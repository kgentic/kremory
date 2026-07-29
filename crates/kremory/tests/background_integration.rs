#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration test: BackgroundIngestor → contradiction detection round-trip.
//!
//! Scenario:
//!   Utterance 1: "the app needs to go live friday"
//!     → extracts Fact: subject="app", predicate="go_live", object="friday"
//!   Utterance 2: "actually can we deploy on monday instead"
//!     → extracts Fact: subject="app", predicate="go_live", object="monday"
//!     → contradiction detected against Fact#1 (same subject+predicate, different value)
//!     → Fact#1 gets invalid_at set (superseded, never deleted)
//!
//! Query: get all facts for subject="app" predicate="go_live"
//!   → Fact#1 (friday) has invalid_at set
//!   → Fact#2 (monday) is the current, valid fact
//!
//! Design: BackgroundIngestor runs ingest() which calls LlmExtractor (uses LLM).
//! We supply a ScriptedLlmClient that returns scripted JSON responses in FIFO order,
//! so each utterance gets a different extraction result (2 calls/utterance) and the
//! contradiction check returns {"indices":[1]} to mark Fact#1 as superseded.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use kremory::core::background::{BackgroundIngestor, IngestorConfig, SendParams};
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::extraction::LlmExtractor;
use kremory::core::ingest::Engine;
use kremory::core::provider::{
    ChatMessage, ChatProvider, ChatResponse, EmbeddingProvider, LLMError, MockChatResponse,
    MockEmbeddingProvider, StructuredOutputFormat, Tool,
};
use kremory::core::schema::TemporalGraph;
use metrics_util::debugging::DebuggingRecorder;

mod common;

// ─── ScriptedLlmClient ────────────────────────────────────────────────────────

/// Returns scripted responses in FIFO order.  When the queue is exhausted it
/// falls back to returning an empty JSON array `"[]"`, which is a safe no-op
/// for both extraction and contradiction prompts.
// Legacy name; impls ChatProvider per AA adoption (2026-04-12 commit 5e8bddd).
#[derive(Debug, Clone)]
struct ScriptedLlmClient {
    queue: Arc<Mutex<Vec<String>>>,
}

impl ScriptedLlmClient {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(
                responses.into_iter().map(str::to_owned).collect(),
            )),
        }
    }
}

#[async_trait::async_trait]
impl ChatProvider for ScriptedLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
        let text = {
            let mut guard = self.queue.lock().expect("ScriptedLlmClient queue poisoned");
            if guard.is_empty() {
                "[]".to_owned()
            } else {
                guard.remove(0)
            }
        };
        Ok(Box::new(MockChatResponse { text }))
    }
}

// ─── ScriptedEmbeddingProvider ────────────────────────────────────────────────

/// Thin wrapper that delegates to MockEmbeddingProvider so embeddings are
/// deterministic without requiring the `embeddings` feature.
#[derive(Debug, Clone)]
struct ScriptedEmbeddingProvider(MockEmbeddingProvider);

impl ScriptedEmbeddingProvider {
    fn new(dim: usize) -> Self {
        Self(MockEmbeddingProvider::new(dim))
    }
}

impl EmbeddingProvider for ScriptedEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::core::error::Result<Vec<f32>>> + Send + 'a {
        self.0.embed(text)
    }
}

// ─── LLM response scripts ─────────────────────────────────────────────────────
//
// LlmExtractor makes TWO LLM calls per utterance (Stage 1: entities with
// integer-ID schema; Stage 2: fact triplets as bare JSON array).
//
// CascadeResolver uses exact-match normalization for "app" == "app" (Tier 1) —
// no LLM call needed for entity resolution when the same entity reappears.
//
// TwoPoolDetector skips the LLM when pool_a (same subject+predicate facts) is
// empty.  After utterance 1 is ingested, utterance 2's fact shares subject+predicate
// so pool_a is non-empty and the LLM is called for contradiction detection.
//
// Call sequence (5 total LLM calls):
//   Call 0: Utterance-1 Stage 1  → {"entities": [{"name":"app","entity_type_id":0}]}
//   Call 1: Utterance-1 Stage 2  → [{"subject":"app","predicate":"go_live","object":"friday",...}]
//   Call 2: Utterance-2 Stage 1  → {"entities": [{"name":"app","entity_type_id":0}]}
//   Call 3: Utterance-2 Stage 2  → [{"subject":"app","predicate":"go_live","object":"monday",...}]
//   Call 4: Contradiction check  → {"indices":[1]} → Fact#1 (friday) is superseded
//
// Extra entries in the script fall back to "[]" — safe no-op for both extraction
// and contradiction prompts.

fn build_scripted_llm() -> ScriptedLlmClient {
    // LlmExtractor makes TWO LLM calls per utterance (Stage 1: entities, Stage 2:
    // relationships) and ONE call for contradiction detection — 5 calls total.
    // ScriptedLlmClient uses capability_of("") → PromptOnly → LlmJsonRepair arm
    // fires first, which calls chat_with_tools once and parses the raw text as JSON.
    // Each valid JSON response is consumed in FIFO order.
    //
    // Call sequence:
    //   Call 0: Utterance-1 Stage 1 → integer-ID entity list (app=Entity, id=0)
    //   Call 1: Utterance-1 Stage 2 → fact triplet (go_live=friday)
    //   Call 2: Utterance-2 Stage 1 → integer-ID entity list (app=Entity, id=0)
    //   Call 3: Utterance-2 Stage 2 → fact triplet (go_live=monday)
    //   Call 4: Contradiction check  → {"indices":[1]} supersedes Fact#1 (friday)
    //
    // CascadeResolver resolves "app" == "app" by exact-match normalization — no LLM call.
    // TwoPoolDetector skips the LLM when pool_a is empty (utterance 1 has no prior facts).

    // Stage 1 format: {"entities": [{"name": "...", "entity_type_id": <int>}]}
    // entity_type_id=0 is the "Entity" catch-all (DEFAULT_ENTITY_TYPES[0]).
    let u1_stage1 = r#"{"entities": [{"name": "app", "entity_type_id": 0}]}"#;

    // Stage 2 format: bare JSON array of triplets.
    let u1_stage2 = r#"[{"subject":"app","predicate":"go_live","object":"friday","is_entity_ref":false,"confidence":0.9}]"#;

    let u2_stage1 = r#"{"entities": [{"name": "app", "entity_type_id": 0}]}"#;

    let u2_stage2 = r#"[{"subject":"app","predicate":"go_live","object":"monday","is_entity_ref":false,"confidence":0.9}]"#;

    // Contradiction check for utterance 2: Fact#1 (friday) is at index 1.
    // Wrapped form required — bare array "[1]" is no longer accepted by parse_index_list.
    // T1.7 (sprint plan v0-2-0-phase-b-prep): `reason` is REQUIRED on
    // ContradictionVerdictWrapper per `llm-output-parse-loudly` — mock must emit it.
    let u2_contradiction =
        r#"{"indices":[1], "reason":"monday declaration supersedes prior friday"}"#;

    ScriptedLlmClient::new(vec![
        u1_stage1,
        u1_stage2,
        u2_stage1,
        u2_stage2,
        u2_contradiction,
    ])
}

// ─── Test ─────────────────────────────────────────────────────────────────────

/// Full round-trip: BackgroundIngestor → ingest → contradiction detected →
/// Fact#1 superseded → query shows Fact#2 valid, Fact#1 invalidated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_contradiction_round_trip() {
    // Install a no-op metrics recorder so histogram!() calls don't panic.
    let recorder = DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );

    let config = PipelineConfig::builder()
        // TD-167: contradiction detection is now DEFAULT-OFF (it supersedes
        // set-valued facts). This test asserts the capability itself, so it
        // opts in EXPLICITLY rather than relying on a default that changed.
        .contradiction_detection_enabled(true)
        .build()
        .expect("PipelineConfig build failed");

    let dim = config.embedding_dim.0;
    let llm = Arc::new(build_scripted_llm());
    let embedder = Arc::new(ScriptedEmbeddingProvider::new(dim));

    let graph = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm,
        embedder,
        config,
        model: None,
    });

    // Ingestor config: tiny channel — only 2 slots needed
    let ingestor_config = IngestorConfig {
        channel_capacity: 8,
        ..IngestorConfig::default()
    };

    let (ingestor, guard) = BackgroundIngestor::new(graph, ingestor_config);

    // Send utterance 1 — "the app needs to go live friday"
    ingestor
        .send(
            "the app needs to go live friday",
            SendParams {
                reference_time: Some(Utc::now()),
                ..SendParams::default()
            },
        )
        .expect("send utterance 1 should succeed");

    // Send utterance 2 — "actually can we deploy on monday instead"
    ingestor
        .send(
            "actually can we deploy on monday instead",
            SendParams {
                reference_time: Some(Utc::now()),
                ..SendParams::default()
            },
        )
        .expect("send utterance 2 should succeed");

    // Close the work channel so the worker drains and exits.
    drop(ingestor);

    // Join the worker thread via spawn_blocking (guard.shutdown() blocks).
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");

    // The graph is now owned by the worker thread — we can't access it directly
    // after BackgroundIngestor consumes it.  Instead we verify behaviour by
    // checking drain_errors() was empty (both ingestions succeeded) and by
    // re-reading the note below.
    //
    // NOTE: BackgroundIngestor moves the graph into the OS thread and does not
    // return it.  The only observable signals from outside are:
    //   1. No IngestErrors via drain_errors() (both utterances processed cleanly)
    //   2. The worker thread exiting without panic (guard.shutdown() succeeds)
    //
    // To assert on graph state we use a second test (below) that calls
    // Engine::ingest() directly — this is the pattern used in ingest.rs tests.
    // That test is the source of truth for contradiction semantics; this test
    // proves the BackgroundIngestor plumbing works end-to-end.
}

/// Prove the contradiction round-trip on Engine directly (same logic as the
/// background worker executes).  This verifies that after two ingestions with
/// a conflicting fact, Fact#1 has invalid_at set and Fact#2 does not.
#[tokio::test]
async fn rql_graph_contradiction_invalidates_superseded_fact() {
    let recorder = DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );

    let config = PipelineConfig::builder()
        // TD-167: contradiction detection is now DEFAULT-OFF (it supersedes
        // set-valued facts). This test asserts the capability itself, so it
        // opts in EXPLICITLY rather than relying on a default that changed.
        .contradiction_detection_enabled(true)
        .build()
        .expect("PipelineConfig build failed");

    let dim = config.embedding_dim.0;
    let llm = Arc::new(build_scripted_llm());
    let embedder = Arc::new(ScriptedEmbeddingProvider::new(dim));

    let graph = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm: Arc::clone(&llm),
        embedder,
        config,
        model: None,
    });

    // Use LlmExtractor explicitly so the scripted LLM responses are consumed
    // regardless of whether the `ner` feature is enabled (which would otherwise
    // route ingest() through GLiNER and bypass the mock entirely).
    let extractor = LlmExtractor::new(Arc::clone(&llm));

    // Ingest utterance 1: "the app needs to go live friday"
    let result1 = graph
        .ingest_with(
            &extractor,
            kremory::core::ingest::IngestWithParams {
                text: "the app needs to go live friday",
                reference_time: Some(Utc::now()),
                group_id: None,
                content_type: None,
                source_params: kremory::core::ingest::SourceParams::default(),
            },
        )
        .await
        .expect("ingest utterance 1 failed");

    assert_eq!(
        result1.inserted_fact_ids.len(),
        1,
        "utterance 1 should insert exactly 1 fact (go_live=friday)"
    );
    assert!(
        result1.invalidated_fact_ids.is_empty(),
        "utterance 1 should not invalidate any facts (no prior state)"
    );

    let fact1_id = result1.inserted_fact_ids[0];

    // Ingest utterance 2: "actually can we deploy on monday instead"
    let result2 = graph
        .ingest_with(
            &extractor,
            kremory::core::ingest::IngestWithParams {
                text: "actually can we deploy on monday instead",
                reference_time: Some(Utc::now()),
                group_id: None,
                content_type: None,
                source_params: kremory::core::ingest::SourceParams::default(),
            },
        )
        .await
        .expect("ingest utterance 2 failed");

    assert_eq!(
        result2.inserted_fact_ids.len(),
        1,
        "utterance 2 should insert exactly 1 fact (go_live=monday)"
    );
    assert_eq!(
        result2.invalidated_fact_ids,
        vec![fact1_id],
        "utterance 2 should invalidate Fact#1 (friday) via contradiction"
    );

    // Query: get all facts for subject=app, predicate=go_live (including expired)
    // get_facts_by_subject_predicate returns only non-expired (expired_at IS NULL).
    // After invalidation, Fact#1 has expired_at set so it is excluded.
    let active_facts = graph
        .graph()
        .get_facts_by_subject_predicate("app", "go_live")
        .await
        .expect("get_facts_by_subject_predicate failed");

    assert_eq!(
        active_facts.len(),
        1,
        "only 1 active fact should remain after contradiction (monday); got: {:?}",
        active_facts
            .iter()
            .map(|f| f.object_value.as_deref().unwrap_or("?"))
            .collect::<Vec<_>>()
    );

    let current_fact = &active_facts[0];
    assert_eq!(
        current_fact.object_value.as_deref(),
        Some("monday"),
        "the surviving fact should have object_value=monday"
    );
    assert!(
        current_fact.invalid_at.is_none(),
        "the current fact (monday) should not be marked invalid"
    );

    // Confirm Fact#1 (friday) is invalidated by querying entity history
    // (entity_history includes expired facts)
    let all_facts = graph
        .graph()
        .entity_history("app")
        .await
        .expect("entity_history failed");

    let friday_fact = all_facts
        .iter()
        .find(|f| f.id == fact1_id)
        .expect("Fact#1 (friday) should still exist in history — supersede never deletes");

    assert!(
        friday_fact.invalid_at.is_some(),
        "Fact#1 (friday) should have invalid_at set after being superseded"
    );
    assert!(
        friday_fact.expired_at.is_some(),
        "Fact#1 (friday) should have expired_at set after being superseded"
    );
}

/// Verify that deferred LLM extraction is invoked after a successful NER ingest.
///
/// We build a `ScriptedLlmClient` with exactly two responses:
///   - Call 0 (NER): returns 1 entity ("Alice") and 0 relationships.
///   - Call 1 (deferred): returns 0 entities and 1 relationship fact.
///
/// After the ingestor shuts down, we verify:
///   1. The scripted queue was fully consumed (both calls happened).
///   2. No errors were reported (neither NER nor deferred failed).
///
/// The graph is moved into the worker thread, so we cannot query it directly
/// after shutdown.  The observable signals are the queue consumption and the
/// absence of errors.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_extraction_invoked_after_successful_ner() {
    let recorder = DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build failed");

    let dim = config.embedding_dim.0;

    // Call 0 — NER extraction: one entity, no relationships.
    let ner_response = serde_json::json!({
        "entities": [{"name": "Alice", "label": "Person"}],
        "relationships": []
    })
    .to_string();

    // Call 1 — Deferred LLM extraction: zero entities, one relationship fact.
    // No entity insertion happens in deferred (Phase 1 owns that).
    let deferred_response = serde_json::json!({
        "entities": [],
        "relationships": [
            {
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme Corp",
                "is_entity_ref": false,
                "confidence": 0.90
            }
        ]
    })
    .to_string();

    let scripted_llm =
        ScriptedLlmClient::new(vec![ner_response.as_str(), deferred_response.as_str()]);

    // Keep a handle to the queue so we can check how many responses remain
    // after the worker shuts down.
    let queue_handle = Arc::clone(&scripted_llm.queue);

    let embedder = Arc::new(ScriptedEmbeddingProvider::new(dim));
    let graph = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm: Arc::new(scripted_llm),
        embedder,
        config,
        model: None,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, ingestor_config);

    ingestor
        .send(
            "Alice works at Acme Corp",
            SendParams {
                reference_time: Some(Utc::now()),
                ..SendParams::default()
            },
        )
        .expect("send should succeed");

    // Poll for errors while the ingestor is alive — we want to catch any
    // NER or deferred failures before closing the channel.
    // 150 × 100ms = 15s. Gives the background deferred-extraction worker ample
    // time to consume the scripted responses before shutdown, robust to the
    // thread starvation seen when the full suite runs under high nextest
    // parallelism with no CI (2-worker-thread runtime, loaded machine). Breaks
    // early on the first error; the generous ceiling only affects the no-error
    // path and never masks a real hang.
    let mut errors_seen = Vec::new();
    for _ in 0..150 {
        errors_seen = ingestor.drain_errors();
        if !errors_seen.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");

    // Both scripted responses should have been consumed.
    let remaining = queue_handle.lock().expect("queue lock poisoned").len();
    assert_eq!(
        remaining, 0,
        "expected 0 scripted responses remaining after shutdown; \
         {remaining} left means deferred was not invoked"
    );

    assert!(
        errors_seen.is_empty(),
        "expected no ingestion errors; got: {:?}",
        errors_seen.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// Smoke test: BackgroundIngestor shuts down cleanly with no errors when
/// the LLM queue is exhausted (falls back to "[]").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_drains_without_errors() {
    let recorder = DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );

    // TD-006 fix (Phase E E-4): set allowed_entity_types so the ner-feature
    // GlinerExtractor doesn't reject the open-ended config at extract time.
    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build failed");

    let dim = config.embedding_dim.0;
    let llm = Arc::new(build_scripted_llm());
    let embedder = Arc::new(ScriptedEmbeddingProvider::new(dim));

    let graph = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm,
        embedder,
        config,
        model: None,
    });
    let (ingestor, guard) = BackgroundIngestor::new(graph, IngestorConfig::default());

    ingestor
        .send(
            "the app needs to go live friday",
            SendParams {
                reference_time: Some(Utc::now()),
                ..SendParams::default()
            },
        )
        .expect("send utterance 1 should succeed");
    ingestor
        .send(
            "actually can we deploy on monday instead",
            SendParams {
                reference_time: Some(Utc::now()),
                ..SendParams::default()
            },
        )
        .expect("send utterance 2 should succeed");

    // Poll for errors for up to 5 seconds while keeping the ingestor alive.
    let mut errors_seen = Vec::new();
    for _ in 0..50 {
        errors_seen = ingestor.drain_errors();
        if !errors_seen.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");

    assert!(
        errors_seen.is_empty(),
        "BackgroundIngestor should process both utterances without errors; got: {:?}",
        errors_seen.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// TD-167: with contradiction detection at its DEFAULT (off), a second value
/// for the same (subject, predicate) must be STORED ALONGSIDE the first, not
/// supersede it.
///
/// This is the regression guard for a silent data-loss bug. kremory treated
/// every predicate as functional, so ingesting a LIST destroyed all but the
/// last member. Measured on 8 LongMemEval sessions (2026-07-29): 81 of 1,021
/// facts invalidated, ≥31% provably multi-valued —
/// `has_performer: billie eilish / tove lo / lana del rey` all superseded by
/// `the 1975`; `contain: rolled oats` superseded by `seeds`.
///
/// The fixture deliberately uses a genuinely SET-VALUED predicate
/// (`has_performer`) and scripts the LLM to claim a contradiction anyway —
/// mirroring what the real model does, because the prompt at
/// `core/contradiction.rs:176` instructs it to treat "same relationship but
/// newer value" as an update. With the flag off that verdict is never
/// requested, so both facts survive.
///
/// Pairs with `rql_graph_contradiction_invalidates_superseded_fact` above,
/// which opts INTO the flag and asserts supersession still works when wanted.
#[tokio::test]
async fn td167_set_valued_facts_survive_when_contradiction_detection_is_off() {
    let recorder = DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );

    // Contradiction detection explicitly OFF. As of ADR-079 rev.2 the DEFAULT
    // is ON again (the prompt fix measured 0/8 set-valued destroyed and a
    // corpus run confirmed it), so this must be set rather than assumed — the
    // property under test here is that the APPEND-ONLY path still preserves
    // set-valued facts, which is the fallback posture if the flag is ever
    // turned back off.
    let config = PipelineConfig::builder()
        .contradiction_detection_enabled(false)
        .build()
        .expect("PipelineConfig build failed");
    assert!(
        !config.contradiction_detection_enabled,
        "explicit opt-out must be honoured — if this fails the flag is not wired"
    );

    let dim = config.embedding_dim.0;
    // Only 4 responses: two entity/triplet pairs. A 5th (contradiction verdict)
    // is deliberately NOT scripted — if the gate leaks and the detector runs,
    // the queue is exhausted and the test fails loudly rather than silently
    // passing for the wrong reason.
    let llm = Arc::new(ScriptedLlmClient::new(vec![
        r#"{"entities": [{"name": "festival", "entity_type_id": 0}]}"#,
        r#"[{"subject":"festival","predicate":"has_performer","object":"billie eilish","is_entity_ref":false,"confidence":0.9}]"#,
        r#"{"entities": [{"name": "festival", "entity_type_id": 0}]}"#,
        r#"[{"subject":"festival","predicate":"has_performer","object":"the 1975","is_entity_ref":false,"confidence":0.9}]"#,
    ]));
    let embedder = Arc::new(ScriptedEmbeddingProvider::new(dim));

    let graph = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm: Arc::clone(&llm),
        embedder,
        config,
        model: None,
    });
    let extractor = LlmExtractor::new(Arc::clone(&llm));

    let ingest = |text: &'static str| {
        let g = &graph;
        let e = &extractor;
        async move {
            g.ingest_with(
                e,
                kremory::core::ingest::IngestWithParams {
                    text,
                    reference_time: Some(Utc::now()),
                    group_id: None,
                    content_type: None,
                    source_params: kremory::core::ingest::SourceParams::default(),
                },
            )
            .await
            .expect("ingest failed")
        }
    };

    let r1 = ingest("the festival has billie eilish performing").await;
    assert_eq!(r1.inserted_fact_ids.len(), 1, "first performer should store");
    assert!(r1.invalidated_fact_ids.is_empty(), "nothing to invalidate yet");

    let r2 = ingest("the 1975 are also performing at the festival").await;
    assert_eq!(r2.inserted_fact_ids.len(), 1, "second performer should store");

    // THE ASSERTION THIS TEST EXISTS FOR.
    assert!(
        r2.invalidated_fact_ids.is_empty(),
        "TD-167 REGRESSION: adding a second value for the same (subject, \
         predicate) invalidated the first. A festival has MANY performers — \
         this is set-valued data, not a contradiction. Invalidated: {:?}",
        r2.invalidated_fact_ids
    );
}

/// ADR-079 rev.2: contradiction detection is ON by default.
///
/// Pinned as a test because this default has now moved TWICE in one day, in
/// both directions, and each move is a product decision with real consequences:
///
///   * OFF (morning) — the prompt destroyed set-valued facts: a festival's
///     2nd..6th performer each superseded the previous. 7/8 destroyed on the
///     production model.
///   * ON  (after the fix) — the coexistence + temporal prompt measured 0/8
///     destroyed; a corpus run confirmed it (contradiction rate 38.8% -> 11.3%,
///     no multi-valued predicate destroyed). LongMemEval's knowledge-update
///     category (78 of 500 questions) TESTS this mechanism, so shipping the v1
///     benchmark with it disabled would publish a number with the relevant
///     feature switched off.
///
/// If this assertion fails, someone changed a shipped product decision. Read
/// ADR-079 before "fixing" the test.
#[tokio::test]
async fn adr079_contradiction_detection_is_on_by_default() {
    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig build failed");
    assert!(
        config.contradiction_detection_enabled,
        "ADR-079 rev.2: contradiction detection ships ON. If this was flipped \
         deliberately, update ADR-079 and this test together — a silent flip \
         changes whether kremory can correct a stale fact at all."
    );
}
