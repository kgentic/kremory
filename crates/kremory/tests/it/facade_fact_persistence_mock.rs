//! Deterministic (no real LLM) isolation of the facade 0-facts bug (ship-build-debug 2026-06-29).
//!
//! The real-LLM e2e (`facade_fact_extraction_e2e.rs`) shows `Memory::open().with_llm().remember()`
//! lands 0 facts even though `Engine::ingest_with` with the same extractor lands 16. This test
//! drives the FULL facade path with a MockChatProvider staged to DEFINITELY return one fact
//! triple for IntegerId's 3 stages. It isolates the question:
//!   - facts land here  → facade persistence path is fine; real-LLM returns empty triplets.
//!   - 0 facts here     → facade path/persistence bug (deterministically debuggable).
//!
//! Gate: `test-utils` (for `temporal_graph_for_test`). NO real LLM — runs in standard CI.
//!   cargo test -p kremory --features test-utils --test it facade_fact_persistence_mock:: -- --nocapture

#![cfg(feature = "test-utils")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::config::PipelineConfig;
use kremory::core::error::Error as KremoryError;
use kremory::core::extraction::IntegerIdLlmExtractor;
use kremory::core::graph::{FactInsert, InsertEntityParams};
use kremory::core::ingest::{
    Engine, EngineNewParams, IngestParams, IngestWithParams, SourceParams,
};
use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};
use metrics_util::debugging::{DebuggingRecorder, Snapshot};

/// Sum a labeled counter across all label-value variants. Mirrors the helper in
/// `with_facts_integration.rs` — labeled counters can have multiple variants, so a
/// single `.find()` would under-report ([[observability-first-class]] failure #9).
fn find_counter_labeled(snapshot: Snapshot, name: &str) -> u64 {
    snapshot
        .into_vec()
        .into_iter()
        .filter_map(|(key, _unit, _desc, value)| {
            if key.key().name() == name {
                if let metrics_util::debugging::DebugValue::Counter(v) = value {
                    return Some(v);
                }
            }
            None
        })
        .sum()
}

/// Mock staged for IntegerIdLlmExtractor's 3 stages (substring-keyed) → one works_at fact.
fn staged_mock() -> MockChatProvider {
    let mut map = HashMap::new();
    // Stage 1 — entities (build_entity_prompt).
    map.insert(
        "Each entity must appear exactly once".to_string(),
        r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme","entity_type_id":2}]}"#
            .to_string(),
    );
    // Stage 2 — relationship names (build_relation_names_prompt).
    map.insert(
        "Output a JSON array of relationship name strings.".to_string(),
        r#"["works_at"]"#.to_string(),
    );
    // Stage 3 — triplets (build_triplet_prompt).
    map.insert(
        "Output a concise JSON array of objects with".to_string(),
        r#"[{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.95}]"#
            .to_string(),
    );
    // CascadeResolver escalation (no-op: distinct entities).
    map.insert(
        "Are these two entities".to_string(),
        "\"different\"".to_string(),
    );
    // TwoPoolDetector (no contradictions).
    map.insert(
        "Output a JSON array of index numbers".to_string(),
        "[]".to_string(),
    );
    MockChatProvider::new(map)
}

/// Mock staged to emit a SET-VALUED predicate: three distinct objects for one
/// subject+predicate, all inside a single episode. DUR-2 fixture.
fn staged_mock_multivalued() -> MockChatProvider {
    let mut map = HashMap::new();
    map.insert(
        "Each entity must appear exactly once".to_string(),
        r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#.to_string(),
    );
    map.insert(
        "Output a JSON array of relationship name strings.".to_string(),
        r#"["speaks"]"#.to_string(),
    );
    // Three literal objects (is_entity_ref:false) — English, French, Spanish.
    map.insert(
        "Output a concise JSON array of objects with".to_string(),
        r#"[{"subject":"Alice","predicate":"speaks","object":"English","is_entity_ref":false,"confidence":0.95},{"subject":"Alice","predicate":"speaks","object":"French","is_entity_ref":false,"confidence":0.95},{"subject":"Alice","predicate":"speaks","object":"Spanish","is_entity_ref":false,"confidence":0.95}]"#
            .to_string(),
    );
    map.insert(
        "Are these two entities".to_string(),
        "\"different\"".to_string(),
    );
    map.insert(
        "Output a JSON array of index numbers".to_string(),
        "[]".to_string(),
    );
    MockChatProvider::new(map)
}

/// DUR-2 (V1-CANONICAL §4.1) — a set-valued predicate asserted once in a SINGLE
/// episode must store every value, not just the first.
///
/// The F5 within-episode pre-check tested `source_episode_id == episode_id` against
/// `pool_a`, which comes from `get_facts_by_subject_predicate` and is therefore
/// **object-agnostic**. So after "Alice speaks English" landed, "Alice speaks French"
/// and "Alice speaks Spanish" each matched an existing same-episode row on the
/// subject+predicate pair alone and were silently dropped — no error, and no dropped
/// count anywhere in `IngestionResult`.
///
/// Within one episode there is no temporal ordering that could make one assertion
/// supersede another; they are co-asserted. Differing objects are multiple values,
/// not a contradiction. Cross-episode supersession is a separate, temporally-ordered
/// mechanism and is untouched by this fix.
///
/// FAILS before the fix with `facts == 1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn within_episode_set_valued_predicate_keeps_every_value() {
    let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
    let config = PipelineConfig::builder().build().expect("config");
    let dim = config.embedding_dim.0;
    let llm = Arc::new(staged_mock_multivalued());
    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::clone(&llm),
        embedder: Arc::new(MockEmbeddingProvider::new(dim)),
        config,
        model: None,
    });
    let extractor = IntegerIdLlmExtractor::new(Arc::clone(&llm));

    let r = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Alice speaks English, French and Spanish.",
                reference_time: None,
                declared_reference_time: None,
                group_id: Some("dur2-multivalued"),
                content_type: None,
                source_params: SourceParams::default(),
            },
        )
        .await
        .expect("ingest_with");

    assert_eq!(
        r.inserted_fact_ids.len(),
        3,
        "DUR-2: all three values of the set-valued predicate `speaks` must persist \
         from a single episode — got {} fact(s). The within-episode pre-check is \
         matching on subject+predicate instead of the full triple.",
        r.inserted_fact_ids.len()
    );
}

/// ADR-045 §11 detection intent — the multi-value counter must actually FIRE.
///
/// The DUR-2 fix stores every value instead of dropping all but the first. ADR-045's
/// contradiction-detection intent is preserved additively via
/// `rql.ingest.within_episode_multivalue`. A counter nobody asserts on is an
/// unverified claim about observability ([[observability-first-class]] failure #4:
/// "counters that are 0 when work happened"), so this drives the real ingest path and
/// reads the metric back.
///
/// Three `speaks` values in one episode ⇒ 2 increments (values 2 and 3 each see a
/// prior same-episode row for the pair; the first does not).
#[test]
fn within_episode_multivalue_counter_fires() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
            let config = PipelineConfig::builder().build().expect("config");
            let dim = config.embedding_dim.0;
            let llm = Arc::new(staged_mock_multivalued());
            let engine = Engine::new(EngineNewParams {
                graph,
                llm: Arc::clone(&llm),
                embedder: Arc::new(MockEmbeddingProvider::new(dim)),
                config,
                model: None,
            });
            let extractor = IntegerIdLlmExtractor::new(Arc::clone(&llm));
            let r = engine
                .ingest_with(
                    &extractor,
                    IngestWithParams {
                        text: "Alice speaks English, French and Spanish.",
                        reference_time: None,
                        declared_reference_time: None,
                        group_id: Some("multivalue-counter"),
                        content_type: None,
                        source_params: SourceParams::default(),
                    },
                )
                .await
                .expect("ingest_with");

            assert_eq!(
                r.inserted_fact_ids.len(),
                3,
                "all three values must persist"
            );

            let fired = find_counter_labeled(
                snapshotter.snapshot(),
                "rql.ingest.within_episode_multivalue",
            );
            assert_eq!(
                fired, 2,
                "the multi-value signal ADR-045 wanted must be observable — expected 2 \
                 increments for 3 co-asserted values, got {fired}"
            );
        });
    });
}

/// DUR-2 companion — an EXACT repeated triple in one episode must still dedup to one.
///
/// Guards the other direction of the same fix: making the check triple-exact must not
/// turn it into a no-op. Without this, a fix that simply deleted the pre-check would
/// pass the test above while silently regressing duplicate suppression.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn within_episode_exact_duplicate_triple_still_dedups() {
    let mut map = HashMap::new();
    map.insert(
        "Each entity must appear exactly once".to_string(),
        r#"{"entities":[{"name":"Alice","entity_type_id":1}]}"#.to_string(),
    );
    map.insert(
        "Output a JSON array of relationship name strings.".to_string(),
        r#"["speaks"]"#.to_string(),
    );
    // The SAME triple three times.
    map.insert(
        "Output a concise JSON array of objects with".to_string(),
        r#"[{"subject":"Alice","predicate":"speaks","object":"English","is_entity_ref":false,"confidence":0.95},{"subject":"Alice","predicate":"speaks","object":"English","is_entity_ref":false,"confidence":0.95},{"subject":"Alice","predicate":"speaks","object":"English","is_entity_ref":false,"confidence":0.95}]"#
            .to_string(),
    );
    map.insert(
        "Are these two entities".to_string(),
        "\"different\"".to_string(),
    );
    map.insert(
        "Output a JSON array of index numbers".to_string(),
        "[]".to_string(),
    );

    let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
    let config = PipelineConfig::builder().build().expect("config");
    let dim = config.embedding_dim.0;
    let llm = Arc::new(MockChatProvider::new(map));
    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::clone(&llm),
        embedder: Arc::new(MockEmbeddingProvider::new(dim)),
        config,
        model: None,
    });
    let extractor = IntegerIdLlmExtractor::new(Arc::clone(&llm));

    let r = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Alice speaks English. Alice speaks English. Alice speaks English.",
                reference_time: None,
                declared_reference_time: None,
                group_id: Some("dur2-exact-dup"),
                content_type: None,
                source_params: SourceParams::default(),
            },
        )
        .await
        .expect("ingest_with");

    assert_eq!(
        r.inserted_fact_ids.len(),
        1,
        "an exact repeated triple within one episode must still collapse to a single \
         fact — got {}",
        r.inserted_fact_ids.len()
    );
}

/// DUR-2 on the DEFERRED path (Path β) — the second, previously-uncited copy.
///
/// V1-CANONICAL §4.1 cited only `ingest_with.rs`, but `deferred.rs` carried an
/// identical object-agnostic within-episode check — and per V1-CANONICAL §6d the
/// deferred path is what `Memory::remember()` actually drives, so this copy is the
/// one most consumers hit. A fix applied only to the inline path would have left
/// the shipping path broken while the inline test went green.
///
/// FAILS before the fix with 1 fact instead of 3.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_path_set_valued_predicate_keeps_every_value() {
    use kremory::core::background::{BackgroundIngestor, IngestorConfig, SendParams};
    let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
    let config = PipelineConfig::builder().build().expect("config");
    let dim = config.embedding_dim.0;
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm: Arc::new(staged_mock_multivalued()),
        embedder: Arc::new(MockEmbeddingProvider::new(dim)),
        config,
        model: None,
    });
    let (ingestor, guard) = BackgroundIngestor::new(
        engine,
        IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        },
    );
    ingestor
        .send(
            "Alice speaks English, French and Spanish.",
            SendParams {
                group_id: Some("dur2-deferred-ns".to_string()),
                ..SendParams::default()
            },
        )
        .expect("send");

    // Poll until the deferred facts land. Wait for the FULL expected set, not just
    // the first row — breaking on `!is_empty()` would pass on the buggy 1-fact
    // outcome, which is exactly the shape of guard this fix exists to prevent.
    let mut facts = Vec::new();
    for _ in 0..50 {
        facts = graph.facts_at(chrono::Utc::now()).await.expect("facts_at");
        if facts.len() >= 3 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown panicked");

    let speaks: Vec<_> = facts.iter().filter(|f| f.predicate == "speaks").collect();
    assert_eq!(
        speaks.len(),
        3,
        "DUR-2 (deferred path): all three `speaks` values must persist from one \
         episode — got {}. This is the path Memory::remember() drives.",
        speaks.len()
    );
}

/// Control: Engine `ingest_with` directly, bisecting `group_id` (None vs a fresh namespace),
/// with the SAME staged mock. Isolates whether passing a `group_id` (as the facade does) is
/// what drops the fact.
async fn engine_facts_for_group(group_id: Option<&str>) -> usize {
    engine_facts_cfg(group_id, false).await
}

async fn engine_facts_cfg(group_id: Option<&str>, with_allowed_types: bool) -> usize {
    let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
    let mut cb = PipelineConfig::builder();
    if with_allowed_types {
        cb = cb.allowed_entity_types(vec!["Person".to_string(), "Organisation".to_string()]);
    }
    let config = cb.build().expect("config");
    let dim = config.embedding_dim.0;
    let llm = Arc::new(staged_mock());
    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::clone(&llm),
        embedder: Arc::new(MockEmbeddingProvider::new(dim)),
        config,
        // model: None → PromptOnly, where the mock's plain-array stage-3 response parses
        // (FormatSchema would reject it). Keeps this control uncontaminated.
        model: None,
    });
    let extractor = IntegerIdLlmExtractor::new(llm);
    let r = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Alice works at Acme Corp.",
                reference_time: None,
                declared_reference_time: None,
                group_id,
                content_type: None,
                source_params: SourceParams::default(),
            },
        )
        .await
        .expect("ingest_with");
    eprintln!(
        "[control] group_id={:?} entities={} facts={}",
        group_id,
        r.upserted_entities.len(),
        r.inserted_fact_ids.len()
    );
    r.inserted_fact_ids.len()
}

/// Regression guard: Engine `ingest_with` writes the same facts whether `group_id` is
/// `None` (default namespace) or a fresh namespace. Before the composite-FK fix
/// (`subject_group_id`/`object_group_id` were unset → FK failed for non-default groups),
/// `group_id=Some(ns)` silently dropped every fact — which broke the entire facade path
/// (`Memory::remember` always passes `group_id=Some(namespace)`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_ingest_writes_facts_in_any_namespace() {
    let none = engine_facts_for_group(None).await;
    let some = engine_facts_for_group(Some("mock-e2e")).await;
    eprintln!("[control] facts: group_none={none} group_some={some}");
    assert!(
        none >= 1 && some >= 1 && none == some,
        "facts must persist regardless of group_id; group_none={none} vs group_some={some}"
    );
}

/// B3 regression guard (2026-07-12) — the ner-gated `Engine::ingest()` dispatch bug.
///
/// `Engine::ingest()` (the wrapper that resolves `self.extractor`) previously
/// special-cased `#[cfg(feature = "ner")]` to ALWAYS route through the process-wide
/// bare `GlinerExtractor` (`ner_singleton()`, whose `::extract` returns `facts: vec![]`),
/// discarding the builder-resolved extractor. Under `--features ner` / `--all-features`,
/// every fact-producing extraction via `ingest()` was a silent no-op — for ~100 commits.
///
/// Why nothing caught it: every other fast test in this file calls `ingest_with()`
/// (which was NEVER buggy — it takes the extractor as an argument), and the full suite
/// was never RUN under `--features ner` (no CI matrix). This guard closes both holes:
/// it calls `ingest()` — the exact buggy wrapper — with the Engine's default
/// `ExtractorKind::IntegerId` (fed the staged mock), and it is **ungated** so it runs in
/// BOTH the `not(ner)` and `--features ner` matrices. Fixed in `5d2ef4a` (always
/// `self.ingest_with(&self.extractor, ...)`).
///
/// Verified to FAIL on the reintroduced bug (under `--features ner` the wrapper routes to
/// bare GLiNER → 0 facts → this assertion fires) and PASS on the fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_ingest_wrapper_dispatches_configured_extractor_persists_facts() {
    let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
    let config = PipelineConfig::builder().build().expect("config");
    let dim = config.embedding_dim.0;
    let llm = Arc::new(staged_mock());
    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::clone(&llm),
        embedder: Arc::new(MockEmbeddingProvider::new(dim)),
        config,
        // model: None → PromptOnly so the mock's plain-array stage-3 response parses.
        model: None,
    });
    // Call `ingest()` — the wrapper with the historical cfg dispatch — NOT `ingest_with()`.
    // Engine::new defaults `self.extractor` to `ExtractorKind::IntegerId(mock)`, which
    // produces one `works_at` fact from the staged mock.
    let r = engine
        .ingest(IngestParams {
            text: "Alice works at Acme Corp.",
            reference_time: None,
            declared_reference_time: None,
            group_id: Some("b3-ner-guard"),
            content_type: None,
            source_params: SourceParams::default(),
        })
        .await
        .expect("ingest");
    assert!(
        !r.inserted_fact_ids.is_empty(),
        "Engine::ingest() must persist >=1 fact via the configured extractor \
         (B3 regression: ner-gated bare-GLiNER dispatch dropped ALL facts); \
         got {} facts, {} entities",
        r.inserted_fact_ids.len(),
        r.upserted_entities.len()
    );
}

/// Regression guard (Quinn REL-001 / TD-080 #1): the DEFERRED/background Phase-2 path
/// (`BackgroundIngestor` → `deferred.rs`) must persist facts in a non-default namespace.
///
/// FIXED 2026-06-29 (TD-080 #1, spec
/// `td-080-facade-fact-quality-and-namespace-clearance-sprint-2026-06-29.md` §P1):
/// `verify_stage::stage3_write` now threads `request.group_id` into the entity INSERT,
/// the idempotency synthetic hash, and the episodic edge — so background entities land
/// in the requested namespace and the deferred fact's composite FK
/// `(facts.subject_group_id) → entities(id, group_id)` resolves. Previously entities were
/// hardcoded to "default" while the namespace-aware fact targeted the requested namespace
/// → silent composite-FK drop. Default-namespace background ingestion was always fine.
#[cfg(not(feature = "ner"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deferred_path_writes_facts_in_namespace() {
    use kremory::core::background::{BackgroundIngestor, IngestorConfig, SendParams};
    let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("graph"));
    let config = PipelineConfig::builder().build().expect("config");
    let dim = config.embedding_dim.0;
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm: Arc::new(staged_mock()),
        embedder: Arc::new(MockEmbeddingProvider::new(dim)),
        config,
        // model: None → PromptOnly, where the mock's plain-array stage-3 parses.
        model: None,
    });
    let (ingestor, guard) = BackgroundIngestor::new(
        engine,
        IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        },
    );
    ingestor
        .send(
            "Alice works at Acme Corp.",
            SendParams {
                group_id: Some("deferred-ns".to_string()),
                ..SendParams::default()
            },
        )
        .expect("send");

    // Poll until the deferred fact lands (worker processes Phase-1 then deferred Phase-2).
    let mut facts = Vec::new();
    for _ in 0..50 {
        facts = graph.facts_at(chrono::Utc::now()).await.expect("facts_at");
        if !facts.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown panicked");

    assert!(
        !facts.is_empty(),
        "deferred/background path must persist >=1 fact in a non-default namespace \
         (composite-FK regression — facts.subject_group_id must match the entity namespace)"
    );
}

/// The full facade path persists facts when the extractor returns a triple.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facade_remember_persists_facts_with_mock_extractor() {
    let dir = tempfile::tempdir().expect("tempdir");

    let llm: Arc<dyn ChatProvider> = Arc::new(staged_mock());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));

    // NO with_model_id → PromptOnly capability → the mock's plain-array stage-3 response
    // parses (FormatSchema arm would reject it). This isolates facade PERSISTENCE from the
    // FormatSchema extraction path: if the fact lands here, remember()'s ingest→persist
    // wiring is fine and the real-LLM 0-facts bug is FormatSchema-extraction-specific.
    let mem = Memory::open(dir.path().join("kremory.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(384)
        .default_namespace(Namespace::new("mock-e2e"))
        .await
        .expect("Memory::open facade");

    let commit = mem
        .remember("Alice works at Acme Corp.")
        .from_chat("mock-session")
        .await
        .expect("remember");

    // Fact extraction is DEFERRED (Phase 2, background worker) — `remember()`
    // returns after Phase 1 (sync embed+NER), so reading facts immediately races
    // the deferred extractor and sees 0. Wait for the episode to reach a terminal
    // processing status before asserting on persisted facts.
    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("episode_entity_id must parse to an i64 rowid");
    mem.wait_for_processing(episode_id, std::time::Duration::from_secs(30))
        .await
        .expect("wait_for_processing must complete");

    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (needs test-utils)");
    let facts = graph.facts_at(chrono::Utc::now()).await.expect("facts_at");
    let history = graph.entity_history("Alice").await.unwrap_or_default();

    eprintln!(
        "[mock-facade] active_facts={} alice_history(incl expired)={}",
        facts.len(),
        history.len()
    );
    for f in &facts {
        eprintln!(
            "[mock-facade]   fact#{}: {} --{}--> obj_id={:?} obj_value={:?}",
            f.id, f.subject_id, f.predicate, f.object_id, f.object_value
        );
    }

    assert!(
        !facts.is_empty(),
        "facade remember() must persist the mock-extracted works_at fact (active_facts>0). \
         Got 0 active, {} in history — if history>0 the fact was written then invalidated; \
         if both 0 the facade ingest path never wrote it.",
        history.len()
    );
}

/// P3 self-loop guard (TD-080 §P3): entity-entity self-loop (subject==object) must be
/// rejected with `Error::SelfLoop`; `try_insert_fact_with_group` must swallow to `Ok(None)`
/// so a self-loop in a batch doesn't abort the surrounding fact writes; and a legitimate
/// different-subject/object triple must be unaffected.
///
/// The `kremory.fact.rejected_total{reason="self_loop"}` counter fires inside
/// `insert_fact_with_group` PRE-swallow so P5 quality metrics capture the defect rate
/// even when callers use the `try_*` path (Vera P3-001).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn self_loop_fact_is_rejected() {
    let graph = TemporalGraph::open_in_memory().await.expect("graph");

    // Seed two entities so the legitimate-triple assertion exercises a real FK path.
    for id in ["Alice", "Bob"] {
        graph
            .insert_entity(InsertEntityParams {
                id,
                entity_type_id: 1,
                properties: serde_json::json!({}),
            })
            .await
            .unwrap_or_else(|e| panic!("insert_entity {id}: {e}"));
    }

    // Self-loop — loud path: insert_fact_with_group must return Error::SelfLoop.
    let err = graph
        .insert_fact_with_group(
            FactInsert::new("Alice", "knows", chrono::Utc::now()).object_id("Alice"),
            None,
        )
        .await
        .expect_err("self-loop must be rejected");
    assert!(
        matches!(err, KremoryError::SelfLoop { .. }),
        "expected Error::SelfLoop, got: {err:?}"
    );

    // Self-loop — try_* path: must swallow to Ok(None) without aborting the batch.
    let swallowed = graph
        .try_insert_fact_with_group(
            FactInsert::new("Alice", "knows", chrono::Utc::now()).object_id("Alice"),
            None,
        )
        .await
        .expect("try_insert_fact_with_group must not Err on self-loop");
    assert!(
        swallowed.is_none(),
        "try_insert_fact_with_group must return Ok(None) for a self-loop"
    );

    // Legitimate triple — must insert without triggering the guard.
    // group_id=Some("default") matches entity rows that `insert_entity` writes via
    // the schema DEFAULT ('default'). Raw None would bind NULL to subject_group_id
    // (NOT NULL column), which the DB rejects — same reason the engine normalises via
    // `group_id.unwrap_or("default")` before calling insert_fact_with_group.
    let fact_id = graph
        .insert_fact_with_group(
            FactInsert::new("Alice", "knows", chrono::Utc::now()).object_id("Bob"),
            Some("default"),
        )
        .await
        .expect("legitimate triple (Alice → knows → Bob) must succeed");
    assert!(fact_id > 0, "inserted fact must have a positive rowid");
}
