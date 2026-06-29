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
//!   cargo test -p kremory --features test-utils --test facade_fact_persistence_mock -- --nocapture

#![cfg(feature = "test-utils")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::config::PipelineConfig;
use kremory::core::error::Error as KremoryError;
use kremory::core::extraction::IntegerIdLlmExtractor;
use kremory::core::graph::{FactInsert, InsertEntityParams};
use kremory::core::ingest::{Engine, EngineNewParams, IngestWithParams, SourceParams};
use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

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

    mem.remember("Alice works at Acme Corp.")
        .from_chat("mock-session")
        .await
        .expect("remember");

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
