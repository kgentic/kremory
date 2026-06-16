//! ADR-052 Gap 1 REGRESSION GUARD — sink callbacks MUST fire from production ingest.
//!
//! Governing contract: ADR `rqlm-async-event-handle-api-design-2026-05-19` §4.8
//! (`IngestEventSink` / `EnrichmentEventSink` sync-inline sink contract) +
//! ADR-052 Gap 1 (sink callsite firing). Canonical fire-sites: commits 737e152
//! (Phase 3) + 34fdc60 (Phase 4); regressed by fb85ba8 (Phase 7 dual-path
//! consolidation), re-established by the cause-fix this test guards.
//!
//! # Why this test exists
//!
//! The pre-existing `event_order.rs` / `sink_trait_shape.rs` / `facade_event_sink.rs`
//! tests drive the sink by calling its methods MANUALLY (or only assert the
//! builder STORES the sink). None of them assert that PRODUCTION ingestion fires
//! the callbacks. That gap let fb85ba8 silently drop every fire-site while every
//! test stayed green — `Memory::remember(..).with_event_sink(..)` became a no-op
//! on the inline path, surfaced only by the real-LLM `golden_path_smoke`.
//!
//! This test closes the gap: it runs REAL ingestion through the PUBLIC `Memory`
//! facade (`remember(..).with_event_sink(..).await`) with a deterministic mock
//! extractor + mock LLM (NO Ollama, default `test-utils` features) and asserts
//! the capturing sink recorded the events. A future consolidation that drops the
//! fire-sites again fails HERE, at near-zero cost, without needing a live model.

#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use kremory::core::error::IngestStatus;
use kremory::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::sink::{ContradictionDetected, IngestEventSink, IngestionError};
use kremory::memory::events::{BatchPhase2Complete, EnrichmentEventSink};
use kremory::{DynEmbeddingProvider, Memory, Namespace};

// ── Capturing sink (records every callback for assertion) ─────────────────────

#[derive(Default)]
struct CapturingSink {
    entities: Mutex<Vec<(String, String)>>,
    edges: Mutex<Vec<(String, String, String)>>,
    stages: Mutex<Vec<IngestStatus>>,
}

impl IngestEventSink for CapturingSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.entities
            .lock()
            .unwrap()
            .push((entity_id.to_owned(), name.to_owned()));
    }
    fn on_edge_added(&self, from: &str, to: &str, predicate: &str) {
        self.edges
            .lock()
            .unwrap()
            .push((from.to_owned(), to.to_owned(), predicate.to_owned()));
    }
    fn on_contradiction(&self, _event: ContradictionDetected) {}
    fn on_dedup_merge(&self, _surviving_id: &str, _absorbed_id: &str) {}
    fn on_stage_change(&self, stage: IngestStatus) {
        self.stages.lock().unwrap().push(stage);
    }
    fn on_ingestion_error(&self, _event: IngestionError) {}
}

impl EnrichmentEventSink for CapturingSink {
    fn on_community_updated(&self, _community_id: &str, _member_count: usize) {}
    fn on_batch_phase2_complete(&self, _event: BatchPhase2Complete) {}
}

// ── Deterministic extractor: returns two fixed entities, no facts ─────────────
//
// Returning `facts: vec![]` keeps the Phase 2 contradiction/fact loop a no-op
// (so the test needs no scripted contradiction JSON), while the entity loop in
// `ingest_with` runs in full — firing on_entity_extracted + on_edge_added per
// persisted entity. The mock LLM below satisfies the resolver/detector
// construction; on a fresh DB the resolver loop never executes (no existing
// entities) so the empty LLM response is never consulted.

struct TwoEntityExtractor;

impl EntityExtractor for TwoEntityExtractor {
    fn name(&self) -> &'static str {
        "two-entity-test-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: vec![
                ExtractedEntity {
                    name: "Alice".to_string(),
                    label: "Person".to_string(),
                    properties: serde_json::json!({"name": "Alice"}),
                },
                ExtractedEntity {
                    name: "Acme Corp".to_string(),
                    label: "Organisation".to_string(),
                    properties: serde_json::json!({"name": "Acme Corp"}),
                },
            ],
            facts: vec![],
        })
    }
}

// ── Deterministic failing extractor: errors AFTER the Extracting fire ─────────
//
// Drives the ADR-051 §4 state-machine invariant guard (Quinn MED-01): a fallible
// `?`-step AFTER `on_stage_change(Extracting)` fires MUST still reach a terminal
// `Failed` transition before the error propagates — never `Extracting` then
// silence. `extractor.extract(..).await?` is one such post-`Extracting` step, so
// returning `Err` here is the cleanest way to exercise that exact path.

struct FailingExtractor;

impl EntityExtractor for FailingExtractor {
    fn name(&self) -> &'static str {
        "failing-test-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Err(kremory::core::error::Error::Other(anyhow::anyhow!(
            "deterministic extractor failure (regression guard)"
        )))
    }
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

fn unique_db(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_sink_fires_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

// ── The regression guard ──────────────────────────────────────────────────────

/// Production ingestion through the PUBLIC facade MUST fire the sink callbacks.
///
/// Mirrors the `golden_path_smoke` consumer journey
/// (`remember(..).with_event_sink(..).await`) but mock-driven, so it runs on
/// the default `test-utils` tier with NO Ollama. Asserts the sink — driven by
/// REAL ingestion, not by the test calling it manually — recorded:
///   - `on_entity_extracted` >= 1 (the exact assertion `golden_path_smoke` makes);
///   - `on_edge_added("mention")` >= 1 (now that the namespaced edge-insert bug
///     below is fixed);
///   - the `Extracting` stage transition (extraction began);
///   - the `EntitiesReady` + `Complete` terminal transitions (success path).
///
/// Edge-insert fix (Quinn FIX 4 / Migration 006): `insert_episodic_edge` now
/// takes the entity's `entity_group_id`. Previously the INSERT omitted that
/// column, so it defaulted to `'default'`; the composite FK `(entity_id,
/// entity_group_id) REFERENCES entities(id, group_id)` (migration 006) then had
/// NO parent row for a non-`'default'` namespace. This was a default-vs-namespace
/// MISMATCH — NOT a NULL / missing value, since the column is
/// `entity_group_id TEXT NOT NULL DEFAULT 'default'` (migrations.rs:2973). The
/// edge silently failed to insert in namespaced mode, so `on_edge_added` never
/// fired for namespaced ingests. `ingest_with` now threads the real ingest
/// `group_id` into the call, the composite FK resolves, the edge persists, and
/// `on_edge_added` fires — which this test now asserts under the namespaced
/// `"sink-regression"` namespace.
#[tokio::test]
async fn production_ingest_fires_sink_callbacks() {
    let mem = Memory::open(unique_db("inline"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_extractor(Arc::new(TwoEntityExtractor))
        .default_namespace(Namespace::new("sink-regression"))
        .await
        .expect("Memory::open must succeed");

    let sink = Arc::new(CapturingSink::default());

    // Drive REAL ingestion via the public facade — the inline path
    // (no .no_wait()) that golden_path_smoke uses.
    mem.remember("Alice works at Acme Corp in London.")
        .from_chat("sink-regression-session")
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await
        .expect("remember must succeed");

    // 1) on_entity_extracted fired at least once — the load-bearing assertion
    //    that fb85ba8 silently broke (got 0 before the cause-fix).
    let entities = sink.entities.lock().unwrap();
    assert!(
        !entities.is_empty(),
        "PRODUCTION ingest must fire on_entity_extracted >= 1 (ADR-052 Gap 1); got 0. \
         The fire-sites in Engine::ingest_with were dropped — re-wire them."
    );

    // 2) on_edge_added("mention") fired at least once — the namespaced
    //    edge-insert regression guard (Quinn FIX 4). Before threading
    //    entity_group_id into insert_episodic_edge, the composite FK
    //    (entity_id, entity_group_id) -> entities(id, group_id) had no parent
    //    row under the "sink-regression" namespace (default-vs-namespace
    //    mismatch), the edge silently failed to insert, and this fire-site
    //    (guarded on insert success) never fired. With the fix the edge persists
    //    and the mention edge fires.
    let edges = sink.edges.lock().unwrap();
    assert!(
        edges.iter().any(|(_, _, predicate)| predicate == "mention"),
        "PRODUCTION namespaced ingest must fire on_edge_added(\"mention\") >= 1 \
         (Quinn FIX 4 / Migration 006 composite FK); got {edges:?}. \
         insert_episodic_edge must thread entity_group_id so the FK resolves."
    );

    // 3) Stage transitions captured: Extracting (began) + EntitiesReady + Complete
    //    (terminal success). These are the inline-path stage sequence.
    let stages = sink.stages.lock().unwrap();
    assert!(
        stages.contains(&IngestStatus::Extracting),
        "expected on_stage_change(Extracting); got {stages:?}"
    );
    assert!(
        stages.contains(&IngestStatus::EntitiesReady),
        "expected on_stage_change(EntitiesReady); got {stages:?}"
    );
    assert!(
        stages.contains(&IngestStatus::Complete),
        "expected on_stage_change(Complete); got {stages:?}"
    );

    // 4) Stage ordering invariant: Extracting precedes EntitiesReady precedes
    //    Complete (the documented IngestStatus success sequence).
    let pos = |target: &IngestStatus| stages.iter().position(|s| s == target);
    let (extracting, ready, complete) = (
        pos(&IngestStatus::Extracting),
        pos(&IngestStatus::EntitiesReady),
        pos(&IngestStatus::Complete),
    );
    assert!(
        extracting < ready && ready < complete,
        "stage transitions must fire in order Extracting < EntitiesReady < Complete; got {stages:?}"
    );

    // NOTE: on_edge_added("mention") IS now asserted above (item 2). The prior
    // version documented-around a namespaced edge-insert bug; that bug is fixed
    // (Quinn FIX 4 — insert_episodic_edge threads entity_group_id), so the edge
    // persists and the fire-site fires even on the mock path. The on_edge_added
    // ("object") variant is NOT asserted here: the mock extractor returns
    // facts: vec![], so the Phase 2 fact loop (the only object-edge producer) is
    // a no-op by construction; object edges are exercised by golden_path_smoke.
}

/// Production ingestion that fails AFTER `Extracting` fires MUST reach a terminal
/// `Failed` transition — never `Extracting` then silence (ADR-051 §4 / Quinn
/// MED-01). The `fire_failed` exit in `Engine::ingest_with` wraps the whole
/// post-`Extracting` body in one `Result`; on ANY `Err` it fires `Failed` once
/// before re-propagating. This guards that invariant: a `FailingExtractor` errors
/// at `extractor.extract(..).await?` (a post-`Extracting` `?`-step), and the sink
/// MUST have recorded `Extracting` THEN `Failed`, and the error MUST surface to
/// the caller (not be swallowed).
#[tokio::test]
async fn production_ingest_fires_failed_on_extractor_error() {
    let mem = Memory::open(unique_db("fail"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_extractor(Arc::new(FailingExtractor))
        .default_namespace(Namespace::new("sink-regression-fail"))
        .await
        .expect("Memory::open must succeed");

    let sink = Arc::new(CapturingSink::default());

    // Drive REAL inline ingestion; the extractor will Err mid-pipeline.
    let result = mem
        .remember("Alice works at Acme Corp in London.")
        .from_chat("sink-regression-fail-session")
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await;

    // 1) The error MUST surface to the caller — `fire_failed` does NOT swallow it.
    assert!(
        result.is_err(),
        "extractor failure must propagate to the caller; got Ok"
    );

    let stages = sink.stages.lock().unwrap();

    // 2) Extracting fired (extraction began before the failing step).
    assert!(
        stages.contains(&IngestStatus::Extracting),
        "expected on_stage_change(Extracting) before the failure; got {stages:?}"
    );

    // 3) A terminal Failed transition fired — NOT silence after Extracting.
    //    (MED-01: the previous code only fired Failed for inner-`'phases` errors;
    //    an extractor Err propagated via `?` left the sink stuck at Extracting.)
    assert!(
        stages
            .iter()
            .any(|s| matches!(s, IngestStatus::Failed(_))),
        "expected a terminal on_stage_change(Failed(..)) after Extracting; got {stages:?}"
    );

    // 4) Ordering: Extracting precedes Failed, and NO success terminal fired.
    let extracting = stages.iter().position(|s| s == &IngestStatus::Extracting);
    let failed = stages
        .iter()
        .position(|s| matches!(s, IngestStatus::Failed(_)));
    assert!(
        extracting < failed,
        "Extracting must precede Failed; got {stages:?}"
    );
    assert!(
        !stages.contains(&IngestStatus::EntitiesReady)
            && !stages.contains(&IngestStatus::Complete),
        "failure path must NOT fire the success terminals (EntitiesReady/Complete); got {stages:?}"
    );
}
