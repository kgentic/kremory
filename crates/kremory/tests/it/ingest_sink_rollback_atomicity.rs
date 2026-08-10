//! **DUR-7 regression pin** (V1-CANONICAL §4.2) — sink events and counters emitted
//! from `Engine::ingest_with` must describe rows that were actually **committed**.
//!
//! # The defect
//!
//! `ingest_with` opens one outer transaction (`ingest_with.rs:1064`) spanning ~1,086
//! lines and commits at `:2150` / rolls back at `:2180`. Inside that span it fires
//! **8 sink callbacks** and **15 counters** *inline*, at the moment each row is
//! written:
//!
//! | site | callback |
//! |---|---|
//! | `:1332` / `:1334` | `on_entity_extracted` + `on_edge_added("mention")` — merged arm |
//! | `:1402` / `:1404` | same pair — L4-merge arm |
//! | `:1590` / `:1592` | same pair — insert-new arm |
//! | `:1865` | `on_contradiction` |
//! | `:2119` | `on_edge_added("object")` |
//!
//! A sink callback cannot be rolled back. So when the transaction aborts, every
//! consumer that built state from those events is holding **references to entity and
//! edge ids that do not exist in the database** — and nothing ever tells them.
//!
//! This is a **correctness** defect for consumers, not a metrics-accuracy one. The
//! napi and MCP event surfaces are exactly such consumers.
//!
//! # How this test proves it
//!
//! `DROP TABLE facts_fts` makes `fts_search_facts` (`ingest_with.rs:1682`) fail. That
//! call sits in Phase 2's fact loop and propagates via `break 'phases Err(e)`, so:
//!
//! - Phase 1's entity loop has **already run in full**, firing the six entity/mention
//!   callbacks above;
//! - the phase result is `Err`, so `outer_guard.rollback()` runs and **no entity or
//!   edge is committed**.
//!
//! The assertion is then simply: **did the consumer hear about rows that do not
//! exist?** Before the fix it hears about all of them.
//!
//! Chosen deliberately over a `BEFORE INSERT` trigger on `facts`: fact-insert errors
//! are **swallowed** (`ingest_with.rs:2040-2085` counts them and continues), so a
//! trigger there aborts nothing and the test would pass vacuously against broken code.
//!
//! # Sensitivity — proven in BOTH directions
//!
//! `rolled_back_ingest_emits_no_sink_events` asserts events are ABSENT after a
//! rollback. On its own that is satisfiable by a sink that never fires at all — the
//! vacuity trap that made two of DUR-4's three instruments report green against
//! broken code. `the_same_fixture_does_emit_when_the_ingest_commits` is the
//! non-vacuity half: **same extractor, same fixture, fault removed**, and it asserts
//! the events DO fire. Neither test is meaningful without the other.

#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use kremory::core::error::IngestStatus;
use kremory::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::MockChatProvider;
use kremory::core::sink::{
    ContradictionDetected, IngestEventSink, IngestionError, OnEdgeAddedParams,
};
use kremory::memory::events::{BatchPhase2Complete, EnrichmentEventSink};
use kremory::{DynEmbeddingProvider, Memory, Namespace};

// ── Capturing sink ────────────────────────────────────────────────────────────

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
    fn on_edge_added(&self, params: OnEdgeAddedParams<'_>) {
        let OnEdgeAddedParams {
            from_entity_id: from,
            to_entity_id: to,
            predicate,
        } = params;
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

// ── Deterministic extractor: two entities AND one fact ────────────────────────
//
// The fact is load-bearing, unlike `sink_fires_through_ingest.rs`'s empty-facts
// extractor: without at least one fact the Phase 2 loop never runs, so
// `fts_search_facts` is never reached and the injected fault never fires.

struct TwoEntityOneFactExtractor;

impl EntityExtractor for TwoEntityOneFactExtractor {
    fn name(&self) -> &'static str {
        "dur7-two-entity-one-fact-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: vec![
                ExtractedEntity {
                    name: "Ada Lovelace".to_string(),
                    label: "Person".to_string(),
                    properties: serde_json::json!({"name": "Ada Lovelace"}),
                },
                ExtractedEntity {
                    name: "Analytical Engine".to_string(),
                    label: "Artifact".to_string(),
                    properties: serde_json::json!({"name": "Analytical Engine"}),
                },
            ],
            facts: vec![ExtractedFact {
                subject: "Ada Lovelace".to_string(),
                predicate: "wrote algorithms for".to_string(),
                object: "Analytical Engine".to_string(),
                is_entity_ref: true,
                confidence: 1.0,
                valid_at: None,
            }],
        })
    }
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_dur7_sink_rollback_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

async fn build_memory(tag: &str) -> Memory {
    Memory::open(unique_db(tag))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_extractor(Arc::new(TwoEntityOneFactExtractor))
        .default_namespace(Namespace::new("dur7"))
        .await
        .expect("Memory::open must succeed")
}

const TEXT: &str = "Ada Lovelace wrote algorithms for the Analytical Engine.";

// ── The regression guard ──────────────────────────────────────────────────────

/// A ROLLED-BACK ingest must emit NO entity/edge sink events.
///
/// Before the DUR-7 fix this fails with six recorded callbacks — three
/// `on_entity_extracted` and three `on_edge_added("mention")` — every one of them
/// naming an id that the rollback erased.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolled_back_ingest_emits_no_sink_events() {
    let mem = build_memory("rollback").await;
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");

    // Fault injection. `facts_fts` is a STANDALONE fts5 table (defs_h.rs:197), not
    // external-content, so dropping it genuinely removes the index rather than
    // leaving a view that reads through to `facts` — the read-through trap that made
    // DUR-4's second instrument report green.
    tg.conn
        .execute("DROP TABLE facts_fts", ())
        .await
        .expect("fault injection must succeed — the whole test rests on it");

    let sink = Arc::new(CapturingSink::default());
    let result = mem
        .remember(TEXT)
        .from_chat("dur7-session")
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await;

    // INSTRUMENT VALIDATION. If the fault did not actually abort the phase, the
    // ingest returns Ok, nothing rolls back, and every assertion below would be
    // asserting the absence of events that were legitimately never due.
    assert!(
        result.is_err(),
        "the injected fault must abort the ingest — an Ok here means \
         `fts_search_facts` never ran (or its error is now swallowed), so this test \
         is proving nothing. Re-derive the injection point before trusting a green."
    );

    // THE ASSERTION. Nothing was committed, so the consumer must have heard nothing.
    let entities = sink.entities.lock().unwrap();
    assert!(
        entities.is_empty(),
        "DUR-7: a rolled-back ingest fired on_entity_extracted for {} entity/entities \
         that DO NOT EXIST: {entities:?}.\n\
         Sink callbacks cannot be rolled back, so any consumer building state from \
         the sink (napi / MCP event surfaces) now holds references to uncommitted \
         rows and will never be told otherwise. Buffer the emissions during the \
         transaction and flush them after `outer_guard.commit()` succeeds.",
        entities.len()
    );

    let edges = sink.edges.lock().unwrap();
    assert!(
        edges.is_empty(),
        "DUR-7: a rolled-back ingest fired on_edge_added for {} edge(s) that DO NOT \
         EXIST: {edges:?}. Same cause and same fix as the entities above.",
        edges.len()
    );

    // The terminal-success transitions must NOT appear either: `EntitiesReady` and
    // `Complete` claim a durable write that did not happen. (These already fire
    // post-commit at `:2151-2153`, so this pins existing correct behaviour rather
    // than driving new work — it fails only if a future refactor moves them inside
    // the transaction along with everything else.)
    let stages = sink.stages.lock().unwrap();
    assert!(
        !stages.iter().any(|s| matches!(s, IngestStatus::Complete)),
        "a rolled-back ingest must never report Complete; got {stages:?}"
    );
}

/// NON-VACUITY HALF — same extractor, same fixture, **no fault**.
///
/// Without this, the test above is satisfiable by a sink that never fires at all,
/// which is precisely how a DUR-7 "fix" that simply deleted the fire-sites would go
/// green. Deleting them must fail HERE.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_same_fixture_does_emit_when_the_ingest_commits() {
    let mem = build_memory("happy").await;
    let sink = Arc::new(CapturingSink::default());

    mem.remember(TEXT)
        .from_chat("dur7-session")
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await
        .expect("the unfaulted ingest must succeed");

    let entities = sink.entities.lock().unwrap();
    assert!(
        !entities.is_empty(),
        "non-vacuity: this fixture MUST fire on_entity_extracted on the committed \
         path, otherwise `rolled_back_ingest_emits_no_sink_events` passes for the \
         wrong reason (a sink that never fires satisfies it trivially). Got 0 — \
         either the fire-sites were deleted rather than deferred, or the flush after \
         `outer_guard.commit()` was never wired."
    );

    let edges = sink.edges.lock().unwrap();
    assert!(
        edges.iter().any(|(_, _, predicate)| predicate == "mention"),
        "non-vacuity: this fixture MUST fire on_edge_added(\"mention\") on the \
         committed path; got {edges:?}"
    );

    let stages = sink.stages.lock().unwrap();
    assert!(
        stages.iter().any(|s| matches!(s, IngestStatus::Complete)),
        "the committed path must reach Complete; got {stages:?}"
    );
}
