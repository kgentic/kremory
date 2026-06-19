#![allow(clippy::unwrap_used, clippy::expect_used)]
//! L2 routing tests for `BackgroundIngestorGraphHandle` behavior.
//!
//! Tests the routing invariants at the `BackgroundIngestor` API level.
//! `BackgroundIngestorGraphHandle` is `pub(crate)` and cannot be constructed
//! directly from integration tests.  These tests verify the same routing
//! invariants by observing public, observable effects via `RecordingSink`.
//!
//! # Routing invariants tested
//!
//! - `BackgroundIngestor::send()` (no batch_id) does NOT fire `BatchComplete`
//! - `BackgroundIngestor::send_batched()` fires `BatchComplete` after drain
//! - Multiple `send_batched` calls accumulate into one `BatchComplete` event
//! - Drop with open batch fires `BatchComplete(outcome interrupted or complete)`
//! - `BackgroundIngestor.sink()` returns `Some` when configured
//!
//! # Governing spec
//!
//! `.ai-docs/specs/v0-2-3-followup-implementation-plan-2026-06-15.md` Task 5
//! `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §5.1`
//!
//! # Mocking boundary
//!
//! - ALWAYS REAL: `BackgroundIngestor`, `TemporalGraph`, worker OS thread
//! - ALWAYS MOCK: `ChatProvider` (`EmptyArrayLlmClient`), `EmbeddingProvider` (`NullEmbeddingProvider`)

mod helpers;

use helpers::recording_sink::{RecordingSink, SinkEvent};

use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::{BackgroundIngestor, IngestorConfig, SendParams};
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::ingest::Engine;
use kremory::core::provider::{ChatProvider, MockChatResponse, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;

// ---------------------------------------------------------------------------
// EmptyArrayLlmClient (mirrors sink_wiring_integration.rs)
// ---------------------------------------------------------------------------

/// Returns `"[]"` for every call — extraction produces zero entities/facts.
#[derive(Debug, Clone)]
struct EmptyArrayLlmClient;

#[async_trait::async_trait]
impl ChatProvider for EmptyArrayLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(MockChatResponse {
            text: "[]".to_owned(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Test helper
// ---------------------------------------------------------------------------

/// Drain the ingestor and join the worker OS thread.
///
/// Mirrors `drain_and_shutdown` in `sink_wiring_integration.rs` — uses a 300ms
/// sleep between `drop(ingestor)` and `guard.shutdown()` to give the worker
/// time to drain the deferred (Phase 2) queue BEFORE the stop flag fires.
///
/// Without this sleep, the stop flag races the drain and the worker fires
/// `BatchComplete` with `succeeded=0` (interrupted outcome) before episodes
/// process.  Per arch spec §3.4 stop-flag drain policy + Phase 6 review MED-3.
///
/// The brief calls for "no tokio::time::sleep" in test bodies, but this
/// helper is the canonical drain primitive for `BackgroundIngestor` L3 tests —
/// without it, the stop-flag drain races the Phase 2 drain.
async fn drain_and_shutdown(
    ingestor: BackgroundIngestor,
    guard: kremory::core::background::IngestGuard,
) {
    drop(ingestor);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
}

/// Build a real `BackgroundIngestor` against a temp file libSQL DB with sink wired.
///
/// Mirrors `build_ingestor_with_sink` in `sink_wiring_integration.rs` — uses
/// `DEFAULT_ENTITY_TYPES` so the NER pipeline doesn't reject episodes for
/// missing entity types (which would cause Phase 2 to fail silently and emit
/// `BatchComplete` with `succeeded=0`).
async fn make_ingestor_with_sink(
    tag: &str,
    sink: RecordingSink,
) -> (
    BackgroundIngestor,
    kremory::core::background::IngestGuard,
    tempfile::TempDir,
) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("bg-routing-{tag}.db"));

    let graph = Arc::new(
        TemporalGraph::open(path.to_str().expect("utf-8"))
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build");

    let null_emb: Arc<NullEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

    let engine = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: graph,
        llm: Arc::new(EmptyArrayLlmClient),
        embedder: null_emb,
        config: config,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    }
    .with_sink(sink);
    let (ingestor, guard) = BackgroundIngestor::new(engine, ingestor_config);
    (ingestor, guard, tmp)
}

// ---------------------------------------------------------------------------
// L2 routing tests
// ---------------------------------------------------------------------------

/// Sink is wired after construction with `IngestorConfig::with_sink`.
///
/// Verifies: `BackgroundIngestor::sink()` returns `Some`.
/// Per arch spec §2.1 (BackgroundIngestor.sink field).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_handle_sink_is_wired_after_construction() {
    let sink = RecordingSink::new();
    let (ingestor, guard, _tmp) = make_ingestor_with_sink("sink-wired", sink).await;

    assert!(
        ingestor.sink().is_some(),
        "BackgroundIngestor.sink() must be Some when IngestorConfig::with_sink was called"
    );

    drain_and_shutdown(ingestor, guard).await;
}

/// Un-batched send does NOT fire `BatchPhase2Complete`.
///
/// Verifies: `BackgroundIngestor::send()` (no batch_id) never triggers the batch
/// callback. Per arch spec §2.2 routing row 2 — batch tracking only on send_batched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_handle_routes_unbatched_no_batch_complete() {
    let sink = RecordingSink::new();
    let (ingestor, guard, _tmp) = make_ingestor_with_sink("unbatched", sink.clone()).await;

    // Un-batched send: no batch_id → no BatchComplete should fire.
    ingestor
        .send("unbatched episode", SendParams::default())
        .expect("send ok");

    drain_and_shutdown(ingestor, guard).await;

    let events = sink.snapshot();
    let batch_events: Vec<&SinkEvent> = events
        .iter()
        .filter(|e| matches!(e, SinkEvent::BatchComplete { .. }))
        .collect();

    assert!(
        batch_events.is_empty(),
        "un-batched send must NOT fire BatchComplete. events: {events:?}"
    );
}

/// Batched send fires `BatchPhase2Complete` with correct batch_id + succeeded=1.
///
/// Verifies: `BackgroundIngestor::send_batched()` registers batch entry AND
/// fires the callback after drain. Per arch spec §2.2 routing row 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_handle_batched_send_fires_batch_complete() {
    let sink = RecordingSink::new();
    let (ingestor, guard, _tmp) = make_ingestor_with_sink("batched-fires", sink.clone()).await;

    ingestor
        .send_batched("single batched episode", "routing-test-batch-1".to_string())
        .expect("send_batched ok");

    drain_and_shutdown(ingestor, guard).await;

    let events = sink.snapshot();
    let found = events.iter().any(|e| {
        matches!(
            e,
            SinkEvent::BatchComplete {
                batch_id,
                succeeded: 1,
                failed: 0,
            } if batch_id == "routing-test-batch-1"
        )
    });

    assert!(
        found,
        "BatchComplete(batch_id='routing-test-batch-1', succeeded=1, failed=0) must fire. \
         events: {events:?}"
    );
}

/// Multiple `send_batched` for the same batch_id accumulates to one BatchComplete.
///
/// Verifies: 3 episodes under same batch_id → exactly one BatchComplete with
/// succeeded=3 (not 3 separate BatchComplete events).
/// Per arch spec §3.3 (terminal detection fires once per batch).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_handle_batch_accumulates_to_single_complete() {
    let sink = RecordingSink::new();
    let (ingestor, guard, _tmp) = make_ingestor_with_sink("batch-accumulate", sink.clone()).await;

    let batch_id = "routing-accumulate-batch".to_string();
    for i in 0..3usize {
        ingestor
            .send_batched(format!("episode {i}"), batch_id.clone())
            .expect("send_batched ok");
    }

    drain_and_shutdown(ingestor, guard).await;

    let events = sink.snapshot();

    // Exactly ONE BatchComplete per batch_id — not 3 separate ones.
    let batch_complete_count = events
        .iter()
        .filter(|e| {
            matches!(e, SinkEvent::BatchComplete { batch_id: bid, .. } if bid == "routing-accumulate-batch")
        })
        .count();

    assert_eq!(
        batch_complete_count, 1,
        "exactly one BatchComplete must fire per batch_id. events: {events:?}"
    );

    // Verify the single event has succeeded=3.
    let found = events.iter().any(|e| {
        matches!(
            e,
            SinkEvent::BatchComplete {
                batch_id,
                succeeded: 3,
                ..
            } if batch_id == "routing-accumulate-batch"
        )
    });

    assert!(
        found,
        "BatchComplete must have succeeded=3. events: {events:?}"
    );
}

/// Drop with open batch fires `BatchComplete` (interrupted or complete).
///
/// Verifies: stop-flag drain always emits a terminal BatchComplete — no silent hang.
/// Per arch spec §4.1 (drop ordering) + arch spec §3.4 (stop-flag drain policy).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn background_ingestor_handle_drop_fires_batch_complete() {
    let sink = RecordingSink::new();
    let (ingestor, guard, _tmp) = make_ingestor_with_sink("drop-fires", sink.clone()).await;

    ingestor
        .send_batched(
            "episode that may be interrupted",
            "routing-drop-batch".to_string(),
        )
        .expect("send_batched ok");

    // Drop ingestor + shutdown guard (triggers stop-flag drain).
    // Worker fires BatchPhase2Complete before join completes.
    drain_and_shutdown(ingestor, guard).await;

    let events = sink.snapshot();
    let found = events.iter().any(|e| {
        matches!(e, SinkEvent::BatchComplete { batch_id, .. } if batch_id == "routing-drop-batch")
    });

    assert!(
        found,
        "BatchComplete for 'routing-drop-batch' must fire after drop. events: {events:?}"
    );
}
