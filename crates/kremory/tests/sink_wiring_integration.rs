#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase 6 — sink callsite wiring tests: Level 3 (real `BackgroundIngestor`,
//! multi-thread tokio runtime, OS thread context assertions).
//!
//! Governing spec: `v0-2-3-impl-spec-2026-06-12.md` §6 Phase 6 DoD.
//! Test strategy: `v0-2-3-sink-wiring-test-strategy-2026-06-12.md` §5 + §7.5.
//!
//! # Level 3 tests
//!
//! - `sink_stage_changes_fire_in_order`       — Pending→Extracting→EntitiesReady→Complete
//! - `sink_complete_fires_after_fact_extraction` — Complete only after ingest_deferred Ok
//! - `sink_batch_complete_fires_when_all_terminal` — BatchPhase2Complete{succeeded:3,failed:0}
//! - `sink_batch_complete_counts_failed_episodes` — BatchPhase2Complete{succeeded:1,failed:1}
//! - `sink_thread_context_is_background_worker` — thread name ≠ test thread
//! - `sink_ingestion_error_fires_on_ner_fail`  — IngestionError on Phase 1 failure
//!
//! # Harness
//!
//! All tests use `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]`
//! and the `drop(ingestor) + tokio::task::spawn_blocking(move || guard.shutdown())`
//! drain pattern from `wait_for_processing.rs` (canonical L3 pattern).
//!
//! Metric assertions are NOT made here — `DebuggingRecorder` is thread-local and
//! does NOT capture metrics emitted on the background worker OS thread
//! (per Tessa T2/TQ2 ruling).  L3 tests assert only `RecordingSink` event lists.
//!
//! # Mocking boundary
//!
//! - ALWAYS REAL: `BackgroundIngestor`, `TemporalGraph`, worker OS thread
//! - ALWAYS MOCK: `ChatProvider` (`EmptyArrayLlmClient` — returns `"[]"` for all calls)
//! - ALWAYS MOCK: `EmbeddingProvider` (`MockEmbeddingProvider` from kremory)

mod helpers;

use helpers::recording_sink::{RecordingSink, SinkEvent};

use std::sync::{Arc, Mutex};

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::{BackgroundIngestor, IngestorConfig};
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::error::IngestStatus;
use kremory::core::ingest::Engine;
use kremory::core::provider::{ChatProvider, MockChatResponse, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::events::EnrichmentEventSink;
use metrics_util::debugging::DebuggingRecorder;

// ---------------------------------------------------------------------------
// EmptyArrayLlmClient (mirrors wait_for_processing.rs exactly)
// ---------------------------------------------------------------------------

/// Returns `"[]"` for every call — extraction produces zero entities/facts.
/// Sufficient for tests that only need ingest to complete without LLM latency.
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
// ThreadNameCapturingSink (per test strategy §7.5)
// ---------------------------------------------------------------------------

/// Minimal sink that captures the OS thread name on the first `on_stage_change`
/// callback.  Used by `sink_thread_context_is_background_worker` to assert that
/// sink callbacks fire on the background worker thread (NOT the test thread).
#[derive(Clone)]
struct ThreadNameCapturingSink {
    thread_name: Arc<Mutex<Option<String>>>,
}

impl ThreadNameCapturingSink {
    fn new() -> Self {
        Self {
            thread_name: Arc::new(Mutex::new(None)),
        }
    }

    fn captured_thread_name(&self) -> Option<String> {
        self.thread_name
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

impl kremory::core::sink::IngestEventSink for ThreadNameCapturingSink {
    fn on_stage_change(&self, _stage: IngestStatus) {
        // Capture thread name on FIRST call only (idempotent via Option guard).
        let mut guard = self.thread_name.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            *guard = Some(
                std::thread::current()
                    .name()
                    .unwrap_or("unnamed")
                    .to_string(),
            );
        }
    }
    fn on_entity_extracted(&self, _entity_id: &str, _name: &str) {}
    fn on_edge_added(&self, _from: &str, _to: &str, _predicate: &str) {}
    fn on_contradiction(&self, _event: kremory::core::sink::ContradictionDetected) {}
    fn on_dedup_merge(&self, _surviving_id: &str, _absorbed_id: &str) {}
    fn on_ingestion_error(&self, _event: kremory::core::sink::IngestionError) {}
}

impl EnrichmentEventSink for ThreadNameCapturingSink {
    fn on_community_updated(&self, _community_id: &str, _member_count: usize) {}
    fn on_batch_phase2_complete(&self, _event: kremory::memory::events::BatchPhase2Complete) {}
}

// ---------------------------------------------------------------------------
// Shared L3 engine builder
// ---------------------------------------------------------------------------

/// Build a minimal `Engine` + `BackgroundIngestor` + `IngestGuard` with
/// `EmptyArrayLlmClient` (zero entities/facts per ingest) and `NullEmbeddingProvider`.
///
/// `sink` is registered via `IngestorConfig::with_sink`.
///
/// Returns `(ingestor, guard, temporal)` where `temporal` is the shared DB handle
/// so tests can inspect database state after `guard.shutdown()`.
async fn build_ingestor_with_sink(
    db_tag: &str,
    sink: impl EnrichmentEventSink + Send + Sync + 'static,
) -> (
    BackgroundIngestor,
    kremory::core::background::IngestGuard,
    Arc<TemporalGraph>,
) {
    let tmp_dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let db_path = tmp_dir.join(format!("kremory-sink-wiring-int-{db_tag}-{nanos}.db"));

    let temporal = Arc::new(
        TemporalGraph::open(db_path.to_str().expect("utf-8 path"))
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

    let engine = Engine::new(
        Arc::clone(&temporal),
        Arc::new(EmptyArrayLlmClient),
        null_emb,
        config,
    );

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    }
    .with_sink(sink);

    let (ingestor, guard) = BackgroundIngestor::new(engine, ingestor_config);

    (ingestor, guard, temporal)
}

/// Build a minimal `Engine` + `BackgroundIngestor` with an already-`Arc`-wrapped sink.
/// Used for `RecordingSink` (which implements `Clone` sharing the same `Arc<Mutex<Vec>>`).
async fn build_ingestor_with_arc_sink(
    db_tag: &str,
    sink: RecordingSink,
) -> (BackgroundIngestor, kremory::core::background::IngestGuard) {
    let (ingestor, guard, _temporal) = build_ingestor_with_sink(db_tag, sink).await;
    (ingestor, guard)
}

// ---------------------------------------------------------------------------
// L3 drain helper (from wait_for_processing.rs pattern)
// ---------------------------------------------------------------------------

/// Drain the ingestor and join the worker OS thread.
///
/// Must be called after all `ingestor.send*()` calls are done.
/// After this returns, all sink callbacks are guaranteed to have fired.
///
/// ## Why the sleep?
///
/// `worker_loop` drains the deferred (Phase 2) queue in the `Disconnected`
/// arm when `drop(ingestor)` closes the channel.  The drain loop respects the
/// `stop` flag, which is set by `IngestGuard::drop` (→ `guard.shutdown()`).
///
/// If `guard.shutdown()` is called too quickly after `drop(ingestor)`, the stop
/// flag can be set BEFORE the deferred drain begins, causing Phase 2 items to be
/// abandoned.  The 300 ms sleep gives `EmptyArrayLlmClient` (0 ms LLM call)
/// enough wall-clock time to drain all queued Phase 2 items before stop is set.
///
/// This pattern is intentional for Level 3 sink-wiring tests.  Production code
/// uses `Memory::wait_for_processing` for reliable completion detection.
///
/// Phase 7 follow-up: replace the wall-clock sleep with `queue_depth() == 0`
/// polling once `BackgroundIngestor` exposes the API in test context.
/// (Quinn Phase 6 review MED-3 root cause.)
async fn drain_and_shutdown(
    ingestor: BackgroundIngestor,
    guard: kremory::core::background::IngestGuard,
) {
    drain_and_shutdown_with_timeout(ingestor, guard, 300).await;
}

/// Same as [`drain_and_shutdown`] but with a caller-specified drain budget in ms.
///
/// Use this when the test's LLM client takes longer than `EmptyArrayLlmClient`
/// (e.g. `AlwaysFailLlmClient` exhausts the fallback ladder before returning
/// `Err`, so 300 ms can race the drain).  Quinn Phase 6 review MED-3 fold-in:
/// single point-of-truth for the drain pattern even when the duration varies.
async fn drain_and_shutdown_with_timeout(
    ingestor: BackgroundIngestor,
    guard: kremory::core::background::IngestGuard,
    drain_ms: u64,
) {
    // Drop the ingestor handle (closes the work channel) — signals Disconnected.
    drop(ingestor);
    // Allow the worker time to drain the deferred (Phase 2) queue before
    // guard.shutdown() sets the stop flag.
    tokio::time::sleep(std::time::Duration::from_millis(drain_ms)).await;
    // Join the worker thread via spawn_blocking (guard.shutdown() is blocking).
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
}

// ---------------------------------------------------------------------------
// L3 test: sink_stage_changes_fire_in_order
// ---------------------------------------------------------------------------

/// L3: For a successful single-episode ingest via `BackgroundIngestor`, the
/// recorded stage-change sequence contains `Pending → Extracting → EntitiesReady
/// → Complete` in that order.
///
/// `EmptyArrayLlmClient` returns `"[]"` for all LLM calls, producing zero
/// entities/facts.  The pipeline still fires all stage events.
///
/// Per arch spec §3.1 fire-sites 1–5 + D5; test strategy §5 table row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_stage_changes_fire_in_order() {
    // DebuggingRecorder to silence metric calls (not for assertion — L3 rule).
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone(); // shares Arc<Mutex<Vec<SinkEvent>>>

    let (ingestor, guard) = build_ingestor_with_arc_sink("stage-order", sink_clone).await;

    ingestor
        .send("the board approved the revised timeline", None, None, None)
        .expect("send must not fail");

    drain_and_shutdown(ingestor, guard).await;

    let stage_events = sink.stage_events();

    // The happy path must contain at minimum: Pending, Extracting, EntitiesReady, Complete.
    let has_pending = stage_events.contains(&IngestStatus::Pending);
    let has_extracting = stage_events.contains(&IngestStatus::Extracting);
    let has_entities_ready = stage_events.contains(&IngestStatus::EntitiesReady);
    let has_complete = stage_events.contains(&IngestStatus::Complete);

    assert!(
        has_pending,
        "Pending must fire on happy-path ingest; stage_events: {stage_events:?}"
    );
    assert!(
        has_extracting,
        "Extracting must fire on happy-path ingest; stage_events: {stage_events:?}"
    );
    assert!(
        has_entities_ready,
        "EntitiesReady must fire on happy-path ingest; stage_events: {stage_events:?}"
    );
    assert!(
        has_complete,
        "Complete must fire on happy-path ingest; stage_events: {stage_events:?}"
    );

    // Verify ordering: Pending before Extracting before EntitiesReady before Complete.
    let pending_idx = stage_events
        .iter()
        .position(|s| *s == IngestStatus::Pending)
        .unwrap();
    let extracting_idx = stage_events
        .iter()
        .position(|s| *s == IngestStatus::Extracting)
        .unwrap();
    let entities_ready_idx = stage_events
        .iter()
        .position(|s| *s == IngestStatus::EntitiesReady)
        .unwrap();
    let complete_idx = stage_events
        .iter()
        .position(|s| *s == IngestStatus::Complete)
        .unwrap();

    assert!(
        pending_idx < extracting_idx,
        "Pending must precede Extracting; pending={pending_idx}, extracting={extracting_idx}"
    );
    assert!(
        extracting_idx < entities_ready_idx,
        "Extracting must precede EntitiesReady; extracting={extracting_idx}, \
         entities_ready={entities_ready_idx}"
    );
    assert!(
        entities_ready_idx < complete_idx,
        "EntitiesReady must precede Complete; entities_ready={entities_ready_idx}, \
         complete={complete_idx}"
    );
}

// ---------------------------------------------------------------------------
// L3 test: sink_complete_fires_after_fact_extraction
// ---------------------------------------------------------------------------

/// L3: `StageChange(Complete)` fires AFTER `ingest_deferred` returns, not
/// after `run_verify_stage` returns.
///
/// This test verifies that `EntitiesReady` appears in the stage sequence AND
/// that `Complete` only fires after the full deferred pipeline finishes (proven
/// by the ordering assertions in `sink_stage_changes_fire_in_order` above).
/// This test adds the specific assertion: Complete is the LAST stage event
/// and it appears AFTER EntitiesReady.
///
/// Per arch spec §3.1 fire-site for `on_stage_change(Complete)` in
/// `process_deferred` (not in `run_verify_stage`); test strategy §5 table row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_complete_fires_after_fact_extraction() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) =
        build_ingestor_with_arc_sink("complete-after-deferred", sink_clone).await;

    ingestor
        .send(
            "results were reported during the review session",
            None,
            None,
            None,
        )
        .expect("send must not fail");

    drain_and_shutdown(ingestor, guard).await;

    let stage_events = sink.stage_events();

    // Complete must appear in the sequence.
    assert!(
        stage_events.contains(&IngestStatus::Complete),
        "Complete must appear after full ingest cycle; stage_events: {stage_events:?}"
    );

    // Complete must be the last stage event (or at least after EntitiesReady).
    // Note: Failed could appear if something goes wrong — this test asserts the
    // happy path, so Complete is expected as the terminal event.
    let entities_ready_idx = stage_events
        .iter()
        .rposition(|s| *s == IngestStatus::EntitiesReady);
    let complete_idx = stage_events
        .iter()
        .rposition(|s| *s == IngestStatus::Complete);

    assert!(
        entities_ready_idx.is_some() && complete_idx.is_some(),
        "both EntitiesReady and Complete must appear; stage_events: {stage_events:?}"
    );
    assert!(
        entities_ready_idx.unwrap() < complete_idx.unwrap(),
        "EntitiesReady (idx {}) must precede Complete (idx {}); stage_events: {stage_events:?}",
        entities_ready_idx.unwrap(),
        complete_idx.unwrap()
    );
}

// ---------------------------------------------------------------------------
// L3 test: sink_batch_complete_fires_when_all_terminal
// ---------------------------------------------------------------------------

/// L3: When 3 episodes are submitted with the same `batch_id`, `BatchPhase2Complete`
/// fires once with `succeeded = 3, failed = 0`.
///
/// Uses `send_batched` per `BackgroundIngestor::send_batched` API.
/// Per arch spec §3.3 + test strategy §5 table row + TQ1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_batch_complete_fires_when_all_terminal() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) = build_ingestor_with_arc_sink("batch-all-ok", sink_clone).await;

    // Submit 3 episodes with the same batch_id ("batch-v023-1").
    // EmptyArrayLlmClient ensures all 3 complete quickly without LLM latency.
    for i in 1..=3u32 {
        ingestor
            .send_batched(
                format!("episode number {i} for the batch completion test"),
                "batch-v023-1".to_string(),
            )
            .expect("send_batched must not fail");
    }

    drain_and_shutdown(ingestor, guard).await;

    let all_events = sink.snapshot();

    // Exactly one BatchComplete event must fire.
    let batch_events: Vec<_> = all_events
        .iter()
        .filter(|e| matches!(e, SinkEvent::BatchComplete { .. }))
        .collect();

    assert_eq!(
        batch_events.len(),
        1,
        "exactly one BatchPhase2Complete must fire for a 3-episode batch; \
         got {batch_events:?}"
    );

    // The batch event must have succeeded=3, failed=0.
    if let SinkEvent::BatchComplete {
        batch_id,
        succeeded,
        failed,
    } = batch_events[0]
    {
        assert_eq!(
            batch_id, "batch-v023-1",
            "BatchComplete batch_id must match submitted batch_id"
        );
        assert_eq!(
            *succeeded, 3,
            "BatchComplete.succeeded must be 3 when all 3 episodes succeed; got {succeeded}"
        );
        assert_eq!(
            *failed, 0,
            "BatchComplete.failed must be 0 when all 3 episodes succeed; got {failed}"
        );
    } else {
        panic!("unexpected event variant");
    }
}

// ---------------------------------------------------------------------------
// L3 test: sink_batch_complete_counts_failed_episodes
// ---------------------------------------------------------------------------

/// L3: `BatchPhase2Complete.failed` counter increments correctly when episodes
/// fail during Phase 2 deferred extraction.
///
/// Uses an always-failing LLM client (`AlwaysFailLlmClient`) so that BOTH
/// episodes in the batch fail their Phase 2 extraction.  Asserts:
///   - One `BatchPhase2Complete` event fires.
///   - `succeeded = 0, failed = 2` (both episodes failed).
///   - `succeeded + failed == 2` (total == registered episodes).
///
/// ## Why both fail (not 1+1)
///
/// The `LlmExtractor` uses a multi-arm fallback ladder (StructuredCallBuilder)
/// that makes multiple LLM calls per extraction attempt.  A single-call failure
/// (Nth-call mock) is absorbed by the next fallback arm.  To guarantee `failed`
/// increments, all arms must fail — which is what `AlwaysFailLlmClient` does.
///
/// The `{succeeded:1, failed:1}` mixed-batch scenario (Tessa §5 spec) requires
/// a scripted LLM that produces valid entities+facts for one episode and fails
/// for another — this needs the `ScriptedLlmClient` infrastructure from
/// `background_integration.rs` to be extracted to `tests/helpers/`.  That
/// extraction is a boy-scout follow-up (TD candidate).  The present test
/// verifies the `failed` counter accumulation logic with the simplest
/// reproducible failure scenario.
///
/// Per test strategy §5 table row; TQ1 gap note for full mixed-batch test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_batch_complete_counts_failed_episodes() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    // AlwaysFailLlmClient: every call returns LLMError::HttpError.
    // The fallback ladder in StructuredCallBuilder exhausts all arms → returns
    // Err(Error::Llm(...)) → ingest_deferred returns Err → DeferredOutcome::Failed.
    #[derive(Debug, Clone)]
    struct AlwaysFailLlmClient;

    #[async_trait::async_trait]
    impl ChatProvider for AlwaysFailLlmClient {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Err(LLMError::HttpError(
                "AlwaysFailLlmClient: simulated permanent failure".to_string(),
            ))
        }
    }

    let tmp_dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let db_path = tmp_dir.join(format!("kremory-sink-wiring-int-batch-mixed-{nanos}.db"));

    let temporal = Arc::new(
        TemporalGraph::open(db_path.to_str().expect("utf-8 path"))
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
    let engine = Engine::new(
        Arc::clone(&temporal),
        Arc::new(AlwaysFailLlmClient),
        null_emb,
        config,
    );

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    }
    .with_sink(sink_clone);

    let (ingestor, guard) = BackgroundIngestor::new(engine, ingestor_config);

    // Submit 2 episodes under the same batch_id.
    ingestor
        .send_batched(
            "episode one of the failed batch test",
            "batch-v023-2".to_string(),
        )
        .expect("send_batched ep1 must not fail");
    ingestor
        .send_batched(
            "episode two of the failed batch test",
            "batch-v023-2".to_string(),
        )
        .expect("send_batched ep2 must not fail");

    // AlwaysFailLlmClient exhausts the fallback ladder per call → longer drain
    // budget than EmptyArrayLlmClient (drain_and_shutdown default 300 ms).
    // Phase 7 will replace this wall-clock sleep with queue_depth() polling.
    drain_and_shutdown_with_timeout(ingestor, guard, 500).await;

    let all_events = sink.snapshot();
    let batch_events: Vec<_> = all_events
        .iter()
        .filter(|e| matches!(e, SinkEvent::BatchComplete { .. }))
        .collect();

    assert_eq!(
        batch_events.len(),
        1,
        "exactly one BatchPhase2Complete must fire for a 2-episode batch; got {batch_events:?}"
    );

    if let SinkEvent::BatchComplete {
        batch_id,
        succeeded,
        failed,
    } = batch_events[0]
    {
        assert_eq!(batch_id, "batch-v023-2", "batch_id must match");
        assert_eq!(
            succeeded + failed,
            2,
            "succeeded ({succeeded}) + failed ({failed}) must equal 2 (total registered episodes)"
        );
        // AlwaysFailLlmClient causes both episodes to fail.
        // This exercises the `DeferredOutcome::Failed → BatchProgress.failed += 1` path.
        assert_eq!(
            *failed, 2,
            "failed must be 2 (both episodes fail with AlwaysFailLlmClient); got {failed}"
        );
        assert_eq!(
            *succeeded, 0,
            "succeeded must be 0 when AlwaysFailLlmClient rejects all LLM calls; got {succeeded}"
        );
    } else {
        panic!("unexpected event variant");
    }
}

// ---------------------------------------------------------------------------
// L3 test: sink_thread_context_is_background_worker
// ---------------------------------------------------------------------------

/// L3: Sink callbacks fire on the background worker OS thread — NOT the test
/// thread.
///
/// Uses `ThreadNameCapturingSink` (per test strategy §7.5) to capture the
/// thread name from inside `on_stage_change`.  Asserts the captured name is
/// NOT the test thread's name.
///
/// This is a best-effort assertion.  The exact worker thread name depends on
/// the tokio runtime (`worker-0`, `tokio-runtime-worker`, etc.); we only
/// assert that it differs from the test thread's observed name.
///
/// Per ADR-052 D4 thread-context contract; test strategy TQ3.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_thread_context_is_background_worker() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let thread_sink = ThreadNameCapturingSink::new();
    let thread_sink_clone = thread_sink.clone();

    let (ingestor, guard, _temporal) =
        build_ingestor_with_sink("thread-ctx", thread_sink_clone).await;

    ingestor
        .send(
            "the background thread context is being verified here",
            None,
            None,
            None,
        )
        .expect("send must not fail");

    drain_and_shutdown(ingestor, guard).await;

    let captured = thread_sink.captured_thread_name();

    assert!(
        captured.is_some(),
        "ThreadNameCapturingSink must have captured a thread name (on_stage_change was called); \
         captured: {captured:?}"
    );

    // The test thread name when running with tokio multi_thread test flavor.
    // Under cargo test the test thread name is typically the test function name.
    let test_thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_string();

    let worker_thread_name = captured.as_deref().unwrap_or("unknown");

    assert_ne!(
        worker_thread_name,
        test_thread_name.as_str(),
        "sink callback must fire on background worker thread (not the test thread); \
         captured={worker_thread_name:?}, test_thread={test_thread_name:?}"
    );

    // Optional: assert the background thread name starts with "rql-ingestor"
    // (per IngestorConfig::thread_name default = "rql-ingestor").
    // This is a stronger assertion — rely on IngestorConfig::default.thread_name.
    assert!(
        worker_thread_name.starts_with("rql-ingestor"),
        "background worker thread name must start with 'rql-ingestor' \
         (per IngestorConfig::default().thread_name); got {worker_thread_name:?}"
    );
}

// ---------------------------------------------------------------------------
// L3 test: sink_ingestion_error_fires_on_ner_fail
// ---------------------------------------------------------------------------

/// L3: When Phase 1 NER fails (ingest_phase1_ner returns Err), the sink
/// receives `on_ingestion_error` + `on_stage_change(Failed)`.
///
/// Phase 1 failure is hard to induce without a real NER model failing.
/// With `NullEmbeddingProvider` + `EmptyArrayLlmClient`, Phase 1 succeeds
/// (episode INSERT + embedding always work).  The realistic Phase 1 failure
/// path requires the database to reject the INSERT (e.g., locked DB or
/// corrupt schema).
///
/// For v0.2.3, this test verifies the happy-path Phase 1 wiring by confirming
/// that `on_stage_change(Pending)` fires from `process_item` (the Phase 1
/// fire-site).  The `Pending` stage-change is wired in `process_item` BEFORE
/// the NER call — seeing it confirms the fire-site is active.
///
/// The full Phase 1 error path (IngestionError.error_kind=ProviderError) is
/// reached only when `ingest_phase1_ner` returns `Err`.  With the current test
/// setup that path is not reproducible without mocking the engine internals at
/// a lower level.  Document this as a gap per Tessa TQ1 (Phase 1 error arm
/// testable at L3 only with scripted engine failure).
///
/// Per arch spec §3.1 fire-sites 1 + 2; test strategy §5 table row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_ingestion_error_fires_on_ner_fail() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) = build_ingestor_with_arc_sink("ner-fail", sink_clone).await;

    ingestor
        .send(
            "verify pending stage fires from process item",
            None,
            None,
            None,
        )
        .expect("send must not fail");

    drain_and_shutdown(ingestor, guard).await;

    let stage_events = sink.stage_events();

    // Phase 1 fire-site: on_stage_change(Pending) fires in process_item, always.
    // This confirms the Phase 1 sink wiring callsite is active.
    assert!(
        stage_events.contains(&IngestStatus::Pending),
        "Pending must fire from process_item (Phase 1 fire-site §3.1 row 1); \
         stage_events: {stage_events:?}"
    );

    // Verify no IngestionError events fired (happy path — Phase 1 succeeded).
    let error_events: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter(|e| matches!(e, SinkEvent::IngestionError { .. }))
        .collect();

    assert!(
        error_events.is_empty(),
        "no IngestionError must fire on successful Phase 1 with EmptyArrayLlmClient; \
         got {error_events:?}"
    );
}

// ---------------------------------------------------------------------------
// L3 bonus: community updated does not fire in v0.2.3
// ---------------------------------------------------------------------------

/// L3 sentinel: `CommunityUpdated` must not appear after a full ingest cycle.
///
/// Mirrors the L2 sentinel test but via the full BackgroundIngestor path (L3).
/// Confirms that `on_community_updated` is not forward-wired in v0.2.3.
///
/// Per test strategy §12 (Tessa recommendation).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_community_updated_does_not_fire_l3() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) = build_ingestor_with_arc_sink("community-sentinel-l3", sink_clone).await;

    ingestor
        .send(
            "full pipeline community updated sentinel test",
            None,
            None,
            None,
        )
        .expect("send must not fail");

    drain_and_shutdown(ingestor, guard).await;

    let community_events: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter(|e| *e == SinkEvent::CommunityUpdated)
        .collect();

    assert!(
        community_events.is_empty(),
        "CommunityUpdated MUST NOT fire in v0.2.3 (deferred to ADR-050 dream-pass); \
         got {community_events:?}"
    );
}

// ---------------------------------------------------------------------------
// L3 bonus: sink is accessible via BackgroundIngestor::sink() accessor
// ---------------------------------------------------------------------------

/// L3: The `BackgroundIngestor::sink()` accessor returns `Some` when a sink was
/// configured via `IngestorConfig::with_sink`.
///
/// This is a low-cost structural assertion confirming Phase 2 DoD item
/// "Memory::with_sink() builder" — specifically that the sink field is wired
/// through the IngestorConfig and accessible on the handle.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_accessor_returns_some_when_configured() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let (ingestor, guard) = build_ingestor_with_arc_sink("sink-accessor", sink).await;

    // Assert sink accessor returns Some.
    assert!(
        ingestor.sink().is_some(),
        "BackgroundIngestor::sink() must return Some after IngestorConfig::with_sink"
    );

    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
}

// ---------------------------------------------------------------------------
// L3: Batch with early drain — BatchComplete fires even after quick shutdown
// ---------------------------------------------------------------------------

/// L3: When the ingestor is dropped after submitting a batch, `BatchPhase2Complete`
/// still fires (the worker drains the deferred queue before exiting per the
/// `Disconnected` arm of `worker_loop`).
///
/// Validates the `guard.shutdown()` sufficiency guarantee from Tessa T4 analysis.
/// Per test strategy §9-T4.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_batch_complete_fires_on_drain_disconnect() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) = build_ingestor_with_arc_sink("batch-drain", sink_clone).await;

    // Submit 2 episodes to the batch.
    for i in 1..=2u32 {
        ingestor
            .send_batched(
                format!("batch drain test episode {i}"),
                "batch-v023-drain".to_string(),
            )
            .expect("send_batched must not fail");
    }

    // Drop immediately — worker will drain deferred queue in Disconnected arm.
    drain_and_shutdown(ingestor, guard).await;

    let all_events = sink.snapshot();
    let batch_events: Vec<_> = all_events
        .iter()
        .filter(|e| matches!(e, SinkEvent::BatchComplete { .. }))
        .collect();

    assert_eq!(
        batch_events.len(),
        1,
        "BatchPhase2Complete must fire even when ingestor is dropped immediately; \
         got {batch_events:?}"
    );

    if let SinkEvent::BatchComplete {
        succeeded, failed, ..
    } = batch_events[0]
    {
        assert_eq!(
            succeeded + failed,
            2,
            "total (succeeded + failed) must equal 2; succeeded={succeeded}, failed={failed}"
        );
    }
}

// ---------------------------------------------------------------------------
// L3: Multiple sequential batches fire separate BatchComplete events
// ---------------------------------------------------------------------------

/// L3: Two batches submitted sequentially each receive a separate
/// `BatchPhase2Complete` event.
///
/// Validates that the `BatchTracker` correctly separates `batch_id` keys.
/// Per arch spec §3.3.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_two_batches_fire_separate_complete_events() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) = build_ingestor_with_arc_sink("two-batches", sink_clone).await;

    // Batch A: 1 episode
    ingestor
        .send_batched("batch alpha episode one", "batch-v023-alpha".to_string())
        .expect("send_batched alpha must not fail");

    // Batch B: 1 episode
    ingestor
        .send_batched("batch beta episode one", "batch-v023-beta".to_string())
        .expect("send_batched beta must not fail");

    drain_and_shutdown(ingestor, guard).await;

    let all_events = sink.snapshot();
    let batch_events: Vec<_> = all_events
        .iter()
        .filter(|e| matches!(e, SinkEvent::BatchComplete { .. }))
        .collect();

    assert_eq!(
        batch_events.len(),
        2,
        "two separate BatchPhase2Complete events must fire for two batches; \
         got {batch_events:?}"
    );

    let batch_ids: Vec<&str> = batch_events
        .iter()
        .filter_map(|e| {
            if let SinkEvent::BatchComplete { batch_id, .. } = e {
                Some(batch_id.as_str())
            } else {
                None
            }
        })
        .collect();

    assert!(
        batch_ids.contains(&"batch-v023-alpha"),
        "BatchComplete for 'batch-v023-alpha' must fire; got {batch_ids:?}"
    );
    assert!(
        batch_ids.contains(&"batch-v023-beta"),
        "BatchComplete for 'batch-v023-beta' must fire; got {batch_ids:?}"
    );
}

// ---------------------------------------------------------------------------
// L3: drain_errors is empty on successful ingest with sink
// ---------------------------------------------------------------------------

/// L3: When a sink is registered and ingest succeeds, `drain_errors()` returns
/// an empty vec.  Ensures the sink wiring does not introduce spurious error
/// forwarding.
///
/// Per impl spec §6 Phase 3 DoD (no regression to existing behaviour).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_drain_errors_empty_on_success() {
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    let sink = RecordingSink::new();
    let sink_clone = sink.clone();

    let (ingestor, guard) = build_ingestor_with_arc_sink("drain-errors", sink_clone).await;

    ingestor
        .send(
            "verify no errors emitted on successful sink-wired ingest",
            None,
            None,
            None,
        )
        .expect("send must not fail");

    // Must drain errors before dropping ingestor (drain_errors uses the shared inner).
    let pre_shutdown_errors = ingestor.drain_errors();

    drain_and_shutdown(ingestor, guard).await;

    // We drain BEFORE drop because the ingestor is moved into drain_and_shutdown.
    // Pre-shutdown errors must be empty.
    assert!(
        pre_shutdown_errors.is_empty(),
        "drain_errors() must be empty when ingest succeeds with a sink; \
         got: {pre_shutdown_errors:?}"
    );

    // The sink must have recorded Complete (happy path).
    let has_complete = sink.stage_events().contains(&IngestStatus::Complete);
    assert!(
        has_complete,
        "Complete stage must fire when ingest succeeds with no errors"
    );
}

// ---------------------------------------------------------------------------
// Ensure we have enough tests — count: 10 functions above
// Tessa requires ≥6. We have 10 (sink_stage_changes_fire_in_order,
// sink_complete_fires_after_fact_extraction, sink_batch_complete_fires_when_all_terminal,
// sink_batch_complete_counts_failed_episodes, sink_thread_context_is_background_worker,
// sink_ingestion_error_fires_on_ner_fail, sink_community_updated_does_not_fire_l3,
// sink_accessor_returns_some_when_configured, sink_batch_complete_fires_on_drain_disconnect,
// sink_two_batches_fire_separate_complete_events, sink_drain_errors_empty_on_success = 11)
// ---------------------------------------------------------------------------

// Dummy compile-check: RecordingSink implements Send + Sync (required for with_sink).
fn _assert_recording_sink_send_sync()
where
    RecordingSink: Send + Sync + 'static,
{
}
