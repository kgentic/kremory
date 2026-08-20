//! `BackgroundIngestor` + `IngestGuard` + send/queue operations.
//!
//! Sprint plan T2.1 / ADR-049 §Decision 6 §5.1 — Stage 1 enqueue module.
//!
//! Owns:
//! - [`BackgroundIngestor`] — cloneable handle; submits work via mpsc channel
//! - [`IngestGuard`]        — RAII shutdown; joins the worker thread on drop
//! - `Inner`               — shared state behind `Arc`
//!
//! The worker thread is spawned in [`BackgroundIngestor::new`] and delegates
//! to [`crate::core::background::deferred_pipeline::worker_loop`].

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use chrono::{DateTime, Utc};

use crate::core::config::ContentType;
use crate::core::ingest::Engine;
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::memory::events::EnrichmentEventSink;

use super::{
    batch_tracker::BatchTracker,
    deferred_pipeline::{worker_loop, WorkerLoopParams},
    IngestError, IngestRequest, IngestSendError, IngestorConfig,
};

// ---------------------------------------------------------------------------
// Inner
// ---------------------------------------------------------------------------

pub(super) struct Inner {
    /// Sender half of the work channel.
    pub(super) work_tx: std::sync::mpsc::SyncSender<IngestRequest>,
    /// Receiver half of the error channel, guarded so multiple clones can
    /// drain it from one caller at a time.
    pub(super) error_rx: Mutex<Receiver<IngestError>>,
    /// Tracks items currently sitting in the work channel.
    pub(super) queued: Arc<AtomicUsize>,
    /// Nominal capacity of the work channel (for error messages).
    pub(super) channel_capacity: usize,
}

// ---------------------------------------------------------------------------
// BackgroundIngestor
// ---------------------------------------------------------------------------

/// Non-blocking background ingestor.  Clone to share across threads.
///
/// The worker thread starts immediately when [`BackgroundIngestor::new`] is
/// called and runs until all clones of this handle are dropped, at which point
/// the work channel closes and the thread exits after processing remaining items.
///
/// Drop the companion [`IngestGuard`] to wait for the worker to finish.
///
/// ## Serialisation invariant
///
/// A single OS thread owns the `Engine` and processes all work items
/// sequentially on a multi-thread tokio runtime pinned to a single worker
/// (`new_multi_thread().worker_threads(1)`).
/// This means:
///
/// - Phase 1 NER ingest calls are serialised — no concurrent schema mutations.
/// - Phase 2 deferred LLM fact extraction calls are also serialised within the
///   same thread, awaited inline after NER-channel idle periods.
/// - The `Engine` is NOT wrapped in an `Arc` or `Mutex` — exclusive ownership
///   lives on the worker thread for the ingestor's lifetime.
///
/// Do **not** share the same `Engine` between a `BackgroundIngestor` and other
/// async tasks — move the engine into the ingestor and interact with the graph
/// through the facade `Memory` handle.
///
/// ## Event sink (ADR-052 Gap 1)
///
/// An optional [`EnrichmentEventSink`] can be attached at construction time via
/// [`IngestorConfig::with_sink`].  When set, the sink receives callbacks at each
/// pipeline stage on the background worker OS thread (D4 thread-context contract).
/// All existing code paths remain unaffected when `sink` is `None`.
/// Bundled (non-generic) parameters for [`BackgroundIngestor::send`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments). The generic
/// `text: impl Into<String>` stays a lead positional param on `send`.
///
/// `Default` yields all-`None` (equivalent to the prior `send(text, None, None,
/// None)` shorthand) — use `SendParams::default()` for a plain enqueue.
#[derive(Debug, Clone, Default)]
pub struct SendParams {
    pub reference_time: Option<DateTime<Utc>>,
    /// TD-187 Gap 1 (2026-08-20): the caller-DECLARED document anchor —
    /// mirrors [`super::IngestRequest::declared_reference_time`]. Distinct
    /// from `reference_time` above; see that field's doc comment for why the
    /// two must never be collapsed.
    pub declared_reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
    pub content_type: Option<ContentType>,
}

#[derive(Clone)]
pub struct BackgroundIngestor {
    pub(super) inner: Arc<Inner>,
    /// Sink for background pipeline stage-change / entity / error callbacks.
    ///
    /// `Arc` so `BackgroundIngestor` remains `Clone`.  `None` when no sink was
    /// configured — all code paths compile and behave identically to pre-v0.2.3.
    ///
    /// Per ADR-052 Gap 1; impl spec §3 Phase 2.
    pub(crate) sink: Option<Arc<dyn EnrichmentEventSink>>,
    /// Per-batch terminal-detection tracker.
    ///
    /// Maps `batch_id → BatchProgress`.  `Arc` so `BackgroundIngestor` stays
    /// `Clone`; `std::sync::Mutex` (not async) because critical sections are
    /// O(1) insert/increment — no `.await` is ever held across the lock.
    ///
    /// `total` is incremented under this mutex BEFORE `work_tx.try_send` so the
    /// race-safety invariant (arch spec §3.3) is maintained.
    ///
    /// The same `Arc` is cloned into `worker_loop` so both the caller side and
    /// the worker side share the same map.
    ///
    /// Per impl spec §6 Phase 4 DoD item 3.
    pub(crate) batch_tracker: BatchTracker,
}

impl BackgroundIngestor {
    /// Create an ingestor + guard.  The worker thread starts immediately.
    ///
    /// `graph` is moved into the worker thread; this is the only place where
    /// the generics `L` and `Emb` appear.
    pub fn new<L, Emb>(graph: Engine<L, Emb>, config: IngestorConfig) -> (Self, IngestGuard)
    where
        L: ChatProvider + 'static,
        Emb: EmbeddingProvider + 'static,
    {
        let (work_tx, work_rx) = sync_channel::<IngestRequest>(config.channel_capacity);
        let (error_tx, error_rx) = sync_channel::<IngestError>(config.error_channel_capacity);

        let queued = Arc::new(AtomicUsize::new(0));
        let queued_worker = Arc::clone(&queued);

        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);

        let deferred_enabled = config.deferred_extraction_enabled;
        let llm_rate_limit = config.llm_rate_limit;
        // Extract sink before config is consumed; clone into worker thread.
        let sink = config.sink.clone();
        let sink_for_worker = config.sink.clone();

        // Shared batch-terminal tracker (impl spec §6 Phase 4 DoD item 3).
        // Arc cloned into the worker so both handle and worker share the map.
        let batch_tracker: BatchTracker = Arc::new(Mutex::new(HashMap::new()));
        let batch_tracker_for_worker = Arc::clone(&batch_tracker);

        let parent_span = tracing::Span::current();
        let handle = thread::Builder::new()
            .name(config.thread_name.clone())
            .spawn(move || {
                let _enter = parent_span.enter();
                worker_loop(WorkerLoopParams {
                    graph,
                    work_rx,
                    error_tx,
                    queued: queued_worker,
                    stop: stop_worker,
                    deferred_enabled,
                    llm_rate_limit,
                    sink: sink_for_worker,
                    batch_tracker: batch_tracker_for_worker,
                });
            })
            .unwrap_or_else(|e| panic!("invariant: OS rejected rql-ingestor thread spawn: {e}"));

        let ingestor = BackgroundIngestor {
            inner: Arc::new(Inner {
                work_tx,
                error_rx: Mutex::new(error_rx),
                queued,
                channel_capacity: config.channel_capacity,
            }),
            sink,
            batch_tracker,
        };

        let guard = IngestGuard {
            stop,
            handle: Some(handle),
        };

        (ingestor, guard)
    }

    /// Enqueue text for background ingestion.  Returns immediately (<1 ms).
    ///
    /// Returns `Err(IngestSendError::Full)` when the channel is at capacity and
    /// `Err(IngestSendError::Disconnected)` when the worker thread has exited.
    pub fn send(&self, text: impl Into<String>, params: SendParams) -> Result<(), IngestSendError> {
        let SendParams {
            reference_time,
            declared_reference_time,
            group_id,
            content_type,
        } = params;
        self.enqueue_req(IngestRequest {
            text: text.into(),
            reference_time,
            declared_reference_time,
            group_id,
            content_type,
            batch_id: None,
        })
    }

    /// Enqueue text for background ingestion as part of a named batch.
    ///
    /// Like [`send`](Self::send) but associates the episode with `batch_id`.
    /// When all episodes in the batch reach Phase 2 terminal state,
    /// `on_batch_phase2_complete` fires on the configured sink.
    ///
    /// `reference_time`, `group_id`, and `content_type` default to `None`.
    /// Full-control batched enqueue is `pub(crate)`-internal; consumers needing
    /// non-default fields on a batched send should use [`Memory`](crate::Memory)
    /// once the facade-level batched method lands (Phase 7 follow-up — Quinn
    /// MED-3 / facade-gap; ADR-052 Gap 1 unresolved at v0.2.3).
    ///
    /// ## Race-safety invariant (arch spec §3.3)
    ///
    /// The `BatchProgress::total` counter is incremented **under the mutex
    /// BEFORE** `work_tx.try_send` fires.  This prevents a fast Phase 2
    /// completion from triggering premature terminal detection.
    ///
    /// ## Drop-before-complete contract (shipped in v0.2.3 Phase 7)
    ///
    /// When [`IngestGuard`] is dropped (stop flag set), the worker fires
    /// `on_batch_phase2_complete` with `outcome="interrupted"` for every batch
    /// that still has outstanding items at stop-flag drain time.  This guarantees
    /// no silent hang — the terminal event always fires.
    ///
    /// Consumers MUST treat `outcome="interrupted"` as terminal.  The
    /// `BatchPhase2Complete` payload reflects only the episodes that completed
    /// before the stop flag fired; `succeeded + failed + skipped ≤ total`.
    ///
    /// Per impl spec §6 Phase 4 DoD item 4 (Quinn MED-1/2/3 folded as
    /// doc-comment per `feedback_boy_scout_includes_quinn_low_findings`).
    pub fn send_batched(
        &self,
        text: impl Into<String>,
        batch_id: String,
    ) -> Result<(), IngestSendError> {
        self.enqueue_req(IngestRequest {
            text: text.into(),
            reference_time: None,
            // Genuinely None, not a gap: this method takes only `text` +
            // `batch_id`, so there is no caller-supplied value to thread.
            // Full-control batched callers needing this should use
            // `enqueue_req` directly (see this method's own doc comment).
            declared_reference_time: None,
            group_id: None,
            content_type: None,
            batch_id: Some(batch_id),
        })
    }

    /// Low-level enqueue accepting a fully-constructed [`IngestRequest`].
    ///
    /// Use when you need `reference_time`, `group_id`, or `content_type` on a
    /// batched send.  The race-safety invariant (arch spec §3.3) is enforced
    /// here: both `BatchProgress::total` and `queued` are incremented BEFORE
    /// `try_send` so the worker thread can never decrement below zero.
    pub(crate) fn enqueue_req(&self, req: IngestRequest) -> Result<(), IngestSendError> {
        // Race-safety invariant: increment BEFORE enqueue (arch spec §3.3).
        // Both counters must be incremented before try_send so the worker
        // cannot fetch_sub(1) on a zero counter and wrap to usize::MAX.
        if let Some(ref bid) = req.batch_id {
            let mut tracker = self.batch_tracker.lock().unwrap_or_else(|p| p.into_inner());
            tracker
                .entry(bid.clone())
                .and_modify(|p| p.total += 1)
                .or_default();
        }
        self.inner.queued.fetch_add(1, Ordering::Relaxed);

        match self.inner.work_tx.try_send(req) {
            Ok(()) => {
                let depth = self.inner.queued.load(Ordering::Relaxed);
                metrics::gauge!("rql.background.queue_depth").set(depth as f64);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.inner.queued.fetch_sub(1, Ordering::Relaxed);
                Err(IngestSendError::Full(self.inner.channel_capacity))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.inner.queued.fetch_sub(1, Ordering::Relaxed);
                Err(IngestSendError::Disconnected)
            }
        }
    }

    /// Poll for ingestion errors since the last call.  Non-blocking.
    pub fn drain_errors(&self) -> Vec<IngestError> {
        let rx =
            self.inner.error_rx.lock().unwrap_or_else(|poisoned| {
                panic!("invariant: error_rx mutex poisoned: {poisoned}")
            });
        let mut errors = Vec::new();
        while let Ok(e) = rx.try_recv() {
            errors.push(e);
        }
        errors
    }

    /// Number of items currently queued in the work channel.
    ///
    /// This is an approximate count maintained by `AtomicUsize`; it is
    /// decremented by the worker just before each `ingest()` call.
    pub fn queue_depth(&self) -> usize {
        self.inner.queued.load(Ordering::Relaxed)
    }

    /// Returns a reference to the event sink, if one was configured via
    /// [`IngestorConfig::with_sink`].
    ///
    /// Provided so Phase 6 integration tests (and future Phase 4 batch-tracker
    /// code) can verify that a sink was wired without poking the private field.
    /// Returns `None` when no sink was configured.
    ///
    /// Per ADR-052 Gap 1; impl spec §3 Phase 2.
    pub fn sink(&self) -> Option<&Arc<dyn EnrichmentEventSink>> {
        self.sink.as_ref()
    }
}

// ---------------------------------------------------------------------------
// IngestGuard
// ---------------------------------------------------------------------------

/// RAII handle for graceful shutdown.
///
/// When all [`BackgroundIngestor`] clones are dropped the work channel closes
/// and the worker exits naturally.  Dropping [`IngestGuard`] joins the thread
/// so the caller can be sure all outstanding work has been processed.
pub struct IngestGuard {
    pub(super) stop: Arc<AtomicBool>,
    pub(super) handle: Option<JoinHandle<()>>,
}

impl IngestGuard {
    /// Explicit shutdown; equivalent to dropping the guard.
    pub fn shutdown(self) {
        drop(self);
    }
}

impl Drop for IngestGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
