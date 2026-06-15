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
    deferred_pipeline::worker_loop, IngestError, IngestRequest, IngestSendError, IngestorConfig,
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
/// sequentially on a current-thread tokio runtime (`worker_threads(1)`).
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

        let parent_span = tracing::Span::current();
        let handle = thread::Builder::new()
            .name(config.thread_name.clone())
            .spawn(move || {
                let _enter = parent_span.enter();
                worker_loop(
                    graph,
                    work_rx,
                    error_tx,
                    queued_worker,
                    stop_worker,
                    deferred_enabled,
                    llm_rate_limit,
                    sink_for_worker,
                );
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
    pub fn send(
        &self,
        text: impl Into<String>,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<String>,
        content_type: Option<ContentType>,
    ) -> Result<(), IngestSendError> {
        let req = IngestRequest {
            text: text.into(),
            reference_time,
            group_id,
            content_type,
        };
        match self.inner.work_tx.try_send(req) {
            Ok(()) => {
                let depth = self.inner.queued.fetch_add(1, Ordering::Relaxed) + 1;
                metrics::gauge!("rql.background.queue_depth").set(depth as f64);
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(IngestSendError::Full(self.inner.channel_capacity)),
            Err(TrySendError::Disconnected(_)) => Err(IngestSendError::Disconnected),
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
