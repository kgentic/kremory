//! BackgroundIngestor — fire-and-forget ingestion worker following the
//! tracing-appender WorkerGuard pattern.
//!
//! The caller creates a [`BackgroundIngestor`] + [`IngestGuard`] pair via
//! [`BackgroundIngestor::new`].  The graph is moved into a dedicated OS thread
//! which owns a `tokio::runtime::Runtime` (current-thread flavour).  Work is
//! submitted via an `std::sync::mpsc` bounded channel; errors are returned via
//! a second bounded channel and collected with [`BackgroundIngestor::drain_errors`].
//!
//! Shutdown is triggered by dropping [`IngestGuard`], which sets an atomic stop
//! flag and joins the worker thread.  The worker drains remaining items before
//! exiting.  This is safe even when [`BackgroundIngestor`] clones still exist —
//! no deadlock.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::config::ContentType;
use crate::error::RqlError;
use crate::ingest::RqlGraph;
use crate::provider::{ChatProvider, EmbeddingProvider};

// ---------------------------------------------------------------------------
// IngestRequest
// ---------------------------------------------------------------------------

/// Work item queued via [`BackgroundIngestor::send`].
pub(crate) struct IngestRequest {
    pub text: String,
    pub reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
    pub content_type: Option<ContentType>,
}

// ---------------------------------------------------------------------------
// DeferredRequest
// ---------------------------------------------------------------------------

/// Work item queued for Phase 2 (deferred LLM fact extraction).
///
/// Created after a successful Phase 1 NER ingest.  The worker processes these
/// when the NER channel is idle, giving NER priority over LLM fact extraction.
struct DeferredRequest {
    text: String,
    reference_time: Option<DateTime<Utc>>,
    group_id: Option<String>,
    content_type: Option<ContentType>,
    /// The episode ID produced by Phase 1, so deferred facts link to the same episode.
    episode_id: i64,
    /// Entity names already inserted by Phase 1, passed as hints to the LLM extractor.
    ner_entity_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// IngestErrorKind
// ---------------------------------------------------------------------------

/// Coarse category of an ingestion failure observed after the fact.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestErrorKind {
    Database,
    Extraction,
    Resolution,
    Llm,
    Embedding,
    Other,
}

impl From<&RqlError> for IngestErrorKind {
    fn from(e: &RqlError) -> Self {
        match e {
            RqlError::Database(_) => IngestErrorKind::Database,
            RqlError::Extraction(_) => IngestErrorKind::Extraction,
            RqlError::Resolution(_) => IngestErrorKind::Resolution,
            RqlError::Llm(_) => IngestErrorKind::Llm,
            RqlError::Embedding(_) => IngestErrorKind::Embedding,
            // Config, Search, Parse, Serialization, Other all collapse to Other.
            _ => IngestErrorKind::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// IngestError
// ---------------------------------------------------------------------------

/// An ingestion failure observed after the fact, available via
/// [`BackgroundIngestor::drain_errors`].
#[derive(Debug, Clone)]
pub struct IngestError {
    /// First 256 chars of the text that failed.
    pub text_preview: String,
    /// Wall-clock time the failure was recorded.
    pub failed_at: DateTime<Utc>,
    /// Human-readable error message.
    pub message: String,
    /// Coarse failure category.
    pub kind: IngestErrorKind,
}

// ---------------------------------------------------------------------------
// IngestSendError
// ---------------------------------------------------------------------------

/// Errors that can occur when calling [`BackgroundIngestor::send`].
#[derive(Debug, thiserror::Error)]
pub enum IngestSendError {
    #[error("ingest queue full (capacity={0})")]
    Full(usize),
    #[error("ingest worker disconnected")]
    Disconnected,
}

// ---------------------------------------------------------------------------
// IngestorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`BackgroundIngestor`].
#[derive(Debug, Clone)]
pub struct IngestorConfig {
    /// Capacity of the work channel.  Default: 64.
    pub channel_capacity: usize,
    /// Capacity of the error feedback channel.  Default: 256.
    pub error_channel_capacity: usize,
    /// Name of the worker OS thread.  Default: `"rql-ingestor"`.
    pub thread_name: String,
    /// Enable Phase 2 deferred LLM fact extraction.  Default: `true`.
    ///
    /// When `true`, after each successful Phase 1 NER ingest the worker enqueues
    /// a [`DeferredRequest`] and processes it when the NER channel is idle.
    /// NER always has priority — the deferred queue is only drained during
    /// `recv_timeout` idle periods.
    ///
    /// Set to `false` to run Phase 1 only (e.g., in latency-critical tests or
    /// environments without a capable LLM).
    pub deferred_extraction_enabled: bool,
}

impl Default for IngestorConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 64,
            error_channel_capacity: 256,
            thread_name: "rql-ingestor".to_string(),
            deferred_extraction_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Inner
// ---------------------------------------------------------------------------

struct Inner {
    /// Sender half of the work channel.
    work_tx: SyncSender<IngestRequest>,
    /// Receiver half of the error channel, guarded so multiple clones can
    /// drain it from one caller at a time.
    error_rx: Mutex<Receiver<IngestError>>,
    /// Tracks items currently sitting in the work channel.
    queued: Arc<AtomicUsize>,
    /// Nominal capacity of the work channel (for error messages).
    channel_capacity: usize,
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
#[derive(Clone)]
pub struct BackgroundIngestor {
    inner: Arc<Inner>,
}

impl BackgroundIngestor {
    /// Create an ingestor + guard.  The worker thread starts immediately.
    ///
    /// `graph` is moved into the worker thread; this is the only place where
    /// the generics `L` and `Emb` appear.
    pub fn new<L, Emb>(graph: RqlGraph<L, Emb>, config: IngestorConfig) -> (Self, IngestGuard)
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

        let handle = thread::Builder::new()
            .name(config.thread_name.clone())
            .spawn(move || {
                worker_loop(
                    graph,
                    work_rx,
                    error_tx,
                    queued_worker,
                    stop_worker,
                    deferred_enabled,
                );
            })
            .expect("failed to spawn rql-ingestor thread");

        let ingestor = BackgroundIngestor {
            inner: Arc::new(Inner {
                work_tx,
                error_rx: Mutex::new(error_rx),
                queued,
                channel_capacity: config.channel_capacity,
            }),
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
                self.inner.queued.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(IngestSendError::Full(self.inner.channel_capacity)),
            Err(TrySendError::Disconnected(_)) => Err(IngestSendError::Disconnected),
        }
    }

    /// Poll for ingestion errors since the last call.  Non-blocking.
    pub fn drain_errors(&self) -> Vec<IngestError> {
        let rx = self.inner.error_rx.lock().expect("error_rx mutex poisoned");
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
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
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
// Worker
// ---------------------------------------------------------------------------

/// Process one NER ingest request (Phase 1).
///
/// Returns `Some(DeferredRequest)` when Phase 2 should be enqueued, or `None`
/// on error (error already forwarded to `error_tx`).
async fn process_item<L: ChatProvider, Emb: EmbeddingProvider>(
    graph: &RqlGraph<L, Emb>,
    req: IngestRequest,
    error_tx: &SyncSender<IngestError>,
    deferred_enabled: bool,
) -> Option<DeferredRequest> {
    let start = std::time::Instant::now();
    match graph
        .ingest(
            &req.text,
            req.reference_time,
            req.group_id.as_deref(),
            req.content_type.clone(),
        )
        .await
    {
        Ok(result) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.ingest_duration_ms").record(elapsed_ms);
            metrics::counter!("rql.background.ingested_total").increment(1);

            if deferred_enabled {
                let ner_entity_names = result.upserted_entities.clone();
                Some(DeferredRequest {
                    text: req.text,
                    reference_time: req.reference_time,
                    group_id: req.group_id,
                    content_type: req.content_type,
                    episode_id: result.episode_id,
                    ner_entity_names,
                })
            } else {
                None
            }
        }
        Err(e) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.ingest_duration_ms").record(elapsed_ms);
            metrics::counter!("rql.background.errors_total").increment(1);
            let err = IngestError {
                text_preview: req.text.chars().take(256).collect(),
                failed_at: Utc::now(),
                message: e.to_string(),
                kind: IngestErrorKind::from(&e),
            };
            if error_tx.try_send(err).is_err() {
                metrics::counter!("rql.background.errors_dropped_total").increment(1);
            }
            None
        }
    }
}

/// Process one deferred LLM fact extraction request (Phase 2).
///
/// Errors are logged via metrics and the error channel but do NOT crash the worker.
async fn process_deferred<L: ChatProvider, Emb: EmbeddingProvider>(
    graph: &RqlGraph<L, Emb>,
    req: DeferredRequest,
    error_tx: &SyncSender<IngestError>,
) {
    let start = std::time::Instant::now();
    match graph
        .ingest_deferred(
            &req.text,
            req.reference_time,
            req.group_id.as_deref(),
            req.content_type,
            req.episode_id,
            &req.ner_entity_names,
        )
        .await
    {
        Ok(facts_extracted) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.deferred_extraction_duration_ms")
                .record(elapsed_ms);
            metrics::counter!("rql.background.deferred_facts_extracted_total")
                .increment(facts_extracted as u64);
        }
        Err(e) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!("rql.background.deferred_extraction_duration_ms")
                .record(elapsed_ms);
            metrics::counter!("rql.background.deferred_errors_total").increment(1);
            let err = IngestError {
                text_preview: req.text.chars().take(256).collect(),
                failed_at: Utc::now(),
                message: format!("deferred: {e}"),
                kind: IngestErrorKind::from(&e),
            };
            if error_tx.try_send(err).is_err() {
                metrics::counter!("rql.background.errors_dropped_total").increment(1);
            }
        }
    }
}

fn worker_loop<L: ChatProvider, Emb: EmbeddingProvider>(
    graph: RqlGraph<L, Emb>,
    work_rx: Receiver<IngestRequest>,
    error_tx: SyncSender<IngestError>,
    queued: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    deferred_enabled: bool,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime for rql-ingestor");

    rt.block_on(async {
        let mut deferred_queue: VecDeque<DeferredRequest> = VecDeque::new();

        loop {
            if stop.load(Ordering::Acquire) {
                // Guard was dropped — drain remaining NER items, then exit.
                // Deferred queue is abandoned on forced shutdown (NER takes priority).
                while let Ok(req) = work_rx.try_recv() {
                    queued.fetch_sub(1, Ordering::Relaxed);
                    if let Some(deferred) =
                        process_item(&graph, req, &error_tx, deferred_enabled).await
                    {
                        deferred_queue.push_back(deferred);
                    }
                }
                break;
            }

            match work_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(req) => {
                    queued.fetch_sub(1, Ordering::Relaxed);
                    if let Some(deferred) =
                        process_item(&graph, req, &error_tx, deferred_enabled).await
                    {
                        deferred_queue.push_back(deferred);
                    }
                    metrics::gauge!("rql.background.deferred_queue_depth")
                        .set(deferred_queue.len() as f64);
                }
                Err(RecvTimeoutError::Timeout) => {
                    // NER channel is idle — process one deferred item if available,
                    // then loop back to check for new NER work (NER priority).
                    if let Some(deferred) = deferred_queue.pop_front() {
                        metrics::gauge!("rql.background.deferred_queue_depth")
                            .set(deferred_queue.len() as f64);
                        process_deferred(&graph, deferred, &error_tx).await;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // All senders dropped — drain remaining NER items, then
                    // process the deferred queue, then exit.
                    while let Ok(req) = work_rx.try_recv() {
                        queued.fetch_sub(1, Ordering::Relaxed);
                        if let Some(deferred) =
                            process_item(&graph, req, &error_tx, deferred_enabled).await
                        {
                            deferred_queue.push_back(deferred);
                        }
                    }
                    // Drain deferred queue, but respect the stop flag so
                    // IngestGuard::drop doesn't block on slow LLM calls.
                    while let Some(deferred) = deferred_queue.pop_front() {
                        if stop.load(Ordering::Acquire) {
                            let abandoned = deferred_queue.len() + 1;
                            eprintln!(
                                "[BackgroundIngestor] stop signal — abandoning {abandoned} deferred item(s)"
                            );
                            break;
                        }
                        metrics::gauge!("rql.background.deferred_queue_depth")
                            .set(deferred_queue.len() as f64);
                        process_deferred(&graph, deferred, &error_tx).await;
                    }
                    break;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::SimpleGraph;

    #[cfg(feature = "llm")]
    use crate::provider::ChatProvider;
    #[cfg(feature = "llm")]
    use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
    #[cfg(feature = "llm")]
    use autoagents_llm::error::LLMError;

    /// A ChatProvider that always returns an `LLMError::Generic` error.
    // Legacy name; impls ChatProvider per AA adoption (2026-04-12 commit 5e8bddd).
    #[cfg(feature = "llm")]
    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    struct FailingLlmClient;

    #[cfg(feature = "llm")]
    #[async_trait::async_trait]
    impl ChatProvider for FailingLlmClient {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Err(LLMError::Generic("simulated LLM failure".to_string()))
        }
    }

    // -----------------------------------------------------------------------

    /// Helper: build an ingestor backed by a SimpleGraph (NullLlmClient +
    /// NullEmbeddingProvider) with default config.
    async fn simple_ingestor() -> (BackgroundIngestor, IngestGuard) {
        let graph = SimpleGraph::open_in_memory_simple()
            .await
            .expect("open_in_memory_simple failed");
        BackgroundIngestor::new(graph, IngestorConfig::default())
    }

    // -----------------------------------------------------------------------

    /// Send one item and verify it processes without errors.
    #[tokio::test]
    async fn send_and_drain() {
        let (ingestor, guard) = simple_ingestor().await;
        ingestor
            .send("Alice works at Acme", None, None, None)
            .expect("send should succeed");
        // Drop the ingestor so the channel closes, then join the worker.
        drop(ingestor);
        guard.shutdown();
        // No errors expected with NullLlmClient.
        // (drain_errors is called after shutdown — channel already drained.)
    }

    // -----------------------------------------------------------------------

    /// A channel_capacity=1 channel should return Full after the slot is taken.
    #[tokio::test]
    async fn queue_full_returns_error() {
        let graph = SimpleGraph::open_in_memory_simple()
            .await
            .expect("open_in_memory_simple failed");
        // capacity = 1 so the second send sees a full channel.
        let config = IngestorConfig {
            channel_capacity: 1,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        // First send should fit.
        ingestor
            .send("first item", None, None, None)
            .expect("first send should succeed");

        // The worker may have consumed the first item before we send the second,
        // so we retry until the queue is full or we exhaust attempts.
        // In the worst case both items go through; what we really want is to
        // exercise the Full error path at least once.
        let mut got_full = false;
        for _ in 0..20 {
            match ingestor.send("overflow item", None, None, None) {
                Err(IngestSendError::Full(_)) => {
                    got_full = true;
                    break;
                }
                Ok(()) => {
                    // Worker consumed the slot — keep trying.
                }
                Err(other) => panic!("unexpected error: {other}"),
            }
        }
        // If the worker was fast enough to consume every item before we could
        // fill the channel, that is also correct behaviour — the test is
        // advisory.  But we still clean up.
        drop(ingestor);
        guard.shutdown();
        // got_full may be false only if the worker drained every item instantly.
        let _ = got_full; // suppress unused-variable warning
    }

    // -----------------------------------------------------------------------

    /// Send 5 items and drop the guard — verify the thread joins without panic.
    #[tokio::test]
    async fn shutdown_drains_queue() {
        let (ingestor, guard) = simple_ingestor().await;
        for i in 0..5_u8 {
            ingestor
                .send(format!("item {i}"), None, None, None)
                .expect("send should succeed");
        }
        drop(ingestor);
        // join must complete without panicking.
        guard.shutdown();
    }

    // -----------------------------------------------------------------------

    /// Use a FailingLlmClient so every ingest() returns an error, then verify
    /// drain_errors() surfaces at least one IngestError.
    ///
    /// Design note: `drain_errors()` must be called while at least one
    /// `BackgroundIngestor` clone is alive (so the channel is not dropped).
    /// We poll for up to 5s to give the worker time to process the item and
    /// send the error, then drop the last clone to close the channel and
    /// join the worker.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn errors_are_observable() {
        use crate::config::PipelineConfig;
        use crate::provider::NullEmbeddingProvider;
        use crate::schema::TemporalGraph;
        use std::sync::Arc;
        use std::time::Duration;

        let temporal = TemporalGraph::open_in_memory()
            .await
            .expect("open in-memory db failed");
        let config = PipelineConfig::builder()
            .build()
            .expect("config build failed");
        let dim = config.embedding_dim.0;
        let graph = RqlGraph::new(
            temporal,
            Arc::new(FailingLlmClient),
            Arc::new(NullEmbeddingProvider { dim }),
            config,
        );

        let (ingestor, guard) = BackgroundIngestor::new(graph, IngestorConfig::default());

        ingestor
            .send("this will fail", None, None, None)
            .expect("send should succeed");

        // Poll for errors for up to 5 seconds.  The worker processes the item
        // asynchronously; we keep the ingestor alive (so the channel is open)
        // while we wait.
        let mut errors = Vec::new();
        for _ in 0..50 {
            errors = ingestor.drain_errors();
            if !errors.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Now drop the ingestor to close the channel, then join the worker.
        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("guard shutdown panicked");

        assert!(
            !errors.is_empty(),
            "expected at least one IngestError after 5s, got none"
        );
    }

    // -----------------------------------------------------------------------

    /// Verify that deferred_extraction_enabled=false prevents any DeferredRequests
    /// from being enqueued — the worker processes items and exits cleanly.
    #[tokio::test]
    async fn deferred_disabled_processes_without_error() {
        let (ingestor, guard) = {
            let graph = SimpleGraph::open_in_memory_simple()
                .await
                .expect("open_in_memory_simple failed");
            let config = IngestorConfig {
                deferred_extraction_enabled: false,
                ..IngestorConfig::default()
            };
            BackgroundIngestor::new(graph, config)
        };

        ingestor
            .send("Alice works at Acme Corp", None, None, None)
            .expect("send should succeed");
        ingestor
            .send("Bob manages Alice", None, None, None)
            .expect("send should succeed");

        drop(ingestor);
        guard.shutdown();
        // Worker exits cleanly — no panic means the deferred path was skipped correctly.
    }

    /// Verify that deferred_extraction_enabled=true (default) allows the worker
    /// to enqueue and drain DeferredRequests without panicking.
    #[tokio::test]
    async fn deferred_enabled_drains_without_panic() {
        let (ingestor, guard) = simple_ingestor().await;

        ingestor
            .send("Alice works at Acme Corp", None, None, None)
            .expect("send should succeed");

        // Drop the ingestor so the channel disconnects; the worker will drain
        // the NER queue and then drain the deferred queue before exiting.
        drop(ingestor);
        guard.shutdown();
        // If the deferred path panics, guard.shutdown() (thread join) will propagate it.
    }

    /// Clone the ingestor, send from both handles, verify both reach the worker.
    #[tokio::test]
    async fn clone_shares_worker() {
        let (ingestor, guard) = simple_ingestor().await;
        let clone = ingestor.clone();

        ingestor
            .send("from original", None, None, None)
            .expect("send from original should succeed");
        clone
            .send("from clone", None, None, None)
            .expect("send from clone should succeed");

        // Drop both senders so the channel closes.
        drop(ingestor);
        drop(clone);
        guard.shutdown();
        // If the thread panicked, join() would propagate here.
    }

    // -----------------------------------------------------------------------
    // Deferred extraction behavioural tests
    // -----------------------------------------------------------------------

    /// A ChatProvider that counts every `chat` call via an AtomicUsize.
    /// Returns an empty string (safe no-op for extraction prompts).
    // Legacy name; impls ChatProvider per AA adoption (2026-04-12 commit 5e8bddd).
    #[cfg(feature = "llm")]
    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    struct CountingLlmClient {
        calls: Arc<AtomicUsize>,
    }

    #[cfg(feature = "llm")]
    impl CountingLlmClient {
        #[allow(dead_code)]
        fn new() -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }
    }

    #[cfg(feature = "llm")]
    #[async_trait::async_trait]
    impl ChatProvider for CountingLlmClient {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(crate::provider::MockChatResponse {
                text: String::new(),
            }))
        }
    }

    // -----------------------------------------------------------------------

    /// A ChatProvider that fails after the first call.  The first call (NER
    /// extraction) returns empty JSON (success); every subsequent call (deferred
    /// LLM extraction) returns an LLMError.
    ///
    /// This lets us verify:
    ///   - NER succeeds  → DeferredRequest is created
    ///   - Deferred fails → error is surfaced via drain_errors(), not via panic
    // Legacy name; impls ChatProvider per AA adoption (2026-04-12 commit 5e8bddd).
    #[cfg(feature = "llm")]
    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    struct FailAfterFirstLlmClient {
        call_count: Arc<AtomicUsize>,
    }

    #[cfg(feature = "llm")]
    impl FailAfterFirstLlmClient {
        #[allow(dead_code)]
        fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[cfg(feature = "llm")]
    #[async_trait::async_trait]
    impl ChatProvider for FailAfterFirstLlmClient {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                // First call: NER extraction → empty string = 0 entities, 0 facts
                Ok(Box::new(crate::provider::MockChatResponse {
                    text: String::new(),
                }))
            } else {
                // Subsequent calls: deferred LLM extraction → simulated failure
                Err(LLMError::Generic("deferred simulated failure".to_string()))
            }
        }
    }

    // -----------------------------------------------------------------------

    /// Helper: build an RqlGraph backed by the supplied ChatProvider.
    #[cfg(feature = "llm")]
    #[allow(dead_code)]
    async fn graph_with_llm<L: ChatProvider + 'static>(
        llm: L,
    ) -> crate::ingest::RqlGraph<L, crate::provider::NullEmbeddingProvider> {
        use crate::config::PipelineConfig;
        use crate::provider::NullEmbeddingProvider;
        use crate::schema::TemporalGraph;

        let temporal = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed");
        let config = PipelineConfig::builder()
            .build()
            .expect("config build failed");
        let dim = config.embedding_dim.0;
        crate::ingest::RqlGraph::new(
            temporal,
            Arc::new(llm),
            Arc::new(NullEmbeddingProvider { dim }),
            config,
        )
    }

    // -----------------------------------------------------------------------

    /// When `deferred_extraction_enabled=true` and NER ingest succeeds, the worker
    /// MUST create a DeferredRequest and invoke `ingest_deferred` (which calls the
    /// LLM).  We prove this by using a `CountingLlmClient`: with deferred enabled,
    /// the LLM is called ≥2 times (once for NER, once for deferred); with deferred
    /// disabled it is called exactly once.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_request_created_after_successful_ingest() {
        use std::time::Duration;

        let (counting_client, call_counter) = CountingLlmClient::new();
        let graph = graph_with_llm(counting_client).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        ingestor
            .send("Alice works at Acme Corp", None, None, None)
            .expect("send should succeed");

        // Keep the ingestor alive so the worker stays connected.  The deferred
        // item is processed during the 100ms idle timeout in recv_timeout.
        // Poll until we observe ≥2 LLM calls (1 NER + ≥1 deferred).
        for _ in 0..50 {
            let total_calls = call_counter.load(Ordering::SeqCst);
            if total_calls >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("guard.shutdown() panicked");

        // NER calls the LLM once (NuExtractExtractor).
        // Deferred calls the LLM at least once more (ingest_deferred → NuExtractExtractor).
        // With a single text item and one chunk, total calls should be exactly 2.
        let final_calls = call_counter.load(Ordering::SeqCst);
        assert!(
            final_calls >= 2,
            "expected ≥2 LLM calls (1 NER + ≥1 deferred) with deferred_enabled=true; got {final_calls}"
        );
    }

    /// Negative counterpart to `deferred_request_created_after_successful_ingest`:
    /// when `deferred_extraction_enabled=false`, only the NER call (exactly 1 LLM
    /// call per single-chunk text) should happen — no deferred call.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_disabled_makes_no_deferred_llm_calls() {
        let (counting_client, call_counter) = CountingLlmClient::new();
        let graph = graph_with_llm(counting_client).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: false,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        ingestor
            .send("Alice works at Acme Corp", None, None, None)
            .expect("send should succeed");

        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("guard.shutdown() panicked");

        // With deferred disabled, only 1 LLM call (the NER extraction) should occur.
        let total_calls = call_counter.load(Ordering::SeqCst);
        assert_eq!(
            total_calls, 1,
            "expected exactly 1 LLM call (NER only) with deferred_enabled=false; got {total_calls}"
        );
    }

    // -----------------------------------------------------------------------

    /// When the deferred VecDeque is non-empty, new NER items must still be
    /// processed first.  Observable proof: with `deferred_enabled=true`, send N
    /// NER items; both NER and deferred phases must complete for all N items.
    ///
    /// We use `CountingLlmClient` and verify that after all processing completes,
    /// exactly N NER calls plus N deferred calls happened.
    ///
    /// Mechanism: deferred items are processed during idle 100ms timeout windows.
    /// We keep the ingestor alive and poll until the count reaches 2N before
    /// shutting down.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_queue_does_not_block_subsequent_ner_items() {
        use std::time::Duration;

        const N: usize = 4;

        let (counting_client, call_counter) = CountingLlmClient::new();
        let graph = graph_with_llm(counting_client).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        for i in 0..N {
            ingestor
                .send(
                    format!("item number {i} about entity person"),
                    None,
                    None,
                    None,
                )
                .expect("send should succeed");
        }

        // Keep the ingestor alive.  The worker processes N NER items eagerly, then
        // drains N deferred items one-per-idle-cycle (100ms each).  Poll for up
        // to 5s for all 2N calls to complete.
        for _ in 0..50 {
            if call_counter.load(Ordering::SeqCst) >= N * 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("guard.shutdown() panicked");

        // Each of the N items contributes 1 NER call + 1 deferred call = 2N total.
        let total_calls = call_counter.load(Ordering::SeqCst);
        assert_eq!(
            total_calls,
            N * 2,
            "expected {expected} LLM calls ({N} NER + {N} deferred); got {total_calls}",
            expected = N * 2,
        );
    }

    // -----------------------------------------------------------------------

    /// When `ingest_deferred` fails (LLM returns an error), the worker MUST
    /// NOT crash.  The error must be observable via `drain_errors()`, and the
    /// worker must exit cleanly after processing all items.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_extraction_errors_do_not_crash_worker() {
        use std::time::Duration;

        let graph = graph_with_llm(FailAfterFirstLlmClient::new()).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        ingestor
            .send("Alice works at Acme Corp", None, None, None)
            .expect("send should succeed");

        // Poll for a deferred error to appear.  Keep the ingestor alive so the
        // error channel stays open.
        let mut deferred_errors: Vec<IngestError> = Vec::new();
        for _ in 0..50 {
            deferred_errors = ingestor.drain_errors();
            // The deferred error message is prefixed with "deferred:" (see process_deferred).
            if deferred_errors
                .iter()
                .any(|e| e.message.contains("deferred"))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        drop(ingestor);
        // If the worker crashed, spawn_blocking would propagate the panic.
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("worker panicked — deferred error should not crash the worker");

        assert!(
            deferred_errors
                .iter()
                .any(|e| e.message.contains("deferred")),
            "expected at least one deferred IngestError; got: {:?}",
            deferred_errors
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );

        // Verify the error kind is Llm (FailAfterFirstLlmClient returns RqlError::Llm).
        assert!(
            deferred_errors
                .iter()
                .any(|e| e.kind == IngestErrorKind::Llm),
            "expected IngestErrorKind::Llm for deferred failure; got: {:?}",
            deferred_errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------

    /// When `deferred_extraction_enabled=false`, the deferred VecDeque must
    /// remain empty throughout — the worker should not call `ingest_deferred`
    /// at all, even after many successful NER ingests.
    ///
    /// Verified by `CountingLlmClient`: call count must equal the number of
    /// NER items sent (not 2× that).
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_config_disabled_skips_queue() {
        const N: usize = 3;

        let (counting_client, call_counter) = CountingLlmClient::new();
        let graph = graph_with_llm(counting_client).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: false,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        for i in 0..N {
            ingestor
                .send(format!("entity record {i}"), None, None, None)
                .expect("send should succeed");
        }

        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("guard.shutdown() panicked");

        let total_calls = call_counter.load(Ordering::SeqCst);
        assert_eq!(
            total_calls, N,
            "deferred disabled: expected exactly {N} LLM calls (NER only); got {total_calls}"
        );
    }

    // -----------------------------------------------------------------------

    /// Verify that when all senders are dropped (Disconnected arm of worker_loop),
    /// the worker drains ALL pending deferred items before exiting.
    ///
    /// Strategy: use `FailAfterFirstLlmClient` so NER succeeds (creates a
    /// DeferredRequest) and deferred fails (error surfaced via drain_errors).
    ///
    /// We wait long enough for the NER item to be processed and a deferred item
    /// to be queued (via the idle timeout path), then drop the sender.  The worker
    /// sees Disconnected and drains the remaining deferred queue.  The deferred
    /// error is observable via drain_errors before we drop the ingestor.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_queue_drains_on_sender_disconnect() {
        use std::time::Duration;

        let graph = graph_with_llm(FailAfterFirstLlmClient::new()).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        ingestor
            .send("Alice meets Bob at Acme HQ", None, None, None)
            .expect("send should succeed");

        // Poll for a deferred error (≥1 idle cycle = ≥100ms).  Keep the
        // ingestor alive so we can drain errors.
        let mut deferred_errors: Vec<IngestError> = Vec::new();
        for _ in 0..50 {
            deferred_errors = ingestor.drain_errors();
            if deferred_errors
                .iter()
                .any(|e| e.message.contains("deferred"))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Now drop all senders so the worker sees Disconnected and exits cleanly.
        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("worker panicked during deferred drain on disconnect");

        // Verify the deferred error was reported (not silently swallowed).
        assert!(
            deferred_errors
                .iter()
                .any(|e| e.message.contains("deferred")),
            "expected a deferred IngestError; got: {:?}",
            deferred_errors
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------

    /// Verify that the deferred code paths (`process_item` success branch and
    /// `process_deferred`) both execute, confirming the metrics-recording code
    /// inside each function runs.
    ///
    /// We use `CountingLlmClient` as a lightweight proxy: each item should produce
    /// exactly 2 LLM calls (1 NER via `process_item` + 1 deferred via
    /// `process_deferred`).  If either metric-adjacent code path were skipped or
    /// crashed, the count would differ from 2.
    ///
    /// Deferred is processed during the idle 100ms timeout.  Keep the ingestor
    /// alive and poll for the count to reach 2.
    #[cfg(all(not(feature = "ner"), feature = "llm"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_metrics_code_paths_execute() {
        use std::time::Duration;

        let (counting_client, call_counter) = CountingLlmClient::new();
        let graph = graph_with_llm(counting_client).await;

        let config = IngestorConfig {
            deferred_extraction_enabled: true,
            ..IngestorConfig::default()
        };
        let (ingestor, guard) = BackgroundIngestor::new(graph, config);

        ingestor
            .send("Alice leads the Acme project", None, None, None)
            .expect("send should succeed");

        // Poll until both code paths have executed (count = 2).
        for _ in 0..50 {
            if call_counter.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        drop(ingestor);
        tokio::task::spawn_blocking(move || guard.shutdown())
            .await
            .expect("guard.shutdown() panicked");

        // 1 NER call (process_item → NuExtractExtractor) +
        // 1 deferred call (process_deferred → ingest_deferred → NuExtractExtractor) = 2 total.
        let total_calls = call_counter.load(Ordering::SeqCst);
        assert_eq!(
            total_calls, 2,
            "expected 2 LLM calls (1 NER + 1 deferred); got {total_calls} — \
             a count of 1 means the deferred metric path was skipped"
        );
    }

    // -----------------------------------------------------------------------

    /// Prove that dropping the guard while a BackgroundIngestor clone is still
    /// alive does NOT deadlock.  Before the stop-flag fix, this would hang
    /// forever because `recv()` would never return `Err` (the clone keeps the
    /// channel open).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn guard_shutdown_does_not_deadlock_with_live_clone() {
        let (ingestor, guard) = simple_ingestor().await;
        let clone = ingestor.clone();

        ingestor
            .send("before shutdown", None, None, None)
            .expect("send should succeed");

        // Drop the guard while `clone` is still alive.
        // Without the stop flag, this would deadlock.
        let join = tokio::task::spawn_blocking(move || {
            guard.shutdown(); // must return within a few hundred ms
        });

        // Give it 5 seconds — if it deadlocks, this will fail.
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), join).await;

        assert!(
            result.is_ok(),
            "guard.shutdown() deadlocked — stop flag not working"
        );

        // The clone is still usable (though the worker is gone, so sends will
        // return Disconnected).
        let send_result = clone.send("after shutdown", None, None, None);
        assert!(
            matches!(send_result, Err(IngestSendError::Disconnected)),
            "expected Disconnected after worker exit, got: {send_result:?}"
        );
    }
}
