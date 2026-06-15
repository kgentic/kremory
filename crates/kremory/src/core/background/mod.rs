//! `kremory::core::background` — fire-and-forget background ingestion pipeline.
//!
//! Split from `background.rs` per sprint plan T2.1 + ADR-049 §Decision 6.
//! Module layout:
//!
//! - [`ingestor`]          — [`BackgroundIngestor`] + [`IngestGuard`] + send/queue ops
//! - [`deferred_pipeline`] — worker loop, spawn_worker, drain logic, error reporting
//! - [`verify_stage`]      — Stage 2 hook (no-op stub until ADR-049 Stage 2 wiring)
//!
//! Shared types live here so both submodules can import without circular deps.
//!
//! # ADR reference
//!
//! ADR-049 §Decision 6 mandates this split as a prerequisite for Phase B Stage 2
//! wiring. See `.ai-docs/adrs/adr-049-c6-async-gate-verify-pre-write-2026-06-10.md`
//! and `.ai-docs/plans/v0-2-0-phase-b-prep-sprint-plan-2026-06-10.md` §T2.1.

use std::sync::mpsc::SyncSender;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::core::config::ContentType;
use crate::core::error::Error;
use crate::memory::events::EnrichmentEventSink;

pub mod deferred_pipeline;
pub mod ingestor;
pub mod verify_stage;

pub use ingestor::{BackgroundIngestor, IngestGuard};
// Quinn MED-02 fix: no re-export of run_verify_stage. The stub is Phase B
// internal scaffolding (ADR-049 §Decision 6 mandate); Phase B will call it via
// `super::verify_stage::run_verify_stage` from deferred_pipeline. Re-export here
// would leak misleadingly-named no-op into the v0.2.0 public surface.

// ---------------------------------------------------------------------------
// IngestRequest  (shared: ingestor enqueues, deferred_pipeline consumes)
// ---------------------------------------------------------------------------

/// Work item queued via [`BackgroundIngestor::send`].
pub(crate) struct IngestRequest {
    pub text: String,
    pub reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
    pub content_type: Option<ContentType>,
}

// ---------------------------------------------------------------------------
// DeferredRequest  (produced by deferred_pipeline after Phase 1 success)
// ---------------------------------------------------------------------------

/// Work item queued for Phase 2 (deferred LLM fact extraction).
///
/// Created after a successful Phase 1 NER ingest.  The worker processes these
/// when the NER channel is idle, giving NER priority over LLM fact extraction.
///
/// `pub` + `#[doc(hidden)]` per MNT-002 pattern (E0365 constraint): integration
/// tests in `tests/verify_stage_integration.rs` need to construct this directly
/// under `feature = "test-utils"`.  Not part of the stable public API.
#[doc(hidden)]
pub struct DeferredRequest {
    pub text: String,
    pub reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
    pub content_type: Option<ContentType>,
    /// The episode ID produced by Phase 1, so deferred facts link to the same episode.
    pub episode_id: i64,
    /// Entity names already inserted by Phase 1, passed as hints to the LLM extractor.
    pub ner_entity_names: Vec<String>,
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

impl From<&Error> for IngestErrorKind {
    fn from(e: &Error) -> Self {
        match e {
            Error::Database(_) => IngestErrorKind::Database,
            Error::Extraction(_) => IngestErrorKind::Extraction,
            Error::Resolution(_) => IngestErrorKind::Resolution,
            Error::Llm(_) => IngestErrorKind::Llm,
            Error::Embedding(_) => IngestErrorKind::Embedding,
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
    /// Episode id produced by Phase 1 when the failure occurs at Phase 2
    /// (deferred LLM fact extraction). `0` when the failure occurs during
    /// Phase 1 itself (episode was never committed) or when the episode_id
    /// is unknown. Non-zero values identify "ghost episodes" — Phase 1
    /// committed, Phase 2 failed — queryable via `Memory::ghost_episodes()`.
    pub episode_id: i64,
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
// RateLimit
// ---------------------------------------------------------------------------

/// Token-bucket rate limit for Phase 2 deferred LLM calls.
///
/// Tokens replenish at `tokens_per_second`; `burst` is the maximum number of
/// tokens that can accumulate (= maximum in-burst request count).  When the
/// bucket is exhausted the worker waits until a token is available before
/// dispatching the next Phase 2 LLM call.
///
/// When throttling fires, the counter
/// `kremory.ingest.llm_rate_limit_deferred_total{namespace}` is incremented
/// per ADR-019 / CLAUDE.md Rule 19 (observability-first-class).
#[derive(Debug, Clone)]
pub struct RateLimit {
    /// Tokens replenished per second.  E.g. `2.0` = max 2 LLM calls/s sustained.
    pub tokens_per_second: f64,
    /// Maximum burst token accumulation.  E.g. `5` = burst up to 5 calls.
    pub burst: usize,
}

// ---------------------------------------------------------------------------
// TokenBucketState  (internal — used only by deferred_pipeline)
// ---------------------------------------------------------------------------

/// Internal mutable state for the token bucket.  Held in a `tokio::sync::Mutex`
/// so it can be awaited inside the async worker loop without blocking the thread.
pub(crate) struct TokenBucketState {
    pub(crate) tokens: f64,
    pub(crate) last_refill: std::time::Instant,
    pub(crate) limit: RateLimit,
}

impl TokenBucketState {
    pub(crate) fn new(limit: RateLimit) -> Self {
        let burst = limit.burst as f64;
        Self {
            tokens: burst, // start full
            last_refill: std::time::Instant::now(),
            limit,
        }
    }

    /// Refill tokens based on elapsed time, capped at burst.
    pub(crate) fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens =
            (self.tokens + elapsed * self.limit.tokens_per_second).min(self.limit.burst as f64);
        self.last_refill = std::time::Instant::now();
    }

    /// Returns `true` if a token was immediately available (no wait).
    /// Returns `false` if we had to wait (rate-limit deferral event).
    pub(crate) fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Compute the wait duration until the next token is available.
    pub(crate) fn wait_duration(&self) -> std::time::Duration {
        if self.tokens >= 1.0 {
            return std::time::Duration::ZERO;
        }
        let needed = 1.0 - self.tokens;
        let secs = needed / self.limit.tokens_per_second;
        std::time::Duration::from_secs_f64(secs)
    }
}

// ---------------------------------------------------------------------------
// IngestorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`BackgroundIngestor`].
///
/// `Debug` is implemented manually because `sink` holds a trait-object
/// (`Arc<dyn EnrichmentEventSink>`) which is not `Debug`-derivable.
#[derive(Clone)]
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
    /// a `DeferredRequest` and processes it when the NER channel is idle.
    /// NER always has priority — the deferred queue is only drained during
    /// `recv_timeout` idle periods.
    ///
    /// Set to `false` to run Phase 1 only (e.g., in latency-critical tests or
    /// environments without a capable LLM).
    pub deferred_extraction_enabled: bool,
    /// Number of concurrent Phase 2 (deferred LLM fact extraction) tasks.
    ///
    /// Default: `1` (serialised).  The `BackgroundIngestor` serialisation
    /// invariant is that a single OS thread owns the `Engine` and processes
    /// work items sequentially on a current-thread tokio runtime (`worker_threads(1)`).
    /// This means Phase 2 tasks are awaited inline on the
    /// current-thread tokio runtime, so this field is reserved for future
    /// multi-engine parallelism.  Only override when you understand the
    /// consequent ordering and idempotency implications.
    pub deferred_concurrency: usize,
    /// Optional token-bucket rate limit for Phase 2 LLM calls.
    ///
    /// `None` (default) = unlimited.  When set, Phase 2 LLM calls are
    /// rate-limited at the configured token rate.  When the bucket is
    /// exhausted the worker sleeps until a token is available, emitting
    /// `kremory.ingest.llm_rate_limit_deferred_total{namespace}` per sleep.
    pub llm_rate_limit: Option<RateLimit>,
    /// Optional event sink for background pipeline callbacks.
    ///
    /// When `Some`, the sink receives [`EnrichmentEventSink`] callbacks at each
    /// stage of background Phase 1 + Phase 2 processing.  Callbacks fire
    /// **sync-inline** on the background worker OS thread (ADR-052 D4 contract).
    ///
    /// Set via [`IngestorConfig::with_sink`] builder method.
    /// `None` (default) — no callbacks emitted; all existing code paths
    /// continue to compile and behave identically.
    ///
    /// Refs: ADR-052 Gap 1 + impl spec §3 Phase 2.
    pub sink: Option<Arc<dyn EnrichmentEventSink>>,
}

impl std::fmt::Debug for IngestorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IngestorConfig")
            .field("channel_capacity", &self.channel_capacity)
            .field("error_channel_capacity", &self.error_channel_capacity)
            .field("thread_name", &self.thread_name)
            .field(
                "deferred_extraction_enabled",
                &self.deferred_extraction_enabled,
            )
            .field("deferred_concurrency", &self.deferred_concurrency)
            .field("llm_rate_limit", &self.llm_rate_limit)
            .field(
                "sink",
                &self.sink.as_ref().map(|_| "<dyn EnrichmentEventSink>"),
            )
            .finish()
    }
}

impl IngestorConfig {
    /// Set the event sink for background pipeline callbacks.
    ///
    /// The sink receives [`crate::core::sink::IngestEventSink`] callbacks at each
    /// stage of background Phase 1 + Phase 2 processing.  All callbacks fire
    /// **sync-inline** on the background worker OS thread — keep them fast
    /// (sub-millisecond ideal, sub-100 ms absolute ceiling).
    ///
    /// Per ADR-052 §Gap 1 D1; impl spec §3 Phase 2 `with_sink` builder.
    pub fn with_sink(mut self, sink: impl EnrichmentEventSink + Send + Sync + 'static) -> Self {
        self.sink = Some(Arc::new(sink));
        self
    }
}

impl Default for IngestorConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 64,
            error_channel_capacity: 256,
            thread_name: "rql-ingestor".to_string(),
            deferred_extraction_enabled: true,
            deferred_concurrency: 1,
            llm_rate_limit: None,
            sink: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal channel helper used by both ingestor and deferred_pipeline
// ---------------------------------------------------------------------------

/// Forward an [`IngestError`] to the error channel; drop + log if the channel is full.
pub(crate) fn try_send_error(error_tx: &SyncSender<IngestError>, err: IngestError) {
    if error_tx.try_send(err).is_err() {
        metrics::counter!("rql.background.errors_dropped_total").increment(1);
        tracing::warn!("kremory.background.ingest error dropped (channel full)");
    }
}
