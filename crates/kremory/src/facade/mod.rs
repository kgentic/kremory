//! `kremory::Memory` — fluent facade over the kremory substrate.
//!
//! # Three-tier API (React philosophy)
//!
//! ```text
//! Tier 1 — Just works           Memory::auto / Memory::with_ollama
//!       ↓
//! Tier 2 — Customizable         Memory::open().with_llm().with_embedder().await?
//!       ↓
//! Tier 3 — Composable           kremory::memory::* substrate free functions
//! ```
//!
//! ## Quick start (Tier 1)
//!
//! ```rust,no_run
//! use kremory::Memory;
//! # async fn ex() -> kremory::memory::Result<()> {
//! let mem = Memory::with_ollama("./agent.db").await?;
//! mem.remember("User prefers concise replies").await?;
//! let context: String = mem.recall("what does user prefer?").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Tier 2 builder
//!
//! ```rust,no_run
//! use kremory::{Memory, Namespace, DynEmbeddingProvider};
//! use std::sync::Arc;
//! # async fn ex() -> kremory::memory::Result<()> {
//! # let my_llm: Arc<dyn kremory::memory::ChatProvider> = todo!();
//! # let my_embedder: Arc<dyn DynEmbeddingProvider> = todo!();
//! let mem = Memory::open("./agent.db")
//!     .with_llm(my_llm)
//!     .with_embedder(my_embedder)
//!     .default_namespace(Namespace::new("acme-corp"))
//!     .await?;
//! mem.remember("Customer reported login failure").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Event sink example (Tier 2)
//!
//! ```rust,no_run
//! use kremory::{EnrichmentEventSink, IngestEventSink, ContradictionDetected, BatchPhase2Complete, IngestStatus, IngestionError};
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! struct CountingSink {
//!     entity_count: Arc<AtomicUsize>,
//! }
//!
//! impl IngestEventSink for CountingSink {
//!     fn on_entity_extracted(&self, _id: &str, _name: &str) {
//!         self.entity_count.fetch_add(1, Ordering::Relaxed);
//!     }
//!     fn on_edge_added(&self, _f: &str, _t: &str, _p: &str) {}
//!     fn on_contradiction(&self, _e: ContradictionDetected) {}
//!     fn on_dedup_merge(&self, _s: &str, _a: &str) {}
//!     fn on_stage_change(&self, _s: IngestStatus) {}
//!     fn on_ingestion_error(&self, _e: IngestionError) {}
//! }
//!
//! impl EnrichmentEventSink for CountingSink {
//!     fn on_community_updated(&self, _id: &str, _count: usize) {}
//!     fn on_batch_phase2_complete(&self, _e: BatchPhase2Complete) {}
//! }
//! ```

pub mod providers;

pub mod dream;
pub mod forget;
pub mod recall;
pub mod remember;
pub mod update;

pub use dream::*;
pub use forget::*;
pub use recall::*;
pub use remember::*;
pub use update::*;

use std::future::IntoFuture;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::core::chat_tracking::TokenTrackingChatProvider;
use crate::core::error::Error as CoreError;
use crate::core::provider::DynEmbeddingProvider;
use crate::core::schema::TemporalGraph;
use crate::memory::engine_handle::namespace_to_group_id;
use crate::memory::{
    self,
    events::EnrichmentEventSink,
    types::{
        AwaitOpts, BatchStatus, CancelOutcome, ContextTemplate, DreamHandle, DreamOpts,
        DreamPhaseResult, DreamStatus, EpisodeCommit, Namespace, NamespacePolicy, RetrievedContext,
        SearchOpts, SourceKind, SourceRef, StructuredFact, SubmitOpts,
    },
    ChatProvider, GraphHandle, MemoryError, Result,
};

// ── Type-state markers ────────────────────────────────────────────────────────

/// Type-state marker: LLM not yet configured.
pub struct NoLlm;
/// Type-state marker: LLM configured.
pub struct WithLlm;
/// Type-state marker: Embedder not yet configured.
pub struct NoEmb;
/// Type-state marker: Embedder configured.
pub struct WithEmb;

// ── DreamSummary ─────────────────────────────────────────────────────────────

/// Summary returned when a dream phase completes via the facade.
///
/// Contains the same statistics as the underlying `DreamPhaseResult` / the
/// substrate's consolidation output, wrapped at the facade level.
///
/// ADR-037 §3 D6: `types_discovered` + `warnings` added for Dream Pass 0.
/// ADR-046 Option E E8: `entities_reclassified` added for Dream Pass 2.
#[derive(Debug, Clone)]
pub struct DreamSummary {
    pub communities_updated: usize,
    pub cross_episode_merges: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    pub duration_ms: u64,
    /// Entity types proposed and accepted by Dream Pass 0 type discovery.
    /// Empty when Pass 0 was not run or produced no accepted proposals.
    pub types_discovered: Vec<crate::core::dream::TypeProposal>,
    /// Total entities reclassified by Dream Pass 2 (catch_all_cascade + low_confidence arms).
    /// Zero when Pass 2 was not run or found no candidates.
    pub entities_reclassified: usize,
    /// Warnings emitted during the dream phase.
    /// Includes degraded-mode notices (e.g. anti-redundancy gate skipped).
    pub warnings: Vec<String>,
}

impl From<DreamPhaseResult> for DreamSummary {
    fn from(r: DreamPhaseResult) -> Self {
        Self {
            communities_updated: r.communities_recomputed,
            cross_episode_merges: r.cross_meeting_merges,
            supersessions_recorded: r.supersessions_recorded,
            facts_archived: r.facts_archived,
            duration_ms: r.duration_ms,
            types_discovered: r.types_discovered,
            entities_reclassified: 0,
            warnings: r.dream_warnings,
        }
    }
}

impl From<crate::core::ingest::DreamPassSummary> for DreamSummary {
    fn from(s: crate::core::ingest::DreamPassSummary) -> Self {
        Self {
            // DreamPassSummary fields map to DreamSummary where applicable.
            // Fields without a direct mapping are zeroed.
            communities_updated: 0,
            cross_episode_merges: s.ghost_episodes_retried,
            supersessions_recorded: 0,
            facts_archived: 0,
            duration_ms: s.duration_ms,
            types_discovered: Vec::new(),
            entities_reclassified: s.entities_reclassified,
            warnings: Vec::new(),
        }
    }
}

// ── RecallTemplate ────────────────────────────────────────────────────────────

/// Render strategy for a `RecallRequest`. Facade-level enum mapping to
/// `ContextTemplate` variants with a stable, serializable `as_str` surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallTemplate {
    /// Render entities with name + summary. Matches `ContextTemplate::Entities`.
    Entities,
    /// Render one-line-per-edge compact summary. Matches `ContextTemplate::EdgeSummary`.
    EdgeSummary,
    /// Render temporal facts with `valid_at` annotations (default).
    TemporalFacts,
}

impl RecallTemplate {
    /// Parse from a string slug. Returns `None` for unknown values.
    pub fn parse_str(s: &str) -> Option<Self> {
        match s {
            "entities" => Some(Self::Entities),
            "edge_summary" => Some(Self::EdgeSummary),
            "temporal_facts" => Some(Self::TemporalFacts),
            _ => None,
        }
    }

    /// Return the stable string slug for this template.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entities => "entities",
            Self::EdgeSummary => "edge_summary",
            Self::TemporalFacts => "temporal_facts",
        }
    }
}

impl From<RecallTemplate> for ContextTemplate {
    fn from(t: RecallTemplate) -> Self {
        match t {
            RecallTemplate::Entities => ContextTemplate::Entities,
            RecallTemplate::EdgeSummary => ContextTemplate::EdgeSummary,
            RecallTemplate::TemporalFacts => ContextTemplate::TemporalFacts,
        }
    }
}

// ── Memory struct ─────────────────────────────────────────────────────────────

/// Fluent facade handle for kremory agent memory.
///
/// `Memory` is `Clone + Send + Sync` — cheaply cloneable (`Arc` internally)
/// and safe to share across tokio tasks.
///
/// # Obtain a handle
///
/// - **Tier 1**: `Memory::auto(path).await?` (env-detected provider)
/// - **Tier 1.5**: `Memory::with_ollama(path).await?` etc.
/// - **Tier 2**: `Memory::open(path).with_llm(l).with_embedder(e).await?`
///
/// # Tier 3 (substrate composition)
///
/// Advanced users requiring raw substrate access can use `kremory::memory::*`
/// free functions directly — they remain public and unchanged.
#[derive(Clone)]
pub struct Memory {
    pub(crate) graph: Arc<dyn GraphHandle>,
    /// `None` when built via the NoLlm path (`Memory::open().with_extractor(…).with_embedder(…)`).
    /// Category B methods (dream, recall_with_disambiguation, detect_contradictions) call
    /// `.llm_or_err("method_name")` which returns `Error::LlmRequired` at call time.
    pub(crate) llm: Option<Arc<dyn ChatProvider>>,
    /// Stored for forward-compat (v0.1.1 will wire to real TemporalGraph::open).
    #[allow(dead_code)]
    pub(crate) embedder: Arc<dyn DynEmbeddingProvider>,
    pub(crate) default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    pub(crate) default_namespace: Option<Namespace>,
    /// Direct handle to the underlying `TemporalGraph` for namespace-policy
    /// substrate calls (ADR-029a `register_namespace` + lazy population).
    /// `None` only when `Memory` is constructed by a test path that bypasses
    /// `providers::open_graph` (e.g. with a stub `GraphHandle`). In that case
    /// `register_namespace` returns `MemoryError::Other("…")`.
    pub(crate) temporal_graph: Option<Arc<TemporalGraph>>,
    /// Soft warning threshold for episode content length (chars). When set,
    /// `remember(...).await` emits `tracing::warn!` + a metrics counter if
    /// `content.len()` exceeds this value. Never enforced — observability only.
    /// `None` disables the warning. Default (via builder): `Some(10_000)`.
    pub(crate) episode_content_warn_threshold: Option<usize>,
    /// Background dream scheduler handle. `None` when `DreamSchedule::Off` (default)
    /// or when `Memory` is constructed by a test stub path. Stored as
    /// `Arc<Mutex<Option<…>>>` so `Clone` works without requiring the handle to be
    /// `Clone` (a `JoinHandle<()>` is not `Clone`).
    pub(crate) dream_scheduler:
        std::sync::Arc<std::sync::Mutex<Option<crate::memory::scheduler::DreamSchedulerHandle>>>,
    /// When `true`, `Memory::remember(...).await` blocks until background
    /// extraction (GLiNER/LLM via the ADR-051 worker) has transitioned the
    /// episode to `Verified` (or returns `Err` on `Failed` / timeout).
    ///
    /// Opt-in: default is `false` (fire-and-forget, per ADR-051 design).
    /// Per D1 peer pattern: equivalent to Cognee's `run_in_background=False`.
    ///
    /// ⚠ Cost: enables synchronous-extraction ergonomics at the expense of the
    /// latency benefit ADR-051 provides. Document this trade-off in consumer
    /// code. Prefer `Memory::wait_for_processing` directly for fine-grained
    /// control. See spec §Risk R-06.
    pub(crate) await_extraction: bool,
    /// Timeout applied when `await_extraction = true`.
    ///
    /// Default: 60 seconds (per spec §Risk R-12 mitigation). Configurable via
    /// `MemoryBuilder::with_await_extraction_timeout`.
    pub(crate) await_extraction_timeout: Duration,
}

impl Memory {
    /// Open a database at `path` and start the type-state builder.
    ///
    /// Requires `.with_llm(…)` then `.with_embedder(…)` before `.await`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # use std::sync::Arc;
    /// # async fn ex() -> kremory::memory::Result<()> {
    /// # let llm: Arc<dyn kremory::memory::ChatProvider> = todo!();
    /// # let emb: Arc<dyn kremory::DynEmbeddingProvider> = todo!();
    /// let mem = Memory::open("./agent.db")
    ///     .with_llm(llm)
    ///     .with_embedder(emb)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: impl AsRef<Path>) -> MemoryBuilder<NoLlm, NoEmb> {
        MemoryBuilder {
            path: path.as_ref().to_path_buf(),
            llm: None,
            embedder: None,
            default_sink: None,
            default_namespace: None,
            embedding_dim: None,
            provider_rates_path: None,
            episode_content_warn_threshold: Some(10_000),
            custom_extractor: None,
            #[cfg(feature = "ner")]
            use_gliner: false,
            allowed_entity_types: vec![],
            dream_schedule: crate::memory::scheduler::DreamSchedule::Off,
            await_extraction: false,
            await_extraction_timeout: Duration::from_secs(60),
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }

    // ── Tier 1 shortcuts — implemented in providers.rs ───────────────────────

    /// Env-detected shortcut: OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY → Err.
    ///
    /// Priority order is intentionally local-first to position kremory as a
    /// privacy-first library (no data leaves the machine when Ollama is available).
    pub async fn auto(path: impl AsRef<Path>) -> Result<Self> {
        providers::auto(path).await
    }

    /// Open with Ollama running at `http://localhost:11434`.
    /// Models: `llama3.2` (chat) + `nomic-embed-text` (embeddings).
    pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_ollama(path).await
    }

    /// Open with Ollama at a custom URL.
    pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Self> {
        providers::with_ollama_at(url, path).await
    }

    /// Open with OpenAI. Requires `$OPENAI_API_KEY`.
    /// Models: `gpt-4o-mini` (chat) + `text-embedding-3-small` (embeddings).
    pub async fn with_openai(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_openai(path).await
    }

    /// Open with Anthropic. Requires `$ANTHROPIC_API_KEY`.
    /// Note: Anthropic has no native embedding API; falls back to a deterministic
    /// FNV-1a embedder (dim=384, not semantic — suitable for exact-match recall only).
    /// A `tracing::warn!` is emitted at construction time.
    pub async fn with_anthropic(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_anthropic(path).await
    }

    // ── Test utilities ────────────────────────────────────────────────────────

    /// Access the underlying `TemporalGraph` for integration tests that need
    /// direct SQL access (e.g. asserting `episode_processing_status`).
    ///
    /// Only available under `test` or `test-utils` feature. Not part of the
    /// stable public API.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn temporal_graph_for_test(&self) -> Option<&Arc<TemporalGraph>> {
        self.temporal_graph.as_ref()
    }

    // ── Ingest ────────────────────────────────────────────────────────────────

    /// Ingest a memory episode.
    ///
    /// Default: blocks until Phase 2 enrichment is done.
    /// Use `.no_wait()` to return after Phase 1 commit only.
    ///
    /// # Namespace resolution
    ///
    /// Either `.in_namespace(ns)` on the request OR a `default_namespace` on
    /// the builder is required. Missing both → `Err(MemoryError::MissingNamespace)`.
    #[must_use = "RememberRequest must be .await-ed or have a terminal called"]
    pub fn remember<'a>(&'a self, content: impl Into<String> + 'a) -> RememberRequest<'a> {
        RememberRequest {
            memory: self,
            content: content.into(),
            source_ref: None,
            namespace: None,
            published_at: None,
            facts: vec![],
            sink: None,
            no_wait: false,
            opts: None,
            skip_extraction: false,
        }
    }

    /// Bulk-ingest multiple episodes in a single batch.
    #[must_use = "RememberBatchBuilder must be .await-ed"]
    pub fn remember_batch(&self) -> RememberBatchBuilder<'_> {
        RememberBatchBuilder {
            memory: self,
            episodes: vec![],
            batch_id: None,
            sink: None,
        }
    }

    /// Search for memories matching `query`.
    ///
    /// Default terminal: `.await?` returns `String` via `TemporalFacts` template.
    /// Use `.raw()` for `Vec<RetrievedContext>` or `.as_template(t)` for other templates.
    #[must_use = "RecallRequest must be .await-ed or have a terminal called"]
    pub fn recall<'a>(&'a self, query: impl Into<String> + 'a) -> RecallRequest<'a> {
        RecallRequest {
            memory: self,
            query: query.into(),
            namespace: None,
            namespaces: None,
            per_namespace_top_k: None,
            best_effort: false,
            recall_id: Uuid::new_v4(),
            k: None,
            as_of: None,
            template: Some(RecallTemplate::TemporalFacts),
            raw_mode: false,
            opts: None,
            metadata_filters: Vec::new(),
            metadata_filters_in: Vec::new(),
            pending_error: None,
        }
    }

    /// Delete all episodes in scope.
    ///
    /// Requires either `.in_namespace(ns)` or a `default_namespace`.
    /// Must call `.execute()` explicitly (destructive terminal — no accidental `.await`).
    #[must_use = "ForgetRequest must call .execute() to run"]
    pub fn forget(&self) -> ForgetRequest<'_> {
        ForgetRequest {
            memory: self,
            namespace: None,
            source_id: None,
        }
    }

    /// Run batch consolidation (dream phase).
    ///
    /// Default: blocks until done (returns `DreamSummary`).
    /// Use `.fire_and_forget()` to return `DreamHandle` without blocking.
    #[must_use = "DreamRequest must be .await-ed or have a terminal called"]
    pub fn dream(&self) -> DreamRequest<'_> {
        DreamRequest {
            memory: self,
            namespace: None,
            batch_id: None,
            batch_size: None,
            sink: None,
            fire_and_forget: false,
            opts: None,
        }
    }

    // ── Handle / polling ──────────────────────────────────────────────────────

    /// Query Phase 2 enrichment status for a committed episode.
    pub async fn status_of(
        &self,
        commit: &EpisodeCommit,
    ) -> Result<crate::core::error::IngestStatus> {
        let run_id = commit.run_id.ok_or_else(|| {
            MemoryError::Other("EpisodeCommit has no run_id (Phase 2 was inline)".into())
        })?;
        self.graph.graph_ingest_status(run_id).await
    }

    /// Block until Phase 2 enrichment reaches a terminal status.
    ///
    /// Uses `AwaitOpts::default()` if `timeout` is translated to the opts shape.
    pub async fn await_enrichment(
        &self,
        commit: &EpisodeCommit,
        timeout: Duration,
    ) -> Result<crate::core::error::IngestStatus> {
        let run_id = commit.run_id.ok_or_else(|| {
            MemoryError::Other("EpisodeCommit has no run_id (Phase 2 was inline)".into())
        })?;
        memory::await_enrichment(
            self.graph.as_ref(),
            run_id,
            AwaitOpts {
                timeout,
                ..AwaitOpts::default()
            },
        )
        .await
    }

    // ── ADR-051 async extraction wait API (v0.2.2, Phase 4) ──────────────────

    /// Poll `episodes.episode_processing_status` for `episode_id` until the
    /// episode reaches a terminal state or `timeout` is exceeded.
    ///
    /// # Terminal states
    ///
    /// | Status      | Return value                          |
    /// |-------------|---------------------------------------|
    /// | `Verified`  | `Ok(())`                              |
    /// | `Failed`    | `Err(MemoryError::Core(ExtractionFailed { episode_id }))` |
    /// | timeout     | `Err(MemoryError::Core(WaitTimeout { episode_id, elapsed }))` |
    ///
    /// `Pending` and `Extracting` keep the poll running.
    ///
    /// # Polling schedule (spec §Risk R-05 mitigation)
    ///
    /// - Interval starts at **50 ms**.
    /// - After 5 s elapsed the interval backs off to **200 ms**.
    ///
    /// # Observability
    ///
    /// Emits `kremory.wait_for_processing.duration_ms{outcome}` histogram on
    /// every terminal exit (outcomes: `verified`, `failed`, `timeout`).
    ///
    /// # Errors
    ///
    /// Returns `Err(MemoryError::Other(...))` when `temporal_graph` is `None`
    /// (test-stub path that bypasses `providers::open_graph`).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # use std::time::Duration;
    /// # async fn ex() -> kremory::memory::Result<()> {
    /// # let mem: Memory = todo!();
    /// # let commit: kremory::memory::types::EpisodeCommit = todo!();
    /// // Get the raw episode rowid from the commit's episode_entity_id.
    /// let episode_id: i64 = commit.episode_entity_id.parse().unwrap_or(0);
    /// mem.wait_for_processing(episode_id, Duration::from_secs(30)).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn wait_for_processing(&self, episode_id: i64, timeout: Duration) -> Result<()> {
        use tokio::time::{sleep, Duration as TokioDuration, Instant};

        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::wait_for_processing requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let start = Instant::now();
        let mut interval = TokioDuration::from_millis(50);
        let timeout_dur = timeout;
        let backoff_threshold = TokioDuration::from_secs(5);

        loop {
            // SELECT the current status.
            let status: String = {
                let mut rows = tg
                    .conn
                    .query(
                        "SELECT episode_processing_status FROM episodes WHERE id = ?1",
                        libsql::params![episode_id],
                    )
                    .await
                    .map_err(|e| MemoryError::Core(CoreError::Database(e)))?;
                match rows
                    .next()
                    .await
                    .map_err(|e| MemoryError::Core(CoreError::Database(e)))?
                {
                    Some(row) => row
                        .get::<String>(0)
                        .map_err(|e| MemoryError::Core(CoreError::Database(e)))?,
                    None => {
                        // Episode not found — treat as timeout (episode may not
                        // have committed yet; callers should ensure Phase 1
                        // completed before calling wait_for_processing).
                        let elapsed = start.elapsed();
                        metrics::histogram!(
                            "kremory.wait_for_processing.duration_ms",
                            "outcome" => "not_found"
                        )
                        .record(start.elapsed().as_secs_f64() * 1000.0);
                        return Err(MemoryError::Core(CoreError::WaitTimeout {
                            episode_id,
                            elapsed,
                        }));
                    }
                }
            };

            tracing::debug!(
                target: "kremory.wait_for_processing",
                episode_id,
                status = %status,
                elapsed_ms = start.elapsed().as_millis(),
                "polling episode_processing_status"
            );

            match status.as_str() {
                "Verified" => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    metrics::histogram!(
                        "kremory.wait_for_processing.duration_ms",
                        "outcome" => "verified"
                    )
                    .record(elapsed_ms);
                    return Ok(());
                }
                "Failed" => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    metrics::histogram!(
                        "kremory.wait_for_processing.duration_ms",
                        "outcome" => "failed"
                    )
                    .record(elapsed_ms);
                    return Err(MemoryError::Core(CoreError::ExtractionFailed {
                        episode_id,
                    }));
                }
                // "Pending" | "Extracting" | any future intermediate state
                _ => {}
            }

            // Check timeout BEFORE sleeping — prevents one extra sleep cycle
            // after the budget is exhausted.
            if start.elapsed() >= timeout_dur {
                let elapsed = start.elapsed();
                metrics::histogram!(
                    "kremory.wait_for_processing.duration_ms",
                    "outcome" => "timeout"
                )
                .record(start.elapsed().as_secs_f64() * 1000.0);
                return Err(MemoryError::Core(CoreError::WaitTimeout {
                    episode_id,
                    elapsed,
                }));
            }

            // Backoff: after 5 s switch from 50 ms to 200 ms interval
            // (spec §Risk R-05 — avoids busy-polling long extractions).
            if start.elapsed() >= backoff_threshold {
                interval = TokioDuration::from_millis(200);
            }

            sleep(interval).await;
        }
    }

    /// Block until the dream phase handle reaches a terminal status.
    pub async fn await_dream(
        &self,
        handle: &DreamHandle,
        timeout: Duration,
    ) -> Result<DreamStatus> {
        memory::await_dream(
            self.graph.as_ref(),
            handle.run_id,
            AwaitOpts {
                timeout,
                ..AwaitOpts::default()
            },
        )
        .await
    }

    /// Block until all episodes in `batch_id` reach a terminal status.
    pub async fn await_batch(&self, batch_id: &str, timeout: Duration) -> Result<BatchStatus> {
        memory::await_batch_enrichment(
            self.graph.as_ref(),
            batch_id,
            AwaitOpts {
                timeout,
                ..AwaitOpts::default()
            },
        )
        .await
    }

    /// Cancel an in-flight Phase 2 or Phase 3 run.
    pub async fn cancel(&self, commit: &EpisodeCommit) -> Result<CancelOutcome> {
        let run_id = commit.run_id.ok_or_else(|| {
            MemoryError::Other("EpisodeCommit has no run_id — cannot cancel inline Phase 2".into())
        })?;
        self.graph.graph_cancel(run_id).await
    }

    /// Cancel a dream phase by its handle.
    pub async fn cancel_dream(&self, handle: &DreamHandle) -> Result<CancelOutcome> {
        self.graph.graph_cancel(handle.run_id).await
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Flush any pending writes and close the memory handle.
    ///
    /// At v0.1.0 this is a no-op stub. v0.1.1 will add WAL flush semantics.
    /// Callers should call this at shutdown to future-proof their code.
    pub async fn close(&self) -> Result<()> {
        // v0.1.0: no-op. WAL flush deferred to v0.1.1.
        Ok(())
    }

    // ── Namespace policy (ADR-029a, v0.1.4) ───────────────────────────────────

    /// Register a namespace + its policy explicitly, ahead of any writes.
    ///
    /// # Idempotency
    ///
    /// Calling `register_namespace` with the SAME `(group_id, policy)` pair
    /// returns `Ok(())`. Calling with a DIFFERENT policy on an existing
    /// `group_id` returns
    /// `Err(MemoryError::Core(Error::NamespacePolicyImmutable { ... }))`.
    /// This makes startup code safe to re-execute (idempotent against
    /// persisted state).
    ///
    /// # Validation
    ///
    /// The policy attached to `namespace` is validated via
    /// [`NamespacePolicy::validate`] before persistence. Incoherent policies
    /// surface as `Err(MemoryError::Core(Error::InvalidPolicy(...)))`.
    ///
    /// # No enforcement at v0.1.4
    ///
    /// The policy is PERSISTED but not yet enforced on
    /// `dream()` / `forget()` / mutation operations. Enforcement lands in
    /// v0.1.5+ per ADR-029b. Every non-default policy registration emits
    /// a `tracing::warn!` on target `kremory.namespace` to make the
    /// declaration vs enforcement gap visible.
    ///
    /// # Race semantics — atomic via `BEGIN IMMEDIATE`
    ///
    /// `register_namespace` wraps the SELECT + INSERT pair in a
    /// `BEGIN IMMEDIATE` transaction (per ADR-022 write_lock invariant). This
    /// acquires SQLite's RESERVED write lock before reading, serializing
    /// against concurrent `remember(...)` calls that would implicitly create
    /// the namespace with default policy.
    ///
    /// The recommended pattern is `register_namespace` AT STARTUP before any
    /// `remember(...)`. See ADR-029a Decision 6 for the three race outcomes.
    pub async fn register_namespace(&self, namespace: Namespace) -> Result<()> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::register_namespace requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let policy = namespace.policy.clone().unwrap_or_default();
        policy
            .validate()
            .map_err(|e| MemoryError::Core(CoreError::InvalidPolicy(e)))?;

        let group_id = namespace_to_group_id(&namespace);
        let is_non_default = policy != NamespacePolicy::default();

        // Atomic INSERT-or-compare via BEGIN IMMEDIATE.
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(MemoryError::Core)?;
        let stored = tg
            .get_namespace_policy(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        let outcome: Result<()> = match stored {
            Some(existing) if existing == policy => Ok(()),
            Some(existing) => Err(MemoryError::Core(CoreError::NamespacePolicyImmutable {
                namespace: group_id.clone(),
                stored: existing,
                attempted: policy.clone(),
            })),
            None => tg
                .set_namespace_policy(&group_id, &policy)
                .await
                .map_err(MemoryError::Core),
        };
        match &outcome {
            Ok(()) => {
                guard.commit().await.map_err(MemoryError::Core)?;
            }
            Err(_) => {
                guard.rollback().await.map_err(MemoryError::Core)?;
            }
        }

        // Operational visibility: every non-default policy DECLARATION emits
        // warn (NOT info) — closes Vera cycle-1 HIGH-1 footgun. Default
        // policies are silent (they would be the existing behaviour).
        if outcome.is_ok() && is_non_default {
            tracing::warn!(
                target: "kremory.namespace",
                group_id = %group_id,
                policy = ?policy,
                "kremory.namespace.policy_declared: POLICY DECLARED BUT NOT \
                 ENFORCED at v0.1.4 — enforcement lands v0.1.5+ per ADR-029b. \
                 See https://docs.rs/kremory/0.1.4/kremory/#adr-029a"
            );
        }

        outcome
    }

    /// Monotonically upgrade a namespace's immutability from `Mutable` to
    /// `AppendOnly` (ADR-029b Decision 5).
    ///
    /// This is a **one-way ratchet**: `Mutable → AppendOnly` is the only
    /// allowed direction. Attempting to downgrade (`AppendOnly → Mutable`)
    /// returns `Err(MemoryError::Core(Error::NamespacePolicyImmutable))`.
    /// Calling on an already-`AppendOnly` namespace is idempotent (`Ok(())`).
    ///
    /// # Atomicity
    ///
    /// The read-decide-write sequence is wrapped in a `BEGIN IMMEDIATE`
    /// transaction to prevent races with concurrent `register_namespace` or
    /// `upgrade_namespace_policy` calls.
    ///
    /// # Errors
    ///
    /// - `MemoryError::Other` — `Memory` not constructed via the builder path.
    /// - `MemoryError::Core(Error::NamespacePolicyImmutable)` — downgrade
    ///   attempted or policy mismatch.
    /// - `MemoryError::Core(Error::Other)` — substrate failure.
    pub async fn upgrade_namespace_policy(&self, namespace: Namespace) -> Result<()> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::upgrade_namespace_policy requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let group_id = namespace_to_group_id(&namespace);
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(MemoryError::Core)?;

        let stored = tg
            .get_namespace_policy(&group_id)
            .await
            .map_err(MemoryError::Core)?;

        let target_policy = crate::memory::types::NamespacePolicy::APPEND_ONLY;

        let outcome: Result<()> = match stored {
            Some(ref existing)
                if existing.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly =>
            {
                // Already AppendOnly — idempotent.
                Ok(())
            }
            Some(ref existing)
                if existing.immutability == crate::memory::types::ImmutabilityLevel::Mutable =>
            {
                // Upgrade Mutable → AppendOnly.
                tg.set_namespace_policy_with_upgraded_at(&group_id, &target_policy)
                    .await
                    .map_err(MemoryError::Core)
            }
            Some(existing) => {
                // Unexpected policy state — treat as immutable conflict.
                Err(MemoryError::Core(CoreError::NamespacePolicyImmutable {
                    namespace: group_id.clone(),
                    stored: existing,
                    attempted: target_policy,
                }))
            }
            None => {
                // Namespace not yet registered — create directly as AppendOnly.
                tg.set_namespace_policy_with_upgraded_at(&group_id, &target_policy)
                    .await
                    .map_err(MemoryError::Core)
            }
        };

        match &outcome {
            Ok(()) => {
                guard.commit().await.map_err(MemoryError::Core)?;
                // Invalidate cache so next read reflects the new AppendOnly policy.
                tg.invalidate_policy_cache(&group_id);
                tracing::info!(
                    target: "kremory.namespace",
                    group_id = %group_id,
                    "kremory.namespace.policy_upgraded: namespace upgraded to AppendOnly"
                );
            }
            Err(_) => {
                guard.rollback().await.map_err(MemoryError::Core)?;
            }
        }
        outcome
    }

    /// Lazy-population helper: ensure a default-policy row exists for the
    /// `namespace` if it has not been observed yet. Invoked from the first-
    /// encounter paths (`remember`, `recall`, `forget`, `dream`) per ADR-029a
    /// Decision 8.
    ///
    /// Best-effort: when `Memory` is constructed without a direct
    /// `Arc<TemporalGraph>` (e.g. test-only stub-handle path) this is a no-op.
    /// Errors from the substrate are converted to `MemoryError::Core` and
    /// returned so the call site can decide whether to fail the user request.
    pub(crate) async fn ensure_namespace_policy(&self, namespace: &Namespace) -> Result<()> {
        let Some(tg) = self.temporal_graph.as_ref() else {
            return Ok(());
        };
        let group_id = namespace_to_group_id(namespace);
        tg.ensure_namespace_policy_row(&group_id)
            .await
            .map_err(MemoryError::Core)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn resolve_namespace(&self, override_ns: Option<Namespace>) -> Result<Namespace> {
        override_ns
            .or_else(|| self.default_namespace.clone())
            .ok_or_else(|| {
                MemoryError::MissingNamespace {
                    request: "namespace is required — call .in_namespace(ns) or set default_namespace on the builder",
                }
            })
    }

    fn resolve_sink(
        &self,
        per_call: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Option<Arc<dyn EnrichmentEventSink>> {
        per_call.or_else(|| self.default_sink.clone())
    }

    /// Return the wired LLM or `Err(MemoryError::Core(Error::LlmRequired))`.
    ///
    /// Used by Category B methods (dream, recall_with_disambiguation,
    /// detect_contradictions) that unconditionally require an LLM.
    pub(crate) fn llm_or_err(
        &self,
        method: &'static str,
        hint: &'static str,
    ) -> Result<Arc<dyn ChatProvider>> {
        self.llm.clone().ok_or_else(|| {
            tracing::warn!(
                method = method,
                "LLM required but not wired — returning LlmRequired"
            );
            metrics::counter!("kremory.llm_required_total", "method" => method).increment(1);
            MemoryError::Core(CoreError::LlmRequired { method, hint })
        })
    }

    /// Return the wired LLM, or a no-op stub when no LLM was configured.
    ///
    /// Used by Category A methods (remember, remember_batch) that pass a
    /// provider arg to `submit_episode` but the engine ignores the arg when
    /// `skip_extraction = true` or when the custom extractor handles extraction.
    /// The real `LlmRequired` error fires from `pipeline.rs` if an LLM-dependent
    /// pipeline step is actually reached without a wired LLM.
    pub(crate) fn llm_or_stub(&self) -> Arc<dyn ChatProvider> {
        self.llm
            .clone()
            .unwrap_or_else(|| Arc::new(crate::core::provider::NullChatProvider))
    }

    // ── Dream pass API (Phase C DoD C1, C4, C5, C11) ─────────────────────────

    /// Run a synchronous dream pass with the given options.
    ///
    /// At-most-one concurrent dream pass per engine (serialised via internal
    /// Mutex). Subsequent calls will block until the active pass completes.
    ///
    /// Pass 0 (type discovery) and Pass 2 (ghost episode retry) are **stubbed**
    /// in Phase C — they return empty/zero counts. Real logic lands in Phase D/E.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C1 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn run_dream_pass_sync(
        &self,
        opts: crate::core::ingest::DreamPassOpts,
    ) -> Result<DreamSummary> {
        self.graph.graph_run_dream_pass_sync(opts).await
    }

    /// Return episode IDs where Phase 1 succeeded but Phase 2 produced no facts
    /// (ghost episodes).
    ///
    /// An optional `group_id` restricts the query to one namespace/thread.
    /// `None` returns ghost episodes across all namespaces.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C4 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        self.graph.graph_ghost_episodes(group_id).await
    }

    /// Pin an entity as `ConsumerPinned`, protecting it from dream reclassification.
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` on the entity row.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C5 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn assert_entity_type(
        &self,
        entity_id: &str,
        entity_type_id: u32,
        group_id: Option<&str>,
    ) -> Result<()> {
        self.graph
            .graph_assert_entity_type(entity_id, entity_type_id, group_id)
            .await
    }

    /// Start a dream scheduler background task at runtime.
    ///
    /// Returns a [`DreamSchedulerHandle`] the caller can use to stop the task.
    /// For scheduler-at-build-time, use [`MemoryBuilder::with_dream_schedule`]
    /// instead.
    ///
    /// If a scheduler was already started via `with_dream_schedule` at build
    /// time, calling this method starts an ADDITIONAL independent scheduler.
    /// Stop the build-time one via [`Memory::stop_dream_scheduler`] first if
    /// you want to replace it.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C11 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub fn start_dream_scheduler(
        &self,
        schedule: crate::memory::scheduler::DreamSchedule,
    ) -> crate::memory::scheduler::DreamSchedulerHandle {
        crate::memory::scheduler::spawn_scheduler(
            Arc::clone(&self.graph),
            schedule,
            crate::core::ingest::DreamPassOpts::default,
        )
    }

    /// Stop the scheduler that was started via `with_dream_schedule` at build
    /// time, if one is running.
    ///
    /// No-op if no build-time scheduler is active. Returns `true` if a
    /// scheduler was stopped, `false` if none was running.
    pub async fn stop_dream_scheduler(&self) -> bool {
        let handle = self
            .dream_scheduler
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(h) = handle {
            h.stop().await;
            true
        } else {
            false
        }
    }
}

// ── MemoryBuilder ─────────────────────────────────────────────────────────────

/// Type-state builder for `Memory`. Compile-time enforced: `.with_llm()` then
/// `.with_embedder()` are both required before `.await`.
///
/// Optional: `.with_event_sink()`, `.default_namespace()`, `.embedding_dim()`.
#[must_use = "MemoryBuilder must be configured with .with_llm() AND .with_embedder() before .await"]
pub struct MemoryBuilder<L, E> {
    path: std::path::PathBuf,
    llm: Option<Arc<dyn ChatProvider>>,
    embedder: Option<Arc<dyn DynEmbeddingProvider>>,
    default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    default_namespace: Option<Namespace>,
    embedding_dim: Option<usize>,
    /// Optional path to a custom `provider-rates.toml`. When `Some`, overrides
    /// the bundled rates file at build time.
    provider_rates_path: Option<PathBuf>,
    /// Soft warning threshold for episode content length (chars).
    /// Default `Some(10_000)` via `Memory::open` — matches Zep's recommended
    /// chunk size. `None` disables the warning. Never enforced as a hard limit;
    /// observability only.
    episode_content_warn_threshold: Option<usize>,
    /// Custom extractor supplied via `.with_extractor(Arc<impl EntityExtractor>)`.
    /// When `Some`, overrides LLM-derived extraction. Mutually exclusive with
    /// `.with_gliner()` — builder errors at `build()` if both are set.
    custom_extractor: Option<Arc<dyn crate::core::intelligence::EntityExtractorDyn>>,
    /// GLiNER enabled via `.with_gliner()`. Requires the `ner` cargo feature.
    /// When set without `.with_llm()`, builder errors at `build()` because
    /// GLiNER candidate-gen still needs one LLM typing call.
    #[cfg(feature = "ner")]
    use_gliner: bool,
    /// Entity type names the extractor is allowed to emit. Forwarded to
    /// `PipelineConfig::allowed_entity_types`. When empty (the default), the
    /// `GlinerExtractor` rejects all entities — callers that activate the `ner`
    /// feature MUST supply this via [`MemoryBuilder::allowed_entity_types`].
    allowed_entity_types: Vec<String>,
    /// Automatic dream-pass scheduling policy.
    /// Default: `DreamSchedule::Off` (no background task).
    dream_schedule: crate::memory::scheduler::DreamSchedule,
    /// When `true`, `Memory::remember(...).await` blocks until the background
    /// extraction pipeline transitions the episode to `Verified` (or returns
    /// `Err` on `Failed` / timeout). Default: `false`.
    /// Per spec §Phase 4 / ADR-051 D1 peer pattern (Cognee `run_in_background=False`).
    await_extraction: bool,
    /// Timeout applied when `await_extraction = true`.
    /// Default: 60 s (spec §Risk R-12 mitigation).
    await_extraction_timeout: Duration,
    _llm_state: std::marker::PhantomData<L>,
    _emb_state: std::marker::PhantomData<E>,
}

impl<L, E> MemoryBuilder<L, E> {
    /// Set the default event sink for all subsequent operations.
    /// Per-call sinks (via `.with_event_sink()` on request builders) override this.
    ///
    /// # Sink callback contract (G7 — v0.1.6)
    ///
    /// Sink methods are called **sync inline** on the Phase 2 enrichment
    /// pipeline thread. Slow callbacks stall ingest. See the
    /// [`EnrichmentEventSink`] trait rustdoc for the full contract +
    /// recommended consumer pattern (buffer + return immediately + drain
    /// async).
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.default_sink = Some(sink);
        self
    }

    /// Set the default namespace used by operations that don't specify `.in_namespace()`.
    pub fn default_namespace(mut self, ns: Namespace) -> Self {
        self.default_namespace = Some(ns);
        self
    }

    /// Set the embedding vector dimensionality.
    ///
    /// **Must match your embedder's output dimension.** A mismatch causes a
    /// SQLite vector index failure on the first `remember()` call:
    /// `vector index(insert): dimensions are different: <actual> != <config>`.
    ///
    /// Common values:
    /// - `384` — MiniLM-L6-v2 (default if not set)
    /// - `768` — `nomic-embed-text` (Ollama), `all-mpnet-base-v2`
    /// - `1536` — OpenAI `text-embedding-3-small`
    /// - `3072` — OpenAI `text-embedding-3-large`
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = Some(dim);
        self
    }

    /// Override the soft warn threshold for episode content length (chars).
    ///
    /// Default: `Some(10_000)` (Zep-compatible). Set `None` to disable the
    /// warning entirely. Never enforced as a hard limit; the threshold only
    /// drives `tracing::warn!` + a `kremory_episode_oversize_total` counter
    /// at ingest time so callers can spot extraction-quality risk early.
    pub fn episode_content_warn_threshold(mut self, threshold: Option<usize>) -> Self {
        self.episode_content_warn_threshold = threshold;
        self
    }

    /// Provide a custom entity extractor (BYOE — bring your own extractor).
    ///
    /// Accepts any type that implements [`EntityExtractor`]. Wraps it in
    /// `Arc<dyn EntityExtractorDyn>` internally for object-safe dispatch.
    ///
    /// - Mutually exclusive with `.with_gliner()` — builder errors at build time
    ///   if both are set.
    /// - Compatible with or without `.with_llm()`: custom extractor runs regardless.
    ///   If LLM is also wired, it remains available for Category B methods.
    pub fn with_extractor<Ext>(mut self, extractor: Arc<Ext>) -> Self
    where
        Ext: crate::core::intelligence::EntityExtractor + 'static,
    {
        self.custom_extractor =
            Some(extractor as Arc<dyn crate::core::intelligence::EntityExtractorDyn>);
        self
    }

    /// Enable GLiNER-based candidate generation (requires the `ner` cargo feature).
    ///
    /// When combined with `.with_llm()`, the builder selects `ExtractorKind::GlinerLlm`
    /// (GLiNER for candidate spans + one LLM typing call per batch).
    ///
    /// Without `.with_llm()`, the builder errors at build time — GLiNER candidate-gen
    /// still requires one LLM call for entity-type classification.
    ///
    /// `_config` is reserved for future tuning knobs (threshold, model path, batch size)
    /// per ADR-039 §A6 deferred architecture path X. Currently has no public fields.
    #[cfg(feature = "ner")]
    pub fn with_gliner(mut self, _config: crate::core::extraction::GlinerConfig) -> Self {
        self.use_gliner = true;
        self
    }

    /// Set the entity type names the extractor is allowed to emit.
    ///
    /// Forwarded to [`PipelineConfig::allowed_entity_types`]. Required when
    /// the `ner` feature is active and you want extraction to produce results.
    /// With an empty list (the default) the `GlinerExtractor` will reject all
    /// candidate entities.
    ///
    /// Typically callers pass the names from `DEFAULT_ENTITY_TYPES`:
    ///
    /// ```rust,no_run
    /// use kremory::Memory;
    /// use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
    ///
    /// let names: Vec<String> = DEFAULT_ENTITY_TYPES
    ///     .iter()
    ///     .map(|(_, name, _)| name.to_string())
    ///     .collect();
    /// // Memory::open("./db").allowed_entity_types(names)…
    /// ```
    pub fn allowed_entity_types(mut self, types: Vec<String>) -> Self {
        self.allowed_entity_types = types;
        self
    }

    /// Configure automatic dream-pass scheduling.
    ///
    /// Default: [`DreamSchedule::Off`] — no background task is spawned.
    ///
    /// When set to a non-`Off` variant, a background tokio task is spawned
    /// during `.await` (i.e. at `MemoryBuilder::into_future`). The task runs
    /// until [`DreamSchedulerHandle::stop`] is called or the `Memory` is
    /// dropped.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kremory::{Memory, DreamSchedule};
    /// use std::time::Duration;
    /// # async fn ex() -> kremory::memory::Result<()> {
    /// # let llm = todo!(); let emb = todo!();
    /// let mem = Memory::open("./agent.db")
    ///     .with_llm(llm)
    ///     .with_embedder(emb)
    ///     .with_dream_schedule(DreamSchedule::Interval(Duration::from_secs(300)))
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C10 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub fn with_dream_schedule(
        mut self,
        schedule: crate::memory::scheduler::DreamSchedule,
    ) -> Self {
        self.dream_schedule = schedule;
        self
    }

    /// Opt-in to synchronous-extraction ergonomics: when `true`,
    /// `Memory::remember(...).await` blocks until the ADR-051 background worker
    /// has transitioned the episode to `Verified` (returns `Ok(())`) or
    /// `Failed` / timeout (returns `Err`).
    ///
    /// Default: `false` (fire-and-forget — the ADR-051 design intent).
    ///
    /// Use [`with_await_extraction_timeout`](Self::with_await_extraction_timeout)
    /// to configure the maximum wait duration (default 60 s, see spec §Risk R-12).
    ///
    /// ⚠ **Cost**: enables sync semantics at the expense of the hot-path latency
    /// benefit that ADR-051 provides. Prefer `Memory::wait_for_processing` for
    /// fine-grained per-episode control (spec §Phase 4, Risk R-06).
    ///
    /// Per D1 peer pattern: equivalent to Cognee's `run_in_background=False`.
    pub fn with_await_extraction(mut self, await_extraction: bool) -> Self {
        self.await_extraction = await_extraction;
        self
    }

    /// Configure the maximum time `Memory::remember` will wait when
    /// `with_await_extraction(true)` is set.
    ///
    /// Default: 60 seconds (per spec §Risk R-12 mitigation — prevents
    /// false-timeout on real 30 s LLM extractions).
    ///
    /// Has no effect when `await_extraction` is `false` (the default).
    pub fn with_await_extraction_timeout(mut self, timeout: Duration) -> Self {
        self.await_extraction_timeout = timeout;
        self
    }
}

impl MemoryBuilder<NoLlm, NoEmb> {
    /// Configure the LLM provider (required).
    ///
    /// The provider is used as-is, without token or cost instrumentation. For
    /// automatic observability, prefer [`with_llm_tracked`](Self::with_llm_tracked).
    pub fn with_llm(self, llm: Arc<dyn ChatProvider>) -> MemoryBuilder<WithLlm, NoEmb> {
        MemoryBuilder {
            path: self.path,
            llm: Some(llm),
            embedder: self.embedder,
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }

    /// Wrap a user-supplied `ChatProvider` in a [`TokenTrackingChatProvider`],
    /// capturing `(provider, model)` labels for metrics emission.
    ///
    /// Preferred over [`with_llm`](Self::with_llm) when the caller wants automatic
    /// token count, cost, and duration observability via the kremory metrics surface
    /// (`kremory_core_tokens_total`, `kremory_core_cost_usd_total`,
    /// `kremory_core_chat_duration_seconds`).
    ///
    /// The `provider` and `model` labels must match entries in
    /// `monitoring/provider-rates.toml` (or a custom rates file via
    /// `with_provider_rates_path`) for
    /// cost counters to emit non-zero values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kremory::Memory;
    /// # async fn ex<L>(
    /// #     my_openai_client: L,
    /// #     my_embedder: std::sync::Arc<dyn kremory::DynEmbeddingProvider>,
    /// # ) -> kremory::memory::Result<()>
    /// # where
    /// #     L: kremory::memory::ChatProvider + Send + Sync + 'static,
    /// # {
    /// let memory = Memory::open("./agent.db")
    ///     .with_llm_tracked("openai", "gpt-4o-mini", my_openai_client)
    ///     .with_embedder(my_embedder)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_llm_tracked<L: ChatProvider + Send + Sync + 'static>(
        self,
        provider: impl Into<String>,
        model: impl Into<String>,
        llm: L,
    ) -> MemoryBuilder<WithLlm, NoEmb> {
        let tracked = TokenTrackingChatProvider::new(llm, provider, model);
        MemoryBuilder {
            path: self.path,
            llm: Some(Arc::new(tracked) as Arc<dyn ChatProvider>),
            embedder: self.embedder,
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }
}

impl MemoryBuilder<WithLlm, NoEmb> {
    /// Configure the embedding provider (required).
    pub fn with_embedder(
        self,
        emb: Arc<dyn DynEmbeddingProvider>,
    ) -> MemoryBuilder<WithLlm, WithEmb> {
        MemoryBuilder {
            path: self.path,
            llm: self.llm,
            embedder: Some(emb),
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }
}

impl MemoryBuilder<NoLlm, NoEmb> {
    /// Configure the embedding provider for the no-LLM path.
    ///
    /// Use this when you supply a custom extractor via `.with_extractor(…)` but
    /// do not need an LLM provider. The resulting `Memory` supports all Category A
    /// operations; Category B operations (`dream`, `recall_with_disambiguation`,
    /// `detect_contradictions`) return `Error::LlmRequired` at call time.
    pub fn with_embedder(
        self,
        emb: Arc<dyn DynEmbeddingProvider>,
    ) -> MemoryBuilder<NoLlm, WithEmb> {
        MemoryBuilder {
            path: self.path,
            llm: self.llm,
            embedder: Some(emb),
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
            embedding_dim: self.embedding_dim,
            provider_rates_path: self.provider_rates_path,
            episode_content_warn_threshold: self.episode_content_warn_threshold,
            custom_extractor: self.custom_extractor,
            #[cfg(feature = "ner")]
            use_gliner: self.use_gliner,
            allowed_entity_types: self.allowed_entity_types,
            dream_schedule: self.dream_schedule,
            await_extraction: self.await_extraction,
            await_extraction_timeout: self.await_extraction_timeout,
            _llm_state: std::marker::PhantomData,
            _emb_state: std::marker::PhantomData,
        }
    }
}

impl IntoFuture for MemoryBuilder<WithLlm, WithEmb> {
    type Output = Result<Memory>;
    type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            // Initialize provider rates (idempotent). Errors are logged but not
            // fatal — cost counters will skip emission with a one-shot warn.
            if let Some(ref custom_path) = self.provider_rates_path {
                if let Err(e) = crate::core::rates::init_from_path(custom_path.as_path()) {
                    tracing::warn!(
                        error = %e,
                        path  = %custom_path.display(),
                        "failed to load custom provider-rates.toml — cost counters will not be emitted"
                    );
                }
            } else if let Err(e) = crate::core::rates::init_bundled() {
                tracing::warn!(
                    error = %e,
                    "failed to load bundled provider-rates.toml — cost counters will not be emitted"
                );
            }

            let llm = self
                .llm
                .ok_or_else(|| MemoryError::Other("llm missing".into()))?;
            let embedder = self
                .embedder
                .ok_or_else(|| MemoryError::Other("embedder missing".into()))?;

            // ── Compat matrix (ADR-039, 7-row table) ────────────────────────
            // Row 6: .with_extractor conflicts with .with_gliner → Err
            #[cfg(feature = "ner")]
            if self.custom_extractor.is_some() && self.use_gliner {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: ".with_extractor conflicts with .with_gliner — \
                             supply one or the other, not both"
                        .into(),
                }));
            }

            // Row 2: LLM only → Llm extractor (default open_graph path)
            // Row 3: LLM + gliner → GlinerLlm extractor
            // Row 5/6 (with LLM): custom extractor wins, LLM still available
            let (graph, temporal_graph) = if let Some(custom) = self.custom_extractor {
                // Rows 5/6 with LLM — open with explicit Custom extractor
                providers::open_graph_with_extractor(
                    providers::GraphOpenParams {
                        path: self.path.clone(),
                        embedder: embedder.clone(),
                        embedding_dim: self.embedding_dim,
                        allowed_entity_types: self.allowed_entity_types,
                    },
                    llm.clone(),
                    crate::core::extraction::factory::ExtractorKind::Custom(custom),
                )
                .await?
            } else {
                #[cfg(feature = "ner")]
                if self.use_gliner {
                    // Row 3: LLM + gliner → GlinerLlm
                    use crate::core::provider::ArcChatProvider;
                    let arc_llm = Arc::new(ArcChatProvider::new(llm.clone()));
                    let gliner_ext =
                        crate::core::extraction::hybrid_typer::GlinerLlmExtractor::new(arc_llm)
                            .map_err(|e| {
                                MemoryError::Core(CoreError::BuilderConflict {
                                    detail: format!("GLiNER extractor init failed: {e}"),
                                })
                            })?;
                    providers::open_graph_with_extractor(
                        providers::GraphOpenParams {
                            path: self.path.clone(),
                            embedder: embedder.clone(),
                            embedding_dim: self.embedding_dim,
                            allowed_entity_types: self.allowed_entity_types,
                        },
                        llm.clone(),
                        crate::core::extraction::factory::ExtractorKind::GlinerLlm(Box::new(
                            gliner_ext,
                        )),
                    )
                    .await?
                } else {
                    // Row 2: LLM only → default Llm extractor
                    providers::open_graph(
                        self.path.as_path(),
                        llm.clone(),
                        embedder.clone(),
                        self.embedding_dim,
                        self.allowed_entity_types,
                    )
                    .await?
                }
                #[cfg(not(feature = "ner"))]
                {
                    // Row 2 (no ner feature): default Llm extractor
                    providers::open_graph(
                        self.path.as_path(),
                        llm.clone(),
                        embedder.clone(),
                        self.embedding_dim,
                        self.allowed_entity_types,
                    )
                    .await?
                }
            };

            // T6.2 — Warm schema caches after graph open, before returning.
            // Best-effort: errors are silently ignored inside warm_schema_caches.
            // Spawned on a background tokio task so engine startup is not blocked
            // by the 10 × LLM round-trips.
            // Gated to non-test builds: unit tests use fast-open paths and must
            // not incur LLM round-trips on every engine construction.
            // model is passed as None: MemoryBuilder does not surface the model
            // string at construction time.  Warmup is connection-pool warm only
            // (LlmJsonRepair arm); NativeSchema/FormatSchema arms are not reached.
            // Tier 1 callers (providers::build_memory) pass the concrete model
            // string and reach the provider-native schema-compilation path.
            #[cfg(not(test))]
            {
                use crate::core::extraction::structured::warm_schema_caches;
                use crate::core::provider::ArcChatProvider;
                let warmup_llm = Arc::new(ArcChatProvider::new(llm.clone()));
                tokio::spawn(async move {
                    warm_schema_caches(warmup_llm.as_ref(), None).await;
                });
            }

            // ── Dream scheduler (C10) ────────────────────────────────────────
            // Spawn background scheduler if a non-Off schedule was requested.
            let dream_scheduler_handle = match self.dream_schedule {
                crate::memory::scheduler::DreamSchedule::Off => None,
                schedule => {
                    let handle = crate::memory::scheduler::spawn_scheduler(
                        Arc::clone(&graph),
                        schedule,
                        crate::core::ingest::DreamPassOpts::default,
                    );
                    Some(handle)
                }
            };

            Ok(Memory {
                graph,
                llm: Some(llm),
                embedder,
                default_sink: self.default_sink,
                default_namespace: self.default_namespace,
                temporal_graph: Some(temporal_graph),
                episode_content_warn_threshold: self.episode_content_warn_threshold,
                dream_scheduler: std::sync::Arc::new(std::sync::Mutex::new(dream_scheduler_handle)),
                await_extraction: self.await_extraction,
                await_extraction_timeout: self.await_extraction_timeout,
            })
        })
    }
}

impl IntoFuture for MemoryBuilder<NoLlm, WithEmb> {
    type Output = Result<Memory>;
    type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            // Row 4: gliner set but no LLM → Err
            #[cfg(feature = "ner")]
            if self.use_gliner && self.custom_extractor.is_none() {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: "GLiNER candidate-gen needs LLM for entity-type classification — \
                             add .with_llm(…) or swap to .with_extractor(…) for a fully \
                             custom extractor that doesn't require LLM"
                        .into(),
                }));
            }

            // Row 6 (conflict): with_extractor + with_gliner → Err
            #[cfg(feature = "ner")]
            if self.custom_extractor.is_some() && self.use_gliner {
                return Err(MemoryError::Core(CoreError::BuilderConflict {
                    detail: ".with_extractor conflicts with .with_gliner — \
                             supply one or the other, not both"
                        .into(),
                }));
            }

            // Row 0: nothing wired → Err
            let custom = self.custom_extractor.ok_or_else(|| {
                MemoryError::Core(CoreError::BuilderConflict {
                    detail: "no extractor wired — call .with_llm(…) for built-in LLM extraction, \
                             or .with_extractor(Arc<impl EntityExtractor>) to bring your own"
                        .into(),
                })
            })?;

            let embedder = self
                .embedder
                .ok_or_else(|| MemoryError::Other("embedder missing".into()))?;

            // Row 4 (no LLM, custom extractor) — open without LLM
            let (graph, temporal_graph) = providers::open_graph_no_llm(
                providers::GraphOpenParams {
                    path: self.path.clone(),
                    embedder: embedder.clone(),
                    embedding_dim: self.embedding_dim,
                    allowed_entity_types: self.allowed_entity_types,
                },
                custom,
            )
            .await?;

            Ok(Memory {
                graph,
                llm: None,
                embedder,
                default_sink: self.default_sink,
                default_namespace: self.default_namespace,
                temporal_graph: Some(temporal_graph),
                episode_content_warn_threshold: self.episode_content_warn_threshold,
                dream_scheduler: std::sync::Arc::new(std::sync::Mutex::new(None)),
                await_extraction: self.await_extraction,
                await_extraction_timeout: self.await_extraction_timeout,
            })
        })
    }
}
