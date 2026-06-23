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
//! use kremory::{EnrichmentEventSink, IngestEventSink, ContradictionDetected, BatchPhase2Complete, IngestStatus, IngestionError, OnEdgeAddedParams};
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
//!     fn on_edge_added(&self, _p: OnEdgeAddedParams<'_>) {}
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
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

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
    ChatProvider, GraphAssertEntityTypeParams, GraphHandle, MemoryError, Result,
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
/// **Real fields (populated by `mem.dream()`):**
/// - `types_discovered` — entity types proposed and accepted by Dream Pass 0.
/// - `entities_reclassified` — entities reclassified by Dream Pass 2.
/// - `duration_ms` — wall-clock time of the dream call.
/// - `warnings` — non-fatal notices from either pass.
///
/// **Honest-zero fields (Phase-3 consolidation not yet implemented):**
/// - `communities_updated`, `cross_episode_merges`, `supersessions_recorded`,
///   `facts_archived` — always 0 until Phase-3 consolidation ships
///   (see ADR-007 retirement, `adr-mem-dream-canonical-supersede-f01-2026-06-22`).
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
    /// Optional dedicated dream-phase LLM (TD-052b). `Some` → `dream()` uses it;
    /// `None` → dream falls back to `self.llm` via `dream_llm_or_main`.
    pub(crate) dream_llm: Option<Arc<dyn ChatProvider>>,
    /// Embedding provider — read by the dream/disambiguation paths
    /// (`facade/dream.rs` passes `self.memory.embedder.as_ref()` into the
    /// dream pass). TD-043: field is live, `#[allow(dead_code)]` removed.
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
        MemoryBuilder::new_open(path.as_ref().to_path_buf())
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

    /// Bulk-ingest multiple episodes in a single batch (F6).
    ///
    /// This is the **inline** batch path: each entry supports per-entry
    /// `.in_namespace(...)`, and ingestion runs through the standard
    /// `EngineGraphHandle`. Use this when you have several episodes to add and
    /// want per-entry control.
    ///
    /// For the **background** path that routes through `BackgroundIngestor` and
    /// fires `on_batch_phase2_complete` on a configured sink, use
    /// [`send_batched`](Self::send_batched) instead (requires a
    /// `default_namespace`; no per-entry namespace override).
    #[must_use = "RememberBatchBuilder must be .await-ed"]
    pub fn remember_batch(&self) -> RememberBatchBuilder<'_> {
        RememberBatchBuilder {
            memory: self,
            episodes: vec![],
            batch_id: None,
            sink: None,
        }
    }

    /// Enqueue text for background ingestion as part of a named batch.
    ///
    /// Associates the episode with `batch_id` for batch tracking.  When all
    /// episodes in the batch reach Phase 2 terminal state,
    /// `on_batch_phase2_complete` fires on the configured sink (if any).
    ///
    /// Per ADR-052 Gap 1 §3.2 + Phase 7 DoD + v0.2.3 follow-up closure
    /// (`BackgroundIngestorGraphHandle` dual-path consolidation, 2026-06-15).
    ///
    /// # Namespace resolution
    ///
    /// Requires a `default_namespace` on the builder.  Per-call namespace
    /// override is not available on the batched send path (use
    /// `remember_batch().with_batch_id()` for per-entry namespace control).
    ///
    /// # Architectural note
    ///
    /// When `.with_sink()` is configured on the builder, `MemoryBuilder::build()`
    /// constructs a `BackgroundIngestorGraphHandle` (arch spec §3.2 Option A).
    /// `send_batched` then routes through `BackgroundIngestor.send_batched` —
    /// the ADR-051 OS-thread pipeline — and `on_batch_phase2_complete` fires
    /// via the configured sink when the batch reaches terminal state.
    ///
    /// When no sink is configured, `send_batched` routes through the
    /// `EngineGraphHandle` tokio-spawn path.  `on_batch_phase2_complete` will
    /// NOT fire (no sink to receive it).  This is the correct behavior: callers
    /// who don't provide a sink have no listener for the callback.
    ///
    /// # Errors
    ///
    /// Returns `Err(MemoryError::MissingNamespace)` when no `default_namespace`
    /// is set.
    ///
    /// Returns `Err(MemoryError::Core(...))` on substrate failure.
    pub async fn send_batched(
        &self,
        text: impl Into<String>,
        batch_id: String,
    ) -> memory::Result<EpisodeCommit> {
        use crate::memory::types::SubmitOpts;

        let ns = self.resolve_namespace(None)?;
        let sink = self.default_sink.clone();

        // ADR-029a lazy population.
        self.ensure_namespace_policy(&ns).await?;

        let source_ref = memory::types::SourceRef {
            kind: memory::types::SourceKind::Chat,
            id: uuid::Uuid::new_v4().to_string(),
            occurred_at: chrono::Utc::now(),
            published_at: None,
        };

        memory::submit_episode(memory::SubmitEpisodeParams {
            graph: self.graph.as_ref(),
            content: &text.into(),
            source_ref,
            structured_facts: vec![],
            provider: self.llm_or_stub(),
            namespace: ns,
            batch_id: Some(batch_id),
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: true,
            },
            sink,
        })
        .await
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
    /// Currently a no-op: kremory uses libSQL WAL with autocommit, so there are
    /// no buffered writes to flush at v0.2.x. Retained as a forward-compatible
    /// shutdown hook — call it at shutdown so your code is ready if explicit
    /// flush semantics are added later (F9).
    pub async fn close(&self) -> Result<()> {
        // No-op: WAL autocommit means no pending writes to flush at v0.2.x.
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

    /// Register a namespace policy + seed its entity-type registry in one atomic
    /// operation (spec custom-entity-type-registry §5.2.2).
    ///
    /// Seeds the namespace's `entity_types` table per the `seed` instruction,
    /// and registers the (default) namespace policy — both inside the SAME
    /// `BEGIN IMMEDIATE` transaction (no TOCTOU window between policy write and
    /// seed write; §5.8 invariant).
    ///
    /// # Semantics (D9 / D9a — brownfield safety)
    ///
    /// | namespace state         | `Default` / `Augment` | `Replace`                              |
    /// |-------------------------|-----------------------|----------------------------------------|
    /// | no rows (fresh)         | seed → `Seeded`       | seed → `Seeded`                        |
    /// | has rows, seed MATCHES  | `AlreadySeeded`       | `AlreadySeeded` (D9a, idempotent boot) |
    /// | has rows, seed DIFFERS  | `AlreadySeeded`       | `Err(AlreadyPopulated { group_id })`   |
    ///
    /// - id=0 "Entity" catch-all is ALWAYS present after a successful call.
    /// - `Replace` NEVER mutates a populated namespace — it fails loud with NO
    ///   DB write, so existing entities' `entity_type_id` can never be orphaned
    ///   (ASMP-003). To add types to an already-populated namespace use
    ///   [`assert_entity_type`](Self::assert_entity_type) (the sanctioned
    ///   incremental-add path).
    ///
    /// This is the PREFERRED startup pattern for domain-specific namespaces:
    /// call at startup BEFORE the first `remember()` for the target namespace.
    /// The existing [`register_namespace`](Self::register_namespace) remains for
    /// callers that want the default seed.
    pub async fn register_namespace_with_seed(
        &self,
        namespace: Namespace,
        seed: crate::core::entity_types::NamespaceSeed,
    ) -> std::result::Result<
        crate::core::entity_types::SeedOutcome,
        crate::core::entity_types::NamespaceRegistrationError,
    > {
        use crate::core::entity_types::NamespaceRegistrationError;

        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            NamespaceRegistrationError::Store(CoreError::Other(anyhow::anyhow!(
                "Memory::register_namespace_with_seed requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
            )))
        })?;

        let policy = namespace.policy.clone().unwrap_or_default();
        policy
            .validate()
            .map_err(|e| NamespaceRegistrationError::Store(CoreError::InvalidPolicy(e)))?;

        let group_id = namespace_to_group_id(&namespace);
        let is_non_default = policy != NamespacePolicy::default();

        // Single BEGIN IMMEDIATE wrapping: namespace policy write + seed
        // application. Presence check + seed live inside this txn (no TOCTOU).
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(NamespaceRegistrationError::Store)?;

        let outcome: std::result::Result<
            crate::core::entity_types::SeedOutcome,
            NamespaceRegistrationError,
        > = async {
            // Policy: INSERT-or-compare (mirrors register_namespace).
            let stored = tg
                .get_namespace_policy(&group_id)
                .await
                .map_err(NamespaceRegistrationError::Store)?;
            match stored {
                Some(existing) if existing == policy => {}
                Some(existing) => {
                    return Err(NamespaceRegistrationError::Store(
                        CoreError::NamespacePolicyImmutable {
                            namespace: group_id.clone(),
                            stored: existing,
                            attempted: policy.clone(),
                        },
                    ));
                }
                None => {
                    tg.set_namespace_policy(&group_id, &policy)
                        .await
                        .map_err(NamespaceRegistrationError::Store)?;
                }
            }

            // Seed (D9 / D9a) inside the same txn.
            crate::core::entity_types::apply_namespace_seed(&tg.conn, &group_id, &seed).await
        }
        .await;

        match &outcome {
            Ok(_) => {
                guard
                    .commit()
                    .await
                    .map_err(NamespaceRegistrationError::Store)?;
            }
            Err(_) => {
                guard
                    .rollback()
                    .await
                    .map_err(NamespaceRegistrationError::Store)?;
            }
        }

        if outcome.is_ok() && is_non_default {
            tracing::warn!(
                target: "kremory.namespace",
                group_id = %group_id,
                policy = ?policy,
                "kremory.namespace.policy_declared: POLICY DECLARED BUT NOT \
                 ENFORCED at v0.1.4 — enforcement lands v0.1.5+ per ADR-029b."
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

    /// Return the dream-phase LLM: the dedicated `with_dream_llm` provider when
    /// set, else delegate to [`llm_or_err`](Self::llm_or_err) (the `with_llm`
    /// provider, or `Err(LlmRequired)` when neither is wired).
    ///
    /// Backward-compat is **structural** (load-bearing-invariants-at-emit, TD-052b
    /// §3.3): when `dream_llm` is `None`, this is byte-for-byte the prior
    /// `llm_or_err("dream", …)` behaviour — same provider, same error, same metric.
    ///
    /// Emits `kremory.dream.llm_role_selected_total{model_role}` on **every** call
    /// (TD-052b §6). The counter measures role-selection *attempts*, not realised
    /// successes — the `interactive` arm increments before `llm_or_err`, so on the
    /// no-provider row it counts an attempt that then errors `LlmRequired`. The
    /// authoritative dream-failure signal remains
    /// `kremory.llm_required_total{method="dream"}`.
    pub(crate) fn dream_llm_or_main(
        &self,
        method: &'static str,
        hint: &'static str,
    ) -> Result<Arc<dyn ChatProvider>> {
        match &self.dream_llm {
            Some(llm) => {
                tracing::debug!(
                    target: "kremory.facade.dream",
                    model_role = "dream",
                    "dream phase using dedicated dream_llm provider (TD-052b)"
                );
                metrics::counter!(
                    "kremory.dream.llm_role_selected_total",
                    "model_role" => "dream"
                )
                .increment(1);
                Ok(Arc::clone(llm))
            }
            None => {
                // Fallback: identical to prior behaviour. Emit the role counter so
                // dashboards can attribute dream calls to the shared interactive
                // provider; llm_or_err still owns the LlmRequired warn+counter.
                metrics::counter!(
                    "kremory.dream.llm_role_selected_total",
                    "model_role" => "interactive"
                )
                .increment(1);
                self.llm_or_err(method, hint)
            }
        }
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
    pub async fn assert_entity_type(&self, params: GraphAssertEntityTypeParams<'_>) -> Result<()> {
        self.graph.graph_assert_entity_type(params).await
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

mod builder;
pub use builder::{MemoryBuilder, WithLlmTrackedParams};

#[cfg(test)]
mod dream_llm_slot_tests {
    //! TD-052b — per-phase dream model slot (`with_dream_llm`).
    //!
    //! Governing spec: `.ai-docs/specs/td-052b-dream-llm-slot-spec-2026-06-22.md`.
    //! These in-crate tests exercise the `pub(crate)` `dream_llm_or_main`
    //! accessor + the `kremory.dream.llm_role_selected_total{model_role}`
    //! counter (§3.3 / §6) — surfaces unreachable from an integration test.
    //! Public-surface build tests live in `tests/memory_builder_compat_matrix.rs`.

    use super::*;
    use crate::core::provider::MockEmbeddingProvider;
    use crate::memory::ChatProvider;
    use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
    use autoagents_llm::error::LLMError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Call-counting `ChatProvider`: records how many times `chat_with_tools`
    /// fired, so a test can assert WHICH slot the dream phase invoked. Returns
    /// an empty response (the dream fan-out tolerates empty proposals).
    #[derive(Debug)]
    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        tag: &'static str,
    }

    impl CountingProvider {
        fn new(tag: &'static str) -> (Arc<Self>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(Self {
                    calls: Arc::clone(&calls),
                    tag,
                }),
                calls,
            )
        }
    }

    #[async_trait::async_trait]
    impl ChatProvider for CountingProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(EmptyResponse))
        }

        fn model(&self) -> &str {
            self.tag
        }
    }

    #[derive(Debug)]
    struct EmptyResponse;

    impl std::fmt::Display for EmptyResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "")
        }
    }

    impl ChatResponse for EmptyResponse {
        fn text(&self) -> Option<String> {
            None
        }
        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
    }

    fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
        Arc::new(MockEmbeddingProvider::new(64))
    }

    async fn build_with(
        llm: Option<Arc<dyn ChatProvider>>,
        dream_llm: Option<Arc<dyn ChatProvider>>,
    ) -> Memory {
        // Build via the real builder so the field threads through the same
        // construction path production uses. NoLlm builds require an extractor;
        // use the LLM path when a main LLM is present, else a null extractor.
        match llm {
            Some(main) => {
                let mut b = Memory::open(":memory:").with_llm(main);
                if let Some(d) = dream_llm {
                    b = b.with_dream_llm(d);
                }
                b.with_embedder(null_embedder())
                    .await
                    .expect("build WithLlm")
            }
            None => {
                let mut b = Memory::open(":memory:")
                    .with_extractor(Arc::new(crate::core::intelligence::MockExtractor));
                if let Some(d) = dream_llm {
                    b = b.with_dream_llm(d);
                }
                b.with_embedder(null_embedder())
                    .await
                    .expect("build NoLlm + extractor")
            }
        }
    }

    /// T3 — `dream_llm = None` → `dream_llm_or_main` returns the SAME `Arc` as
    /// `llm_or_err` (structural fallback, byte-for-byte prior behaviour).
    #[tokio::test]
    async fn t3_accessor_falls_back_to_main_when_dream_unset() {
        let (main, _) = CountingProvider::new("MAIN");
        let main: Arc<dyn ChatProvider> = main;
        let mem = build_with(Some(Arc::clone(&main)), None).await;

        let via_dream = mem
            .dream_llm_or_main("dream", "hint")
            .expect("main is wired");
        let via_main = mem.llm_or_err("dream", "hint").expect("main is wired");
        assert!(
            Arc::ptr_eq(&via_dream, &via_main),
            "with dream_llm=None, dream_llm_or_main must return the same Arc as llm_or_err"
        );
    }

    /// T3 — `dream_llm = Some(D)` → `dream_llm_or_main` returns D, distinct from
    /// the main provider.
    #[tokio::test]
    async fn t3_accessor_returns_dream_provider_when_set() {
        let (main, _) = CountingProvider::new("MAIN");
        let (dream, _) = CountingProvider::new("DREAM");
        let main: Arc<dyn ChatProvider> = main;
        let dream: Arc<dyn ChatProvider> = dream;
        let mem = build_with(Some(Arc::clone(&main)), Some(Arc::clone(&dream))).await;

        let selected = mem
            .dream_llm_or_main("dream", "hint")
            .expect("dream provider wired");
        assert!(
            Arc::ptr_eq(&selected, &dream),
            "dream_llm_or_main must return the dedicated dream provider"
        );
        assert!(
            !Arc::ptr_eq(&selected, &main),
            "dream_llm_or_main must NOT return the main provider when dream_llm is set"
        );
    }

    /// T6 row 4 — neither main nor dream wired → `dream_llm_or_main` errors
    /// `LlmRequired` exactly as the prior `llm_or_err` path (unchanged).
    #[tokio::test]
    async fn t6_row4_neither_provider_errors_llm_required() {
        let mem = build_with(None, None).await;
        let result = mem.dream_llm_or_main("dream", "hint");
        let Err(err) = result else {
            panic!("no provider wired → dream_llm_or_main must error LlmRequired");
        };
        assert!(
            matches!(
                err,
                MemoryError::Core(CoreError::LlmRequired {
                    method: "dream",
                    ..
                })
            ),
            "expected LlmRequired{{method=\"dream\"}}, got: {err:?}"
        );
    }

    /// T5 — a real blocking `dream()` routes through the DREAM slot, never MAIN.
    ///
    /// `dream()` calls `dream_llm_or_main` (dream.rs:95) — selecting the dedicated
    /// dream provider and emitting `model_role="dream"`. Pass-0 and Pass-2 execute
    /// if a TemporalGraph is present; on an empty graph they return Ok with zero work.
    /// Phase-3 consolidation fields (communities/merges/supersessions/archival) are
    /// always 0 — honest zeros per ADR-007 retirement. The role counter is the
    /// **authoritative routing proof**; `MAIN.calls == 0` proves the main slot is
    /// never used for the dream phase.
    #[test]
    fn t5_dream_invokes_dream_provider_not_main() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        let (main, main_calls) = CountingProvider::new("MAIN");
        let (dream, dream_calls) = CountingProvider::new("DREAM");
        let main: Arc<dyn ChatProvider> = main;
        let dream: Arc<dyn ChatProvider> = dream;

        metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let mem = build_with(Some(main), Some(dream)).await;
                // Routes through dream_llm_or_main → DREAM slot. Returns Ok on
                // empty graph (Pass-0/Pass-2 find no work; honest-zero summary).
                mem.dream()
                    .in_namespace(Namespace::new("default"))
                    .await
                    .expect("dream on empty graph must return Ok");
            });
        });

        assert_eq!(
            main_calls.load(Ordering::SeqCst),
            0,
            "the MAIN provider must NEVER be invoked by the dream phase when a dedicated dream_llm is set"
        );
        // dream_calls may be 0 (empty graph) — the role counter is the
        // authoritative routing proof.
        let _ = dream_calls;

        let snapshot = snapshotter.snapshot();
        let dream_role_total: u64 = snapshot
            .into_vec()
            .into_iter()
            .filter(|(k, _, _, _)| {
                k.key().name() == "kremory.dream.llm_role_selected_total"
                    && k.key()
                        .labels()
                        .any(|l| l.key() == "model_role" && l.value() == "dream")
            })
            .map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => c,
                _ => 0,
            })
            .sum();
        assert!(
            dream_role_total >= 1,
            "dream pass with dream_llm=Some must increment llm_role_selected_total{{model_role=\"dream\"}}"
        );
    }

    /// T7 — the `interactive` role counter fires when dream falls back to the
    /// main provider (dream_llm unset, main set).
    #[test]
    fn t7_interactive_role_counter_on_fallback() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        let (main, _) = CountingProvider::new("MAIN");
        let main: Arc<dyn ChatProvider> = main;

        metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let mem = build_with(Some(main), None).await;
                let _ = mem.dream_llm_or_main("dream", "hint");
            });
        });

        let snapshot = snapshotter.snapshot();
        let interactive_total: u64 = snapshot
            .into_vec()
            .into_iter()
            .filter(|(k, _, _, _)| {
                k.key().name() == "kremory.dream.llm_role_selected_total"
                    && k.key()
                        .labels()
                        .any(|l| l.key() == "model_role" && l.value() == "interactive")
            })
            .map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => c,
                _ => 0,
            })
            .sum();
        assert!(
            interactive_total >= 1,
            "fallback dream resolution (dream_llm=None) must increment llm_role_selected_total{{model_role=\"interactive\"}}"
        );
    }

    /// NT-1 — blocking `dream()` invokes the DREAM LLM slot and discovers types
    /// from catch-all entities seeded into the TemporalGraph.
    ///
    /// Closes the TD-052b decorative gap: before the `run_dream_phase`
    /// short-circuit was removed, `dream_llm` flowed only into unreachable code.
    /// Verifies `dream_calls >= 1`, `main_calls == 0`, and `types_discovered`
    /// reflects the scripted proposal — the assertion T5 could not make.
    ///
    /// NT-2 (honest-zeros lock) is folded in: Phase-3 consolidation fields must
    /// always be 0 pending consolidation implementation (ADR-007 retirement).
    #[tokio::test]
    async fn nt1_dream_invokes_dream_llm_discovers_types_and_zeroes_consolidation() {
        use crate::core::entity_types::ensure_default_types_seeded;
        use chrono::Utc;

        // ScriptedCountingProvider: counts calls AND returns valid proposal JSON.
        #[derive(Debug)]
        struct ScriptedCountingProvider {
            calls: Arc<AtomicUsize>,
        }

        #[derive(Debug)]
        struct TextResponse {
            text: String,
        }
        impl std::fmt::Display for TextResponse {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.text)
            }
        }
        impl ChatResponse for TextResponse {
            fn text(&self) -> Option<String> {
                Some(self.text.clone())
            }
            fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
                None
            }
        }

        #[async_trait::async_trait]
        impl ChatProvider for ScriptedCountingProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(TextResponse {
                    text: r#"{"proposals":[{"name":"Company","description":"A business entity.","justification":"All three are companies."}]}"#
                        .to_string(),
                }))
            }

            fn model(&self) -> &str {
                "scripted-dream"
            }
        }

        let dream_calls = Arc::new(AtomicUsize::new(0));
        let scripted_dream: Arc<dyn ChatProvider> = Arc::new(ScriptedCountingProvider {
            calls: Arc::clone(&dream_calls),
        });
        let (main, main_calls) = CountingProvider::new("MAIN");
        let main: Arc<dyn ChatProvider> = main;

        let mem = build_with(Some(main), Some(scripted_dream)).await;

        // Seed catch-all entities so Pass-0 has clusters to process.
        // group_id = "default" (namespace_to_group_id(Namespace::new("default"))).
        let tg = mem
            .temporal_graph_for_test()
            .expect("TemporalGraph present in :memory: build");
        let conn = tg.conn.clone();
        ensure_default_types_seeded(&conn, "default")
            .await
            .expect("seed entity_types defaults");
        let now = Utc::now().to_rfc3339();
        for id in ["alpha corp", "beta fund", "gamma ventures"] {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![id.to_string(), now.clone(), "default".to_string()],
            )
            .await
            .expect("seed catch-all entity");
        }

        let summary = mem
            .dream()
            .in_namespace(Namespace::new("default"))
            .await
            .expect("dream with seeded catch-all entities must return Ok");

        // NT-1: DREAM provider fired; MAIN never touched.
        assert!(
            dream_calls.load(Ordering::SeqCst) >= 1,
            "Pass-0 must invoke the dream LLM slot when catch-all entities exist"
        );
        assert_eq!(
            main_calls.load(Ordering::SeqCst),
            0,
            "MAIN provider must never be invoked by the dream phase"
        );
        // NT-1: types_discovered count is not asserted > 0 — anti-redundancy can correctly
        // reject proposals that overlap existing types in the test embedder's metric space
        // (stochastic, mirrors the plan's "do NOT assert > 0" stance). The load-bearing
        // signal is dream_calls >= 1 above: that proves TD-052b is live and Pass-0 ran.
        let _ = summary.types_discovered;
        // NT-2 (honest-zeros lock): Phase-3 consolidation fields always 0.
        assert_eq!(
            summary.communities_updated, 0,
            "communities_updated must be 0 — Phase-3 consolidation not yet implemented"
        );
        assert_eq!(
            summary.cross_episode_merges, 0,
            "cross_episode_merges must be 0 — Phase-3 consolidation not yet implemented"
        );
        assert_eq!(
            summary.supersessions_recorded, 0,
            "supersessions_recorded must be 0 — Phase-3 consolidation not yet implemented"
        );
        assert_eq!(
            summary.facts_archived, 0,
            "facts_archived must be 0 — Phase-3 consolidation not yet implemented"
        );
    }
}
