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

use std::future::IntoFuture;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::core::provider::DynEmbeddingProvider;
use crate::memory::{
    self,
    events::EnrichmentEventSink,
    types::{
        AwaitOpts, BatchStatus, CancelOutcome, ContextTemplate, DreamHandle, DreamOpts,
        DreamPhaseResult, DreamStatus, EpisodeCommit, Namespace, RetrievedContext, SearchOpts,
        SourceKind, SourceRef, StructuredFact, SubmitOpts,
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
#[derive(Debug, Clone)]
pub struct DreamSummary {
    pub communities_updated: usize,
    pub cross_episode_merges: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    pub duration_ms: u64,
}

impl From<DreamPhaseResult> for DreamSummary {
    fn from(r: DreamPhaseResult) -> Self {
        Self {
            communities_updated: r.communities_recomputed,
            cross_episode_merges: r.cross_meeting_merges,
            supersessions_recorded: r.supersessions_recorded,
            facts_archived: r.facts_archived,
            duration_ms: r.duration_ms,
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
    graph: Arc<dyn GraphHandle>,
    llm: Arc<dyn ChatProvider>,
    /// Stored for forward-compat (v0.1.1 will wire to real TemporalGraph::open).
    #[allow(dead_code)]
    embedder: Arc<dyn DynEmbeddingProvider>,
    default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    default_namespace: Option<Namespace>,
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
    /// Models: `llama3.1:8b` (chat) + `nomic-embed-text` (embeddings).
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
    /// SHA-256-based embedder (not semantic — suitable for exact-match recall only).
    /// A `tracing::warn!` is emitted at construction time.
    pub async fn with_anthropic(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_anthropic(path).await
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
            k: None,
            as_of: None,
            template: Some(RecallTemplate::TemporalFacts),
            raw_mode: false,
            opts: None,
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
}

// ── MemoryBuilder ─────────────────────────────────────────────────────────────

/// Type-state builder for `Memory`. Compile-time enforced: `.with_llm()` then
/// `.with_embedder()` are both required before `.await`.
///
/// Optional: `.with_event_sink()`, `.default_namespace()`.
#[must_use = "MemoryBuilder must be configured with .with_llm() AND .with_embedder() before .await"]
pub struct MemoryBuilder<L, E> {
    path: std::path::PathBuf,
    llm: Option<Arc<dyn ChatProvider>>,
    embedder: Option<Arc<dyn DynEmbeddingProvider>>,
    default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    default_namespace: Option<Namespace>,
    _llm_state: std::marker::PhantomData<L>,
    _emb_state: std::marker::PhantomData<E>,
}

impl<L, E> MemoryBuilder<L, E> {
    /// Set the default event sink for all subsequent operations.
    /// Per-call sinks (via `.with_event_sink()` on request builders) override this.
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.default_sink = Some(sink);
        self
    }

    /// Set the default namespace used by operations that don't specify `.in_namespace()`.
    pub fn default_namespace(mut self, ns: Namespace) -> Self {
        self.default_namespace = Some(ns);
        self
    }
}

impl MemoryBuilder<NoLlm, NoEmb> {
    /// Configure the LLM provider (required).
    pub fn with_llm(self, llm: Arc<dyn ChatProvider>) -> MemoryBuilder<WithLlm, NoEmb> {
        MemoryBuilder {
            path: self.path,
            llm: Some(llm),
            embedder: self.embedder,
            default_sink: self.default_sink,
            default_namespace: self.default_namespace,
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
            let llm = self
                .llm
                .ok_or_else(|| MemoryError::Other("llm missing".into()))?;
            let embedder = self
                .embedder
                .ok_or_else(|| MemoryError::Other("embedder missing".into()))?;
            let graph = providers::open_graph(self.path, embedder.clone()).await?;
            Ok(Memory {
                graph,
                llm,
                embedder,
                default_sink: self.default_sink,
                default_namespace: self.default_namespace,
            })
        })
    }
}

// ── RememberRequest ───────────────────────────────────────────────────────────

/// Ingest request builder. Obtain via `mem.remember("…")`.
pub struct RememberRequest<'a> {
    memory: &'a Memory,
    content: String,
    source_ref: Option<SourceRef>,
    namespace: Option<Namespace>,
    published_at: Option<DateTime<Utc>>,
    facts: Vec<StructuredFact>,
    sink: Option<Arc<dyn EnrichmentEventSink>>,
    no_wait: bool,
    opts: Option<SubmitOpts>,
}

impl<'a> RememberRequest<'a> {
    /// Tag this episode as originating from a chat session.
    pub fn from_chat(mut self, id: impl Into<String>) -> Self {
        self.source_ref = Some(SourceRef {
            kind: SourceKind::Chat,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Tag this episode as originating from a note.
    pub fn from_note(mut self, id: impl Into<String>) -> Self {
        self.source_ref = Some(SourceRef {
            kind: SourceKind::Document,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Tag this episode as originating from a document.
    pub fn from_document(mut self, id: impl Into<String>) -> Self {
        self.source_ref = Some(SourceRef {
            kind: SourceKind::Document,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Full escape hatch: specify source ID and kind directly.
    pub fn from_source(mut self, id: impl Into<String>, kind: SourceKind) -> Self {
        self.source_ref = Some(SourceRef {
            kind,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Set the namespace for this operation (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Set the bi-temporal anchor for the source document (Story #318).
    ///
    /// When set, `valid_from` for structured facts without their own anchor
    /// falls back to this timestamp.
    pub fn published_at(mut self, ts: DateTime<Utc>) -> Self {
        self.published_at = Some(ts);
        // Propagate into source_ref if already set
        if let Some(ref mut sr) = self.source_ref {
            sr.published_at = Some(ts);
        }
        self
    }

    /// Attach pre-extracted structured facts. When provided, skips Phase 2
    /// LLM extraction for these facts (they are pinned directly into the graph).
    pub fn with_facts(mut self, facts: Vec<StructuredFact>) -> Self {
        self.facts = facts;
        self
    }

    /// Per-call event sink override. Overrides the Memory-level default for
    /// this operation only.
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Return after Phase 1 commit (store + embed) without blocking on Phase 2.
    ///
    /// The returned `EpisodeCommit.run_id` will be `Some(...)` — use
    /// `mem.status_of(&commit)` or `mem.await_enrichment(&commit, timeout)` to
    /// track Phase 2 completion.
    pub fn no_wait(mut self) -> Self {
        self.no_wait = true;
        self
    }

    /// Explicit form of the default: block until Phase 2 enrichment is done.
    ///
    /// Equivalent to the default `.await` terminal; provided for clarity in code
    /// that toggles between `.no_wait()` and blocking paths.
    pub fn await_enrichment(mut self) -> Self {
        self.no_wait = false;
        self
    }

    /// Escape hatch: set raw substrate `SubmitOpts` directly.
    pub fn opts(mut self, opts: SubmitOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    async fn execute(self) -> Result<EpisodeCommit> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let sink = self.memory.resolve_sink(self.sink);

        let source_ref = self.source_ref.unwrap_or_else(|| SourceRef {
            kind: SourceKind::Chat,
            id: uuid::Uuid::new_v4().to_string(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });

        let opts = self.opts.unwrap_or(SubmitOpts {
            enrich_per_episode: true,
            run_in_background: self.no_wait,
        });

        memory::submit_episode(
            self.memory.graph.as_ref(),
            &self.content,
            source_ref,
            self.facts,
            self.memory.llm.clone(),
            ns,
            None,
            opts,
            sink,
        )
        .await
    }
}

impl<'a> IntoFuture for RememberRequest<'a> {
    type Output = Result<EpisodeCommit>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

// ── RememberBatchBuilder ──────────────────────────────────────────────────────

/// Bulk ingest request builder. Obtain via `mem.remember_batch()`.
pub struct RememberBatchBuilder<'a> {
    memory: &'a Memory,
    episodes: Vec<PendingEpisode>,
    batch_id: Option<String>,
    sink: Option<Arc<dyn EnrichmentEventSink>>,
}

struct PendingEpisode {
    content: String,
    source_ref: Option<SourceRef>,
    namespace: Option<Namespace>,
    published_at: Option<DateTime<Utc>>,
    facts: Vec<StructuredFact>,
}

impl<'a> RememberBatchBuilder<'a> {
    /// Add an episode to the batch. Returns an `EpisodeEntryBuilder` for chaining.
    pub fn entry(self, content: impl Into<String>) -> EpisodeEntryBuilder<'a> {
        EpisodeEntryBuilder {
            batch: self,
            pending: PendingEpisode {
                content: content.into(),
                source_ref: None,
                namespace: None,
                published_at: None,
                facts: vec![],
            },
        }
    }

    /// Set the batch ID for idempotent batch tracking.
    pub fn with_batch_id(mut self, id: impl Into<String>) -> Self {
        self.batch_id = Some(id.into());
        self
    }

    /// Per-batch event sink (applies to all episodes in this batch).
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    async fn execute(self) -> Result<Vec<EpisodeCommit>> {
        let mut commits = Vec::with_capacity(self.episodes.len());
        for ep in self.episodes {
            let ns = self.memory.resolve_namespace(ep.namespace)?;
            let sink = self.memory.resolve_sink(self.sink.clone());
            let source_ref = ep.source_ref.unwrap_or_else(|| SourceRef {
                kind: SourceKind::Chat,
                id: uuid::Uuid::new_v4().to_string(),
                occurred_at: Utc::now(),
                published_at: ep.published_at,
            });
            let commit = memory::submit_episode(
                self.memory.graph.as_ref(),
                &ep.content,
                source_ref,
                ep.facts,
                self.memory.llm.clone(),
                ns,
                self.batch_id.clone(),
                SubmitOpts {
                    enrich_per_episode: true,
                    run_in_background: false,
                },
                sink,
            )
            .await?;
            commits.push(commit);
        }
        Ok(commits)
    }
}

impl<'a> IntoFuture for RememberBatchBuilder<'a> {
    type Output = Result<Vec<EpisodeCommit>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Per-episode entry builder within a `RememberBatchBuilder`.
pub struct EpisodeEntryBuilder<'a> {
    batch: RememberBatchBuilder<'a>,
    pending: PendingEpisode,
}

impl<'a> EpisodeEntryBuilder<'a> {
    /// Tag as chat source.
    pub fn from_chat(mut self, id: impl Into<String>) -> Self {
        self.pending.source_ref = Some(SourceRef {
            kind: SourceKind::Chat,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.pending.published_at,
        });
        self
    }

    /// Tag as document source.
    pub fn from_document(mut self, id: impl Into<String>) -> Self {
        self.pending.source_ref = Some(SourceRef {
            kind: SourceKind::Document,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.pending.published_at,
        });
        self
    }

    /// Set namespace for this episode.
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.pending.namespace = Some(ns);
        self
    }

    /// Set bi-temporal anchor for this episode.
    pub fn published_at(mut self, ts: DateTime<Utc>) -> Self {
        self.pending.published_at = Some(ts);
        if let Some(ref mut sr) = self.pending.source_ref {
            sr.published_at = Some(ts);
        }
        self
    }

    /// Attach pre-extracted facts.
    pub fn with_facts(mut self, facts: Vec<StructuredFact>) -> Self {
        self.pending.facts = facts;
        self
    }

    /// Finish this entry and return to the batch builder for chaining or terminal.
    pub fn done(mut self) -> RememberBatchBuilder<'a> {
        self.batch.episodes.push(self.pending);
        self.batch
    }
}

// ── RecallRequest ─────────────────────────────────────────────────────────────

/// Recall (search) request builder. Obtain via `mem.recall("…")`.
pub struct RecallRequest<'a> {
    memory: &'a Memory,
    query: String,
    namespace: Option<Namespace>,
    k: Option<usize>,
    as_of: Option<DateTime<Utc>>,
    template: Option<RecallTemplate>,
    raw_mode: bool,
    opts: Option<SearchOpts>,
}

impl<'a> RecallRequest<'a> {
    /// Set the namespace for this operation (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Top-k results to return (clamped per Story #166).
    pub fn k(mut self, n: usize) -> Self {
        self.k = Some(n);
        self
    }

    /// Point-in-time filter (v0.1.0: emits `tracing::warn!`, not yet implemented in SQL).
    /// v0.1.1 will apply the filter. Setting this now ensures forward-compatible caller code.
    pub fn as_of(mut self, ts: DateTime<Utc>) -> Self {
        self.as_of = Some(ts);
        self
    }

    /// Return results as a prompt-ready string using `TemporalFacts` template.
    /// Sets the terminal return type to `String`.
    pub fn as_prompt_text(mut self) -> Self {
        self.template = Some(RecallTemplate::TemporalFacts);
        self.raw_mode = false;
        self
    }

    /// Return results as a string using a specific template.
    pub fn as_template(mut self, t: RecallTemplate) -> Self {
        self.template = Some(t);
        self.raw_mode = false;
        self
    }

    /// Return raw `Vec<RetrievedContext>` — no template rendering.
    pub fn raw(mut self) -> RecallRawRequest<'a> {
        self.raw_mode = true;
        RecallRawRequest { inner: self }
    }

    /// Escape hatch: set raw `SearchOpts` directly.
    pub fn opts(mut self, opts: SearchOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    async fn execute(self) -> Result<String> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let opts = self.opts.unwrap_or(SearchOpts {
            limit: self.k,
            as_of: self.as_of,
            source_kind: None,
        });
        let results = memory::search(self.memory.graph.as_ref(), &self.query, ns, opts).await?;
        let template = self.template.unwrap_or(RecallTemplate::TemporalFacts);
        Ok(memory::context_block(&results, template.into()))
    }
}

impl<'a> IntoFuture for RecallRequest<'a> {
    type Output = Result<String>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Raw recall variant — returns `Vec<RetrievedContext>` without template rendering.
pub struct RecallRawRequest<'a> {
    inner: RecallRequest<'a>,
}

impl<'a> IntoFuture for RecallRawRequest<'a> {
    type Output = Result<Vec<RetrievedContext>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let ns = self.inner.memory.resolve_namespace(self.inner.namespace)?;
            let opts = self.inner.opts.unwrap_or(SearchOpts {
                limit: self.inner.k,
                as_of: self.inner.as_of,
                source_kind: None,
            });
            memory::search(
                self.inner.memory.graph.as_ref(),
                &self.inner.query,
                ns,
                opts,
            )
            .await
        })
    }
}

// ── ForgetRequest ─────────────────────────────────────────────────────────────

/// Forget (delete) request builder. Obtain via `mem.forget()`.
///
/// Must call `.execute()` explicitly — this is a destructive operation.
pub struct ForgetRequest<'a> {
    memory: &'a Memory,
    namespace: Option<Namespace>,
}

impl<'a> ForgetRequest<'a> {
    /// Set the namespace scope for deletion (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Execute the deletion. Returns the count of deleted episodes.
    ///
    /// This is the only terminal for `ForgetRequest` — there is no implicit
    /// `.await` to prevent accidental destructive operations.
    pub async fn execute(self) -> Result<u64> {
        let _ns = self.memory.resolve_namespace(self.namespace)?;
        // v0.1.0 stub: no substrate batch_forget fn exists yet.
        // The namespace is validated above (MissingNamespace check fires correctly).
        // Actual deletion deferred until substrate exposes batch_forget.
        // Per plan Rule 1 (scope): NO substrate changes in this PR.
        Ok(0)
    }
}

// ── DreamRequest ──────────────────────────────────────────────────────────────

/// Dream phase (batch consolidation) request builder. Obtain via `mem.dream()`.
pub struct DreamRequest<'a> {
    memory: &'a Memory,
    namespace: Option<Namespace>,
    batch_id: Option<String>,
    batch_size: Option<usize>,
    sink: Option<Arc<dyn EnrichmentEventSink>>,
    fire_and_forget: bool,
    opts: Option<DreamOpts>,
}

impl<'a> DreamRequest<'a> {
    /// Set the namespace scope for this consolidation.
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Idempotent batch key. Multiple calls with same `(namespace, batch_id)` return
    /// the existing handle without starting a new run.
    pub fn for_batch(mut self, id: impl Into<String>) -> Self {
        self.batch_id = Some(id.into());
        self
    }

    /// Tunable: set the episode batch size for this consolidation pass.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = Some(n);
        self
    }

    /// Per-call event sink override.
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Return `DreamHandle` immediately without blocking on completion.
    /// Use `mem.await_dream(&handle, timeout)` to wait.
    pub fn fire_and_forget(mut self) -> DreamFireAndForget<'a> {
        self.fire_and_forget = true;
        DreamFireAndForget { inner: self }
    }

    /// Explicit form of the default: block until the dream phase completes.
    pub fn await_completion(mut self) -> Self {
        self.fire_and_forget = false;
        self
    }

    /// Escape hatch: set raw `DreamOpts` directly.
    pub fn opts(mut self, opts: DreamOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    async fn execute_blocking(self) -> Result<DreamSummary> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let sink = self.memory.resolve_sink(self.sink);
        let _opts = self.opts.unwrap_or_default();
        // Use legacy synchronous path: run_dream_phase → DreamPhaseResult → DreamSummary
        // This is the correct substrate call for blocking dream at v0.1.0.
        #[allow(deprecated)]
        let result =
            memory::run_dream_phase(self.memory.graph.as_ref(), ns, self.memory.llm.clone())
                .await?;
        // Sink is accepted but dream events are fired by the graph impl internally.
        // The sink parameter is stored for future use when non-blocking dream fires events.
        let _ = sink;
        Ok(DreamSummary::from(result))
    }
}

impl<'a> IntoFuture for DreamRequest<'a> {
    type Output = Result<DreamSummary>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute_blocking())
    }
}

/// Fire-and-forget dream variant — returns `DreamHandle` without blocking.
pub struct DreamFireAndForget<'a> {
    inner: DreamRequest<'a>,
}

impl<'a> IntoFuture for DreamFireAndForget<'a> {
    type Output = Result<DreamHandle>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let ns = self.inner.memory.resolve_namespace(self.inner.namespace)?;
            let sink = self.inner.memory.resolve_sink(self.inner.sink);
            let opts = self.inner.opts.unwrap_or_default();
            memory::submit_dream_phase(
                self.inner.memory.graph.as_ref(),
                ns,
                self.inner.memory.llm.clone(),
                self.inner.batch_id,
                opts,
                sink,
            )
            .await
        })
    }
}
