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
//! use kremory::{Memory, Namespace};
//! # async fn ex() -> kremory::memory::Result<()> {
//! let mem = Memory::with_ollama("./agent.db").await?;
//! // `remember(...)` requires a namespace: call `.in_namespace(ns)` on the
//! // request (or set a `default_namespace` on the builder — see Tier 2 below).
//! mem.remember("User prefers concise replies")
//!     .in_namespace(Namespace::new("agent"))
//!     .await?;
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
pub mod reverse;
pub mod supersede;
pub mod update;

mod embeddings;
pub use embeddings::*;

mod namespaces;

#[cfg(test)]
mod dream_llm_slot_tests;

mod summary;
pub use summary::*;

pub use dream::*;
pub use forget::*;
pub use recall::*;
pub use remember::*;
pub use reverse::*;
pub use supersede::*;
pub use update::*;

// Reversible-graph-mutations honest outcome types (arch-spec §3.1) — re-exported
// from the (`pub(crate)`) provenance module so `Memory::unmerge` /
// `restore_archived_fact` / `unsupersede` return a nameable public type.
pub use crate::core::dream::provenance::{
    DeleteEntityOutcome, DeleteFactOutcome, EditEntityOutcome, RestoreArchivedOutcome,
    UnmergeOutcome, UnsupersedeOutcome,
};

// Reversible-graph-mutations consumer INSPECT surface (arch-spec §3 "Inspect
// surface") — the SEE half of the see+fix story. `MutationRecord` is the
// consumer-facing view a `mutation_history` / `list_mutations` query returns;
// `MutationKind` tags it; `MutationFilter` shapes `list_mutations`.
pub use crate::core::dream::provenance::{MutationFilter, MutationKind, MutationRecord};

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
        AwaitOpts, BatchStatus, CancelOutcome, ContextTemplate, CrossEpisodeMode, DreamHandle,
        DreamOpts, DreamPhaseResult, DreamStatus, EpisodeCommit, Namespace, NamespacePolicy,
        RetrievedContext, SearchOpts, SourceKind, SourceRef, StructuredFact, SubmitOpts,
    },
    ChatProvider, GraphAssertEntityTypeParams, GraphHandle, MemoryError, Result,
};

// ── Type-state markers ────────────────────────────────────────────────────────

/// Type-state marker: LLM not yet configured.
pub struct NoLlm;
/// Type-state marker: LLM configured.
pub struct WithLlm;
/// Type-state marker: Embedder not yet configured.
pub struct NoEmbedder;
/// Type-state marker: Embedder configured.
pub struct WithEmbedder;

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
///
#[derive(Clone)]
pub struct Memory {
    pub(crate) graph: Arc<dyn GraphHandle>,
    /// `None` when built via the NoLlm path (`Memory::open().with_extractor(…).with_embedder(…)`).
    /// Category B methods (dream, recall_with_disambiguation, detect_contradictions) call
    /// `.llm_or_err("method_name")` which returns `Error::LlmRequired` at call time.
    pub(crate) llm: Option<Arc<dyn ChatProvider>>,
    /// Optional dedicated dream-phase LLM. `Some` → `dream()` uses it;
    /// `None` → dream falls back to `self.llm` via `dream_llm_or_main`.
    pub(crate) dream_llm: Option<Arc<dyn ChatProvider>>,
    /// Concrete model id for the MAIN chat provider (`with_model_id` / Tier-1
    /// shortcut). Threaded into the dream LLM passes for capability detection
    /// (empty → `PromptOnly` degrade). `None` when the provider was wired via
    /// raw `with_llm` without a model id — dream then degrades exactly as the
    /// interactive path does. Previously baked only into the ingest
    /// pipeline, never reaching the dream facade — the root cause of the
    /// silent empty-model → zero-output degrade in the LLM dream passes.
    pub(crate) model_id: Option<String>,
    /// Optional dedicated dream-phase model id (`with_dream_model_id`). Pairs
    /// with `dream_llm` the way `model_id` pairs with `llm`: when a dedicated
    /// dream *provider* is set, its model *string* usually differs from the
    /// interactive model, so capability detection needs its own id. `None` →
    /// dream falls back to `model_id` via `dream_model_id_or_main`.
    pub(crate) dream_model_id: Option<String>,
    /// Embedding provider — read by the dream/disambiguation paths
    /// (`facade/dream.rs` passes `self.memory.embedder.as_ref()` into the
    /// dream pass). Field is live, `#[allow(dead_code)]` removed.
    pub(crate) embedder: Arc<dyn DynEmbeddingProvider>,
    pub(crate) default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    pub(crate) default_namespace: Option<Namespace>,
    /// Direct handle to the underlying `TemporalGraph` for namespace-policy
    /// substrate calls (`register_namespace` + lazy population).
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
    /// extraction (GLiNER/LLM via the background worker) has transitioned the
    /// episode to `Verified` (or returns `Err` on `Failed` / timeout).
    ///
    /// Opt-in: default is `false` (fire-and-forget by design).
    /// Per D1 peer pattern: equivalent to Cognee's `run_in_background=False`.
    ///
    /// ⚠ Cost: enables synchronous-extraction ergonomics at the expense of the
    /// latency benefit the background path provides. Document this trade-off in consumer
    /// code. Prefer `Memory::wait_for_processing` directly for fine-grained
    /// control.
    pub(crate) await_extraction: bool,
    /// Timeout applied when `await_extraction = true`.
    ///
    /// Default: 60 seconds. Configurable via
    /// `MemoryBuilder::with_await_extraction_timeout`.
    pub(crate) await_extraction_timeout: Duration,
}

// Hand-written `Debug` (C-DEBUG): `Memory` holds `Arc<dyn ChatProvider>` /
// `Arc<dyn GraphHandle>` / `Arc<dyn DynEmbeddingProvider>` trait objects that
// don't themselves implement `Debug`, so a `#[derive(Debug)]` wouldn't
// compile. Print what CAN be shown (presence of a provider, the resolved
// model ids, the default namespace, the tunables) and elide the rest via
// `finish_non_exhaustive()` — the same shape `tokio::runtime::Runtime` and
// `reqwest::Client` use for the same reason. Without this, `println!("{:?}",
// mem)` / `dbg!(mem)` — the first thing most Rust developers reach for —
// does not compile for the type consumers interact with most.
impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("llm_configured", &self.llm.is_some())
            .field("dream_llm_configured", &self.dream_llm.is_some())
            .field("model_id", &self.model_id)
            .field("dream_model_id", &self.dream_model_id)
            .field("event_sink_configured", &self.default_sink.is_some())
            .field("default_namespace", &self.default_namespace)
            .field(
                "episode_content_warn_threshold",
                &self.episode_content_warn_threshold,
            )
            .field("await_extraction", &self.await_extraction)
            .field("await_extraction_timeout", &self.await_extraction_timeout)
            .finish_non_exhaustive()
    }
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
    pub fn open(path: impl AsRef<Path>) -> MemoryBuilder<NoLlm, NoEmbedder> {
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
    /// Models: `gemma4:e4b` (chat, reasoning disabled) + `nomic-embed-text` (embeddings).
    /// See [`providers::with_ollama`] for the benchmark rationale + lighter alternatives.
    pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_ollama(path).await
    }

    /// Open with Ollama at a custom URL.
    pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Self> {
        providers::with_ollama_at(url, path).await
    }

    /// Open with Ollama at a custom URL and an optional custom chat model.
    ///
    /// When `model` is `None`, the default `gemma4:e4b` (reasoning disabled) is
    /// used. Pass e.g. `Some("qwen2.5:7b".into())` for a lighter footprint.
    /// This is the inherent, discoverable counterpart to
    /// [`providers::with_ollama_at_model`] — previously that free function was
    /// the only way to reach this path, inconsistent with every other Tier-1
    /// shortcut being an inherent `Memory::` associated function. Both remain
    /// callable; this one just shows up in `Memory::<tab>` completion.
    pub async fn with_ollama_at_model(
        url: impl Into<String>,
        model: Option<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        providers::with_ollama_at_model(url, model, path).await
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

    // ── Introspection ───────────────────────────────────────────────────────

    /// The live search-fusion configuration this `Memory` uses for recall — the
    /// [`SearchConfig`](crate::core::config::SearchConfig) carried by the
    /// underlying graph handle, reflecting any `KREMORY_CONTENT_WEIGHT` /
    /// `KREMORY_RRF_K` boot overrides applied at construction (via
    /// `providers::search_env_overrides`). Read-only; cheap (a clone of an
    /// in-memory struct — no I/O, hence not `async`).
    ///
    /// Exposed so a transport/consumer — e.g. the `kremory-http` bench
    /// server's `GET /health` endpoint — can report the ACTUAL active scoring
    /// config as a single source of truth, rather than re-reading env
    /// independently (which can silently drift from what the search path
    /// actually uses, reporting a wrong config to whoever asks). Stub/test
    /// graph handles that carry no `Engine` return `SearchConfig::default()`.
    pub fn search_config(&self) -> crate::core::config::SearchConfig {
        self.graph.search_config()
    }

    /// Whether ingest-time contradiction detection is live on this `Memory` —
    /// reflecting the compiled-in default,
    /// any `KREMORY_CONTRADICTION_DETECTION` boot override, and any
    /// [`MemoryBuilder::with_contradiction_detection_enabled`](crate::facade::builder::MemoryBuilder::with_contradiction_detection_enabled)
    /// call, resolved by that same precedence. Read-only; cheap (a `bool` copy
    /// — no I/O, hence not `async`).
    ///
    /// Exposed for the same reason as its sibling [`Memory::search_config`]:
    /// a consumer or transport must be able to report the config the
    /// pipeline ACTUALLY runs, rather than re-reading env — and since the
    /// builder seam can now override env, env is no longer a reliable proxy
    /// for this flag at all. It matters more than the sibling because it gates
    /// the pipeline's only DESTRUCTIVE default-ON path: when `true`, an
    /// ingested fact judged to contradict a stored one supersedes it.
    pub fn contradiction_detection_enabled(&self) -> bool {
        self.graph.contradiction_detection_enabled()
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

    /// Resolve a `Namespace` to the internal group-id string, so integration
    /// tests can plant graph rows under the exact group `mem.dream()` operates
    /// on for that namespace.
    ///
    /// Only available under `test` or `test-utils`. Not part of the stable
    /// public API — external callers MUST NOT depend on the group-id string
    /// layout (it is substrate detail; see `namespace_to_group_id`).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn group_id_for_test(&self, ns: &crate::Namespace) -> String {
        crate::memory::engine_handle::namespace_to_group_id(ns)
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
    /// the OS-thread pipeline — and `on_batch_phase2_complete` fires
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

        // Lazy population.
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
            rerank_k: None,
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

    /// Bound a fact's world-time `valid_to` window explicitly — the
    /// consumer-facing, consumer-EXPLICIT half of the
    /// supersession gap (auto-detected supersession is deferred, TD-P1-AUTO).
    ///
    /// Requires `.at(valid_to)` before `.execute()`. The dream supersession
    /// sweep (`include_supersession_sweep: true`) later observes the bounded
    /// `valid_to` and closes the window (`expired_at = valid_to`,
    /// `DreamSummary.supersessions_recorded` increments) — see
    /// [`supersede::SupersedeRequest`] for the full two-phase mechanism.
    #[must_use = "SupersedeRequest must call .execute() to run"]
    pub fn supersede(&self, fact_id: i64) -> SupersedeRequest<'_> {
        SupersedeRequest {
            memory: self,
            fact_id,
            namespace: None,
            valid_to: None,
            reason: None,
            close_now: false,
        }
    }

    /// Reverse ANY logged, reversible mutation by its `mutation_id` — the unified
    /// undo umbrella, and the one recommended entry point.
    ///
    /// Reads the `graph_mutation_log` row for `mutation_id`, matches on its kind,
    /// and dispatches to the correct per-kind undo, returning the honest
    /// [`UndoOutcome`]. This is the method to reach for after iterating
    /// [`list_mutations`](Self::list_mutations) / [`mutation_history`](Self::mutation_history):
    /// a consumer can uniformly `mem.undo(record.mutation_id)` without switching on
    /// the kind by hand. The per-kind methods ([`unmerge`](Self::unmerge),
    /// [`undo_entity_edit`](Self::undo_entity_edit),
    /// [`undo_delete_entity`](Self::undo_delete_entity),
    /// [`undo_delete_fact`](Self::undo_delete_fact)) still work and remain the
    /// escape hatch when you already know the kind.
    ///
    /// Five LOGGED kinds dispatch (`entity_merge` / `entity_edit` /
    /// `entity_delete` / `fact_delete` / `fact_archive`). The other three
    /// [`MutationKind`] variants are RESERVED (never produced into the log today),
    /// so a would-be row of that kind returns a loud `Error::UndoUnsupportedKind`
    /// — see [`UndoRequest`].
    ///
    /// Optional `.in_namespace(ns)` guards the undo to the mutation's original
    /// namespace. Must call `.execute()` (mutating op).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::{Memory, Namespace};
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// // SEE what dream() did, then UNDO the most recent mutation uniformly.
    /// let history = mem.mutation_history("alice")
    ///     .in_namespace(Namespace::new("agent"))
    ///     .await?;
    /// if let Some(rec) = history.first() {
    ///     let outcome = mem.undo(rec.mutation_id).execute().await?;
    ///     println!("reversed: {outcome:?}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "UndoRequest must call .execute() to run"]
    pub fn undo(&self, mutation_id: i64) -> UndoRequest<'_> {
        UndoRequest {
            memory: self,
            mutation_id,
            namespace: None,
        }
    }

    /// Reverse a prior entity-merge, fully restoring the loser entity, its facts,
    /// its episodic edges, and the keeper's overwritten `access_count` /
    /// `ner_confidence` (reversible-graph-mutations arch-spec §4.2). Records a
    /// merge NOGOOD so the next `dream()` will NOT re-merge the split pair (§6.2).
    ///
    /// Idempotent: a second call returns `already_undone = true`. Must call
    /// `.execute()` (mutating op). Prefer [`undo`](Self::undo) when iterating
    /// [`list_mutations`](Self::list_mutations) — it dispatches by kind for you.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::{Memory, Namespace};
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// // Find a merge in an entity's history, then split the pair back apart.
    /// let history = mem.mutation_history("alice j")
    ///     .in_namespace(Namespace::new("agent"))
    ///     .await?;
    /// if let Some(rec) = history.first() {
    ///     let outcome = mem.unmerge(rec.mutation_id).execute().await?;
    ///     println!("restored '{}' (nogood recorded: {})",
    ///         outcome.restored_entity, outcome.nogood_recorded);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "UnmergeRequest must call .execute() to run"]
    pub fn unmerge(&self, mutation_id: i64) -> UnmergeRequest<'_> {
        UnmergeRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Restore a fact previously moved to `facts_archive` (P2 archival) back into
    /// `facts` (arch-spec §3.2 / §4.4). Idempotent: `already_live = true` when the
    /// fact is already live. Must call `.execute()`.
    ///
    /// # Where `archived_fact_id` comes from
    ///
    /// [`list_mutations`](Self::list_mutations)`.kind(MutationKind::FactArchive)` —
    /// each record's `summary` names the archived fact, and its `mutation_id` is
    /// what [`undo`](Self::undo) takes. Prefer `undo(mutation_id)`: it performs the
    /// same restore AND marks the mutation reversed, which this method cannot do
    /// because it is not given the log row. This method stays for the case where
    /// you already hold the fact's id (it is the fact's ORIGINAL `facts.id` —
    /// `facts_archive.id` carries it unchanged).
    #[must_use = "RestoreArchivedRequest must call .execute() to run"]
    pub fn restore_archived_fact(&self, archived_fact_id: i64) -> RestoreArchivedRequest<'_> {
        RestoreArchivedRequest {
            memory: self,
            archived_fact_id,
        }
    }

    /// Clear a supersession bound (`valid_to` / `expired_at`) set by
    /// `supersede(...)`, re-opening the fact as currently-true (arch-spec §4.5).
    /// Idempotent: `NotSuperseded` when no bound was set. Must call `.execute()`.
    #[must_use = "UnsupersedeRequest must call .execute() to run"]
    pub fn unsupersede(&self, fact_id: i64) -> UnsupersedeRequest<'_> {
        UnsupersedeRequest {
            memory: self,
            fact_id,
        }
    }

    /// Edit an entity — retype or rename — with full FK-propagation, provenance
    /// snapshot, and reconciler-freeze re-open (reversible-graph-mutations
    /// arch-spec §4.3). Completes the diarization flow: after `unmerge`, rename
    /// `"Speaker 1"` to `"Alice"` and every one of its facts / episodic edges /
    /// archived facts / community membership re-points to `alice`.
    ///
    /// Choose exactly one operation on the builder:
    /// - `.rename(new_id)` — REKEY the entity's id (rejects renaming INTO an
    ///   existing id with `Error::EntityEditConflict`; merge explicitly instead).
    /// - `.retype(type_id)` — change the entity's type (pins it `ConsumerPinned`).
    ///   `type_id` must be REGISTERED in the namespace (or be the id=0 "Entity"
    ///   catch-all, always allowed); an unregistered id errors
    ///   `Error::EntityEditInvalid` rather than being coerced to the catch-all.
    ///
    /// The edit is undoable via [`undo_entity_edit`](Self::undo_entity_edit) with
    /// the returned `EditEntityOutcome.mutation_id`. Must call `.execute()`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::{Memory, Namespace};
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// // Rename a diarization placeholder; every fact / edge re-points to the new id.
    /// let edit = mem.edit_entity("Speaker 1")
    ///     .rename("alice")
    ///     .in_namespace(Namespace::new("meeting"))
    ///     .execute()
    ///     .await?;
    /// // ...and reverse it later via the returned mutation_id.
    /// mem.undo_entity_edit(edit.mutation_id).execute().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "EditEntityRequest must call .execute() to run"]
    pub fn edit_entity<'a>(&'a self, entity_id: impl Into<String> + 'a) -> EditEntityRequest<'a> {
        EditEntityRequest {
            memory: self,
            entity_id: entity_id.into(),
            namespace: None,
            new_id: None,
            new_type_id: None,
        }
    }

    /// Reverse a prior `edit_entity` (retype or rename/rekey) from its provenance
    /// snapshot (arch-spec §4.3). Pass the `mutation_id` from the
    /// `EditEntityOutcome` (or from `mutation_history` / `list_mutations`).
    /// Idempotent: a second call is a zero-count no-op. Must call `.execute()`.
    #[must_use = "UndoEntityEditRequest must call .execute() to run"]
    pub fn undo_entity_edit(&self, mutation_id: i64) -> UndoEntityEditRequest<'_> {
        UndoEntityEditRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Delete an entity, reversibly (reversible-graph-mutations arch-spec §4.4).
    /// The entity's facts are ARCHIVED (recoverable — never hard-deleted), its
    /// edges / community membership / FTS / row are removed, and any neighbour whose
    /// live-fact support drops to zero has its DERIVED community membership retracted
    /// (the base entity is never auto-deleted). Undoable via
    /// [`undo_delete_entity`](Self::undo_delete_entity) with the returned
    /// `DeleteEntityOutcome.mutation_id`. Must call `.execute()` (destructive op).
    #[must_use = "DeleteEntityRequest must call .execute() to run"]
    pub fn delete_entity<'a>(
        &'a self,
        entity_id: impl Into<String> + 'a,
    ) -> DeleteEntityRequest<'a> {
        DeleteEntityRequest {
            memory: self,
            entity_id: entity_id.into(),
            namespace: None,
        }
    }

    /// Delete a single fact, reversibly (reversible-graph-mutations arch-spec §4.5).
    /// The fact is archived (recoverable via
    /// [`restore_archived_fact`](Self::restore_archived_fact)); either endpoint whose
    /// support drops to zero has its DERIVED community membership retracted. Undoable
    /// via [`undo_delete_fact`](Self::undo_delete_fact). A fact id is global. Must
    /// call `.execute()`.
    #[must_use = "DeleteFactRequest must call .execute() to run"]
    pub fn delete_fact(&self, fact_id: i64) -> DeleteFactRequest<'_> {
        DeleteFactRequest {
            memory: self,
            fact_id,
        }
    }

    /// Reverse a prior `delete_entity` from its provenance snapshot (arch-spec §4.4)
    /// — re-inserts the entity + FTS, restores its archived facts + episodic edges,
    /// and un-retracts every community membership the cascade retracted. Pass the
    /// `mutation_id` from the `DeleteEntityOutcome` (or `mutation_history` /
    /// `list_mutations`). Idempotent. Must call `.execute()`.
    #[must_use = "UndoDeleteEntityRequest must call .execute() to run"]
    pub fn undo_delete_entity(&self, mutation_id: i64) -> UndoDeleteEntityRequest<'_> {
        UndoDeleteEntityRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Reverse a prior `delete_fact` from its provenance snapshot (arch-spec §4.5) —
    /// restores the archived fact + un-retracts any neighbour the cascade retracted.
    /// Idempotent. Must call `.execute()`.
    #[must_use = "UndoDeleteFactRequest must call .execute() to run"]
    pub fn undo_delete_fact(&self, mutation_id: i64) -> UndoDeleteFactRequest<'_> {
        UndoDeleteFactRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Inspect the mutations `dream()` applied to a single entity — the **SEE**
    /// half of the reversible-mutation story (arch-spec §3 "Inspect surface").
    ///
    /// Returns a newest-first `Vec<MutationRecord>` of every logged graph mutation
    /// that touched `entity_id` (for an `entity_merge`, whether the entity was the
    /// loser OR the keeper), each carrying the `mutation_id` to pass to
    /// [`unmerge`](Self::unmerge). Includes already-undone mutations
    /// (`undone = true`), so a reversed merge is still visible.
    ///
    /// Namespace: `.in_namespace(ns)` or a `default_namespace` on the builder is
    /// required (an entity id is namespace-scoped). Read-only — `.await` it.
    ///
    /// Only the five LOGGED [`MutationKind`]s appear here (`EntityMerge` /
    /// `EntityEdit` / `EntityDelete` / `FactDelete` / `FactArchive`); the other
    /// three are reserved and never surface — see
    /// [`list_mutations`](Self::list_mutations) for the tracked-kind boundary.
    #[must_use = "MutationHistoryRequest must be .await-ed"]
    pub fn mutation_history<'a>(
        &'a self,
        entity_id: impl Into<String> + 'a,
    ) -> MutationHistoryRequest<'a> {
        MutationHistoryRequest {
            memory: self,
            entity_id: entity_id.into(),
            namespace: None,
        }
    }

    /// List the mutations `dream()` applied, newest-first — the **SEE** surface
    /// for a whole namespace (arch-spec §3 "Inspect surface").
    ///
    /// Returns `Vec<MutationRecord>` (each carrying its `mutation_id` to undo via
    /// [`undo`](Self::undo)). Filter with `.kind(k)` / `.since(ts)` /
    /// `.include_undone(true)`; scope with `.in_namespace(ns)` (else the
    /// `default_namespace`, else ALL namespaces). Default view is LIVE
    /// (still-reversible) mutations only. Read-only — `.await` it.
    ///
    /// # Tracked-kind boundary (5 of 8)
    ///
    /// FIVE [`MutationKind`] variants are currently LOGGED (hence listable and
    /// reversible via [`undo`](Self::undo)): `EntityMerge`, `EntityEdit`,
    /// `EntityDelete`, `FactDelete`, `FactArchive`. The other three
    /// (`FactSupersede`, `CommunityAssign`, `CanonicalForm`) are RESERVED — not yet
    /// produced into the `graph_mutation_log` — so
    /// `list_mutations().kind(<a reserved kind>)` returns EMPTY by construction
    /// (not "nothing changed"). `FactSupersede` is itself reversible, but through
    /// the domain-id method [`unsupersede`](Self::unsupersede), not this
    /// inspect+undo surface.
    ///
    /// **This is where an archived fact's id comes from.** `DreamSummary` reports
    /// `facts_archived` as a COUNT, so before `FactArchive` was logged nothing named
    /// WHICH facts a dream retired, and
    /// [`restore_archived_fact`](Self::restore_archived_fact) took an id no public
    /// read returned (TD-250).
    #[must_use = "ListMutationsRequest must be .await-ed"]
    pub fn list_mutations(&self) -> ListMutationsRequest<'_> {
        ListMutationsRequest {
            memory: self,
            namespace: None,
            kind: None,
            since: None,
            include_undone: false,
        }
    }

    /// Run batch consolidation (dream phase).
    ///
    /// Must call `.execute()` explicitly — like [`forget`](Self::forget) /
    /// [`undo`](Self::undo) / the rest of the mutating surface, this is a
    /// destructive-terminal builder, not a bare-`.await` one. `dream()` is
    /// arguably the single most consequential call in the whole API — by
    /// default it commits entity merges, fact archival, and a supersession
    /// sweep across the namespace — so the explicit terminal makes that
    /// intent visible in code review, the same rationale `forget()` already
    /// states for itself.
    ///
    /// Default: blocks until done (returns [`DreamSummary`]). All consolidation ops
    /// default ON and are REVERSIBLE — inspect what changed with
    /// [`mutation_history`](Self::mutation_history) / [`list_mutations`](Self::list_mutations),
    /// reverse anything with [`undo`](Self::undo). Use `.fire_and_forget()` to
    /// return a `DreamHandle` without blocking, or `.cross_episode(mode)` /
    /// `.with_opts(opts)` to tune the consolidation surface.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// let summary = mem.dream().execute().await?;
    /// println!(
    ///     "communities updated: {}, entities reclassified: {}",
    ///     summary.communities_updated, summary.entities_reclassified,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "DreamRequest must call .execute() to run"]
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

    // ── Async extraction wait API ─────────────────────────────────────────────

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
                // DUR-3 (V1-CANONICAL §4.2): `skip_extraction` ingests are
                // TERMINAL on commit — nothing is ever enqueued for them, so
                // polling on would never resolve. Before this arm existed they
                // sat at `Pending` and every caller burned its full timeout
                // budget to be told `WaitTimeout`, i.e. FAILURE for an ingest
                // that succeeded and is durably stored.
                "Skipped" => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    metrics::histogram!(
                        "kremory.wait_for_processing.duration_ms",
                        "outcome" => "skipped"
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

    // ── Namespace policy (v0.1.4) ──────────────────────────────────────────────

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
    /// Backward-compat is **structural**: when `dream_llm` is `None`, this is
    /// byte-for-byte the prior
    /// `llm_or_err("dream", …)` behaviour — same provider, same error, same metric.
    ///
    /// Emits `kremory.dream.llm_role_selected_total{model_role}` on **every** call.
    /// The counter measures role-selection *attempts*, not realised
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
                    "dream phase using dedicated dream_llm provider"
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

    /// Return the model id the dream phase should thread into its LLM passes for
    /// capability detection: the dedicated `with_dream_model_id` string when set,
    /// else the main `with_model_id` string, else `None`.
    ///
    /// Mirrors [`dream_llm_or_main`](Self::dream_llm_or_main) at the model-id
    /// layer: `dream_model_id` pairs with `dream_llm` the way `model_id` pairs
    /// with `llm`. `None` (raw `with_llm` without a model id) means the dream
    /// passes degrade to `PromptOnly` exactly as the interactive path does —
    /// no worse than before, and the correct behaviour when the model is unknown.
    ///
    /// Before this resolver, the facade dream path hardcoded an empty
    /// model string in every LLM pass, silently degrading every configuration
    /// (even a fully-specified `with_model_id`) to zero structured output.
    pub(crate) fn dream_model_id_or_main(&self) -> Option<&str> {
        self.dream_model_id.as_deref().or(self.model_id.as_deref())
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
    pub async fn ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        self.graph.graph_ghost_episodes(group_id).await
    }

    /// Pin an entity as `ConsumerPinned`, protecting it from dream reclassification.
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` on the entity row.
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

