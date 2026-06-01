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
    pub(crate) graph: Arc<dyn GraphHandle>,
    pub(crate) llm: Arc<dyn ChatProvider>,
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
            let (graph, temporal_graph) = providers::open_graph(
                self.path.as_path(),
                llm.clone(),
                embedder.clone(),
                self.embedding_dim,
            )
            .await?;

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

            Ok(Memory {
                graph,
                llm,
                embedder,
                default_sink: self.default_sink,
                default_namespace: self.default_namespace,
                temporal_graph: Some(temporal_graph),
                episode_content_warn_threshold: self.episode_content_warn_threshold,
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

        // G4: soft warn at content threshold. Never enforced; observability only.
        if let Some(threshold) = self.memory.episode_content_warn_threshold {
            let chars = self.content.chars().count();
            if chars > threshold {
                tracing::warn!(
                    episode_chars = chars,
                    threshold = threshold,
                    "episode content exceeds soft threshold — extraction quality may degrade; consider pre-chunking"
                );
                metrics::counter!("kremory_episode_oversize_total").increment(1);
            }
        }

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

        // ADR-029a lazy population: ensure a default-policy row exists for
        // the namespace before the first write.
        self.memory.ensure_namespace_policy(&ns).await?;

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
            // ADR-029a lazy population.
            self.memory.ensure_namespace_policy(&ns).await?;
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
    /// Single-namespace selector. Mutually exclusive with `namespaces`.
    namespace: Option<Namespace>,
    /// Multi-namespace selector (ADR-029c Decision 6). Mutually exclusive with
    /// `namespace`. When set, fan-out via `tokio::join_all` executes one
    /// sub-query per namespace and blends results via cross-namespace RRF.
    namespaces: Option<Vec<Namespace>>,
    /// Per-namespace top-K cap before cross-namespace RRF blend (ADR-029c
    /// Decision 4). Default: effective `k`. Raising this value improves recall
    /// diversity for low-coverage namespaces at the cost of extra sub-query work.
    per_namespace_top_k: Option<usize>,
    /// When `true`, sub-query errors emit `tracing::warn!` and the failing
    /// namespace is skipped rather than propagating `Err` to the caller
    /// (ADR-029c Decision 4 sub-decision M1). Default `false` (fail-all).
    best_effort: bool,
    /// Stable id for correlating tracing spans across multi-namespace fan-out.
    /// Auto-generated at `RecallRequest` construction; override via
    /// `with_recall_id`. (ADR-029c Decision 5).
    recall_id: Uuid,
    k: Option<usize>,
    as_of: Option<DateTime<Utc>>,
    template: Option<RecallTemplate>,
    raw_mode: bool,
    opts: Option<SearchOpts>,
    /// G6 — single-value AND filters: `metadata[key] == value` (Vera F6).
    /// Multiple calls AND together. Keys validated via [`validate_metadata_key`].
    metadata_filters: Vec<(String, serde_json::Value)>,
    /// G6.b — multi-value OR-within-key filters: `metadata[key] IN values`
    /// (Vera F6). Multiple calls AND across keys. Empty `values` slice causes
    /// .await to return `Err` via [`Self::pending_error`].
    metadata_filters_in: Vec<(String, Vec<serde_json::Value>)>,
    /// G6 — deferred-error slot. Set by validating builders (filter_metadata,
    /// filter_metadata_in). Surfaces at `.await` time via `Err`. Matches the
    /// `ConflictingNamespaceSelectors` pattern at facade/mod.rs:1373.
    pending_error: Option<MemoryError>,
}

/// G6 — Apply metadata filters as a post-filter to recall results.
///
/// Looks up each result's episode metadata via the
/// `entities → episodic_edges → episodes` JOIN (entity-scoped within the
/// recall namespace), then applies AND-across-filters / OR-within-`filter_in`
/// semantics. An entity is retained iff AT LEAST ONE of its episodes within
/// the namespace satisfies all filters.
///
/// # Why post-filter at v0.1.6
///
/// The substrate-level `GraphHandle::graph_search` predates metadata filters;
/// promoting them to a SQL pre-filter via `json_extract(...)` requires
/// extending the substrate trait + an index opportunity is scheduled for
/// v0.1.7. Post-filter is correct + traceable here.
async fn apply_metadata_post_filter(
    memory: &Memory,
    namespace: &Namespace,
    results: Vec<RetrievedContext>,
    metadata_filters: &[(String, serde_json::Value)],
    metadata_filters_in: &[(String, Vec<serde_json::Value>)],
) -> Result<Vec<RetrievedContext>> {
    if metadata_filters.is_empty() && metadata_filters_in.is_empty() {
        return Ok(results);
    }
    let tg = memory.temporal_graph.as_ref().ok_or_else(|| {
        MemoryError::Other(
            "filter_metadata post-filter requires a Memory constructed via the \
             builder/providers path (no Arc<TemporalGraph> attached)"
                .to_string(),
        )
    })?;
    let conn = &tg.conn;
    let mut filtered: Vec<RetrievedContext> = Vec::with_capacity(results.len());
    for r in results {
        let mut rows = conn
            .query(
                "SELECT DISTINCT e.metadata FROM episodes e \
                 JOIN episodic_edges ee ON ee.episode_id = e.id \
                 WHERE ee.entity_id = ?1 AND e.group_id = ?2",
                libsql::params![r.entity_id.clone(), namespace.namespace.clone()],
            )
            .await
            .map_err(CoreError::Database)?;
        let mut any_pass = false;
        while let Some(row) = rows.next().await.map_err(CoreError::Database)? {
            let meta_text = row.get::<Option<String>>(0).map_err(CoreError::Database)?;
            let meta: Option<serde_json::Value> = match meta_text {
                Some(t) => match serde_json::from_str(&t) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        // Quinn C2 — surface parse failures via tracing rather
                        // than silently dropping the episode. Schema corruption
                        // or encoding bugs would otherwise be invisible.
                        tracing::warn!(
                            entity_id = %r.entity_id,
                            namespace = %namespace.namespace,
                            error = %e,
                            "filter_metadata post-filter: episode.metadata JSON parse failed, treating as absent"
                        );
                        None
                    }
                },
                None => None,
            };
            if metadata_matches(meta.as_ref(), metadata_filters, metadata_filters_in) {
                any_pass = true;
                break;
            }
        }
        if any_pass {
            filtered.push(r);
        }
    }
    Ok(filtered)
}

/// AND-across-filters / OR-within-`filter_in`. A metadata object that lacks
/// any required key fails the filter (missing-key counts as no-match).
fn metadata_matches(
    meta: Option<&serde_json::Value>,
    filters: &[(String, serde_json::Value)],
    filters_in: &[(String, Vec<serde_json::Value>)],
) -> bool {
    let Some(m) = meta else {
        return false;
    };
    for (k, v) in filters {
        if m.get(k) != Some(v) {
            return false;
        }
    }
    for (k, vs) in filters_in {
        let Some(actual) = m.get(k) else {
            return false;
        };
        if !vs.iter().any(|v| actual == v) {
            return false;
        }
    }
    true
}

/// G6 — Vera F6 validation for metadata filter keys. Rejects path-injection
/// metachars + length + empty + digit-prefix. Returns `Err` describing the
/// reject reason (caller stores in `pending_error` for deferred surface).
fn validate_metadata_key(key: &str) -> Result<()> {
    if key.is_empty() {
        return Err(MemoryError::Other(
            "filter_metadata: key must not be empty".to_string(),
        ));
    }
    if key.len() > 128 {
        return Err(MemoryError::Other(format!(
            "filter_metadata: key length must be <= 128 (got {})",
            key.len()
        )));
    }
    if let Some(first) = key.chars().next() {
        if first.is_ascii_digit() {
            return Err(MemoryError::Other(format!(
                "filter_metadata: key must not start with a digit (got {key:?})"
            )));
        }
    }
    // Vera F6 path-injection guard. Bracket / quote / dollar / star / backslash
    // / dot are all interpreted by SQLite's json_extract path grammar; rejecting
    // here prevents consumer-controlled keys from escaping `'$.{key}'`.
    for ch in key.chars() {
        // Quinn C3: `{` and `}` added for defence-in-depth. They have no valid
        // top-level metadata key use, and could enable template-string escape
        // in any future code path that wraps the path in `{...}`.
        if matches!(
            ch,
            '.' | '[' | ']' | '\'' | '"' | '\\' | '$' | '*' | '{' | '}'
        ) {
            return Err(MemoryError::Other(format!(
                "filter_metadata: key must not contain JSON-path metachars \
                 (. [ ] ' \" \\ $ * {{ }}); got {key:?}"
            )));
        }
    }
    Ok(())
}

/// SQLite parameter cap (libsql ≥ 3.32.0). G6.b values must stay under this
/// to avoid `SQLITE_TOOBIG` at the SQL pre-filter promotion in v0.1.7.
const SQLITE_MAX_VARIABLE_NUMBER: usize = 32766;

impl<'a> RecallRequest<'a> {
    /// Set the namespace for this operation (overrides Memory default).
    /// Mutually exclusive with `in_namespaces` — setting both returns
    /// `Err(Error::ConflictingNamespaceSelectors)` at `.await` time.
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Recall across multiple namespaces concurrently, blending results via
    /// per-namespace top-K RRF. Each result in the returned `Vec` carries
    /// `namespace: Some(ns)` identifying its source namespace.
    ///
    /// # Empty slice
    ///
    /// `in_namespaces(&[])` returns `Err(MemoryError::MissingNamespace { ... })`.
    ///
    /// # Single-element equivalence
    ///
    /// `in_namespaces(&[ns])` is equivalent to `in_namespace(ns)` — same query
    /// plan, same result semantics, same `namespace: Some(ns)` attribution.
    ///
    /// # Mutual exclusion with `in_namespace`
    ///
    /// Calling both `in_namespace` and `in_namespaces` on the same request is a
    /// programming error and returns `Err(MemoryError::Core(Error::ConflictingNamespaceSelectors))`
    /// at `.await` time (checked by `check_selectors`).
    ///
    /// # Cold-cache startup latency
    ///
    /// Each namespace may trigger a DB policy lookup on first call if the
    /// `NamespacePolicyCache` is cold. For N > 4 namespaces at server startup,
    /// consider pre-warming via `register_namespace` on each namespace before
    /// the first `in_namespaces` call to avoid the cold-cache thundering-herd
    /// (see ADR-029c Decision 4 for details).
    pub fn in_namespaces(mut self, namespaces: &[Namespace]) -> Self {
        self.namespaces = Some(namespaces.to_vec());
        self
    }

    /// Cap the number of results fetched from each individual namespace before
    /// cross-namespace RRF blending. Default: `k` (or `Memory::default_k` if
    /// `k` is unset). Raising this value improves recall diversity for
    /// low-coverage namespaces at the cost of extra per-namespace sub-query work.
    pub fn per_namespace_top_k(mut self, n: usize) -> Self {
        self.per_namespace_top_k = Some(n);
        self
    }

    /// When `true`, sub-query errors emit `tracing::warn!` and the failing
    /// namespace is skipped rather than returning `Err` to the caller. The
    /// returned `Vec<RetrievedContext>` contains results from all namespaces
    /// that succeeded. Default: `false` (fail-all — appropriate for audit
    /// consumers where partial results are worse than no results).
    ///
    /// If ALL sub-queries fail, `best_effort(true)` still returns `Err`
    /// (returning an empty result set silently is worse than surfacing the error).
    pub fn best_effort(mut self, enabled: bool) -> Self {
        self.best_effort = enabled;
        self
    }

    /// Override the auto-generated `recall_id`. Use when correlating kremory
    /// tracing spans with an application-level request id.
    pub fn with_recall_id(mut self, id: Uuid) -> Self {
        self.recall_id = id;
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

    /// G6 — single-value equality filter on a top-level episode metadata key.
    /// Multiple calls AND together. Fluent — invalid keys surface as `Err`
    /// at `.await` time via the deferred-error pattern (matches
    /// `ConflictingNamespaceSelectors` at facade/mod.rs:1373).
    ///
    /// # Key validation (Vera F6 — JSON-path injection)
    ///
    /// `key` MUST be a top-level metadata field name only. Rejected at
    /// `.await` time when the key:
    /// - Contains any of `. [ ] ' " \ $ *` (path-traversal / escape chars)
    /// - Length > 128
    /// - Empty
    /// - Starts with an ASCII digit
    ///
    /// The key is later interpolated into SQLite's `json_extract` path
    /// (`'$.{key}'`); SQLite has no parameterized path equivalent so this
    /// validation is the only guard against path injection from
    /// consumer-controlled keys. Values are ALWAYS bound via libsql `?`.
    ///
    /// # v0.1.6 perf characteristics
    ///
    /// At v0.1.6 the filter is applied as a post-filter at facade level
    /// (after `memory::search` returns). Index promotion to a SQL pre-filter
    /// is scheduled for v0.1.7 (per kremory-v016-api-gaps spec G6 note
    /// "Index opportunity deferred to v0.1.7").
    pub fn filter_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        if let Err(e) = validate_metadata_key(key) {
            // Keep the first error; subsequent invalid keys won't overwrite.
            if self.pending_error.is_none() {
                self.pending_error = Some(e);
            }
            return self;
        }
        self.metadata_filters.push((key.to_string(), value));
        self
    }

    /// G6.b — multi-value OR-within-key equality filter:
    /// `metadata[key] ∈ values`. Same `key` validation as
    /// [`Self::filter_metadata`]. Multiple `filter_metadata_in` calls
    /// combine ACROSS keys with AND; values WITHIN a single call combine
    /// with OR.
    ///
    /// # Empty values slice
    ///
    /// Calling `filter_metadata_in("k", &[])` would never match anything
    /// and is almost always a caller bug — rejected at `.await` time as
    /// `Err(MemoryError::Other(...))`.
    ///
    /// # Limits
    ///
    /// libsql ≥ 3.32.0 enforces `SQLITE_MAX_VARIABLE_NUMBER = 32766` total
    /// bound parameters across a single statement (Vera cycle-2 RISK-004).
    /// Caller-side cap: keep `values.len()` under a few thousand per call
    /// to leave room for the query's other parameters.
    pub fn filter_metadata_in(mut self, key: &str, values: &[serde_json::Value]) -> Self {
        if let Err(e) = validate_metadata_key(key) {
            if self.pending_error.is_none() {
                self.pending_error = Some(e);
            }
            return self;
        }
        if values.is_empty() {
            if self.pending_error.is_none() {
                self.pending_error = Some(MemoryError::Other(format!(
                    "filter_metadata_in: values slice must not be empty (key {key:?})"
                )));
            }
            return self;
        }
        // Quinn C7 / Vera RISK-004 — cap values per call to leave room for
        // other bound params in the v0.1.7 SQL pre-filter promotion. The cap
        // already lands at v0.1.6 (post-filter) so behaviour stays stable when
        // the SQL plumbing arrives.
        if values.len() > SQLITE_MAX_VARIABLE_NUMBER {
            if self.pending_error.is_none() {
                self.pending_error = Some(MemoryError::Other(format!(
                    "filter_metadata_in: values length {} exceeds SQLite parameter cap {} (key {key:?})",
                    values.len(),
                    SQLITE_MAX_VARIABLE_NUMBER
                )));
            }
            return self;
        }
        self.metadata_filters_in
            .push((key.to_string(), values.to_vec()));
        self
    }

    /// Check that `in_namespace` and `in_namespaces` were not both set on this
    /// request. Called from both `execute()` and `RecallRawRequest::into_future`
    /// (ADR-029c Decision 6, closes M3).
    fn check_selectors(&self) -> Result<()> {
        if self.namespace.is_some() && self.namespaces.is_some() {
            return Err(MemoryError::Core(
                crate::core::error::Error::ConflictingNamespaceSelectors {
                    request: "in_namespace and in_namespaces both set on the same RecallRequest; \
                              use one or the other"
                        .to_string(),
                },
            ));
        }
        Ok(())
    }

    async fn execute(mut self) -> Result<String> {
        self.check_selectors()?;
        if let Some(err) = self.pending_error.take() {
            return Err(err);
        }

        let template = self.template.unwrap_or(RecallTemplate::TemporalFacts);

        // Multi-namespace fan-out path (ADR-029c Decision 4 + 6).
        if self.namespaces.is_some() {
            if self.namespaces.as_ref().is_none_or(|v| v.is_empty()) {
                return Err(MemoryError::MissingNamespace {
                    request:
                        "in_namespaces called with empty slice; provide at least one namespace",
                });
            }
            let results = self.execute_multi_namespace().await?;
            return Ok(memory::context_block(&results, template.into()));
        }

        // Single-namespace path (original behaviour).
        let ns = self.memory.resolve_namespace(self.namespace.clone())?;
        let opts = self.opts.clone().unwrap_or(SearchOpts {
            limit: self.k,
            as_of: self.as_of,
            source_kind: None,
        });
        // ADR-029a lazy population.
        self.memory.ensure_namespace_policy(&ns).await?;
        let recall_id = self.recall_id;
        let span = tracing::info_span!(
            "kremory.recall.single_ns",
            recall_id = %recall_id,
            namespace = %ns.namespace,
        );
        let _enter = span.enter();
        let mut results = memory::search(self.memory.graph.as_ref(), &self.query, ns.clone(), opts)
            .await?
            .into_iter()
            .map(|r| r.with_namespace(ns.clone()))
            .collect::<Vec<_>>();
        // G6 — apply metadata post-filter before template rendering.
        results = apply_metadata_post_filter(
            self.memory,
            &ns,
            results,
            &self.metadata_filters,
            &self.metadata_filters_in,
        )
        .await?;
        Ok(memory::context_block(&results, template.into()))
    }

    /// Fan-out recall across all namespaces in `self.namespaces`, blend via RRF,
    /// trim to `self.k`, and return results with `namespace: Some(ns)` attribution.
    ///
    /// Precondition: `self.namespaces` is `Some` and non-empty (caller checks).
    async fn execute_multi_namespace(self) -> Result<Vec<RetrievedContext>> {
        // Extract all fields upfront before consuming `self`.
        let memory = self.memory;
        let query = self.query;
        let namespaces = self
            .namespaces
            .ok_or_else(|| MemoryError::MissingNamespace {
                request:
                    "execute_multi_namespace called without namespaces (internal precondition)",
            })?;
        let opts_template = self.opts;
        let per_ns_k = self.per_namespace_top_k.or(self.k);
        let final_k = self.k;
        let as_of = self.as_of;
        let best_effort = self.best_effort;
        let recall_id = self.recall_id;
        // G6 — Quinn C1 fix: filters MUST apply per-namespace inside the sub-
        // future, otherwise multi-namespace recall silently drops them. Wrap in
        // Arc so the closures (one per namespace) can share without cloning the
        // Vecs N times.
        let metadata_filters = std::sync::Arc::new(self.metadata_filters);
        let metadata_filters_in = std::sync::Arc::new(self.metadata_filters_in);

        let outer_span = tracing::info_span!(
            "kremory.recall.multi_ns",
            recall_id = %recall_id,
            namespace_count = %namespaces.len(),
        );
        let _outer = outer_span.enter();

        // Collect sub-query futures — one per namespace.
        let mut sub_futures = Vec::with_capacity(namespaces.len());
        for ns in namespaces {
            let query = query.clone();
            let opts = opts_template.clone().unwrap_or(SearchOpts {
                limit: per_ns_k,
                as_of,
                source_kind: None,
            });
            let metadata_filters = std::sync::Arc::clone(&metadata_filters);
            let metadata_filters_in = std::sync::Arc::clone(&metadata_filters_in);
            sub_futures.push(async move {
                let span = tracing::info_span!(
                    "kremory.recall.sub_query",
                    recall_id = %recall_id,
                    namespace = %ns.namespace,
                    per_namespace_top_k = per_ns_k,
                );
                let _enter = span.enter();
                memory.ensure_namespace_policy(&ns).await?;
                let hits = memory::search(memory.graph.as_ref(), &query, ns.clone(), opts).await?;
                let attributed: Vec<RetrievedContext> = hits
                    .into_iter()
                    .map(|r| r.with_namespace(ns.clone()))
                    .collect();
                // G6 — Quinn C1: filters must apply per-namespace (entity_id
                // scoping requires the namespace's own conn lookup).
                let filtered = apply_metadata_post_filter(
                    memory,
                    &ns,
                    attributed,
                    &metadata_filters,
                    &metadata_filters_in,
                )
                .await?;
                Ok::<Vec<RetrievedContext>, MemoryError>(filtered)
            });
        }

        let sub_results = futures::future::join_all(sub_futures).await;

        // Collect results, honouring best_effort semantics.
        let mut all_results: Vec<RetrievedContext> = Vec::new();
        let mut last_err: Option<MemoryError> = None;
        for outcome in sub_results {
            match outcome {
                Ok(hits) => all_results.extend(hits),
                Err(e) => {
                    if best_effort {
                        tracing::warn!(
                            target: "kremory.recall",
                            recall_id = %recall_id,
                            error = %e,
                            "best_effort: namespace sub-query failed, skipping"
                        );
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        // If best_effort and ALL sub-queries failed, surface the last error.
        if best_effort && all_results.is_empty() {
            if let Some(e) = last_err {
                return Err(e);
            }
        }

        // Sort blended results by score descending (RRF scores from sub-queries).
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to final_k if set.
        if let Some(k) = final_k {
            all_results.truncate(k);
        }

        Ok(all_results)
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
        let mut inner = self.inner;
        Box::pin(async move {
            // ADR-029c Decision 6 / M3: check_selectors fires on the raw path too.
            inner.check_selectors()?;
            // G6 — surface deferred validation error from filter_metadata /
            // filter_metadata_in setters.
            if let Some(err) = inner.pending_error.take() {
                return Err(err);
            }

            // Multi-namespace fan-out path.
            if inner.namespaces.is_some() {
                if inner.namespaces.as_ref().is_none_or(|v| v.is_empty()) {
                    return Err(MemoryError::MissingNamespace {
                        request:
                            "in_namespaces called with empty slice; provide at least one namespace",
                    });
                }
                return inner.execute_multi_namespace().await;
            }

            let ns = inner.memory.resolve_namespace(inner.namespace.clone())?;
            let opts = inner.opts.unwrap_or(SearchOpts {
                limit: inner.k,
                as_of: inner.as_of,
                source_kind: None,
            });
            // ADR-029a lazy population.
            inner.memory.ensure_namespace_policy(&ns).await?;
            let recall_id = inner.recall_id;
            let span = tracing::info_span!(
                "kremory.recall.single_ns",
                recall_id = %recall_id,
                namespace = %ns.namespace,
            );
            let _enter = span.enter();
            let results: Vec<RetrievedContext> =
                memory::search(inner.memory.graph.as_ref(), &inner.query, ns.clone(), opts)
                    .await?
                    .into_iter()
                    .map(|r| r.with_namespace(ns.clone()))
                    .collect();
            // G6 — apply metadata post-filter before returning.
            let filtered = apply_metadata_post_filter(
                inner.memory,
                &ns,
                results,
                &inner.metadata_filters,
                &inner.metadata_filters_in,
            )
            .await?;
            Ok(filtered)
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
    /// G8 — narrow forget to entities derived from episodes with this
    /// `source_id`. Composes with `in_namespace`. Vera F17 shared-entity
    /// preservation applies: an entity is deleted only when all of its
    /// `episodic_edges` resolve to episodes matching the filter.
    source_id: Option<String>,
}

impl<'a> ForgetRequest<'a> {
    /// Set the namespace scope for deletion (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// G8 — narrow forget to entities derived from episodes matching this
    /// `source_id` (the column added in G1). Composes with `in_namespace`.
    ///
    /// # Shared-entity preservation (Vera F17)
    ///
    /// Entities referenced by ANY episode outside this `source_id` are NOT
    /// deleted. Only entities whose entire `episodic_edges` set falls within
    /// the matched episodes are removed. This prevents cross-source data loss
    /// when a single entity ("Acme Corp") is mentioned in multiple ingested
    /// documents.
    ///
    /// # AppendOnly enforcement
    ///
    /// Per ADR-029b §3.1, AppendOnly enforcement applies regardless of
    /// `by_source_id` scope — narrowing the forget set does not weaken the
    /// policy gate.
    pub fn by_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Execute the deletion. Returns the count of deleted entity rows.
    ///
    /// This is the only terminal for `ForgetRequest` — there is no implicit
    /// `.await` to prevent accidental destructive operations.
    ///
    /// # AppendOnly enforcement (ADR-029b §3.1)
    ///
    /// If the namespace has `AppendOnly` policy, returns
    /// `Err(MemoryError::Core(CoreError::NamespacePolicyViolation))`.
    pub async fn execute(self) -> Result<u64> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        // ADR-029a lazy population: ensure namespace row exists before read.
        self.memory.ensure_namespace_policy(&ns).await?;

        // ADR-029b §3.1: AppendOnly enforcement — forget is a mutation.
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::forget requires a Memory constructed via the builder/providers path \
                 (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let group_id = namespace_to_group_id(&ns);
        let policy = tg
            .get_namespace_policy_cached(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        if let Some(p) = &policy {
            if p.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                // ADR-029b §3.1 enforcement — v0.1.5 closure of the v0.1.4
                // declare-but-don't-enforce contract. ForgetRequest is a
                // mutating operation and is prohibited on AppendOnly
                // namespaces. Returns the canonical CoreError variant so
                // callers can pattern-match on the policy violation.
                return Err(MemoryError::Core(
                    crate::core::error::Error::NamespacePolicyViolation {
                        namespace: group_id.clone(),
                        operation: "forget".to_string(),
                        policy: p.clone(),
                    },
                ));
            }
        }

        // G8 — narrowed source_id forget path. Walks the
        // episodes(source_id) → episodic_edges(episode_id, entity_id) chain to
        // collect candidate entities, then applies Vera F17 shared-entity
        // preservation: an entity is deleted only when EVERY one of its
        // episodic_edges row falls inside the matched episode set. Entities
        // with edges to any episode outside the filter are pinned.
        if let Some(sid) = self.source_id {
            let conn = &tg.conn;
            // Step 1: candidate entity_ids = entities that have at least one
            // edge to an episode matching (source_id, group_id).
            let mut cand_rows = conn
                .query(
                    "SELECT DISTINCT ee.entity_id \
                     FROM episodic_edges ee \
                     JOIN episodes e ON e.id = ee.episode_id \
                     WHERE e.source_id = ?1 AND e.group_id = ?2",
                    libsql::params![sid.clone(), group_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?;
            let mut candidates: Vec<String> = Vec::new();
            while let Some(row) = cand_rows.next().await.map_err(CoreError::Database)? {
                candidates.push(row.get::<String>(0).map_err(CoreError::Database)?);
            }
            // Quinn C2 — N+1 visibility: log candidate count so a regression
            // (e.g. document with 200+ entities) surfaces in tracing before
            // the v0.1.7 SQL pre-filter promotion lands.
            tracing::debug!(
                source_id = %sid,
                namespace = %group_id,
                candidate_count = candidates.len(),
                "forget by_source_id: candidate entities collected"
            );
            // Step 2: pin any candidate that has ANY edge to an episode
            // OUTSIDE the matched set (Vera F17 shared-entity preservation).
            let mut to_delete: Vec<String> = Vec::with_capacity(candidates.len());
            for entity_id in candidates {
                let mut count_rows = conn
                    .query(
                        // De Morgan: NOT (source_id = ?2 AND group_id = ?3)
                        // expands to (source_id IS NULL OR source_id != ?2 OR
                        // group_id != ?3). NULL guard is load-bearing —
                        // episodes seeded before G1 land with source_id=NULL
                        // and must count as "outside" the filter. DO NOT
                        // "simplify" this clause without re-running the Vera
                        // F17 shared-entity preservation tests.
                        "SELECT COUNT(*) FROM episodic_edges ee \
                         JOIN episodes e ON e.id = ee.episode_id \
                         WHERE ee.entity_id = ?1 \
                           AND (e.source_id IS NULL OR e.source_id != ?2 OR e.group_id != ?3)",
                        libsql::params![entity_id.clone(), sid.clone(), group_id.clone()],
                    )
                    .await
                    .map_err(CoreError::Database)?;
                let outside_count: i64 = count_rows
                    .next()
                    .await
                    .map_err(CoreError::Database)?
                    .ok_or_else(|| {
                        MemoryError::Other(
                            "forget by_source_id: COUNT(*) returned no rows".to_string(),
                        )
                    })?
                    .get::<i64>(0)
                    .map_err(CoreError::Database)?;
                if outside_count == 0 {
                    to_delete.push(entity_id);
                }
            }
            let entities_deleted = if to_delete.is_empty() {
                0
            } else {
                tg.batch_forget(&to_delete)
                    .await
                    .map_err(MemoryError::Core)?
            };

            // Quinn C3 — spec §G8 says "only the episode row(s) AND edges
            // exclusively owned by this source_id are removed". Episode rows
            // are 1:1 with source_id (not shared across consumers), so
            // delete the matched episode rows after entity cleanup. The
            // FK (episodic_edges.episode_id → episodes.id) means we must
            // also drop any remaining episodic_edges pointing to these
            // episodes first (entities sharing with other sources stayed
            // pinned, but their edges to THIS source's episodes go).
            conn.execute(
                "DELETE FROM episodic_edges WHERE episode_id IN \
                 (SELECT id FROM episodes WHERE source_id = ?1 AND group_id = ?2)",
                libsql::params![sid.clone(), group_id.clone()],
            )
            .await
            .map_err(CoreError::Database)?;
            conn.execute(
                "DELETE FROM episodes WHERE source_id = ?1 AND group_id = ?2",
                libsql::params![sid, group_id],
            )
            .await
            .map_err(CoreError::Database)?;
            return Ok(entities_deleted);
        }

        // Wire to substrate: list entities in group, then batch_forget.
        let entities = tg
            .list_entities_in_group(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        if entities.is_empty() {
            return Ok(0);
        }
        let ids: Vec<String> = entities.into_iter().map(|e| e.id).collect();
        tg.batch_forget(&ids).await.map_err(MemoryError::Core)
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
        // ADR-029a lazy population.
        self.memory.ensure_namespace_policy(&ns).await?;

        // ADR-029b §3.1 enforcement — v0.1.5 closure of the v0.1.4
        // declare-but-don't-enforce contract. DreamRequest mutates the
        // graph (consolidation rewrites facts) and is prohibited on
        // AppendOnly namespaces. Returns the canonical CoreError variant
        // so callers can pattern-match on the policy violation.
        if let Some(tg) = self.memory.temporal_graph.as_ref() {
            let group_id = namespace_to_group_id(&ns);
            let policy = tg
                .get_namespace_policy_cached(&group_id)
                .await
                .map_err(MemoryError::Core)?;
            if let Some(p) = &policy {
                if p.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                    return Err(MemoryError::Core(
                        crate::core::error::Error::NamespacePolicyViolation {
                            namespace: group_id.clone(),
                            operation: "dream".to_string(),
                            policy: p.clone(),
                        },
                    ));
                }
            }
        }

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
            // ADR-029a lazy population.
            self.inner.memory.ensure_namespace_policy(&ns).await?;
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

// ── UpdateSourceUriRequest ────────────────────────────────────────────────────

/// Request to update an episode's `source_uri`. Built via [`Memory::update_source_uri`].
///
/// Must call `.to(new_uri)` before `.await`. Errors if no episodes match the
/// given `source_id`. Does NOT mutate facts, bi-temporal axes (`valid_from`,
/// `valid_to`, `recorded_at`), or other episode columns.
///
/// # Substrate-purity
///
/// `source_id` and `source_uri` are substrate-generic universal identifiers —
/// chat-grain, doc-grain, and code-grain consumers all use the same surface.
#[must_use = "UpdateSourceUriRequest must be .await-ed after calling .to(new_uri)"]
pub struct UpdateSourceUriRequest<'a> {
    memory: &'a Memory,
    source_id: String,
    new_uri: Option<String>,
}

impl<'a> UpdateSourceUriRequest<'a> {
    /// Set the new `source_uri` value.
    pub fn to(mut self, new_uri: impl Into<String>) -> Self {
        self.new_uri = Some(new_uri.into());
        self
    }
}

impl<'a> IntoFuture for UpdateSourceUriRequest<'a> {
    type Output = Result<u64>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let new_uri = self.new_uri.ok_or_else(|| {
                MemoryError::Other(
                    "update_source_uri: .to(new_uri) must be called before .await".into(),
                )
            })?;
            let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
                MemoryError::Other(
                    "Memory::update_source_uri requires a Memory constructed via \
                     the builder/providers path (no Arc<TemporalGraph> attached)"
                        .into(),
                )
            })?;
            let conn = &tg.conn;
            // Verify at least one episode with this source_id exists.
            let count: i64 = conn
                .query(
                    "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
                    libsql::params![self.source_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?
                .next()
                .await
                .map_err(CoreError::Database)?
                .ok_or_else(|| {
                    MemoryError::Other(
                        "update_source_uri: COUNT query returned no rows".to_string(),
                    )
                })?
                .get(0)
                .map_err(CoreError::Database)?;
            if count == 0 {
                return Err(MemoryError::Other(format!(
                    "update_source_uri: no episode found with source_id={}",
                    self.source_id
                )));
            }
            let updated = conn
                .execute(
                    "UPDATE episodes SET source_uri = ?1 WHERE source_id = ?2",
                    libsql::params![new_uri, self.source_id],
                )
                .await
                .map_err(CoreError::Database)?;
            Ok(updated)
        })
    }
}

// ── Memory::update_source_uri entry point ─────────────────────────────────────

impl Memory {
    /// Update the `source_uri` for all episode(s) with the given `source_id`.
    ///
    /// Does NOT mutate facts, bi-temporal axes (`valid_from`, `valid_to`,
    /// `recorded_at`), or any other episode column. Returns `Err` if no
    /// episodes match the given `source_id`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// let n = mem.update_source_uri("doc-abc").to("path/v2").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "UpdateSourceUriRequest must be .await-ed after calling .to(new_uri)"]
    pub fn update_source_uri<'a>(
        &'a self,
        source_id: impl Into<String> + 'a,
    ) -> UpdateSourceUriRequest<'a> {
        UpdateSourceUriRequest {
            memory: self,
            source_id: source_id.into(),
            new_uri: None,
        }
    }
}

// ── UpdateEpisodeMetadataRequest builder ──────────────────────────────────────

/// Request to shallow-merge a JSON patch into episode metadata.
/// Built via [`Memory::update_episode_metadata`]. See that method for
/// merge semantics (shallow, arrays-replace, non-object rejected).
pub struct UpdateEpisodeMetadataRequest<'a> {
    memory: &'a Memory,
    source_id: String,
    patch: Option<serde_json::Value>,
    pending_error: Option<MemoryError>,
}

impl<'a> UpdateEpisodeMetadataRequest<'a> {
    /// Set the JSON patch to merge.
    ///
    /// The patch MUST be a JSON object (`Value::Object(_)`). Non-object
    /// patches (numbers, strings, arrays, booleans, null) cause `.await`
    /// to return `Err(MemoryError::Other(...))`.
    pub fn patch(mut self, patch: serde_json::Value) -> Self {
        if !patch.is_object() {
            self.pending_error = Some(MemoryError::Other(format!(
                "update_episode_metadata: patch must be a JSON object, got: {patch}"
            )));
        } else {
            self.patch = Some(patch);
        }
        self
    }
}

impl<'a> std::future::IntoFuture for UpdateEpisodeMetadataRequest<'a> {
    type Output = Result<usize>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            if let Some(err) = self.pending_error {
                return Err(err);
            }
            let patch_value = self.patch.ok_or_else(|| {
                MemoryError::Other(
                    "update_episode_metadata: .patch(value) must be called before .await"
                        .to_string(),
                )
            })?;
            let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
                MemoryError::Other(
                    "Memory::update_episode_metadata requires a Memory constructed via \
                     the builder/providers path (no Arc<TemporalGraph> attached)"
                        .to_string(),
                )
            })?;
            let conn = &tg.conn;

            // Verify at least one episode with this source_id exists.
            let count: i64 = conn
                .query(
                    "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
                    libsql::params![self.source_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?
                .next()
                .await
                .map_err(CoreError::Database)?
                .ok_or_else(|| {
                    MemoryError::Other(
                        "update_episode_metadata: COUNT query returned no rows".to_string(),
                    )
                })?
                .get(0)
                .map_err(CoreError::Database)?;

            if count == 0 {
                return Err(MemoryError::Other(format!(
                    "update_episode_metadata: no episode found with source_id={}",
                    self.source_id
                )));
            }

            // Read existing metadata TEXT for the source_id.
            let existing_text: Option<String> = conn
                .query(
                    "SELECT metadata FROM episodes WHERE source_id = ?1",
                    libsql::params![self.source_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?
                .next()
                .await
                .map_err(CoreError::Database)?
                .ok_or_else(|| {
                    MemoryError::Other(
                        "update_episode_metadata: metadata SELECT returned no rows".to_string(),
                    )
                })?
                .get(0)
                .map_err(CoreError::Database)?;

            // Parse existing (NULL → empty object).
            let mut existing_obj: serde_json::Map<String, serde_json::Value> = match existing_text {
                Some(ref s) => serde_json::from_str(s).map_err(|e| {
                    MemoryError::Other(format!(
                        "update_episode_metadata: failed to parse existing metadata as JSON: {e}"
                    ))
                })?,
                None => serde_json::Map::new(),
            };

            // Shallow-merge: iterate patch top-level keys, overwrite/insert.
            // Arrays are replaced, not merged — this is automatic because we
            // overwrite the top-level key, not recurse into nested structures.
            if let Some(patch_obj) = patch_value.as_object() {
                for (k, v) in patch_obj {
                    existing_obj.insert(k.clone(), v.clone());
                }
            }

            // Serialize and UPDATE.
            let merged_text = serde_json::to_string(&existing_obj).map_err(|e| {
                MemoryError::Other(format!(
                    "update_episode_metadata: failed to serialize merged metadata: {e}"
                ))
            })?;

            let updated = conn
                .execute(
                    "UPDATE episodes SET metadata = ?1 WHERE source_id = ?2",
                    libsql::params![merged_text, self.source_id],
                )
                .await
                .map_err(CoreError::Database)?;

            Ok(updated as usize)
        })
    }
}

// ── Memory::update_episode_metadata entry point ───────────────────────────────

impl Memory {
    /// Update the `metadata` JSON column for the episode(s) with the given
    /// `source_id` via shallow-merge of `patch` into existing metadata.
    ///
    /// # Semantics
    ///
    /// - **Shallow merge**: patch keys at the top level merge with existing
    ///   metadata; nested objects are NOT recursively merged.
    /// - **Arrays REPLACE not merge**: a patch `{"refs": [b]}` FULLY REPLACES
    ///   an existing `{"refs": [a]}` — no concatenation, no de-duplication.
    /// - **NULL → empty object**: if `episode.metadata` is NULL, treats as
    ///   `{}` and merges.
    /// - **Patch must be Object**: a non-object patch (`json!(42)`, array,
    ///   string) returns `Err` at `.await` time.
    /// - **Does NOT mutate**: facts, bi-temporal axes (`valid_from`,
    ///   `valid_to`, `recorded_at` on facts), `source_id`, `source_uri`,
    ///   `recorded_at` on episodes — only the `metadata` column changes.
    ///
    /// # Substrate-purity
    ///
    /// `metadata` is opaque JSON. The substrate enforces no schema on its
    /// contents — consumers carry their own taxonomy.
    #[must_use = "UpdateEpisodeMetadataRequest must be .await-ed or have a terminal called"]
    pub fn update_episode_metadata<'a>(
        &'a self,
        source_id: impl Into<String> + 'a,
    ) -> UpdateEpisodeMetadataRequest<'a> {
        UpdateEpisodeMetadataRequest {
            memory: self,
            source_id: source_id.into(),
            patch: None,
            pending_error: None,
        }
    }
}

// ── G2 Red tests (AC.3) ───────────────────────────────────────────────────────
//
// These tests INTENTIONALLY FAIL to compile until the Green agent ships:
//   - `Memory::update_source_uri(&self, source_id) -> UpdateSourceUriRequest<'_>`
//   - `UpdateSourceUriRequest::to(self, new_uri) -> Self`
//   - `UpdateSourceUriRequest::execute(self) -> Result<u64>` (async)
//
// Placement: inline in facade/mod.rs per Tessa AC.3 + test-strategy T1 spec.
// See: .ai-docs/specs/test-strategy-kremory-v016-substrate-2026-05-29.md §5 AC.3

#[cfg(test)]
mod update_source_uri_tests {
    use std::sync::Arc;

    use crate::{
        core::provider::{MockChatProvider, NullEmbeddingProvider},
        memory::types::Namespace,
    };

    use super::Memory;

    /// Build an in-memory `Memory` instance wired with null providers.
    ///
    /// Uses `TemporalGraph::open_with_dim(":memory:", 384)` under the hood
    /// via `providers::open_graph`. Both LLM and embedder are null — the G2
    /// invariant under test (source_uri update) requires neither.
    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });

        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("in-memory Memory construction must not fail")
    }

    /// Seed an episode row directly via SQL, bypassing the facade ingest path.
    ///
    /// Required because `RememberRequest::with_source_id` / `with_source_uri`
    /// do not yet exist (G1 partial — struct fields added, facade builder
    /// methods deferred to G5). The `conn_for_test()` accessor is gated behind
    /// `#[cfg(any(test, feature = "test-utils"))]`.
    ///
    /// Returns the `group_id` used for the episode row.
    async fn seed_episode_with_source(
        mem: &Memory,
        source_id: &str,
        source_uri: &str,
        namespace: &Namespace,
    ) -> String {
        use crate::memory::engine_handle::namespace_to_group_id;

        let group_id = namespace_to_group_id(namespace);
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some for in-memory Memory");
        let conn = tg.conn_for_test();

        conn.execute(
            "INSERT INTO episodes \
             (group_id, content, timestamp, recorded_at, source_id, source_uri) \
             VALUES (?1, ?2, unixepoch('now'), datetime('now'), ?3, ?4)",
            libsql::params![
                group_id.clone(),
                "test content for source identity.",
                source_id,
                source_uri,
            ],
        )
        .await
        .expect("seed_episode_with_source INSERT must succeed");

        group_id
    }

    // ── AC.3 happy path ───────────────────────────────────────────────────────

    /// AC.3 — happy path: `update_source_uri` updates `source_uri` on the
    /// target episode without mutating any facts or bi-temporal axes.
    ///
    /// Invariants asserted:
    /// - `source_uri` changed to the new value on the episode row.
    /// - `source_id` is unchanged.
    /// - `recorded_at` on the episode row is unchanged.
    /// - Fact count in the `facts` table is unchanged (zero facts were
    ///   inserted by the seed, so the count remains zero).
    /// - No facts rows have `valid_from` mutated (vacuously true for zero
    ///   facts, but the assertion pattern confirms the contract explicitly).
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_source_uri` + `UpdateSourceUriRequest`.
    #[tokio::test]
    async fn update_source_uri_does_not_mutate_facts_or_timestamps() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g2-happy");

        seed_episode_with_source(&mem, "doc-a", "path/old", &ns).await;

        // Capture pre-update state from episodes table.
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let pre_row = conn
            .query(
                "SELECT source_uri, source_id, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("pre-update query must succeed")
            .next()
            .await
            .expect("pre-update query row_next must succeed")
            .expect("pre-update episode row must exist");

        let pre_source_uri: String = pre_row.get(0).expect("source_uri column");
        let pre_source_id: String = pre_row.get(1).expect("source_id column");
        let pre_recorded_at: String = pre_row.get(2).expect("recorded_at column");

        assert_eq!(
            pre_source_uri, "path/old",
            "seed: source_uri must be path/old before update"
        );
        assert_eq!(pre_source_id, "doc-a", "seed: source_id must be doc-a");

        // Capture pre-update fact count. NullLlm produces zero facts — assert
        // zero and confirm it stays zero after the update.
        let pre_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts f \
                 JOIN episodes e ON e.id = f.subject_id \
                 WHERE e.source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("pre-update fact count query must succeed")
            .next()
            .await
            .expect("fact count row_next must succeed")
            .expect("fact count row must exist")
            .get(0)
            .expect("fact count column");

        // ── G2 builder call — FAILS TO COMPILE until Green ships the builder ──
        let updated_count: u64 = mem
            .update_source_uri("doc-a")
            .to("path/new")
            .await
            .expect("update_source_uri must succeed for existing source_id");

        assert!(
            updated_count >= 1,
            "at least one episode row must be updated"
        );

        // Post-update: verify source_uri changed, source_id unchanged.
        let post_row = conn
            .query(
                "SELECT source_uri, source_id, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("post-update query must succeed")
            .next()
            .await
            .expect("post-update query row_next must succeed")
            .expect("post-update episode row must exist");

        let post_source_uri: String = post_row.get(0).expect("source_uri column post");
        let post_source_id: String = post_row.get(1).expect("source_id column post");
        let post_recorded_at: String = post_row.get(2).expect("recorded_at column post");

        assert_eq!(
            post_source_uri, "path/new",
            "source_uri must be updated to path/new"
        );
        assert_eq!(
            post_source_id, "doc-a",
            "source_id must be unchanged after update"
        );
        assert_eq!(
            post_recorded_at, pre_recorded_at,
            "recorded_at must not be mutated by update_source_uri"
        );

        // Post-update: fact count must be unchanged.
        let post_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts f \
                 JOIN episodes e ON e.id = f.subject_id \
                 WHERE e.source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("post-update fact count query must succeed")
            .next()
            .await
            .expect("post-update fact count row_next must succeed")
            .expect("post-update fact count row must exist")
            .get(0)
            .expect("post-update fact count column");

        assert_eq!(
            post_fact_count, pre_fact_count,
            "fact count must not change after update_source_uri"
        );
    }

    // ── AC.3 error path ───────────────────────────────────────────────────────

    /// AC.3 — error path: `update_source_uri` with a `source_id` that does not
    /// exist in any episode must return `Err`.
    ///
    /// The exact error variant is determined by the Green agent (expected:
    /// `MemoryError::Other` or a new `SourceIdNotFound` variant). This test
    /// asserts only `is_err()` so it remains valid regardless of which named
    /// variant Green introduces.
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_source_uri` + `UpdateSourceUriRequest`.
    #[tokio::test]
    async fn update_source_uri_missing_source_id_returns_err() {
        let mem = make_memory().await;

        // ── G2 builder call — FAILS TO COMPILE until Green ships the builder ──
        let result = mem
            .update_source_uri("nonexistent-doc-12345")
            .to("any/path")
            .await;

        assert!(
            result.is_err(),
            "update_source_uri with unknown source_id must return Err, got: {:?}",
            result
        );
    }

    // ── AC.3 idempotency ──────────────────────────────────────────────────────

    /// AC.3 — idempotency: calling `update_source_uri` with the same URI as
    /// the current value must return `Ok` without error and leave the row
    /// unchanged.
    ///
    /// Per spec §G2: "Idempotent: re-running with same new_uri is a no-op
    /// (returns count)."
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_source_uri` + `UpdateSourceUriRequest`.
    #[tokio::test]
    async fn update_source_uri_idempotent_same_uri() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g2-idempotent");

        seed_episode_with_source(&mem, "x", "p", &ns).await;

        // ── G2 builder call — FAILS TO COMPILE until Green ships the builder ──
        let result = mem.update_source_uri("x").to("p").await;

        assert!(
            result.is_ok(),
            "update_source_uri with same uri must return Ok, got: {:?}",
            result
        );

        // Verify the source_uri row is still "p" — no spurious mutation.
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT source_uri FROM episodes WHERE source_id = ?1",
                libsql::params!["x"],
            )
            .await
            .expect("idempotency check query must succeed")
            .next()
            .await
            .expect("idempotency check row_next must succeed")
            .expect("idempotency check row must exist");

        let uri: String = row.get(0).expect("source_uri column in idempotency check");
        assert_eq!(
            uri, "p",
            "source_uri must remain 'p' after idempotent re-run"
        );
    }
}

// ── Memory::recall_by_source_id (G5) ──────────────────────────────────────────

impl Memory {
    /// Direct lookup of episodes matching `source_id`. Returns episodes
    /// ordered by `recorded_at DESC` (newest first).
    ///
    /// # Namespace scope (Vera F11 fold-in)
    ///
    /// When `namespace` is `Some`, the query is restricted to that namespace
    /// only. When `None`, the Memory's default namespace is used; if no
    /// default is set, results span all namespaces (the only path where
    /// cross-namespace results can leak — opt-in via explicit `None` with no
    /// builder-default).
    ///
    /// # Returns
    ///
    /// `Ok(Vec<Episode>)` — empty when no episode rows match the filter.
    /// Never returns Err for "no match"; only DB errors propagate.
    pub async fn recall_by_source_id(
        &self,
        source_id: impl AsRef<str>,
        namespace: Option<Namespace>,
    ) -> Result<Vec<crate::core::schema::Episode>> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::recall_by_source_id requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .to_string(),
            )
        })?;
        let conn = &tg.conn;

        let source_id = source_id.as_ref().to_string();
        let group_filter: Option<String> = namespace
            .or_else(|| self.default_namespace.clone())
            .map(|ns| ns.namespace);

        // `(? IS NULL OR group_id = ?)` evaluates to TRUE when filter is NULL
        // → no namespace scope applied.
        let mut rows = conn
            .query(
                "SELECT id, content, timestamp, source_type, metadata, group_id, \
                        saga_id, sequence_number, recorded_at \
                 FROM episodes \
                 WHERE source_id = ?1 \
                   AND (?2 IS NULL OR group_id = ?2) \
                 ORDER BY recorded_at DESC",
                libsql::params![source_id, group_filter],
            )
            .await
            .map_err(CoreError::Database)?;

        let mut out: Vec<crate::core::schema::Episode> = Vec::new();
        while let Some(row) = rows.next().await.map_err(CoreError::Database)? {
            let id = row.get::<i64>(0).map_err(CoreError::Database)?;
            let content = row.get::<String>(1).map_err(CoreError::Database)?;
            let timestamp_text = row.get::<String>(2).map_err(CoreError::Database)?;
            let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_text)
                .map_err(|e| {
                    MemoryError::Other(format!(
                        "recall_by_source_id: episode.timestamp not RFC3339: {e}"
                    ))
                })?
                .with_timezone(&chrono::Utc);
            let source_type = row.get::<Option<String>>(3).map_err(CoreError::Database)?;
            let metadata_text = row.get::<Option<String>>(4).map_err(CoreError::Database)?;
            let metadata = match metadata_text {
                Some(t) => Some(serde_json::from_str::<serde_json::Value>(&t).map_err(|e| {
                    MemoryError::Other(format!(
                        "recall_by_source_id: episode.metadata not JSON: {e}"
                    ))
                })?),
                None => None,
            };
            let group_id = row.get::<Option<String>>(5).map_err(CoreError::Database)?;
            let saga_id = row.get::<Option<String>>(6).map_err(CoreError::Database)?;
            let sequence_number = row.get::<Option<i64>>(7).map_err(CoreError::Database)?;
            // Episode.content_hash field is currently unsourced (no column on
            // the episodes table — only facts.content_hash exists). Default
            // None until/unless a future migration adds the column to episodes.
            let content_hash: Option<String> = None;
            let recorded_at = row.get::<Option<String>>(8).map_err(CoreError::Database)?;

            out.push(crate::core::schema::Episode {
                id,
                content,
                timestamp,
                source_type,
                metadata,
                group_id,
                saga_id,
                sequence_number,
                content_hash,
                recorded_at,
            });
        }
        Ok(out)
    }
}

// ── G3 Red tests (AC.4 + AC.10) ──────────────────────────────────────────────
//
// These tests INTENTIONALLY FAIL to compile until the Green agent ships:
//   - `Memory::update_episode_metadata(&self, source_id) -> UpdateEpisodeMetadataRequest<'_>`
//   - `UpdateEpisodeMetadataRequest::patch(self, patch: serde_json::Value) -> Self`
//   - `IntoFuture for UpdateEpisodeMetadataRequest` returning `Result<()>`
//
// Placement: inline in facade/mod.rs per Tessa AC.4 + AC.10 + test-strategy T1.
// See: .ai-docs/specs/test-strategy-kremory-v016-substrate-2026-05-29.md §5 AC.4 + AC.10

#[cfg(test)]
mod update_episode_metadata_tests {
    use std::sync::Arc;

    use serde_json::json;

    use crate::{
        core::provider::{MockChatProvider, NullEmbeddingProvider},
        memory::types::Namespace,
    };

    use super::Memory;

    /// Build an in-memory `Memory` instance wired with null providers.
    ///
    /// Mirrors the pattern established in `update_source_uri_tests::make_memory`.
    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });

        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("in-memory Memory construction must not fail")
    }

    /// Seed an episode row with a given `source_id` and `metadata` JSON text
    /// directly via SQL, bypassing the facade ingest path.
    ///
    /// `metadata_json` is `Option<&str>`: pass `Some(r#"{"key":"val"}"#)` for a
    /// JSON payload, or `None` to insert a NULL metadata column (AC.4 edge case).
    ///
    /// Returns the `group_id` used for the episode row.
    async fn seed_episode_with_metadata(
        mem: &Memory,
        source_id: &str,
        source_uri: &str,
        metadata_json: Option<&str>,
        namespace: &Namespace,
    ) -> String {
        use crate::memory::engine_handle::namespace_to_group_id;

        let group_id = namespace_to_group_id(namespace);
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some for in-memory Memory");
        let conn = tg.conn_for_test();

        // Two separate INSERT statements to handle NULL vs non-NULL metadata,
        // because libsql param binding does not cleanly accept Option<&str> as
        // a nullable TEXT without a concrete libsql::Value variant.
        match metadata_json {
            Some(json_text) => {
                conn.execute(
                    "INSERT INTO episodes \
                     (group_id, content, timestamp, recorded_at, source_id, source_uri, metadata) \
                     VALUES (?1, ?2, unixepoch('now'), datetime('now'), ?3, ?4, ?5)",
                    libsql::params![
                        group_id.clone(),
                        "test episode content for metadata patch.",
                        source_id,
                        source_uri,
                        json_text,
                    ],
                )
                .await
                .expect("seed_episode_with_metadata INSERT (with metadata) must succeed");
            }
            None => {
                conn.execute(
                    "INSERT INTO episodes \
                     (group_id, content, timestamp, recorded_at, source_id, source_uri) \
                     VALUES (?1, ?2, unixepoch('now'), datetime('now'), ?3, ?4)",
                    libsql::params![
                        group_id.clone(),
                        "test episode content for metadata patch.",
                        source_id,
                        source_uri,
                    ],
                )
                .await
                .expect("seed_episode_with_metadata INSERT (NULL metadata) must succeed");
            }
        }

        group_id
    }

    // ── AC.4 happy path: shallow merge preserves existing keys ───────────────

    /// AC.4 — shallow merge: patching one key must update that key while
    /// preserving all other top-level keys in the existing metadata object.
    ///
    /// Invariants asserted:
    /// - Updated key (`status`) takes the new value (`accepted`).
    /// - Untouched key (`count`) retains its original value (`1`).
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_episode_metadata` + `UpdateEpisodeMetadataRequest`.
    #[tokio::test]
    async fn update_metadata_shallow_merge_preserves_existing_keys() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-shallow-merge");

        seed_episode_with_metadata(
            &mem,
            "doc-b",
            "path/b",
            Some(r#"{"status":"draft","count":1}"#),
            &ns,
        )
        .await;

        // ── G3 builder call — FAILS TO COMPILE until Green ships the builder ──
        mem.update_episode_metadata("doc-b")
            .patch(json!({"status": "accepted"}))
            .await
            .expect("update_episode_metadata shallow merge must succeed");

        // Read back the metadata column.
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-b"],
            )
            .await
            .expect("post-patch metadata query must succeed")
            .next()
            .await
            .expect("post-patch metadata row_next must succeed")
            .expect("post-patch episode row must exist");

        let metadata_text: String = row.get(0).expect("metadata column");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        assert_eq!(
            metadata,
            json!({"status": "accepted", "count": 1}),
            "shallow merge must update status and preserve count"
        );
    }

    // ── AC.10: arrays REPLACE, not append ────────────────────────────────────

    /// AC.10 — array replace semantics: patching an array-valued key must
    /// REPLACE the existing array entirely, not merge or append to it.
    ///
    /// This is Vera F3 — explicitly documented + tested. Shallow-merge means
    /// top-level array values are overwritten, not concatenated.
    ///
    /// Invariants asserted:
    /// - Result `refs` array contains only the patched entry (`id=b`).
    /// - Original entry (`id=a`) is gone from `refs`.
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_episode_metadata` + `UpdateEpisodeMetadataRequest`.
    #[tokio::test]
    async fn update_metadata_array_replace_not_append() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-array-replace");

        seed_episode_with_metadata(
            &mem,
            "doc-c",
            "path/c",
            Some(r#"{"refs":[{"id":"a","rel":"depends_on"}]}"#),
            &ns,
        )
        .await;

        // ── G3 builder call — FAILS TO COMPILE until Green ships the builder ──
        mem.update_episode_metadata("doc-c")
            .patch(json!({"refs": [{"id": "b", "rel": "supersedes"}]}))
            .await
            .expect("update_episode_metadata array replace must succeed");

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-c"],
            )
            .await
            .expect("post-patch metadata query must succeed")
            .next()
            .await
            .expect("post-patch metadata row_next must succeed")
            .expect("post-patch episode row must exist");

        let metadata_text: String = row.get(0).expect("metadata column");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        let refs = metadata.get("refs").expect("refs key must exist");
        let refs_arr = refs.as_array().expect("refs must be a JSON array");

        assert_eq!(
            refs_arr.len(),
            1,
            "refs array must contain exactly one entry (REPLACE, not append); got: {refs_arr:?}"
        );
        assert_eq!(
            refs_arr[0],
            json!({"id": "b", "rel": "supersedes"}),
            "refs[0] must be the patched entry (id=b), not the original (id=a)"
        );
    }

    // ── AC.4 edge: NULL metadata column creates empty object then merges ──────

    /// AC.4 edge — NULL metadata: patching an episode with a NULL metadata
    /// column must treat existing state as `{}` and merge the patch into it,
    /// producing a result equal to the patch object itself.
    ///
    /// Invariants asserted:
    /// - Result metadata equals `{"first": "value"}`.
    /// - No Err is returned for a valid object patch against a NULL column.
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_episode_metadata` + `UpdateEpisodeMetadataRequest`.
    #[tokio::test]
    async fn update_metadata_null_metadata_creates_and_merges() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-null-metadata");

        // Seed with NULL metadata (no metadata_json argument).
        seed_episode_with_metadata(&mem, "doc-d", "path/d", None, &ns).await;

        // ── G3 builder call — FAILS TO COMPILE until Green ships the builder ──
        mem.update_episode_metadata("doc-d")
            .patch(json!({"first": "value"}))
            .await
            .expect("update_episode_metadata on NULL metadata column must succeed");

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-d"],
            )
            .await
            .expect("post-patch metadata query must succeed")
            .next()
            .await
            .expect("post-patch metadata row_next must succeed")
            .expect("post-patch episode row must exist");

        let metadata_text: String = row
            .get(0)
            .expect("metadata column must be non-NULL after patch");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        assert_eq!(
            metadata,
            json!({"first": "value"}),
            "NULL metadata + patch must produce the patch object itself"
        );
    }

    // ── AC.4 input validation: non-object patch is rejected ──────────────────

    /// AC.4 — input validation: calling `.patch(value)` with a non-Object JSON
    /// value (e.g. a number, string, array) must return `Err` at build/await
    /// time without touching any database row.
    ///
    /// Invariants asserted:
    /// - Call returns `Err` — exact variant is determined by Green agent.
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_episode_metadata` + `UpdateEpisodeMetadataRequest`.
    #[tokio::test]
    async fn update_metadata_non_object_patch_rejects() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-non-object-patch");

        seed_episode_with_metadata(
            &mem,
            "doc-e-reject",
            "path/e-reject",
            Some(r#"{"existing":"value"}"#),
            &ns,
        )
        .await;

        // ── G3 builder call — FAILS TO COMPILE until Green ships the builder ──
        let result = mem
            .update_episode_metadata("doc-e-reject")
            .patch(json!(42))
            .await;

        assert!(
            result.is_err(),
            "update_episode_metadata with non-object patch must return Err, got: {:?}",
            result
        );
    }

    // ── AC.4 error path: missing source_id returns Err ───────────────────────

    /// AC.4 error path — missing source_id: patching a `source_id` that does
    /// not exist in any episode row must return `Err`.
    ///
    /// Mirrors the error-path contract established in `update_source_uri` (G2).
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_episode_metadata` + `UpdateEpisodeMetadataRequest`.
    #[tokio::test]
    async fn update_metadata_missing_source_id_returns_err() {
        let mem = make_memory().await;

        // ── G3 builder call — FAILS TO COMPILE until Green ships the builder ──
        let result = mem
            .update_episode_metadata("nonexistent-source-99999")
            .patch(json!({"any": "patch"}))
            .await;

        assert!(
            result.is_err(),
            "update_episode_metadata with unknown source_id must return Err, got: {:?}",
            result
        );
    }

    // ── AC.4 non-mutation: facts, source_uri, recorded_at are unchanged ───────

    /// AC.4 non-mutation guarantee: `update_episode_metadata` must only mutate
    /// the `metadata` column. All other episode columns (`source_uri`,
    /// `recorded_at`) and the facts table (count) must be unchanged after
    /// a successful patch.
    ///
    /// Invariants asserted:
    /// - `source_uri` on the episode row is unchanged.
    /// - `recorded_at` on the episode row is unchanged.
    /// - `facts` COUNT for this source_id is unchanged (zero facts seeded).
    /// - `metadata` key (`status`) has the new value (`accepted`) — sanity
    ///   guard to confirm the patch was not silently a no-op.
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_episode_metadata` + `UpdateEpisodeMetadataRequest`.
    #[tokio::test]
    async fn update_metadata_does_not_mutate_facts_or_uri() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-non-mutation");

        seed_episode_with_metadata(&mem, "doc-f", "path/f", Some(r#"{"status":"draft"}"#), &ns)
            .await;

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        // Capture pre-update state.
        let pre_row = conn
            .query(
                "SELECT source_uri, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-f"],
            )
            .await
            .expect("pre-patch episode query must succeed")
            .next()
            .await
            .expect("pre-patch episode row_next must succeed")
            .expect("pre-patch episode row must exist");

        let pre_source_uri: String = pre_row.get(0).expect("source_uri column pre");
        let pre_recorded_at: String = pre_row.get(1).expect("recorded_at column pre");

        let pre_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts WHERE source_episode_id IN \
                 (SELECT id FROM episodes WHERE source_id = ?1)",
                libsql::params!["doc-f"],
            )
            .await
            .expect("pre-patch fact count query must succeed")
            .next()
            .await
            .expect("pre-patch fact count row_next must succeed")
            .expect("pre-patch fact count row must exist")
            .get(0)
            .expect("fact count column pre");

        // ── G3 builder call — FAILS TO COMPILE until Green ships the builder ──
        mem.update_episode_metadata("doc-f")
            .patch(json!({"status": "accepted"}))
            .await
            .expect("update_episode_metadata non-mutation test must succeed");

        // Post-update assertions.
        let post_row = conn
            .query(
                "SELECT source_uri, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-f"],
            )
            .await
            .expect("post-patch episode query must succeed")
            .next()
            .await
            .expect("post-patch episode row_next must succeed")
            .expect("post-patch episode row must exist");

        let post_source_uri: String = post_row.get(0).expect("source_uri column post");
        let post_recorded_at: String = post_row.get(1).expect("recorded_at column post");

        assert_eq!(
            post_source_uri, pre_source_uri,
            "source_uri must not be mutated by update_episode_metadata"
        );
        assert_eq!(
            post_recorded_at, pre_recorded_at,
            "recorded_at must not be mutated by update_episode_metadata"
        );

        let post_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts WHERE source_episode_id IN \
                 (SELECT id FROM episodes WHERE source_id = ?1)",
                libsql::params!["doc-f"],
            )
            .await
            .expect("post-patch fact count query must succeed")
            .next()
            .await
            .expect("post-patch fact count row_next must succeed")
            .expect("post-patch fact count row must exist")
            .get(0)
            .expect("fact count column post");

        assert_eq!(
            post_fact_count, pre_fact_count,
            "fact count must not change after update_episode_metadata"
        );

        // Verify metadata was actually updated (sanity: confirm the patch was
        // not silently a no-op that would make the test vacuously pass).
        let meta_row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-f"],
            )
            .await
            .expect("post-patch metadata sanity query must succeed")
            .next()
            .await
            .expect("post-patch metadata sanity row_next must succeed")
            .expect("post-patch metadata sanity row must exist");

        let metadata_text: String = meta_row.get(0).expect("metadata column sanity");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        assert_eq!(
            metadata.get("status").and_then(|v| v.as_str()),
            Some("accepted"),
            "metadata status must be updated to 'accepted' (sanity: patch was not a no-op)"
        );
    }
}

// ── G4 tests — episode_content_warn_threshold ────────────────────────────────
//
// Spec AC.5 — soft warn at configurable content threshold. Asserts the warn
// fires above threshold, does not fire below, can be disabled via `None`, and
// is never enforced (never returns Err on its own).

#[cfg(test)]
mod episode_content_warn_threshold_tests {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory_with_threshold(threshold: Option<usize>) -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .episode_content_warn_threshold(threshold)
            .await
            .expect("in-memory Memory construction must not fail")
    }

    /// AC.5 — warn fires when content.chars().count() exceeds threshold.
    /// Threshold set deliberately small (5 chars) so test content trivially
    /// triggers; production default of 10_000 is irrelevant here.
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn warn_fires_above_threshold() {
        let mem = make_memory_with_threshold(Some(5)).await;
        let ns = Namespace::new("test-g4-warn-fires");

        // 11 chars > threshold 5 → warn must fire.
        // .await may fail downstream (null providers + enrichment), but the
        // warn emits BEFORE submit_episode so logs_contain captures it.
        let _ = mem.remember("hello world").in_namespace(ns).no_wait().await;

        assert!(
            logs_contain("episode content exceeds soft threshold"),
            "warn must fire when content (11 chars) exceeds threshold (5)"
        );
    }

    /// AC.5 — warn does NOT fire when content is at or below threshold.
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn warn_does_not_fire_below_threshold() {
        let mem = make_memory_with_threshold(Some(100)).await;
        let ns = Namespace::new("test-g4-no-warn");

        // 11 chars <= threshold 100 → warn must NOT fire.
        let _ = mem.remember("hello world").in_namespace(ns).no_wait().await;

        assert!(
            !logs_contain("episode content exceeds soft threshold"),
            "warn must NOT fire when content (11 chars) is below threshold (100)"
        );
    }

    /// AC.5 — `None` disables the warning entirely (even for very large content).
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn warn_disabled_when_threshold_is_none() {
        let mem = make_memory_with_threshold(None).await;
        let ns = Namespace::new("test-g4-disabled");

        // 50_000-char content; without threshold the warn must NOT fire.
        let huge = "x".repeat(50_000);
        let _ = mem.remember(huge).in_namespace(ns).no_wait().await;

        assert!(
            !logs_contain("episode content exceeds soft threshold"),
            "warn must NOT fire when threshold is None, regardless of content size"
        );
    }

    /// AC.5 — default threshold via `Memory::open` is `Some(10_000)`.
    /// Validates the documented default + builder default propagation.
    #[tokio::test(flavor = "multi_thread")]
    async fn default_threshold_is_some_10_000() {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

        let mem = Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("default-config Memory must build");

        assert_eq!(
            mem.episode_content_warn_threshold,
            Some(10_000),
            "default threshold must be Some(10_000) per spec"
        );
    }
}

// ── G5 tests — recall_by_source_id ───────────────────────────────────────────
//
// Spec AC.6 — direct lookup of episodes matching source_id, ordered
// recorded_at DESC, optional namespace scope (Vera F11 fold-in).

#[cfg(test)]
mod recall_by_source_id_tests {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    /// Direct seeding helper; bypasses the facade ingest path so the test
    /// only exercises recall_by_source_id. `recorded_at` is set to control
    /// ORDER BY ordering directly.
    async fn seed_episode(
        mem: &Memory,
        source_id: &str,
        ns: &Namespace,
        recorded_at: &str,
        metadata_json: Option<&str>,
    ) {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        conn.execute(
            "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, source_id, source_uri, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            libsql::params![
                "test content",
                "2026-05-29T12:00:00Z",
                "Document",
                metadata_json,
                ns.namespace.as_str(),
                source_id,
                "uri/x",
                recorded_at
            ],
        )
        .await
        .expect("seed insert must succeed");
    }

    /// AC.6 — happy path: returns matching episodes ordered recorded_at DESC.
    #[tokio::test]
    async fn returns_matching_episodes_ordered_recorded_at_desc() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g5-order");

        seed_episode(&mem, "doc-1", &ns, "2026-05-27T10:00:00Z", None).await;
        seed_episode(&mem, "doc-1", &ns, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "doc-1", &ns, "2026-05-28T10:00:00Z", None).await;

        let eps = mem
            .recall_by_source_id("doc-1", Some(ns))
            .await
            .expect("recall must succeed");

        assert_eq!(eps.len(), 3, "must return all 3 episodes for doc-1");
        // recorded_at DESC = newest first.
        assert_eq!(eps[0].recorded_at.as_deref(), Some("2026-05-29T10:00:00Z"));
        assert_eq!(eps[1].recorded_at.as_deref(), Some("2026-05-28T10:00:00Z"));
        assert_eq!(eps[2].recorded_at.as_deref(), Some("2026-05-27T10:00:00Z"));
        // Quinn C6 — proves column-ordinal mapping is live, not a silent zero.
        assert!(
            eps[0].id > 0,
            "Episode.id must be populated (column-ordinal mapping check)"
        );
    }

    /// AC.6 / Quinn C4 — placeholder anchor (do not remove).
    #[allow(dead_code)]
    fn _g5_anchor() {}
}

// ── G6 + G6.b tests — filter_metadata + filter_metadata_in ───────────────────
//
// Spec G6 (key validation + AND-across-filters) + G6.b (OR-within-key +
// multi-value). Covers the Vera F6 path-injection guard and the deferred-
// error pattern surfacing at `.await` time.

#[cfg(test)]
mod filter_metadata_tests {
    use std::sync::Arc;

    use serde_json::json;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::{metadata_matches, validate_metadata_key, Memory};

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    // ── validate_metadata_key — Vera F6 path-injection guard ─────────────────

    #[test]
    fn validate_key_accepts_simple_alphanumeric() {
        assert!(validate_metadata_key("status").is_ok());
        assert!(validate_metadata_key("doc_type").is_ok());
        assert!(validate_metadata_key("key2").is_ok());
        assert!(validate_metadata_key("a").is_ok());
    }

    #[test]
    fn validate_key_rejects_empty() {
        assert!(validate_metadata_key("").is_err());
    }

    #[test]
    fn validate_key_rejects_over_128_chars() {
        let long = "a".repeat(129);
        assert!(validate_metadata_key(&long).is_err());
    }

    #[test]
    fn validate_key_rejects_digit_prefix() {
        assert!(validate_metadata_key("1foo").is_err());
        assert!(validate_metadata_key("0").is_err());
    }

    #[test]
    fn validate_key_rejects_path_metachars() {
        // Vera F6 reject list + Quinn C3 defence-in-depth (`{` and `}`).
        for ch in ['.', '[', ']', '\'', '"', '\\', '$', '*', '{', '}'] {
            let bad = format!("foo{ch}bar");
            assert!(
                validate_metadata_key(&bad).is_err(),
                "key with {ch:?} must be rejected (path-injection guard)"
            );
        }
    }

    // ── metadata_matches — semantics ─────────────────────────────────────────

    #[test]
    fn matches_empty_filters_passes_when_meta_is_some() {
        // No filters at all → caller of helper shouldn't invoke; defensive
        // check: even with Some(meta), empty filter set must short-circuit to
        // true via the for-loops being empty.
        let meta = json!({"k": "v"});
        assert!(metadata_matches(Some(&meta), &[], &[]));
    }

    #[test]
    fn matches_returns_false_when_meta_is_none_and_any_filter() {
        let filters = vec![("k".to_string(), json!("v"))];
        assert!(!metadata_matches(None, &filters, &[]));
    }

    #[test]
    fn matches_and_across_filters_all_must_pass() {
        let meta = json!({"a": 1, "b": 2});
        assert!(metadata_matches(
            Some(&meta),
            &[("a".to_string(), json!(1)), ("b".to_string(), json!(2)),],
            &[]
        ));
        // Mismatch on one → false.
        assert!(!metadata_matches(
            Some(&meta),
            &[("a".to_string(), json!(1)), ("b".to_string(), json!(99)),],
            &[]
        ));
    }

    #[test]
    fn matches_in_or_within_key() {
        let meta = json!({"status": "accepted"});
        // status ∈ {"accepted", "approved"} → match.
        assert!(metadata_matches(
            Some(&meta),
            &[],
            &[(
                "status".to_string(),
                vec![json!("accepted"), json!("approved")],
            )]
        ));
        // status ∈ {"draft"} → no match.
        assert!(!metadata_matches(
            Some(&meta),
            &[],
            &[("status".to_string(), vec![json!("draft")])]
        ));
    }

    #[test]
    fn matches_missing_required_key_fails() {
        let meta = json!({"a": 1});
        assert!(!metadata_matches(
            Some(&meta),
            &[("b".to_string(), json!(2))],
            &[]
        ));
    }

    // ── End-to-end deferred-error path via .await ────────────────────────────

    #[tokio::test]
    async fn filter_metadata_invalid_key_errs_at_await() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g6-bad-key");
        let result = mem
            .recall("q")
            .in_namespace(ns)
            .filter_metadata("bad.key", json!("v"))
            .raw()
            .await;
        assert!(
            result.is_err(),
            "filter_metadata with path-metachar key must err at .await, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn filter_metadata_in_empty_values_errs_at_await() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g6-empty-vals");
        let result = mem
            .recall("q")
            .in_namespace(ns)
            .filter_metadata_in("status", &[])
            .raw()
            .await;
        assert!(
            result.is_err(),
            "filter_metadata_in with empty values slice must err at .await, got: {result:?}"
        );
    }

    /// G8 cap test runner moved to separate `forget_by_source_id_tests` mod
    /// below; this anchor keeps the filter_metadata module focused on G6/G6.b.
    #[allow(dead_code)]
    fn _g6_anchor() {}

    #[tokio::test]
    async fn filter_metadata_first_error_wins() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g6-first-err");
        // First call: invalid (digit prefix). Second call: empty key.
        let result = mem
            .recall("q")
            .in_namespace(ns)
            .filter_metadata("1bad", json!("a"))
            .filter_metadata("", json!("b"))
            .raw()
            .await;
        let err = result.expect_err("must err");
        assert!(
            err.to_string().contains("1bad"),
            "first error must surface (the digit-prefix one); got: {err}"
        );
    }
}

#[cfg(test)]
mod recall_by_source_id_tests_part2 {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    async fn seed_episode(
        mem: &Memory,
        source_id: &str,
        ns: &Namespace,
        recorded_at: &str,
        metadata_json: Option<&str>,
    ) {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        conn.execute(
            "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, source_id, source_uri, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            libsql::params![
                "test content",
                "2026-05-29T12:00:00Z",
                "Document",
                metadata_json,
                ns.namespace.as_str(),
                source_id,
                "uri/x",
                recorded_at
            ],
        )
        .await
        .expect("seed insert must succeed");
    }

    /// AC.6 / Quinn C4 — when no `namespace` is passed and Memory has a default
    /// namespace set, results must be scoped to that default (not span all NS).
    #[tokio::test]
    async fn no_namespace_with_memory_default_scopes_to_default() {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        let ns_default = Namespace::new("test-g5-default-scope");
        let ns_other = Namespace::new("test-g5-other-scope");

        let mem = Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .default_namespace(ns_default.clone())
            .await
            .expect("Memory with default namespace must build");

        seed_episode(&mem, "doc-x", &ns_default, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "doc-x", &ns_other, "2026-05-29T11:00:00Z", None).await;

        let eps = mem
            .recall_by_source_id("doc-x", None)
            .await
            .expect("recall with no explicit namespace must succeed");

        assert_eq!(
            eps.len(),
            1,
            "namespace=None + Memory default set must scope to the default ns"
        );
        assert_eq!(
            eps[0].group_id.as_deref(),
            Some("test-g5-default-scope"),
            "returned episode must come from the default namespace"
        );
    }

    /// AC.6 — missing source_id returns empty Vec, not Err.
    #[tokio::test]
    async fn missing_source_id_returns_empty_vec() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g5-empty");

        let eps = mem
            .recall_by_source_id("does-not-exist-xyz", Some(ns))
            .await
            .expect("recall must succeed even with no matches");

        assert!(eps.is_empty(), "no matches must return empty Vec, not Err");
    }

    /// AC.6 — Vera F11: namespace filter restricts results when Some.
    #[tokio::test]
    async fn namespace_filter_restricts_results() {
        let mem = make_memory().await;
        let ns_a = Namespace::new("test-g5-ns-a");
        let ns_b = Namespace::new("test-g5-ns-b");

        seed_episode(&mem, "shared-doc", &ns_a, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "shared-doc", &ns_b, "2026-05-29T11:00:00Z", None).await;

        // Filter to ns_a only.
        let eps_a = mem
            .recall_by_source_id("shared-doc", Some(ns_a.clone()))
            .await
            .expect("recall ns_a must succeed");
        assert_eq!(eps_a.len(), 1, "ns_a filter must return only the ns_a row");
        assert_eq!(eps_a[0].group_id.as_deref(), Some("test-g5-ns-a"));

        // Filter to ns_b only.
        let eps_b = mem
            .recall_by_source_id("shared-doc", Some(ns_b))
            .await
            .expect("recall ns_b must succeed");
        assert_eq!(eps_b.len(), 1, "ns_b filter must return only the ns_b row");
        assert_eq!(eps_b[0].group_id.as_deref(), Some("test-g5-ns-b"));
    }

    /// AC.6 — Vera F11: when namespace=None and no Memory default, returns
    /// episodes across ALL namespaces (the only opt-in cross-namespace leak path).
    #[tokio::test]
    async fn no_namespace_and_no_default_spans_all_namespaces() {
        let mem = make_memory().await;
        let ns_a = Namespace::new("test-g5-span-a");
        let ns_b = Namespace::new("test-g5-span-b");

        seed_episode(&mem, "doc-span", &ns_a, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "doc-span", &ns_b, "2026-05-29T11:00:00Z", None).await;

        let eps = mem
            .recall_by_source_id("doc-span", None)
            .await
            .expect("recall with no namespace must succeed");

        assert_eq!(
            eps.len(),
            2,
            "namespace=None + no Memory default must span all namespaces"
        );
    }

    /// AC.6 — metadata JSON is parsed into serde_json::Value, NULL → None.
    #[tokio::test]
    async fn metadata_parsing_round_trips() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g5-meta");

        seed_episode(
            &mem,
            "doc-meta",
            &ns,
            "2026-05-29T10:00:00Z",
            Some(r#"{"k":"v","n":42}"#),
        )
        .await;
        seed_episode(&mem, "doc-meta-null", &ns, "2026-05-29T11:00:00Z", None).await;

        let with_meta = mem
            .recall_by_source_id("doc-meta", Some(ns.clone()))
            .await
            .expect("recall must succeed");
        assert_eq!(with_meta.len(), 1);
        let meta = with_meta[0].metadata.as_ref().expect("metadata Some");
        assert_eq!(meta.get("k").and_then(|v| v.as_str()), Some("v"));
        assert_eq!(meta.get("n").and_then(|v| v.as_i64()), Some(42));

        let null_meta = mem
            .recall_by_source_id("doc-meta-null", Some(ns))
            .await
            .expect("recall must succeed");
        assert_eq!(null_meta.len(), 1);
        assert!(null_meta[0].metadata.is_none(), "NULL metadata → None");
    }
}

// ── G8 tests — ForgetRequest::by_source_id + Vera F17 ────────────────────────
//
// Spec G8 — narrow forget to entities derived from episodes matching
// source_id. Vera F17: entities referenced by ANY episode outside the
// matched source_id must be preserved (cross-source data-loss guard).

#[cfg(test)]
mod forget_by_source_id_tests {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    /// Seed: episode + entity + episodic_edge. Returns the inserted
    /// episode_id (SQLite AUTOINCREMENT).
    async fn seed_link(mem: &Memory, source_id: &str, ns: &Namespace, entity_id: &str) -> i64 {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;

        // Insert entity (idempotent via INSERT OR IGNORE on PRIMARY KEY).
        conn.execute(
            "INSERT OR IGNORE INTO entities (id, label, properties, recorded_at, group_id) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                entity_id,
                entity_id,
                "{}",
                "2026-05-29T12:00:00Z",
                ns.namespace.as_str()
            ],
        )
        .await
        .expect("entity seed must succeed");

        // Insert episode with explicit source_id.
        conn.execute(
            "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, source_id, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            libsql::params![
                "seed",
                "2026-05-29T12:00:00Z",
                "Document",
                None::<String>,
                ns.namespace.as_str(),
                source_id,
                "2026-05-29T12:00:00Z"
            ],
        )
        .await
        .expect("episode seed must succeed");

        // Recover the inserted episode id.
        let mut rows = conn
            .query(
                "SELECT id FROM episodes WHERE source_id = ?1 AND group_id = ?2 \
                 ORDER BY id DESC LIMIT 1",
                libsql::params![source_id, ns.namespace.as_str()],
            )
            .await
            .expect("episode id lookup must succeed");
        let episode_id: i64 = rows
            .next()
            .await
            .expect("episode id row_next must succeed")
            .expect("episode id row must exist")
            .get::<i64>(0)
            .expect("episode id column");

        // Wire the episodic_edge. Migration 006 enforces composite FK
        // (entity_id, entity_group_id) → entities(id, group_id), so we must
        // bind entity_group_id explicitly.
        conn.execute(
            "INSERT INTO episodic_edges (episode_id, entity_id, entity_group_id, role, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                episode_id,
                entity_id,
                ns.namespace.as_str(),
                "mentioned",
                "2026-05-29T12:00:00Z"
            ],
        )
        .await
        .expect("episodic_edge seed must succeed");

        episode_id
    }

    async fn entity_exists(mem: &Memory, entity_id: &str) -> bool {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        let mut rows = conn
            .query(
                "SELECT 1 FROM entities WHERE id = ?1",
                libsql::params![entity_id],
            )
            .await
            .expect("entity exist check must succeed");
        rows.next().await.expect("row_next must succeed").is_some()
    }

    /// AC.8 — happy path: entity referenced only by source_A is forgotten
    /// when forget(by_source_id="A") runs; entity referenced only by source_B
    /// is preserved.
    #[tokio::test]
    async fn by_source_id_deletes_only_matched_unique_entities() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-unique");
        seed_link(&mem, "doc-A", &ns, "entity-alpha").await;
        seed_link(&mem, "doc-B", &ns, "entity-beta").await;

        let deleted = mem
            .forget()
            .in_namespace(ns.clone())
            .by_source_id("doc-A")
            .execute()
            .await
            .expect("forget by_source_id must succeed");

        assert_eq!(deleted, 1, "only entity-alpha must be deleted");
        assert!(
            !entity_exists(&mem, "entity-alpha").await,
            "alpha must be gone"
        );
        assert!(entity_exists(&mem, "entity-beta").await, "beta must remain");
    }

    /// AC.8 / Vera F17 — shared-entity preservation: entity referenced by
    /// BOTH source_A and source_B must NOT be deleted when only source_A is
    /// forgotten.
    #[tokio::test]
    async fn by_source_id_preserves_shared_entity_vera_f17() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-shared");
        // Same entity linked from two distinct sources.
        seed_link(&mem, "doc-A", &ns, "entity-shared").await;
        seed_link(&mem, "doc-B", &ns, "entity-shared").await;

        let deleted = mem
            .forget()
            .in_namespace(ns.clone())
            .by_source_id("doc-A")
            .execute()
            .await
            .expect("forget by_source_id must succeed");

        assert_eq!(
            deleted, 0,
            "shared entity must NOT be deleted (Vera F17 preservation)"
        );
        assert!(
            entity_exists(&mem, "entity-shared").await,
            "shared entity must remain (still referenced by doc-B)"
        );
    }

    async fn episode_count_for(mem: &Memory, source_id: &str, ns: &Namespace) -> i64 {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM episodes WHERE source_id = ?1 AND group_id = ?2",
                libsql::params![source_id, ns.namespace.as_str()],
            )
            .await
            .expect("episode count must succeed");
        rows.next()
            .await
            .expect("row_next")
            .expect("row")
            .get::<i64>(0)
            .expect("count column")
    }

    /// AC.8 / Quinn C3 — episode rows for the matched source_id are
    /// removed (not just entities). Spec: "only the episode row(s) and
    /// edges exclusively owned by this source_id are removed."
    #[tokio::test]
    async fn by_source_id_deletes_matched_episode_rows() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-episodes");
        seed_link(&mem, "doc-A", &ns, "entity-foo").await;
        seed_link(&mem, "doc-B", &ns, "entity-bar").await;

        assert_eq!(episode_count_for(&mem, "doc-A", &ns).await, 1);
        assert_eq!(episode_count_for(&mem, "doc-B", &ns).await, 1);

        mem.forget()
            .in_namespace(ns.clone())
            .by_source_id("doc-A")
            .execute()
            .await
            .expect("forget must succeed");

        assert_eq!(
            episode_count_for(&mem, "doc-A", &ns).await,
            0,
            "doc-A episodes must be deleted (Quinn C3)"
        );
        assert_eq!(
            episode_count_for(&mem, "doc-B", &ns).await,
            1,
            "doc-B episodes must be preserved (out of scope)"
        );
    }

    /// AC.8 / Quinn C4 — AppendOnly policy gate still fires when
    /// `by_source_id` is set. Narrowing scope does not weaken the gate.
    #[tokio::test]
    async fn by_source_id_appendonly_blocks_forget() {
        use crate::memory::types::{ImmutabilityLevel, NamespacePolicy};

        let mem = make_memory().await;
        let ns_inner = Namespace::new("test-g8-appendonly");

        // Register AppendOnly policy. AppendOnly mandates forgettable=false
        // AND dream_eligible=false (coherence check rejects mutating ops).
        let policy = NamespacePolicy {
            immutability: ImmutabilityLevel::AppendOnly,
            forgettable: false,
            dream_eligible: false,
            ..NamespacePolicy::default()
        };
        let ns_with_policy = ns_inner.clone().with_policy(policy).expect("policy");
        mem.register_namespace(ns_with_policy)
            .await
            .expect("register_namespace must succeed");

        // Seed an episode so the source_id path has a candidate (even
        // though the policy gate fires first and the entity stays).
        seed_link(&mem, "doc-A", &ns_inner, "entity-zeta").await;

        let result = mem
            .forget()
            .in_namespace(ns_inner)
            .by_source_id("doc-A")
            .execute()
            .await;

        let err = result.expect_err("AppendOnly must block forget by_source_id");
        let msg = err.to_string();
        assert!(
            msg.contains("policy") || msg.contains("Policy") || msg.contains("AppendOnly"),
            "error must reference policy violation; got: {msg}"
        );
    }

    /// AC.8 — unknown source_id returns 0, no Err.
    #[tokio::test]
    async fn by_source_id_no_match_returns_zero() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-empty");
        seed_link(&mem, "doc-A", &ns, "entity-x").await;

        let deleted = mem
            .forget()
            .in_namespace(ns.clone())
            .by_source_id("does-not-exist-zzz")
            .execute()
            .await
            .expect("forget by_source_id must succeed even with no matches");

        assert_eq!(deleted, 0, "no match must return 0, not Err");
        assert!(
            entity_exists(&mem, "entity-x").await,
            "non-matched entity must remain"
        );
    }
}
