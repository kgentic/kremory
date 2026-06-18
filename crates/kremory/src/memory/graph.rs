//! `GraphHandle` trait — the storage-backend boundary rqlm sits behind.
//!
//! ## Why a trait
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface" + the canonical Zep
//! pattern (verified context7 2026-05-13): rqlm orchestrates over an
//! opaque graph handle that downstream SDK consumers implement against
//! their own concrete graph. Generics on the
//! public fns would propagate `<L: ChatProvider, Emb: EmbeddingProvider>`
//! through every consumer signature — that's a stability hazard for the
//! SDK contract. A trait-object behind `&dyn GraphHandle` erases both
//! generics at the boundary.
//!
//! ## API shape is rqlm-shaped, not rqlc-shaped
//!
//! The trait method signatures speak in rqlm vocabulary
//! (`Namespace`, `IngestResult`, `RetrievedContext`) — NOT
//! kremory::core's `Engine<L, Emb>::ingest_document(source, title, text)`
//! signature. This is deliberate: each consumer translates its
//! concrete-graph API into the trait, so a future backend swap (libsql
//! → kuzu → Neo4j) doesn't break rqlm's public surface.
//!
//! ## D.6.4 extension — canonical definition (supersedes D.0a)
//!
//! Per ADR rqlm-async-event-handle-api-design-2026-05-19 §4.9:
//! all methods are required — NO defaults. Every impl (e.g. StubGraphHandle
//! in tests, or a consumer-supplied concrete handle) must explicitly implement all methods.
//! Compile failure is the enforcement mechanism.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::core::error::IngestStatus;

use super::{
    events::EnrichmentEventSink,
    types::{
        BatchStatus, CancelOutcome, DreamHandle, DreamOpts, DreamPhaseResult, DreamStatus,
        EpisodeCommit, Namespace, RetrievedContext, SearchOpts, SourceRef, StructuredFact,
        SubmitOpts,
    },
    ChatProvider, Result,
};

/// Bundled parameters for [`GraphHandle::graph_ingest_episode`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). Borrows live for the
/// duration of a single ingest call; callers construct a fresh value per call.
pub struct GraphIngestEpisodeParams<'a> {
    pub namespace: &'a Namespace,
    pub source_ref: &'a SourceRef,
    pub content: &'a str,
    pub structured_facts: &'a [StructuredFact],
    pub provider: Arc<dyn ChatProvider>,
    pub batch_id: Option<String>,
    pub opts: SubmitOpts,
    pub sink: Option<Arc<dyn EnrichmentEventSink>>,
}

/// Storage-backend boundary for rqlm. Consumers implement this trait
/// against their concrete graph storage; rqlm orchestrates over the
/// trait object.
///
/// All methods are `async` via `async_trait` so the trait is object-safe
/// (`Box<dyn GraphHandle>` / `&dyn GraphHandle` work). The trait extends
/// `Send + Sync` so it can be cloned into `tokio::spawn` closures by
/// downstream consumers.
///
/// **All methods are required — no defaults.** This is the compiler-enforced
/// shape-stability constraint per ADR §4.9 (cycle-1 Vera finding #1 resolved).
#[async_trait]
pub trait GraphHandle: Send + Sync {
    // ── Phase 1 + 2: episode ingest ──────────────────────────────────────────

    /// Ingest one episode.
    ///
    /// Phase 1 (store + embed) commits synchronously and returns before
    /// Phase 2 (LLM enrich) if `opts.run_in_background = true`.
    /// Episode is searchable from the moment this fn returns.
    ///
    /// `sink` receives Phase 2 events if `enrich_per_episode = true`.
    /// `batch_id` groups this episode with others under a single caller-set string.
    async fn graph_ingest_episode(
        &self,
        params: GraphIngestEpisodeParams<'_>,
    ) -> Result<EpisodeCommit>;

    /// Query Phase 2 status for a `run_id` returned by `graph_ingest_episode`.
    async fn graph_ingest_status(&self, run_id: Uuid) -> Result<IngestStatus>;

    /// Cancel an in-flight Phase 2 or Phase 3 run.
    async fn graph_cancel(&self, run_id: Uuid) -> Result<CancelOutcome>;

    // ── Phase 3: batch consolidation (dream) ─────────────────────────────────

    /// Submit a dream-phase batch consolidation over the scope.
    /// Returns immediately with `DreamHandle`.
    ///
    /// Idempotent: if a run is already active for `(namespace, thread, batch_id)`,
    /// returns the existing `DreamHandle` without starting a new run.
    /// Parameter mismatch on existing key → `tracing::warn!` (not error).
    // Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
    #[allow(clippy::too_many_arguments)]
    async fn graph_submit_dream(
        &self,
        namespace: &Namespace,
        provider: Arc<dyn ChatProvider>,
        batch_id: Option<String>,
        opts: DreamOpts,
        sink: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Result<DreamHandle>;

    /// Query Phase 3 status for a `DreamHandle` run_id.
    async fn graph_dream_status(&self, run_id: Uuid) -> Result<DreamStatus>;

    // ── Batch / policy support ───────────────────────────────────────────────

    /// Track whether all Phase 2 runs for a `batch_id` are terminal.
    /// Returns `BatchStatus` with explicit completed/skipped/failed counts.
    async fn graph_batch_status(&self, batch_id: &str) -> Result<BatchStatus>;

    /// When did the last dream phase complete for this scope? `None` if never run.
    async fn graph_last_consolidated_at(
        &self,
        namespace: &Namespace,
    ) -> Result<Option<DateTime<Utc>>>;

    /// Count of episodes committed since the last dream phase completed.
    async fn graph_episodes_since_last_dream(&self, namespace: &Namespace) -> Result<usize>;

    /// `true` if a dream phase is currently running for this namespace.
    async fn graph_is_consolidating(&self, namespace: &Namespace) -> Result<bool>;

    // ── Search (from D.0a — signature unchanged) ─────────────────────────────

    /// Hybrid retrieval over the scoped graph.
    async fn graph_search(
        &self,
        namespace: &Namespace,
        query: &str,
        opts: &SearchOpts,
    ) -> Result<Vec<RetrievedContext>>;

    // ── Legacy consolidation (from D.0a — retained for backwards compat) ─────
    //
    // New code should use `graph_submit_dream`. This method runs dream
    // synchronously, blocking until complete. Backing implementation for
    // the `run_dream_phase()` backwards-compat wrapper in mod.rs.
    async fn graph_run_consolidation(
        &self,
        namespace: &Namespace,
        provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult>;

    // ── Dream pass sync (Phase C — v0.1.1) ───────────────────────────────────

    /// Run a synchronous dream pass with the given options.
    ///
    /// Serialised internally (at-most-one concurrent pass per engine instance).
    /// Pass 0 (type discovery) and Pass 2 (ghost episode retry) are stubbed in
    /// Phase C and wired in Phase D/E.
    ///
    /// Returns a [`DreamSummary`] with pass statistics. All counts are zero in
    /// the Phase C stub.
    ///
    /// # ADR reference
    ///
    /// ADR-045 §3; Phase C DoD C1 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    async fn graph_run_dream_pass_sync(
        &self,
        opts: crate::core::ingest::DreamPassOpts,
    ) -> Result<crate::facade::DreamSummary>;

    /// Return episode IDs where Phase 1 succeeded but Phase 2 produced no facts.
    ///
    /// An optional `group_id` restricts the query to one namespace/thread.
    /// `None` returns ghost episodes across all namespaces.
    ///
    /// # ADR reference
    ///
    /// ADR-045 §3; Phase C DoD C4.
    async fn graph_ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>>;

    /// Pin an entity as `ConsumerPinned`, protecting it from dream reclassification.
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` on the entity row.
    ///
    /// # ADR reference
    ///
    /// ADR-045 §3; Phase C DoD C5.
    async fn graph_assert_entity_type(
        &self,
        entity_id: &str,
        entity_type_id: u32,
        group_id: Option<&str>,
    ) -> Result<()>;
}
