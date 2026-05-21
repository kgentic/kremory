//! `GraphHandle` trait — the storage-backend boundary rqlm sits behind.
//!
//! ## Why a trait
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface" + the canonical Zep
//! pattern (verified context7 2026-05-13): rqlm orchestrates over an
//! opaque graph handle that consumers (the host application, paying SDK customers,
//! aidocs) implement against their own concrete graph. Generics on the
//! 4 public fns would propagate `<L: ChatProvider, Emb: EmbeddingProvider>`
//! through every consumer signature — that's a stability hazard for the
//! SDK contract. A trait-object behind `&dyn GraphHandle` erases both
//! generics at the boundary.
//!
//! ## API shape is rqlm-shaped, not rqlc-shaped
//!
//! The trait method signatures speak in rqlm vocabulary
//! (`WorkspaceScope`, `IngestResult`, `RetrievedContext`) — NOT
//! rqlc's `RqlGraph<L, Emb>::ingest_document(source, title, text)`
//! signature. This is deliberate: each consumer translates its
//! concrete-graph API into the trait, so a future backend swap (libsql
//! → kuzu → Neo4j) doesn't break rqlm's public surface.
//!
//! ## Greenfield D.2b-trait
//!
//! This module ships the trait definition. The rqlm public fns
//! (`ingest_episode` / `run_dream_phase` / `search`) take
//! `&dyn GraphHandle` instead of the D.1b `_graph: &()` placeholder.
//! Bodies still return `Unimplemented` — wiring the trait calls is
//! the D.2b-impl slice. The signature change is what locks the API.

use async_trait::async_trait;
use std::sync::Arc;

use super::{
    ChatProvider, DreamPhaseResult, IngestResult, RetrievedContext, Result, SearchOpts, SourceRef,
    StructuredFact, WorkspaceScope,
};

/// Storage-backend boundary for rqlm. Consumers implement this trait
/// against their concrete graph storage; rqlm orchestrates over the
/// trait object.
///
/// All methods are `async` via `async_trait` so the trait is object-safe
/// (`Box<dyn GraphHandle>` / `&dyn GraphHandle` work). The trait extends
/// `Send + Sync` so it can be cloned into `tokio::spawn` closures by
/// downstream consumers.
#[async_trait]
pub trait GraphHandle: Send + Sync {
    /// Ingest one episode into the scoped graph.
    ///
    /// Implementations are responsible for:
    /// - LLM entity + edge extraction (via `provider`) per Graphiti's
    ///   `add_episode` cycle.
    /// - Temporal validity inference (`valid_at` / `invalid_at` on facts).
    /// - Entity deduplication against the scoped existing graph.
    /// - Per-episode fact invalidation when new facts contradict prior.
    /// - Honouring `structured_facts` as caller-pinned high-confidence
    ///   data alongside LLM extraction.
    ///
    /// Returns counts of new entities + edges added + facts invalidated
    /// + duration of the cycle.
    async fn graph_ingest_episode(
        &self,
        scope: &WorkspaceScope,
        source_ref: &SourceRef,
        content: &str,
        structured_facts: &[StructuredFact],
        provider: Arc<dyn ChatProvider>,
    ) -> Result<IngestResult>;

    /// Hybrid retrieval over the scoped graph. Implementations apply
    /// rqlc's underlying primitives (FTS5 + vector + graph traversal)
    /// then rerank per rqlm's opinionated defaults. Returns top results
    /// ordered by score descending; caller renders via `context_block`.
    ///
    /// `opts.limit` defaults to 10 when None; `opts.as_of` switches the
    /// retrieval to bi-temporal mode (return rows valid at the supplied
    /// instant); `opts.source_kind` filters by SourceKind variant.
    async fn graph_search(
        &self,
        scope: &WorkspaceScope,
        query: &str,
        opts: &SearchOpts,
    ) -> Result<Vec<RetrievedContext>>;

    /// Run the batch consolidation cycle over the scope. Composes
    /// rqlc's batch primitives: community recompute + cross-meeting
    /// distillation + supersession sweep + stale-fact archival. NOT a
    /// daemon — consumer-triggered (the host application fires on meeting-end;
    /// aidocs on doc-batch-flush).
    ///
    /// Idempotent within a scope/version: a re-run that finds nothing
    /// to consolidate returns zero counts cleanly.
    async fn graph_run_consolidation(
        &self,
        scope: &WorkspaceScope,
        provider: Arc<dyn ChatProvider>,
    ) -> Result<DreamPhaseResult>;
}
