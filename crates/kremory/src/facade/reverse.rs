//! Consumer-facing reversal + INSPECT builders (arch-spec
//! `reversible-graph-mutations-arch-spec-2026-07-10.md` §3.1 + §3 "Inspect
//! surface") — the `Memory` surface for the Stage-1 reversal primitives AND the
//! read-only inspect surface.
//!
//! The three FIX builders (`Unmerge` / `RestoreArchived` / `Unsupersede`) are each
//! a `#[must_use]` handle that must call `.execute()` (mirrors `SupersedeRequest` /
//! `ForgetRequest`'s "no accidental `.await`" discipline for a mutating op), each
//! delegating to the deterministic substrate reversal
//! (`core::dream::provenance::reversal`) and returning the HONEST outcome type
//! (§3.1) — never a bare `Applied`. The two SEE builders (`MutationHistory` /
//! `ListMutations`) are read-only and `IntoFuture` (await directly, like
//! `RecallRequest`), returning `Vec<MutationRecord>`. The napi mirror is
//! mechanical (Tier-0, not built here).

use std::future::IntoFuture;

use super::*;

use crate::core::dream::provenance::{inspect, MutationKind, MutationRecord};
use crate::memory::engine_handle::namespace_to_group_id;

// ── UnmergeRequest ─────────────────────────────────────────────────────────

/// Reverse a prior entity-merge by its `mutation_id` (§4.2). Obtain via
/// `mem.unmerge(mutation_id)`.
///
/// Fully restores the loser entity, its facts, its episodic edges, and the
/// keeper's overwritten `access_count` / `ner_confidence`, then records a merge
/// NOGOOD so the next `dream()` will NOT re-merge the split pair (§6.2).
/// Idempotent: a second `.execute()` returns `already_undone = true`.
///
/// Must call `.execute()` — this is a mutating operation.
pub struct UnmergeRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) mutation_id: i64,
}

impl UnmergeRequest<'_> {
    /// Execute the reversal.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` was constructed without a `TemporalGraph`
    /// (test-stub path), if `mutation_id` names no `entity_merge` log row, or if
    /// the snapshot fails to deserialize (parse-loudly — an un-reversible row is
    /// a hard error, never a silent no-op).
    pub async fn execute(self) -> Result<crate::core::dream::provenance::UnmergeOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::unmerge requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::reversal::unmerge(tg, self.mutation_id)
            .await
            .map_err(MemoryError::Core)
    }
}

// ── RestoreArchivedRequest ─────────────────────────────────────────────────

/// Restore a fact previously moved to `facts_archive` (P2 archival) back into
/// `facts` (§3.2 / §4.4). Obtain via `mem.restore_archived_fact(archived_fact_id)`.
///
/// Idempotent: if the fact is already live, returns `already_live = true` and
/// writes nothing. Must call `.execute()`.
pub struct RestoreArchivedRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) archived_fact_id: i64,
}

impl RestoreArchivedRequest<'_> {
    /// Execute the restore.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, or if
    /// `archived_fact_id` names no `facts_archive` row.
    pub async fn execute(self) -> Result<crate::core::dream::provenance::RestoreArchivedOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::restore_archived_fact requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::reversal::restore_archived_fact(tg, self.archived_fact_id)
            .await
            .map_err(MemoryError::Core)
    }
}

// ── UnsupersedeRequest ─────────────────────────────────────────────────────

/// Clear a supersession bound (`valid_to` / `expired_at`) set by
/// `supersede(...)`, re-opening the fact as currently-true (§4.5). Obtain via
/// `mem.unsupersede(fact_id)`.
///
/// Idempotent: a fact with no bound set returns `NotSuperseded` (an honest
/// no-op). Must call `.execute()`.
pub struct UnsupersedeRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) fact_id: i64,
}

impl UnsupersedeRequest<'_> {
    /// Execute the un-supersede.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, or if `fact_id` names
    /// no fact.
    pub async fn execute(self) -> Result<crate::core::dream::provenance::UnsupersedeOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::unsupersede requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::reversal::unsupersede(tg, self.fact_id)
            .await
            .map_err(MemoryError::Core)
    }
}

// ── MutationHistoryRequest (§3 inspect surface — SEE) ───────────────────────

/// Inspect the mutations that touched one entity (§3 "Inspect surface"). Obtain
/// via `mem.mutation_history(entity_id)`. Read-only — `.await` it for a
/// newest-first `Vec<MutationRecord>` (includes already-undone mutations).
///
/// An entity id is namespace-scoped: set `.in_namespace(ns)` or a
/// `default_namespace` on the builder, else `.await` returns
/// `MemoryError::MissingNamespace`.
#[must_use = "MutationHistoryRequest must be .await-ed"]
pub struct MutationHistoryRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) entity_id: String,
    pub(super) namespace: Option<Namespace>,
}

impl<'a> MutationHistoryRequest<'a> {
    /// Scope the lookup to `ns` (overrides the `Memory` default namespace).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    async fn execute(self) -> Result<Vec<MutationRecord>> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::mutation_history requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let group_id = namespace_to_group_id(&ns);
        inspect::mutation_history(tg, &self.entity_id, &group_id)
            .await
            .map_err(MemoryError::Core)
    }
}

impl<'a> IntoFuture for MutationHistoryRequest<'a> {
    type Output = Result<Vec<MutationRecord>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

// ── ListMutationsRequest (§3 inspect surface — SEE) ─────────────────────────

/// List logged graph mutations, newest-first (§3 "Inspect surface"). Obtain via
/// `mem.list_mutations()`. Read-only — `.await` it for a `Vec<MutationRecord>`.
///
/// Knobs (all optional): `.in_namespace(ns)` scopes to one namespace (else the
/// `default_namespace`, else ALL namespaces); `.kind(k)` restricts the mutation
/// kind; `.since(ts)` sets a `created_at` lower bound; `.include_undone(true)`
/// adds already-reversed mutations (default: LIVE / still-reversible only).
#[must_use = "ListMutationsRequest must be .await-ed"]
pub struct ListMutationsRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) namespace: Option<Namespace>,
    pub(super) kind: Option<MutationKind>,
    pub(super) since: Option<DateTime<Utc>>,
    pub(super) include_undone: bool,
}

impl<'a> ListMutationsRequest<'a> {
    /// Scope to `ns` (else the `Memory` default namespace, else all namespaces).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Restrict to a single mutation kind.
    pub fn kind(mut self, kind: MutationKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Only mutations at/after `ts` (`created_at >=`, RFC3339-compared).
    pub fn since(mut self, ts: DateTime<Utc>) -> Self {
        self.since = Some(ts);
        self
    }

    /// Include already-undone mutations (default `false` — live only).
    pub fn include_undone(mut self, yes: bool) -> Self {
        self.include_undone = yes;
        self
    }

    async fn execute(self) -> Result<Vec<MutationRecord>> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::list_mutations requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        // Namespace is OPTIONAL for listing: explicit override → default → None
        // (all namespaces). No error path — an unscoped list is a valid admin view.
        let group_id = self
            .namespace
            .or_else(|| self.memory.default_namespace.clone())
            .map(|ns| namespace_to_group_id(&ns));
        let filter = inspect::MutationFilter {
            group_id,
            kind: self.kind,
            since: self.since.map(|ts| ts.to_rfc3339()),
            include_undone: self.include_undone,
        };
        inspect::list_mutations(tg, filter)
            .await
            .map_err(MemoryError::Core)
    }
}

impl<'a> IntoFuture for ListMutationsRequest<'a> {
    type Output = Result<Vec<MutationRecord>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}
