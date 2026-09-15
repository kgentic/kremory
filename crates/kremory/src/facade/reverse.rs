//! Consumer-facing reversal + INSPECT builders — the `Memory` surface for
//! the Stage-1 reversal primitives AND the read-only inspect surface.
//!
//! The three FIX builders (`Unmerge` / `RestoreArchived` / `Unsupersede`) are each
//! a `#[must_use]` handle that must call `.execute()` (mirrors `SupersedeRequest` /
//! `ForgetRequest`'s "no accidental `.await`" discipline for a mutating op), each
//! delegating to the deterministic substrate reversal
//! (`core::dream::provenance::reversal`) and returning the HONEST outcome type
//! (e.g. `already_undone`/`already_live`/`NotSuperseded`) — never a bare
//! `Applied`. The two SEE builders (`MutationHistory` /
//! `ListMutations`) are read-only and `IntoFuture` (await directly, like
//! `RecallRequest`), returning `Vec<MutationRecord>`. The napi mirror is
//! mechanical (Tier-0, not built here).

use std::future::IntoFuture;

use super::*;

use crate::core::dream::provenance::delete::DeleteEntityParams;
use crate::core::dream::provenance::edit::{EntityEditOp, EntityEditParams};
use crate::core::dream::provenance::{
    inspect, DeleteEntityOutcome, DeleteFactOutcome, EditEntityOutcome, MutationKind,
    MutationRecord, UnmergeOutcome,
};
use crate::memory::engine_handle::namespace_to_group_id;

// ── UnmergeRequest ─────────────────────────────────────────────────────────

/// Reverse a prior entity-merge by its `mutation_id`. Obtain via
/// `mem.unmerge(mutation_id)`.
///
/// Fully restores the loser entity, its facts, its episodic edges, and the
/// keeper's overwritten `access_count` / `ner_confidence`, then records a merge
/// NOGOOD so the next `dream()` will NOT re-merge the split pair.
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
/// `facts`. Obtain via `mem.restore_archived_fact(archived_fact_id)`.
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

/// Clear a supersession bound (`valid_to` / `expired_at` / `invalid_at`) set
/// by `supersede(...)` or by contradiction detection, re-opening the fact as
/// currently-true and re-eligible for consolidation (TD-178). Obtain via
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

// ── EditEntityRequest (the diarization rename/retype cascade) ────────────────

/// Edit an entity (retype or rename/rekey) with full FK-propagation + provenance.
/// Obtain via `mem.edit_entity(entity_id)`, then set exactly one of
/// `.rename(new_id)` / `.retype(type_id)`. Must call `.execute()`.
pub struct EditEntityRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) entity_id: String,
    pub(super) namespace: Option<Namespace>,
    pub(super) new_id: Option<String>,
    pub(super) new_type_id: Option<i64>,
}

impl<'a> EditEntityRequest<'a> {
    /// Scope the entity to `ns` (overrides the `Memory` default namespace).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Rename (REKEY) the entity to `new_id`, re-pointing every dependent FK. A
    /// rename INTO an existing id errors `EntityEditConflict`. Mutually exclusive
    /// with `.retype(...)`.
    #[must_use]
    pub fn rename(mut self, new_id: impl Into<String>) -> Self {
        self.new_id = Some(new_id.into());
        self
    }

    /// Re-type the entity to `type_id` (pins it `ConsumerPinned`, invalidates its
    /// community membership, re-opens the freeze). Mutually exclusive with
    /// `.rename(...)`.
    ///
    /// `type_id` must be a type REGISTERED in the target namespace (see
    /// `NamespaceSeed` / `Memory::register_namespace_with_seed`), or the id=0
    /// "Entity" catch-all, which is always admissible so a mis-typed entity can be
    /// demoted back to unclassified. An unregistered id fails with
    /// [`Error::EntityEditInvalid`](crate::core::error::Error::EntityEditInvalid)
    /// naming the offending id — it is NOT silently coerced to the catch-all,
    /// because that would absorb an explicit caller mistake.
    #[must_use]
    pub fn retype(mut self, type_id: i64) -> Self {
        self.new_type_id = Some(type_id);
        self
    }

    /// Execute the edit.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, if the namespace cannot
    /// be resolved, if neither/both of `rename`/`retype` were set
    /// (`EntityEditInvalid`), if `retype` named a type not registered in the
    /// namespace (`EntityEditInvalid` — id=0 excepted), if the entity does not
    /// exist (`EntityEditNotFound`), or if a rename targets an occupied id
    /// (`EntityEditConflict`). A rejected edit writes nothing: no type change and
    /// no `graph_mutation_log` row.
    pub async fn execute(self) -> Result<EditEntityOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::edit_entity requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let op = match (self.new_id, self.new_type_id) {
            (Some(new_id), None) => EntityEditOp::Rename { new_id },
            (None, Some(new_type_id)) => EntityEditOp::Retype { new_type_id },
            (None, None) => {
                return Err(MemoryError::Core(
                    crate::core::error::Error::EntityEditInvalid {
                        detail: "specify exactly one of .rename(new_id) or .retype(type_id)".into(),
                    },
                ));
            }
            (Some(_), Some(_)) => {
                return Err(MemoryError::Core(
                    crate::core::error::Error::EntityEditInvalid {
                        detail: ".rename(new_id) and .retype(type_id) are mutually exclusive — \
                             call one per edit"
                            .into(),
                    },
                ));
            }
        };
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let group_id = namespace_to_group_id(&ns);
        crate::core::dream::provenance::edit::edit_entity(
            tg,
            EntityEditParams {
                entity_id: self.entity_id,
                group_id,
                op,
            },
        )
        .await
        .map_err(MemoryError::Core)
    }
}

// ── UndoEntityEditRequest (reverse a prior edit) ──────────────────────────────

/// Reverse a prior `edit_entity` from its provenance snapshot. Obtain via
/// `mem.undo_entity_edit(mutation_id)`. Idempotent: a second call is a zero-count
/// no-op. Must call `.execute()`.
pub struct UndoEntityEditRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) mutation_id: i64,
}

impl UndoEntityEditRequest<'_> {
    /// Execute the reversal.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, if `mutation_id` names
    /// no `entity_edit` log row (`EntityEditNotFound`), or if the snapshot fails to
    /// deserialize (parse-loudly).
    pub async fn execute(self) -> Result<EditEntityOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::undo_entity_edit requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::edit::undo_entity_edit(tg, self.mutation_id)
            .await
            .map_err(MemoryError::Core)
    }
}

// ── DeleteEntityRequest (reversible delete cascade) ───────────────────────────

/// Delete an entity, reversibly. Obtain via `mem.delete_entity(entity_id)`.
/// Archives the entity's facts (recoverable — never hard-deleted), removes its edges
/// / community membership / FTS / row, and retracts the DERIVED artifacts of
/// neighbours whose live-fact support drops to zero. Undoable via
/// `undo_delete_entity(mutation_id)`. Must call `.execute()` (destructive op).
pub struct DeleteEntityRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) entity_id: String,
    pub(super) namespace: Option<Namespace>,
}

impl<'a> DeleteEntityRequest<'a> {
    /// Scope the entity to `ns` (overrides the `Memory` default namespace).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Execute the delete.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, if the namespace cannot be
    /// resolved, or if the entity does not exist (`EntityDeleteNotFound`).
    pub async fn execute(self) -> Result<DeleteEntityOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::delete_entity requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let group_id = namespace_to_group_id(&ns);
        crate::core::dream::provenance::delete::delete_entity(
            tg,
            DeleteEntityParams {
                entity_id: self.entity_id,
                group_id,
            },
        )
        .await
        .map_err(MemoryError::Core)
    }
}

// ── DeleteFactRequest (reversible fact delete) ────────────────────────────────

/// Delete a single fact, reversibly. Obtain via `mem.delete_fact(fact_id)`.
/// The fact is archived (recoverable via `restore_archived_fact`) and the DERIVED
/// artifacts of either endpoint whose support drops to zero are retracted. Undoable
/// via `undo_delete_fact(mutation_id)`. A fact id is global (not namespace-scoped).
/// Must call `.execute()`.
pub struct DeleteFactRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) fact_id: i64,
}

impl DeleteFactRequest<'_> {
    /// Execute the delete.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, or if `fact_id` names no
    /// fact (`FactDeleteNotFound`).
    pub async fn execute(self) -> Result<DeleteFactOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::delete_fact requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::delete::delete_fact(tg, self.fact_id)
            .await
            .map_err(MemoryError::Core)
    }
}

// ── UndoDeleteEntityRequest / UndoDeleteFactRequest ───────────────────────────

/// Reverse a prior `delete_entity` from its provenance snapshot. Obtain via
/// `mem.undo_delete_entity(mutation_id)`. Idempotent: a second call is a zero-count
/// no-op. Must call `.execute()`.
pub struct UndoDeleteEntityRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) mutation_id: i64,
}

impl UndoDeleteEntityRequest<'_> {
    /// Execute the reversal.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, if `mutation_id` names no
    /// `entity_delete` log row (`EntityDeleteNotFound`), or if the snapshot fails to
    /// deserialize (parse-loudly).
    pub async fn execute(self) -> Result<DeleteEntityOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::undo_delete_entity requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::delete::undo_delete_entity(tg, self.mutation_id)
            .await
            .map_err(MemoryError::Core)
    }
}

/// Reverse a prior `delete_fact` from its provenance snapshot. Obtain via
/// `mem.undo_delete_fact(mutation_id)`. Idempotent. Must call `.execute()`.
pub struct UndoDeleteFactRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) mutation_id: i64,
}

impl UndoDeleteFactRequest<'_> {
    /// Execute the reversal.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `Memory` lacks a `TemporalGraph`, if `mutation_id` names no
    /// `fact_delete` log row (`FactDeleteNotFound`), or if the snapshot fails to
    /// deserialize (parse-loudly).
    pub async fn execute(self) -> Result<DeleteFactOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::undo_delete_fact requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        crate::core::dream::provenance::delete::undo_delete_fact(tg, self.mutation_id)
            .await
            .map_err(MemoryError::Core)
    }
}

// ── MutationHistoryRequest (inspect surface — SEE) ────────────────────────────

/// Inspect the mutations that touched one entity. Obtain
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

// ── ListMutationsRequest (inspect surface — SEE) ──────────────────────────────

/// List logged graph mutations, newest-first. Obtain via
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

// ── UndoOutcome / UndoRequest (the unified reversal dispatcher) ───────────────

/// What a unified [`Memory::undo`](super::Memory::undo) call ACTUALLY reversed —
/// one variant per log-dispatchable [`MutationKind`], each wrapping that kind's
/// HONEST per-op outcome type, never a bare `Applied`. This is the return
/// of the ONE umbrella undo a consumer reaches for after iterating
/// [`list_mutations`](super::Memory::list_mutations): match the variant when you
/// need the per-kind counts, or ignore it for fire-and-forget reversal.
///
/// `#[non_exhaustive]`: a future log-dispatchable kind becomes a new variant with
/// zero breaking change (matching the `MutationKind` `#[non_exhaustive]` posture).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum UndoOutcome {
    /// The mutation was an `entity_merge`; reversed via
    /// [`unmerge`](super::Memory::unmerge).
    Unmerge(UnmergeOutcome),
    /// The mutation was an `entity_edit`; reversed via
    /// [`undo_entity_edit`](super::Memory::undo_entity_edit).
    EditEntity(EditEntityOutcome),
    /// The mutation was an `entity_delete`; reversed via
    /// [`undo_delete_entity`](super::Memory::undo_delete_entity).
    DeleteEntity(DeleteEntityOutcome),
    /// The mutation was a `fact_delete`; reversed via
    /// [`undo_delete_fact`](super::Memory::undo_delete_fact).
    DeleteFact(DeleteFactOutcome),
    /// A dream's fact archival, reversed — the fact is back in `facts`.
    RestoreArchived(crate::core::dream::provenance::RestoreArchivedOutcome),
    /// The mutation was a `fact_supersede`; reversed via the same
    /// [`unsupersede`](super::Memory::unsupersede) mechanism, reached by
    /// `mutation_id` instead of the domain `fact_id`.
    Unsupersede(crate::core::dream::provenance::UnsupersedeOutcome),
}

/// The unified undo dispatcher. Obtain via
/// `mem.undo(mutation_id)`.
///
/// Reads the `graph_mutation_log` row for `mutation_id`, matches on its `kind`,
/// and dispatches to the correct per-kind undo — so a consumer iterating
/// [`list_mutations`](super::Memory::list_mutations) can uniformly
/// `mem.undo(record.mutation_id)` without first switching on the kind by hand.
/// Returns the honest [`UndoOutcome`] carrying the reversed op's counts.
///
/// # The 6-of-8 honesty boundary
///
/// Six LOGGED, reversible kinds dispatch here: `entity_merge` →
/// [`unmerge`](super::Memory::unmerge), `entity_edit` →
/// [`undo_entity_edit`](super::Memory::undo_entity_edit), `entity_delete` →
/// [`undo_delete_entity`](super::Memory::undo_delete_entity), `fact_delete` →
/// [`undo_delete_fact`](super::Memory::undo_delete_fact), and `fact_archive` →
/// the same restore [`restore_archived_fact`](super::Memory::restore_archived_fact)
/// performs, reached by `mutation_id` instead of by an `archived_fact_id` no
/// public read returned (TD-250).
///
/// `fact_supersede` → the same [`unsupersede`](super::Memory::unsupersede)
/// mechanism, reached by `mutation_id` instead of the domain `fact_id` no
/// public read used to correlate back to a live row (2026-09-14 — the fix
/// that removed the need for a sixth MCP tool).
///
/// The other two [`MutationKind`] variants (`community_assign` /
/// `canonical_form`) remain RESERVED — not produced into the log today — so a
/// would-be row of that kind returns a LOUD
/// [`Error::UndoUnsupportedKind`](crate::core::error::Error::UndoUnsupportedKind).
///
/// Must call `.execute()` (mutating op).
#[must_use = "UndoRequest must call .execute() to run"]
pub struct UndoRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) mutation_id: i64,
    pub(super) namespace: Option<Namespace>,
}

impl<'a> UndoRequest<'a> {
    /// Guard the undo to `ns`: the dispatcher verifies the mutation's logged
    /// `group_id` matches this namespace, returning
    /// [`Error::UndoWrongNamespace`](crate::core::error::Error::UndoWrongNamespace)
    /// on a mismatch. Optional — omit it to undo by `mutation_id` alone (a
    /// `mutation_id` is a global, namespace-independent key).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Execute the dispatched reversal.
    ///
    /// # Errors
    ///
    /// - [`Error::MutationNotFound`](crate::core::error::Error::MutationNotFound)
    ///   — `mutation_id` names no `graph_mutation_log` row.
    /// - [`Error::UndoWrongNamespace`](crate::core::error::Error::UndoWrongNamespace)
    ///   — `.in_namespace(ns)` was set and its group differs from the mutation's.
    /// - [`Error::UndoUnsupportedKind`](crate::core::error::Error::UndoUnsupportedKind)
    ///   — the row's kind is one of the two reserved (non-log-dispatchable) kinds.
    /// - Whatever the dispatched per-kind undo returns (e.g.
    ///   [`Error::UnmergeOutOfOrder`](crate::core::error::Error::UnmergeOutOfOrder)).
    /// - `Err` if `Memory` was constructed without a `TemporalGraph` (test-stub path).
    pub async fn execute(self) -> Result<UndoOutcome> {
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::undo requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        // Read the log row's kind + group_id (the ONLY columns dispatch needs).
        // A missing row is a LOUD MutationNotFound (parse-loudly), never a silent
        // no-op — reversing a mutation that was never logged is a caller bug.
        let (kind_tag, group_id): (String, String) = {
            let mut rows = tg
                .conn
                .query(
                    "SELECT kind, group_id FROM graph_mutation_log WHERE id = ?1",
                    libsql::params![self.mutation_id],
                )
                .await
                .map_err(|e| MemoryError::Core(CoreError::Database(e)))?;
            match rows
                .next()
                .await
                .map_err(|e| MemoryError::Core(CoreError::Database(e)))?
            {
                Some(row) => (
                    row.get::<String>(0)
                        .map_err(|e| MemoryError::Core(CoreError::Database(e)))?,
                    row.get::<String>(1)
                        .map_err(|e| MemoryError::Core(CoreError::Database(e)))?,
                ),
                None => {
                    return Err(MemoryError::Core(CoreError::MutationNotFound {
                        mutation_id: self.mutation_id,
                    }));
                }
            }
        };

        // Optional namespace guard (C2): an undo targets the mutation's ORIGINAL
        // namespace. If the caller scoped a namespace, verify it matches the logged
        // group so we never silently reverse a mutation in an unintended namespace.
        if let Some(ns) = self.namespace {
            let requested_group = namespace_to_group_id(&ns);
            if requested_group != group_id {
                return Err(MemoryError::Core(CoreError::UndoWrongNamespace {
                    mutation_id: self.mutation_id,
                    mutation_group: group_id,
                    requested_group,
                }));
            }
        }

        // Parse-loudly: an unrecognised kind tag is a hard error, never
        // silently skipped — a log row we cannot classify must surface.
        let kind = MutationKind::from_tag(&kind_tag).map_err(MemoryError::Core)?;

        match kind {
            MutationKind::EntityMerge => Ok(UndoOutcome::Unmerge(
                self.memory.unmerge(self.mutation_id).execute().await?,
            )),
            MutationKind::EntityEdit => Ok(UndoOutcome::EditEntity(
                self.memory
                    .undo_entity_edit(self.mutation_id)
                    .execute()
                    .await?,
            )),
            MutationKind::EntityDelete => Ok(UndoOutcome::DeleteEntity(
                self.memory
                    .undo_delete_entity(self.mutation_id)
                    .execute()
                    .await?,
            )),
            MutationKind::FactDelete => Ok(UndoOutcome::DeleteFact(
                self.memory
                    .undo_delete_fact(self.mutation_id)
                    .execute()
                    .await?,
            )),
            // TD-250: reached by `mutation_id`, unlike `restore_archived_fact`,
            // which takes an `archived_fact_id` that no public read returns. Same
            // restore underneath — this arm exists so the archival is reversible
            // through the surface a consumer actually has, and so the log row is
            // marked `undone_at` like every other reversal.
            MutationKind::FactArchive => Ok(UndoOutcome::RestoreArchived(
                crate::core::dream::provenance::reversal::undo_fact_archive(tg, self.mutation_id)
                    .await
                    .map_err(MemoryError::Core)?,
            )),
            // 2026-09-14: reached by `mutation_id`, mirrors the `fact_archive` arm
            // above — same underlying `unsupersede` mechanism, reached the same
            // uniform way as every other logged kind instead of requiring the
            // caller to already know the domain `fact_id`.
            MutationKind::FactSupersede => Ok(UndoOutcome::Unsupersede(
                crate::core::dream::provenance::reversal::undo_fact_supersede(
                    tg,
                    self.mutation_id,
                )
                .await
                .map_err(MemoryError::Core)?,
            )),
            // The two RESERVED kinds are never produced into the log today; a
            // would-be row of that kind is loudly unsupported (R3 honesty).
            other => Err(MemoryError::Core(CoreError::UndoUnsupportedKind {
                mutation_id: self.mutation_id,
                kind: other.as_tag().to_string(),
            })),
        }
    }
}

#[cfg(test)]
mod undo_dispatch_tests {
    //! `mem.undo(mutation_id)` dispatches each of the six
    //! LOGGED kinds to the correct per-kind undo (same effect as the per-kind
    //! method), returns `MutationNotFound` for an unknown id, and returns a loud
    //! `UndoUnsupportedKind` for a would-be RESERVED-kind row. Deterministic,
    //! zero-LLM — fast tier, no VCR (`llm-test-pyramid-vcr-seams`).

    use std::sync::Arc;

    use crate::core::canonicalization::{
        canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD,
    };
    use crate::core::error::Error as CoreError;
    use crate::core::graph::InsertEntityWithGroupParams;
    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};

    use super::{namespace_to_group_id, MemoryError, Namespace, UndoOutcome};
    use crate::facade::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    fn ns() -> Namespace {
        Namespace::new("agent")
    }

    fn unit_vec() -> Vec<f32> {
        let v = 1.0_f32 / (384.0_f32).sqrt();
        vec![v; 384]
    }

    async fn insert_bare(mem: &Memory, id: &str) {
        let group = namespace_to_group_id(&ns());
        mem.temporal_graph
            .as_ref()
            .unwrap()
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({ "name": id }),
                group_id: Some(group.as_str()),
            })
            .await
            .expect("insert bare entity");
    }

    async fn insert_embedded(mem: &Memory, id: &str, description: &str) {
        let group = namespace_to_group_id(&ns());
        let tg = mem.temporal_graph.as_ref().unwrap();
        tg.insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0,
            properties: serde_json::json!({ "name": id, "description": description }),
            group_id: Some(group.as_str()),
        })
        .await
        .expect("insert embedded entity");
        tg.set_entity_embedding(id, &unit_vec())
            .await
            .expect("set embedding");
    }

    /// Drive a real canonicalize merge and return the logged `entity_merge` id.
    async fn make_merge(mem: &Memory) -> i64 {
        let group = namespace_to_group_id(&ns());
        insert_embedded(
            mem,
            "alice johnson",
            "A detailed description of Alice Johnson, engineer at Acme.",
        )
        .await;
        insert_embedded(mem, "alice j", "Alice.").await;
        let report = canonicalize_surface_forms(
            mem.temporal_graph.as_ref().unwrap(),
            &group,
            L5_CANONICALIZATION_THRESHOLD,
        )
        .await
        .expect("canonicalize");
        assert_eq!(report.merges_applied, 1, "exactly one merge produced");
        let mut rows = mem
            .temporal_graph
            .as_ref()
            .unwrap()
            .conn
            .query(
                "SELECT id FROM graph_mutation_log WHERE kind = 'entity_merge'",
                (),
            )
            .await
            .expect("log query");
        rows.next()
            .await
            .expect("row")
            .expect("one entity_merge row")
            .get::<i64>(0)
            .expect("id")
    }

    #[tokio::test]
    async fn undo_routes_entity_merge_to_unmerge() {
        let mem = make_memory().await;
        let mutation_id = make_merge(&mem).await;

        let outcome = mem.undo(mutation_id).execute().await.expect("undo merge");
        match outcome {
            UndoOutcome::Unmerge(o) => {
                // Same effect as `mem.unmerge(id)`: the loser is restored + a nogood
                // is recorded so the next dream() will not re-merge the pair.
                assert_eq!(o.restored_entity, "alice j", "loser restored");
                assert_eq!(o.keeper, "alice johnson", "keeper named");
                assert!(
                    o.nogood_recorded,
                    "unmerge records the anti-re-merge nogood"
                );
                assert!(!o.already_undone, "first undo is a real reversal");
            }
            other => panic!("expected Unmerge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn undo_routes_entity_edit_to_undo_entity_edit() {
        let mem = make_memory().await;
        insert_bare(&mem, "speaker 1").await;
        let edit = mem
            .edit_entity("speaker 1")
            .rename("alice")
            .in_namespace(ns())
            .execute()
            .await
            .expect("rename");
        assert!(edit.rekeyed, "rename rekeys");

        let outcome = mem
            .undo(edit.mutation_id)
            .execute()
            .await
            .expect("undo edit");
        match outcome {
            UndoOutcome::EditEntity(o) => {
                // Same effect as `mem.undo_entity_edit(id)`: the prior id is restored.
                assert_eq!(
                    o.entity_id, "speaker 1",
                    "rename undo restores the prior id"
                );
                assert!(!o.already_undone, "first undo is a real reversal");
            }
            other => panic!("expected EditEntity, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn undo_routes_entity_delete_to_undo_delete_entity() {
        let mem = make_memory().await;
        insert_bare(&mem, "bob").await;
        let del = mem
            .delete_entity("bob")
            .in_namespace(ns())
            .execute()
            .await
            .expect("delete entity");

        let outcome = mem
            .undo(del.mutation_id)
            .execute()
            .await
            .expect("undo delete");
        match outcome {
            UndoOutcome::DeleteEntity(o) => {
                assert_eq!(o.entity_id, "bob", "delete undo names the restored entity");
                assert!(!o.already_undone, "first undo is a real reversal");
            }
            other => panic!("expected DeleteEntity, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn undo_routes_fact_delete_to_undo_delete_fact() {
        let mem = make_memory().await;
        let group = namespace_to_group_id(&ns());
        insert_bare(&mem, "carol").await;
        insert_bare(&mem, "dave").await;
        let now = chrono::Utc::now().to_rfc3339();
        mem.temporal_graph
            .as_ref()
            .unwrap()
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
                  subject_group_id, object_group_id, confidence) \
                 VALUES ('carol', 'knows', 'dave', ?1, ?1, ?2, ?2, ?2, 1.0)",
                libsql::params![now, group.clone()],
            )
            .await
            .expect("plant fact");
        let fact_id: i64 = {
            let mut rows = mem
                .temporal_graph
                .as_ref()
                .unwrap()
                .conn
                .query(
                    "SELECT id FROM facts WHERE subject_id = 'carol' AND object_id = 'dave'",
                    (),
                )
                .await
                .expect("fact id query");
            rows.next()
                .await
                .expect("row")
                .expect("fact")
                .get::<i64>(0)
                .expect("id")
        };

        let del = mem
            .delete_fact(fact_id)
            .execute()
            .await
            .expect("delete fact");
        let outcome = mem
            .undo(del.mutation_id)
            .execute()
            .await
            .expect("undo delete fact");
        match outcome {
            UndoOutcome::DeleteFact(o) => {
                assert_eq!(
                    o.fact_id, fact_id,
                    "delete-fact undo names the restored fact"
                );
                assert!(o.fact_restored, "the archived fact was moved back to live");
                assert!(!o.already_undone, "first undo is a real reversal");
            }
            other => panic!("expected DeleteFact, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn undo_unknown_id_is_mutation_not_found() {
        let mem = make_memory().await;
        let err = mem
            .undo(999_999)
            .execute()
            .await
            .expect_err("unknown id must error");
        match err {
            MemoryError::Core(CoreError::MutationNotFound { mutation_id }) => {
                assert_eq!(mutation_id, 999_999);
            }
            other => panic!("expected MutationNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn undo_reserved_kind_is_unsupported_kind() {
        let mem = make_memory().await;
        let group = namespace_to_group_id(&ns());
        let now = chrono::Utc::now().to_rfc3339();
        // Plant a would-be `community_assign` row directly — nothing writes this
        // kind to the log today (the reserved-kind boundary), so `undo` must
        // refuse it loudly rather than dispatch. `fact_supersede` is NOT this
        // example any more — it became a LOGGED, dispatchable kind on 2026-09-14
        // (see `undo_routes_fact_supersede_to_unsupersede` below).
        mem.temporal_graph
            .as_ref()
            .unwrap()
            .conn
            .execute(
                "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
                 VALUES ('community_assign', ?1, ?2, '{}', '{}')",
                libsql::params![group, now],
            )
            .await
            .expect("plant reserved-kind row");
        let planted_id: i64 = {
            let mut rows = mem
                .temporal_graph
                .as_ref()
                .unwrap()
                .conn
                .query(
                    "SELECT id FROM graph_mutation_log WHERE kind = 'community_assign'",
                    (),
                )
                .await
                .expect("planted id query");
            rows.next()
                .await
                .expect("row")
                .expect("planted row")
                .get::<i64>(0)
                .expect("id")
        };

        let err = mem
            .undo(planted_id)
            .execute()
            .await
            .expect_err("reserved kind must error");
        match err {
            MemoryError::Core(CoreError::UndoUnsupportedKind { mutation_id, kind }) => {
                assert_eq!(mutation_id, planted_id);
                assert_eq!(
                    kind, "community_assign",
                    "the offending kind tag is surfaced"
                );
            }
            other => panic!("expected UndoUnsupportedKind, got {other:?}"),
        }
    }

    /// AC — `mem.undo(mutation_id)` on a LOGGED `fact_supersede` row dispatches to
    /// the same `unsupersede` mechanism the domain-id door uses, and reports it
    /// through `UndoOutcome::Unsupersede` — proves the 2026-09-14 fix that made a
    /// supersede retraction listable + undoable by `mutation_id` (removing the
    /// need for a sixth MCP tool), the same uniform way as every other logged
    /// kind.
    #[tokio::test]
    async fn undo_routes_fact_supersede_to_unsupersede() {
        let mem = make_memory().await;
        let group_ns = ns();
        let group = namespace_to_group_id(&group_ns);
        let now = chrono::Utc::now();
        let valid_from = now - chrono::Duration::days(10);

        let tg = mem.temporal_graph.as_ref().unwrap();
        tg.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "entity-undo-supersede-subject",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(group.as_str()),
        })
        .await
        .expect("seed subject entity");
        let fact_id = tg
            .insert_fact_with_group(
                crate::core::graph::FactInsert::new(
                    "entity-undo-supersede-subject",
                    "status",
                    valid_from,
                )
                .object_value("active"),
                Some(group.as_str()),
            )
            .await
            .expect("seed fact");

        let bound_at = now - chrono::Duration::days(1);
        let outcome = mem
            .supersede(fact_id)
            .in_namespace(group_ns.clone())
            .at(bound_at)
            .execute()
            .await
            .expect("supersede must succeed");
        assert!(matches!(
            outcome,
            crate::facade::SupersedeOutcome::Bounded { retired: 0 }
        ));

        let mutation_id = {
            let records = mem
                .list_mutations()
                .kind(crate::core::dream::provenance::MutationKind::FactSupersede)
                .in_namespace(group_ns.clone())
                .await
                .expect("list_mutations must succeed");
            assert_eq!(
                records.len(),
                1,
                "the supersede must be listable by kind — this is the whole fix"
            );
            records[0].mutation_id
        };

        let undo_outcome = mem
            .undo(mutation_id)
            .execute()
            .await
            .expect("undo of a fact_supersede row must succeed, not UndoUnsupportedKind");
        match undo_outcome {
            UndoOutcome::Unsupersede(
                crate::core::dream::provenance::UnsupersedeOutcome::Cleared {
                    fact_id: cleared_id,
                    cleared_valid_to,
                    ..
                },
            ) => {
                assert_eq!(cleared_id, fact_id);
                assert!(cleared_valid_to, "the bound this test set must be cleared");
            }
            other => panic!("expected UndoOutcome::Unsupersede(Cleared), got {other:?}"),
        }

        let fact = tg
            .get_fact_by_id(fact_id, &group)
            .await
            .expect("get_fact_by_id must succeed")
            .expect("fact must exist");
        assert_eq!(
            fact.valid_to, None,
            "undo must clear the DB row's valid_to bound"
        );
    }

    /// Regression for the HIGH-severity gap an adversarial review caught before
    /// this shipped: a fact superseded TWICE has a second `fact_supersede` row
    /// whose `prior_valid_to` is the FIRST bound, not `NULL`. Undoing the FIRST
    /// (older) mutation while the SECOND (newer) one is still live must be
    /// REFUSED, not silently clobber the live newer bound — `unsupersede`'s
    /// unconditional NULL-both would have destroyed it while reporting success.
    #[tokio::test]
    async fn undo_fact_supersede_out_of_order_is_rejected() {
        let mem = make_memory().await;
        let group_ns = ns();
        let group = namespace_to_group_id(&group_ns);
        let now = chrono::Utc::now();
        let valid_from = now - chrono::Duration::days(30);

        let tg = mem.temporal_graph.as_ref().unwrap();
        tg.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "entity-undo-ooo-subject",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(group.as_str()),
        })
        .await
        .expect("seed subject entity");
        let fact_id = tg
            .insert_fact_with_group(
                crate::core::graph::FactInsert::new(
                    "entity-undo-ooo-subject",
                    "status",
                    valid_from,
                )
                .object_value("active"),
                Some(group.as_str()),
            )
            .await
            .expect("seed fact");

        // First supersede: None -> t1.
        let t1 = now - chrono::Duration::days(20);
        mem.supersede(fact_id)
            .in_namespace(group_ns.clone())
            .at(t1)
            .execute()
            .await
            .expect("first supersede must succeed");
        let first_mutation_id = {
            let records = mem
                .list_mutations()
                .kind(crate::core::dream::provenance::MutationKind::FactSupersede)
                .in_namespace(group_ns.clone())
                .await
                .expect("list_mutations after first supersede");
            assert_eq!(records.len(), 1);
            records[0].mutation_id
        };

        // Second supersede: t1 -> t2 (a later, narrower bound).
        let t2 = now - chrono::Duration::days(10);
        mem.supersede(fact_id)
            .in_namespace(group_ns.clone())
            .at(t2)
            .execute()
            .await
            .expect("second supersede must succeed");

        // Undoing the FIRST (older) mutation must be refused — the second bound
        // is still live and would otherwise be silently destroyed.
        let err = mem
            .undo(first_mutation_id)
            .execute()
            .await
            .expect_err("undoing an out-of-order supersede must error, not succeed");
        match err {
            MemoryError::Core(CoreError::UndoStale { mutation_id, .. }) => {
                assert_eq!(mutation_id, first_mutation_id);
            }
            other => panic!("expected UndoStale, got {other:?}"),
        }

        // The live (second) bound must be UNTOUCHED by the rejected undo attempt.
        let fact = tg
            .get_fact_by_id(fact_id, &group)
            .await
            .expect("get_fact_by_id must succeed")
            .expect("fact must exist");
        assert_eq!(
            fact.valid_to.map(|t| t.timestamp()),
            Some(t2.timestamp()),
            "the still-live second bound must survive the rejected out-of-order undo"
        );
    }

    /// Companion to the out-of-order test: undoing the fact's supersedes
    /// newest-first must RESTORE each row's captured prior state, not blind-clear
    /// to `None` — proves `undo_fact_supersede` reads `prior_valid_to` back
    /// rather than delegating to `unsupersede`'s unconditional NULL-both.
    #[tokio::test]
    async fn undo_fact_supersede_restores_prior_bound_not_blind_clear() {
        let mem = make_memory().await;
        let group_ns = ns();
        let group = namespace_to_group_id(&group_ns);
        let now = chrono::Utc::now();
        let valid_from = now - chrono::Duration::days(30);

        let tg = mem.temporal_graph.as_ref().unwrap();
        tg.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "entity-undo-lifo-subject",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(group.as_str()),
        })
        .await
        .expect("seed subject entity");
        let fact_id = tg
            .insert_fact_with_group(
                crate::core::graph::FactInsert::new(
                    "entity-undo-lifo-subject",
                    "status",
                    valid_from,
                )
                .object_value("active"),
                Some(group.as_str()),
            )
            .await
            .expect("seed fact");

        let t1 = now - chrono::Duration::days(20);
        mem.supersede(fact_id)
            .in_namespace(group_ns.clone())
            .at(t1)
            .execute()
            .await
            .expect("first supersede must succeed");

        let t2 = now - chrono::Duration::days(10);
        mem.supersede(fact_id)
            .in_namespace(group_ns.clone())
            .at(t2)
            .execute()
            .await
            .expect("second supersede must succeed");

        let second_mutation_id = {
            let records = mem
                .list_mutations()
                .kind(crate::core::dream::provenance::MutationKind::FactSupersede)
                .in_namespace(group_ns.clone())
                .await
                .expect("list_mutations after second supersede");
            assert_eq!(records.len(), 2);
            // Newest-first (undone rows excluded by default) — the live row with
            // the LATEST created_at is the second supersede.
            records
                .iter()
                .max_by_key(|r| r.created_at.clone())
                .expect("at least one record")
                .mutation_id
        };

        // Undo the SECOND (latest) mutation — must restore to t1, the captured
        // `prior_valid_to`, NOT to `None`.
        let outcome = mem
            .undo(second_mutation_id)
            .execute()
            .await
            .expect("undo of the latest supersede must succeed");
        match outcome {
            UndoOutcome::Unsupersede(
                crate::core::dream::provenance::UnsupersedeOutcome::Cleared {
                    fact_id: cleared_id,
                    ..
                },
            ) => assert_eq!(cleared_id, fact_id),
            other => panic!("expected UndoOutcome::Unsupersede(Cleared), got {other:?}"),
        }

        let fact = tg
            .get_fact_by_id(fact_id, &group)
            .await
            .expect("get_fact_by_id must succeed")
            .expect("fact must exist");
        assert_eq!(
            fact.valid_to.map(|t| t.timestamp()),
            Some(t1.timestamp()),
            "undoing the second supersede must restore the FIRST bound, not wipe to None"
        );
    }

    #[tokio::test]
    async fn undo_wrong_namespace_guard_rejects() {
        let mem = make_memory().await;
        let mutation_id = make_merge(&mem).await;
        // The merge was logged under `ns()`; scoping the undo to a DIFFERENT
        // namespace must refuse loudly rather than reverse it.
        let err = mem
            .undo(mutation_id)
            .in_namespace(Namespace::new("some-other-namespace"))
            .execute()
            .await
            .expect_err("cross-namespace undo must error");
        assert!(
            matches!(err, MemoryError::Core(CoreError::UndoWrongNamespace { .. })),
            "expected UndoWrongNamespace, got {err:?}"
        );
    }
}
