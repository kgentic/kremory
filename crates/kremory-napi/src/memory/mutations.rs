//! Reversible graph mutations (Tier-1): forget/supersede/undo/unmerge/restore,
//! entity edit + delete (with their undo counterparts), and the mutation-history
//! / list-mutations inspection surface.

use napi_derive::napi;
use super::JsMemory;
use crate::convert;
use crate::convert::*;
use kremory::Namespace;

#[napi]
impl JsMemory {
    /// Forget (hard-delete) all episodes matching `source_id` in `namespace`.
    ///
    /// Wraps `Memory::forget().by_source_id(source_id).in_namespace(namespace)`.
    /// AppendOnly namespaces reject with `NamespacePolicyViolation`. When `namespace`
    /// is omitted, the Memory handle's default namespace is used. If neither is set
    /// (no per-call namespace AND no default registered on the handle), the call
    /// rejects with a namespace-required error.
    ///
    /// Returns the full per-table breakdown (`ForgetOutcome`) — until the Node
    /// parity pass (2026-09-14) this returned a bare `number` of ENTITY rows
    /// deleted only, which was actively misleading (see below).
    ///
    /// ⚠️ `entities === 0` does NOT mean nothing was erased. Shared-entity
    /// preservation pins any subject that also appears in another source, which
    /// is the normal case, so a complete erasure routinely leaves `entities: 0`
    /// while `facts`/`edges`/`episodes` were removed. Check `isEmpty` for the
    /// honest "did anything happen?" answer.
    #[napi]
    pub async fn forget(
        &self,
        source_id: String,
        namespace: Option<String>,
    ) -> napi::Result<JsForgetOutcome> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone())
            .ok_or_else(|| {
                napi::Error::from_reason(
                    "kremory forget failed: namespace required — set per-call or open with defaultNamespace"
                )
            })?;

        let deleted = self
            .inner
            .forget()
            .by_source_id(source_id)
            .in_namespace(ns)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory forget failed: {e}")))?;

        Ok(convert::forget_outcome_to_js(deleted))
    }

    /// Bound a fact's world-time `valid_to` window explicitly — the
    /// consumer-facing, consumer-EXPLICIT half of the supersession gap
    /// (auto-detected supersession is deferred to a future release).
    ///
    /// Wraps `Memory::supersede(factId).at(validTo).with_reason(reason?)
    /// .in_namespace(namespace)`. `validTo` is an RFC-3339 timestamp string; a
    /// malformed value rejects the returned Promise. When `namespace` is
    /// omitted, the Memory handle's default namespace is used. If neither is
    /// set, the call rejects with a namespace-required error.
    ///
    /// The dream supersession sweep (`dream({ includeSupersessionSweep: true
    /// })`) later observes the bounded `validTo` and closes the window
    /// (`expiredAt = validTo`, `supersessionsRecorded` increments) —
    /// see `kremory::SupersedeRequest` for the full two-phase mechanism.
    ///
    /// When `closeNow == true`, the deterministic
    /// `window_closeout` sweep runs INLINE right after bounding, retiring
    /// already-past-dated bounds in this one call — mirrors
    /// `SupersedeRequest::close_now()`. Only past-dated bounds retire; a
    /// future-dated bound returns `retired == 0` (deferred to a later dream sweep).
    ///
    /// Returns a `JsSupersedeOutcome` `{ outcome, retired }` where `outcome` is
    /// `"bounded"` | `"rejected_time_inversion"` | `"not_found"` (mirrors the
    /// honest `kremory::SupersedeOutcome::Bounded` rename) and `retired` is the
    /// in-band inline-close count (`0` unless `closeNow` retired past-dated bounds).
    #[napi]
    pub async fn supersede(
        &self,
        fact_id: i64,
        valid_to: String,
        reason: Option<String>,
        namespace: Option<String>,
        close_now: Option<bool>,
    ) -> napi::Result<JsSupersedeOutcome> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone())
            .ok_or_else(|| {
                napi::Error::from_reason(
                    "kremory supersede failed: namespace required — set per-call or open with defaultNamespace"
                )
            })?;

        // Loud parse — caller's malformed timestamp surfaces (mirrors the
        // `remember` wrapper's `reference_time` handling above).
        let valid_to_dt = chrono::DateTime::parse_from_rfc3339(&valid_to)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| {
                napi::Error::from_reason(format!(
                    "kremory supersede failed: invalid RFC-3339 validTo {valid_to:?}: {e}"
                ))
            })?;

        let mut req = self
            .inner
            .supersede(fact_id)
            .in_namespace(ns)
            .at(valid_to_dt);
        if let Some(r) = reason {
            req = req.with_reason(r);
        }
        if close_now.unwrap_or(false) {
            req = req.close_now();
        }

        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory supersede failed: {e}")))?;

        Ok(convert::supersede_outcome_to_js(outcome))
    }

    // ── Reversible-graph-mutations (Tier-1) — undo + inspect ──────────────────

    /// Reverse ANY logged, reversible mutation by its `mutationId` — the unified
    /// undo dispatcher. Wraps `Memory::undo`.
    ///
    /// Reads the mutation's kind and routes to the correct per-kind undo, returning
    /// a flat `UndoOutcome` `{ kind, unmerge?, editEntity?, deleteEntity?,
    /// deleteFact? }` — switch on `kind` and read the matching field. This is the
    /// method to reach for after iterating `listMutations` / `mutationHistory`
    /// (uniform `mem.undo(record.mutationId)` without switching on the kind by
    /// hand). The per-kind methods (`unmerge` / `undoEntityEdit` / `undoDeleteEntity`
    /// / `undoDeleteFact`) still work as the escape hatch.
    ///
    /// Only the four LOGGED kinds dispatch; a would-be row of a reserved kind
    /// (`fact_supersede` / `fact_archive` / `community_assign` / `canonical_form`)
    /// rejects the Promise (loud `UndoUnsupportedKind`). The optional `namespace`
    /// GUARDS the undo to the mutation's original namespace (mismatch rejects) — it
    /// is only applied when EXPLICITLY passed. The handle default is deliberately
    /// NOT injected here: a `mutationId` is a global key, so undoing a record from a
    /// cross-namespace `listMutations` must not be blocked by the handle default.
    #[napi]
    pub async fn undo(
        &self,
        mutation_id: i64,
        namespace: Option<String>,
    ) -> napi::Result<JsUndoOutcome> {
        let mut req = self.inner.undo(mutation_id);
        if let Some(ns) = namespace.as_deref().map(Namespace::new) {
            req = req.in_namespace(ns);
        }
        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory undo failed: {e}")))?;
        Ok(convert::undo_outcome_to_js(outcome))
    }

    /// Reverse a prior entity-merge by its `mutationId` (Tier-1).
    ///
    /// Fully restores the loser entity, its facts, its episodic edges, and the
    /// keeper's overwritten `access_count` / `ner_confidence`, then records a merge
    /// NOGOOD so the next `dream()` will NOT re-merge the split pair. Idempotent: a
    /// second call returns `alreadyUndone = true`. Wraps `Memory::unmerge`.
    ///
    /// Obtain the `mutationId` from `mutationHistory` / `listMutations`.
    #[napi]
    pub async fn unmerge(&self, mutation_id: i64) -> napi::Result<JsUnmergeOutcome> {
        let outcome = self
            .inner
            .unmerge(mutation_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory unmerge failed: {e}")))?;
        Ok(convert::unmerge_outcome_to_js(outcome))
    }

    /// Restore a fact previously moved to `facts_archive` (P2 archival) back into
    /// the live `facts` table (Tier-1). Wraps
    /// `Memory::restore_archived_fact`.
    ///
    /// Idempotent: if the fact is already live, returns `alreadyLive = true` and
    /// writes nothing.
    #[napi]
    pub async fn restore_archived_fact(
        &self,
        archived_fact_id: i64,
    ) -> napi::Result<JsRestoreArchivedOutcome> {
        let outcome = self
            .inner
            .restore_archived_fact(archived_fact_id)
            .execute()
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory restoreArchivedFact failed: {e}"))
            })?;
        Ok(convert::restore_archived_outcome_to_js(outcome))
    }

    /// Clear a supersession bound (`valid_to` / `expired_at`) set by `supersede`,
    /// re-opening the fact as currently-true (Tier-1). Wraps
    /// `Memory::unsupersede`.
    ///
    /// Idempotent: a fact with no bound set returns `outcome = "not_superseded"`
    /// (an honest no-op).
    #[napi]
    pub async fn unsupersede(&self, fact_id: i64) -> napi::Result<JsUnsupersedeOutcome> {
        let outcome = self
            .inner
            .unsupersede(fact_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory unsupersede failed: {e}")))?;
        Ok(convert::unsupersede_outcome_to_js(outcome))
    }

    /// Edit an entity — retype or rename — with full FK-propagation, provenance,
    /// and reconciler-freeze re-open (Tier-2a). Completes the
    /// diarization flow (rename `"Speaker 1"` → `"Alice"` propagating to all its
    /// facts). Wraps `Memory::edit_entity`.
    ///
    /// Set exactly one of `opts.newId` (rename/rekey — rejects an occupied id) or
    /// `opts.typeId` (retype — rejects a type id not registered in the namespace;
    /// id 0, the "Entity" catch-all, is always allowed). `opts.namespace` scopes
    /// the entity (else the handle default). Reverse via `undoEntityEdit` with the
    /// returned `mutationId`.
    #[napi]
    pub async fn edit_entity(
        &self,
        entity_id: String,
        opts: Option<JsEditEntityOptions>,
    ) -> napi::Result<JsEditEntityOutcome> {
        let opts = opts.unwrap_or(JsEditEntityOptions {
            new_id: None,
            type_id: None,
            namespace: None,
        });
        let ns = opts
            .namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let mut req = self.inner.edit_entity(entity_id);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }
        if let Some(new_id) = opts.new_id {
            req = req.rename(new_id);
        }
        if let Some(type_id) = opts.type_id {
            req = req.retype(type_id);
        }

        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory editEntity failed: {e}")))?;
        Ok(convert::edit_entity_outcome_to_js(outcome))
    }

    /// Reverse a prior `editEntity` from its provenance snapshot (Tier-2a).
    /// Pass the `mutationId` from the `EditEntityOutcome` (or from
    /// `mutationHistory` / `listMutations`). Idempotent. Wraps
    /// `Memory::undo_entity_edit`.
    #[napi]
    pub async fn undo_entity_edit(&self, mutation_id: i64) -> napi::Result<JsEditEntityOutcome> {
        let outcome = self
            .inner
            .undo_entity_edit(mutation_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory undoEntityEdit failed: {e}")))?;
        Ok(convert::edit_entity_outcome_to_js(outcome))
    }

    /// Delete an entity, reversibly (Tier-2b). Archives the entity's
    /// facts (recoverable — never hard-deleted), removes its edges / community
    /// membership / FTS / row, and retracts the DERIVED artifacts of neighbours whose
    /// live-fact support drops to zero. Reverse via `undoDeleteEntity` with the
    /// returned `mutationId`. `namespace` scopes the entity (else the handle default).
    /// Wraps `Memory::delete_entity`.
    #[napi]
    pub async fn delete_entity(
        &self,
        entity_id: String,
        namespace: Option<String>,
    ) -> napi::Result<JsDeleteEntityOutcome> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());
        let mut req = self.inner.delete_entity(entity_id);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }
        let outcome = req
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory deleteEntity failed: {e}")))?;
        Ok(convert::delete_entity_outcome_to_js(outcome))
    }

    /// Delete a single fact, reversibly (Tier-2b). The fact is archived
    /// (recoverable via `restoreArchivedFact`); either endpoint whose support drops to
    /// zero has its DERIVED community membership retracted. Reverse via `undoDeleteFact`.
    /// A fact id is global (not namespace-scoped). Wraps `Memory::delete_fact`.
    #[napi]
    pub async fn delete_fact(&self, fact_id: i64) -> napi::Result<JsDeleteFactOutcome> {
        let outcome = self
            .inner
            .delete_fact(fact_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory deleteFact failed: {e}")))?;
        Ok(convert::delete_fact_outcome_to_js(outcome))
    }

    /// Reverse a prior `deleteEntity` from its provenance snapshot (Tier-2b)
    /// — re-inserts the entity + FTS, restores its archived facts + episodic
    /// edges, and un-retracts every community membership the cascade retracted.
    /// Idempotent. Wraps `Memory::undo_delete_entity`.
    #[napi]
    pub async fn undo_delete_entity(
        &self,
        mutation_id: i64,
    ) -> napi::Result<JsDeleteEntityOutcome> {
        let outcome = self
            .inner
            .undo_delete_entity(mutation_id)
            .execute()
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory undoDeleteEntity failed: {e}"))
            })?;
        Ok(convert::delete_entity_outcome_to_js(outcome))
    }

    /// Reverse a prior `deleteFact` from its provenance snapshot (Tier-2b) —
    /// restores the archived fact + un-retracts any neighbour the cascade
    /// retracted. Idempotent. Wraps `Memory::undo_delete_fact`.
    #[napi]
    pub async fn undo_delete_fact(&self, mutation_id: i64) -> napi::Result<JsDeleteFactOutcome> {
        let outcome = self
            .inner
            .undo_delete_fact(mutation_id)
            .execute()
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory undoDeleteFact failed: {e}")))?;
        Ok(convert::delete_fact_outcome_to_js(outcome))
    }

    /// Inspect the mutations that touched one entity, newest-first (Tier-1) —
    /// the SEE half of the see+fix story. Each record
    /// carries the `mutationId` to pass to `unmerge`. Includes already-undone
    /// mutations. Wraps `Memory::mutation_history`.
    ///
    /// An entity id is namespace-scoped: pass `namespace` or open the handle with a
    /// `defaultNamespace`, else the call rejects with a namespace-required error.
    #[napi]
    pub async fn mutation_history(
        &self,
        entity_id: String,
        namespace: Option<String>,
    ) -> napi::Result<Vec<JsMutationRecord>> {
        let ns = namespace
            .as_deref()
            .map(Namespace::new)
            .or_else(|| self.default_namespace.clone());

        let mut req = self.inner.mutation_history(entity_id);
        if let Some(ns) = ns {
            req = req.in_namespace(ns);
        }

        let records = req.await.map_err(|e| {
            napi::Error::from_reason(format!("kremory mutationHistory failed: {e}"))
        })?;

        Ok(records
            .into_iter()
            .map(convert::mutation_record_to_js)
            .collect())
    }

    /// List logged graph mutations, newest-first (Tier-1). Each record
    /// carries its `mutationId` to undo. Wraps `Memory::list_mutations`.
    ///
    /// Filter via `opts`: `namespace` scopes to one namespace (else the
    /// `defaultNamespace`, else ALL namespaces); `kind` restricts the mutation kind
    /// (unknown tag rejects); `since` sets an RFC3339 `created_at` lower bound
    /// (malformed rejects); `includeUndone` adds already-reversed mutations
    /// (default: LIVE / still-reversible only).
    #[napi]
    pub async fn list_mutations(
        &self,
        opts: Option<JsMutationFilter>,
    ) -> napi::Result<Vec<JsMutationRecord>> {
        let mut req = self.inner.list_mutations();

        if let Some(f) = opts {
            let ns = f
                .namespace
                .as_deref()
                .map(Namespace::new)
                .or_else(|| self.default_namespace.clone());
            if let Some(ns) = ns {
                req = req.in_namespace(ns);
            }
            // Parse the kind tag loudly via the substrate enum's snake_case serde —
            // an unknown tag is a caller error, surfaced not silently dropped.
            if let Some(ref kind_str) = f.kind {
                let kind: kremory::MutationKind = serde_json::from_value(
                    serde_json::Value::String(kind_str.clone()),
                )
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "JsMutationFilter.kind: unknown value {kind_str:?}: {e}"
                    ))
                })?;
                req = req.kind(kind);
            }
            if let Some(ref since_str) = f.since {
                let ts = chrono::DateTime::parse_from_rfc3339(since_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        napi::Error::from_reason(format!(
                            "JsMutationFilter.since: invalid RFC-3339 timestamp {since_str:?}: {e}"
                        ))
                    })?;
                req = req.since(ts);
            }
            if f.include_undone.unwrap_or(false) {
                req = req.include_undone(true);
            }
        }

        let records = req
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory listMutations failed: {e}")))?;

        Ok(records
            .into_iter()
            .map(convert::mutation_record_to_js)
            .collect())
    }

}
