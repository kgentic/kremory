//! Mutation-outcome napi conversions — supersede/unmerge/restore/unsupersede/edit/delete/forget/undo/mutation-record types.
//! Split out of `convert.rs` (TD-243); see `convert/mod.rs` for the domain map.

use napi_derive::napi;

/// Outcome of `JsMemory.supersede`.
///
/// `outcome` is one of `"bounded"` | `"rejected_time_inversion"` | `"not_found"`
/// (mirrors `kremory::SupersedeOutcome` + the `outcome` label on
/// `kremory.dream.consolidation.supersede_request_total` 1:1). The honest rename
/// `Applied` → `Bounded`: `execute()` only BOUNDS `valid_to`; it does not
/// retire the fact. `retired` carries the inline `.closeNow()` window-closeout
/// count IN-BAND — `0` for a plain supersede OR a future-dated bound whose
/// window has not yet closed, so a consumer observes the deferral from the return
/// value, not a doc caveat.
#[napi(object, js_name = "SupersedeOutcome")]
pub struct JsSupersedeOutcome {
    /// `"bounded"` | `"rejected_time_inversion"` | `"not_found"`.
    pub outcome: String,
    /// Facts retired by an inline `.closeNow()` sweep (`0` otherwise).
    /// Namespace-scoped total.
    pub retired: f64,
}

/// Convert a substrate `kremory::SupersedeOutcome` to `JsSupersedeOutcome`.
pub fn supersede_outcome_to_js(o: kremory::SupersedeOutcome) -> JsSupersedeOutcome {
    match o {
        kremory::SupersedeOutcome::Bounded { retired } => JsSupersedeOutcome {
            outcome: "bounded".to_string(),
            retired: retired as f64,
        },
        kremory::SupersedeOutcome::RejectedTimeInversion => JsSupersedeOutcome {
            outcome: "rejected_time_inversion".to_string(),
            retired: 0.0,
        },
        kremory::SupersedeOutcome::NotFound => JsSupersedeOutcome {
            outcome: "not_found".to_string(),
            retired: 0.0,
        },
        // Non-exhaustive guard: a variant added after this build maps to an honest
        // "unknown" marker (never a silent mis-label as "bounded").
        _ => JsSupersedeOutcome {
            outcome: "unknown".to_string(),
            retired: 0.0,
        },
    }
}

// ── Reversible-graph-mutations (Tier-1) outcome + inspect mirrors ─────────────

/// Outcome of `JsMemory.unmerge` — mirrors `kremory::UnmergeOutcome`. Every
/// count is the ACTUAL number reversed — never a placeholder or optimistic
/// estimate. All counts as JS `number` (f64): usize values far below 2^53 at
/// practical scale.
#[napi(object, js_name = "UnmergeOutcome")]
pub struct JsUnmergeOutcome {
    /// The restored loser entity id (the entity that had been hard-DELETEd).
    pub restored_entity: String,
    /// The keeper whose overwritten access_count / ner_confidence were restored.
    pub keeper: String,
    /// Facts whose endpoint + corroboration_inert flag were reverted.
    pub facts_repointed: f64,
    /// Episodic edges re-inserted (collided) or re-pointed back (non-collided).
    pub edges_restored: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// A NOGOOD was recorded for the split pair (next `dream()` will not re-merge).
    pub nogood_recorded: bool,
    /// `true` if the mutation was already undone — an idempotent no-op (all counts
    /// zero, `nogoodRecorded = false`), never a re-application.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::UnmergeOutcome` to `JsUnmergeOutcome`.
pub fn unmerge_outcome_to_js(o: kremory::UnmergeOutcome) -> JsUnmergeOutcome {
    JsUnmergeOutcome {
        restored_entity: o.restored_entity,
        keeper: o.keeper,
        facts_repointed: o.facts_repointed as f64,
        edges_restored: o.edges_restored as f64,
        entities_reopened: o.entities_reopened as f64,
        nogood_recorded: o.nogood_recorded,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.restoreArchivedFact` — mirrors
/// `kremory::RestoreArchivedOutcome`.
#[napi(object, js_name = "RestoreArchivedOutcome")]
pub struct JsRestoreArchivedOutcome {
    /// The `facts` row id restored from `facts_archive`.
    pub restored_fact_id: i64,
    /// `true` if the fact was already live — an honest no-op, nothing restored.
    pub already_live: bool,
}

/// Convert a substrate `kremory::RestoreArchivedOutcome` to
/// `JsRestoreArchivedOutcome`.
pub fn restore_archived_outcome_to_js(
    o: kremory::RestoreArchivedOutcome,
) -> JsRestoreArchivedOutcome {
    JsRestoreArchivedOutcome {
        restored_fact_id: o.restored_fact_id,
        already_live: o.already_live,
    }
}

/// Result of `JsMemory.backfillEntityEmbeddings` / `.backfillFactEmbeddings` —
/// mirrors `kremory::facade::EpisodeEmbeddingBackfill`. Not crate-root
/// re-exported by the substrate (only reachable as
/// `kremory::facade::EpisodeEmbeddingBackfill`), and not one of the
/// mechanical parity test's tracked struct types (`tracked_struct_types()`,
/// `tests/api_parity.rs`) — only `DreamSummary` is field-tracked there — so
/// this napi-side name does not need to match the substrate struct's name,
/// only its shape.
///
/// `embedded`/`failed` are `u64` on the substrate; cast to `f64` here per the
/// existing `usize`/`u64` → `f64` convention (`JsDreamSummary`'s doc comment
/// above) — safe up to 2^53, far beyond any realistic corpus size.
#[cfg(feature = "content-search")]
#[napi(object, js_name = "EmbeddingBackfillResult")]
pub struct JsEmbeddingBackfill {
    /// Items whose text was embedded + stored this run.
    pub embedded: f64,
    /// Items skipped due to a per-item embed/store failure (WARN-logged in
    /// the substrate log; re-run to retry them).
    pub failed: f64,
}

/// Convert a substrate `kremory::facade::EpisodeEmbeddingBackfill` to
/// `JsEmbeddingBackfill`. Feature-gated behind `content-search` — the
/// substrate type only exists there.
#[cfg(feature = "content-search")]
pub fn embedding_backfill_to_js(
    o: kremory::facade::EpisodeEmbeddingBackfill,
) -> JsEmbeddingBackfill {
    JsEmbeddingBackfill {
        embedded: o.embedded as f64,
        failed: o.failed as f64,
    }
}

/// Outcome of `JsMemory.unsupersede` — mirrors `kremory::UnsupersedeOutcome`.
///
/// `outcome` is `"cleared"` (a bound was cleared, fact re-opened as
/// currently-true) or `"not_superseded"` (no bound was set — honest no-op). The
/// `clearedValidTo` / `clearedExpiredAt` booleans record WHICH bound(s) were
/// cleared (both `false` for `"not_superseded"`).
#[napi(object, js_name = "UnsupersedeOutcome")]
pub struct JsUnsupersedeOutcome {
    /// `"cleared"` | `"not_superseded"`.
    pub outcome: String,
    /// The fact id this un-supersede targeted.
    pub fact_id: i64,
    /// `true` if the world-time `valid_to` bound was cleared.
    pub cleared_valid_to: bool,
    /// `true` if the system-time `expired_at` bound was cleared.
    pub cleared_expired_at: bool,
    /// `true` if the contradiction-resolver's domain-time marker (`invalid_at`)
    /// was also cleared, making the fact re-eligible for consolidation (TD-178).
    pub cleared_invalid_at: bool,
}

/// Convert a substrate `kremory::UnsupersedeOutcome` to `JsUnsupersedeOutcome`.
pub fn unsupersede_outcome_to_js(o: kremory::UnsupersedeOutcome) -> JsUnsupersedeOutcome {
    match o {
        kremory::UnsupersedeOutcome::Cleared {
            fact_id,
            cleared_valid_to,
            cleared_expired_at,
            cleared_invalid_at,
        } => JsUnsupersedeOutcome {
            outcome: "cleared".to_string(),
            fact_id,
            cleared_valid_to,
            cleared_expired_at,
            cleared_invalid_at,
        },
        kremory::UnsupersedeOutcome::NotSuperseded { fact_id } => JsUnsupersedeOutcome {
            outcome: "not_superseded".to_string(),
            fact_id,
            cleared_valid_to: false,
            cleared_expired_at: false,
            cleared_invalid_at: false,
        },
        // Non-exhaustive guard: unknown future variant maps to an honest marker
        // with fact_id = -1 (no real fact id is negative).
        _ => JsUnsupersedeOutcome {
            outcome: "unknown".to_string(),
            fact_id: -1,
            cleared_valid_to: false,
            cleared_expired_at: false,
            cleared_invalid_at: false,
        },
    }
}

/// Options for `JsMemory.editEntity` — set exactly one of `newId` (rename/rekey)
/// or `typeId` (retype). `namespace` scopes the entity (else the handle default).
#[napi(object, js_name = "EditEntityOptions")]
pub struct JsEditEntityOptions {
    /// Rename/REKEY target id. Mutually exclusive with `typeId`.
    pub new_id: Option<String>,
    /// Retype target `entity_type_id`. Mutually exclusive with `newId`.
    pub type_id: Option<i64>,
    /// Namespace scope. Omit → the Memory handle's `defaultNamespace`.
    pub namespace: Option<String>,
}

/// Outcome of `JsMemory.editEntity` / `JsMemory.undoEntityEdit` — mirrors
/// `kremory::EditEntityOutcome`. Every count is the ACTUAL rows affected —
/// never a placeholder or optimistic estimate. Counts as JS `number` (f64):
/// usize values far below 2^53 at practical scale.
#[napi(object, js_name = "EditEntityOutcome")]
pub struct JsEditEntityOutcome {
    /// The entity id AFTER the edit (new id for a rename; unchanged for a retype;
    /// the restored prior id for an undo).
    pub entity_id: String,
    /// `true` if this was a rename/rekey (id changed, FKs re-pointed).
    pub rekeyed: bool,
    /// `true` if the entity's type was changed (retype).
    pub retyped: bool,
    /// `facts` rows re-pointed (subject + object). Zero for a retype.
    pub facts_repointed: f64,
    /// `facts_archive` rows re-pointed. Zero for a retype.
    pub archived_repointed: f64,
    /// `episodic_edges` rows re-pointed. Zero for a retype.
    pub edges_repointed: f64,
    /// `entity_communities` rows re-pointed (rename) or invalidated (retype).
    pub communities_repointed: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// The `graph_mutation_log.id` — pass to `undoEntityEdit` to reverse.
    pub mutation_id: i64,
    /// `true` if `undoEntityEdit` found the mutation already reversed — a zero-count
    /// idempotent no-op. Always `false` for a forward `editEntity`.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::EditEntityOutcome` to `JsEditEntityOutcome`.
pub fn edit_entity_outcome_to_js(o: kremory::EditEntityOutcome) -> JsEditEntityOutcome {
    JsEditEntityOutcome {
        entity_id: o.entity_id,
        rekeyed: o.rekeyed,
        retyped: o.retyped,
        facts_repointed: o.facts_repointed as f64,
        archived_repointed: o.archived_repointed as f64,
        edges_repointed: o.edges_repointed as f64,
        communities_repointed: o.communities_repointed as f64,
        entities_reopened: o.entities_reopened as f64,
        mutation_id: o.mutation_id,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.deleteEntity` / `JsMemory.undoDeleteEntity` — mirrors
/// `kremory::DeleteEntityOutcome` (Tier-2b). Every count is the ACTUAL rows
/// affected — never a placeholder or optimistic estimate. On a forward
/// delete the counts are what was retracted/removed; on an undo they are the
/// inverse (rows restored). Counts as JS `number` (f64): usize values far
/// below 2^53 at practical scale.
#[napi(object, js_name = "DeleteEntityOutcome")]
pub struct JsDeleteEntityOutcome {
    /// The deleted (or, on undo, restored) entity id.
    pub entity_id: String,
    /// Facts archived (forward) or restored from archive (undo) — never hard-deleted.
    pub facts_retracted: f64,
    /// Episodic edges removed (forward) or re-inserted (undo).
    pub edges_removed: f64,
    /// The entity's OWN community memberships removed (forward) or restored (undo).
    pub communities_removed: f64,
    /// Neighbours whose community membership the retract-on-zero cascade retracted
    /// (forward) or restored (undo).
    pub neighbors_retracted: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// The `graph_mutation_log.id` — pass to `undoDeleteEntity` to reverse.
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — a zero-count idempotent no-op.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::DeleteEntityOutcome` to `JsDeleteEntityOutcome`.
pub fn delete_entity_outcome_to_js(o: kremory::DeleteEntityOutcome) -> JsDeleteEntityOutcome {
    JsDeleteEntityOutcome {
        entity_id: o.entity_id,
        facts_retracted: o.facts_retracted as f64,
        edges_removed: o.edges_removed as f64,
        communities_removed: o.communities_removed as f64,
        neighbors_retracted: o.neighbors_retracted as f64,
        entities_reopened: o.entities_reopened as f64,
        mutation_id: o.mutation_id,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.deleteFact` / `JsMemory.undoDeleteFact` — mirrors
/// `kremory::DeleteFactOutcome` (Tier-2b).
#[napi(object, js_name = "DeleteFactOutcome")]
pub struct JsDeleteFactOutcome {
    /// The deleted (or, on undo, restored) fact id.
    pub fact_id: i64,
    /// `true` only when an undo actually moved the fact back out of the archive.
    /// `false` on a forward `deleteFact` (archives, restores nothing) and on an undo
    /// that found the fact already live (an honest no-op restore). Distinct from
    /// `already_undone`, which flags the whole mutation as previously reversed.
    pub fact_restored: bool,
    /// Neighbours (endpoint entities) whose community membership the cascade retracted
    /// (forward) or restored (undo).
    pub neighbors_retracted: f64,
    /// Entities whose reconciler-freeze stamp was re-opened.
    pub entities_reopened: f64,
    /// The `graph_mutation_log.id` — pass to `undoDeleteFact` to reverse.
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — a zero-count idempotent no-op.
    pub already_undone: bool,
}

/// Convert a substrate `kremory::DeleteFactOutcome` to `JsDeleteFactOutcome`.
pub fn delete_fact_outcome_to_js(o: kremory::DeleteFactOutcome) -> JsDeleteFactOutcome {
    JsDeleteFactOutcome {
        fact_id: o.fact_id,
        fact_restored: o.fact_restored,
        neighbors_retracted: o.neighbors_retracted as f64,
        entities_reopened: o.entities_reopened as f64,
        mutation_id: o.mutation_id,
        already_undone: o.already_undone,
    }
}

/// Outcome of `JsMemory.forget` — mirrors `kremory::ForgetOutcome`. Was a bare
/// `number` (entity count only) until the Node parity pass (2026-09-14): the
/// Rust facade always returned the full per-table breakdown, and `0` entities
/// routinely masked a real erasure (shared-entity preservation), so the JS
/// caller had no honest way to tell "nothing happened" from "the entities
/// were pinned but everything else was removed."
#[napi(object, js_name = "ForgetOutcome")]
pub struct JsForgetOutcome {
    /// Entity rows deleted. `0` is normal and correct when every subject is
    /// shared with a source that was not erased.
    pub entities: f64,
    /// Fact rows deleted — both those whose endpoints went, and (on the
    /// source-scoped path) those sourced from an erased episode.
    pub facts: f64,
    /// Episode rows deleted: the stored source text itself.
    pub episodes: f64,
    /// `episodic_edges` rows deleted.
    pub edges: f64,
    /// `true` when this erasure removed nothing at all — the only honest way
    /// to ask "did anything happen?"; `entities === 0` alone is NOT that
    /// answer (see the struct doc comment).
    pub is_empty: bool,
}

/// Convert a substrate `kremory::ForgetOutcome` to `JsForgetOutcome`.
pub fn forget_outcome_to_js(o: kremory::ForgetOutcome) -> JsForgetOutcome {
    JsForgetOutcome {
        entities: o.entities as f64,
        facts: o.facts as f64,
        episodes: o.episodes as f64,
        edges: o.edges as f64,
        is_empty: o.is_empty(),
    }
}

/// Outcome of `JsMemory.undo` — the unified undo dispatcher.
/// Mirrors `kremory::UndoOutcome` as a FLAT JS object carrying the dispatched
/// `kind` PLUS exactly one populated per-kind outcome field (the other three are
/// omitted). A JS consumer switches on `kind`, then reads the matching field:
///
/// ```js
/// const r = await mem.undo(mutationId);
/// switch (r.kind) {
///   case "unmerge":       console.log(r.unmerge.restoredEntity); break;
///   case "edit_entity":   console.log(r.editEntity.entityId);    break;
///   case "delete_entity": console.log(r.deleteEntity.entityId);  break;
///   case "delete_fact":   console.log(r.deleteFact.factId);      break;
/// }
/// ```
#[napi(object, js_name = "UndoOutcome")]
pub struct JsUndoOutcome {
    /// Which per-kind undo ran: `"unmerge"` | `"edit_entity"` | `"delete_entity"`
    /// | `"delete_fact"` (mirrors the `kremory::UndoOutcome` variant).
    pub kind: String,
    /// Populated iff `kind === "unmerge"`.
    pub unmerge: Option<JsUnmergeOutcome>,
    /// Populated iff `kind === "edit_entity"`.
    pub edit_entity: Option<JsEditEntityOutcome>,
    /// Populated iff `kind === "delete_entity"`.
    pub delete_entity: Option<JsDeleteEntityOutcome>,
    /// Populated iff `kind === "delete_fact"`.
    pub delete_fact: Option<JsDeleteFactOutcome>,
}

/// Convert a substrate `kremory::UndoOutcome` to `JsUndoOutcome` (flat object with
/// a `kind` discriminator + the one populated per-kind outcome field).
pub fn undo_outcome_to_js(o: kremory::UndoOutcome) -> JsUndoOutcome {
    let mut js = JsUndoOutcome {
        kind: String::new(),
        unmerge: None,
        edit_entity: None,
        delete_entity: None,
        delete_fact: None,
    };
    match o {
        kremory::UndoOutcome::Unmerge(u) => {
            js.kind = "unmerge".to_string();
            js.unmerge = Some(unmerge_outcome_to_js(u));
        }
        kremory::UndoOutcome::EditEntity(e) => {
            js.kind = "edit_entity".to_string();
            js.edit_entity = Some(edit_entity_outcome_to_js(e));
        }
        kremory::UndoOutcome::DeleteEntity(d) => {
            js.kind = "delete_entity".to_string();
            js.delete_entity = Some(delete_entity_outcome_to_js(d));
        }
        kremory::UndoOutcome::DeleteFact(d) => {
            js.kind = "delete_fact".to_string();
            js.delete_fact = Some(delete_fact_outcome_to_js(d));
        }
        // Non-exhaustive guard: a future log-dispatchable kind maps to an honest
        // "unknown" marker with every outcome field omitted (never a mis-label).
        _ => {
            js.kind = "unknown".to_string();
        }
    }
    js
}

/// A single logged graph mutation from `JsMemory.mutationHistory` /
/// `JsMemory.listMutations` — mirrors `kremory::MutationRecord` (the consumer
/// INSPECT view). Carries the `mutationId` to pass to `unmerge` (or the
/// matching undo method for the kind).
#[napi(object, js_name = "MutationRecord")]
pub struct JsMutationRecord {
    /// The `graph_mutation_log.id` — pass to `unmerge` to reverse the mutation.
    pub mutation_id: i64,
    /// The mutation kind tag, e.g. `"entity_merge"` | `"fact_supersede"` |
    /// `"fact_archive"` | `"entity_edit"` | … (mirrors `kremory::MutationKind`).
    pub kind: String,
    /// RFC3339 timestamp when the mutation was applied.
    pub created_at: String,
    /// `true` once the mutation has been reversed.
    pub undone: bool,
    /// The namespace (group) this mutation scoped.
    pub group_id: String,
    /// Entity ids this mutation touched. For `entity_merge` this is `[keeper, loser]`.
    pub affected_entities: Vec<String>,
    /// A short human/agent-readable summary of what the mutation did.
    pub summary: String,
}

/// Render a `kremory::MutationKind` to its snake_case tag string. Uses serde (the
/// enum's `rename_all = "snake_case"` derive) so the tag matches the substrate's
/// `graph_mutation_log.kind` values 1:1 without reaching for the `pub(crate)`
/// `as_tag` helper. A serialize failure (unreachable for a unit enum) surfaces as
/// `"unknown"` rather than a panic.
fn mutation_kind_to_str(k: kremory::MutationKind) -> String {
    serde_json::to_value(k)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Convert a substrate `kremory::MutationRecord` to `JsMutationRecord`.
pub fn mutation_record_to_js(r: kremory::MutationRecord) -> JsMutationRecord {
    JsMutationRecord {
        mutation_id: r.mutation_id,
        kind: mutation_kind_to_str(r.kind),
        created_at: r.created_at,
        undone: r.undone,
        group_id: r.group_id,
        affected_entities: r.affected_entities,
        summary: r.summary,
    }
}

/// Filter for `JsMemory.listMutations` — mirrors `kremory::MutationFilter`. All
/// fields optional.
#[napi(object, js_name = "MutationFilter")]
pub struct JsMutationFilter {
    /// Restrict to one namespace. Omit → the Memory handle's default, else ALL
    /// namespaces.
    pub namespace: Option<String>,
    /// Restrict to one mutation kind tag (e.g. `"entity_merge"`). An unknown tag
    /// rejects the Promise. Omit → every kind.
    pub kind: Option<String>,
    /// Only mutations at/after this RFC3339 timestamp (`created_at >=`). Omit → no
    /// lower bound. A malformed timestamp rejects the Promise.
    pub since: Option<String>,
    /// Include already-undone mutations. Default `false` (live/still-reversible only).
    pub include_undone: Option<bool>,
}
