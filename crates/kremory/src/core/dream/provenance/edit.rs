//! Deterministic EDIT-ENTITY cascade (Tier-2a) for the reversible-graph-mutations
//! substrate (arch-spec `reversible-graph-mutations-arch-spec-2026-07-10.md` §4.3
//! + §2.3 `entity_edit`).
//!
//! Completes the diarization flow (`merge → unmerge → rename "Speaker 1" →
//! "Alice"`). Two edit flavors, one provenance shape:
//!
//! - **`retype`** — change `entity_type_id` (+ `entity_type_source` /
//!   `entity_type_assigned_at`) only; the id is unchanged so NO fact/edge rekey.
//!   Community membership is invalidated (dropped) + the reconciler freeze is
//!   re-opened so the next `dream()` re-places + re-evaluates the entity.
//! - **`rename`** — a **REKEY** of the composite TEXT PK `(id, group_id)`: every
//!   entity-id foreign key is re-pointed `old_id → new_id` inside ONE
//!   `BEGIN IMMEDIATE`. The full FK set (arch-spec §4.3, enumerated against the
//!   migrations):
//!   `facts.subject_id`/`object_id`, `facts_archive.subject_id`/`object_id`
//!   (V5 — archived facts carry TEXT endpoint ids too), `episodic_edges.entity_id`,
//!   `entity_communities.entity_id`, the `entities_fts` shadow, and the `entities`
//!   PK row itself. A rename INTO an occupied id is refused with a structured
//!   [`Error::EntityEditConflict`] (never a silent auto-merge — ADR-059).
//!
//! ## FK-off caveat (arch-spec §4.3 V5, load-bearing)
//!
//! The runtime DB open does NOT set `PRAGMA foreign_keys = ON` (the `=ON` lines in
//! `schema.rs` are all inside `#[tokio::test]` bodies). So the composite FKs on
//! `facts` / `facts_archive` are declarative-only at runtime: a *missed* rekey
//! UPDATE would NOT error today — it would silently ship a stale id. Therefore the
//! rekey MUST update EVERY table that carries an entity-id FK (this module's
//! `rekey_all_fks` is the single source of truth for that column set) and MUST NOT
//! rely on FK enforcement to catch an omission. The same UPDATE also snapshots the
//! affected ids so [`undo_entity_edit`] reverses precisely.
//!
//! Q5: `identity_verdict_audit.candidate_a` / `.candidate_b` (TEXT entity ids, no FK)
//! are DELIBERATELY excluded from the rekey — they are pure adjudication history that
//! must reference the id-at-decision-time, not be rewritten by a later rename.
//!
//! ## Provenance + undo
//!
//! Before the destructive writes, an `entity_edit` [`EntityEditPreState`] snapshot
//! is INSERTed into `graph_mutation_log` in the SAME txn (spec §8.1 — a table write
//! is transactional, so provenance can never diverge). [`undo_entity_edit`] replays
//! it: the inverse rekey (`new_id → old_id`) over the SNAPSHOTTED id set (precise —
//! it never touches a fact/edge added AFTER the rename), or the type/community
//! restore for a retype.
//!
//! ## Freeze re-open (§6.1)
//!
//! Every edit drops the affected entity's `dream_idempotency_keys` rows by INTEGER
//! rowid (`entity_id` is the rowid, `defs_g1.rs:80`) so the next `dream()`
//! re-processes it. A rename UPDATEs the entity row in place (rowid stable), so the
//! rowid is resolved by the FINAL id.
//!
//! ## `pub` + `#[doc(hidden)]` (MNT-002 precedent)
//!
//! `edit_entity` / `undo_entity_edit` are `pub` + `#[doc(hidden)]` (re-exported
//! under `feature = "test-utils"` from `core/dream/mod.rs`) for the same E0365
//! reason as the reversal free-functions: external integration-test binaries cannot
//! import `pub(crate)` items. They are NOT the stable public API — the consumer
//! surface is the `Memory` facade (`facade/reverse.rs`).

use metrics::counter;

use crate::core::error::{Error, Result};
use crate::core::schema::TemporalGraph;

use super::{EditEntityOutcome, EditInputs, EntityEditPreState, EpisodeEdgeKey};

// ─── caller-facing params ────────────────────────────────────────────────────

/// The single edit operation to apply — exactly one per [`edit_entity`] call
/// (the builder rejects zero/both, arch-spec §4.3).
#[derive(Debug, Clone)]
pub enum EntityEditOp {
    /// Rename/REKEY: change the entity id (`old → new_id`), re-pointing every FK.
    Rename { new_id: String },
    /// Retype: change `entity_type_id` only; id unchanged, no rekey.
    Retype { new_type_id: i64 },
}

/// Parameters for [`edit_entity`] — the resolved entity id, its namespace group,
/// and the edit op. Built by the facade (`facade/reverse.rs`) from
/// `mem.edit_entity(id).rename(...)` / `.retype(...)`.
#[derive(Debug, Clone)]
pub struct EntityEditParams {
    pub entity_id: String,
    pub group_id: String,
    pub op: EntityEditOp,
}

// ─── current-entity snapshot (pre-edit) ──────────────────────────────────────

/// The pre-edit entity state the snapshot + undo need. Loaded BEFORE any write.
struct CurrentEntity {
    entity_type_id: i64,
    entity_type_source: Option<String>,
    entity_type_assigned_at: Option<String>,
    properties: Option<String>,
    community_id: Option<i64>,
}

/// Load the current entity's type fields, `properties` (for FTS rebuild), and its
/// community membership. A missing entity is a hard [`Error::EntityEditNotFound`]
/// (parse-loudly, §2.1) — editing a non-existent entity is a caller bug, never a
/// silent no-op.
async fn load_current_entity(
    conn: &libsql::Connection,
    entity_id: &str,
    group_id: &str,
) -> Result<CurrentEntity> {
    let (entity_type_id, entity_type_source, entity_type_assigned_at, properties) = {
        let mut rows = conn
            .query(
                "SELECT entity_type_id, entity_type_source, entity_type_assigned_at, properties \
                 FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![entity_id, group_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::EntityEditNotFound {
                detail: format!("no entity `{entity_id}` in namespace `{group_id}`"),
            });
        };
        (
            row.get::<i64>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<Option<String>>(2)?,
            row.get::<Option<String>>(3)?,
        )
    };
    let community_id: Option<i64> = {
        let mut rows = conn
            .query(
                "SELECT community_id FROM entity_communities WHERE group_id = ?1 AND entity_id = ?2",
                libsql::params![group_id, entity_id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Some(row.get::<i64>(0)?),
            None => None,
        }
    };
    Ok(CurrentEntity {
        entity_type_id,
        entity_type_source,
        entity_type_assigned_at,
        properties,
        community_id,
    })
}

// ─── FK rekey (the load-bearing column set, §4.3) ────────────────────────────

/// Per-table row counts affected by a rekey.
struct RekeyCounts {
    facts: usize,
    archived: usize,
    edges: usize,
    communities: usize,
}

/// Bundled params for [`rekey_all_fks`] — args-as-object per `clippy.toml`
/// `too-many-arguments-threshold = 3` (`#[allow]` banned in src). `conn` stays a
/// lead positional param (receiver-like, project convention).
struct RekeyParams<'a> {
    from_id: &'a str,
    to_id: &'a str,
    group_id: &'a str,
}

/// Re-point EVERY entity-id foreign key `from_id → to_id` within `group_id` — the
/// single source of truth for the rekey column set (§4.3 FK-off caveat). Used
/// forward (`old → new`) by [`edit_entity`]'s rename branch; the undo path re-points
/// by snapshotted id instead (precise). The `entities` PK row + FTS are handled by
/// the caller (they need the `properties` for the FTS rebuild).
///
/// Namespace-scoped on the `*_group_id` columns: the composite FK is
/// `(subject_id, subject_group_id) REFERENCES entities(id, group_id)`, so a row
/// referencing THIS entity necessarily carries `*_group_id = group_id` — a same-id
/// entity in a different namespace is untouched.
async fn rekey_all_fks(conn: &libsql::Connection, params: RekeyParams<'_>) -> Result<RekeyCounts> {
    let RekeyParams {
        from_id,
        to_id,
        group_id,
    } = params;
    let facts_subject = conn
        .execute(
            "UPDATE facts SET subject_id = ?1 WHERE subject_id = ?2 AND subject_group_id = ?3",
            libsql::params![to_id, from_id, group_id],
        )
        .await?;
    let facts_object = conn
        .execute(
            "UPDATE facts SET object_id = ?1 WHERE object_id = ?2 AND object_group_id = ?3",
            libsql::params![to_id, from_id, group_id],
        )
        .await?;
    // V5: archived facts carry TEXT endpoint ids too — a rekey that skips
    // `facts_archive` leaves stale ids that `restore_archived_fact` would resurrect
    // pointing at the dead `from_id`.
    let archived_subject = conn
        .execute(
            "UPDATE facts_archive SET subject_id = ?1 \
             WHERE subject_id = ?2 AND subject_group_id = ?3",
            libsql::params![to_id, from_id, group_id],
        )
        .await?;
    let archived_object = conn
        .execute(
            "UPDATE facts_archive SET object_id = ?1 \
             WHERE object_id = ?2 AND object_group_id = ?3",
            libsql::params![to_id, from_id, group_id],
        )
        .await?;
    let edges = conn
        .execute(
            "UPDATE episodic_edges SET entity_id = ?1 \
             WHERE entity_id = ?2 AND entity_group_id = ?3",
            libsql::params![to_id, from_id, group_id],
        )
        .await?;
    let communities = conn
        .execute(
            "UPDATE entity_communities SET entity_id = ?1 \
             WHERE entity_id = ?2 AND group_id = ?3",
            libsql::params![to_id, from_id, group_id],
        )
        .await?;
    Ok(RekeyCounts {
        facts: (facts_subject + facts_object) as usize,
        archived: (archived_subject + archived_object) as usize,
        edges: edges as usize,
        communities: communities as usize,
    })
}

/// Snapshot the affected id sets for the rename `pre_state` (so undo reverses
/// precisely, never touching a row added AFTER the rename). Read BEFORE the rekey.
struct AffectedIds {
    fact_ids: Vec<i64>,
    archived_fact_ids: Vec<i64>,
    episode_edge_keys: Vec<EpisodeEdgeKey>,
}

async fn collect_affected_ids(
    conn: &libsql::Connection,
    entity_id: &str,
    group_id: &str,
) -> Result<AffectedIds> {
    let mut fact_ids = Vec::new();
    {
        let mut rows = conn
            .query(
                "SELECT id FROM facts \
                 WHERE (subject_id = ?1 AND subject_group_id = ?2) \
                    OR (object_id = ?1 AND object_group_id = ?2)",
                libsql::params![entity_id, group_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            fact_ids.push(row.get::<i64>(0)?);
        }
    }
    let mut archived_fact_ids = Vec::new();
    {
        let mut rows = conn
            .query(
                "SELECT id FROM facts_archive \
                 WHERE (subject_id = ?1 AND subject_group_id = ?2) \
                    OR (object_id = ?1 AND object_group_id = ?2)",
                libsql::params![entity_id, group_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            archived_fact_ids.push(row.get::<i64>(0)?);
        }
    }
    let mut episode_edge_keys = Vec::new();
    {
        let mut rows = conn
            .query(
                "SELECT episode_id, entity_group_id FROM episodic_edges \
                 WHERE entity_id = ?1 AND entity_group_id = ?2",
                libsql::params![entity_id, group_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            episode_edge_keys.push(EpisodeEdgeKey {
                episode_id: row.get::<i64>(0)?,
                entity_group_id: row.get::<String>(1)?,
            });
        }
    }
    Ok(AffectedIds {
        fact_ids,
        archived_fact_ids,
        episode_edge_keys,
    })
}

/// Defer foreign-key constraint checks to the transaction's COMMIT (SQLite
/// `PRAGMA defer_foreign_keys`). A rename re-points child-table FKs onto an id
/// that only becomes live when the entity row is renamed at the END of the txn, so
/// under IMMEDIATE FK enforcement an intermediate statement would violate. Deferring
/// makes the whole rekey atomic w.r.t. FK checks; the pragma auto-resets at the next
/// COMMIT/ROLLBACK. Harmless no-op when FKs are OFF (production, §4.3 V5).
pub(super) async fn defer_foreign_keys(conn: &libsql::Connection) -> Result<()> {
    conn.execute("PRAGMA defer_foreign_keys = ON", ()).await?;
    Ok(())
}

// ─── freeze re-open (§6.1) ───────────────────────────────────────────────────

/// Drop the entity's `dream_idempotency_keys` rows so the next `dream()`
/// re-processes it (§6.1). `dream_idempotency_keys.entity_id` is the INTEGER rowid
/// (`defs_g1.rs:80`), resolved from the entity's CURRENT id. Returns 1 if the
/// entity was found + any keys dropped (0 otherwise), for the honest outcome count.
pub(super) async fn reopen_freeze(
    conn: &libsql::Connection,
    entity_id: &str,
    group_id: &str,
) -> Result<usize> {
    let rowid: Option<i64> = {
        let mut rows = conn
            .query(
                "SELECT rowid FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![entity_id, group_id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Some(row.get::<i64>(0)?),
            None => None,
        }
    };
    let Some(rid) = rowid else {
        return Ok(0);
    };
    let dropped = conn
        .execute(
            "DELETE FROM dream_idempotency_keys WHERE entity_id = ?1",
            libsql::params![rid],
        )
        .await?;
    Ok(if dropped > 0 { 1 } else { 0 })
}

// ─── edit_entity (forward, §4.3) ─────────────────────────────────────────────

/// Apply a deterministic entity edit (retype or rename/rekey), snapshotting the
/// pre-state into `graph_mutation_log` in the SAME txn so it is undoable, then
/// re-opening the reconciler freeze (§4.3 / §6.1). All in ONE `BEGIN IMMEDIATE`.
///
/// # Errors
///
/// - [`Error::EntityEditNotFound`] if `entity_id` names no entity in the namespace.
/// - [`Error::EntityEditConflict`] if a `rename` targets an already-occupied id.
/// - A serialize / DB error rolls the whole edit back (atomic with the snapshot).
#[doc(hidden)]
pub async fn edit_entity(
    graph: &TemporalGraph,
    params: EntityEditParams,
) -> Result<EditEntityOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match edit_entity_txn(graph, &params).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    // §8.1: the `graph_mutation_log` row is a durable in-txn write (correct on
    // rollback); its summary counter fires only AFTER the real commit.
    if committed_here {
        let op_label = if outcome.rekeyed { "rename" } else { "retype" };
        counter!(
            "kremory.graph.mutation_logged_total",
            "kind" => "entity_edit",
            "source" => "edit",
        )
        .increment(1);
        tracing::debug!(
            target: "kremory.graph.provenance",
            kind = "entity_edit",
            op = op_label,
            group_id = %params.group_id,
            entity_id = %outcome.entity_id,
            facts_repointed = outcome.facts_repointed,
            archived_repointed = outcome.archived_repointed,
            edges_repointed = outcome.edges_repointed,
            "kremory.graph.mutation_logged: entity_edit provenance row committed"
        );
    }
    Ok(outcome)
}

async fn edit_entity_txn(
    graph: &TemporalGraph,
    params: &EntityEditParams,
) -> Result<EditEntityOutcome> {
    let conn = &graph.conn;
    let EntityEditParams {
        entity_id,
        group_id,
        op,
    } = params;

    // A rekey re-points FKs onto an id that does not exist YET (the entity row is
    // renamed IN PLACE at the end to keep its rowid stable). Under immediate FK
    // enforcement (enabled in the test harness; OFF in production per §4.3 V5) an
    // intermediate step would therefore trip a FOREIGN KEY constraint. Defer FK
    // checks to COMMIT — by which point the whole rekey is consistent. The pragma
    // is auto-reset at this txn's COMMIT/ROLLBACK, and is a harmless no-op when FKs
    // are off. Correct treatment (make the multi-statement rekey atomic w.r.t. FK
    // checks) — NOT a suppression.
    defer_foreign_keys(conn).await?;

    let current = load_current_entity(conn, entity_id, group_id).await?;

    match op {
        EntityEditOp::Retype { new_type_id } => {
            retype_txn(
                conn,
                RetypeTxnParams {
                    entity_id,
                    group_id,
                    current: &current,
                    new_type_id: *new_type_id,
                },
            )
            .await
        }
        EntityEditOp::Rename { new_id } => {
            rename_txn(conn, RenameTxnParams {
                old_id: entity_id,
                new_id,
                group_id,
                current: &current,
            })
            .await
        }
    }
}

/// Bundled params for [`retype_txn`] — args-as-object (clippy threshold 3).
struct RetypeTxnParams<'a> {
    entity_id: &'a str,
    group_id: &'a str,
    current: &'a CurrentEntity,
    new_type_id: i64,
}

/// Retype: change `entity_type_id`, stamp `entity_type_source = 'ConsumerPinned'`
/// (§4.3 — a consumer-directed retype is an explicit pin; `ConsumerPinned` is an
/// admissible CHECK value and is excluded from reclassify so the pin sticks) +
/// `entity_type_assigned_at = now`. Invalidate community membership (drop) so
/// community detection re-places it, and re-open the freeze. No fact/edge rekey.
async fn retype_txn(
    conn: &libsql::Connection,
    params: RetypeTxnParams<'_>,
) -> Result<EditEntityOutcome> {
    let RetypeTxnParams {
        entity_id,
        group_id,
        current,
        new_type_id,
    } = params;
    let pre_state = EntityEditPreState {
        old_id: entity_id.to_string(),
        new_id: entity_id.to_string(),
        old_label: None,
        new_label: None,
        old_type_id: current.entity_type_id,
        new_type_id,
        rekey: false,
        affected_fact_ids: Vec::new(),
        affected_archived_fact_ids: Vec::new(),
        affected_episode_edge_keys: Vec::new(),
        prior_type_source: current.entity_type_source.clone(),
        prior_type_assigned_at: current.entity_type_assigned_at.clone(),
        prior_community_id: current.community_id,
    };
    let inputs = EditInputs {
        old_id: entity_id.to_string(),
        new_id: entity_id.to_string(),
        rekey: false,
        old_type_id: current.entity_type_id,
        new_type_id,
    };
    let mutation_id = write_edit_log(conn, WriteEditLogParams { group_id, pre_state: &pre_state, inputs: &inputs }).await?;

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE entities SET entity_type_id = ?1, entity_type_source = 'ConsumerPinned', \
         entity_type_assigned_at = ?2, updated_at = ?2 WHERE id = ?3 AND group_id = ?4",
        libsql::params![new_type_id, now, entity_id, group_id],
    )
    .await?;

    // Invalidate community membership so detection re-places the re-typed entity.
    let communities = conn
        .execute(
            "DELETE FROM entity_communities WHERE group_id = ?1 AND entity_id = ?2",
            libsql::params![group_id, entity_id],
        )
        .await?;

    let entities_reopened = reopen_freeze(conn, entity_id, group_id).await?;

    Ok(EditEntityOutcome {
        entity_id: entity_id.to_string(),
        rekeyed: false,
        retyped: true,
        facts_repointed: 0,
        archived_repointed: 0,
        edges_repointed: 0,
        communities_repointed: communities as usize,
        entities_reopened,
        mutation_id,
        already_undone: false,
    })
}

/// Bundled params for [`rename_txn`] — args-as-object (clippy threshold 3).
struct RenameTxnParams<'a> {
    old_id: &'a str,
    new_id: &'a str,
    group_id: &'a str,
    current: &'a CurrentEntity,
}

/// Rename/REKEY: re-point every entity-id FK `old → new`, rename the entity row +
/// FTS, snapshot the affected ids, re-open the freeze. Rejects rename INTO an
/// occupied id ([`Error::EntityEditConflict`], §4.3).
async fn rename_txn(
    conn: &libsql::Connection,
    params: RenameTxnParams<'_>,
) -> Result<EditEntityOutcome> {
    let RenameTxnParams {
        old_id,
        new_id,
        group_id,
        current,
    } = params;

    // Reject rename into an existing id (never a silent fuse — ADR-059).
    if old_id != new_id {
        let occupied = {
            let mut rows = conn
                .query(
                    "SELECT 1 FROM entities WHERE id = ?1 AND group_id = ?2",
                    libsql::params![new_id, group_id],
                )
                .await?;
            rows.next().await?.is_some()
        };
        if occupied {
            return Err(Error::EntityEditConflict {
                from: old_id.to_string(),
                existing: new_id.to_string(),
                group_id: group_id.to_string(),
            });
        }
    }

    // Snapshot the affected id sets BEFORE the rekey (undo reverses precisely).
    let affected = collect_affected_ids(conn, old_id, group_id).await?;
    let pre_state = EntityEditPreState {
        old_id: old_id.to_string(),
        new_id: new_id.to_string(),
        old_label: None,
        new_label: None,
        old_type_id: current.entity_type_id,
        new_type_id: current.entity_type_id,
        rekey: true,
        affected_fact_ids: affected.fact_ids.clone(),
        affected_archived_fact_ids: affected.archived_fact_ids.clone(),
        affected_episode_edge_keys: affected.episode_edge_keys.clone(),
        prior_type_source: current.entity_type_source.clone(),
        prior_type_assigned_at: current.entity_type_assigned_at.clone(),
        prior_community_id: current.community_id,
    };
    let inputs = EditInputs {
        old_id: old_id.to_string(),
        new_id: new_id.to_string(),
        rekey: true,
        old_type_id: current.entity_type_id,
        new_type_id: current.entity_type_id,
    };
    let mutation_id = write_edit_log(conn, WriteEditLogParams { group_id, pre_state: &pre_state, inputs: &inputs }).await?;

    // Re-point every FK old → new (the load-bearing column set).
    let counts = rekey_all_fks(
        conn,
        RekeyParams {
            from_id: old_id,
            to_id: new_id,
            group_id,
        },
    )
    .await?;

    // Rename the entity PK row IN PLACE (rowid stable → soft rowid refs, e.g.
    // dream_pass4_audit + idempotency keys, survive) + rebuild its FTS shadow.
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE entities SET id = ?1, updated_at = ?2 WHERE id = ?3 AND group_id = ?4",
        libsql::params![new_id, now, old_id, group_id],
    )
    .await?;
    // entities_fts.label is always '' (the entities.label column was dropped Mig
    // 009); rebuild from the snapshotted properties under the new id.
    conn.execute(
        "DELETE FROM entities_fts WHERE entity_id = ?1",
        libsql::params![old_id],
    )
    .await?;
    conn.execute(
        "INSERT INTO entities_fts (entity_id, label, properties) VALUES (?1, '', ?2)",
        libsql::params![new_id, current.properties.clone().unwrap_or_default()],
    )
    .await?;

    // Freeze re-open by the NEW (current) id — rowid is stable across the in-place
    // rename, but the id key changed.
    let entities_reopened = reopen_freeze(conn, new_id, group_id).await?;

    Ok(EditEntityOutcome {
        entity_id: new_id.to_string(),
        rekeyed: true,
        retyped: false,
        facts_repointed: counts.facts,
        archived_repointed: counts.archived,
        edges_repointed: counts.edges,
        communities_repointed: counts.communities,
        entities_reopened,
        mutation_id,
        already_undone: false,
    })
}

/// Bundled params for [`write_edit_log`] — args-as-object (clippy threshold 3).
struct WriteEditLogParams<'a> {
    group_id: &'a str,
    pre_state: &'a EntityEditPreState,
    inputs: &'a EditInputs,
}

/// INSERT the `entity_edit` provenance row on `conn` (same txn) and return its id.
/// Snapshot BEFORE the destructive writes (§8.1). Parse-loudly on serialize error.
async fn write_edit_log(conn: &libsql::Connection, params: WriteEditLogParams<'_>) -> Result<i64> {
    let WriteEditLogParams {
        group_id,
        pre_state,
        inputs,
    } = params;
    let pre_state_json = serde_json::to_string(pre_state)
        .map_err(|e| Error::Other(anyhow::anyhow!("serialize entity_edit pre_state: {e}")))?;
    let inputs_json = serde_json::to_string(inputs)
        .map_err(|e| Error::Other(anyhow::anyhow!("serialize entity_edit inputs: {e}")))?;
    let now = chrono::Utc::now().to_rfc3339();
    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.graph.provenance",
            kind = "entity_edit",
            group_id = %group_id,
            pre_state = %pre_state_json,
            inputs = %inputs_json,
            "reversible-mutations entity_edit snapshot captured (pre-mutate)"
        );
    }
    conn.execute(
        "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        libsql::params![
            "entity_edit",
            group_id,
            now,
            pre_state_json,
            inputs_json
        ],
    )
    .await?;
    let mutation_id: i64 = {
        let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::InsertReturnedNoRowId {
                operation: "insert_entity_edit_log",
            });
        };
        row.get::<i64>(0)?
    };
    Ok(mutation_id)
}

// ─── undo_entity_edit (reversal, §4.3) ───────────────────────────────────────

/// Reverse a prior `entity_edit` from its `graph_mutation_log` snapshot (§4.3):
/// the inverse rekey (`new → old`) over the SNAPSHOTTED id set for a rename, or the
/// type/community restore for a retype. Idempotent: an already-undone edit returns
/// a zero-count outcome (never a double-reversal). A missing / non-`entity_edit`
/// `mutation_id`, or a `pre_state` that fails to deserialize, is a hard `Error`
/// (parse-loudly). All in ONE `BEGIN IMMEDIATE`.
#[doc(hidden)]
pub async fn undo_entity_edit(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<EditEntityOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match undo_entity_edit_txn(graph, mutation_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    // Q3: an already-undone edit is a no-op — its counter must NOT fire (matching the
    // `unmerge` / delete-undo `!already_undone` precedent), else a repeat undo inflates
    // `mutation_undone_total`.
    if committed_here && !outcome.already_undone {
        counter!(
            "kremory.graph.mutation_undone_total",
            "kind" => "entity_edit",
        )
        .increment(1);
    }
    Ok(outcome)
}

async fn undo_entity_edit_txn(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<EditEntityOutcome> {
    let conn = &graph.conn;

    // Same rationale as the forward edit: the inverse rekey re-points FKs onto the
    // restored id before the entity row is renamed back. Defer FK checks to commit.
    defer_foreign_keys(conn).await?;

    let (group_id, undone_at, pre_state_json): (String, Option<String>, String) = {
        let mut rows = conn
            .query(
                "SELECT group_id, undone_at, pre_state FROM graph_mutation_log \
                 WHERE id = ?1 AND kind = 'entity_edit'",
                libsql::params![mutation_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::EntityEditNotFound {
                detail: format!(
                    "no entity_edit mutation-log row with id {mutation_id} — \
                     cannot reverse an edit that was never logged"
                ),
            });
        };
        (
            row.get::<String>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<String>(2)?,
        )
    };

    let pre: EntityEditPreState = serde_json::from_str(&pre_state_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "undo_entity_edit: deserialize pre_state for mutation {mutation_id}: {e}"
        ))
    })?;

    // Idempotent: already reversed → zero-count no-op.
    if undone_at.is_some() {
        return Ok(EditEntityOutcome {
            entity_id: pre.old_id,
            rekeyed: pre.rekey,
            retyped: !pre.rekey,
            facts_repointed: 0,
            archived_repointed: 0,
            edges_repointed: 0,
            communities_repointed: 0,
            entities_reopened: 0,
            mutation_id,
            already_undone: true,
        });
    }

    let outcome = if pre.rekey {
        undo_rename(conn, &group_id, &pre).await?
    } else {
        undo_retype(conn, &group_id, &pre).await?
    };

    // Mark reversed.
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE graph_mutation_log SET undone_at = ?1 WHERE id = ?2",
        libsql::params![now, mutation_id],
    )
    .await?;

    Ok(EditEntityOutcome {
        mutation_id,
        ..outcome
    })
}

/// Inverse rekey `new → old` over the SNAPSHOTTED id set (precise — a fact/edge
/// added AFTER the rename is never touched). Then rename the entity row + FTS back.
async fn undo_rename(
    conn: &libsql::Connection,
    group_id: &str,
    pre: &EntityEditPreState,
) -> Result<EditEntityOutcome> {
    let old_id = &pre.old_id;
    let new_id = &pre.new_id;

    let mut facts_repointed = 0usize;
    for fid in &pre.affected_fact_ids {
        facts_repointed += conn
            .execute(
                "UPDATE facts SET subject_id = ?1 WHERE id = ?2 AND subject_id = ?3",
                libsql::params![old_id.clone(), *fid, new_id.clone()],
            )
            .await? as usize;
        facts_repointed += conn
            .execute(
                "UPDATE facts SET object_id = ?1 WHERE id = ?2 AND object_id = ?3",
                libsql::params![old_id.clone(), *fid, new_id.clone()],
            )
            .await? as usize;
    }
    let mut archived_repointed = 0usize;
    for fid in &pre.affected_archived_fact_ids {
        archived_repointed += conn
            .execute(
                "UPDATE facts_archive SET subject_id = ?1 WHERE id = ?2 AND subject_id = ?3",
                libsql::params![old_id.clone(), *fid, new_id.clone()],
            )
            .await? as usize;
        archived_repointed += conn
            .execute(
                "UPDATE facts_archive SET object_id = ?1 WHERE id = ?2 AND object_id = ?3",
                libsql::params![old_id.clone(), *fid, new_id.clone()],
            )
            .await? as usize;
    }
    let mut edges_repointed = 0usize;
    for key in &pre.affected_episode_edge_keys {
        edges_repointed += conn
            .execute(
                "UPDATE episodic_edges SET entity_id = ?1 \
                 WHERE episode_id = ?2 AND entity_group_id = ?3 AND entity_id = ?4",
                libsql::params![
                    old_id.clone(),
                    key.episode_id,
                    key.entity_group_id.clone(),
                    new_id.clone()
                ],
            )
            .await? as usize;
    }
    // Community membership was re-pointed old → new by the rename; re-point it back.
    let communities_repointed = conn
        .execute(
            "UPDATE entity_communities SET entity_id = ?1 WHERE entity_id = ?2 AND group_id = ?3",
            libsql::params![old_id.clone(), new_id.clone(), group_id],
        )
        .await? as usize;

    // Rename the entity row back (in place) + rebuild FTS under the old id.
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE entities SET id = ?1, updated_at = ?2 WHERE id = ?3 AND group_id = ?4",
        libsql::params![old_id.clone(), now, new_id.clone(), group_id],
    )
    .await?;
    let properties: Option<String> = {
        let mut rows = conn
            .query(
                "SELECT properties FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![old_id.clone(), group_id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => row.get::<Option<String>>(0)?,
            None => None,
        }
    };
    conn.execute(
        "DELETE FROM entities_fts WHERE entity_id = ?1",
        libsql::params![new_id.clone()],
    )
    .await?;
    conn.execute(
        "INSERT INTO entities_fts (entity_id, label, properties) VALUES (?1, '', ?2)",
        libsql::params![old_id.clone(), properties.unwrap_or_default()],
    )
    .await?;

    let entities_reopened = reopen_freeze(conn, old_id, group_id).await?;

    Ok(EditEntityOutcome {
        entity_id: old_id.clone(),
        rekeyed: true,
        retyped: false,
        facts_repointed,
        archived_repointed,
        edges_repointed,
        communities_repointed,
        entities_reopened,
        mutation_id: 0,
        already_undone: false,
    })
}

/// Restore the prior type + community membership for a retype undo.
async fn undo_retype(
    conn: &libsql::Connection,
    group_id: &str,
    pre: &EntityEditPreState,
) -> Result<EditEntityOutcome> {
    let entity_id = &pre.old_id; // id unchanged for a retype
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE entities SET entity_type_id = ?1, entity_type_source = ?2, \
         entity_type_assigned_at = ?3, updated_at = ?4 WHERE id = ?5 AND group_id = ?6",
        libsql::params![
            pre.old_type_id,
            pre.prior_type_source.clone(),
            pre.prior_type_assigned_at.clone(),
            now.clone(),
            entity_id.clone(),
            group_id
        ],
    )
    .await?;

    // The retype dropped community membership; restore it if there was one.
    let communities_repointed = if let Some(cid) = pre.prior_community_id {
        conn.execute(
            "INSERT OR REPLACE INTO entity_communities (group_id, entity_id, community_id, updated_at) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![group_id, entity_id.clone(), cid, now],
        )
        .await? as usize
    } else {
        0
    };

    let entities_reopened = reopen_freeze(conn, entity_id, group_id).await?;

    Ok(EditEntityOutcome {
        entity_id: entity_id.clone(),
        rekeyed: false,
        retyped: true,
        facts_repointed: 0,
        archived_repointed: 0,
        edges_repointed: 0,
        communities_repointed,
        entities_reopened,
        mutation_id: 0,
        already_undone: false,
    })
}
