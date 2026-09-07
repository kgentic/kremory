//! Deterministic DELETE cascade (Tier-2b) for the reversible-graph-mutations
//! substrate (`delete-entity` + `delete-fact` + `entity_delete` / `fact_delete`).
//!
//! A delete is **REVERSIBLE**, not a hard drop. Two ops, one provenance shape:
//!
//! - **`delete_entity`** (§4.4) — deleting entity `E` in group `g`, in ONE
//!   `BEGIN IMMEDIATE`: **archive** each of E's facts to `facts_archive` (recoverable
//!   by id — NEVER hard-deleted), **remove** E from `episodic_edges` /
//!   `entity_communities` / `entities_fts`, **delete** the `entities` row (snapshotted
//!   first), then run the **retract-on-zero** cascade (B5/DRed) — for every OTHER
//!   entity reachable over the fact FK graph whose live-fact support drops to zero,
//!   retract its DERIVED community membership (the base entity is NEVER auto-deleted;
//!   HippoRAG #17: teardown the derived, keep the base).
//! - **`delete_fact`** (§4.5) — archive the fact (recoverable via
//!   `restore_archived_fact`) + run the same retract-on-zero over its two endpoints.
//!
//! Each op snapshots an [`EntityDeletePreState`] / [`FactDeletePreState`] into
//! `graph_mutation_log` in the SAME txn BEFORE the destructive writes (§8.1 — a table
//! write is transactional, so provenance can never diverge), so
//! [`undo_delete_entity`] / [`undo_delete_fact`] restore the **recall-relevant** graph
//! (entities / facts / episodic edges / community memberships) exactly.
//!
//! ## Undo scope — recall data, NOT append-only audit history (Q4, honest scoping)
//!
//! `DELETE FROM entities` fires the `trg_dream_pass4_audit_cascade_delete` trigger,
//! which removes that entity's `dream_pass4_audit` type-correction rows (they FK the
//! old integer rowid). `undo_delete_entity` re-INSERTs the entity under a NEW rowid, so
//! it does NOT — and deliberately cannot — restore those audit rows: they reference the
//! rowid-at-decision-time, and re-inserting them against the new rowid would fabricate
//! history. This is correct: `dream_pass4_audit` is append-only CORRECTION history, not
//! recall data — its loss does not affect what the deleted-then-restored entity recalls.
//! The undo restores everything a query can observe; it does not resurrect the reconciler
//! audit trail of a delete-then-undo round-trip.
//!
//! ## Reachability (§4.1 — cycle-safe)
//!
//! The retract-on-zero candidate set is enumerated by a recursive-CTE walk over the
//! EXISTING fact FKs ([`reachable_dependents`]). The structural FK graph HAS cycles
//! (`A—fact—B` and `B—fact—A`), so the CTE uses `UNION` (set-dedup — terminates on
//! cycles) plus a `MAX_CASCADE_DEPTH` cap as belt-and-suspenders. Only DIRECT
//! neighbours can actually drop to zero support when a single node is removed; the
//! reachability walk is the cycle-safe enumeration mechanism the spec calls for, and
//! transitively-reachable entities simply recompute to nonzero support and are left
//! alone.
//!
//! ## `pub` + `#[doc(hidden)]` (MNT-002 precedent)
//!
//! `delete_entity` / `delete_fact` / `undo_delete_entity` / `undo_delete_fact` +
//! `DeleteEntityParams` are `pub` + `#[doc(hidden)]` (re-exported under
//! `feature = "test-utils"` from `core/dream/mod.rs`) for the same E0365 reason as the
//! reversal / edit free-functions: external integration-test binaries cannot import
//! `pub(crate)` items. They are NOT the stable public API — the consumer surface is the
//! `Memory` facade (`facade/reverse.rs`).

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use metrics::counter;

use crate::core::dream::consolidation::archive::ARCHIVE_INSERT_SQL;
use crate::core::error::{Error, Result};
use crate::core::schema::TemporalGraph;

use super::edit::{defer_foreign_keys, reopen_freeze};
use super::reversal::restore_archived_fact;
use super::{
    DeleteEntityInputs, DeleteEntityOutcome, DeleteFactInputs, DeleteFactOutcome,
    DeletedEpisodicEdge, EntityDeletePreState, FactDeletePreState, LoserEntityRow, MutationKind,
    RetractedNeighbor,
};

/// Cycle-safety depth cap for the reachability CTE (§4.1). Belt-and-suspenders atop
/// the `UNION` set-dedup that already terminates on cycles.
const MAX_CASCADE_DEPTH: i64 = 32;

// ─── caller-facing params ────────────────────────────────────────────────────

/// Parameters for [`delete_entity`] — the resolved entity id + its namespace group.
/// Built by the facade (`facade/reverse.rs`) from `mem.delete_entity(id)`.
#[derive(Debug, Clone)]
pub struct DeleteEntityParams {
    pub entity_id: String,
    pub group_id: String,
}

// ─── shared reads ─────────────────────────────────────────────────────────────

/// Load an entity's full 11-column row (the same live `entities` column set a merge
/// snapshots, §2.3(1)) as a [`LoserEntityRow`] for the delete snapshot. A missing
/// entity is a hard [`Error::EntityDeleteNotFound`] (parse-loudly, §2.1) — deleting a
/// non-existent entity is a caller bug, never a silent no-op.
async fn load_entity_row(
    conn: &libsql::Connection,
    entity_id: &str,
    group_id: &str,
) -> Result<LoserEntityRow> {
    let mut rows = conn
        .query(
            "SELECT id, group_id, properties, embedding, recorded_at, updated_at, \
                    access_count, entity_type_id, entity_type_source, \
                    entity_type_assigned_at, ner_confidence \
             FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![entity_id, group_id],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Err(Error::EntityDeleteNotFound {
            detail: format!("no entity `{entity_id}` in namespace `{group_id}`"),
        });
    };
    let embedding: Option<Vec<u8>> = row.get::<Option<Vec<u8>>>(3)?;
    Ok(LoserEntityRow {
        id: row.get::<String>(0)?,
        group_id: row.get::<String>(1)?,
        properties: row.get::<Option<String>>(2)?,
        embedding_b64: embedding.as_deref().map(|b| BASE64_STANDARD.encode(b)),
        recorded_at: row.get::<String>(4)?,
        updated_at: row.get::<Option<String>>(5)?,
        access_count: row.get::<i64>(6)?,
        entity_type_id: row.get::<i64>(7)?,
        entity_type_source: row.get::<Option<String>>(8)?,
        entity_type_assigned_at: row.get::<Option<String>>(9)?,
        ner_confidence: row.get::<Option<f64>>(10)?,
    })
}

/// The community membership id of `entity_id` in `group_id`, or `None`.
async fn community_of(
    conn: &libsql::Connection,
    entity_id: &str,
    group_id: &str,
) -> Result<Option<i64>> {
    let mut rows = conn
        .query(
            "SELECT community_id FROM entity_communities WHERE group_id = ?1 AND entity_id = ?2",
            libsql::params![group_id, entity_id],
        )
        .await?;
    match rows.next().await? {
        Some(row) => Ok(Some(row.get::<i64>(0)?)),
        None => Ok(None),
    }
}

/// Cycle-safe reachable dependent entities from `seed_id` over the fact FK graph
/// (§4.1). `UNION` set-dedups (terminates on cycles) + a `MAX_CASCADE_DEPTH` cap.
/// Returns DISTINCT reachable entity ids INCLUDING the seed (depth 0); the caller
/// excludes the seed when scanning for retract-on-zero neighbours.
async fn reachable_dependents(
    conn: &libsql::Connection,
    seed_id: &str,
    group_id: &str,
) -> Result<Vec<String>> {
    // The recursive term steps to the OTHER endpoint of each incident fact; the
    // `IS NOT NULL` guard drops literal-object facts (no entity to reach), and the
    // depth cap bounds any pathological high-degree hub.
    let mut rows = conn
        .query(
            "WITH RECURSIVE reach(entity_id, depth) AS ( \
                 SELECT ?1, 0 \
                 UNION \
                 SELECT CASE WHEN f.subject_id = r.entity_id THEN f.object_id \
                             ELSE f.subject_id END, \
                        r.depth + 1 \
                 FROM reach r \
                 JOIN facts f ON (f.subject_id = r.entity_id OR f.object_id = r.entity_id) \
                 WHERE r.depth < ?2 \
                   AND f.group_id = ?3 \
                   AND (CASE WHEN f.subject_id = r.entity_id THEN f.object_id \
                             ELSE f.subject_id END) IS NOT NULL \
             ) \
             SELECT DISTINCT entity_id FROM reach",
            libsql::params![seed_id, MAX_CASCADE_DEPTH, group_id],
        )
        .await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push(row.get::<String>(0)?);
    }
    Ok(out)
}

// ─── retract-on-zero support checks (B5/DRed, §4.4 / §4.5) ────────────────────

/// Bundled params for [`neighbor_retracted_on_entity_delete`] — args-as-object per
/// `clippy.toml` `too-many-arguments-threshold = 3` (`#[allow]` banned in src).
struct EntityRetractCheck<'a> {
    deleted_id: &'a str,
    candidate: &'a str,
    group_id: &'a str,
}

/// If archiving ALL of `deleted_id`'s facts would drop `candidate`'s live-fact
/// support to zero AND `candidate` has a community membership, return the neighbour
/// to retract (with its prior community id). Otherwise `None`. Support is recomputed
/// at cascade-time (no stored counter — that is Stage 2's `support_count`).
async fn neighbor_retracted_on_entity_delete(
    conn: &libsql::Connection,
    params: EntityRetractCheck<'_>,
) -> Result<Option<RetractedNeighbor>> {
    let EntityRetractCheck {
        deleted_id,
        candidate,
        group_id,
    } = params;
    // Live facts referencing `candidate` that do NOT involve `deleted_id` — the
    // support that SURVIVES the delete.
    let post_support: i64 = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM facts \
                 WHERE group_id = ?1 AND (subject_id = ?2 OR object_id = ?2) \
                   AND subject_id != ?3 AND (object_id IS NULL OR object_id != ?3)",
                libsql::params![group_id, candidate, deleted_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "delete cascade: post-support COUNT(*) returned no row"
            )));
        };
        row.get::<i64>(0)?
    };
    if post_support > 0 {
        return Ok(None);
    }
    Ok(community_of(conn, candidate, group_id)
        .await?
        .map(|cid| RetractedNeighbor {
            entity_id: candidate.to_string(),
            prior_community_id: cid,
        }))
}

/// Bundled params for [`neighbor_retracted_on_fact_delete`] — args-as-object.
struct FactRetractCheck<'a> {
    endpoint: &'a str,
    fact_id: i64,
    group_id: &'a str,
}

/// If deleting fact `fact_id` would drop endpoint `endpoint`'s live-fact support to
/// zero AND it has a community membership, return the neighbour to retract.
async fn neighbor_retracted_on_fact_delete(
    conn: &libsql::Connection,
    params: FactRetractCheck<'_>,
) -> Result<Option<RetractedNeighbor>> {
    let FactRetractCheck {
        endpoint,
        fact_id,
        group_id,
    } = params;
    let post_support: i64 = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM facts \
                 WHERE group_id = ?1 AND (subject_id = ?2 OR object_id = ?2) AND id != ?3",
                libsql::params![group_id, endpoint, fact_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "delete cascade: post-support COUNT(*) returned no row"
            )));
        };
        row.get::<i64>(0)?
    };
    if post_support > 0 {
        return Ok(None);
    }
    Ok(community_of(conn, endpoint, group_id)
        .await?
        .map(|cid| RetractedNeighbor {
            entity_id: endpoint.to_string(),
            prior_community_id: cid,
        }))
}

// ─── shared writes ────────────────────────────────────────────────────────────

/// Archive one fact `facts → facts_archive` (recoverable by id) + drop its FTS
/// shadow + delete the live row, all on `conn` (the caller's open txn). Mirrors
/// `consolidation::archive::move_fact` but without opening a nested guard (this runs
/// inside the delete's `BEGIN IMMEDIATE`). Reuses `ARCHIVE_INSERT_SQL` (the single
/// source of truth for the 19-column archive projection).
async fn archive_fact(conn: &libsql::Connection, fact_id: i64, group_id: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(ARCHIVE_INSERT_SQL, libsql::params![now, fact_id, group_id])
        .await?;
    conn.execute(
        "DELETE FROM facts_fts WHERE fact_id = ?1",
        libsql::params![fact_id],
    )
    .await?;
    conn.execute(
        "DELETE FROM facts WHERE id = ?1 AND group_id = ?2",
        libsql::params![fact_id, group_id],
    )
    .await?;
    Ok(())
}

/// Retract a neighbour's DERIVED community membership + re-open its freeze (§4.4).
/// Returns 1 if the freeze was re-opened (keys dropped), for the honest count.
///
/// The `cascade_retracted_total` counter is NOT emitted here: this runs INSIDE the
/// delete's open `BEGIN IMMEDIATE`, and a metric increment cannot be rolled back.
/// The forward `delete_entity` / `delete_fact`
/// paths emit it post-commit, labelled with THAT op's `MutationKind` (Q1/Q2) — so a
/// fact-delete-triggered retraction is attributed to `fact_delete`, not `entity_delete`.
async fn retract_neighbor(
    conn: &libsql::Connection,
    neighbor: &RetractedNeighbor,
    group_id: &str,
) -> Result<usize> {
    conn.execute(
        "DELETE FROM entity_communities WHERE group_id = ?1 AND entity_id = ?2",
        libsql::params![group_id, neighbor.entity_id.clone()],
    )
    .await?;
    reopen_freeze(conn, &neighbor.entity_id, group_id).await
}

/// Restore a retracted neighbour's community membership on undo (§4.4). Returns 1.
async fn restore_neighbor(
    conn: &libsql::Connection,
    neighbor: &RetractedNeighbor,
    group_id: &str,
) -> Result<usize> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO entity_communities (group_id, entity_id, community_id, updated_at) \
         VALUES (?1, ?2, ?3, ?4)",
        libsql::params![
            group_id,
            neighbor.entity_id.clone(),
            neighbor.prior_community_id,
            now
        ],
    )
    .await?;
    Ok(1)
}

/// Bundled params for [`write_delete_log`] — args-as-object (clippy threshold 3).
struct WriteDeleteLogParams<'a> {
    kind: MutationKind,
    group_id: &'a str,
    pre_state_json: &'a str,
    inputs_json: &'a str,
}

/// INSERT a delete-kind provenance row on `conn` (same txn, BEFORE the destructive
/// writes, §8.1) and return its id. `pre_state` / `inputs` are pre-serialized by the
/// caller (parse-loudly on serialize error there).
async fn write_delete_log(
    conn: &libsql::Connection,
    params: WriteDeleteLogParams<'_>,
) -> Result<i64> {
    let WriteDeleteLogParams {
        kind,
        group_id,
        pre_state_json,
        inputs_json,
    } = params;
    let now = chrono::Utc::now().to_rfc3339();
    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.graph.provenance",
            kind = kind.as_tag(),
            group_id = %group_id,
            pre_state = %pre_state_json,
            inputs = %inputs_json,
            "reversible-mutations delete snapshot captured (pre-mutate)"
        );
    }
    conn.execute(
        "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        libsql::params![kind.as_tag(), group_id, now, pre_state_json, inputs_json],
    )
    .await?;
    let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
    let Some(row) = rows.next().await? else {
        return Err(Error::InsertReturnedNoRowId {
            operation: "insert_delete_log",
        });
    };
    Ok(row.get::<i64>(0)?)
}

/// Serialize a snapshot / inputs value to JSON, mapping a serialize failure to a loud
/// `Error` (parse-loudly, §2.1 — our own structured emit, but a snapshot we cannot
/// serialize means the delete would be un-reversible).
fn to_json<T: serde::Serialize>(value: &T, what: &str) -> Result<String> {
    serde_json::to_string(value).map_err(|e| Error::Other(anyhow::anyhow!("serialize {what}: {e}")))
}

// ─── delete_entity (forward, §4.4) ───────────────────────────────────────────

/// Delete an entity, reversibly (§4.4). Archives its facts (recoverable), removes its
/// edges / community membership / FTS / row, retracts DERIVED artifacts of neighbours
/// whose support drops to zero, and snapshots everything undo needs — all in ONE
/// `BEGIN IMMEDIATE`.
///
/// # Errors
///
/// - [`Error::EntityDeleteNotFound`] if `entity_id` names no entity in the namespace.
/// - A serialize / DB error rolls the whole delete back (atomic with the snapshot).
#[doc(hidden)]
pub async fn delete_entity(
    graph: &TemporalGraph,
    params: DeleteEntityParams,
) -> Result<DeleteEntityOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match delete_entity_txn(graph, &params).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    // §8.1: the `graph_mutation_log` row + archive moves are durable in-txn writes
    // (correct on rollback); the summary counter fires only AFTER the real commit.
    if committed_here {
        counter!(
            "kremory.graph.mutation_logged_total",
            "kind" => "entity_delete",
            "source" => "delete",
        )
        .increment(1);
        // Q2: the retract-on-zero cascade retractions are
        // durable in-txn writes; their counter fires ONLY post-commit, and Q1: labelled
        // with THIS op's kind (`entity_delete`) so it is never mis-attributed.
        if outcome.neighbors_retracted > 0 {
            counter!(
                "kremory.graph.cascade_retracted_total",
                "kind" => "entity_delete",
                "artifact" => "community_membership",
            )
            .increment(outcome.neighbors_retracted as u64);
        }
    }
    Ok(outcome)
}

async fn delete_entity_txn(
    graph: &TemporalGraph,
    params: &DeleteEntityParams,
) -> Result<DeleteEntityOutcome> {
    let conn = &graph.conn;
    // Bind as `&str` (Copy) so the ids can be used across the many `libsql::params!`
    // sites without per-site clones (a `&String` would try to move into the macro).
    let entity_id = params.entity_id.as_str();
    let group_id = params.group_id.as_str();

    // Archiving E's facts then deleting E re-points nothing, but the entity row is
    // removed while archived rows still carry its id; defer FK checks to COMMIT for
    // the same atomicity reason as the edit cascade. Harmless no-op when FKs are OFF.
    defer_foreign_keys(conn).await?;

    // ── (1) Reads — capture everything the snapshot + undo need, BEFORE any write ──
    let entity_row = load_entity_row(conn, entity_id, group_id).await?;
    let prior_community_id = community_of(conn, entity_id, group_id).await?;

    // E's facts (ids to archive). Group-scoped.
    let mut fact_ids: Vec<i64> = Vec::new();
    {
        let mut rows = conn
            .query(
                "SELECT id FROM facts \
                 WHERE group_id = ?1 AND (subject_id = ?2 OR object_id = ?2)",
                libsql::params![group_id, entity_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            fact_ids.push(row.get::<i64>(0)?);
        }
    }

    // E's episodic edges (full non-key columns for the undo re-INSERT).
    let mut episodic_edges: Vec<DeletedEpisodicEdge> = Vec::new();
    {
        let mut rows = conn
            .query(
                "SELECT episode_id, entity_group_id, role, recorded_at FROM episodic_edges \
                 WHERE entity_id = ?1 AND entity_group_id = ?2",
                libsql::params![entity_id, group_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            episodic_edges.push(DeletedEpisodicEdge {
                episode_id: row.get::<i64>(0)?,
                entity_group_id: row.get::<String>(1)?,
                role: row.get::<String>(2)?,
                recorded_at: row.get::<String>(3)?,
            });
        }
    }

    // Retract-on-zero candidate set — cycle-safe reachability over the fact FKs
    // (§4.1). Exclude the seed E itself; a neighbour whose support drops to zero has
    // its DERIVED community membership retracted (base entity kept).
    let reachable = reachable_dependents(conn, entity_id, group_id).await?;
    let mut retracted_neighbors: Vec<RetractedNeighbor> = Vec::new();
    for cand in &reachable {
        if cand == entity_id {
            continue;
        }
        if let Some(n) = neighbor_retracted_on_entity_delete(
            conn,
            EntityRetractCheck {
                deleted_id: entity_id,
                candidate: cand,
                group_id,
            },
        )
        .await?
        {
            retracted_neighbors.push(n);
        }
    }

    // ── (2) Snapshot BEFORE the destructive writes (§8.1) ─────────────────────────
    let pre_state = EntityDeletePreState {
        entity_row,
        archived_fact_ids: fact_ids.clone(),
        episodic_edges: episodic_edges.clone(),
        prior_community_id,
        retracted_neighbors: retracted_neighbors.clone(),
    };
    let inputs = DeleteEntityInputs {
        entity_id: entity_id.to_string(),
        facts_retracted: fact_ids.len(),
    };
    let mutation_id = write_delete_log(
        conn,
        WriteDeleteLogParams {
            kind: MutationKind::EntityDelete,
            group_id,
            pre_state_json: &to_json(&pre_state, "entity_delete pre_state")?,
            inputs_json: &to_json(&inputs, "entity_delete inputs")?,
        },
    )
    .await?;

    // ── (3) Destructive writes ────────────────────────────────────────────────────
    for fid in &fact_ids {
        archive_fact(conn, *fid, group_id).await?;
    }
    // Drop E's own idempotency freeze keys (resolve rowid while the row still exists).
    let mut entities_reopened = reopen_freeze(conn, entity_id, group_id).await?;
    let edges_removed = conn
        .execute(
            "DELETE FROM episodic_edges WHERE entity_id = ?1 AND entity_group_id = ?2",
            libsql::params![entity_id, group_id],
        )
        .await? as usize;
    let communities_removed = conn
        .execute(
            "DELETE FROM entity_communities WHERE group_id = ?1 AND entity_id = ?2",
            libsql::params![group_id, entity_id],
        )
        .await? as usize;
    conn.execute(
        "DELETE FROM entities_fts WHERE entity_id = ?1",
        libsql::params![entity_id],
    )
    .await?;
    conn.execute(
        "DELETE FROM entities WHERE id = ?1 AND group_id = ?2",
        libsql::params![entity_id, group_id],
    )
    .await?;

    // Retract-on-zero cascade over neighbours (base entity NEVER deleted).
    for neighbor in &retracted_neighbors {
        entities_reopened += retract_neighbor(conn, neighbor, group_id).await?;
    }

    Ok(DeleteEntityOutcome {
        entity_id: entity_id.to_string(),
        facts_retracted: fact_ids.len(),
        edges_removed,
        communities_removed,
        neighbors_retracted: retracted_neighbors.len(),
        entities_reopened,
        mutation_id,
        already_undone: false,
    })
}

// ─── delete_fact (forward, §4.5) ─────────────────────────────────────────────

/// Delete a single fact, reversibly (§4.5). Archives it (recoverable via
/// `restore_archived_fact`) + retracts the DERIVED community membership of either
/// endpoint whose support drops to zero. All in ONE `BEGIN IMMEDIATE`.
///
/// ## Embedding caveat on undo (Q6, inherited archive tradeoff)
///
/// Archival intentionally drops `facts.embedding` (`restore_archived_fact` re-projects
/// only the 19 archived columns; embedding / `access_count` / `corroboration_inert`
/// take their column defaults). So [`undo_delete_fact`] returns the fact with a
/// column-default (NULL) embedding — it is present for lexical (FTS) recall but will
/// NOT match semantic (vector) recall until the fact is re-embedded. Documented, not
/// hidden: the tradeoff is inherited from the P2 archive projection, not introduced here.
///
/// # Errors
///
/// - [`Error::FactDeleteNotFound`] if `fact_id` names no fact.
#[doc(hidden)]
pub async fn delete_fact(graph: &TemporalGraph, fact_id: i64) -> Result<DeleteFactOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match delete_fact_txn(graph, fact_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    if committed_here {
        counter!(
            "kremory.graph.mutation_logged_total",
            "kind" => "fact_delete",
            "source" => "delete",
        )
        .increment(1);
        // Q2: cascade retractions counted ONLY post-commit,
        // Q1: labelled `fact_delete` (a fact-delete-triggered retraction, correctly
        // attributed — the same `retract_neighbor` helper serves both delete paths).
        if outcome.neighbors_retracted > 0 {
            counter!(
                "kremory.graph.cascade_retracted_total",
                "kind" => "fact_delete",
                "artifact" => "community_membership",
            )
            .increment(outcome.neighbors_retracted as u64);
        }
    }
    Ok(outcome)
}

async fn delete_fact_txn(graph: &TemporalGraph, fact_id: i64) -> Result<DeleteFactOutcome> {
    let conn = &graph.conn;
    defer_foreign_keys(conn).await?;

    // Load the fact's endpoints + group (a fact id is globally unique — its row
    // carries the namespace). Loud if absent.
    let (subject_id, object_id, group_id): (String, Option<String>, String) = {
        let mut rows = conn
            .query(
                "SELECT subject_id, object_id, group_id FROM facts WHERE id = ?1",
                libsql::params![fact_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::FactDeleteNotFound {
                detail: format!("no fact with id {fact_id}"),
            });
        };
        (
            row.get::<String>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<String>(2)?,
        )
    };

    // Retract-on-zero over the two endpoints (§4.5 — no deeper recursion in Stage 1;
    // derived-fact tracking is Stage 2's `derivation_edges`).
    let mut endpoints: Vec<String> = vec![subject_id.clone()];
    if let Some(obj) = &object_id {
        if obj != &subject_id {
            endpoints.push(obj.clone());
        }
    }
    let mut retracted_neighbors: Vec<RetractedNeighbor> = Vec::new();
    for ep in &endpoints {
        if let Some(n) = neighbor_retracted_on_fact_delete(
            conn,
            FactRetractCheck {
                endpoint: ep,
                fact_id,
                group_id: &group_id,
            },
        )
        .await?
        {
            retracted_neighbors.push(n);
        }
    }

    // Snapshot BEFORE the destructive writes (§8.1).
    let pre_state = FactDeletePreState {
        archived_fact_id: fact_id,
        subject_id: subject_id.clone(),
        object_id: object_id.clone(),
        retracted_neighbors: retracted_neighbors.clone(),
    };
    let inputs = DeleteFactInputs {
        fact_id,
        subject_id,
        object_id,
    };
    let mutation_id = write_delete_log(
        conn,
        WriteDeleteLogParams {
            kind: MutationKind::FactDelete,
            group_id: &group_id,
            pre_state_json: &to_json(&pre_state, "fact_delete pre_state")?,
            inputs_json: &to_json(&inputs, "fact_delete inputs")?,
        },
    )
    .await?;

    // Archive the fact (recoverable) + retract neighbours.
    archive_fact(conn, fact_id, &group_id).await?;
    let mut entities_reopened = 0usize;
    for neighbor in &retracted_neighbors {
        entities_reopened += retract_neighbor(conn, neighbor, &group_id).await?;
    }

    Ok(DeleteFactOutcome {
        fact_id,
        // Forward delete archives the fact; it restores nothing.
        fact_restored: false,
        neighbors_retracted: retracted_neighbors.len(),
        entities_reopened,
        mutation_id,
        already_undone: false,
    })
}

// ─── undo_delete_entity (reversal, §4.4) ─────────────────────────────────────

/// Reverse a prior `delete_entity` from its `graph_mutation_log` snapshot (§4.4):
/// re-INSERT the entity row + FTS, restore its archived facts, re-insert its episodic
/// edges, restore its OWN + the retracted neighbours' community memberships.
/// Idempotent: an already-undone delete returns a zero-count outcome. A missing /
/// non-`entity_delete` `mutation_id`, or a `pre_state` that fails to deserialize, is a
/// hard `Error` (parse-loudly). All in ONE `BEGIN IMMEDIATE`.
#[doc(hidden)]
pub async fn undo_delete_entity(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<DeleteEntityOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match undo_delete_entity_txn(graph, mutation_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    if committed_here && !outcome.already_undone {
        counter!(
            "kremory.graph.mutation_undone_total",
            "kind" => "entity_delete",
        )
        .increment(1);
    }
    Ok(outcome)
}

async fn undo_delete_entity_txn(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<DeleteEntityOutcome> {
    let conn = &graph.conn;
    defer_foreign_keys(conn).await?;

    let (group_id, undone_at, pre_state_json): (String, Option<String>, String) = {
        let mut rows = conn
            .query(
                "SELECT group_id, undone_at, pre_state FROM graph_mutation_log \
                 WHERE id = ?1 AND kind = 'entity_delete'",
                libsql::params![mutation_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::EntityDeleteNotFound {
                detail: format!(
                    "no entity_delete mutation-log row with id {mutation_id} — \
                     cannot reverse a delete that was never logged"
                ),
            });
        };
        (
            row.get::<String>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<String>(2)?,
        )
    };

    let pre: EntityDeletePreState = serde_json::from_str(&pre_state_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "undo_delete_entity: deserialize pre_state for mutation {mutation_id}: {e}"
        ))
    })?;
    let entity_id = pre.entity_row.id.clone();

    // Idempotent: already reversed → zero-count no-op.
    if undone_at.is_some() {
        return Ok(DeleteEntityOutcome {
            entity_id,
            facts_retracted: 0,
            edges_removed: 0,
            communities_removed: 0,
            neighbors_retracted: 0,
            entities_reopened: 0,
            mutation_id,
            already_undone: true,
        });
    }

    // (a) Re-INSERT the entity row (11 live columns) + FTS shadow (label always '';
    //     FTS reconstructed from the snapshotted properties, §4.2 step 1).
    let lr = &pre.entity_row;
    let embedding_val: libsql::Value = match lr.embedding_b64.as_deref() {
        Some(b64) => libsql::Value::Blob(BASE64_STANDARD.decode(b64).map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "undo_delete_entity: decode `{entity_id}` embedding base64: {e}"
            ))
        })?),
        None => libsql::Value::Null,
    };
    conn.execute(
        "INSERT INTO entities \
         (id, group_id, properties, embedding, recorded_at, updated_at, access_count, \
          entity_type_id, entity_type_source, entity_type_assigned_at, ner_confidence) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        libsql::params![
            lr.id.clone(),
            lr.group_id.clone(),
            lr.properties.clone(),
            embedding_val,
            lr.recorded_at.clone(),
            lr.updated_at.clone(),
            lr.access_count,
            lr.entity_type_id,
            lr.entity_type_source.clone(),
            lr.entity_type_assigned_at.clone(),
            lr.ner_confidence,
        ],
    )
    .await?;
    conn.execute(
        "INSERT INTO entities_fts (entity_id, label, properties) VALUES (?1, '', ?2)",
        libsql::params![lr.id.clone(), lr.properties.clone().unwrap_or_default()],
    )
    .await?;

    // (b) Restore the archived facts (facts_archive → facts + facts_fts). Reuses the
    //     tested `restore_archived_fact` helper; the nested guard is a no-op that
    //     participates in this txn (no double-count).
    let mut facts_restored = 0usize;
    for fid in &pre.archived_fact_ids {
        // Honest count (§3.1): `restore_archived_fact` no-ops when the fact is ALREADY
        // live (its `already_live` flag). Count only facts this undo actually moved
        // back out of the archive — never a no-op restore.
        let restored = restore_archived_fact(graph, *fid).await?;
        if !restored.already_live {
            facts_restored += 1;
        }
    }

    // (c) Re-insert the entity's episodic edges.
    let mut edges_restored = 0usize;
    for edge in &pre.episodic_edges {
        conn.execute(
            "INSERT INTO episodic_edges \
             (episode_id, entity_id, entity_group_id, role, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                edge.episode_id,
                entity_id.clone(),
                edge.entity_group_id.clone(),
                edge.role.clone(),
                edge.recorded_at.clone(),
            ],
        )
        .await?;
        edges_restored += 1;
    }

    // (d) Restore the entity's OWN community membership.
    let communities_restored = if let Some(cid) = pre.prior_community_id {
        restore_neighbor(
            conn,
            &RetractedNeighbor {
                entity_id: entity_id.clone(),
                prior_community_id: cid,
            },
            &group_id,
        )
        .await?
    } else {
        0
    };

    // (e) Restore the retracted neighbours' community memberships.
    let mut neighbors_restored = 0usize;
    for neighbor in &pre.retracted_neighbors {
        neighbors_restored += restore_neighbor(conn, neighbor, &group_id).await?;
    }

    // (f) Re-open freeze on the restored entity + neighbours (idempotent; the restored
    //     entity gets a NEW rowid with no cached key regardless).
    let mut entities_reopened = reopen_freeze(conn, &entity_id, &group_id).await?;
    for neighbor in &pre.retracted_neighbors {
        entities_reopened += reopen_freeze(conn, &neighbor.entity_id, &group_id).await?;
    }

    // (g) Mark reversed.
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE graph_mutation_log SET undone_at = ?1 WHERE id = ?2",
        libsql::params![now, mutation_id],
    )
    .await?;

    Ok(DeleteEntityOutcome {
        entity_id,
        facts_retracted: facts_restored,
        edges_removed: edges_restored,
        communities_removed: communities_restored,
        neighbors_retracted: neighbors_restored,
        entities_reopened,
        mutation_id,
        already_undone: false,
    })
}

// ─── undo_delete_fact (reversal, §4.5) ───────────────────────────────────────

/// Reverse a prior `delete_fact` from its snapshot (§4.5): restore the archived fact +
/// un-retract any neighbour whose community membership the delete retracted.
/// Idempotent; parse-loudly on a missing / undeserializable row.
///
/// Q6: the restored fact returns with a column-default (NULL) embedding — an inherited
/// archive tradeoff (`facts.embedding` is dropped on archival), so it will not match
/// semantic recall until re-embedded. See [`delete_fact`] for the full caveat.
#[doc(hidden)]
pub async fn undo_delete_fact(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<DeleteFactOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match undo_delete_fact_txn(graph, mutation_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    if committed_here && !outcome.already_undone {
        counter!(
            "kremory.graph.mutation_undone_total",
            "kind" => "fact_delete",
        )
        .increment(1);
    }
    Ok(outcome)
}

async fn undo_delete_fact_txn(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<DeleteFactOutcome> {
    let conn = &graph.conn;
    defer_foreign_keys(conn).await?;

    let (group_id, undone_at, pre_state_json): (String, Option<String>, String) = {
        let mut rows = conn
            .query(
                "SELECT group_id, undone_at, pre_state FROM graph_mutation_log \
                 WHERE id = ?1 AND kind = 'fact_delete'",
                libsql::params![mutation_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::FactDeleteNotFound {
                detail: format!(
                    "no fact_delete mutation-log row with id {mutation_id} — \
                     cannot reverse a delete that was never logged"
                ),
            });
        };
        (
            row.get::<String>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<String>(2)?,
        )
    };

    let pre: FactDeletePreState = serde_json::from_str(&pre_state_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "undo_delete_fact: deserialize pre_state for mutation {mutation_id}: {e}"
        ))
    })?;

    if undone_at.is_some() {
        return Ok(DeleteFactOutcome {
            fact_id: pre.archived_fact_id,
            // Whole mutation already reversed → this call restored nothing.
            fact_restored: false,
            neighbors_retracted: 0,
            entities_reopened: 0,
            mutation_id,
            already_undone: true,
        });
    }

    // Restore the archived fact (facts_archive → facts + facts_fts). Bind the outcome
    // so the returned `fact_restored` is HONEST: `restore_archived_fact` no-ops when
    // the fact is ALREADY live (nothing moved out of the archive), so a no-op restore
    // is not reported as a real one.
    let restored = restore_archived_fact(graph, pre.archived_fact_id).await?;

    // Un-retract the neighbours' community memberships + re-open their freeze.
    let mut neighbors_restored = 0usize;
    let mut entities_reopened = 0usize;
    for neighbor in &pre.retracted_neighbors {
        neighbors_restored += restore_neighbor(conn, neighbor, &group_id).await?;
        entities_reopened += reopen_freeze(conn, &neighbor.entity_id, &group_id).await?;
    }

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE graph_mutation_log SET undone_at = ?1 WHERE id = ?2",
        libsql::params![now, mutation_id],
    )
    .await?;

    Ok(DeleteFactOutcome {
        fact_id: pre.archived_fact_id,
        // Honest: `true` only if the fact was genuinely moved out of the archive here
        // (`false` if `restore_archived_fact` found it already live — a no-op restore).
        fact_restored: !restored.already_live,
        neighbors_retracted: neighbors_restored,
        entities_reopened,
        mutation_id,
        already_undone: false,
    })
}
