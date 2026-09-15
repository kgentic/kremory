//! Reversal core for the Stage-1 reversible-graph-mutations substrate.
//!
//! Sub-phase 1c. This module makes a merge REVERSIBLE:
//!
//! - [`unmerge`] — replays the `graph_mutation_log` snapshot to fully reverse an
//!   `entity_merge` (§4.2): re-INSERTs the loser row + FTS, restores the keeper's
//!   overwritten `access_count` / `ner_confidence`, un-repoints every fact
//!   (restoring the PRIOR `corroboration_inert`, not blanket-clearing — §12 CH-3),
//!   re-inserts/re-points the loser's episodic edges by their `collided` flag, drops
//!   the reconciler-freeze idempotency keys so the pass re-processes, writes the
//!   anti-re-merge nogood, and stamps `undone_at`. All in ONE `BEGIN IMMEDIATE`.
//! - [`load_merge_nogoods`] — the sorted-pair set consulted at all THREE merge sites
//!   (§6.2 V3 fix) so an unmerged pair is never silently re-merged by the next
//!   `dream()`.
//! - [`restore_archived_fact`] (§4.4) / [`unsupersede`] (§4.5) — the two additive
//!   reversal helpers (archive-restore + supersession-bound clear).
//!
//! Every reversal is DETERMINISTIC (no LLM), so its restore is EXACT and it sits at
//! the fast test tier with no VCR (`llm-test-pyramid-vcr-seams`).
//!
//! ## `pub` + `#[doc(hidden)]` (MNT-002 precedent)
//!
//! The reversal free-functions are `pub` + `#[doc(hidden)]` (re-exported under
//! `feature = "test-utils"` from `core/dream/mod.rs`) for the same E0365 reason as
//! `acronym_nickname_recall` / `cross_episode` / `archive`: external integration-test
//! binaries cannot import `pub(crate)` items. They are NOT part of the stable public
//! API — the consumer surface is the `Memory` facade (`facade/reverse.rs`).
//!
//! ## Freeze re-open (§6.1) — idempotency-key drop only, deliberately
//!
//! `unmerge` re-opens the reconciler freeze via the `dream_idempotency_keys` drop
//! (§6.1(b), the amended INTEGER-rowid fix). It deliberately does **NOT** reset the
//! reclassify `entity_type_source` stamp to a `'reopened'` sentinel as §6.1(a)
//! sketched: (a) the `entities.entity_type_source` CHECK constraint
//! (`defs_g2.rs:190-193`) only admits `Phase1Ner | Phase2Llm | DreamPass0 | DreamPass1
//! | ConsumerPinned | DreamPass4` — `'reopened'` would be a hard CHECK violation; and
//! (b) writing ANY value over the restored loser's `entity_type_source` would BREAK
//! the byte-identical-restore invariant the unmerge test asserts (the loser's source
//! is restored verbatim from the snapshot). The idempotency-key drop is sufficient:
//! the re-INSERTed loser gets a NEW rowid with no cached key, and the keeper's key is
//! dropped because its neighbourhood changed — both re-enter the next pass.

use std::collections::HashSet;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use metrics::counter;

use crate::core::error::{Error, Result};
use crate::core::schema::TemporalGraph;

use super::{
    EntityMergePreState, FactArchivePreState, FactEndpoint, FactSupersedeInputs,
    FactSupersedePreState, MergeInputs, RestoreArchivedOutcome, UnmergeOutcome, UnsupersedeOutcome,
};

// ─── nogood key ─────────────────────────────────────────────────────────────

/// The SORTED unordered pair used as the anti-re-merge nogood key (§6.2). Sorting
/// makes the key stable under a keeper/loser role-flip between passes, so a merge
/// applied `(a→b)` and its unmerge ban `(b, a)` map to the SAME key.
pub(crate) fn sorted_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Load the anti-re-merge nogood set for `group_id` (§6.2): the sorted pairs an
/// `unmerge` recorded into `merge_nogood`. Called ONCE per dream pass (like
/// `load_entity_slots`), NOT per-candidate. A merge site consults
/// `nogoods.contains(&sorted_pair(a, b))` before every `apply_merge_with_audit`.
#[doc(hidden)]
pub async fn load_merge_nogoods(
    graph: &TemporalGraph,
    group_id: &str,
) -> Result<HashSet<(String, String)>> {
    let mut set = HashSet::new();
    let mut rows = graph
        .conn
        .query(
            "SELECT pair_lo, pair_hi FROM merge_nogood WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await?;
    while let Some(row) = rows.next().await? {
        set.insert((row.get::<String>(0)?, row.get::<String>(1)?));
    }
    Ok(set)
}

// ─── unmerge (§4.2) ─────────────────────────────────────────────────────────

/// Reverse a prior `entity_merge` (§4.2), fully restoring the loser entity, its
/// facts, its episodic edges, and the keeper's overwritten `access_count` /
/// `ner_confidence`, then record a merge NOGOOD so the next dream will NOT re-merge
/// the split pair (§6.2).
///
/// Idempotent: if the mutation-log row's `undone_at` is already set, returns an
/// [`UnmergeOutcome`] with `already_undone = true` and all-zero counts — never a
/// double-restore. A missing / non-`entity_merge` `mutation_id`, or a `pre_state`
/// that fails to deserialize, is a hard `Error` (parse-loudly, §2.1) — an
/// un-reversible log row must never silently no-op.
///
/// **LIFO-only for chained merges.** When an entity is merged
/// more than once over a SHARED endpoint (e.g. `A→K1` then `K1→K2`), the merges
/// MUST be unwound Last-In-First-Out. Attempting to unmerge an earlier merge
/// while a later merge still chains on one of its endpoints returns
/// [`Error::UnmergeOutOfOrder`] (naming the blocking later `mutation_id`) rather
/// than corrupting the fact chain — unmerge the later chained merge(s) first.
///
/// The whole reversal runs in ONE `BEGIN IMMEDIATE` (or participates in the caller's
/// open txn); the summary counter fires only in the post-commit `Ok` arm (§8.1).
#[doc(hidden)]
pub async fn unmerge(graph: &TemporalGraph, mutation_id: i64) -> Result<UnmergeOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match unmerge_txn(graph, mutation_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    // §8.1: the durable writes (loser re-INSERT, undone_at, nogood) commit atomically
    // in the txn; the summary counter fires only AFTER commit, and only for an ACTUAL
    // reversal (an already-undone no-op must not inflate the undo counter).
    if committed_here && !outcome.already_undone {
        counter!(
            "kremory.graph.mutation_undone_total",
            "kind" => "entity_merge",
        )
        .increment(1);
    }
    Ok(outcome)
}

/// The transactional body of [`unmerge`] — every statement runs on `graph.conn`
/// inside the caller's `BeginGuard` so the reversal is atomic.
async fn unmerge_txn(graph: &TemporalGraph, mutation_id: i64) -> Result<UnmergeOutcome> {
    let conn = &graph.conn;

    // (a) Load the mutation-log row (kind='entity_merge'); loud if absent.
    let (group_id, undone_at, pre_state_json): (String, Option<String>, String) = {
        let mut rows = conn
            .query(
                "SELECT group_id, undone_at, pre_state FROM graph_mutation_log \
                 WHERE id = ?1 AND kind = 'entity_merge'",
                libsql::params![mutation_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "unmerge: no entity_merge mutation-log row with id {mutation_id} — \
                 cannot reverse a merge that was never logged"
            )));
        };
        (
            row.get::<String>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<String>(2)?,
        )
    };

    // parse-loudly (§2.1): our own structured emit, but a corrupt snapshot means the
    // merge is un-reversible — hard error, never a silent partial restore.
    let pre: EntityMergePreState = serde_json::from_str(&pre_state_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "unmerge: deserialize pre_state for mutation {mutation_id}: {e}"
        ))
    })?;
    let loser_id = pre.loser_entity_row.id.clone();
    let keeper_id = pre.keeper_pre.id.clone();

    // Idempotency (§3.1): already reversed → no-op with honest zeroes.
    if undone_at.is_some() {
        return Ok(UnmergeOutcome {
            restored_entity: loser_id,
            keeper: keeper_id,
            facts_repointed: 0,
            edges_restored: 0,
            entities_reopened: 0,
            nogood_recorded: false,
            already_undone: true,
        });
    }

    // M1 (LIFO guard, §4.2): a merge whose keeper or loser was RE-USED as an
    // endpoint by a LATER still-live merge is CHAINED — that later merge
    // re-pointed facts/edges THROUGH the shared endpoint, so reversing THIS one
    // first would leave a broken fact chain. Reject loudly and instruct LIFO
    // order; the consumer unmerges the later chained merge(s) first. Scoped to
    // `group_id` (namespace) — a re-used id in a different namespace is a
    // different entity and cannot chain (§6.2 namespace-scoping).
    if let Some(blocking_mutation_id) = later_chained_merge(
        conn,
        LaterChainedMergeParams {
            mutation_id,
            group_id: &group_id,
            loser_id: &loser_id,
            keeper_id: &keeper_id,
        },
    )
    .await?
    {
        return Err(Error::UnmergeOutOfOrder {
            mutation_id,
            blocking_mutation_id,
        });
    }

    // (b) Re-INSERT the loser entity row (all 11 live columns, §2.3(1)) + FTS.
    let lr = &pre.loser_entity_row;
    let embedding_val: libsql::Value = match lr.embedding_b64.as_deref() {
        Some(b64) => libsql::Value::Blob(BASE64_STANDARD.decode(b64).map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "unmerge: decode loser `{loser_id}` embedding base64: {e}"
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
    // FTS shadow: `entities_fts.label` is always '' (the `entities.label` column was
    // dropped by Mig 009), so FTS is reconstructed from the snapshotted `properties`
    // only (§4.2 step 1, `entities.rs:68`).
    conn.execute(
        "INSERT INTO entities_fts (entity_id, label, properties) VALUES (?1, '', ?2)",
        libsql::params![lr.id.clone(), lr.properties.clone().unwrap_or_default()],
    )
    .await?;

    // (c) Restore the keeper's PRE-merge access_count + ner_confidence (§4.2 step 2)
    //     — SET the snapshotted values (noisy-OR is non-invertible; subtracting is
    //     lossy), NOT subtract the loser back.
    let keeper_restored = conn
        .execute(
            "UPDATE entities SET access_count = ?1, ner_confidence = ?2 \
             WHERE id = ?3 AND group_id = ?4",
            libsql::params![
                pre.keeper_pre.access_count,
                pre.keeper_pre.ner_confidence,
                keeper_id.clone(),
                group_id.clone(),
            ],
        )
        .await?;
    // LOUD-on-stale (§4.2, honest-outcome): the keeper is the entity the merge
    // folded the loser INTO — its overwritten access_count / ner_confidence are what
    // this UPDATE sets back. If the keeper was RENAMED or DELETED between the merge
    // and this unmerge (a cross-kind chain, e.g. `merge(A→B) → rename(B→C) →
    // unmerge`), the UPDATE matches ZERO rows: the restore silently no-ops and the
    // returned `UnmergeOutcome` would name a `keeper` that no longer exists — a false
    // success masking cross-kind-chain corruption. Fail LOUD instead; the whole
    // reversal rolls back (the loser re-INSERT above is undone with it) so the
    // consumer learns the merge is no longer cleanly reversible.
    if keeper_restored == 0 {
        return Err(Error::UndoStale {
            mutation_id,
            reason: format!(
                "keeper entity `{keeper_id}` no longer exists in namespace \
                 `{group_id}` (renamed or deleted after the merge) — cannot restore \
                 its pre-merge access_count / ner_confidence"
            ),
        });
    }

    // (d) Un-repoint facts keeper→loser + restore the PRIOR corroboration_inert
    //     (§4.2 step 3) — a fact inert BEFORE the merge stays inert; one made inert
    //     BY the merge is cleared (the monotone-undo trap, §12 CH-3).
    let mut facts_repointed = 0usize;
    for rf in &pre.repointed_facts {
        let sql = match rf.endpoint {
            FactEndpoint::Subject => {
                "UPDATE facts SET subject_id = ?1, corroboration_inert = ?2 WHERE id = ?3"
            }
            FactEndpoint::Object => {
                "UPDATE facts SET object_id = ?1, corroboration_inert = ?2 WHERE id = ?3"
            }
        };
        // Gate the count on the affected-row count, matching the non-collided edges
        // path below (L1, honest outcome): a fact externally DELETEd between merge and
        // unmerge affects ZERO rows — never claim a re-point that didn't happen.
        let updated = conn
            .execute(
                sql,
                libsql::params![loser_id.clone(), rf.prior_corroboration_inert, rf.fact_id],
            )
            .await?;
        if updated > 0 {
            facts_repointed += 1;
        }
    }

    // (d2) D1 — REVIVE the facts the merge expired because re-pointing
    //      collapsed both endpoints onto the keeper. Ordered immediately after
    //      (d) and NOT before it: while the endpoints still both read `keeper`
    //      the fact is a meaningless self-loop, so un-expiring first would make
    //      a `keeper pred keeper` row briefly live. (d) has just pointed one
    //      endpoint back at the loser, so the fact is meaningful again here.
    //
    //      ⚠️ The `expired_at IS NOT NULL` predicate does NOT discriminate by
    //      WHICH mechanism expired the row — it clears the stamp on any expired
    //      row in the list, and merely no-ops on one that is already live. An
    //      earlier version of this comment claimed it protected against reviving
    //      a fact expired by a later unrelated mechanism; it does
    //      not, and saying so would be a guarantee the code has not got.
    //
    //      What ACTUALLY closes that window is two facts, both external to this
    //      statement: `unmerge` short-circuits on `undone_at` (:179), so this
    //      block runs at most ONCE per merge; and every other expiring mechanism
    //      filters `expired_at IS NULL`, so it cannot re-expire a row this merge
    //      already tombstoned. If either ever stops holding, this needs the
    //      merge's expiry timestamp in `pre_state` and an equality check on it.
    let mut self_loops_revived = 0usize;
    for fact_id in &pre.self_loops_expired {
        let updated = conn
            .execute(
                "UPDATE facts SET expired_at = NULL WHERE id = ?1 AND expired_at IS NOT NULL",
                libsql::params![*fact_id],
            )
            .await?;
        if updated > 0 {
            self_loops_revived += 1;
        }
    }
    if self_loops_revived > 0 {
        tracing::info!(
            target: "kremory.unmerge",
            loser_id = %loser_id,
            revived = self_loops_revived,
            "kremory.unmerge.self_loop_facts_revived"
        );
    }

    // (e) Episodic edges (§4.2 step 4): collided → re-INSERT the dropped loser edge
    //     (keeper's untouched); non-collided → re-point the remapped row back to the
    //     loser (exactly one such row exists — non-collision means keeper had none).
    let mut edges_restored = 0usize;
    for edge in &pre.episodic_edges {
        if edge.collided {
            // Collided → re-INSERT the dropped loser row. INSERT always adds a
            // row (or errors loudly), so this path counts unconditionally (L1).
            conn.execute(
                "INSERT INTO episodic_edges \
                 (episode_id, entity_id, entity_group_id, role, recorded_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                libsql::params![
                    edge.episode_id,
                    loser_id.clone(),
                    edge.entity_group_id.clone(),
                    edge.cols.role.clone(),
                    edge.cols.recorded_at.clone(),
                ],
            )
            .await?;
            edges_restored += 1;
        } else {
            // Non-collided → re-point the remapped row back to the loser. If that
            // row was externally DELETEd between merge and unmerge, the UPDATE
            // affects 0 rows — count only an ACTUAL re-point (L1, honest outcome),
            // never claim a restore that didn't happen.
            let updated = conn
                .execute(
                    "UPDATE episodic_edges SET entity_id = ?1 \
                     WHERE episode_id = ?2 AND entity_id = ?3 AND entity_group_id = ?4",
                    libsql::params![
                        loser_id.clone(),
                        edge.episode_id,
                        keeper_id.clone(),
                        edge.entity_group_id.clone(),
                    ],
                )
                .await?;
            if updated > 0 {
                edges_restored += 1;
            }
        }
    }

    // (f) Freeze re-open (§6.1(b)): drop the idempotency keys so the pass
    //     re-processes. Resolve the CURRENT rowid — the loser was just re-INSERTed
    //     (NEW rowid); the keeper's neighbourhood changed. `entity_id` is the INTEGER
    //     rowid (defs_g1.rs:80), NOT the TEXT id (the amended freeze-reopen key fix).
    let mut entities_reopened = 0usize;
    for eid in [&loser_id, &keeper_id] {
        let rowid: Option<i64> = {
            let mut rows = conn
                .query(
                    "SELECT rowid FROM entities WHERE id = ?1 AND group_id = ?2",
                    libsql::params![eid.clone(), group_id.clone()],
                )
                .await?;
            match rows.next().await? {
                Some(row) => Some(row.get::<i64>(0)?),
                None => None,
            }
        };
        if let Some(rid) = rowid {
            let dropped = conn
                .execute(
                    "DELETE FROM dream_idempotency_keys WHERE entity_id = ?1",
                    libsql::params![rid],
                )
                .await?;
            if dropped > 0 {
                entities_reopened += 1;
            }
        }
    }

    // (g) Record the anti-re-merge nogood (§6.2) — the sorted pair, namespace-scoped,
    //     durable the instant this txn commits. `INSERT OR IGNORE` keeps it idempotent
    //     under the composite PK.
    let (lo, hi) = sorted_pair(&loser_id, &keeper_id);
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR IGNORE INTO merge_nogood (pair_lo, pair_hi, group_id, created_at) \
         VALUES (?1, ?2, ?3, ?4)",
        libsql::params![lo, hi, group_id.clone(), now.clone()],
    )
    .await?;

    // (h) Mark the mutation reversed.
    conn.execute(
        "UPDATE graph_mutation_log SET undone_at = ?1 WHERE id = ?2",
        libsql::params![now, mutation_id],
    )
    .await?;

    Ok(UnmergeOutcome {
        restored_entity: loser_id,
        keeper: keeper_id,
        facts_repointed,
        edges_restored,
        entities_reopened,
        nogood_recorded: true,
        already_undone: false,
    })
}

/// Bundled parameters for [`later_chained_merge`] — args-as-object
/// (`clippy.toml` `too-many-arguments-threshold = 3`; `#[allow]` banned in src).
/// `conn` stays a lead positional param (receiver-like, project convention).
struct LaterChainedMergeParams<'a> {
    mutation_id: i64,
    group_id: &'a str,
    loser_id: &'a str,
    keeper_id: &'a str,
}

/// M1 LIFO guard (§4.2): find the earliest LATER (higher-id) still-live
/// `entity_merge` in `group_id` whose endpoint pair shares `loser_id` or
/// `keeper_id` — the direct chain successor that must be unmerged FIRST (LIFO).
/// Returns its `mutation_id`, or `None` when no later live merge chains on this
/// pair.
///
/// Ordered by `id ASC` so the *immediate* successor is reported: unmerging it
/// first (then re-invoking on this one) walks the chain top-down correctly, even
/// for a 3+-deep chain where only the direct neighbour shares an endpoint (the
/// transitive links surface one hop at a time as each layer is peeled).
///
/// Parse-loudly (§2.1): a later row whose `inputs` fail to deserialize is a hard
/// `Error` — a merge we cannot classify must never be silently treated as a
/// non-blocker (that would permit the corrupting out-of-order unmerge this guard
/// exists to prevent).
async fn later_chained_merge(
    conn: &libsql::Connection,
    params: LaterChainedMergeParams<'_>,
) -> Result<Option<i64>> {
    let LaterChainedMergeParams {
        mutation_id,
        group_id,
        loser_id,
        keeper_id,
    } = params;
    let mut rows = conn
        .query(
            "SELECT id, inputs FROM graph_mutation_log \
             WHERE kind = 'entity_merge' AND group_id = ?1 \
               AND undone_at IS NULL AND id > ?2 \
             ORDER BY id ASC",
            libsql::params![group_id, mutation_id],
        )
        .await?;
    while let Some(row) = rows.next().await? {
        let later_id = row.get::<i64>(0)?;
        let inputs_json = row.get::<String>(1)?;
        let inputs: MergeInputs = serde_json::from_str(&inputs_json).map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "unmerge LIFO guard: deserialize inputs for later merge {later_id}: {e}"
            ))
        })?;
        // Does this later merge re-use EITHER of our endpoints as EITHER of its
        // endpoints? (keeper/loser role is irrelevant — any shared endpoint means
        // it re-pointed facts through it.)
        if inputs.keeper == loser_id
            || inputs.keeper == keeper_id
            || inputs.loser == loser_id
            || inputs.loser == keeper_id
        {
            return Ok(Some(later_id));
        }
    }
    Ok(None)
}

// ─── restore_archived_fact (§4.4 / §3.2) ────────────────────────────────────

/// Restore a fact previously moved to `facts_archive` (P2 archival) back into
/// `facts` + its FTS shadow (§3.2). Idempotent: if the fact is already live returns
/// `already_live = true` and writes nothing. A missing `facts_archive` row is a hard
/// `Error` (parse-loudly) — restoring a non-existent archived id is a caller bug.
///
/// The re-INSERT projects the SAME 19 columns the archive op stored (`archive.rs`
/// `ARCHIVE_INSERT_SQL`, reversed) — `facts.embedding` / `access_count` /
/// `corroboration_inert` were intentionally dropped on archival and take their column
/// defaults on restore.
#[doc(hidden)]
pub async fn restore_archived_fact(
    graph: &TemporalGraph,
    archived_fact_id: i64,
) -> Result<RestoreArchivedOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match restore_archived_txn(graph, archived_fact_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    if committed_here {
        let label = if outcome.already_live {
            "already_live"
        } else {
            "applied"
        };
        counter!("kremory.graph.restore_archived_total", "outcome" => label).increment(1);
    }
    Ok(outcome)
}

/// Reverse a LOGGED `fact_archive` mutation by its `mutation_id` (TD-250).
///
/// The domain-id door — `restore_archived_fact(graph, archived_fact_id)` — has
/// always existed and was always uncallable from outside, because nothing public
/// returned an `archived_fact_id`. Now that the archive op logs its mutation, a
/// consumer reaches this the same way it reaches every other reversal: iterate
/// `list_mutations`, pass the `mutation_id` to `undo`.
///
/// Marks `undone_at` on the log row, so the reversal shows up in
/// `mutation_history` like its four siblings rather than silently succeeding and
/// leaving the record reading un-reversed. Idempotent: a row already marked
/// undone returns `already_live = true` and writes nothing.
///
/// # Errors
///
/// - [`Error::MutationNotFound`] — `mutation_id` names no `fact_archive` row.
/// - Whatever `restore_archived_fact` returns.
pub async fn undo_fact_archive(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<RestoreArchivedOutcome> {
    let (undone_at, pre_state_json): (Option<String>, String) = {
        let mut rows = graph
            .conn
            .query(
                "SELECT undone_at, pre_state FROM graph_mutation_log \
                 WHERE id = ?1 AND kind = 'fact_archive'",
                libsql::params![mutation_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::MutationNotFound { mutation_id });
        };
        (row.get::<Option<String>>(0)?, row.get::<String>(1)?)
    };

    let pre: FactArchivePreState = serde_json::from_str(&pre_state_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "undo_fact_archive: deserialize pre_state for mutation {mutation_id}: {e}"
        ))
    })?;

    if undone_at.is_some() {
        return Ok(RestoreArchivedOutcome {
            restored_fact_id: pre.archived_fact_id,
            // Already reversed → this call restored nothing, and the fact is live.
            already_live: true,
        });
    }

    let outcome = restore_archived_fact(graph, pre.archived_fact_id).await?;
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "UPDATE graph_mutation_log SET undone_at = ?1 WHERE id = ?2",
            libsql::params![now, mutation_id],
        )
        .await?;
    Ok(outcome)
}

/// Reverse of `archive.rs::ARCHIVE_INSERT_SQL`: the 19 columns the archive op moved
/// `facts → facts_archive`, projected back `facts_archive → facts`.
const RESTORE_INSERT_SQL: &str = "INSERT INTO facts \
     (id, subject_id, predicate, object_id, object_value, properties, \
      valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, \
      source_episode_id, memory_type, content_hash, subject_group_id, object_group_id, \
      is_dream_generated) \
     SELECT id, subject_id, predicate, object_id, object_value, properties, \
      valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, \
      source_episode_id, memory_type, content_hash, subject_group_id, object_group_id, \
      is_dream_generated \
     FROM facts_archive WHERE id = ?1";

async fn restore_archived_txn(
    graph: &TemporalGraph,
    archived_fact_id: i64,
) -> Result<RestoreArchivedOutcome> {
    let conn = &graph.conn;

    // F44 cause-fix: check `facts` (already-live, idempotent) FIRST, before
    // checking `facts_archive` presence. A successful restore's LAST step
    // deletes the `facts_archive` row (below) — so the NATURAL double-call a
    // consumer would try first (`restore_archived_fact(id)` twice) used to
    // hit the (then-first) `facts_archive` presence check on the SECOND
    // call, find nothing (the first call already deleted it), and throw a
    // hard error, never reaching this idempotent branch at all. Checking
    // `facts` first makes the double-call return `already_live: true` as
    // documented; a genuinely bogus id (never archived, not live either)
    // still falls through to the `facts_archive` presence check below and
    // errors loudly.
    let already_live = {
        let mut rows = conn
            .query(
                "SELECT 1 FROM facts WHERE id = ?1",
                libsql::params![archived_fact_id],
            )
            .await?;
        rows.next().await?.is_some()
    };
    if already_live {
        // Idempotent cleanup (L2): the fact is already live (the source of truth).
        // DELETE is a no-op if no `facts_archive` row exists (the natural
        // double-call case, post F44); it clears a REAL row when some other
        // reversal path re-inserted the live row without cleaning up the
        // archive. Safe either way: the live `facts` row already carries the
        // data.
        conn.execute(
            "DELETE FROM facts_archive WHERE id = ?1",
            libsql::params![archived_fact_id],
        )
        .await?;
        return Ok(RestoreArchivedOutcome {
            restored_fact_id: archived_fact_id,
            already_live: true,
        });
    }

    // Not live AND no archive row → loud error (§2.1 parse-loudly extended to
    // reversal input) — a genuinely never-archived-or-live id is a caller bug.
    let in_archive = {
        let mut rows = conn
            .query(
                "SELECT 1 FROM facts_archive WHERE id = ?1",
                libsql::params![archived_fact_id],
            )
            .await?;
        rows.next().await?.is_some()
    };
    if !in_archive {
        // No mid-txn counter here (rollback-overcount): this runs
        // INSIDE `restore_archived_txn`, and when nested under `undo_delete_*` the
        // OUTER txn rolls back on this `Err` while a fired increment could NOT — it
        // would overcount a "restore" that never durably happened. The `Err`
        // propagation IS the honest signal; the post-commit `applied` / `already_live`
        // counter in `restore_archived_fact` covers the durable outcomes.
        return Err(Error::Other(anyhow::anyhow!(
            "restore_archived_fact: no facts_archive row with id {archived_fact_id}"
        )));
    }

    conn.execute(RESTORE_INSERT_SQL, libsql::params![archived_fact_id])
        .await?;
    conn.execute(
        "INSERT INTO facts_fts (fact_id, predicate, object_value) \
         SELECT id, predicate, object_value FROM facts WHERE id = ?1",
        libsql::params![archived_fact_id],
    )
    .await?;
    conn.execute(
        "DELETE FROM facts_archive WHERE id = ?1",
        libsql::params![archived_fact_id],
    )
    .await?;

    Ok(RestoreArchivedOutcome {
        restored_fact_id: archived_fact_id,
        already_live: false,
    })
}

/// Reverse a LOGGED `fact_supersede` mutation by its `mutation_id` (2026-09-14,
/// the fix that removed the need for a sixth MCP tool —
/// `.ai-docs/plans/mcp-agent-api-design-2026-09-14.md` §"the sixth tool").
///
/// Restores the row's captured `prior_valid_to` / `prior_expired_at` directly —
/// it does NOT delegate to [`unsupersede`], because `unsupersede` unconditionally
/// NULLs both fields, which is correct only for the single-supersede case. A
/// fact bounded by TWO supersedes in sequence has a second log row whose
/// `prior_valid_to` is the FIRST bound, not `NULL`; blindly nulling on undo of
/// either row would silently discard that history.
///
/// **LIFO guard (mirrors [`Error::UnmergeOutOfOrder`]/[`Error::UndoStale`]):**
/// `bound_valid_to` is a total overwrite, not a delta, so undoing an OLDER
/// supersede while a NEWER one is still live would clobber the newer bound with
/// this row's `prior_valid_to` and report success — destroying live data. Before
/// restoring, this checks the fact's CURRENT `valid_to` still equals what THIS
/// mutation itself set (`inputs.valid_to`); if a later supersede has since run,
/// the current value has moved on, and the undo is refused loudly rather than
/// guessed. A consumer must undo the fact's supersedes newest-first.
///
/// Idempotent at the LOG level: a row already marked `undone_at` short-circuits
/// to `NotSuperseded` WITHOUT touching `facts` again — mirrors
/// `undo_fact_archive` checking `undone_at` before re-restoring. Note this
/// reports "not superseded" for THIS row, not a claim that the fact carries no
/// bound at all — a separate, later `fact_supersede` row may still be live.
///
/// # Errors
///
/// - [`Error::MutationNotFound`] — `mutation_id` names no `fact_supersede` row.
/// - [`Error::UndoStale`] — a later supersede on the same fact is still live
///   (the LIFO guard above).
pub async fn undo_fact_supersede(
    graph: &TemporalGraph,
    mutation_id: i64,
) -> Result<UnsupersedeOutcome> {
    let (undone_at, pre_state_json, inputs_json): (Option<String>, String, String) = {
        let mut rows = graph
            .conn
            .query(
                "SELECT undone_at, pre_state, inputs FROM graph_mutation_log \
                 WHERE id = ?1 AND kind = 'fact_supersede'",
                libsql::params![mutation_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::MutationNotFound { mutation_id });
        };
        (
            row.get::<Option<String>>(0)?,
            row.get::<String>(1)?,
            row.get::<String>(2)?,
        )
    };

    let pre: FactSupersedePreState = serde_json::from_str(&pre_state_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "undo_fact_supersede: deserialize pre_state for mutation {mutation_id}: {e}"
        ))
    })?;

    if undone_at.is_some() {
        return Ok(UnsupersedeOutcome::NotSuperseded { fact_id: pre.fact_id });
    }

    let inputs: FactSupersedeInputs = serde_json::from_str(&inputs_json).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "undo_fact_supersede: deserialize inputs for mutation {mutation_id}: {e}"
        ))
    })?;

    let (current_valid_to, current_expired_at): (Option<String>, Option<String>) = {
        let mut rows = graph
            .conn
            .query(
                "SELECT valid_to, expired_at FROM facts WHERE id = ?1",
                libsql::params![pre.fact_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "undo_fact_supersede: no fact with id {} (mutation {mutation_id})",
                pre.fact_id
            )));
        };
        (row.get::<Option<String>>(0)?, row.get::<Option<String>>(1)?)
    };

    if current_valid_to.as_deref() != Some(inputs.valid_to.as_str()) {
        return Err(Error::UndoStale {
            mutation_id,
            reason: format!(
                "fact {} has been superseded again since this mutation ran (current \
                 valid_to {current_valid_to:?}, this mutation set {:?}) — undo the \
                 LATEST supersede on this fact first (LIFO)",
                pre.fact_id, inputs.valid_to
            ),
        });
    }

    graph
        .conn
        .execute(
            "UPDATE facts SET valid_to = ?1, expired_at = ?2 WHERE id = ?3",
            libsql::params![
                pre.prior_valid_to.clone(),
                pre.prior_expired_at.clone(),
                pre.fact_id
            ],
        )
        .await?;

    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "UPDATE graph_mutation_log SET undone_at = ?1 WHERE id = ?2",
            libsql::params![now, mutation_id],
        )
        .await?;

    Ok(UnsupersedeOutcome::Cleared {
        fact_id: pre.fact_id,
        cleared_valid_to: pre.prior_valid_to != current_valid_to,
        cleared_expired_at: pre.prior_expired_at != current_expired_at,
        // `SupersedeRequest::execute()` only ever binds `valid_to`/`expired_at`
        // (see `facade/supersede.rs`) — it never touches `invalid_at`, which is
        // exclusively a contradiction-resolver marker (`facts.rs::
        // invalidate_fact_with_reason`). So restoring THIS mutation's captured
        // prior state never clears it; that column is untouched by construction.
        cleared_invalid_at: false,
    })
}

// ─── unsupersede (§4.5) ─────────────────────────────────────────────────────

/// Clear a supersession bound (`valid_to` / `expired_at` / `invalid_at`) set by
/// `supersede(...)` or by contradiction detection, re-opening the fact as
/// currently-true AND re-eligible for consolidation (§4.5, TD-178). Idempotent:
/// a fact with none of the three set returns [`UnsupersedeOutcome::NotSuperseded`]
/// (an honest no-op). A missing `fact_id` is a hard `Error`.
#[doc(hidden)]
pub async fn unsupersede(graph: &TemporalGraph, fact_id: i64) -> Result<UnsupersedeOutcome> {
    let guard = graph.begin_immediate_if_needed().await?;
    let committed_here = guard.opened();
    let outcome = match unsupersede_txn(graph, fact_id).await {
        Ok(o) => o,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };
    guard.commit().await?;
    if committed_here {
        let label = match outcome {
            UnsupersedeOutcome::Cleared { .. } => "cleared",
            UnsupersedeOutcome::NotSuperseded { .. } => "not_superseded",
        };
        counter!("kremory.graph.unsupersede_total", "outcome" => label).increment(1);
    }
    Ok(outcome)
}

async fn unsupersede_txn(graph: &TemporalGraph, fact_id: i64) -> Result<UnsupersedeOutcome> {
    let conn = &graph.conn;

    let (valid_to, expired_at, invalid_at): (Option<String>, Option<String>, Option<String>) = {
        let mut rows = conn
            .query(
                "SELECT valid_to, expired_at, invalid_at FROM facts WHERE id = ?1",
                libsql::params![fact_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "unsupersede: no fact with id {fact_id}"
            )));
        };
        (
            row.get::<Option<String>>(0)?,
            row.get::<Option<String>>(1)?,
            row.get::<Option<String>>(2)?,
        )
    };

    let cleared_valid_to = valid_to.is_some();
    let cleared_expired_at = expired_at.is_some();
    let cleared_invalid_at = invalid_at.is_some();
    if !cleared_valid_to && !cleared_expired_at && !cleared_invalid_at {
        return Ok(UnsupersedeOutcome::NotSuperseded { fact_id });
    }

    // TD-178: `invalidate_fact_with_reason` (contradiction detection) sets
    // `expired_at` AND `invalid_at` together as one retirement. Clearing only
    // `expired_at` here used to leave `invalid_at` behind — invisible to
    // recall (which never checks it), but permanently excluded from
    // cross-episode merge, archival and supersession, which all gate on
    // `invalid_at IS NULL`. Undo must be symmetric with the operation it
    // reverses: clear both, unconditionally (NULLing an already-NULL column
    // is a no-op).
    conn.execute(
        "UPDATE facts SET valid_to = NULL, expired_at = NULL, invalid_at = NULL WHERE id = ?1",
        libsql::params![fact_id],
    )
    .await?;

    Ok(UnsupersedeOutcome::Cleared {
        fact_id,
        cleared_valid_to,
        cleared_expired_at,
        cleared_invalid_at,
    })
}
