// ─── Migration 025 ─────────────────────────────────────────────────────────

/// Migration 025 (TD-133 B2): make `idx_facts_content_hash_unique` a partial
/// UNIQUE index over ACTIVE (non-expired) rows only, so its predicate matches
/// the dedup pre-check SELECT in `graph/facts.rs`
/// (`WHERE content_hash = ?1 AND expired_at IS NULL`).
///
/// ## The bug (measured)
///
/// The dedup pre-check SELECT excludes expired rows (`expired_at IS NULL`),
/// but the UNIQUE index covered ALL rows (`WHERE content_hash IS NOT NULL`).
/// So re-asserting a triple that had been superseded/expired — a legitimate
/// bi-temporal assert→expire→re-assert — passed the dedup SELECT (no ACTIVE
/// duplicate) yet collided with the stale EXPIRED row still present in the
/// global index → `UNIQUE constraint failed` → the re-assertion was silently
/// lost. Measured on the TD-133 instrumented conv0 run (2026-07-21): 21
/// `kremory_ingest_phase2_fact_insert_failed_total{reason=unique_violation}`
/// in a single namespace — NOT the cross-namespace collision the plan
/// hypothesised (conv0 ingests under one namespace).
///
/// ## The fix
///
/// Recreate the index with `WHERE content_hash IS NOT NULL AND expired_at IS
/// NULL`: uniqueness is enforced only among active rows, so an expired row no
/// longer blocks re-assertion, while two concurrently-active identical triples
/// are still rejected (the dedup invariant holds). Strictly MORE permissive
/// than the prior index — it can never violate existing data.
///
/// ## Ordering (load-bearing)
///
/// MUST run as the LAST schema step in `run_migrations` (after `migrate_023`,
/// which recreates the old-form index via `defs_j`), otherwise a later
/// migration would re-introduce the un-partitioned form and undo this fix.
///
/// Idempotent + conditional: only DROP+recreate when the live index SQL lacks
/// the `expired_at` predicate; a no-op on every subsequent open once migrated.
///
/// NB: this deliberately does NOT group-scope the index/dedup by `group_id`.
/// The measured failure is single-namespace (expired-collision). Cross-namespace
/// dedup semantics (same triple in different namespaces = distinct facts) is a
/// separate, latent multi-tenant concern tracked under TD-126 — it is not
/// exercised by this data and would change the semantics asserted by
/// `test_try_insert_fact_with_group_dedups_cross_variant`, so it is out of
/// scope here.
pub(crate) async fn migrate_025_fact_dedup_expired_partial(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn other_err(context: &str, e: impl std::fmt::Display) -> crate::core::error::Error {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_025_fact_dedup_expired_partial: {context}: {e}"
        ))
    }

    // Read the current index definition (if any) from sqlite_master.
    //
    // The `rows` cursor is confined to this block and dropped before the DDL
    // below: an un-finalized libsql cursor holds a read lock, and a DROP INDEX
    // then fails with `database table is locked`. (migrate_024 avoids this the
    // same way — its query rows fall out of loop scope before its execute.)
    let existing_sql: Option<String> = {
        let mut rows = conn
            .query(
                "SELECT sql FROM sqlite_master \
                 WHERE type='index' AND name='idx_facts_content_hash_unique'",
                (),
            )
            .await
            .map_err(|e| other_err("querying sqlite_master for index sql", e))?;

        match rows
            .next()
            .await
            .map_err(|e| other_err("reading index sql row", e))?
        {
            Some(row) => Some(
                row.get::<String>(0)
                    .map_err(|e| other_err("reading index sql column", e))?,
            ),
            None => None,
        }
    };

    // Already migrated (index carries the expired_at predicate) → no-op.
    if let Some(sql) = &existing_sql {
        if sql.contains("expired_at") {
            return Ok(());
        }
    }

    // Drop the old-form index (if present) and recreate as an active-only
    // partial UNIQUE index matching the dedup SELECT predicate.
    conn.execute("DROP INDEX IF EXISTS idx_facts_content_hash_unique", ())
        .await
        .map_err(|e| other_err("dropping old idx_facts_content_hash_unique", e))?;
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_content_hash_unique \
         ON facts(content_hash) WHERE content_hash IS NOT NULL AND expired_at IS NULL",
        (),
    )
    .await
    .map_err(|e| {
        other_err(
            "creating active-only partial idx_facts_content_hash_unique",
            e,
        )
    })?;

    Ok(())
}
