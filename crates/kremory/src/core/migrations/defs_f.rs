// ─── Migration 015a ───────────────────────────────────────────────────────────

/// Migration 015a: add `episode_processing_status` column to `episodes`.
///
/// Introduces an explicit state-machine column for the async extraction
/// gate.  Values follow a strict lifecycle:
///
/// ```text
/// Pending → Extracting → Verified
///                      ↘ Failed
/// ```
///
/// ### Steps
///
/// 1. **PRAGMA-guard** — scan `PRAGMA table_info('episodes')` for the column.
///    If already present, skip the ALTER and proceed to the index step (idempotent
///    on resume or re-run).
/// 2. **Pre-ALTER backup** — `episodes_bak_015a` via `CREATE TABLE IF NOT EXISTS …
///    AS SELECT` for row-level recovery on partial-apply.
/// 3. **ALTER TABLE** — `ADD COLUMN episode_processing_status TEXT NOT NULL DEFAULT 'Pending'`.
///    SQLite 3.37+ allows `NOT NULL` + constant `DEFAULT` on `ALTER TABLE ADD COLUMN`.
/// 4. **Backfill** — existing episodes that already have entities extracted are
///    logically in the `Verified` state. Backfill via:
///    `UPDATE episodes SET episode_processing_status = 'Verified'
///     WHERE id IN (SELECT DISTINCT episode_id FROM episodic_edges)`
///    Uses `episodic_edges` (not `entities`) because the FK from `episodic_edges.episode_id`
///    is the authoritative link between an episode and its extracted entities.
/// 5. **Index** — `CREATE INDEX IF NOT EXISTS idx_episodes_processing_status
///    ON episodes(episode_processing_status)` for state-transition queries.
///
/// ### Idempotency
///
/// G1 — column already present → skip ALTER + backup steps, proceed to index.
/// Index uses `IF NOT EXISTS` — always idempotent.
/// Backup uses `CREATE TABLE IF NOT EXISTS` — safe on resume-from-partial.
/// Backfill UPDATE is idempotent: `WHERE id IN (SELECT DISTINCT episode_id FROM
/// episodic_edges)` is a pure SELECT-based predicate — double-apply sets the
/// same value again, no-op in practice.
///
/// ### No CHECK constraint
///
/// The status values (`Pending`, `Extracting`, `Verified`, `Failed`) are NOT
/// constrained by a `CHECK` clause.  Reason: SQLite `CHECK` is per-row but
/// does NOT prevent invalid values when `PRAGMA ignore_check_constraints = ON`,
/// and it creates migration friction for future state-machine extensions.
/// Validation is enforced at the application layer (engine.rs state transitions).
///
/// ### Emergency downgrade — see [`migrate_014c_downgrade_episode_processing_status`]
///
/// SQLite has no `DROP COLUMN` alternative that works with `NOT NULL DEFAULT`
/// columns in a table-recreation pattern.  The named downgrade fn uses
/// table-recreation (rename → create-without-column → copy → drop-old).
/// It is emergency-only and carries data-loss surface for any NOT NULL columns
/// added after 015a.
pub async fn migrate_015a_episode_processing_status(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_015a step `{name}` failed: {e}"
            ))
        }
    }

    tracing::info!(
        target: "kremory::migrations",
        migration = "015a",
        target_table = "episodes",
        "migrate_015a: starting episode_processing_status migration"
    );

    // ── G1: PRAGMA-guard — check whether column already exists ───────────────
    //
    // Scan table_info('episodes'). If `episode_processing_status` is present
    // we skip the backup + ALTER but still run the index step (idempotent via IF NOT EXISTS).

    let mut pragma_rows = conn
        .query("PRAGMA table_info('episodes')", ())
        .await
        .map_err(step("g1_pragma_table_info"))?;

    let mut has_status_col = false;
    while let Some(row) = pragma_rows
        .next()
        .await
        .map_err(step("g1_pragma_table_info_next"))?
    {
        let col_name: String = row.get(1).map_err(step("g1_pragma_col_name_read"))?;
        if col_name == "episode_processing_status" {
            has_status_col = true;
            break;
        }
    }
    // Explicitly drop the Rows cursor so it releases its open statement on the
    // episodes table before any DDL operates on it. libsql (SQLite) reports
    // "database table is locked" if a prepared statement cursor is open when
    // DDL targeting the same table runs on the same connection.
    drop(pragma_rows);

    if !has_status_col {
        // ── Step 1: pre-ALTER backup ─────────────────────────────────────────
        //
        // Row-level snapshot for recovery if ALTER crashes mid-run.
        // CREATE TABLE IF NOT EXISTS makes this step safe on resume-from-partial.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes_bak_015a AS SELECT * FROM episodes",
            (),
        )
        .await
        .map_err(step("backup_episodes"))?;

        // ── Step 2: ADD COLUMN episode_processing_status ─────────────────────
        //
        // SQLite ALTER TABLE ADD COLUMN supports NOT NULL + constant DEFAULT
        // (since 3.37.0, 2021-11-27). libsql ships a bundled SQLite meeting
        // this requirement. Existing rows receive DEFAULT 'Pending'.
        conn.execute(
            "ALTER TABLE episodes \
             ADD COLUMN episode_processing_status TEXT NOT NULL DEFAULT 'Pending'",
            (),
        )
        .await
        .map_err(step("add_column_episode_processing_status"))?;

        // ── Step 3: backfill — episodes with entities are already Verified ───
        //
        // Existing episodes that have at least
        // one entry in `episodic_edges` (the join table linking episodes to their
        // extracted entities) are logically already in the Verified state.
        // `episodic_edges.episode_id` is the authoritative post-extraction FK.
        // Backfill to 'Verified' so Phase 2's async gate is not re-triggered
        // for episodes already fully processed.
        conn.execute(
            "UPDATE episodes \
             SET episode_processing_status = 'Verified' \
             WHERE id IN (SELECT DISTINCT episode_id FROM episodic_edges)",
            (),
        )
        .await
        .map_err(step("backfill_verified_episodes"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_015a: episode_processing_status already present — skipping ALTER + backfill"
        );
    }

    // ── Step 4: index — always idempotent via IF NOT EXISTS ─────────────────
    //
    // Required for efficient state-transition queries:
    //   SELECT id FROM episodes WHERE episode_processing_status = 'Pending'
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_processing_status \
         ON episodes(episode_processing_status)",
        (),
    )
    .await
    .map_err(step("create_index_processing_status"))?;

    tracing::info!(
        target: "kremory::migrations",
        migration = "015a",
        target_table = "episodes",
        "migrate_015a: episode_processing_status column added + backfilled + index created"
    );
    Ok(())
}

// ─── Migration 014c — emergency downgrade for Migration 015a ──────────────────
//
// Emergency-only: data loss surface (any NOT NULL column without a constant
// DEFAULT that was added to episodes AFTER migration 015a would be dropped
// during table-recreation, returning those rows with SQLite's column-omit
// semantics which may violate NOT NULL constraints).
//
// SQLite has no `ALTER TABLE DROP COLUMN` that is safe for `NOT NULL DEFAULT`
// columns on all supported versions. Table-recreation is the canonical pattern
// for removing a column in SQLite < 3.35.0. Even on 3.35.0+, DROP COLUMN
// on a column with a NOT NULL DEFAULT is supported; however the migration
// framework uses table-recreation for maximum portability and to ensure index
// recreation is explicit.
//
// Named "014c" per the kremory convention of naming downgrade migrations after
// the preceding stable migration number + letter suffix (mirrors migrate_012b
// which downgrades migrate_013).

/// Migration 014c (emergency downgrade for 015a): remove `episode_processing_status`
/// from `episodes` via table-recreation.
///
/// # Emergency-only
///
/// This function is an emergency rollback path — it is NOT called from
/// `run_migrations`.  Invoke it only when a hard rollback of Migration 015a is
/// required and you have verified a backup exists (see `backup_workspace`).
///
/// Data-loss surface: any column added to `episodes` BETWEEN the 015a upgrade
/// and this downgrade that is `NOT NULL` without a constant `DEFAULT` will
/// be absent from the recreated table, leaving those rows without that column's
/// data. Inspect the live schema before invoking.
///
/// # Steps
///
/// 1. **Idempotency gate** — if `episode_processing_status` is NOT present,
///    the downgrade is already complete; return early.
/// 2. **PRAGMA foreign_keys = OFF** for the duration (recreating the referenced
///    `episodes` table while FK enforcement is on causes referential errors in
///    `facts` and `episodic_edges` which reference `episodes(id)`).
/// 3. **Rename** `episodes` → `episodes_bak_014c`.
/// 4. **Recreate** `episodes` with the pre-015a column set (no `episode_processing_status`).
/// 5. **Copy** rows from `episodes_bak_014c` into the new `episodes`.
/// 6. **Drop** `episodes_bak_014c`.
/// 7. **Recreate** all indexes that existed on `episodes` (saga, source_id, content_hash).
/// 8. **Restore** `PRAGMA foreign_keys = ON`.
///
/// # Idempotency
///
/// G1 — `episode_processing_status` already absent → return immediately (no-op).
/// The function is safe to call if 015a was never applied.
///
/// # Stuck-state warning (operator-only)
///
/// If `migrate_014c` fails between the RENAME (step 3) and the INSERT (step 5),
/// the connection is left with `episodes_bak_014c` (data) + a freshly-recreated
/// empty `episodes` table without the status column. The G1 PRAGMA gate on a
/// re-run will see `episodes` lacks `episode_processing_status` and short-circuit
/// to no-op — leaving the partial state forever. This matches the same shape in
/// `migrate_012b`. Recovery: manual `INSERT INTO episodes SELECT * FROM
/// episodes_bak_014c` + `DROP TABLE episodes_bak_014c`. Inspect schema before
/// invoking downgrade in production.
pub async fn migrate_014c_downgrade_episode_processing_status(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_014c step `{name}` failed: {e}"
            ))
        }
    }

    // ── G1: idempotency gate ─────────────────────────────────────────────────

    let mut pragma_rows = conn
        .query("PRAGMA table_info('episodes')", ())
        .await
        .map_err(step("g1_pragma_table_info"))?;

    let mut has_status_col = false;
    while let Some(row) = pragma_rows
        .next()
        .await
        .map_err(step("g1_pragma_table_info_next"))?
    {
        let col_name: String = row.get(1).map_err(step("g1_pragma_col_name_read"))?;
        if col_name == "episode_processing_status" {
            has_status_col = true;
            break;
        }
    }
    // Explicitly drop the Rows cursor so it releases its open statement on the
    // episodes table before any DDL operates on it. libsql (SQLite) reports
    // "database table is locked" if a prepared statement cursor is open when
    // DDL targeting the same table runs on the same connection.
    drop(pragma_rows);

    if !has_status_col {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_014c: episode_processing_status already absent — downgrade is a no-op"
        );
        return Ok(());
    }

    // ── PRAGMA foreign_keys = OFF ────────────────────────────────────────────
    //
    // `episodes` is referenced by `facts.source_episode_id` and
    // `episodic_edges.episode_id`. Table-recreation (DROP + CREATE + RENAME)
    // requires FK enforcement to be suspended so referential checks don't fire
    // against the intermediate state.

    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        // ── Step 1: rename existing table to backup ──────────────────────────
        conn.execute("ALTER TABLE episodes RENAME TO episodes_bak_014c", ())
            .await
            .map_err(step("rename_to_backup"))?;

        // ── Step 2: recreate episodes WITHOUT episode_processing_status ───────
        //
        // Column set matches the base DDL in schema.rs run_migrations +
        // columns added by migrations 007 (source_id, source_uri, recorded_at)
        // and 011 (content_hash). All columns that existed before 015a.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                content          TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                recorded_at      TEXT NOT NULL DEFAULT (datetime('now')),
                source_type      TEXT,
                metadata         TEXT,
                group_id         TEXT,
                saga_id          TEXT,
                sequence_number  INTEGER,
                source_id        TEXT,
                source_uri       TEXT,
                content_hash     TEXT
            )",
            (),
        )
        .await
        .map_err(step("recreate_episodes_table"))?;

        // ── Step 3: copy rows from backup ────────────────────────────────────
        conn.execute(
            "INSERT INTO episodes (
                 id, content, timestamp, recorded_at, source_type, metadata,
                 group_id, saga_id, sequence_number, source_id, source_uri, content_hash
             )
             SELECT
                 id, content, timestamp, recorded_at, source_type, metadata,
                 group_id, saga_id, sequence_number, source_id, source_uri, content_hash
             FROM episodes_bak_014c",
            (),
        )
        .await
        .map_err(step("copy_rows_from_backup"))?;

        // ── Step 4: drop the backup table ────────────────────────────────────
        conn.execute("DROP TABLE episodes_bak_014c", ())
            .await
            .map_err(step("drop_backup"))?;

        // ── Step 5: recreate indexes ──────────────────────────────────────────
        //
        // Three indexes existed on `episodes` before migration 015a:
        //   idx_episodes_saga         (saga_id)            — from base DDL
        //   idx_episodes_source_id    (source_id)          — from Migration 007
        //   idx_episodes_content_hash (content_hash)       — from Migration 011
        // The 015a-added index (idx_episodes_processing_status) is intentionally
        // NOT recreated — it referred to the column we just removed.
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_saga \
                 ON episodes(saga_id)",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_source_id \
                 ON episodes(source_id)",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_content_hash \
                 ON episodes(content_hash)",
                (),
            )
            .await;

        Ok(())
    }
    .await;

    // Always restore PRAGMA foreign_keys = ON regardless of body outcome.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_014c: failed to re-enable PRAGMA foreign_keys — \
             connection FK state is inconsistent (still OFF); operator must reconnect"
        );
    }

    body_result?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_014c: episodes table recreated without episode_processing_status; \
         downgrade complete."
    );
    Ok(())
}
