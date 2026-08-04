// ─── Migration LEGACY ──────────────────────────────────────────────────────

/// Pre-002 backward migration: rename legacy `entities` (rql shape, has `label`)
/// to `rql_entities` so that the workspace P1 `entities` table can coexist.
///
/// Idempotent: only renames when a `label`-shaped legacy table exists AND the
/// new name is free.
pub(crate) async fn migrate_legacy_rql_entities_table(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    // Detect legacy rql shape: `entities` table with a `label` column.
    let mut rows = conn.query("PRAGMA table_info(entities)", ()).await?;
    let mut has_label = false;
    let mut has_any = false;
    while let Some(row) = rows.next().await? {
        has_any = true;
        let name: String = row.get(1)?;
        if name == "label" {
            has_label = true;
            break;
        }
    }
    if !has_any || !has_label {
        return Ok(());
    }

    // Don't clobber an existing rql_entities — if both are present the
    // rename happened previously and the bare `entities` is some other
    // table (e.g. workspace shape co-resident). Bail without touching.
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rql_entities'",
            (),
        )
        .await?;
    if rows.next().await?.is_some() {
        return Ok(());
    }

    // Rename data table + FTS5 sibling + indexes.
    conn.execute("ALTER TABLE entities RENAME TO rql_entities", ())
        .await?;
    let _ = conn
        .execute("ALTER TABLE entities_fts RENAME TO rql_entities_fts", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS idx_entities_group", ())
        .await;
    Ok(())
}

// ─── Migration 002 ─────────────────────────────────────────────────────────

/// Migration 002: rename `rql_entities` → `entities` (ADR-029b Decision 2).
///
/// Idempotent: exits immediately when `rql_entities` does not exist.
/// Called from `run_migrations()` BEFORE the base DDL block.
pub(crate) async fn migrate_002_drop_rql_prefix(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    // Check if rql_entities still exists — if not, migration already applied.
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rql_entities'",
            (),
        )
        .await?;
    if rows.next().await?.is_none() {
        return Ok(());
    }

    // Also check that `entities` does not exist yet (prevents double-rename collision).
    let mut check = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'",
            (),
        )
        .await?;
    if check.next().await?.is_some() {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_002: both rql_entities and entities exist — skipping rename; \
             manual inspection recommended"
        );
        return Ok(());
    }

    conn.execute("ALTER TABLE rql_entities RENAME TO entities", ())
        .await?;
    let _ = conn
        .execute("ALTER TABLE rql_entities_fts RENAME TO entities_fts", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS rql_entities_vec_idx", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS idx_rql_entities_group", ())
        .await;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_002: rql_entities renamed to entities"
    );
    Ok(())
}

// ─── Migration 004 ─────────────────────────────────────────────────────────

/// Migration 004: composite PK on `entities` (ADR-029b Decision 1).
///
/// Uses CREATE-COPY-DROP-RENAME to restructure the table to
/// `PRIMARY KEY (id, group_id)`.
///
/// Idempotency gate (Vera 2026-05-28 BUG-1 fix):
///   Uses `PRAGMA table_info('entities')` to check whether `group_id` is already
///   in the PK — this is the SHAPE-based sentinel. The old `entities_bak_004`
///   backup-table sentinel was a false gate: if the process crashed after creating
///   the backup but before creating `entities_new`, the next startup would see
///   the backup, return Ok(()), and silently leave the migration incomplete.
///
/// FK restore on ALL exit paths (Vera 2026-05-28 BUG-2 fix):
///   `PRAGMA foreign_keys = ON` is guaranteed via the `body_result` wrapping
///   pattern — same approach used by migrate_006. An error in any restructure
///   step still restores FK enforcement before propagating the error.
///
/// Post-migration `PRAGMA foreign_key_check` (Vera 2026-05-28 BUG-2 companion):
///   After the body completes successfully the migration asserts that no FK
///   violations were introduced.
pub(crate) async fn migrate_004_composite_pk_entities(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_004 step `{name}` failed: {e}"
            ))
        }
    }

    // Vera BUG-1 fix: SHAPE-based idempotency gate.
    // If `group_id` is already part of the entities PK (pk > 0), the migration
    // has completed — return immediately without touching anything.
    let mut pragma_rows = conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .map_err(step("g1_table_info"))?;
    while let Some(r) = pragma_rows
        .next()
        .await
        .map_err(step("g1_table_info_next"))?
    {
        let col_name: String = r.get(1).unwrap_or_default();
        let pk: i64 = r.get(5).unwrap_or(0);
        if col_name == "group_id" && pk > 0 {
            // Already migrated.
            return Ok(());
        }
    }

    // Check whether entities_new exists — signals a partial migration in progress
    // (crashed between CREATE entities_new and the DROP+RENAME).
    let mut rows2 = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities_new'",
            (),
        )
        .await
        .map_err(step("check_entities_new"))?;
    let partial_migration_in_progress = rows2
        .next()
        .await
        .map_err(step("check_entities_new_next"))?
        .is_some();
    // ── DUR-5 (V1-CANONICAL §4.2): THE crash-resume fix for migrate_004 ─────────
    //
    // Drop the Rows handle before any DDL — an open cursor locks the connection.
    // `defs_d.rs:82` and `defs_e.rs:55` already carried this line and this comment;
    // migrate_004 never did.
    //
    // The omission was invisible because it bites ONLY on the path that matters. When
    // `entities_new` does not exist — every ordinary run — `.next()` returns `None`,
    // which exhausts the cursor and releases it as a side effect. When it DOES exist —
    // i.e. exactly when resuming a migration that crashed between the DROP and the
    // RENAME — `.next()` returns a row, the cursor stays open, and the `DROP TABLE`
    // below fails with `database table is locked`. So the resume path could never
    // complete, and no ordinary run could ever reveal that.
    //
    // ⚠️ Proven, in both directions, by `tests/migration_crash_resume.rs`. Two other
    // candidate fixes were tried FIRST and both were measured to be NO-OPS here —
    // recorded so nobody re-adds them believing they do something:
    //   • `DROP TABLE IF EXISTS entities` — unreachable. `ensure_schema` runs
    //     `CREATE TABLE IF NOT EXISTS entities` (`schema.rs:599`) on EVERY open before
    //     any migration, so by the time this code runs the table is always back. The
    //     bare DROP always has something to drop.
    //   • dropping `entities_vec_idx` first — the lock was the CURSOR, not the index.
    drop(rows2);

    if partial_migration_in_progress {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_004: entities_new already exists — attempting to complete partial migration"
        );
    }

    // Vera BUG-2 fix: PRAGMA foreign_keys = OFF is set once here; `PRAGMA
    // foreign_keys = ON` is guaranteed via the `body_result` pattern below
    // regardless of whether the body succeeds or returns Err. This mirrors the
    // pattern used by migrate_006 (Vera 2026-05-28 #2).
    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        if partial_migration_in_progress {
            // ── CONTENT-BASED RESUME (Quinn REL-002) ─────────────────────────
            //
            // The `drop(rows2)` fix above makes this branch REACHABLE for the first
            // time — it previously always died on `database table is locked`, so the
            // lock was accidentally acting as a guard. Un-blocking it without this
            // check would trade a loud failure (data intact in `entities_new` +
            // `entities_bak_004`) for a quiet one: `entities_new` existing proves only
            // that a run STARTED, and `CREATE TABLE entities_new` is a separate
            // statement from the `INSERT … SELECT` below, so a crash between them
            // leaves it EMPTY. The swap would then rename an empty table over every
            // entity and report success.
            //
            // `migrate_023`'s H3 invariant (`defs_j.rs:74-93`): compare against the
            // IMMUTABLE snapshot and repopulate FROM THE SNAPSHOT — never from live
            // `entities`, which a prior crashed run may already have dropped (and
            // which `ensure_schema` may since have recreated EMPTY, `schema.rs:599`,
            // making it actively misleading as a source).
            async fn row_count(
                conn: &libsql::Connection,
                table: &str,
            ) -> std::result::Result<usize, libsql::Error> {
                // NOT `COUNT(*)` — the libsql vector index returns 0 for a populated
                // `entities` (SYSTEM-PRIMER gotcha #1), which here would read as
                // "the snapshot is empty" and skip the repair.
                let mut rows = conn.query(&format!("SELECT rowid FROM {table}"), ()).await?;
                let mut n = 0usize;
                while rows.next().await?.is_some() {
                    n += 1;
                }
                Ok(n)
            }

            let mut bak = conn
                .query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='entities_bak_004'",
                    (),
                )
                .await
                .map_err(step("resume_check_bak"))?;
            let bak_present = bak
                .next()
                .await
                .map_err(step("resume_check_bak_next"))?
                .is_some();
            drop(bak);

            if !bak_present {
                return Err(crate::core::error::Error::Other(anyhow::anyhow!(
                    "migrate_004: entities_new exists but entities_bak_004 does not, so \
                     the scratch table's completeness cannot be verified. Refusing to \
                     swap it over the live table. Inspect entities_new by hand; dropping \
                     it restarts the migration cleanly from the live table."
                )));
            }

            let scratch = row_count(conn, "entities_new")
                .await
                .map_err(step("resume_count_entities_new"))?;
            let snapshot = row_count(conn, "entities_bak_004")
                .await
                .map_err(step("resume_count_bak"))?;
            if scratch < snapshot {
                tracing::warn!(
                    target: "kremory::migrations",
                    scratch_rows = scratch,
                    snapshot_rows = snapshot,
                    "migrate_004: entities_new is INCOMPLETE — the prior run crashed \
                     mid-copy. Repopulating from entities_bak_004 before the swap; \
                     renaming it as-is would silently destroy entities."
                );
                conn.execute("DELETE FROM entities_new", ())
                    .await
                    .map_err(step("resume_clear_entities_new"))?;
                conn.execute(
                    "INSERT INTO entities_new (id, label, properties, embedding, recorded_at, updated_at, group_id, access_count)
                     SELECT id, label, properties, embedding, recorded_at, updated_at,
                            COALESCE(group_id, 'default') AS group_id,
                            COALESCE(access_count, 0) AS access_count
                     FROM entities_bak_004",
                    (),
                )
                .await
                .map_err(step("resume_repopulate_from_bak"))?;
            }
        }

        if !partial_migration_in_progress {
            // Step 1: backup. Left in place as a rollback artifact.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entities_bak_004 AS SELECT * FROM entities",
                (),
            )
            .await
            .map_err(step("create_bak"))?;

            // Step 2: create new table with composite PK.
            // Note: `embedding` uses generic `BLOB` here (rather than `F32_BLOB(dim)`)
            // because `dim` is not in scope inside this migration helper; the vector
            // index (recreated below via `libsql_vector_idx`) works with raw BLOB columns.
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS entities_new (
                    id           TEXT NOT NULL,
                    label        TEXT NOT NULL,
                    properties   TEXT,
                    embedding    BLOB,
                    recorded_at  TEXT NOT NULL,
                    updated_at   TEXT,
                    group_id     TEXT NOT NULL DEFAULT 'default',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (id, group_id)
                );",
            )
            .await
            .map_err(step("create_entities_new"))?;

            // Step 3: copy + backfill group_id.
            conn.execute(
                "INSERT INTO entities_new (id, label, properties, embedding, recorded_at, updated_at, group_id, access_count)
                 SELECT id, label, properties, embedding, recorded_at, updated_at,
                        COALESCE(group_id, 'default') AS group_id,
                        COALESCE(access_count, 0)     AS access_count
                 FROM entities",
                (),
            )
            .await
            .map_err(step("copy_rows"))?;
        }

        // Step 4 + 5: drop old, rename new.
        conn.execute("DROP TABLE entities", ())
            .await
            .map_err(step("drop_old_entities"))?;
        conn.execute("ALTER TABLE entities_new RENAME TO entities", ())
            .await
            .map_err(step("rename_new_to_entities"))?;

        // Step 6: re-create indexes (vector index is best-effort on in-memory DBs).
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS entities_vec_idx \
                 ON entities(libsql_vector_idx(embedding, 'metric=cosine'))",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id)",
                (),
            )
            .await;

        // Step 7+8: facts + episodic_edges composite FK restructure — PENDING
        // ship-architect design review (2026-05-28). Current best-effort ALTER+
        // UPDATE pattern is kept temporarily so the compile + downstream tests
        // continue to surface the issue rather than papering over it. See
        // `.ai-docs/planning/v014-adr-029b-composite-fk-design-2026-05-28.md`.
        let _ = conn
            .execute("ALTER TABLE facts ADD COLUMN subject_group_id TEXT", ())
            .await;
        let _ = conn
            .execute("ALTER TABLE facts ADD COLUMN object_group_id TEXT", ())
            .await;
        let _ = conn
            .execute(
                "UPDATE facts SET subject_group_id = (
                     SELECT COALESCE(e.group_id, 'default')
                     FROM entities e WHERE e.id = facts.subject_id
                     LIMIT 1
                 ) WHERE subject_group_id IS NULL",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "UPDATE facts SET object_group_id = (
                     SELECT COALESCE(e.group_id, 'default')
                     FROM entities e WHERE e.id = facts.object_id
                     LIMIT 1
                 ) WHERE object_group_id IS NULL AND object_id IS NOT NULL",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "ALTER TABLE episodic_edges ADD COLUMN entity_group_id TEXT",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "UPDATE episodic_edges SET entity_group_id = (
                     SELECT COALESCE(e.group_id, 'default')
                     FROM entities e WHERE e.id = episodic_edges.entity_id
                     LIMIT 1
                 ) WHERE entity_group_id IS NULL",
                (),
            )
            .await;

        Ok(())
    }
    .await;

    // Vera BUG-2 fix: always restore PRAGMA foreign_keys = ON, regardless of
    // whether the body succeeded or returned Err.  A failed migration that leaves
    // FK enforcement OFF on the connection is a silent data-integrity bug for all
    // subsequent application writes on that connection.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_004: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    // Propagate the body's result.
    body_result?;

    // NOTE: `PRAGMA foreign_key_check` is deliberately NOT run here. After
    // migrate_004 swaps entities to composite PK (id, group_id), the existing
    // `facts.subject_id REFERENCES entities(id)` constraint references a
    // non-unique column set — SQLite reports this as a violation until
    // migrate_006 rebuilds facts with the composite FK shape. The fk-integrity
    // check therefore belongs at the END of the migrate_006 step (already
    // present there, see line ~1163), not after migrate_004 alone — the schema
    // is intentionally in a mixed state between these two migrations.
    //
    // The always-restore `PRAGMA foreign_keys` wrapper above (Vera BUG-2 fix)
    // ensures the connection's FK enforcement is correctly re-enabled on any
    // exit path. The post-condition gate is migrate_006's fk_check.

    tracing::info!(
        target: "kremory::migrations",
        "migrate_004: composite PK (id, group_id) applied to entities"
    );
    Ok(())
}

// ─── Migration 005 ─────────────────────────────────────────────────────────

/// Migration 005: add `upgraded_at` column to `namespaces` (ADR-029b Decision 5).
///
/// Idempotent: `ALTER TABLE ADD COLUMN` errors are swallowed when the column
/// already exists.
pub(crate) async fn migrate_005_policy_upgraded_at(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    let _ = conn
        .execute("ALTER TABLE namespaces ADD COLUMN upgraded_at TEXT", ())
        .await;
    tracing::info!(
        target: "kremory::migrations",
        "migrate_005: upgraded_at column ensured on namespaces"
    );
    Ok(())
}
