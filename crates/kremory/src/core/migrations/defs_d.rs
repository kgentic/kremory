// ─── Migration 013 ─────────────────────────────────────────────────────────

/// Migration 013: extend `entity_type_source` CHECK to accept `'DreamPass4'` +
/// create `dream_pass4_audit` table (ADR-047, v0.1.2 Phase B).
///
/// ### Steps
///
/// 1. Idempotency guard: query `sqlite_master` for `entities` DDL containing
///    `'DreamPass4'`. If already present → no-op + return Ok(()).
/// 2. Table-rebuild to extend the CHECK constraint on `entity_type_source`:
///    - `PRAGMA foreign_keys = OFF`
///    - BEGIN TRANSACTION
///    - CREATE `entities_new` with CHECK accepting all 6 values
///      (`Phase1Ner, Phase2Llm, DreamPass0, DreamPass1, ConsumerPinned, DreamPass4`)
///    - INSERT INTO entities_new SELECT * FROM entities
///    - DROP indexes on `entities`
///    - DROP TABLE entities
///    - ALTER TABLE entities_new RENAME TO entities
///    - Recreate indexes
///    - COMMIT
///    - `PRAGMA foreign_keys = ON`
/// 3. Create `dream_pass4_audit` table (FK → `entities(rowid)`) with 2 indexes.
///
/// ### Idempotency
///
/// G1 — `entities` DDL in `sqlite_master` already contains `'DreamPass4'` →
/// skip all steps. Safe to call multiple times.
///
/// ### FK note
///
/// `dream_pass4_audit.entity_id` stores `entities.rowid` (physical integer
/// row id) as a SOFT reference. With composite PK `(id, group_id)` the rowid
/// is the physical row id, queryable via `SELECT rowid FROM entities WHERE id = ?1`.
///
/// **FK omission + cascade semantic via TRIGGER (ADR-047 IRREV-001 amendment)**:
/// SQLite FK resolution requires the referenced column to be UNIQUE or a single-column
/// INTEGER PRIMARY KEY. Entities uses composite PK (id, group_id), so a single-column
/// FK on rowid is invalid. To preserve IRREV-001's `ON DELETE CASCADE` provenance
/// semantic without a FK constraint, this migration creates an `AFTER DELETE` trigger
/// (`trg_dream_pass4_audit_cascade_delete`) that mirrors FK CASCADE behaviour: when an
/// entity is deleted, its audit history is also deleted. Per [[treat-cause-not-symptom]]
/// the TRIGGER is the cause-fix; FK omission alone would have been the band-aid.
///
/// **Visibility (`pub` not `pub(crate)`)**: integration tests in
/// `crates/kremory/tests/migration_013_pass4_source_tier.rs` need direct access for
/// per-function acceptance testing (B1-B6 + trigger cascade test). The standard
/// `pub(crate)` convention from migrations 001-014 is intentionally departed-from here.
/// `migrate_012b_*` (downgrade) similarly `pub` for explicit downgrade-tool invocation.
pub async fn migrate_013_pass4_source_tier(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_013 step `{name}` failed: {e}"
            ))
        }
    }

    // ── G1: idempotency gate ─────────────────────────────────────────────────
    //
    // Check whether `entities` DDL in sqlite_master already includes 'DreamPass4'.
    // If it does, the migration has already been applied — return immediately.

    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master \
             WHERE type = 'table' AND name = 'entities' \
             LIMIT 1",
            (),
        )
        .await
        .map_err(step("g1_sqlite_master_query"))?;

    let already_applied =
        if let Some(row) = rows.next().await.map_err(step("g1_sqlite_master_next"))? {
            let ddl: Option<String> = row.get(0).map_err(step("g1_ddl_read"))?;
            ddl.map(|s| s.contains("DreamPass4")).unwrap_or(false)
        } else {
            false
        };
    // Drop the Rows handle before any DDL — an open cursor locks the connection.
    drop(rows);

    if already_applied {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_013: entities DDL already contains 'DreamPass4' — skipping (idempotent)"
        );
        // Still ensure the audit table + indexes + cascade trigger exist
        // (CREATE IF NOT EXISTS is safe).
        //
        // entity_id stores entities.rowid (physical integer row ID) as a soft reference.
        // A FK constraint is omitted because SQLite FK resolution requires the referenced
        // column to be explicitly UNIQUE or a single-column INTEGER PRIMARY KEY.
        // entities uses a composite PK (id, group_id), so rowid is not a valid FK target.
        //
        // To preserve the IRREV-001 cascade-delete semantic from ADR-047 amendment,
        // an AFTER DELETE trigger mirrors `ON DELETE CASCADE` behaviour: when an entity
        // is deleted, its audit-history rows are also deleted.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS dream_pass4_audit (
                audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id       INTEGER NOT NULL,
                pre_type_id     INTEGER NOT NULL,
                post_type_id    INTEGER NOT NULL,
                verify_confidence REAL NOT NULL,
                verify_model    TEXT NOT NULL,
                run_id          TEXT NOT NULL,
                correction_ts   INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
            );
            CREATE INDEX IF NOT EXISTS idx_dream_pass4_audit_entity \
                ON dream_pass4_audit(entity_id);
            CREATE INDEX IF NOT EXISTS idx_dream_pass4_audit_run \
                ON dream_pass4_audit(run_id);
            CREATE TRIGGER IF NOT EXISTS trg_dream_pass4_audit_cascade_delete \
                AFTER DELETE ON entities \
                BEGIN \
                    DELETE FROM dream_pass4_audit WHERE entity_id = OLD.rowid; \
                END;",
        )
        .await
        .map_err(step("ensure_audit_table_idempotent"))?;
        return Ok(());
    }

    // ── Step 2: table-rebuild to extend CHECK constraint ─────────────────────
    //
    // SQLite does not support ALTER TABLE ... ALTER COLUMN CHECK.
    // The only way to change a constraint is to recreate the table.
    //
    // FK enforcement is disabled for the duration of the swap. The body_result
    // wrapper guarantees PRAGMA foreign_keys = ON is restored on all exit paths.

    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        // Drop any view that references `entities` — SQLite rejects RENAME when
        // a view references the table by name. Per ADR-046 Amendment 2026-06-09,
        // v_entity_drift_candidates is deferred and its self-JOIN approach is
        // structurally impossible in kremory; safe to discard permanently.
        let _ = conn
            .execute("DROP VIEW IF EXISTS v_entity_drift_candidates", ())
            .await;

        // Clean up any leftover scratch table from a prior failed migration run.
        let _ = conn.execute("DROP TABLE IF EXISTS entities_new", ()).await;

        // Create entities_new with the extended CHECK constraint.
        // Column list must match the post-migration-012 entities shape:
        //   id TEXT NOT NULL
        //   properties TEXT
        //   embedding BLOB (generic; vector index recreated below)
        //   recorded_at TEXT NOT NULL
        //   updated_at TEXT
        //   group_id TEXT NOT NULL DEFAULT 'default'
        //   access_count INTEGER NOT NULL DEFAULT 0
        //   entity_type_id INTEGER NOT NULL DEFAULT 0
        //   entity_type_source TEXT CHECK(...)  ← extended to include DreamPass4
        //   entity_type_assigned_at TEXT
        //   ner_confidence REAL
        //   PRIMARY KEY (id, group_id)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entities_new (
                id                      TEXT NOT NULL,
                properties              TEXT,
                embedding               BLOB,
                recorded_at             TEXT NOT NULL,
                updated_at              TEXT,
                group_id                TEXT NOT NULL DEFAULT 'default',
                access_count            INTEGER NOT NULL DEFAULT 0,
                entity_type_id          INTEGER NOT NULL DEFAULT 0,
                entity_type_source      TEXT CHECK (entity_type_source IN (
                    'Phase1Ner', 'Phase2Llm', 'DreamPass0',
                    'DreamPass1', 'ConsumerPinned', 'DreamPass4'
                )),
                entity_type_assigned_at TEXT,
                ner_confidence          REAL,
                PRIMARY KEY (id, group_id)
            )",
            (),
        )
        .await
        .map_err(step("create_entities_new"))?;

        // Copy all rows. Column list is explicit to guard against any future
        // schema drift between entities and entities_new.
        conn.execute(
            "INSERT INTO entities_new (
                id, properties, embedding, recorded_at, updated_at,
                group_id, access_count, entity_type_id,
                entity_type_source, entity_type_assigned_at, ner_confidence
             )
             SELECT
                id, properties, embedding, recorded_at, updated_at,
                group_id, access_count, entity_type_id,
                entity_type_source, entity_type_assigned_at, ner_confidence
             FROM entities",
            (),
        )
        .await
        .map_err(step("copy_rows"))?;

        // Drop indexes that reference `entities` before dropping the table.
        // These are recreated after the rename. Use DROP INDEX (not IF EXISTS on
        // the rename path) — indexes go away when the table is dropped anyway, but
        // SQLite requires explicit DROP INDEX for named indexes when FK is OFF.
        let _ = conn
            .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
            .await;
        let _ = conn
            .execute("DROP INDEX IF EXISTS idx_entities_group", ())
            .await;
        let _ = conn
            .execute("DROP INDEX IF EXISTS idx_entities_type_id", ())
            .await;

        // Drop old table and rename new one into place.
        conn.execute("DROP TABLE entities", ())
            .await
            .map_err(step("drop_entities"))?;
        conn.execute("ALTER TABLE entities_new RENAME TO entities", ())
            .await
            .map_err(step("rename_entities_new"))?;

        // Recreate indexes. Vector index is best-effort (may fail on in-memory DBs).
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
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type_id \
                 ON entities(group_id, entity_type_id)",
                (),
            )
            .await;

        Ok(())
    }
    .await;

    // Always restore PRAGMA foreign_keys = ON regardless of body success/failure.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_013: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    // Propagate any body error before proceeding to audit table creation.
    body_result?;

    // ── Step 3: create dream_pass4_audit table + indexes ─────────────────────
    //
    // entity_id stores entities.rowid (physical integer row ID) as a soft reference.
    // A FK constraint is omitted because SQLite FK resolution requires the referenced
    // column to be explicitly UNIQUE or a single-column INTEGER PRIMARY KEY.
    // entities uses a composite PK (id, group_id), so rowid is not a valid FK target.
    // Callers obtain the rowid via `SELECT rowid FROM entities WHERE id = ?1`.

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS dream_pass4_audit (
            audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id       INTEGER NOT NULL,
            pre_type_id     INTEGER NOT NULL,
            post_type_id    INTEGER NOT NULL,
            verify_confidence REAL NOT NULL,
            verify_model    TEXT NOT NULL,
            run_id          TEXT NOT NULL,
            correction_ts   INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
        );
        CREATE INDEX IF NOT EXISTS idx_dream_pass4_audit_entity \
            ON dream_pass4_audit(entity_id);
        CREATE INDEX IF NOT EXISTS idx_dream_pass4_audit_run \
            ON dream_pass4_audit(run_id);",
    )
    .await
    .map_err(step("create_audit_table"))?;

    // ── Step 4: cascade-delete trigger (IRREV-001 amendment cause-fix) ──────
    //
    // Per ADR-047 §IRREV-001 fold: when an entity is deleted, its audit-history
    // rows must be deleted too. SQLite FK with composite PK on entities is
    // structurally impossible (see step 3 comment), so the cascade semantic is
    // implemented via an AFTER DELETE TRIGGER instead. Per [[treat-cause-not-symptom]],
    // the TRIGGER is the cause-fix; FK omission alone would have been the band-aid.

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS trg_dream_pass4_audit_cascade_delete \
            AFTER DELETE ON entities \
            BEGIN \
                DELETE FROM dream_pass4_audit WHERE entity_id = OLD.rowid; \
            END",
        (),
    )
    .await
    .map_err(step("create_cascade_trigger"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_013: entities CHECK extended with 'DreamPass4'; \
         dream_pass4_audit table + indexes + cascade-delete trigger created."
    );
    Ok(())
}
