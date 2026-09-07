// ─── Migration 015b — emergency downgrade for Migration 016 ────────────────────
//
// Emergency-only: data loss surface for is_dream_generated=1 rows — any entities
// or facts whose is_dream_generated flag was set to 1 will lose that information
// on downgrade. The three new tables are dropped without data migration.
//
// Table-recreation is required for is_dream_generated column removal because
// SQLite's DROP COLUMN (available since 3.35.0) has additional constraints on
// NOT NULL columns and may not be available on all libsql bundled SQLite builds.
// Table-recreation matches the 014c precedent for maximum portability.
//
// Named "015b" per the kremory convention: the downgrade migration is named after
// the preceding stable migration number + letter suffix, where the numeric portion
// is one less than the forward migration being reverted:
//   012b downgrades 013; 014c downgrades 015a; 015b downgrades 016.

/// Migration 015b (emergency downgrade for Migration 016): revert the
/// crash-safety schema cluster.
///
/// # Emergency-only
///
/// This function is NOT called from `run_migrations`. Invoke it only when a hard
/// rollback of Migration 016 is required AND you have verified a backup exists
/// (see `backup_workspace`). Production consumers are expected to upgrade once
/// and never downgrade.
///
/// # Operations performed
///
/// 1. **Drop** `dream_idempotency_keys` (all rows lost — emergency only)
/// 2. **Drop** `op_checkpoints` (all rows lost — emergency only)
/// 3. **Drop** `dream_pass_budget_usage` (all rows lost — emergency only)
/// 4. **Table-recreation** for `entities` — remove `is_dream_generated` column
/// 5. **Table-recreation** for `facts` — remove `is_dream_generated` column
///
/// # Data-loss surface
///
/// - All rows in `dream_idempotency_keys`, `op_checkpoints`, and
///   `dream_pass_budget_usage` are permanently deleted.
/// - The `is_dream_generated=1` flag on existing `entities` and `facts` rows
///   is permanently lost (column stripped by table-recreation).
/// - All other columns and data in `entities` and `facts` are preserved.
///
/// # Idempotency
///
/// The column-absence PRAGMA-guard at the top returns immediately when
/// `is_dream_generated` is already absent from `entities`. This is a
/// meaningful idempotency shortcut: if 016 was never applied, the guard
/// fires and the function is a no-op. If the column is present on entities
/// but not facts (partial apply), the function proceeds from step 4 onward.
///
/// The table drops use `DROP TABLE IF EXISTS` — idempotent.
///
/// # Stuck-state warning (operator-only)
///
/// If `migrate_015b` fails between the RENAME (step 4a) and the INSERT (step 4c)
/// on either table, the connection is left with `entities_bak_015b` or
/// `facts_bak_015b` (data) and a freshly-recreated but potentially empty canonical
/// table. Recovery: run `INSERT INTO entities SELECT <cols> FROM entities_bak_015b`
/// then `DROP TABLE entities_bak_015b` (and the same for facts). Inspect schema
/// before invoking downgrade in production.
pub async fn migrate_015b_downgrade_crash_safety_schema(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_015b step `{name}` failed: {e}"
            ))
        }
    }

    tracing::info!(
        target: "kremory::migrations",
        migration = "015b",
        "migrate_015b: starting crash-safety schema downgrade"
    );

    // ── G1: idempotency gate ─────────────────────────────────────────────────
    //
    // If entities.is_dream_generated is already absent, Migration 016 was never
    // applied (or was already downgraded). Return immediately as a no-op.
    let mut pragma_rows = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('entities') WHERE name = 'is_dream_generated'",
            (),
        )
        .await
        .map_err(step("g1_pragma_entities"))?;
    let has_entities_col = if let Some(row) = pragma_rows
        .next()
        .await
        .map_err(step("g1_pragma_entities_next"))?
    {
        let count: i64 = row.get(0).map_err(step("g1_pragma_entities_get"))?;
        count > 0
    } else {
        false
    };
    drop(pragma_rows);

    if !has_entities_col {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_015b: entities.is_dream_generated absent — downgrade is a no-op"
        );
        return Ok(());
    }

    // ── Step 1: drop new tables (data loss — emergency only) ─────────────────
    //
    // DROP TABLE IF EXISTS is idempotent — safe if the table was never created
    // or was already dropped.
    conn.execute("DROP TABLE IF EXISTS dream_idempotency_keys", ())
        .await
        .map_err(step("drop_dream_idempotency_keys"))?;

    conn.execute("DROP TABLE IF EXISTS op_checkpoints", ())
        .await
        .map_err(step("drop_op_checkpoints"))?;

    conn.execute("DROP TABLE IF EXISTS dream_pass_budget_usage", ())
        .await
        .map_err(step("drop_dream_pass_budget_usage"))?;

    tracing::debug!(
        target: "kremory::migrations",
        "migrate_015b: 3 dream-phase tables dropped"
    );

    // ── PRAGMA foreign_keys = OFF ────────────────────────────────────────────
    //
    // `entities` is referenced by `facts` (subject_id, object_id) and
    // `episodic_edges` (entity_id). `facts` is not referenced by others.
    // Table-recreation (RENAME → CREATE → INSERT → DROP) requires FK enforcement
    // to be suspended so referential checks don't fire against the intermediate state.
    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        // ── Step 2: table-recreation for `entities` ──────────────────────────
        //
        // Recreates `entities` without `is_dream_generated`. The column set
        // matches the post-Migration-013 shape (the schema that was live BEFORE
        // Migration 016 added is_dream_generated):
        //   id TEXT NOT NULL, properties TEXT, embedding BLOB,
        //   recorded_at TEXT NOT NULL, updated_at TEXT,
        //   group_id TEXT NOT NULL DEFAULT 'default',
        //   access_count INTEGER NOT NULL DEFAULT 0,
        //   entity_type_id INTEGER NOT NULL DEFAULT 0,
        //   entity_type_source TEXT CHECK (...), entity_type_assigned_at TEXT,
        //   ner_confidence REAL
        //   PRIMARY KEY (id, group_id)
        //   + is_dream_generated INTEGER NOT NULL DEFAULT 0  ← this is what we DROP
        //
        // NOTE: label was dropped by Migration 009. The composite PK (id, group_id)
        // replaced the single-column PK in Migration 008+009 era.
        //
        // We cannot know the embedding dimension at this point (it's stored in the
        // column type) — but SQLite stores the type as a text annotation only;
        // PRAGMA table_info returns 'F32_BLOB(N)' as the type string. For the
        // recreation we use the generic form; libsql treats unknown types as BLOB.
        // The CREATE TABLE ... AS SELECT pattern is NOT used here (unlike the backup
        // step in migrate_015a) because we need explicit column definitions to
        // control the final schema precisely.
        //
        // Backup first for recovery on partial apply.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entities_bak_015b AS SELECT * FROM entities",
            (),
        )
        .await
        .map_err(step("backup_entities"))?;

        // Recreate entities without is_dream_generated.
        // Use the schema that was live BEFORE Migration 016 added the column.
        // This is the post-Migration-013 shape: label was dropped by Migration 009;
        // entity_type_id/source/assigned_at/ner_confidence were added by Migrations 008-010.
        // The embedding type is preserved as a generic BLOB alias — libsql maps it to BLOB.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entities_new_015b (
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
        .map_err(step("recreate_entities"))?;

        // Copy rows from backup — explicitly list pre-016 columns only.
        conn.execute(
            "INSERT INTO entities_new_015b (
                 id, properties, embedding, recorded_at, updated_at,
                 group_id, access_count, entity_type_id,
                 entity_type_source, entity_type_assigned_at, ner_confidence
             )
             SELECT
                 id, properties, embedding, recorded_at, updated_at,
                 group_id, access_count, entity_type_id,
                 entity_type_source, entity_type_assigned_at, ner_confidence
             FROM entities_bak_015b",
            (),
        )
        .await
        .map_err(step("copy_entities_from_backup"))?;

        // Drop the old entities table (the one WITH is_dream_generated).
        conn.execute("DROP TABLE entities", ())
            .await
            .map_err(step("drop_entities_old"))?;

        // Rename the new table to the canonical name.
        conn.execute("ALTER TABLE entities_new_015b RENAME TO entities", ())
            .await
            .map_err(step("rename_entities_new_to_entities"))?;

        // Drop the backup.
        conn.execute("DROP TABLE entities_bak_015b", ())
            .await
            .map_err(step("drop_entities_backup"))?;

        // Recreate the indexes on entities that existed before Migration 016.
        //   idx_entities_group (group_id) — from base DDL + Migration 008
        // Note: entities_vec_idx (vector index) is omitted — it requires
        // libsql_vector_idx() which may not be available in all builds, and
        // it was created as a non-fatal step in run_migrations originally.
        // Consumers needing the vector index should re-open via TemporalGraph::open().
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id)",
                (),
            )
            .await;

        tracing::debug!(
            target: "kremory::migrations",
            "migrate_015b: entities table recreated without is_dream_generated"
        );

        // ── Step 3: table-recreation for `facts` ─────────────────────────────
        //
        // Recreates `facts` without `is_dream_generated`. The column set matches
        // the DDL in schema.rs PLUS Migration 006 (composite FK) columns.
        //
        // Current canonical facts schema (post all migrations including 016):
        //   id INTEGER PK AUTOINCREMENT, subject_id TEXT NOT NULL,
        //   predicate TEXT NOT NULL, object_id TEXT, object_value TEXT,
        //   properties TEXT, embedding BLOB, valid_from TEXT NOT NULL,
        //   valid_to TEXT, recorded_at TEXT NOT NULL, expired_at TEXT,
        //   invalid_at TEXT, group_id TEXT, confidence REAL DEFAULT 1.0,
        //   source_episode_id INTEGER, memory_type TEXT, content_hash TEXT,
        //   access_count INTEGER NOT NULL DEFAULT 0,
        //   subject_group_id TEXT, object_group_id TEXT   ← from Migration 006
        //   + is_dream_generated INTEGER NOT NULL DEFAULT 0  ← this is what we DROP
        //
        // FK relationships: subject_id → entities(id), object_id → entities(id),
        // source_episode_id → episodes(id). After Migration 006: composite FKs
        // on (subject_id, subject_group_id) and (object_id, object_group_id).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts_bak_015b AS SELECT * FROM facts",
            (),
        )
        .await
        .map_err(step("backup_facts"))?;

        // Recreate facts without is_dream_generated and without FK constraints
        // (FK constraints are enforced OFF during this block; they will be
        // re-enabled after the body_result).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts_new_015b (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id       TEXT    NOT NULL,
                predicate        TEXT    NOT NULL,
                object_id        TEXT,
                object_value     TEXT,
                properties       TEXT,
                embedding        BLOB,
                valid_from       TEXT    NOT NULL,
                valid_to         TEXT,
                recorded_at      TEXT    NOT NULL,
                expired_at       TEXT,
                invalid_at       TEXT,
                group_id         TEXT,
                confidence       REAL    DEFAULT 1.0,
                source_episode_id INTEGER,
                memory_type      TEXT,
                content_hash     TEXT,
                access_count     INTEGER NOT NULL DEFAULT 0,
                subject_group_id TEXT,
                object_group_id  TEXT
            )",
            (),
        )
        .await
        .map_err(step("recreate_facts"))?;

        // Copy rows — explicitly list pre-016 columns only.
        conn.execute(
            "INSERT INTO facts_new_015b (
                 id, subject_id, predicate, object_id, object_value,
                 properties, embedding, valid_from, valid_to, recorded_at,
                 expired_at, invalid_at, group_id, confidence,
                 source_episode_id, memory_type, content_hash, access_count,
                 subject_group_id, object_group_id
             )
             SELECT
                 id, subject_id, predicate, object_id, object_value,
                 properties, embedding, valid_from, valid_to, recorded_at,
                 expired_at, invalid_at, group_id, confidence,
                 source_episode_id, memory_type, content_hash, access_count,
                 subject_group_id, object_group_id
             FROM facts_bak_015b",
            (),
        )
        .await
        .map_err(step("copy_facts_from_backup"))?;

        conn.execute("DROP TABLE facts", ())
            .await
            .map_err(step("drop_facts_old"))?;

        conn.execute("ALTER TABLE facts_new_015b RENAME TO facts", ())
            .await
            .map_err(step("rename_facts_new_to_facts"))?;

        conn.execute("DROP TABLE facts_bak_015b", ())
            .await
            .map_err(step("drop_facts_backup"))?;

        // Recreate facts indexes that existed before Migration 016.
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_temporal \
                 ON facts(subject_id, valid_from, expired_at)",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_predicate \
                 ON facts(predicate, expired_at)",
                (),
            )
            .await;

        tracing::debug!(
            target: "kremory::migrations",
            "migrate_015b: facts table recreated without is_dream_generated"
        );

        Ok(())
    }
    .await;

    // Always restore PRAGMA foreign_keys = ON regardless of body outcome.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_015b: failed to re-enable PRAGMA foreign_keys — \
             connection FK state is inconsistent (still OFF); operator must reconnect"
        );
    }

    body_result?;

    tracing::info!(
        target: "kremory::migrations",
        migration = "015b",
        "migrate_015b: crash-safety schema downgrade complete \
         (3 tables dropped + 2 columns removed via table-recreation)"
    );
    Ok(())
}
