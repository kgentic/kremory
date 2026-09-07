// ─── Migration 012b — downgrade for Migration 013 ──────────────────────────

/// Migration 012b: revert `entity_type_source` CHECK to pre-013 set (drop
/// `'DreamPass4'`) + drop `dream_pass4_audit` table.
///
/// This is a DOWNGRADE function — NOT registered in `run_migrations`. It is
/// called explicitly by tests and future downgrade scripts only.
///
/// ### Steps
///
/// 1. Idempotency guard: query `sqlite_master` for entities DDL containing
///    `'DreamPass4'`. If NOT present → already downgraded, return Ok(()).
/// 2. Re-stamp DreamPass4 entities: UPDATE entities SET
///    entity_type_source = 'Phase1Ner' WHERE entity_type_source = 'DreamPass4'.
/// 3. Table-rebuild to revert CHECK constraint (same PRAGMA FK OFF / body_result
///    pattern as migrate_013, but CHECK omits 'DreamPass4').
/// 4. DROP TABLE IF EXISTS dream_pass4_audit (indexes auto-dropped with it).
///
/// ### Idempotency
///
/// G1 — entities DDL does NOT contain 'DreamPass4' → already downgraded → skip.
pub async fn migrate_012b_revert_pass4_source_tier(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_012b step `{name}` failed: {e}"
            ))
        }
    }

    // ── G1: idempotency gate ─────────────────────────────────────────────────
    //
    // If entities DDL does NOT contain 'DreamPass4', the downgrade has already
    // been applied (or was never needed). Return immediately.

    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master \
             WHERE type = 'table' AND name = 'entities' \
             LIMIT 1",
            (),
        )
        .await
        .map_err(step("g1_sqlite_master_query"))?;

    let needs_downgrade =
        if let Some(row) = rows.next().await.map_err(step("g1_sqlite_master_next"))? {
            let ddl: Option<String> = row.get(0).map_err(step("g1_ddl_read"))?;
            ddl.map(|s| s.contains("DreamPass4")).unwrap_or(false)
        } else {
            false
        };
    // Drop the Rows handle before any DDL — an open cursor locks the connection.
    drop(rows);

    if !needs_downgrade {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_012b: entities DDL does not contain 'DreamPass4' — already downgraded (idempotent no-op)"
        );
        return Ok(());
    }

    // ── Step 2: re-stamp DreamPass4 entities to Phase1Ner ────────────────────
    //
    // Before reverting the CHECK, reclassify any rows that would violate the
    // narrowed constraint.

    conn.execute(
        "UPDATE entities \
         SET entity_type_source = 'Phase1Ner' \
         WHERE entity_type_source = 'DreamPass4'",
        (),
    )
    .await
    .map_err(step("restamp_dreampass4_entities"))?;

    // ── Step 3: table-rebuild to revert CHECK constraint ─────────────────────

    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        // Create entities_new with the original 5-value CHECK (no DreamPass4).
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
                    'DreamPass1', 'ConsumerPinned'
                )),
                entity_type_assigned_at TEXT,
                ner_confidence          REAL,
                PRIMARY KEY (id, group_id)
            )",
            (),
        )
        .await
        .map_err(step("create_entities_new"))?;

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

        let _ = conn
            .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
            .await;
        let _ = conn
            .execute("DROP INDEX IF EXISTS idx_entities_group", ())
            .await;
        let _ = conn
            .execute("DROP INDEX IF EXISTS idx_entities_type_id", ())
            .await;

        conn.execute("DROP TABLE entities", ())
            .await
            .map_err(step("drop_entities"))?;
        conn.execute("ALTER TABLE entities_new RENAME TO entities", ())
            .await
            .map_err(step("rename_entities_new"))?;

        // Recreate indexes.
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

    // Always restore PRAGMA foreign_keys = ON.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_012b: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    body_result?;

    // ── Step 4: drop dream_pass4_audit table ─────────────────────────────────
    //
    // ── Step 4a: drop cascade trigger BEFORE dropping audit table ────────────
    //
    // The trigger created by Migration 013 must be dropped explicitly. SQLite
    // does NOT auto-drop triggers that reference dropped tables — the trigger
    // would remain but become invalid, blocking entity deletes with a runtime
    // error. Downgrading removes ALL schema artifacts this migration cluster
    // added, so: drop the trigger before the table.

    conn.execute(
        "DROP TRIGGER IF EXISTS trg_dream_pass4_audit_cascade_delete",
        (),
    )
    .await
    .map_err(step("drop_cascade_trigger"))?;

    // Indexes are auto-dropped when the table is dropped.

    conn.execute("DROP TABLE IF EXISTS dream_pass4_audit", ())
        .await
        .map_err(step("drop_audit_table"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_012b: entities CHECK reverted to 5-value set (DreamPass4 removed); \
         dream_pass4_audit table dropped."
    );
    Ok(())
}

// ─── Migration 014 ────────────────────────────────────────────────────────────

/// Migration 014: add provenance columns to `entity_types` (Dream Pass 0).
///
/// Adds four columns that record how a type was discovered:
/// - `discovered_at TEXT`  — ISO-8601 timestamp when the type was discovered
/// - `discovered_by TEXT`  — source identifier (e.g. `'DreamPass0'`, `'seed'`)
/// - `evidence_count INTEGER` — count of catch-all entities that prompted the proposal
/// - `confidence REAL`     — LLM-assigned confidence [0, 1] at proposal time
///
/// Pre-existing seed types get `discovered_by = 'seed'` in the backfill.
/// All other nullable columns default to NULL for pre-migration rows.
///
/// Idempotency: PRAGMA table_info gate per column — safe to re-run.
pub(crate) async fn migrate_014_entity_types_provenance(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_014 step `{name}` failed: {e}"
            ))
        }
    }

    // ── Step 1: Inspect existing entity_types columns ────────────────────────

    let mut rows = conn
        .query("PRAGMA table_info(entity_types)", ())
        .await
        .map_err(step("pragma_table_info"))?;

    let mut has_discovered_at = false;
    let mut has_discovered_by = false;
    let mut has_evidence_count = false;
    let mut has_confidence = false;

    while let Some(row) = rows.next().await.map_err(step("pragma_row_next"))? {
        let col_name: String = row.get(1).map_err(step("pragma_col_name"))?;
        match col_name.as_str() {
            "discovered_at" => has_discovered_at = true,
            "discovered_by" => has_discovered_by = true,
            "evidence_count" => has_evidence_count = true,
            "confidence" => has_confidence = true,
            _ => {}
        }
    }

    // ── Step 2: ADD COLUMN discovered_at ─────────────────────────────────────

    if !has_discovered_at {
        conn.execute("ALTER TABLE entity_types ADD COLUMN discovered_at TEXT", ())
            .await
            .map_err(step("alter_table_add_discovered_at"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_014: discovered_at already present — skipping ADD COLUMN"
        );
    }

    // ── Step 3: ADD COLUMN discovered_by ─────────────────────────────────────

    if !has_discovered_by {
        conn.execute("ALTER TABLE entity_types ADD COLUMN discovered_by TEXT", ())
            .await
            .map_err(step("alter_table_add_discovered_by"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_014: discovered_by already present — skipping ADD COLUMN"
        );
    }

    // ── Step 4: ADD COLUMN evidence_count ────────────────────────────────────

    if !has_evidence_count {
        conn.execute(
            "ALTER TABLE entity_types ADD COLUMN evidence_count INTEGER",
            (),
        )
        .await
        .map_err(step("alter_table_add_evidence_count"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_014: evidence_count already present — skipping ADD COLUMN"
        );
    }

    // ── Step 5: ADD COLUMN confidence ────────────────────────────────────────

    if !has_confidence {
        conn.execute("ALTER TABLE entity_types ADD COLUMN confidence REAL", ())
            .await
            .map_err(step("alter_table_add_confidence"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_014: confidence already present — skipping ADD COLUMN"
        );
    }

    // ── Step 6: Backfill seed types ───────────────────────────────────────────
    //
    // Rows inserted before Migration 014 have NULL discovered_by.
    // Mark them as 'seed' so downstream queries can rely on the column
    // being non-NULL for all pre-migration rows.
    // Idempotent: WHERE clause restricts to NULL-discovered_by rows only.

    conn.execute(
        "UPDATE entity_types SET discovered_by = 'seed' WHERE discovered_by IS NULL",
        (),
    )
    .await
    .map_err(step("backfill_discovered_by_seed"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_014: discovered_at / discovered_by / evidence_count / confidence \
         added to entity_types; seed backfill done."
    );
    Ok(())
}
