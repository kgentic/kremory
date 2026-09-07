// ─── Migration 007 ─────────────────────────────────────────────────────────

/// Migration 007: add `source_id` and `source_uri` columns to the `episodes`
/// table (v0.1.6 substrate, G1).
///
/// Both columns are TEXT NULL — no NOT NULL constraint, no backfill required.
/// An index on `source_id` is created for efficient source-scoped recall.
///
/// # Idempotency
///
/// Primary gate: `PRAGMA table_info('episodes')` — if both `source_id` and
/// `source_uri` are already present the function returns `Ok(())` immediately.
/// Each `ALTER TABLE ADD COLUMN` is additionally guarded by the individual
/// column-presence flags so a partial prior run (one column added, then crash)
/// is correctly completed on the next startup.
///
/// `CREATE INDEX IF NOT EXISTS` is natively idempotent in SQLite.
///
/// # Backup
///
/// `episodes_bak_007` is created via `CREATE TABLE IF NOT EXISTS … AS SELECT`
/// before any `ALTER TABLE` statement, giving a row-level snapshot for
/// recovery. The `IF NOT EXISTS` makes this step idempotent on resume.
pub(crate) async fn migrate_007_source_id_source_uri(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    // Idempotency gate: scan PRAGMA table_info('episodes') for both columns.
    let mut info = conn
        .query("PRAGMA table_info('episodes')", ())
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_007 step `pragma_table_info` failed: {e}"
            ))
        })?;
    let mut has_source_id = false;
    let mut has_source_uri = false;
    let mut has_recorded_at = false;
    while let Some(row) = info.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_007 step `pragma_table_info_next` failed: {e}"
        ))
    })? {
        // Propagate row.get errors rather than silently
        // mapping to empty string — surface malformed PRAGMA rows to the runner.
        let col_name: String = row.get(1).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_007 step `pragma_table_info_row_get` failed: {e}"
            ))
        })?;
        if col_name == "source_id" {
            has_source_id = true;
        }
        if col_name == "source_uri" {
            has_source_uri = true;
        }
        if col_name == "recorded_at" {
            has_recorded_at = true;
        }
    }

    if has_source_id && has_source_uri && has_recorded_at {
        // All columns already present — migration already applied.
        return Ok(());
    }

    // Pre-ALTER backup: row-level snapshot of episodes in its current shape.
    // CREATE TABLE IF NOT EXISTS makes this step safe on resume-from-partial.
    // Propagate via `?` rather than silent discard so
    // disk-full / permission errors surface to the migration runner.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes_bak_007 AS SELECT * FROM episodes",
        (),
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_007 step `backup_episodes` failed: {e}"
        ))
    })?;

    // ADD COLUMN source_id TEXT (NULL) if not yet present.
    if !has_source_id {
        conn.execute("ALTER TABLE episodes ADD COLUMN source_id TEXT", ())
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "migrate_007 step `add_column_source_id` failed: {e}"
                ))
            })?;
    }

    // ADD COLUMN source_uri TEXT (NULL) if not yet present.
    if !has_source_uri {
        conn.execute("ALTER TABLE episodes ADD COLUMN source_uri TEXT", ())
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "migrate_007 step `add_column_source_uri` failed: {e}"
                ))
            })?;
    }

    // ADD COLUMN recorded_at TEXT with default if not yet present.
    // `NOT NULL DEFAULT (datetime('now'))` is valid in SQLite ALTER TABLE when
    // a DEFAULT is supplied — existing rows get the default value backfilled.
    if !has_recorded_at {
        conn.execute(
            "ALTER TABLE episodes ADD COLUMN recorded_at TEXT NOT NULL DEFAULT (datetime('now'))",
            (),
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_007 step `add_column_recorded_at` failed: {e}"
            ))
        })?;
    }

    // Index on source_id for source-scoped recall queries.
    // CREATE INDEX IF NOT EXISTS is natively idempotent.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_source_id ON episodes(source_id)",
        (),
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_007 step `create_index_source_id` failed: {e}"
        ))
    })?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_007: source_id + source_uri columns added to episodes; idx_episodes_source_id created"
    );
    Ok(())
}

// ─── Migration 008 ─────────────────────────────────────────────────────────

/// Migration 008: introduce `entity_types` registry table + `entity_type_id`
/// column on `entities` (unified extraction architecture).
///
/// ### Steps
///
/// 1. Create `entity_types (group_id, id, name, description, ...)` table —
///    composite PK `(group_id, id)`, UNIQUE on `(group_id, name)`.
/// 2. Seed id=0 `"Entity"` catch-all row for every existing `group_id` in
///    `entities`. The anti-junk description is embedded so the LLM's id=0
///    fallback is governed by an explicit guard rather than a bare placeholder.
/// 3. Seed observed labels (distinct `(group_id, label)` pairs from `entities`)
///    as `entity_types` rows starting at `id=1`, alphabetically ordered.
///    `label = 'Entity'` or `NULL` skips (already covered by id=0).
/// 4. Add `entity_type_id INTEGER NOT NULL DEFAULT 0` column to `entities`.
/// 5. Backfill `entity_type_id` from the newly seeded registry by matching
///    `entity_types.name = entities.label` within the same `group_id`.
///    Entities whose label is not found (or is NULL / `'Entity'`) keep id=0.
/// 6. Create composite index `idx_entities_type_id ON entities(group_id, entity_type_id)`.
///
/// ### Label DROP deferred to Phase 2
///
/// The spec §2 Step 4 specifies `ALTER TABLE entities DROP COLUMN label`.
/// That step CANNOT be applied in Phase 1 because `Entity.label` is a
/// load-bearing struct field: `graph.rs::row_to_entity` reads label at
/// column index 1, all INSERT/SELECT queries include `label`, and several
/// callers (ingest, search, resolver, engine_handle) access `entity.label`
/// directly. Dropping the column while the Rust code still reads it from DB
/// rows causes runtime row-get failures for every entity query.
///
/// Phase 2 will: (a) replace `Entity.label` with `label()` as a computed
/// accessor that resolves via `entity_type_id → entity_types.name`, (b) update
/// all SQL projections in `graph.rs` and `search.rs`, (c) then ship the
/// `ALTER TABLE entities DROP COLUMN label` as part of that atomic commit.
///
/// This Phase 1 migration is safe to apply before Phase 2: `entity_type_id`
/// is populated, the registry is live, and all new code in Phase 2 can rely
/// on both columns being present during the transition window.
///
/// ### Idempotency
///
/// G1 — `entity_type_id` column already present on `entities` → skip (already ran).
/// `CREATE TABLE IF NOT EXISTS`, `INSERT OR IGNORE`, `CREATE INDEX IF NOT EXISTS`
/// are individually idempotent for steps 1–3 and 6. Step 4–5 are gated on G1.
pub(crate) async fn migrate_008_entity_types(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_008 step `{name}` failed: {e}"
            ))
        }
    }

    // ── Step 1: create entity_types table ──────────────────────────────────

    conn.execute(
        "CREATE TABLE IF NOT EXISTS entity_types (
            id           INTEGER NOT NULL,
            group_id     TEXT NOT NULL,
            name         TEXT NOT NULL,
            description  TEXT NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            last_used_at TEXT,
            use_count    INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (group_id, id),
            UNIQUE (group_id, name)
        )",
        (),
    )
    .await
    .map_err(step("create_entity_types_table"))?;

    // ── Step 2: seed id=0 "Entity" catch-all per distinct group_id ─────────
    //
    // INSERT OR IGNORE ensures this is a no-op on re-run (PK conflict skips).
    // The anti-junk description (per spike #1c) prevents LLM from treating
    // "Entity" as a valid extraction target.

    conn.execute(
        "INSERT OR IGNORE INTO entity_types (group_id, id, name, description)
         SELECT DISTINCT
             COALESCE(group_id, 'default') AS group_id,
             0 AS id,
             'Entity' AS name,
             'Generic catch-all. Use ONLY when entity does not match any other type. \
              DO NOT use for placeholders, pronouns, or generic nouns like \
              ''thing'', ''item'', ''person''.' AS description
         FROM entities",
        (),
    )
    .await
    .map_err(step("seed_entity_catch_all"))?;

    // ── Step 3: seed observed labels per group_id (alphabetical → id=1,2,…) ─
    //
    // ROW_NUMBER() OVER (PARTITION BY group_id ORDER BY label) assigns ids
    // starting at 1 within each group. Skips 'Entity' and NULL labels (covered
    // by id=0). INSERT OR IGNORE is a no-op on re-run (UNIQUE(group_id,name)).
    //
    // Gate: migration 009 (Phase 2 co-commit) drops entities.label. On a second
    // run of all migrations, label is absent. Skip Step 3 when label is gone —
    // the seeding was already performed on the first migration run.
    let mut label_col_info = conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .map_err(step("step3_pragma_table_info"))?;
    let mut entities_has_label = false;
    while let Some(row) = label_col_info
        .next()
        .await
        .map_err(step("step3_pragma_next"))?
    {
        let col_name: String = row.get(1).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_008 step `step3_col_name_get` failed: {e}"
            ))
        })?;
        if col_name == "label" {
            entities_has_label = true;
            break;
        }
    }

    if entities_has_label {
        conn.execute(
            "WITH labelled AS (
                 SELECT
                     COALESCE(group_id, 'default') AS group_id,
                     label,
                     CAST(ROW_NUMBER() OVER (
                         PARTITION BY COALESCE(group_id, 'default')
                         ORDER BY label
                     ) AS INTEGER) AS rn
                 FROM (
                     SELECT DISTINCT
                         COALESCE(group_id, 'default') AS group_id,
                         label
                     FROM entities
                     WHERE label IS NOT NULL
                       AND label != 'Entity'
                 ) AS distinct_labels
             )
             INSERT OR IGNORE INTO entity_types (group_id, id, name, description)
             SELECT
                 group_id,
                 rn,
                 label,
                 'Auto-seeded from v0.1.6 migration. Label observed in entities table. Refine description post-migration.'
             FROM labelled",
            (),
        )
        .await
        .map_err(step("seed_observed_labels"))?;
    }

    // ── G1: idempotency gate for column-level changes ──────────────────────
    //
    // Scan PRAGMA table_info('entities') for entity_type_id.
    // If present, steps 4–5 have already been applied — skip them.

    let mut info = conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .map_err(step("g1_pragma_table_info"))?;
    let mut has_entity_type_id = false;
    while let Some(row) = info
        .next()
        .await
        .map_err(step("g1_pragma_table_info_next"))?
    {
        let col_name: String = row.get(1).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_008 step `g1_pragma_row_get` failed: {e}"
            ))
        })?;
        if col_name == "entity_type_id" {
            has_entity_type_id = true;
            break;
        }
    }

    if !has_entity_type_id {
        // ── Step 4: add entity_type_id column to entities ──────────────────

        // Pre-alter backup: row-level snapshot for recovery.
        // CREATE TABLE IF NOT EXISTS makes this idempotent on resume.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entities_bak_008 AS SELECT * FROM entities",
            (),
        )
        .await
        .map_err(step("backup_entities"))?;

        conn.execute(
            "ALTER TABLE entities ADD COLUMN entity_type_id INTEGER NOT NULL DEFAULT 0",
            (),
        )
        .await
        .map_err(step("add_column_entity_type_id"))?;

        // ── Step 5: backfill entity_type_id from registry ──────────────────
        //
        // Match entity_types.name = entities.label within the same group_id.
        // Entities whose label is NULL, 'Entity', or not present in the registry
        // keep the DEFAULT 0 (catch-all). This is safe: the INSERT OR IGNORE
        // steps above guarantee every group_id has an id=0 row.

        conn.execute(
            "UPDATE entities
             SET entity_type_id = COALESCE(
                 (SELECT et.id
                  FROM entity_types et
                  WHERE et.group_id = COALESCE(entities.group_id, 'default')
                    AND et.name = entities.label
                  LIMIT 1),
                 0
             )
             WHERE entity_type_id = 0
               AND label IS NOT NULL
               AND label != 'Entity'",
            (),
        )
        .await
        .map_err(step("backfill_entity_type_id"))?;
    }

    // ── Step 6: composite index (IF NOT EXISTS — always idempotent) ────────

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_type_id \
         ON entities(group_id, entity_type_id)",
        (),
    )
    .await
    .map_err(step("create_idx_entities_type_id"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_008: entity_types table created + seeded; \
         entity_type_id column added to entities + backfilled; \
         idx_entities_type_id created. \
         NOTE: label column retained; Phase 2 will swap callers then DROP label."
    );
    Ok(())
}

// ─── Migration 009 ─────────────────────────────────────────────────────────

/// Migration 009: DROP the `label` column from `entities`.
///
/// ### Pre-conditions
///
/// Migration 008 must have already run (entity_type_id column must exist on
/// `entities` and the `entity_types` registry table must exist).  Migration 009
/// is gated on Phase 2 code: all INSERT/SELECT paths already use `entity_type_id`
/// and resolve label via LEFT JOIN on `entity_types` at query time.
///
/// ### Steps
///
/// 1. PRAGMA-gate: scan `PRAGMA table_info('entities')` for the `label` column.
///    If absent (fresh DB or already dropped), return early — no-op.
/// 2. `ALTER TABLE entities DROP COLUMN label` — drops the column.
/// 3. Prune `entities_fts` of the now-redundant `label` column entries.
///    FTS5 cannot ALTER; a full FTS5 rebuild is triggered via
///    `INSERT INTO entities_fts(entities_fts) VALUES('rebuild')`.
///
/// ### Idempotency
///
/// G1 — `label` column absent on `entities` → skip all steps. Safe to call
/// `run_migrations` multiple times (the gate prevents double-apply).
///
/// ### FTS5 rebuild note
///
/// The `entities_fts` virtual table is an FTS5 content table pointing at
/// `entities`.  Dropping `entities.label` desynchronises the FTS index.
/// A `rebuild` command re-indexes all rows from the base table.  This may
/// be slow on large datasets but is correct and idempotent.
pub(crate) async fn migrate_009_drop_label_column(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_009 step `{name}` failed: {e}"
            ))
        }
    }

    // ── G1 gate: check whether label column still exists ────────────────────

    let mut pragma_rows = conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .map_err(step("pragma_table_info"))?;

    let mut has_label = false;
    while let Some(row) = pragma_rows.next().await.map_err(step("pragma_row_next"))? {
        // PRAGMA table_info columns: cid(0), name(1), type(2), notnull(3), dflt_value(4), pk(5)
        let col_name: String = row.get::<String>(1).map_err(step("pragma_col_name_read"))?;
        if col_name == "label" {
            has_label = true;
            break;
        }
    }

    if !has_label {
        // Already dropped (fresh DB or second run) — idempotent no-op.
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_009: label column already absent from entities — skipping"
        );
        return Ok(());
    }

    // ── Step 1: drop the label column ───────────────────────────────────────
    //
    // ALTER TABLE ... DROP COLUMN is supported in SQLite ≥ 3.35.0 (2021-03-12).
    // libsql and the bundled sqlite3 shipped with kremory meet this requirement.
    // The column is NOT a PRIMARY KEY component nor referenced in any index that
    // still needs to serve queries (entities_fts is rebuilt below).

    conn.execute("ALTER TABLE entities DROP COLUMN label", ())
        .await
        .map_err(step("alter_table_drop_label"))?;

    // ── Step 2: rebuild FTS5 index ───────────────────────────────────────────
    //
    // FTS5 content tables track base table columns by position.  After dropping
    // `label` the position-based column references inside the FTS index are
    // stale.  A `rebuild` command flushes all FTS data and re-indexes the base
    // table from scratch.  This is the canonical SQLite FTS5 repair pattern.

    conn.execute(
        "INSERT INTO entities_fts(entities_fts) VALUES('rebuild')",
        (),
    )
    .await
    .map_err(step("fts_rebuild"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_009: entities.label column dropped; FTS5 index rebuilt."
    );

    Ok(())
}
