// ─── Migration 010 ─────────────────────────────────────────────────────────

/// Migration 010: seed default entity_types vocabulary per group_id.
///
/// Backfills Migration 008's gap. Migration 008 only seeded id=0 ("Entity")
/// for groups that already had entities at migration time. Fresh DBs (no
/// prior entities) end up with an empty entity_types table for any group_id
/// used at runtime — which breaks L2 prompt rendering and yields zero-entity
/// extraction.
///
/// Migration 010 ensures every observed group_id has the full default
/// OntoNotes-style vocabulary (Entity catch-all + Person + Organisation +
/// Location + Date + Time + Money + Quantity + Event + Concept). Domain-
/// specific extensions augment via `SourceParams.entity_types_override`.
///
/// Idempotency: delegates to `ensure_default_types_seeded` which no-ops
/// when the group_id already has any entity_types rows. Safe to re-run.
pub(crate) async fn migrate_010_default_entity_types(
    conn: &libsql::Connection,
) -> anyhow::Result<()> {
    use std::collections::BTreeSet;

    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert("default".to_string());

    let mut rows = conn
        .query(
            "SELECT DISTINCT group_id FROM entities WHERE group_id IS NOT NULL \
             UNION \
             SELECT DISTINCT group_id FROM entity_types WHERE group_id IS NOT NULL",
            (),
        )
        .await
        .map_err(|e| anyhow::anyhow!("migrate_010 discover groups query failed: {e}"))?;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| anyhow::anyhow!("migrate_010 discover groups row read failed: {e}"))?
    {
        let gid: String = row
            .get(0)
            .map_err(|e| anyhow::anyhow!("migrate_010 group_id read failed: {e}"))?;
        seen.insert(gid);
    }

    let mut total_seeded: usize = 0;
    for gid in &seen {
        let inserted = crate::core::entity_types::ensure_default_types_seeded(conn, gid)
            .await
            .map_err(|e| anyhow::anyhow!("migrate_010 seed for group_id={gid} failed: {e}"))?;
        total_seeded += inserted;
    }

    tracing::info!(
        target: "kremory::migrations",
        groups = seen.len(),
        seeded_rows = total_seeded,
        "migrate_010: default entity_types vocabulary applied (idempotent)."
    );

    Ok(())
}

// ─── Migration 011 ─────────────────────────────────────────────────────────

/// Migration 011 (Phase G, ADR-042, TD-003): add `content_hash` column to
/// `episodes` with SHA-256 backfill over existing rows.
///
/// ### Motivation
///
/// The `Episode` struct has carried `content_hash: Option<String>` since
/// approximately v0.1.4.  The `episodes` table never had a matching column —
/// so the field was always `None` when populated from any `SELECT`.  This
/// migration adds the column and backfills it so that existing rows return a
/// real hash immediately after the migration runs.
///
/// SQLite has no built-in SHA-256 function, so backfill is performed in Rust:
/// we iterate all rows with `content_hash IS NULL`, compute
/// `sha2::Sha256::digest(content)`, and issue a batched UPDATE.
///
/// ### Steps
///
/// 1. PRAGMA-gate: scan `PRAGMA table_info('episodes')` for `content_hash`.
///    If already present, return early — no-op.
/// 2. `ALTER TABLE episodes ADD COLUMN content_hash TEXT` (NULL default for
///    existing rows; populated by the backfill below).
/// 3. Rust-side backfill: query `id, content` for all rows where
///    `content_hash IS NULL`, compute hex-encoded SHA-256, batch-UPDATE.
/// 4. `CREATE INDEX IF NOT EXISTS idx_episodes_content_hash ON episodes(content_hash)`.
///    Enables O(log n) future dedup queries.
///
/// ### Idempotency
///
/// G1 — `content_hash` column already present → return early.
/// Index uses `IF NOT EXISTS` — always idempotent.
/// Backfill UPDATE is filtered to `content_hash IS NULL` — safe on re-run
/// if the migration crashes between the ALTER and the UPDATE.
pub(crate) async fn migrate_011_episodes_content_hash(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_011 step `{name}` failed: {e}"
            ))
        }
    }

    // ── G1: idempotency gate ─────────────────────────────────────────────────

    let mut info = conn
        .query("PRAGMA table_info('episodes')", ())
        .await
        .map_err(step("pragma_table_info"))?;

    let mut has_content_hash = false;
    while let Some(row) = info.next().await.map_err(step("pragma_table_info_next"))? {
        let col_name: String = row.get(1).map_err(step("pragma_table_info_row_get"))?;
        if col_name == "content_hash" {
            has_content_hash = true;
            break;
        }
    }

    if has_content_hash {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_011: content_hash column already present on episodes — skipping"
        );
        return Ok(());
    }

    // ── Step 2: ADD COLUMN ───────────────────────────────────────────────────

    conn.execute("ALTER TABLE episodes ADD COLUMN content_hash TEXT", ())
        .await
        .map_err(step("alter_table_add_content_hash"))?;

    // ── Step 3: Rust-side SHA-256 backfill ───────────────────────────────────
    //
    // SQLite has no built-in sha256().  We iterate all rows that need a hash
    // (content_hash IS NULL, which is every row immediately after the ALTER)
    // and issue individual UPDATE statements within a single logical batch.
    // For fixture-sized databases (~50 rows) this is negligible.  For larger
    // production databases the one-time cost is still bounded and acceptable
    // (hashing is CPU-only; no I/O per row beyond the UPDATE).

    {
        use sha2::Digest as _;

        let mut rows = conn
            .query(
                "SELECT id, content FROM episodes WHERE content_hash IS NULL",
                (),
            )
            .await
            .map_err(step("backfill_select"))?;

        let mut updates: Vec<(i64, String)> = Vec::new();
        while let Some(row) = rows.next().await.map_err(step("backfill_row_next"))? {
            let id: i64 = row.get(0).map_err(step("backfill_row_get_id"))?;
            let content: String = row.get(1).map_err(step("backfill_row_get_content"))?;
            let hash = format!("{:x}", sha2::Sha256::digest(content.as_bytes()));
            updates.push((id, hash));
        }

        let count = updates.len();
        for (id, hash) in updates {
            conn.execute(
                "UPDATE episodes SET content_hash = ?1 WHERE id = ?2",
                libsql::params![hash, id],
            )
            .await
            .map_err(step("backfill_update"))?;
        }

        tracing::info!(
            target: "kremory::migrations",
            backfilled = count,
            "migrate_011: SHA-256 backfill complete"
        );
    }

    // ── Step 4: index ────────────────────────────────────────────────────────

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_content_hash ON episodes(content_hash)",
        (),
    )
    .await
    .map_err(step("create_idx_content_hash"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_011: content_hash column added to episodes; SHA-256 backfill done; \
         idx_episodes_content_hash created."
    );
    Ok(())
}

// ─── Migration 012 ─────────────────────────────────────────────────────────

/// Migration 012: add source-tier columns to `entities` + legacy backfill.
///
/// Adds three columns per ADR-045 §2 and ADR-046 §1:
///   - `entity_type_source TEXT CHECK(...)` — which tier last set the entity type
///   - `entity_type_assigned_at TEXT`        — ISO-8601 timestamp of the last assignment
///   - `ner_confidence REAL`                 — GLiNER / NER span confidence (Phase 1 only)
///
/// Note: `v_entity_drift_candidates` view is NOT created by this migration. Drift detection
/// is deferred to Phase E reclassify implementation per ADR-046 Amendment 2026-06-09
/// (Option E) — the original row-comparison design is structurally impossible under
/// kremory's `id = normalize_name(text)` dedup model (same name → same row always).
/// See ADR-046 §Amendment-2026-06-09 and TD-032 for rationale.
///
/// Legacy backfill: sets `entity_type_source = 'Phase1Ner'` and
/// `entity_type_assigned_at = COALESCE(recorded_at, datetime('now'))` on all existing rows
/// that have a NULL source, so queries can always rely on the column being non-NULL for rows
/// written before this migration.
///
/// Idempotency: each ADD COLUMN is guarded by a PRAGMA table_info check, so running
/// this migration twice is a no-op (no error, no duplication). The view uses
/// `CREATE VIEW IF NOT EXISTS`. The backfill UPDATE applies only to NULL-source rows.
///
/// NOTE: `'DreamPass1'` is a historical enum name (schema-locked at this migration);
/// it means "Dream Pass 2 reclassify confident output" per ADR-046 §6. The name
/// predates the ADR-046 pass-numbering convention and cannot be renamed without a
/// subsequent migration altering the CHECK constraint and existing rows.
pub(crate) async fn migrate_012_source_tier_columns(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_012 step `{name}` failed: {e}"
            ))
        }
    }

    // ── G1: idempotency gate — read existing columns ─────────────────────────

    let mut info = conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .map_err(step("pragma_table_info"))?;

    let mut has_entity_type_source = false;
    let mut has_entity_type_assigned_at = false;
    let mut has_ner_confidence = false;

    while let Some(row) = info.next().await.map_err(step("pragma_table_info_next"))? {
        let col_name: String = row.get(1).map_err(step("pragma_table_info_row_get"))?;
        match col_name.as_str() {
            "entity_type_source" => has_entity_type_source = true,
            "entity_type_assigned_at" => has_entity_type_assigned_at = true,
            "ner_confidence" => has_ner_confidence = true,
            _ => {}
        }
    }

    // ── Step 2: ADD COLUMN entity_type_source ────────────────────────────────

    if !has_entity_type_source {
        conn.execute(
            "ALTER TABLE entities ADD COLUMN entity_type_source TEXT \
             CHECK (entity_type_source IN (\
               'Phase1Ner', 'Phase2Llm', 'DreamPass0', 'DreamPass1', 'ConsumerPinned'\
             ))",
            (),
        )
        .await
        .map_err(step("alter_table_add_entity_type_source"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_012: entity_type_source already present — skipping ADD COLUMN"
        );
    }

    // ── Step 3: ADD COLUMN entity_type_assigned_at ───────────────────────────

    if !has_entity_type_assigned_at {
        conn.execute(
            "ALTER TABLE entities ADD COLUMN entity_type_assigned_at TEXT",
            (),
        )
        .await
        .map_err(step("alter_table_add_entity_type_assigned_at"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_012: entity_type_assigned_at already present — skipping ADD COLUMN"
        );
    }

    // ── Step 4: ADD COLUMN ner_confidence ────────────────────────────────────

    if !has_ner_confidence {
        conn.execute("ALTER TABLE entities ADD COLUMN ner_confidence REAL", ())
            .await
            .map_err(step("alter_table_add_ner_confidence"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_012: ner_confidence already present — skipping ADD COLUMN"
        );
    }

    // ── Step 5: Legacy backfill ───────────────────────────────────────────────
    //
    // NOTE: v_entity_drift_candidates view is intentionally NOT created here.
    // Drift detection deferred to Phase E reclassify per ADR-046 Amendment
    // 2026-06-09 (Option E). The row-comparison approach (self-JOIN on entities
    // looking for same id, different entity_type_id) is structurally impossible
    // in kremory: `id = normalize_name(text)` means same name → same row always;
    // the join condition `e2.id != e1.id AND TRIM(e2.id) = TRIM(e1.id)` can
    // never be satisfied. See TD-032 + ADR-046 §Amendment-2026-06-09 for rationale.
    //
    // All existing rows written before this migration have NULL entity_type_source.
    // Backfill them to 'Phase1Ner' (the only tier active before v0.1.1) so
    // downstream queries can rely on the column being non-NULL for pre-migration rows.
    // entity_type_assigned_at is set from recorded_at (the original ingest timestamp).
    // Idempotent: WHERE clause restricts to NULL-source rows only.

    conn.execute(
        "UPDATE entities \
         SET entity_type_source = 'Phase1Ner', \
             entity_type_assigned_at = COALESCE(recorded_at, datetime('now')) \
         WHERE entity_type_source IS NULL",
        (),
    )
    .await
    .map_err(step("backfill_source_tier"))?;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_012: entity_type_source / entity_type_assigned_at / ner_confidence \
         added to entities; legacy backfill done. \
         (drift view deferred to Phase E per ADR-046 Amendment 2026-06-09 Option E)"
    );
    Ok(())
}
