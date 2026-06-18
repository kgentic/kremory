// ─── Migration 016 ─────────────────────────────────────────────────────────────

/// Migration 016 (ADR-050 v0.2.4): dream-pass crash-safety schema cluster.
///
/// Adds three new tables and two additive PRAGMA-guarded columns required by
/// ADR-050's crash-safety + idempotency design:
///
/// ## New tables
///
/// - **`dream_idempotency_keys`** — content-hash idempotency key per
///   `(pass_name, entity_id, content_hash)` triple. Prevents re-processing
///   identical entity states across dream-pass cycles. PK is the triple;
///   supporting index on `(pass_name, entity_id)` for single-entity lookups.
///
/// - **`op_checkpoints`** — mid-corpus crash-resume cursor per
///   `(op_name, op_run_id)`. The background worker writes a cursor every N
///   episodes and reads it on boot to resume from the last safe position.
///   Supporting index on `(op_name, updated_at)` for latest-checkpoint queries.
///
/// - **`dream_pass_budget_usage`** — per-pass token spend. One row per
///   `(pass_run_id, pass_name)` pair recording provider, model, token counts,
///   and cost in micro-USD. Used for cumulative budget observability.
///
/// ## Additive columns
///
/// - **`entities.is_dream_generated`** — `INTEGER NOT NULL DEFAULT 0` flag.
///   Set to 1 by verify_stage / Pass 4 writers for entities they create or
///   amend. Pass 2 reclassify candidate-selection filters `WHERE is_dream_generated = 0`
///   to prevent dream-phase self-referential feedback loops.
///
/// - **`facts.is_dream_generated`** — same semantics for the `facts` table.
///
/// ## Idempotency
///
/// - Tables use `CREATE TABLE IF NOT EXISTS` — safe on re-run.
/// - Indexes use `CREATE INDEX IF NOT EXISTS` — safe on re-run.
/// - Columns use PRAGMA-guard (`SELECT COUNT(*) FROM pragma_table_info(...)
///   WHERE name = '...'`) before each `ALTER TABLE ADD COLUMN`. This matches
///   the [`migrate_015a_episode_processing_status`] style and prevents the
///   duplicate-column error that SQLite would otherwise raise.
///
/// ## Emergency downgrade — see [`migrate_015b_downgrade_crash_safety_schema`]
///
/// Migration 015b is the emergency downgrade for this migration. It drops the
/// three new tables and removes `is_dream_generated` from `entities` and `facts`
/// via the table-recreation pattern (per the `014c` precedent). It is
/// emergency-only and carries a data-loss surface for `is_dream_generated=1` rows.
///
/// ## Migration numbering rationale (arch spec §4.1)
///
/// v0.2.3 consumed Migration 015a (`episode_processing_status`) and its
/// downgrade was numbered 014c. This sprint's forward migration is 016; the
/// downgrade is 015b (following the `X-1` prefix convention: 015b downgrades 016,
/// mirroring 012b-downgrades-013 and 014c-downgrades-015a).
pub async fn migrate_016_crash_safety_schema(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_016 step `{name}` failed: {e}"
            ))
        }
    }

    tracing::info!(
        target: "kremory::migrations",
        migration = "016",
        "migrate_016: starting crash-safety schema migration (ADR-050)"
    );

    // ── Table 1: dream_idempotency_keys ─────────────────────────────────────
    //
    // Content-hash idempotency key per (pass_name, entity_id, content_hash).
    // PK is the triple. Supporting index on (pass_name, entity_id) allows
    // efficient single-entity idempotency lookups in run_verify_stage.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dream_idempotency_keys (
            pass_name    TEXT    NOT NULL,
            entity_id    INTEGER NOT NULL,
            content_hash TEXT    NOT NULL,
            completed_at INTEGER NOT NULL,
            PRIMARY KEY (pass_name, entity_id, content_hash)
        )",
        (),
    )
    .await
    .map_err(step("create_dream_idempotency_keys"))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dream_idempotency_keys_entity \
         ON dream_idempotency_keys (pass_name, entity_id)",
        (),
    )
    .await
    .map_err(step("create_idx_dream_idempotency_keys_entity"))?;

    // ── Table 2: op_checkpoints ─────────────────────────────────────────────
    //
    // Mid-corpus crash-resume cursor per (op_name, op_run_id).
    // The background worker writes every N episodes; on boot it reads the
    // latest row for op_name='verify_stage' and resumes from that cursor.
    // Supporting index on (op_name, updated_at) for efficient latest-row lookup.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS op_checkpoints (
            op_name    TEXT    NOT NULL,
            op_run_id  TEXT    NOT NULL,
            cursor     TEXT    NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (op_name, op_run_id)
        )",
        (),
    )
    .await
    .map_err(step("create_op_checkpoints"))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_op_checkpoints_name_updated \
         ON op_checkpoints (op_name, updated_at)",
        (),
    )
    .await
    .map_err(step("create_idx_op_checkpoints_name_updated"))?;

    // ── Table 3: dream_pass_budget_usage ────────────────────────────────────
    //
    // Per-pass token spend. One row per (pass_run_id, pass_name) pair
    // recording provider, model, token counts, and cost in micro-USD.
    // cost_usd_micro is nullable: Ollama calls record 0 cost; Anthropic calls
    // record the computed micro-USD figure. NULL means cost was not measured
    // (e.g. the provider did not report usage).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dream_pass_budget_usage (
            pass_run_id   TEXT    NOT NULL,
            pass_name     TEXT    NOT NULL,
            provider      TEXT    NOT NULL,
            model         TEXT    NOT NULL,
            tokens_input  INTEGER NOT NULL,
            tokens_output INTEGER NOT NULL,
            cost_usd_micro INTEGER,
            recorded_at   INTEGER NOT NULL,
            PRIMARY KEY (pass_run_id, pass_name)
        )",
        (),
    )
    .await
    .map_err(step("create_dream_pass_budget_usage"))?;

    // ── Column: entities.is_dream_generated ─────────────────────────────────
    //
    // PRAGMA-guard pattern (matches migrate_015a style): query pragma_table_info
    // and count matching rows before attempting ALTER TABLE ADD COLUMN.
    // A count > 0 means the column already exists; skip the ALTER to avoid
    // SQLite's duplicate-column error.
    let mut pragma_rows = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('entities') WHERE name = 'is_dream_generated'",
            (),
        )
        .await
        .map_err(step("g_pragma_entities_is_dream_generated"))?;
    let has_entities_col = if let Some(row) = pragma_rows
        .next()
        .await
        .map_err(step("g_pragma_entities_is_dream_generated_next"))?
    {
        let count: i64 = row
            .get(0)
            .map_err(step("g_pragma_entities_is_dream_generated_get"))?;
        count > 0
    } else {
        false
    };
    drop(pragma_rows);

    if !has_entities_col {
        conn.execute(
            "ALTER TABLE entities ADD COLUMN is_dream_generated INTEGER NOT NULL DEFAULT 0",
            (),
        )
        .await
        .map_err(step("add_column_entities_is_dream_generated"))?;

        tracing::debug!(
            target: "kremory::migrations",
            "migrate_016: entities.is_dream_generated column added"
        );
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_016: entities.is_dream_generated already present — skipping ALTER"
        );
    }

    // ── Column: facts.is_dream_generated ────────────────────────────────────
    //
    // Same PRAGMA-guard pattern for the facts table.
    let mut pragma_rows_facts = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('facts') WHERE name = 'is_dream_generated'",
            (),
        )
        .await
        .map_err(step("g_pragma_facts_is_dream_generated"))?;
    let has_facts_col = if let Some(row) = pragma_rows_facts
        .next()
        .await
        .map_err(step("g_pragma_facts_is_dream_generated_next"))?
    {
        let count: i64 = row
            .get(0)
            .map_err(step("g_pragma_facts_is_dream_generated_get"))?;
        count > 0
    } else {
        false
    };
    drop(pragma_rows_facts);

    if !has_facts_col {
        conn.execute(
            "ALTER TABLE facts ADD COLUMN is_dream_generated INTEGER NOT NULL DEFAULT 0",
            (),
        )
        .await
        .map_err(step("add_column_facts_is_dream_generated"))?;

        tracing::debug!(
            target: "kremory::migrations",
            "migrate_016: facts.is_dream_generated column added"
        );
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_016: facts.is_dream_generated already present — skipping ALTER"
        );
    }

    tracing::info!(
        target: "kremory::migrations",
        migration = "016",
        "migrate_016: crash-safety schema migration complete \
         (3 tables + 2 columns + 2 indexes)"
    );
    Ok(())
}
