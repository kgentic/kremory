#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 016 — crash-safety schema tests.
//!
//! Governing spec: `.ai-docs/specs/v0-2-4-crash-safety-arch-spec-2026-06-12.md` §4.4
//! Governing ADR:  ADR-050 — dream-pass crash-safety + idempotency cluster
//!
//! Test coverage (4 cases per arch spec §4.4):
//!
//! 1. `migration_016_up_creates_all_tables` — 3 new tables + 2 indexes exist post-upgrade.
//! 2. `migration_016_up_adds_is_dream_generated_columns` — both columns present, DEFAULT 0.
//! 3. `migration_016_round_trip` — 015a → 016 → 015b → 016 preserves schema + data.
//! 4. `migration_016_idempotent_when_already_applied` — second run_migrations is a no-op.

use kremory::core::schema::TemporalGraph;

// ─── helpers ─────────────────────────────────────────────────────────────────

async fn open_file_backed_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-mig-016.db");
    let graph = TemporalGraph::open(path.to_str().expect("path utf-8"))
        .await
        .expect("TemporalGraph::open must succeed on fresh DB");
    (graph, tmp)
}

async fn object_exists(graph: &TemporalGraph, kind: &str, name: &str) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = ?1 AND name = ?2",
            libsql::params![kind, name],
        )
        .await
        .expect("sqlite_master query must succeed");
    let row = rows.next().await.expect("iter").expect("row");
    row.get::<i64>(0).expect("count") > 0
}

async fn column_names(graph: &TemporalGraph, table: &str) -> Vec<String> {
    let mut rows = graph
        .conn
        .query(&format!("PRAGMA table_info('{table}')"), ())
        .await
        .unwrap_or_else(|_| panic!("PRAGMA table_info('{table}')"));
    let mut names = Vec::new();
    while let Some(row) = rows.next().await.expect("iter") {
        names.push(row.get::<String>(1).expect("col name"));
    }
    names
}

async fn has_col(graph: &TemporalGraph, table: &str, col: &str) -> bool {
    column_names(graph, table).await.contains(&col.to_string())
}

async fn row_count(graph: &TemporalGraph, table: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(&format!("SELECT COUNT(*) FROM {table}"), ())
        .await
        .unwrap_or_else(|_| panic!("COUNT(*) FROM {table}"));
    let row = rows.next().await.expect("iter").expect("row");
    row.get::<i64>(0).expect("count")
}

async fn insert_entity(graph: &TemporalGraph, id: &str) {
    // Minimal insert — only NOT NULL columns required by the current entities schema
    // (post-Migration 009: entity_type_id replaces the old label text column).
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, recorded_at, group_id) \
             VALUES (?1, datetime('now'), 'default')",
            libsql::params![id],
        )
        .await
        .expect("entity insert must succeed");
}

// ─── Test 1: all tables and indexes created ───────────────────────────────────

/// Migration 016 must create the 3 new tables and 2 supporting indexes.
///
/// Verifies arch spec §4.2 table/index inventory:
///   - `dream_idempotency_keys` + `idx_dream_idempotency_keys_entity`
///   - `op_checkpoints` + `idx_op_checkpoints_name_updated`
///   - `dream_pass_budget_usage` (no supporting index beyond PK)
///
/// All checked via `sqlite_master` (source of truth — not PRAGMA table_info
/// which does not report indexes).
#[tokio::test]
async fn migration_016_up_creates_all_tables() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // ── Tables ─────────────────────────────────────────────────────────────────

    assert!(
        object_exists(&graph, "table", "dream_idempotency_keys").await,
        "dream_idempotency_keys table must exist after Migration 016"
    );

    assert!(
        object_exists(&graph, "table", "op_checkpoints").await,
        "op_checkpoints table must exist after Migration 016"
    );

    assert!(
        object_exists(&graph, "table", "dream_pass_budget_usage").await,
        "dream_pass_budget_usage table must exist after Migration 016"
    );

    // ── Indexes ────────────────────────────────────────────────────────────────

    assert!(
        object_exists(&graph, "index", "idx_dream_idempotency_keys_entity").await,
        "idx_dream_idempotency_keys_entity must exist after Migration 016 \
         (arch spec §4.2 — supports single-entity idempotency lookups)"
    );

    assert!(
        object_exists(&graph, "index", "idx_op_checkpoints_name_updated").await,
        "idx_op_checkpoints_name_updated must exist after Migration 016 \
         (arch spec §4.2 — supports latest-checkpoint queries)"
    );

    // ── Spot-check column presence on dream_idempotency_keys ──────────────────
    //
    // Full column set from arch spec §4.2:
    //   pass_name TEXT NOT NULL, entity_id INTEGER NOT NULL,
    //   content_hash TEXT NOT NULL, completed_at INTEGER NOT NULL
    let dik_cols = column_names(&graph, "dream_idempotency_keys").await;
    for col in &["pass_name", "entity_id", "content_hash", "completed_at"] {
        assert!(
            dik_cols.contains(&(*col).to_string()),
            "dream_idempotency_keys must have column '{col}'; present: {dik_cols:?}"
        );
    }

    // ── Spot-check op_checkpoints columns ─────────────────────────────────────
    let ocp_cols = column_names(&graph, "op_checkpoints").await;
    for col in &["op_name", "op_run_id", "cursor", "updated_at"] {
        assert!(
            ocp_cols.contains(&(*col).to_string()),
            "op_checkpoints must have column '{col}'; present: {ocp_cols:?}"
        );
    }

    // ── Spot-check dream_pass_budget_usage columns ────────────────────────────
    let dpb_cols = column_names(&graph, "dream_pass_budget_usage").await;
    for col in &[
        "pass_run_id",
        "pass_name",
        "provider",
        "model",
        "tokens_input",
        "tokens_output",
        "cost_usd_micro",
        "recorded_at",
    ] {
        assert!(
            dpb_cols.contains(&(*col).to_string()),
            "dream_pass_budget_usage must have column '{col}'; present: {dpb_cols:?}"
        );
    }
}

// ─── Test 2: is_dream_generated columns added on entities and facts ───────────

/// Migration 016 must add `is_dream_generated INTEGER NOT NULL DEFAULT 0` to
/// both `entities` and `facts`.
///
/// Verifies arch spec §4.2 additive columns.  Also verifies the DEFAULT 0
/// applies to rows inserted after migration via a raw INSERT + SELECT.
#[tokio::test]
async fn migration_016_up_adds_is_dream_generated_columns() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // ── Column presence ────────────────────────────────────────────────────────

    assert!(
        has_col(&graph, "entities", "is_dream_generated").await,
        "entities.is_dream_generated must be present after Migration 016"
    );
    assert!(
        has_col(&graph, "facts", "is_dream_generated").await,
        "facts.is_dream_generated must be present after Migration 016"
    );

    // ── DEFAULT 0 on entities ─────────────────────────────────────────────────
    //
    // Insert without specifying is_dream_generated and verify it defaults to 0.
    graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, recorded_at, group_id) \
             VALUES ('test-entity-mig016-default', datetime('now'), 'default')",
            (),
        )
        .await
        .expect("entity insert without is_dream_generated must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT is_dream_generated FROM entities WHERE id = 'test-entity-mig016-default'",
            (),
        )
        .await
        .expect("SELECT is_dream_generated from entities must succeed");
    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("inserted entity must be retrievable");
    let val: i64 = row.get(0).expect("is_dream_generated at column 0");
    assert_eq!(
        val, 0,
        "entities.is_dream_generated must default to 0 for new rows; got {val}"
    );

    // ── Explicit is_dream_generated = 1 roundtrip ──────────────────────────────
    //
    // Verify the column is writable (not just readable at DEFAULT).
    graph
        .conn
        .execute(
            "UPDATE entities SET is_dream_generated = 1 \
             WHERE id = 'test-entity-mig016-default'",
            (),
        )
        .await
        .expect("UPDATE entities.is_dream_generated must succeed");

    let mut rows2 = graph
        .conn
        .query(
            "SELECT is_dream_generated FROM entities WHERE id = 'test-entity-mig016-default'",
            (),
        )
        .await
        .expect("SELECT is_dream_generated after update must succeed");
    let row2 = rows2
        .next()
        .await
        .expect("row iteration must not error")
        .expect("updated entity must be retrievable");
    let val2: i64 = row2.get(0).expect("is_dream_generated at column 0");
    assert_eq!(
        val2, 1,
        "entities.is_dream_generated must be writable; expected 1 after UPDATE, got {val2}"
    );
}

// ─── Test 3: round-trip — 015a → 016 → 015b → 016 ────────────────────────────

/// Round-trip test: migrate up to 016, downgrade via 015b, then re-apply 016.
///
/// Verifies arch spec §4.4 round-trip DoD item:
///   - After 016 forward: 3 tables + 2 columns present.
///   - After 015b rollback: 3 tables absent, 2 columns absent.
///   - After 016 re-apply: 3 tables + 2 columns present again.
///   - Data rows inserted before downgrade are preserved through the
///     table-recreation (only the `is_dream_generated` column is stripped).
///
/// This test calls the migration functions directly rather than via
/// `run_migrations` to exercise the downgrade path, which is not wired into
/// the normal startup sequence (015b is emergency-only).
#[tokio::test]
async fn migration_016_round_trip() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // After normal open, all migrations including 016 have run.
    // Verify the 016 forward state is correct.
    assert!(
        object_exists(&graph, "table", "dream_idempotency_keys").await,
        "[round-trip SETUP] dream_idempotency_keys must exist before downgrade"
    );
    assert!(
        object_exists(&graph, "table", "op_checkpoints").await,
        "[round-trip SETUP] op_checkpoints must exist before downgrade"
    );
    assert!(
        object_exists(&graph, "table", "dream_pass_budget_usage").await,
        "[round-trip SETUP] dream_pass_budget_usage must exist before downgrade"
    );

    assert!(
        has_col(&graph, "entities", "is_dream_generated").await,
        "[round-trip SETUP] entities.is_dream_generated must exist before downgrade"
    );
    assert!(
        has_col(&graph, "facts", "is_dream_generated").await,
        "[round-trip SETUP] facts.is_dream_generated must exist before downgrade"
    );

    // Insert a data row in entities to verify table-recreation preserves data.
    insert_entity(&graph, "round-trip-entity-01").await;
    let entity_count_before = row_count(&graph, "entities").await;
    assert!(
        entity_count_before >= 1,
        "[round-trip SETUP] entities must have ≥1 row after insert; got {entity_count_before}"
    );

    // ── 015b downgrade ────────────────────────────────────────────────────────
    kremory::core::migrations::migrate_015b_downgrade_crash_safety_schema(&graph.conn)
        .await
        .expect("015b downgrade must succeed");

    // Verify tables are gone.
    assert!(
        !object_exists(&graph, "table", "dream_idempotency_keys").await,
        "[round-trip DOWN] dream_idempotency_keys must be absent after 015b downgrade"
    );
    assert!(
        !object_exists(&graph, "table", "op_checkpoints").await,
        "[round-trip DOWN] op_checkpoints must be absent after 015b downgrade"
    );
    assert!(
        !object_exists(&graph, "table", "dream_pass_budget_usage").await,
        "[round-trip DOWN] dream_pass_budget_usage must be absent after 015b downgrade"
    );

    // Verify columns are gone.
    assert!(
        !has_col(&graph, "entities", "is_dream_generated").await,
        "[round-trip DOWN] entities.is_dream_generated must be absent after 015b"
    );
    assert!(
        !has_col(&graph, "facts", "is_dream_generated").await,
        "[round-trip DOWN] facts.is_dream_generated must be absent after 015b"
    );

    // Data must be preserved through table-recreation.
    let entity_count_after_down = row_count(&graph, "entities").await;
    assert_eq!(
        entity_count_before, entity_count_after_down,
        "[round-trip DOWN] entity row count must be preserved through table-recreation: \
         before={entity_count_before}, after={entity_count_after_down}"
    );

    // Verify the specific entity row survived.
    let mut rows = graph
        .conn
        .query(
            "SELECT id FROM entities WHERE id = 'round-trip-entity-01'",
            (),
        )
        .await
        .expect("entity lookup after downgrade must succeed");
    let row = rows.next().await.expect("row iter must not error");
    assert!(
        row.is_some(),
        "[round-trip DOWN] inserted entity must survive 015b table-recreation"
    );

    // ── 016 re-apply ──────────────────────────────────────────────────────────
    kremory::core::migrations::migrate_016_crash_safety_schema(&graph.conn)
        .await
        .expect("016 re-apply after 015b must succeed");

    // Tables must be back.
    assert!(
        object_exists(&graph, "table", "dream_idempotency_keys").await,
        "[round-trip UP2] dream_idempotency_keys must exist after 016 re-apply"
    );
    assert!(
        object_exists(&graph, "table", "op_checkpoints").await,
        "[round-trip UP2] op_checkpoints must exist after 016 re-apply"
    );
    assert!(
        object_exists(&graph, "table", "dream_pass_budget_usage").await,
        "[round-trip UP2] dream_pass_budget_usage must exist after 016 re-apply"
    );

    // Columns must be back.
    assert!(
        has_col(&graph, "entities", "is_dream_generated").await,
        "[round-trip UP2] entities.is_dream_generated must be present after 016 re-apply"
    );
    assert!(
        has_col(&graph, "facts", "is_dream_generated").await,
        "[round-trip UP2] facts.is_dream_generated must be present after 016 re-apply"
    );

    // Data still intact.
    let entity_count_up2 = row_count(&graph, "entities").await;
    assert_eq!(
        entity_count_before, entity_count_up2,
        "[round-trip UP2] entity row count must be preserved after 016 re-apply: \
         before={entity_count_before}, up2={entity_count_up2}"
    );
}

// ─── Test 4: idempotency — second run_migrations is a no-op ──────────────────

/// Migration 016 must be idempotent: running `run_migrations` twice on the same
/// DB must not error. All three 016 invariants are verified:
///
/// 1. `CREATE TABLE IF NOT EXISTS` — no error on double-apply.
/// 2. `CREATE INDEX IF NOT EXISTS` — no error on double-apply.
/// 3. PRAGMA-guard — `ALTER TABLE ADD COLUMN` skipped when column already present;
///    no duplicate-column SQLite error.
///
/// Also verifies column count on entities and facts is unchanged after second run
/// (the PRAGMA-guard prevented any extra columns from being added).
#[tokio::test]
async fn migration_016_idempotent_when_already_applied() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Both columns must be present after first open.
    assert!(
        has_col(&graph, "entities", "is_dream_generated").await,
        "entities.is_dream_generated must be present before idempotency test"
    );
    assert!(
        has_col(&graph, "facts", "is_dream_generated").await,
        "facts.is_dream_generated must be present before idempotency test"
    );

    // Baseline column counts — must be unchanged after second run.
    let entity_col_count = column_names(&graph, "entities").await.len();
    let facts_col_count = column_names(&graph, "facts").await.len();

    // Second run — must not error.
    graph.run_migrations_again_for_test().await.expect(
        "second run_migrations on same DB must be idempotent — \
             any error means a PRAGMA-guard, IF NOT EXISTS, or table guard is missing",
    );

    // Column counts must be unchanged.
    assert_eq!(
        entity_col_count,
        column_names(&graph, "entities").await.len(),
        "entities column count must be unchanged after second migration run"
    );
    assert_eq!(
        facts_col_count,
        column_names(&graph, "facts").await.len(),
        "facts column count must be unchanged after second migration run"
    );

    // All 3 tables must still exist.
    assert!(
        object_exists(&graph, "table", "dream_idempotency_keys").await,
        "dream_idempotency_keys must still exist after second run_migrations"
    );
    assert!(
        object_exists(&graph, "table", "op_checkpoints").await,
        "op_checkpoints must still exist after second run_migrations"
    );
    assert!(
        object_exists(&graph, "table", "dream_pass_budget_usage").await,
        "dream_pass_budget_usage must still exist after second run_migrations"
    );

    // Both indexes must still exist.
    assert!(
        object_exists(&graph, "index", "idx_dream_idempotency_keys_entity").await,
        "idx_dream_idempotency_keys_entity must still exist after second run_migrations"
    );
    assert!(
        object_exists(&graph, "index", "idx_op_checkpoints_name_updated").await,
        "idx_op_checkpoints_name_updated must still exist after second run_migrations"
    );
}
