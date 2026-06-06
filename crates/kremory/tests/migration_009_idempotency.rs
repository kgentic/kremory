#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 009 schema tests — DROP entities.label column.
//!
//! ## Acceptance criteria
//!
//! AC.a — After migration 009 runs, `entities` table must NOT have a `label` column.
//! AC.b — `entity_type_id` column is RETAINED after migration 009.
//! AC.c — FTS5 `entities_fts` virtual table still exists and is usable after migration.
//! AC.d — Re-run is a no-op (idempotency): calling `run_migrations` twice must not error
//!         and must not alter entities column shape.
//! AC.e — PRAGMA foreign_key_check returns empty after migration.
//! AC.f — Static source-code gate: migrations.rs contains the PRAGMA gate and
//!         `ALTER TABLE entities DROP COLUMN label` strings for migration 009.

use kremory::core::schema::TemporalGraph;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Open a file-backed `TemporalGraph` in an isolated temp directory.
/// Migration 009 runs automatically at open() time (after 008).
async fn open_file_backed_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("kremory-mig-009.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed on fresh file-backed DB");
    (graph, tmp)
}

/// Collect column names from `PRAGMA table_info('<table>')`.
async fn table_columns(graph: &TemporalGraph, table: &str) -> Vec<String> {
    let sql = format!("PRAGMA table_info('{table}')");
    let mut rows = graph
        .conn
        .query(&sql, ())
        .await
        .unwrap_or_else(|_| panic!("PRAGMA table_info('{table}') must succeed"));

    let mut cols = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .expect("PRAGMA table_info row iteration must not error")
    {
        let name: String = row.get(1).expect("column name at index 1");
        cols.push(name);
    }
    cols
}

// ─── AC.a: label column absent after migration 009 ───────────────────────────

/// After migration 009 the `entities` table must NOT have a `label` column.
/// Label is now resolved at query time via LEFT JOIN on entity_types.
#[tokio::test]
async fn migration_009_label_column_dropped() {
    let (graph, _tmp) = open_file_backed_graph().await;
    let cols = table_columns(&graph, "entities").await;

    assert!(
        !cols.iter().any(|c| c == "label"),
        "entities must NOT have `label` column after migration 009 \
         (label is resolved at query time via JOIN entity_types); found columns: {cols:?}"
    );
}

// ─── AC.b: entity_type_id retained ───────────────────────────────────────────

/// `entity_type_id` column must be present in `entities` after migration 009.
#[tokio::test]
async fn migration_009_entity_type_id_retained() {
    let (graph, _tmp) = open_file_backed_graph().await;
    let cols = table_columns(&graph, "entities").await;

    assert!(
        cols.iter().any(|c| c == "entity_type_id"),
        "entities must RETAIN `entity_type_id` column after migration 009; found: {cols:?}"
    );
}

// ─── AC.c: entities_fts virtual table still usable ───────────────────────────

/// The FTS5 `entities_fts` virtual table must still exist and accept queries
/// after migration 009. After the DROP, new FTS inserts use empty string for
/// the `label` column; the FTS index itself is not destroyed.
#[tokio::test]
async fn migration_009_fts_still_usable() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Insert an entity using the new shape (entity_type_id, no label).
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities (id, entity_type_id, properties, recorded_at, group_id) \
             VALUES ('test-fts-ent', 0, '{}', datetime('now'), 'default')",
            (),
        )
        .await
        .expect("entity insert must succeed post-migration-009");

    // Insert into FTS (label column uses empty string after DROP).
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities_fts(entity_id, label, properties) VALUES ('test-fts-ent', '', '{}')",
            (),
        )
        .await
        .expect("entities_fts insert with empty label must succeed after migration 009");

    // FTS query must return a result.
    // Use a non-MATCH scan to confirm the virtual table is readable; a bare '*'
    // is not valid FTS5 syntax and the table may be empty (properties='{}').
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id FROM entities_fts",
            (),
        )
        .await
        .expect("entities_fts table-scan must succeed after migration 009");

    // Must be able to iterate without error.
    let first = rows
        .next()
        .await
        .expect("entities_fts row iteration must not error");

    assert!(
        first.is_some(),
        "entities_fts FTS query must return at least one row after inserting test entity"
    );
}

// ─── AC.d: idempotency — double-apply ────────────────────────────────────────

/// Running `run_migrations` twice must be a no-op: no errors and the `label`
/// column must remain absent after the second run.
#[tokio::test]
async fn migration_009_idempotent_double_apply() {
    let (graph, _tmp) = open_file_backed_graph().await;

    let cols_first = table_columns(&graph, "entities").await;
    assert!(
        !cols_first.iter().any(|c| c == "label"),
        "pre-condition: label must be absent after first open(); found: {cols_first:?}"
    );

    // Second run via the test hook — must not error.
    graph
        .run_migrations_again_for_test()
        .await
        .expect(
            "second run_migrations must be idempotent for migration 009 — \
             PRAGMA gate must detect label column already absent and skip DROP",
        );

    // Column shape unchanged.
    let cols_second = table_columns(&graph, "entities").await;
    assert!(
        !cols_second.iter().any(|c| c == "label"),
        "entities must still NOT have `label` column after second migration run; \
         found: {cols_second:?}"
    );
    assert!(
        cols_second.iter().any(|c| c == "entity_type_id"),
        "entity_type_id must still be present after second migration run; \
         found: {cols_second:?}"
    );
}

// ─── AC.e: PRAGMA foreign_key_check clean ────────────────────────────────────

/// `PRAGMA foreign_key_check` must return empty after migration 009.
#[tokio::test]
async fn migration_009_foreign_key_check_clean() {
    let (graph, _tmp) = open_file_backed_graph().await;

    let cols = table_columns(&graph, "entities").await;
    assert!(
        cols.iter().any(|c| c == "entity_type_id"),
        "pre-condition: entity_type_id must exist — migrations 008+009 must have run"
    );

    let mut violations = graph
        .conn
        .query("PRAGMA foreign_key_check", ())
        .await
        .expect("PRAGMA foreign_key_check must execute without error");

    let first_violation = violations
        .next()
        .await
        .expect("row iteration must not error");

    assert!(
        first_violation.is_none(),
        "PRAGMA foreign_key_check must return empty result set after migration 009"
    );
}

// ─── AC.f: static source-code gate ───────────────────────────────────────────

/// migrations.rs must statically contain the migration 009 DDL and guard strings.
#[test]
fn migration_009_source_gates_present() {
    use std::path::Path;

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let migrations_src = manifest_dir.join("src/core/migrations.rs");

    let content = std::fs::read_to_string(&migrations_src)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", migrations_src.display()));

    assert!(
        content.contains("migrate_009_drop_label_column"),
        "migrations.rs must define `migrate_009_drop_label_column` function; \
         found in: {}",
        migrations_src.display()
    );

    assert!(
        content.contains("ALTER TABLE entities DROP COLUMN label"),
        "migrations.rs must contain `ALTER TABLE entities DROP COLUMN label` DDL; \
         found in: {}",
        migrations_src.display()
    );

    assert!(
        content.contains("entities_fts") && content.contains("rebuild"),
        "migrations.rs must contain FTS5 rebuild after DROP COLUMN label; \
         found in: {}",
        migrations_src.display()
    );
}
