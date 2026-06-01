#![allow(clippy::unwrap_used, clippy::expect_used)]
//! G1 Day 1 — Migration 007 schema tests (Red phase).
//!
//! AC.1  — `source_id` + `source_uri` columns present on `episodes` post-migration.
//! AC.13 — `idx_episodes_source_id` index exists; FK-check clean; migration is
//!          idempotent (double-apply without error); static PRAGMA gate present in
//!          migration source.
//!
//! All five tests MUST FAIL until Green wires `migrate_007_source_id_source_uri`
//! in `crates/kremory/src/core/migrations.rs` and calls it from
//! `TemporalGraph::run_migrations`.

use kremory::core::schema::TemporalGraph;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Open a file-backed `TemporalGraph` in an isolated temp directory.
///
/// `TempDir` is returned so the caller owns the lifetime — the directory (and
/// the DB file inside it) is deleted when the returned value is dropped.
async fn open_file_backed_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("kremory-mig-007.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed on fresh file-backed DB");
    (graph, tmp)
}

/// Collect all column names from `PRAGMA table_info('<table>')`.
/// Returns a `Vec<(name, not_null_flag)>` where `not_null` is the integer
/// SQLite stores (0 = nullable, 1 = NOT NULL).
async fn episode_columns(graph: &TemporalGraph) -> Vec<(String, i64)> {
    let mut rows = graph
        .conn
        .query("PRAGMA table_info('episodes')", ())
        .await
        .expect("PRAGMA table_info('episodes') must succeed");

    let mut cols: Vec<(String, i64)> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let name: String = row.get(1).expect("column name at index 1");
        let not_null: i64 = row.get(3).expect("not_null flag at index 3");
        cols.push((name, not_null));
    }
    cols
}

// ─── AC.1: columns present post-apply ─────────────────────────────────────────

/// Migration 007 must add `source_id TEXT NULL` and `source_uri TEXT NULL` to
/// the `episodes` table and must preserve all pre-G1 columns.
///
/// Fails until `migrate_007_source_id_source_uri` is wired in `run_migrations`.
#[tokio::test]
async fn migration_007_columns_present_post_apply() {
    let (graph, _tmp) = open_file_backed_graph().await;
    let cols = episode_columns(&graph).await;

    let col_names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();

    // ── new columns (AC.1) ───────────────────────────────────────────────────
    assert!(
        col_names.contains(&"source_id"),
        "episodes must have column `source_id` after migration 007; found: {col_names:?}"
    );
    assert!(
        col_names.contains(&"source_uri"),
        "episodes must have column `source_uri` after migration 007; found: {col_names:?}"
    );

    // source_id must be nullable (not-null = 0)
    let source_id_notnull = cols
        .iter()
        .find(|(n, _)| n == "source_id")
        .map(|(_, nn)| *nn)
        .expect("source_id column must exist");
    assert_eq!(
        source_id_notnull, 0,
        "source_id must be TEXT NULL (not_null=0); got not_null={source_id_notnull}"
    );

    // source_uri must be nullable (not-null = 0)
    let source_uri_notnull = cols
        .iter()
        .find(|(n, _)| n == "source_uri")
        .map(|(_, nn)| *nn)
        .expect("source_uri column must exist");
    assert_eq!(
        source_uri_notnull, 0,
        "source_uri must be TEXT NULL (not_null=0); got not_null={source_uri_notnull}"
    );

    // ── pre-G1 columns must still be present (AC.1 non-regression) ───────────
    //
    // Verified against actual SQL DDL at `crates/kremory/src/core/schema.rs::428`
    // (CREATE TABLE episodes). Note: `content_hash` exists on the Rust Episode
    // struct (`memory/types.rs`) as a runtime-only field — NOT a SQL column.
    // Migration 007 does not add or touch content_hash.
    let required_pre_g1 = [
        "id",
        "content",
        "timestamp",
        "source_type",
        "metadata",
        "group_id",
        "saga_id",
        "sequence_number",
    ];
    for col in required_pre_g1 {
        assert!(
            col_names.contains(&col),
            "pre-G1 column `{col}` must still be present after migration 007; found: {col_names:?}"
        );
    }
}

// ─── AC.1 + AC.13: idempotency ────────────────────────────────────────────────

/// Running `run_migrations` twice on the same DB must be a no-op.
///
/// Specifically for migration 007: the `ALTER TABLE ADD COLUMN` statements must
/// not fail with "duplicate column" on the second run — the idempotency gate
/// (`PRAGMA table_info` check or `IF NOT EXISTS` equivalent) must absorb the
/// repeat call without error. Column count must also be unchanged.
///
/// Fails until migration 007 is wired with a proper idempotency gate.
#[tokio::test]
async fn migration_007_idempotent_double_apply() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Capture baseline column count after first migration run (inside open).
    let cols_first = episode_columns(&graph).await;
    let count_first = cols_first.len();

    // Second run — must not error.
    graph.run_migrations_again_for_test().await.expect(
        "second run_migrations on same DB must be idempotent for migration 007 — \
                  duplicate-column error means the idempotency gate is missing",
    );

    // Column count must be identical (no phantom duplicates, no drops).
    let cols_second = episode_columns(&graph).await;
    let count_second = cols_second.len();
    assert_eq!(
        count_first, count_second,
        "episode column count must be unchanged after second migration run: \
         first={count_first}, second={count_second}"
    );

    // Both new columns must still be present.
    let names_second: Vec<&str> = cols_second.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names_second.contains(&"source_id"),
        "source_id must still be present after second migration run; columns: {names_second:?}"
    );
    assert!(
        names_second.contains(&"source_uri"),
        "source_uri must still be present after second migration run; columns: {names_second:?}"
    );
}

// ─── AC.13: FK check clean post-migration ─────────────────────────────────────

/// `PRAGMA foreign_key_check` must return an empty result set after migration 007
/// runs. An additive `ALTER TABLE ADD COLUMN` should never introduce FK violations,
/// but the check is mandated by AC.13 as a belt-and-suspenders integrity gate.
///
/// Fails until migration 007 is wired (the test itself passes once columns exist,
/// but the companion `migration_007_columns_present_post_apply` failure ensures
/// the overall suite is Red until Green ships).
#[tokio::test]
async fn migration_007_foreign_key_check_clean_post_apply() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Verify migration 007 columns are present (ensures we're testing post-007 state).
    let cols = episode_columns(&graph).await;
    let col_names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        col_names.contains(&"source_id"),
        "pre-condition: source_id must exist — migration 007 must have run; \
         columns: {col_names:?}"
    );

    // Run FK integrity check.
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
        "PRAGMA foreign_key_check must return empty result set after migration 007 — \
         FK violations found; inspect backup tables for recovery"
    );
}

// ─── AC.13: static PRAGMA gate present in migration source ────────────────────

/// Static source-code grep: the migration source file must contain the
/// `PRAGMA table_info('episodes')` guard (proving the idempotency gate is coded
/// before the `ALTER TABLE ADD COLUMN` statements) and the `idx_episodes_source_id`
/// index DDL.
///
/// This test does NOT require the migration to run — it validates the static
/// source text. It fails if:
///   (a) the migration function does not exist yet (substring absent → FAIL), OR
///   (b) Green adds the column without the PRAGMA gate (FAIL), OR
///   (c) Green omits the index DDL (FAIL).
#[test]
fn migration_007_pragma_gates_present_in_source() {
    use std::path::Path;

    // Resolve migrations.rs relative to the crate manifest directory.
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let migrations_src = manifest_dir.join("src/core/migrations.rs");

    let content = std::fs::read_to_string(&migrations_src)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", migrations_src.display()));

    // Gate 1: PRAGMA table_info idempotency guard must be coded.
    assert!(
        content.contains("PRAGMA table_info('episodes')"),
        "migrations.rs must contain `PRAGMA table_info('episodes')` as the migration 007 \
         idempotency gate — this gate prevents duplicate-column errors on re-run; \
         found in: {}",
        migrations_src.display()
    );

    // Gate 2: index DDL for `idx_episodes_source_id` must be coded.
    assert!(
        content.contains("idx_episodes_source_id"),
        "migrations.rs must contain `idx_episodes_source_id` (the CREATE INDEX DDL for \
         the new source_id column) — found in: {}",
        migrations_src.display()
    );
}

// ─── AC.1: index exists post-migration ───────────────────────────────────────

/// `PRAGMA index_list('episodes')` must include `idx_episodes_source_id` after
/// migration 007 runs.
///
/// Fails until `migrate_007_source_id_source_uri` creates the index and is wired
/// in `run_migrations`.
#[tokio::test]
async fn migration_007_index_exists() {
    let (graph, _tmp) = open_file_backed_graph().await;

    // Verify migration 007 columns are present first (ensures correct state).
    let cols = episode_columns(&graph).await;
    let col_names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        col_names.contains(&"source_id"),
        "pre-condition: source_id must exist — migration 007 must have run; \
         columns: {col_names:?}"
    );

    // Collect all index names on the `episodes` table.
    let mut rows = graph
        .conn
        .query("PRAGMA index_list('episodes')", ())
        .await
        .expect("PRAGMA index_list('episodes') must succeed");

    let mut index_names: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        // index_list columns: seq(0), name(1), unique(2), origin(3), partial(4)
        let name: String = row.get(1).expect("index name at column 1");
        index_names.push(name);
    }

    assert!(
        index_names.iter().any(|n| n == "idx_episodes_source_id"),
        "idx_episodes_source_id must exist in PRAGMA index_list('episodes') after migration 007; \
         found indexes: {index_names:?}"
    );
}
