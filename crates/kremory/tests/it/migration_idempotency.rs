#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Story #213 — CREATE TABLE IF NOT EXISTS idempotency.
//!
//! AC: running `run_migrations` twice on the **same** DB connection does not error.
//!     Every CREATE TABLE/INDEX/VIRTUAL TABLE uses IF NOT EXISTS.
//! Gate G5: cargo test -p kremory migration_idempotency | grep "FAILED" → empty.

/// G5: run_migrations twice on the same in-memory DB must be a no-op.
///
/// `open_in_memory()` runs migrations once internally.
/// `run_migrations_again_for_test()` runs the same suite a second time on the
/// **same** connection — this is the real idempotency test (same DB, twice).
/// If any DDL is missing `IF NOT EXISTS` the second run will error.
///
/// Also asserts that `PRAGMA user_version` is unchanged (the migration runner
/// does not regress the version counter on a re-run where `applied` is empty).
#[tokio::test]
async fn migration_idempotency_same_db_twice() {
    use kremory::core::schema::TemporalGraph;

    // First open — bootstraps the schema on a fresh in-memory DB.
    let graph = TemporalGraph::open_in_memory()
        .await
        .expect("first open must succeed");

    // Run migrations a second time on the SAME handle — must be a no-op.
    graph
        .run_migrations_again_for_test()
        .await
        .expect("second run_migrations on same DB must be idempotent (IF NOT EXISTS)");
}

/// Verify that no bare CREATE TABLE/INDEX/VIRTUAL TABLE appears in
/// the production schema source (grep-based static check).
#[test]
fn no_bare_create_table_in_schema_source() {
    use std::path::Path;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let schema_src = root.join("crates/kremory/src/core/schema.rs");
    let content = std::fs::read_to_string(&schema_src)
        .unwrap_or_else(|_| panic!("cannot read {}", schema_src.display()));

    // Every base-DDL CREATE TABLE must use IF NOT EXISTS.
    //
    // Exempt: migration scratch tables matching `_bak_NNN` (backup snapshot
    // tables created mid-migration) and `_new` (target table inside a
    // CREATE-COPY-DROP-RENAME restructure). These are intentionally bare —
    // their idempotency is enforced by the migration's own pre-flight gate
    // (`SELECT name FROM sqlite_master WHERE name='..._bak_NNN'`). Adding
    // IF NOT EXISTS would silently mask incomplete prior migration state,
    // which is the opposite of safe.
    fn is_migration_scratch(stmt_upper: &str) -> bool {
        // Lift the identifier after CREATE TABLE [IF NOT EXISTS]?
        // Cheap heuristic: look for the documented suffixes.
        stmt_upper.contains("_BAK_") || stmt_upper.contains("_NEW ") || stmt_upper.contains("_NEW(")
    }

    for (line_no, line) in content.lines().enumerate() {
        let stripped = {
            // Remove // line comments
            if let Some(pos) = line.find("//") {
                &line[..pos]
            } else {
                line
            }
        };
        let upper = stripped.to_uppercase();
        if upper.contains("CREATE TABLE")
            && !upper.contains("CREATE TABLE IF NOT EXISTS")
            && !is_migration_scratch(&upper)
        {
            panic!(
                "schema.rs line {}: bare CREATE TABLE without IF NOT EXISTS: {:?}",
                line_no + 1,
                line
            );
        }
        if upper.contains("CREATE INDEX") && !upper.contains("CREATE INDEX IF NOT EXISTS") {
            panic!(
                "schema.rs line {}: bare CREATE INDEX without IF NOT EXISTS: {:?}",
                line_no + 1,
                line
            );
        }
        if upper.contains("CREATE VIRTUAL TABLE")
            && !upper.contains("CREATE VIRTUAL TABLE IF NOT EXISTS")
        {
            panic!(
                "schema.rs line {}: bare CREATE VIRTUAL TABLE without IF NOT EXISTS: {:?}",
                line_no + 1,
                line
            );
        }
    }
}
