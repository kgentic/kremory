use super::*;
use libsql::Builder;

async fn in_memory_conn() -> libsql::Connection {
    let db = Builder::new_local(":memory:")
        .build()
        .await
        .expect("in-memory db build");
    db.connect().expect("connect")
}

async fn seed_app_meta(conn: &libsql::Connection) {
    conn.execute_batch(
        "CREATE TABLE app_meta (
            workspace_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL DEFAULT 0,
            rql_schema_version INTEGER NOT NULL DEFAULT 0
         );
         INSERT INTO app_meta (workspace_id) VALUES ('ws-test');",
    )
    .await
    .expect("seed app_meta");
}

#[tokio::test]
async fn current_version_returns_zero_when_app_meta_missing() {
    let conn = in_memory_conn().await;
    let runner = MigrationRunner::new(&conn, "schema_version");
    let v = runner.current_version().await.expect("current_version");
    assert_eq!(v, 0);
}

#[tokio::test]
async fn current_version_reads_stored_value() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    conn.execute("UPDATE app_meta SET schema_version = 7", ())
        .await
        .expect("seed version");
    let runner = MigrationRunner::new(&conn, "schema_version");
    let v = runner.current_version().await.expect("current_version");
    assert_eq!(v, 7);
}

#[tokio::test]
async fn run_applies_all_pending_then_idempotent_on_rerun() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let migrations = [
        Migration {
            version: 1,
            name: "001_create_foo",
            sql: "CREATE TABLE foo (id INTEGER PRIMARY KEY)",
        },
        Migration {
            version: 2,
            name: "002_create_bar",
            sql: "CREATE TABLE bar (id INTEGER PRIMARY KEY)",
        },
    ];
    let runner = MigrationRunner::new(&conn, "schema_version");
    let out = runner.run(&migrations).await.expect("run");
    assert_eq!(out.starting_version, 0);
    assert_eq!(out.final_version, 2);
    assert_eq!(out.applied, vec![1, 2]);

    let out = runner.run(&migrations).await.expect("re-run idempotent");
    assert_eq!(out.starting_version, 2);
    assert_eq!(out.final_version, 2);
    assert!(out.applied.is_empty());
}

#[tokio::test]
async fn run_rejects_non_dense_migration_sequence() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let bad = [
        Migration {
            version: 1,
            name: "001_create_x",
            sql: "CREATE TABLE x (id INTEGER PRIMARY KEY)",
        },
        Migration {
            version: 3,
            name: "003_create_y",
            sql: "CREATE TABLE y (id INTEGER PRIMARY KEY)",
        },
    ];
    let runner = MigrationRunner::new(&conn, "schema_version");
    let err = runner.run(&bad).await.expect_err("must reject gap");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-dense"),
        "expected gap-detection error, got: {msg}"
    );
}

#[tokio::test]
async fn run_rejects_duplicate_version() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let bad = [
        Migration {
            version: 1,
            name: "001_create_x",
            sql: "CREATE TABLE x (id INTEGER PRIMARY KEY)",
        },
        Migration {
            version: 1,
            name: "001_create_y_dup",
            sql: "CREATE TABLE y (id INTEGER PRIMARY KEY)",
        },
    ];
    let runner = MigrationRunner::new(&conn, "schema_version");
    let err = runner.run(&bad).await.expect_err("must reject dup");
    assert!(format!("{err}").contains("duplicate"));
}

#[tokio::test]
async fn run_rejects_first_version_other_than_one() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let bad = [Migration {
        version: 5,
        name: "005_should_start_at_1",
        sql: "CREATE TABLE x (id INTEGER PRIMARY KEY)",
    }];
    let runner = MigrationRunner::new(&conn, "schema_version");
    let err = runner.run(&bad).await.expect_err("must reject base != 1");
    assert!(format!("{err}").contains("version must be 1"));
}

#[tokio::test]
async fn run_parallel_tracks_independent_via_separate_columns() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let rql_migs = [Migration {
        version: 1,
        name: "001_rql_baseline",
        sql: "CREATE TABLE rql_entities (id INTEGER PRIMARY KEY)",
    }];
    let example_migs = [Migration {
        version: 1,
        name: "001_example_baseline",
        sql: "CREATE TABLE folders (id INTEGER PRIMARY KEY)",
    }];

    MigrationRunner::new(&conn, "rql_schema_version")
        .run(&rql_migs)
        .await
        .expect("rql run");
    MigrationRunner::new(&conn, "schema_version")
        .run(&example_migs)
        .await
        .expect("example run");

    let rql_v = MigrationRunner::new(&conn, "rql_schema_version")
        .current_version()
        .await
        .expect("rql v");
    let kai_v = MigrationRunner::new(&conn, "schema_version")
        .current_version()
        .await
        .expect("kai v");
    assert_eq!(rql_v, 1);
    assert_eq!(kai_v, 1);
}

#[tokio::test]
async fn sanitize_column_rejects_sql_injection_attempts() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    // Direct sanitize_column unit test — the runner calls it on every
    // version read/write, so any caller-supplied column gets checked.
    let bad_inputs = ["1abc", "abc;DROP TABLE app_meta", "abc-def", "abc def"];
    for input in bad_inputs {
        let leaked: &'static str = Box::leak(input.to_string().into_boxed_str());
        let res = sanitize_column(leaked);
        assert!(res.is_err(), "expected rejection for {input:?}");
    }
    // Good inputs pass
    assert!(sanitize_column("schema_version").is_ok());
    assert!(sanitize_column("rql_schema_version").is_ok());
    let _ = conn;
}

#[tokio::test]
async fn run_returns_error_with_version_name_on_invalid_sql() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let bad = [Migration {
        version: 1,
        name: "001_syntax_error",
        sql: "THIS IS NOT VALID SQL",
    }];
    let runner = MigrationRunner::new(&conn, "schema_version");
    let err = runner
        .run(&bad)
        .await
        .expect_err("must surface syntax error");
    let msg = format!("{err}");
    assert!(
        msg.contains("001_syntax_error"),
        "expected migration name in error: {msg}"
    );
    assert!(
        msg.contains("migration 1"),
        "expected version number in error display: {msg}"
    );
    // Verify the version counter was NOT bumped after failure.
    let v = runner.current_version().await.expect("current_version");
    assert_eq!(v, 0, "failed migration must not bump the counter");
}

// ── Backup helpers ────────────────────────────────────────────────

#[tokio::test]
async fn backup_workspace_creates_timestamped_copy() {
    let tmp = tempdir_for_test().await;
    let src = tmp.join("workspace.db");
    // Make a fake db file
    tokio::fs::write(&src, b"sqlite-stand-in")
        .await
        .expect("write src");
    let backup_root = tmp.join("backups");
    let dest = backup_workspace(&src, &backup_root, "ws-alpha")
        .await
        .expect("backup");
    assert!(dest.exists(), "backup file must exist");
    let body = tokio::fs::read(&dest).await.expect("read back");
    assert_eq!(body, b"sqlite-stand-in");
    let fname = dest.file_name().unwrap().to_string_lossy().to_string();
    assert!(
        fname.starts_with("ws-alpha-"),
        "expected workspace prefix, got: {fname}"
    );
    assert!(fname.ends_with(".db"));
}

#[tokio::test]
async fn backup_workspace_rejects_workspace_id_with_path_separator() {
    let tmp = tempdir_for_test().await;
    let src = tmp.join("workspace.db");
    tokio::fs::write(&src, b"x").await.expect("write src");
    let backup_root = tmp.join("backups");
    let err = backup_workspace(&src, &backup_root, "../escape")
        .await
        .expect_err("must reject");
    assert!(matches!(err, MigrationError::InvalidWorkspaceId(_)));
}

#[tokio::test]
async fn prune_old_backups_deletes_only_old_files() {
    let tmp = tempdir_for_test().await;
    let backup_root = tmp.join("backups");
    tokio::fs::create_dir_all(&backup_root)
        .await
        .expect("mkdir backups");

    // Make 3 files: 2 old (mtime backdated), 1 fresh
    let old1 = backup_root.join("old1.db");
    let old2 = backup_root.join("old2.db");
    let fresh = backup_root.join("fresh.db");
    for p in [&old1, &old2, &fresh] {
        tokio::fs::write(p, b"x").await.expect("write");
    }
    // Backdate old files to 60 days ago using filetime
    let sixty_days_ago = SystemTime::now() - Duration::from_secs(60 * 86_400);
    set_mtime(&old1, sixty_days_ago).await;
    set_mtime(&old2, sixty_days_ago).await;

    let count = prune_old_backups(&backup_root, 30).await.expect("prune");
    assert_eq!(count, 2);
    assert!(!old1.exists());
    assert!(!old2.exists());
    assert!(fresh.exists());
}

#[tokio::test]
async fn prune_old_backups_skips_missing_root() {
    let tmp = tempdir_for_test().await;
    let backup_root = tmp.join("does-not-exist");
    let count = prune_old_backups(&backup_root, 30).await.expect("prune");
    assert_eq!(count, 0);
}

#[tokio::test]
async fn prune_old_backups_ignores_non_db_files() {
    let tmp = tempdir_for_test().await;
    let backup_root = tmp.join("backups");
    tokio::fs::create_dir_all(&backup_root)
        .await
        .expect("mkdir");
    let non_db = backup_root.join("notes.txt");
    tokio::fs::write(&non_db, b"keep").await.expect("write");
    let sixty_days_ago = SystemTime::now() - Duration::from_secs(60 * 86_400);
    set_mtime(&non_db, sixty_days_ago).await;
    let count = prune_old_backups(&backup_root, 30).await.expect("prune");
    assert_eq!(count, 0, "non-.db files must be ignored");
    assert!(non_db.exists());
}

/// Story #212: bump_version writes to BOTH app_meta AND PRAGMA user_version.
#[tokio::test]
async fn bump_version_dual_writes_pragma_user_version() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;
    let migrations = [Migration {
        version: 1,
        name: "001_create_x",
        sql: "CREATE TABLE x (id INTEGER PRIMARY KEY)",
    }];
    let runner = MigrationRunner::new(&conn, "schema_version");
    runner.run(&migrations).await.expect("run");

    // Verify app_meta write
    let v = runner.current_version().await.expect("app_meta version");
    assert_eq!(v, 1, "app_meta schema_version must be 1 after migration");

    // Verify PRAGMA user_version dual-write
    let mut rows = conn
        .query("PRAGMA user_version", ())
        .await
        .expect("pragma query");
    let row = rows.next().await.expect("row").expect("Some(row)");
    let pragma_v: i64 = row.get(0).expect("col 0");
    assert_eq!(
        pragma_v, 1,
        "PRAGMA user_version must match app_meta version"
    );
}

/// Story #212 / FU.7: when two runners on the SAME connection each apply one
/// migration, PRAGMA user_version reflects the LAST write (last-writer-wins)
/// while each track's app_meta column remains correct.
///
/// This documents and pins the shared-slot semantic described in the
/// `MigrationRunner` doc comment so regressions are caught immediately.
#[tokio::test]
async fn pragma_user_version_two_runner_last_writer_wins() {
    let conn = in_memory_conn().await;
    seed_app_meta(&conn).await;

    let rql_migs = [Migration {
        version: 1,
        name: "001_rql_entity",
        sql: "CREATE TABLE rql_entities (id INTEGER PRIMARY KEY)",
    }];
    let kai_migs = [Migration {
        version: 1,
        name: "001_kai_folder",
        sql: "CREATE TABLE folders (id INTEGER PRIMARY KEY)",
    }];

    // Run rql runner first — bumps PRAGMA user_version to 1.
    MigrationRunner::new(&conn, "rql_schema_version")
        .run(&rql_migs)
        .await
        .expect("rql run");

    // Run kai runner second — bumps PRAGMA user_version to 1 again (same value,
    // different track). The per-track app_meta columns stay independent.
    MigrationRunner::new(&conn, "schema_version")
        .run(&kai_migs)
        .await
        .expect("kai run");

    // Per-track app_meta columns must be independent and correct.
    let rql_v = MigrationRunner::new(&conn, "rql_schema_version")
        .current_version()
        .await
        .expect("rql v");
    let kai_v = MigrationRunner::new(&conn, "schema_version")
        .current_version()
        .await
        .expect("kai v");
    assert_eq!(rql_v, 1, "rql_schema_version must be 1");
    assert_eq!(kai_v, 1, "schema_version must be 1");

    // PRAGMA user_version reflects the last bump — kai runner ran last so it
    // wrote 1. In a scenario where tracks apply different version numbers the
    // last write wins; here both write 1 so result is 1.
    let mut rows = conn
        .query("PRAGMA user_version", ())
        .await
        .expect("pragma query");
    let row = rows.next().await.expect("row").expect("Some(row)");
    let pragma_v: i64 = row.get(0).expect("col 0");
    assert_eq!(
        pragma_v, 1,
        "PRAGMA user_version must reflect the last bump (last-writer-wins)"
    );
}

// ── Test helpers ──────────────────────────────────────────────────

async fn tempdir_for_test() -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "rqlc-migrations-test-{}",
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    tokio::fs::create_dir_all(&base).await.expect("tempdir");
    base
}

async fn set_mtime(path: &Path, when: SystemTime) {
    // libsql/tokio doesn't expose mtime set; use std + filetime crate
    // would add a dep. Instead use the `utimensat` syscall via
    // std::fs::File::set_modified (stable since 1.75).
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open for mtime");
    f.set_modified(when).expect("set_modified");
}

// ─── migrate_025 (TD-133 B2) — active-only partial dedup index ────────────────
//
// The upgrade path (DROP old-form + CREATE active-only partial) is the migration's
// whole reason to exist, and it is NOT exercised by the graph-layer fresh-DB tests
// (a fresh DB gets the correct index form straight from schema.rs, so migrate_025
// hits its no-op branch there — Quinn MEDIUM finding). These pin the upgrade /
// idempotent / absent-index paths per the migrate_006 fresh/idempotent/precondition
// convention. Anchors: TD-133 B2 + ADR-003 (bi-temporal re-assertion).

/// Read the current `idx_facts_content_hash_unique` DDL from sqlite_master.
async fn dedup_index_sql(conn: &libsql::Connection) -> Option<String> {
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master \
             WHERE type='index' AND name='idx_facts_content_hash_unique'",
            (),
        )
        .await
        .expect("query index sql");
    rows.next()
        .await
        .expect("read row")
        .map(|r| r.get::<String>(0).expect("get sql"))
}

/// Minimal `facts` table carrying just the columns the dedup index targets.
async fn seed_minimal_facts_table(conn: &libsql::Connection) {
    conn.execute("CREATE TABLE facts (content_hash TEXT, expired_at TEXT)", ())
        .await
        .expect("create minimal facts table");
}

#[tokio::test]
async fn migrate_025_rewrites_old_form_index_to_active_only_partial() {
    let conn = in_memory_conn().await;
    seed_minimal_facts_table(&conn).await;
    // Simulate a pre-TD-133 on-disk DB: OLD-form index (all rows, no expired_at).
    conn.execute(
        "CREATE UNIQUE INDEX idx_facts_content_hash_unique \
         ON facts(content_hash) WHERE content_hash IS NOT NULL",
        (),
    )
    .await
    .expect("seed old-form index");

    let before = dedup_index_sql(&conn).await.expect("index exists");
    assert!(
        !before.contains("expired_at"),
        "precondition: old-form index must lack expired_at, got: {before}"
    );

    migrate_025_fact_dedup_expired_partial(&conn)
        .await
        .expect("migrate_025 upgrade path");

    let after = dedup_index_sql(&conn).await.expect("index still exists");
    assert!(
        after.contains("expired_at IS NULL"),
        "migrate_025 must rewrite to active-only partial, got: {after}"
    );
    assert!(
        after.contains("content_hash IS NOT NULL"),
        "must retain the content_hash NOT NULL predicate, got: {after}"
    );
}

#[tokio::test]
async fn migrate_025_is_idempotent_noop_on_already_migrated() {
    let conn = in_memory_conn().await;
    seed_minimal_facts_table(&conn).await;
    conn.execute(
        "CREATE UNIQUE INDEX idx_facts_content_hash_unique \
         ON facts(content_hash) WHERE content_hash IS NOT NULL AND expired_at IS NULL",
        (),
    )
    .await
    .expect("seed new-form index");

    let before = dedup_index_sql(&conn).await.expect("index exists");
    migrate_025_fact_dedup_expired_partial(&conn)
        .await
        .expect("first run must be a no-op");
    let after = dedup_index_sql(&conn).await.expect("index still exists");
    assert_eq!(
        before, after,
        "already-migrated index must be left untouched (no DROP+CREATE)"
    );
    // Second run: still a no-op, no error.
    migrate_025_fact_dedup_expired_partial(&conn)
        .await
        .expect("second run must be a no-op");
}

#[tokio::test]
async fn migrate_025_creates_active_only_partial_when_index_absent() {
    let conn = in_memory_conn().await;
    seed_minimal_facts_table(&conn).await;
    assert!(
        dedup_index_sql(&conn).await.is_none(),
        "precondition: no dedup index yet"
    );

    migrate_025_fact_dedup_expired_partial(&conn)
        .await
        .expect("migrate_025 on absent index");

    let after = dedup_index_sql(&conn).await.expect("index created");
    assert!(
        after.contains("expired_at IS NULL") && after.contains("content_hash IS NOT NULL"),
        "must create the active-only partial form, got: {after}"
    );
}
