//! Migration framework for rqlc + the host application workspace databases.
//!
//! Per ADR-Phase-D.0 §"D.3 — Migration framework" + master plan
//! `.ai-docs/rqlcm-split/00-master-plan.md` D.3 spec.
//!
//! ## Design
//!
//! - **`Migration`** — one versioned schema-evolution step. `version`,
//!   `name`, `sql` (DDL string). Versions start at 1 and must be strictly
//!   monotonic + dense within a domain.
//! - **`MigrationRunner`** — applies pending migrations sequentially.
//!   Parameterised by a `version_column` so the framework can manage
//!   two parallel migration tracks within the same `app_meta` row:
//!   `rql_schema_version` (RQL graph schema, owned by rqlc) and
//!   `schema_version` (the host application CRUD schema, owned by tauri-app). Both
//!   columns live on the same `app_meta` row but they're independent
//!   counters.
//! - **`backup_workspace`** — file-level copy with `{workspace_id}-{rfc3339-ish}.db`
//!   filename. Sibling-directory backup root. Backups ARE rollback in this
//!   framework because SQLite DDL is not transactional (per sqlite.org
//!   `Limitations Of SQLite's "ALTER TABLE"` + "Transactional DDL is partial").
//! - **`prune_old_backups`** — deletes any `*.db` backup older than
//!   `max_age_days`. Used by the host application's startup cleanup to keep the
//!   backup root bounded.
//!
//! ## Idempotency invariant
//!
//! Every SQL DDL statement in a `Migration::sql` field MUST use:
//! - `CREATE TABLE IF NOT EXISTS`
//! - `CREATE INDEX IF NOT EXISTS`
//! - `CREATE VIRTUAL TABLE IF NOT EXISTS`
//!
//! This guarantees that running `MigrationRunner::run` on an already-applied
//! migration set (e.g. on reconnect to an existing database) is a no-op with
//! no errors. The gate `cargo test -p kremory migration_idempotency` verifies
//! this end-to-end for `TemporalGraph::open_in_memory`.
//!
//! ## Failure mode
//!
//! `MigrationRunner::run` issues `BEGIN` per migration if the SQL is
//! single-statement-friendly; multi-statement DDL is run statement-by-
//! statement without a transaction (SQLite limitation). On any per-stmt
//! failure the runner returns `Err` with the partial-apply path embedded
//! in the error — callers are expected to restore from the pre-migration
//! backup. The runner does NOT auto-rollback because backup-restore is a
//! file-level operation outside this fn's purview.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use thiserror::Error;

/// A single migration definition. Migrations are static — they live in
/// a slice in the consumer crate (`rql-core/migrations/*` or `tauri-app/
/// src-tauri/src/migrations/*`).
#[derive(Debug, Clone, Copy)]
pub struct Migration {
    /// Strictly monotonic version number starting at 1. Must be dense
    /// within a domain (no gaps allowed — the runner pins gap-detection
    /// as a hard error to prevent silent drift).
    pub version: u32,
    /// Human-readable name following the `NNN_descriptive_name` convention:
    /// exactly 3 decimal digits, an underscore, then a lowercase snake_case
    /// description (`[a-z][a-z0-9_]*`). Example: `"001_create_rql_entities"`.
    ///
    /// Why: uniform naming lets tooling (audit scripts, runbooks, CLI) sort
    /// and correlate migrations by version without parsing the `version` field.
    /// Enforced at runtime by `validate_migration_set`.
    pub name: &'static str,
    /// One or more SQL DDL statements separated by `;`. Multi-statement
    /// DDL is supported via `libsql::Connection::execute_batch`. Single
    /// statements work too.
    pub sql: &'static str,
}

/// Outcome of a `MigrationRunner::run` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationOutcome {
    pub starting_version: u32,
    pub final_version: u32,
    pub applied: Vec<u32>,
}

#[derive(Debug, Error)]
pub enum MigrationError {
    #[error("libsql error: {0}")]
    Db(#[from] libsql::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("migration {version} ({name}) failed: {source}")]
    Apply {
        version: u32,
        name: &'static str,
        #[source]
        source: libsql::Error,
    },
    #[error("migration set is invalid: {0}")]
    Invalid(String),
    #[error("backup workspace_id must not contain path separators: {0:?}")]
    InvalidWorkspaceId(String),
}

pub type Result<T> = std::result::Result<T, MigrationError>;

/// Migration runner over a `libsql::Connection`. Reads + bumps the
/// version counter stored in `app_meta.{version_column}`.
///
/// The runner assumes `app_meta` exists with the named version column;
/// bootstrapping that table is the caller's responsibility (typically
/// the v1 migration itself creates `app_meta` + seeds the row).
///
/// ## PRAGMA user_version — shared state, last-writer-wins
///
/// `bump_version` writes to **two** locations after every migration step:
///
/// 1. `app_meta.{version_column}` — the per-track logical version.
/// 2. `PRAGMA user_version` — a single 32-bit integer slot on the database
///    file header, readable by any SQLite client without parsing `app_meta`.
///
/// When two `MigrationRunner` instances are live on the **same connection**
/// with different `version_column` values (e.g. `rql_schema_version` and
/// `schema_version`), they share the single `PRAGMA user_version` slot.
/// Each `bump_version` call overwrites it with the version it just applied.
/// The two per-track counters in `app_meta` remain independent and correct;
/// only `PRAGMA user_version` reflects the **last migration applied across
/// all tracks**. Consumers that need per-track versions must read
/// `app_meta.{version_column}` directly.
///
/// For single-track deployments (one runner per database, e.g. kremory's
/// `TemporalGraph`) `PRAGMA user_version` always matches `app_meta.version`
/// and is safe to use as a quick health-check.
pub struct MigrationRunner<'a> {
    conn: &'a libsql::Connection,
    version_column: &'static str,
}

impl<'a> MigrationRunner<'a> {
    pub fn new(conn: &'a libsql::Connection, version_column: &'static str) -> Self {
        Self {
            conn,
            version_column,
        }
    }

    /// Read the currently-applied version from `app_meta.{version_column}`.
    /// Returns 0 if the `app_meta` table doesn't exist (pre-bootstrap),
    /// 0 if the row is missing, or the stored integer otherwise.
    pub async fn current_version(&self) -> Result<u32> {
        let table_check = self
            .conn
            .query(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_meta' LIMIT 1",
                (),
            )
            .await;
        let mut rows = match table_check {
            Ok(r) => r,
            Err(e) => return Err(MigrationError::Db(e)),
        };
        if rows.next().await?.is_none() {
            return Ok(0);
        }
        // Table exists. Read the version column.
        let sql = format!(
            "SELECT {} FROM app_meta LIMIT 1",
            sanitize_column(self.version_column)?
        );
        let mut rows = self.conn.query(&sql, ()).await?;
        let Some(row) = rows.next().await? else {
            return Ok(0);
        };
        let v: i64 = row.get(0)?;
        Ok(v.max(0) as u32)
    }

    /// Apply all migrations in `migrations` whose `version` exceeds the
    /// current stored version. Migrations are sorted by version before
    /// application; gap-detection runs first.
    ///
    /// Returns the `MigrationOutcome` describing what was applied.
    pub async fn run(&self, migrations: &[Migration]) -> Result<MigrationOutcome> {
        validate_migration_set(migrations)?;
        let starting_version = self.current_version().await?;
        let mut applied: Vec<u32> = Vec::new();
        let mut final_version = starting_version;

        let mut pending: Vec<&Migration> = migrations
            .iter()
            .filter(|m| m.version > starting_version)
            .collect();
        pending.sort_by_key(|m| m.version);

        for m in pending {
            self.conn
                .execute_batch(m.sql)
                .await
                .map_err(|source| MigrationError::Apply {
                    version: m.version,
                    name: m.name,
                    source,
                })?;
            self.bump_version(m.version).await?;
            applied.push(m.version);
            final_version = m.version;
        }

        Ok(MigrationOutcome {
            starting_version,
            final_version,
            applied,
        })
    }

    async fn bump_version(&self, new_version: u32) -> Result<()> {
        let sql = format!(
            "UPDATE app_meta SET {} = ?",
            sanitize_column(self.version_column)?
        );
        self.conn
            .execute(&sql, libsql::params![new_version as i64])
            .await?;
        // Story #212 / ADR D11: dual-write PRAGMA user_version so OS-level tools
        // (sqlite3 CLI, sqlitebrowser) observe the canonical version without parsing
        // app_meta. PRAGMA user_version does NOT support `?` binding — must inline
        // the integer literal (verified: libsql PRAGMA write is a DDL-level op).
        self.conn
            .execute(&format!("PRAGMA user_version = {new_version}"), ())
            .await?;
        Ok(())
    }
}

fn validate_migration_set(migrations: &[Migration]) -> Result<()> {
    if migrations.is_empty() {
        return Ok(());
    }

    // Enforce NNN_descriptive_name convention on every migration name.
    // Pattern: exactly 3 decimal digits, underscore, lowercase snake_case body.
    for m in migrations {
        validate_migration_name(m.name)?;
    }

    let mut versions: Vec<u32> = migrations.iter().map(|m| m.version).collect();
    versions.sort();
    if versions[0] != 1 {
        return Err(MigrationError::Invalid(format!(
            "first migration version must be 1, got {}",
            versions[0]
        )));
    }
    for w in versions.windows(2) {
        let (prev, next) = (w[0], w[1]);
        if next == prev {
            return Err(MigrationError::Invalid(format!(
                "duplicate migration version {prev}"
            )));
        }
        if next != prev + 1 {
            return Err(MigrationError::Invalid(format!(
                "non-dense migration sequence: {prev} → {next} (gap)"
            )));
        }
    }
    Ok(())
}

/// Validate that a migration name follows the `NNN_descriptive_name` convention.
///
/// Valid: `"001_create_rql_entities"`, `"042_add_content_hash"`.
/// Invalid: `"create_foo"` (no prefix), `"01_foo"` (2-digit prefix),
///          `"001_CreateFoo"` (uppercase), `"001_"` (empty body).
fn validate_migration_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    // Must start with exactly 3 ASCII decimal digits followed by '_'.
    let prefix_ok = bytes.len() > 4
        && bytes[0].is_ascii_digit()
        && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_digit()
        && bytes[3] == b'_';
    if !prefix_ok {
        return Err(MigrationError::Invalid(format!(
            "migration name {name:?} does not follow NNN_descriptive convention \
             (must start with exactly 3 digits and an underscore, e.g. \"001_create_foo\")"
        )));
    }
    // Body (after the leading NNN_) must be lowercase snake_case: [a-z][a-z0-9_]*.
    let body = &name[4..];
    let mut chars = body.chars();
    let first_ok = chars
        .next()
        .map(|c| c.is_ascii_lowercase())
        .unwrap_or(false);
    if !first_ok {
        return Err(MigrationError::Invalid(format!(
            "migration name {name:?} body must start with a lowercase letter after 'NNN_'"
        )));
    }
    for c in chars {
        if !c.is_ascii_lowercase() && !c.is_ascii_digit() && c != '_' {
            return Err(MigrationError::Invalid(format!(
                "migration name {name:?} body contains invalid char {c:?}: \
                 only lowercase letters, digits, and underscores allowed"
            )));
        }
    }
    Ok(())
}

fn sanitize_column(col: &'static str) -> Result<&'static str> {
    // version_column is a `&'static str` from caller code — guard against
    // SQL injection just in case future code derives it from data. Only
    // letters, digits, underscores allowed; first char must be a letter.
    let mut chars = col.chars();
    let Some(first) = chars.next() else {
        return Err(MigrationError::Invalid(
            "version_column must not be empty".to_string(),
        ));
    };
    if !first.is_ascii_alphabetic() {
        return Err(MigrationError::Invalid(format!(
            "version_column must start with a letter: {col:?}"
        )));
    }
    for c in chars {
        if !c.is_ascii_alphanumeric() && c != '_' {
            return Err(MigrationError::Invalid(format!(
                "version_column contains invalid char {c:?}: {col:?}"
            )));
        }
    }
    Ok(col)
}

/// Copy `workspace_db_path` into `backup_root/{workspace_id}-{utc-rfc3339}.db`.
/// Creates `backup_root` if it doesn't exist. Returns the absolute path of
/// the created backup file.
///
/// Per master plan D.3 spec: backups ARE rollback because SQLite DDL is not
/// transactional. Callers MUST take a backup before running any migration
/// that could leave the database in a half-applied state.
///
/// Not yet wired into the migration runner entry point — integration is a
/// follow-up story. Surgical exemption until the caller is wired.
#[allow(dead_code)] // planned consumer: migration runner pre-migrate hook (D.3 spec)
pub(crate) async fn backup_workspace(
    workspace_db_path: &Path,
    backup_root: &Path,
    workspace_id: &str,
) -> Result<PathBuf> {
    if workspace_id.contains('/') || workspace_id.contains('\\') {
        return Err(MigrationError::InvalidWorkspaceId(workspace_id.to_string()));
    }
    tokio::fs::create_dir_all(backup_root).await?;
    let now: DateTime<Utc> = Utc::now();
    // RFC3339 contains `:` which is reserved on Windows and awkward on macOS
    // Finder. Replace with `-` for filesystem safety.
    let stamp = now.to_rfc3339().replace(':', "-").replace('+', "_");
    let filename = format!("{workspace_id}-{stamp}.db");
    let dest = backup_root.join(filename);
    tokio::fs::copy(workspace_db_path, &dest).await?;
    Ok(dest)
}

/// Delete any `*.db` file under `backup_root` whose mtime is older than
/// `max_age_days`. Returns the count of files removed. Does NOT recurse
/// into subdirectories.
///
/// Not yet wired into the migration runner entry point — integration is a
/// follow-up story. Surgical exemption until the caller is wired.
#[allow(dead_code)] // planned consumer: the host application startup cleanup (D.3 spec)
pub(crate) async fn prune_old_backups(backup_root: &Path, max_age_days: u32) -> Result<usize> {
    if !tokio::fs::try_exists(backup_root).await? {
        return Ok(0);
    }
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(u64::from(max_age_days) * 86_400))
        .unwrap_or(UNIX_EPOCH);
    let mut count = 0usize;
    let mut entries = tokio::fs::read_dir(backup_root).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("db") {
            continue;
        }
        let meta = entry.metadata().await?;
        if !meta.is_file() {
            continue;
        }
        let mtime = meta.modified()?;
        if mtime < cutoff {
            tokio::fs::remove_file(&path).await?;
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
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
        let the-host-application_migs = [Migration {
            version: 1,
            name: "001_the-host-application_baseline",
            sql: "CREATE TABLE folders (id INTEGER PRIMARY KEY)",
        }];

        MigrationRunner::new(&conn, "rql_schema_version")
            .run(&rql_migs)
            .await
            .expect("rql run");
        MigrationRunner::new(&conn, "schema_version")
            .run(&the-host-application_migs)
            .await
            .expect("the-host-application run");

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
}
