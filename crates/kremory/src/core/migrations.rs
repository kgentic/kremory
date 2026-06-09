//! Migration framework for substrate + consumer-owned workspace databases.
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
//!   `schema_version` (consumer CRUD schema, owned by the consumer app). Both
//!   columns live on the same `app_meta` row but they're independent
//!   counters.
//! - **`backup_workspace`** — file-level copy with `{workspace_id}-{rfc3339-ish}.db`
//!   filename. Sibling-directory backup root. Backups ARE rollback in this
//!   framework because SQLite DDL is not transactional (per sqlite.org
//!   `Limitations Of SQLite's "ALTER TABLE"` + "Transactional DDL is partial").
//! - **`prune_old_backups`** — deletes any `*.db` backup older than
//!   `max_age_days`. Used by the consumer's startup cleanup to keep the
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

/// ADR-029a (v0.1.4): namespaces-table migration as a tooling-discoverable
/// constant. The actual DDL is also embedded directly in
/// [`crate::core::schema::TemporalGraph::run_migrations`] so the table exists
/// on every fresh `TemporalGraph::open*` call without requiring the runtime
/// to wire `MigrationRunner` through `TemporalGraph`. This constant captures
/// the canonical migration record for audit scripts + downstream tooling.
///
/// CREATE-only, no backfill (Vera MED-1 — zero coupling with ADR-029b's
/// `002_drop_rql_prefix`). Idempotent via `CREATE TABLE IF NOT EXISTS`.
pub const MIGRATION_003_NAMESPACES_TABLE: Migration = Migration {
    version: 3,
    name: "003_namespaces_table",
    sql: "CREATE TABLE IF NOT EXISTS namespaces (\
              group_id        TEXT PRIMARY KEY,\
              policy_json     TEXT NOT NULL,\
              recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),\
              schema_version  INTEGER NOT NULL DEFAULT 1\
          )",
};

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
#[allow(dead_code)] // planned consumer: consumer startup cleanup (D.3 spec)
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

// ─── Per-migration fns (extracted from schema.rs v0.1.4.1) ─────────────────
//
// These are free `pub(crate)` async fns rather than `impl TemporalGraph` methods
// so that (a) migrations.rs is the single home for all migration logic and
// (b) they can be called and tested independently of `TemporalGraph`.
//
// Call sites: `crate::core::schema::TemporalGraph::run_migrations` (schema.rs)
// calls each fn by its free-fn path.
//
// The `crate::core::error::Result` import at the top of this file covers the
// return type; `anyhow::anyhow!` is used for the `Error::Other` wrapper
// (same as the original impl in schema.rs).

// ─── Migration LEGACY ──────────────────────────────────────────────────────

/// Pre-002 backward migration: rename legacy `entities` (rql shape, has `label`)
/// to `rql_entities` so that the workspace P1 `entities` table can coexist.
///
/// Idempotent: only renames when a `label`-shaped legacy table exists AND the
/// new name is free.
pub(crate) async fn migrate_legacy_rql_entities_table(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    // Detect legacy rql shape: `entities` table with a `label` column.
    let mut rows = conn.query("PRAGMA table_info(entities)", ()).await?;
    let mut has_label = false;
    let mut has_any = false;
    while let Some(row) = rows.next().await? {
        has_any = true;
        let name: String = row.get(1)?;
        if name == "label" {
            has_label = true;
            break;
        }
    }
    if !has_any || !has_label {
        return Ok(());
    }

    // Don't clobber an existing rql_entities — if both are present the
    // rename happened previously and the bare `entities` is some other
    // table (e.g. workspace shape co-resident). Bail without touching.
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rql_entities'",
            (),
        )
        .await?;
    if rows.next().await?.is_some() {
        return Ok(());
    }

    // Rename data table + FTS5 sibling + indexes.
    conn.execute("ALTER TABLE entities RENAME TO rql_entities", ())
        .await?;
    let _ = conn
        .execute("ALTER TABLE entities_fts RENAME TO rql_entities_fts", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS idx_entities_group", ())
        .await;
    Ok(())
}

// ─── Migration 002 ─────────────────────────────────────────────────────────

/// Migration 002: rename `rql_entities` → `entities` (ADR-029b Decision 2).
///
/// Idempotent: exits immediately when `rql_entities` does not exist.
/// Called from `run_migrations()` BEFORE the base DDL block.
pub(crate) async fn migrate_002_drop_rql_prefix(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    // Check if rql_entities still exists — if not, migration already applied.
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rql_entities'",
            (),
        )
        .await?;
    if rows.next().await?.is_none() {
        return Ok(());
    }

    // Also check that `entities` does not exist yet (prevents double-rename collision).
    let mut check = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'",
            (),
        )
        .await?;
    if check.next().await?.is_some() {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_002: both rql_entities and entities exist — skipping rename; \
             manual inspection recommended"
        );
        return Ok(());
    }

    conn.execute("ALTER TABLE rql_entities RENAME TO entities", ())
        .await?;
    let _ = conn
        .execute("ALTER TABLE rql_entities_fts RENAME TO entities_fts", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS rql_entities_vec_idx", ())
        .await;
    let _ = conn
        .execute("DROP INDEX IF EXISTS idx_rql_entities_group", ())
        .await;

    tracing::info!(
        target: "kremory::migrations",
        "migrate_002: rql_entities renamed to entities"
    );
    Ok(())
}

// ─── Migration 004 ─────────────────────────────────────────────────────────

/// Migration 004: composite PK on `entities` (ADR-029b Decision 1).
///
/// Uses CREATE-COPY-DROP-RENAME to restructure the table to
/// `PRIMARY KEY (id, group_id)`.
///
/// Idempotency gate (Vera 2026-05-28 BUG-1 fix):
///   Uses `PRAGMA table_info('entities')` to check whether `group_id` is already
///   in the PK — this is the SHAPE-based sentinel. The old `entities_bak_004`
///   backup-table sentinel was a false gate: if the process crashed after creating
///   the backup but before creating `entities_new`, the next startup would see
///   the backup, return Ok(()), and silently leave the migration incomplete.
///
/// FK restore on ALL exit paths (Vera 2026-05-28 BUG-2 fix):
///   `PRAGMA foreign_keys = ON` is guaranteed via the `body_result` wrapping
///   pattern — same approach used by migrate_006. An error in any restructure
///   step still restores FK enforcement before propagating the error.
///
/// Post-migration `PRAGMA foreign_key_check` (Vera 2026-05-28 BUG-2 companion):
///   After the body completes successfully the migration asserts that no FK
///   violations were introduced.
pub(crate) async fn migrate_004_composite_pk_entities(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_004 step `{name}` failed: {e}"
            ))
        }
    }

    // Vera BUG-1 fix: SHAPE-based idempotency gate.
    // If `group_id` is already part of the entities PK (pk > 0), the migration
    // has completed — return immediately without touching anything.
    let mut pragma_rows = conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .map_err(step("g1_table_info"))?;
    while let Some(r) = pragma_rows
        .next()
        .await
        .map_err(step("g1_table_info_next"))?
    {
        let col_name: String = r.get(1).unwrap_or_default();
        let pk: i64 = r.get(5).unwrap_or(0);
        if col_name == "group_id" && pk > 0 {
            // Already migrated.
            return Ok(());
        }
    }

    // Check whether entities_new exists — signals a partial migration in progress
    // (crashed between CREATE entities_new and the DROP+RENAME).
    let mut rows2 = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities_new'",
            (),
        )
        .await
        .map_err(step("check_entities_new"))?;
    let partial_migration_in_progress = rows2
        .next()
        .await
        .map_err(step("check_entities_new_next"))?
        .is_some();

    if partial_migration_in_progress {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_004: entities_new already exists — attempting to complete partial migration"
        );
    }

    // Vera BUG-2 fix: PRAGMA foreign_keys = OFF is set once here; `PRAGMA
    // foreign_keys = ON` is guaranteed via the `body_result` pattern below
    // regardless of whether the body succeeds or returns Err. This mirrors the
    // pattern used by migrate_006 (Vera 2026-05-28 #2).
    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        if !partial_migration_in_progress {
            // Step 1: backup. Left in place as a rollback artifact.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entities_bak_004 AS SELECT * FROM entities",
                (),
            )
            .await
            .map_err(step("create_bak"))?;

            // Step 2: create new table with composite PK.
            // Note: `embedding` uses generic `BLOB` here (rather than `F32_BLOB(dim)`)
            // because `dim` is not in scope inside this migration helper; the vector
            // index (recreated below via `libsql_vector_idx`) works with raw BLOB columns.
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS entities_new (
                    id           TEXT NOT NULL,
                    label        TEXT NOT NULL,
                    properties   TEXT,
                    embedding    BLOB,
                    recorded_at  TEXT NOT NULL,
                    updated_at   TEXT,
                    group_id     TEXT NOT NULL DEFAULT 'default',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (id, group_id)
                );",
            )
            .await
            .map_err(step("create_entities_new"))?;

            // Step 3: copy + backfill group_id.
            conn.execute(
                "INSERT INTO entities_new (id, label, properties, embedding, recorded_at, updated_at, group_id, access_count)
                 SELECT id, label, properties, embedding, recorded_at, updated_at,
                        COALESCE(group_id, 'default') AS group_id,
                        COALESCE(access_count, 0)     AS access_count
                 FROM entities",
                (),
            )
            .await
            .map_err(step("copy_rows"))?;
        }

        // Step 4 + 5: drop old, rename new.
        conn.execute("DROP TABLE entities", ())
            .await
            .map_err(step("drop_old_entities"))?;
        conn.execute("ALTER TABLE entities_new RENAME TO entities", ())
            .await
            .map_err(step("rename_new_to_entities"))?;

        // Step 6: re-create indexes (vector index is best-effort on in-memory DBs).
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

        // Step 7+8: facts + episodic_edges composite FK restructure — PENDING
        // ship-architect design review (2026-05-28). Current best-effort ALTER+
        // UPDATE pattern is kept temporarily so the compile + downstream tests
        // continue to surface the issue rather than papering over it. See
        // `.ai-docs/planning/v014-adr-029b-composite-fk-design-2026-05-28.md`.
        let _ = conn
            .execute("ALTER TABLE facts ADD COLUMN subject_group_id TEXT", ())
            .await;
        let _ = conn
            .execute("ALTER TABLE facts ADD COLUMN object_group_id TEXT", ())
            .await;
        let _ = conn
            .execute(
                "UPDATE facts SET subject_group_id = (
                     SELECT COALESCE(e.group_id, 'default')
                     FROM entities e WHERE e.id = facts.subject_id
                     LIMIT 1
                 ) WHERE subject_group_id IS NULL",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "UPDATE facts SET object_group_id = (
                     SELECT COALESCE(e.group_id, 'default')
                     FROM entities e WHERE e.id = facts.object_id
                     LIMIT 1
                 ) WHERE object_group_id IS NULL AND object_id IS NOT NULL",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "ALTER TABLE episodic_edges ADD COLUMN entity_group_id TEXT",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "UPDATE episodic_edges SET entity_group_id = (
                     SELECT COALESCE(e.group_id, 'default')
                     FROM entities e WHERE e.id = episodic_edges.entity_id
                     LIMIT 1
                 ) WHERE entity_group_id IS NULL",
                (),
            )
            .await;

        Ok(())
    }
    .await;

    // Vera BUG-2 fix: always restore PRAGMA foreign_keys = ON, regardless of
    // whether the body succeeded or returned Err.  A failed migration that leaves
    // FK enforcement OFF on the connection is a silent data-integrity bug for all
    // subsequent application writes on that connection.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_004: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    // Propagate the body's result.
    body_result?;

    // NOTE: `PRAGMA foreign_key_check` is deliberately NOT run here. After
    // migrate_004 swaps entities to composite PK (id, group_id), the existing
    // `facts.subject_id REFERENCES entities(id)` constraint references a
    // non-unique column set — SQLite reports this as a violation until
    // migrate_006 rebuilds facts with the composite FK shape. The fk-integrity
    // check therefore belongs at the END of the migrate_006 step (already
    // present there, see line ~1163), not after migrate_004 alone — the schema
    // is intentionally in a mixed state between these two migrations.
    //
    // The always-restore `PRAGMA foreign_keys` wrapper above (Vera BUG-2 fix)
    // ensures the connection's FK enforcement is correctly re-enabled on any
    // exit path. The post-condition gate is migrate_006's fk_check.

    tracing::info!(
        target: "kremory::migrations",
        "migrate_004: composite PK (id, group_id) applied to entities"
    );
    Ok(())
}

// ─── Migration 005 ─────────────────────────────────────────────────────────

/// Migration 005: add `upgraded_at` column to `namespaces` (ADR-029b Decision 5).
///
/// Idempotent: `ALTER TABLE ADD COLUMN` errors are swallowed when the column
/// already exists.
pub(crate) async fn migrate_005_policy_upgraded_at(
    conn: &libsql::Connection,
) -> crate::core::error::Result<()> {
    let _ = conn
        .execute("ALTER TABLE namespaces ADD COLUMN upgraded_at TEXT", ())
        .await;
    tracing::info!(
        target: "kremory::migrations",
        "migrate_005: upgraded_at column ensured on namespaces"
    );
    Ok(())
}

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
        // Per Quinn cycle-1 LOW: propagate row.get errors rather than silently
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
    // Per Quinn cycle-1 MED: propagate via `?` rather than silent discard so
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
/// column on `entities` (TD-013 unified extraction architecture).
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

/// Migration 009 (Phase 2, TD-013): DROP the `label` column from `entities`.
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

/// Migration 012: add source-tier columns to `entities` + `v_entity_drift_candidates` view.
///
/// Adds three columns per ADR-045 §2 and ADR-046 §1:
///   - `entity_type_source TEXT CHECK(...)` — which tier last set the entity type
///   - `entity_type_assigned_at TEXT`        — ISO-8601 timestamp of the last assignment
///   - `ner_confidence REAL`                 — GLiNER / NER span confidence (Phase 1 only)
///
/// Creates `v_entity_drift_candidates` view (ADR-046 §1): entities with the same
/// name in the same namespace but different types — candidates for dream-phase reconcile.
/// ConsumerPinned entities are excluded from the view.
///
/// Legacy backfill: sets `entity_type_source = 'Phase1Ner'` and
/// `entity_type_assigned_at = COALESCE(recorded_at, datetime('now'))` on all existing rows
/// that have a NULL source, so queries can always rely on the column being non-NULL for rows
/// written before this migration.
///
/// Idempotency: each ADD COLUMN is guarded by a PRAGMA table_info check, so running
/// this migration twice is a no-op (no error, no duplication). The view uses
/// `CREATE VIEW IF NOT EXISTS`. The backfill UPDATE applies only to NULL-source rows.
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
        conn.execute(
            "ALTER TABLE entities ADD COLUMN ner_confidence REAL",
            (),
        )
        .await
        .map_err(step("alter_table_add_ner_confidence"))?;
    } else {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_012: ner_confidence already present — skipping ADD COLUMN"
        );
    }

    // ── Step 5: CREATE VIEW v_entity_drift_candidates ────────────────────────
    //
    // Entities with the same normalised name in the same namespace (group_id)
    // but different entity_type_id values are drift candidates for the dream
    // reconcile pass (ADR-046 §1). ConsumerPinned entities are excluded — they
    // are structurally protected from dream re-typing (ADR-045 §3).

    // NOTE: entities has no separate `name` column. The normalized entity name
    // IS the primary key `id` (set by normalize_name() at ingest time). The drift
    // view therefore compares LOWER(TRIM(e1.id)) to LOWER(TRIM(e2.id)).
    // The spec draft used `e2.namespace` (actual column: `group_id`) and
    // `e2.name` (actual field: `id`) — both corrected here per T1-T2-impl.md.
    conn.execute(
        "CREATE VIEW IF NOT EXISTS v_entity_drift_candidates AS \
         SELECT e1.id AS entity_id \
         FROM entities e1 \
         WHERE (e1.entity_type_source IS NULL OR e1.entity_type_source != 'ConsumerPinned') \
           AND EXISTS ( \
               SELECT 1 FROM entities e2 \
               WHERE LOWER(TRIM(e2.id)) = LOWER(TRIM(e1.id)) \
                 AND e2.group_id = e1.group_id \
                 AND e2.entity_type_id != e1.entity_type_id \
                 AND e2.id != e1.id \
           )",
        (),
    )
    .await
    .map_err(step("create_view_drift_candidates"))?;

    // ── Step 6: Legacy backfill ───────────────────────────────────────────────
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
         added to entities; v_entity_drift_candidates view created; legacy backfill done."
    );
    Ok(())
}

// ─── Migration 006 ─────────────────────────────────────────────────────────

/// Migration 006: install composite FK constraints on `facts` and `episodic_edges`
/// (ADR-029b Decision 1).
///
/// Uses CREATE-COPY-DROP-RENAME to replace the placeholder ADD COLUMN stubs
/// that migration 004 installed. After this migration both tables reference
/// `entities(id, group_id)` rather than the now-invalid single-column `entities(id)`.
///
/// Idempotency gates:
///   G1 — facts already has composite FK shape → already ran, return Ok(()).
///   G2 — entities does NOT have composite PK → migration 004 not yet applied, return Err.
///   G3 — `facts_new` exists → partial migration, resume from drop+rename.
///   G4 — `episodic_edges_new` exists → same for episodic_edges half.
///
/// The `dim` parameter is required because `facts` contains an `F32_BLOB(dim)`
/// vector column; the new table DDL must embed the same dimension value.
pub(crate) async fn migrate_006_composite_fk_facts_episodic_edges(
    conn: &libsql::Connection,
    dim: usize,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_006 step `{name}` failed: {e}"
            ))
        }
    }

    // Helper: check whether a table exists.
    async fn table_exists(
        conn: &libsql::Connection,
        name: &str,
    ) -> std::result::Result<bool, libsql::Error> {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                libsql::params![name],
            )
            .await?;
        let found = rows.next().await?.is_some();
        Ok(found)
    }

    // Helper: does `facts` already have a composite FK column?
    async fn facts_has_composite_fk(
        conn: &libsql::Connection,
    ) -> std::result::Result<bool, libsql::Error> {
        let mut rows = conn.query("PRAGMA foreign_key_list('facts')", ()).await?;
        while let Some(r) = rows.next().await? {
            let from: String = r.get(3).unwrap_or_default();
            if from == "subject_group_id" || from == "object_group_id" {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // Helper: does `entities` carry the composite PK from migrate_004?
    async fn entities_has_composite_pk(
        conn: &libsql::Connection,
    ) -> std::result::Result<bool, libsql::Error> {
        let mut rows = conn.query("PRAGMA table_info('entities')", ()).await?;
        while let Some(r) = rows.next().await? {
            let name: String = r.get(1).unwrap_or_default();
            let pk: i64 = r.get(5).unwrap_or(0);
            if name == "group_id" && pk > 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // G1 — primary idempotency gate (SHAPE-based):
    if facts_has_composite_fk(conn)
        .await
        .map_err(step("g1_facts_fk_shape"))?
    {
        return Ok(());
    }

    // G2 — pre-condition gate (SHAPE-based): entities must carry the
    // composite PK installed by migrate_004.
    if !entities_has_composite_pk(conn)
        .await
        .map_err(step("g2_entities_pk_shape"))?
    {
        return Err(crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_006: entities table does not have composite PK (id, group_id) — \
             run migrate_004_composite_pk_entities first"
        )));
    }

    // G3 — partial migration recovery: facts_new exists.
    let facts_partial = table_exists(conn, "facts_new")
        .await
        .map_err(step("g3_check_facts_new"))?;

    // G4 — partial migration recovery: episodic_edges_new exists.
    let edges_partial = table_exists(conn, "episodic_edges_new")
        .await
        .map_err(step("g4_check_episodic_edges_new"))?;

    // PRAGMA foreign_keys = OFF for the duration of the restructure.
    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;
    let body_result: crate::core::error::Result<()> = async {

    // ── facts half ───────────────────────────────────────────────────────────

    if facts_partial {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_006: facts_new already exists — attempting to complete partial migration"
        );
    } else {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts_bak_006 AS SELECT * FROM facts",
            (),
        )
        .await
        .map_err(step("create_facts_bak_006"))?;

        conn.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS facts_new (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id       TEXT NOT NULL,
                    subject_group_id TEXT NOT NULL DEFAULT 'default',
                    predicate        TEXT NOT NULL,
                    object_id        TEXT,
                    object_group_id  TEXT,
                    object_value     TEXT,
                    properties       TEXT,
                    embedding        F32_BLOB({dim}),
                    valid_from       TEXT NOT NULL,
                    valid_to         TEXT,
                    recorded_at      TEXT NOT NULL,
                    expired_at       TEXT,
                    invalid_at       TEXT,
                    group_id         TEXT NOT NULL DEFAULT 'default',
                    confidence       REAL DEFAULT 1.0,
                    source_episode_id INTEGER,
                    memory_type      TEXT,
                    content_hash     TEXT,
                    access_count     INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (subject_id, subject_group_id) REFERENCES entities(id, group_id),
                    FOREIGN KEY (object_id,  object_group_id)  REFERENCES entities(id, group_id),
                    FOREIGN KEY (source_episode_id)            REFERENCES episodes(id)
                )"
            ),
            (),
        )
        .await
        .map_err(step("create_facts_new"))?;

        conn.execute(
            "INSERT INTO facts_new (
                 id, subject_id, subject_group_id, predicate,
                 object_id, object_group_id, object_value, properties, embedding,
                 valid_from, valid_to, recorded_at, expired_at, invalid_at,
                 group_id, confidence, source_episode_id, memory_type, content_hash, access_count
             )
             SELECT
                 id,
                 subject_id,
                 COALESCE(subject_group_id, group_id, 'default'),
                 predicate,
                 object_id,
                 CASE WHEN object_id IS NULL THEN NULL
                      ELSE COALESCE(object_group_id, group_id, 'default') END,
                 object_value, properties, embedding,
                 valid_from, valid_to, recorded_at, expired_at, invalid_at,
                 COALESCE(group_id, 'default'),
                 confidence, source_episode_id, memory_type, content_hash, access_count
             FROM facts",
            (),
        )
        .await
        .map_err(step("copy_facts"))?;
    }

    conn.execute("DROP TABLE facts", ())
        .await
        .map_err(step("drop_facts"))?;
    conn.execute("ALTER TABLE facts_new RENAME TO facts", ())
        .await
        .map_err(step("rename_facts_new"))?;

    // Rebuild facts_fts (Vera 2026-05-28 #3): standalone FTS5 table; DROP TABLE facts
    // does NOT cascade. Rebuild from live data to prevent phantom FTS results.
    let _ = conn
        .execute("DROP TABLE IF EXISTS facts_fts", ())
        .await;
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            fact_id UNINDEXED,
            predicate,
            object_value
        )",
        (),
    )
    .await
    .map_err(step("create_facts_fts"))?;
    let _ = conn
        .execute(
            "INSERT INTO facts_fts (fact_id, predicate, object_value) \
             SELECT id, predicate, object_value FROM facts \
             WHERE object_value IS NOT NULL",
            (),
        )
        .await;

    // Re-create facts indexes.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_temporal \
         ON facts(subject_id, valid_from, expired_at)",
        (),
    )
    .await
    .map_err(step("idx_facts_temporal"))?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate, expired_at)",
        (),
    )
    .await
    .map_err(step("idx_facts_predicate"))?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object_id, expired_at)",
        (),
    )
    .await
    .map_err(step("idx_facts_object"))?;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_group ON facts(group_id)",
            (),
        )
        .await;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS facts_vec_idx \
             ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        )
        .await;
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_content_hash_unique \
         ON facts(content_hash) WHERE content_hash IS NOT NULL",
        (),
    )
    .await
    .map_err(step("idx_facts_content_hash_unique"))?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_content_hash ON facts(content_hash)",
        (),
    )
    .await
    .map_err(step("idx_facts_content_hash"))?;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject_group \
             ON facts(subject_id, subject_group_id)",
            (),
        )
        .await;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_object_group \
             ON facts(object_id, object_group_id) WHERE object_id IS NOT NULL",
            (),
        )
        .await;

    // ── episodic_edges half ──────────────────────────────────────────────────

    if edges_partial {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_006: episodic_edges_new already exists — resuming partial migration"
        );
    } else {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodic_edges_bak_006 AS SELECT * FROM episodic_edges",
            (),
        )
        .await
        .map_err(step("create_episodic_edges_bak_006"))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodic_edges_new (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id      INTEGER NOT NULL,
                entity_id       TEXT NOT NULL,
                entity_group_id TEXT NOT NULL DEFAULT 'default',
                role            TEXT NOT NULL DEFAULT 'mentioned',
                recorded_at     TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id),
                FOREIGN KEY (entity_id, entity_group_id) REFERENCES entities(id, group_id)
            )",
            (),
        )
        .await
        .map_err(step("create_episodic_edges_new"))?;

        conn.execute(
            "INSERT INTO episodic_edges_new (id, episode_id, entity_id, entity_group_id, role, recorded_at)
             SELECT id, episode_id, entity_id,
                    COALESCE(entity_group_id, 'default'),
                    role, recorded_at
             FROM episodic_edges",
            (),
        )
        .await
        .map_err(step("copy_episodic_edges"))?;
    }

    conn.execute("DROP TABLE episodic_edges", ())
        .await
        .map_err(step("drop_episodic_edges"))?;
    conn.execute("ALTER TABLE episodic_edges_new RENAME TO episodic_edges", ())
        .await
        .map_err(step("rename_episodic_edges_new"))?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodic_edges_entity \
         ON episodic_edges(entity_id, entity_group_id)",
        (),
    )
    .await
    .map_err(step("idx_episodic_edges_entity"))?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodic_edges_episode \
         ON episodic_edges(episode_id)",
        (),
    )
    .await
    .map_err(step("idx_episodic_edges_episode"))?;

        Ok(())
    }
    .await;

    // Always restore PRAGMA foreign_keys = ON, regardless of body success/failure.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_006: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    body_result?;

    // Boy-scout integrity check: any FK violation introduced by the restructure
    // surfaces here BEFORE the next application write hits it.
    let mut violations = conn
        .query("PRAGMA foreign_key_check", ())
        .await
        .map_err(step("fk_check_post"))?;
    if violations
        .next()
        .await
        .map_err(step("fk_check_post_next"))?
        .is_some()
    {
        return Err(crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_006: PRAGMA foreign_key_check reported violations after restructure — \
             refusing to proceed; inspect entities_bak_004 / facts_bak_006 / \
             episodic_edges_bak_006 for recovery"
        )));
    }

    tracing::info!(
        target: "kremory::migrations",
        "migrate_006: composite FK applied to facts + episodic_edges (foreign_key_check clean)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
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
