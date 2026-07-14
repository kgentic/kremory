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
/// a slice in the consumer crate's migrations module.
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

// ─── Split sub-modules ────────────────────────────────────────────────────────
//
// TD-045: migrate_migrations_god_file_split — the per-migration fns that lived
// inline in this file have been moved to per-file modules under `migrations/`.
// All public/pub(crate) paths are preserved via wildcard re-exports below so
// call sites in `core/schema.rs` require no import changes.

mod defs_a;
mod defs_b;
mod defs_c;
mod defs_d;
mod defs_e;
mod defs_f;
mod defs_g1;
mod defs_g2;
mod defs_h;
// defs_i (ADR-072 seq1 impl-spec §1): migrate_022_episodes_content_recall.
mod defs_i;
// defs_j (TD-115): migrate_023_vector_index_column_type.
mod defs_j;

pub(crate) use defs_a::*;
pub(crate) use defs_b::*;
pub(crate) use defs_c::*;
// defs_d, defs_e, defs_f, defs_g1, defs_g2 contain `pub async fn` items
// (emergency downgrade entry-points callable by external tooling). These must
// be re-exported as `pub use` rather than `pub(crate) use` to preserve the
// same external-crate visibility they had in the original flat migrations.rs.
pub use defs_d::*;
pub use defs_e::*;
pub use defs_f::*;
pub use defs_g1::*;
pub use defs_g2::*;
pub(crate) use defs_h::*;
// defs_i (ADR-072 seq1) is entirely empty when `content-search` is off (its
// sole item is feature-gated at the same level) — gate the re-export itself
// too, or the glob becomes a literal "unused import" under `-D warnings`.
#[cfg(feature = "content-search")]
pub(crate) use defs_i::*;
pub(crate) use defs_j::*;

#[cfg(test)]
mod tests;
