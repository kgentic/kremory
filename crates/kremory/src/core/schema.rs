use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::core::error::Result;

/// Dirty flag: set to `true` by write paths after a successful commit.
/// `SpeculativeCache` reads and clears this flag to invalidate tier-2 cache
/// entries. Process-global because there is one engine per process (ADR-007).
/// Story #215.
pub static DIRTY: AtomicBool = AtomicBool::new(false);

/// Last TTL sweep timestamp as Unix milliseconds. Zero = never swept.
/// CAS-protected — only one goroutine wins the sweep window. Story #235.
pub static LAST_TTL_SWEEP: AtomicI64 = AtomicI64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub label: String,
    pub properties: serde_json::Value,
    /// When this row was recorded in the database (audit timestamp).
    /// Renamed from `created_at` per Story #A1 (honesty fix).
    pub recorded_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: i64,
    pub subject_id: String,
    pub predicate: String,
    pub object_id: Option<String>,
    pub object_value: Option<String>,
    pub properties: Option<serde_json::Value>,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    /// When this row was recorded in the database (audit timestamp).
    /// Renamed from `created_at` per Story #A1 (honesty fix).
    pub recorded_at: DateTime<Utc>,
    pub expired_at: Option<DateTime<Utc>>,
    /// SQL column `invalid_at` = contradiction-resolver invalidation timestamp.
    /// NOT the struct field `valid_to` (window boundary). Distinct concept per
    /// Story #318 HITL.
    pub invalid_at: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
    pub confidence: f64,
    pub source_episode_id: Option<i64>,
    /// Semantic memory type classification. None = unclassified. Story #208.
    pub memory_type: Option<crate::memory::types::MemoryType>,
    /// SHA-256 content hash for dedup (Story #209). Absent on legacy rows.
    pub content_hash: Option<String>,
    /// Number of times this fact was retrieved (Story #247).
    pub access_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: i64,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub source_type: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub group_id: Option<String>,
    pub saga_id: Option<String>,
    pub sequence_number: Option<i64>,
    /// SHA-256 content hash for insert-level dedup (Story #209).
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEdge {
    pub id: i64,
    pub episode_id: i64,
    pub entity_id: String,
    pub role: String,
    /// When this row was recorded in the database (audit timestamp).
    /// Renamed from `created_at` per Story #A1 (honesty fix).
    pub recorded_at: DateTime<Utc>,
}

/// Smart pointer for `BEGIN IMMEDIATE` transactions.
///
/// Returned by `TemporalGraph::begin_immediate_if_needed`. Must be explicitly
/// committed via `.commit().await?` or rolled back via `.rollback().await?`.
/// Dropping without an explicit dispatch emits a `tracing::warn!` + metrics
/// counter and resets `has_outer_transaction` (defensive). Story #246 / ADR-020.
#[must_use = "BeginGuard must be committed or rolled back explicitly"]
pub struct BeginGuard<'a> {
    graph: &'a TemporalGraph,
    /// `true` when this guard opened the transaction; `false` when an outer
    /// transaction was already active (nested call — no BEGIN issued).
    opened: bool,
    /// Whether `commit()` or `rollback()` has been called (prevents double-dispatch).
    dispatched: bool,
}

impl<'a> BeginGuard<'a> {
    pub(crate) fn new(graph: &'a TemporalGraph, opened: bool) -> Self {
        Self {
            graph,
            opened,
            dispatched: false,
        }
    }

    /// Commit the transaction. No-op when this guard did not open a transaction
    /// (nested call with an outer transaction already active).
    pub async fn commit(mut self) -> Result<()> {
        self.dispatched = true;
        if self.opened {
            self.graph.conn.execute("COMMIT", ()).await?;
            DIRTY.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Roll back the transaction. No-op when this guard did not open a transaction.
    pub async fn rollback(mut self) -> Result<()> {
        self.dispatched = true;
        if self.opened {
            let _ = self.graph.conn.execute("ROLLBACK", ()).await;
        }
        Ok(())
    }
}

impl Drop for BeginGuard<'_> {
    fn drop(&mut self) {
        if !self.dispatched && self.opened {
            tracing::warn!(
                target: "kremory::db",
                "BeginGuard dropped without explicit commit or rollback — rolling back defensively"
            );
            metrics::counter!("rql.db.begin_guard_drop_without_explicit_commit").increment(1);
            // Defensive reset — the DB connection will auto-rollback on drop/reuse anyway,
            // but we reset the flag so the next caller doesn't see a stale outer-tx state.
            self.graph
                .has_outer_transaction
                .store(false, Ordering::Release);
        }
    }
}

pub struct TemporalGraph {
    pub(crate) _db: libsql::Database,
    pub(crate) conn: libsql::Connection,
    /// Write-serialiser mutex (ADR-022). Acquired first on every write path.
    /// Prevents concurrent `BEGIN IMMEDIATE` races on a single libsql connection.
    pub(crate) write_lock: Arc<Mutex<()>>,
    /// `true` when a `BEGIN IMMEDIATE` is already active on this connection.
    /// Used by `begin_immediate_if_needed` to skip nested BEGIN. Story #246.
    pub(crate) has_outer_transaction: AtomicBool,
}

impl TemporalGraph {
    pub async fn open(path: &str) -> Result<Self> {
        let db = libsql::Builder::new_local(path).build().await?;
        let conn = db.connect()?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;").await?;
        let graph = Self {
            _db: db,
            conn,
            write_lock: Arc::new(Mutex::new(())),
            has_outer_transaction: AtomicBool::new(false),
        };
        graph.run_migrations().await?;
        Ok(graph)
    }

    pub async fn open_in_memory() -> Result<Self> {
        let db = libsql::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        let graph = Self {
            _db: db,
            conn,
            write_lock: Arc::new(Mutex::new(())),
            has_outer_transaction: AtomicBool::new(false),
        };
        graph.run_migrations().await?;
        Ok(graph)
    }

    /// Begin an IMMEDIATE transaction if one is not already active.
    ///
    /// Acquires the `write_lock` mutex (ADR-022) first to serialise concurrent write
    /// paths on the single libsql connection. Returns a `BeginGuard` that must be
    /// explicitly committed or rolled back. Story #246 / ADR-020.
    ///
    /// If an outer `BEGIN IMMEDIATE` is already in progress on this connection
    /// (i.e., `has_outer_transaction == true`), the guard is returned without issuing
    /// another BEGIN — the outer transaction covers the nested operation.
    pub async fn begin_immediate_if_needed(&self) -> Result<BeginGuard<'_>> {
        let _guard = self.write_lock.lock().await;
        // Deliberately drop `_guard` after the mutex is taken but BEFORE await — the
        // write_lock is a serialiser (ensures single concurrent writer), not a
        // transaction scope holder. SQLite's BEGIN IMMEDIATE itself holds the writer
        // lock at the DB level for the duration of the transaction.
        let already_open = self
            .has_outer_transaction
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err();
        if already_open {
            // Nested call — outer transaction already active; return a no-op guard.
            return Ok(BeginGuard::new(self, false));
        }
        self.conn
            .execute("BEGIN IMMEDIATE", ())
            .await
            .inspect_err(|_e| {
                // Reset the flag — we failed to open the transaction.
                self.has_outer_transaction.store(false, Ordering::Release);
            })?;
        Ok(BeginGuard::new(self, true))
    }

    /// One-shot rename for legacy `entities` (rql shape) → `rql_entities`.
    ///
    /// **Why**: S5.C P1 introduces a workspace `entities` table on the same
    /// DB file as rql's graph. The two cannot coexist by name. The rqlc
    /// rename to `rql_entities` is the structural resolution (P1.F1, see
    /// CLAUDE.md `feedback_no_shortcuts_zero_tech_debt`).
    ///
    /// **Detection**: a legacy table is identified by `entities` having a
    /// `label` column (rql shape) — the workspace amend uses `type`. If
    /// the legacy table is found AND `rql_entities` is free, rename it
    /// in place and migrate the FTS5 + index siblings. Otherwise no-op.
    ///
    /// **Idempotent**: post-rename the legacy `entities` is gone and the
    /// next open finds either no entities table (fresh DB) or only the
    /// workspace one (no label column).
    async fn migrate_legacy_rql_entities_table(conn: &libsql::Connection) -> Result<()> {
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

        // Rename data table + FTS5 sibling + indexes. SQLite supports
        // ALTER TABLE RENAME TO across regular and virtual tables.
        conn.execute("ALTER TABLE entities RENAME TO rql_entities", ())
            .await?;
        // FTS5 rename is best-effort — the FTS virtual table may not
        // have been installed yet on partially-migrated DBs.
        let _ = conn
            .execute("ALTER TABLE entities_fts RENAME TO rql_entities_fts", ())
            .await;
        // Indexes — vector + group_id. Both are CREATE INDEX IF NOT EXISTS
        // downstream, so on failure (missing index) the re-create path
        // covers them.
        let _ = conn
            .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
            .await;
        let _ = conn
            .execute("DROP INDEX IF EXISTS idx_entities_group", ())
            .await;
        Ok(())
    }

    async fn run_migrations(&self) -> Result<()> {
        // Backward migration (S5.C P1.F1, 2026-05-19): pre-rename dev DBs
        // hold the rql graph table at bare name `entities`. The workspace
        // P1 amend installed a colliding workspace `entities` on the same
        // file. Rename the legacy rql table out of the way before installing
        // the canonical `rql_entities` shape. Idempotent: only renames when
        // a label-shaped legacy table exists AND the new name is free.
        Self::migrate_legacy_rql_entities_table(&self.conn).await?;

        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS rql_entities (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties TEXT,
                    embedding F32_BLOB(384),
                    recorded_at TEXT NOT NULL,
                    updated_at TEXT,
                    group_id TEXT
                )",
                (),
            )
            .await?;
        // Vector index — may fail on in-memory DBs, non-fatal
        let _ = self.conn.execute(
            "CREATE INDEX IF NOT EXISTS rql_entities_vec_idx ON rql_entities(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        ).await;
        let _ = self
            .conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_rql_entities_group ON rql_entities(group_id)",
                (),
            )
            .await;
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_type TEXT,
                    metadata TEXT,
                    group_id TEXT,
                    saga_id TEXT,
                    sequence_number INTEGER
                )",
                (),
            )
            .await?;
        let _ = self
            .conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_saga ON episodes(saga_id)",
                (),
            )
            .await;
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id TEXT,
                    object_value TEXT,
                    properties TEXT,
                    embedding F32_BLOB(384),
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    recorded_at TEXT NOT NULL,
                    expired_at TEXT,
                    invalid_at TEXT,
                    group_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    source_episode_id INTEGER,
                    FOREIGN KEY (subject_id) REFERENCES rql_entities(id),
                    FOREIGN KEY (object_id) REFERENCES rql_entities(id),
                    FOREIGN KEY (source_episode_id) REFERENCES episodes(id)
                )",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_temporal ON facts(subject_id, valid_from, expired_at)",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate, expired_at)",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object_id, expired_at)",
                (),
            )
            .await?;
        let _ = self
            .conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_group ON facts(group_id)",
                (),
            )
            .await;
        // Vector index for fact embeddings — may fail on in-memory DBs, non-fatal
        let _ = self.conn.execute(
            "CREATE INDEX IF NOT EXISTS facts_vec_idx ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        ).await;
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS episodic_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id INTEGER NOT NULL,
                    entity_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'mentioned',
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id),
                    FOREIGN KEY (entity_id) REFERENCES rql_entities(id)
                )",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_edges_entity ON episodic_edges(entity_id)",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_edges_episode ON episodic_edges(episode_id)",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS rql_entities_fts USING fts5(
                    entity_id UNINDEXED,
                    label,
                    properties
                )",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    fact_id UNINDEXED,
                    predicate,
                    object_value
                )",
                (),
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod schema_tests {
    /// G3 gate: DDL must use `recorded_at` not `created_at`. Story #A1.
    #[test]
    fn schema_uses_recorded_at_not_created_at() {
        // Inline the DDL strings that run_migrations executes and verify
        // they contain `recorded_at` and NOT `created_at`.
        let ddl_entities = "CREATE TABLE IF NOT EXISTS rql_entities (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties TEXT,
                    embedding F32_BLOB(384),
                    recorded_at TEXT NOT NULL,
                    updated_at TEXT,
                    group_id TEXT
                )";
        let ddl_facts = "CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id TEXT,
                    object_value TEXT,
                    properties TEXT,
                    embedding F32_BLOB(384),
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    recorded_at TEXT NOT NULL,
                    expired_at TEXT,
                    invalid_at TEXT,
                    group_id TEXT,
                    confidence REAL DEFAULT 1.0
                )";
        let ddl_edges = "CREATE TABLE IF NOT EXISTS episodic_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id INTEGER NOT NULL,
                    entity_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'mentioned',
                    recorded_at TEXT NOT NULL
                )";
        for (name, ddl) in [
            ("rql_entities", ddl_entities),
            ("facts", ddl_facts),
            ("episodic_edges", ddl_edges),
        ] {
            assert!(
                ddl.contains("recorded_at"),
                "DDL for `{name}` must contain `recorded_at`"
            );
            assert!(
                !ddl.contains("created_at"),
                "DDL for `{name}` must NOT contain `created_at` (G3)"
            );
        }
    }
}
