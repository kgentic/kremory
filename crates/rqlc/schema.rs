use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub label: String,
    pub properties: serde_json::Value,
    pub created_at: DateTime<Utc>,
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
    pub created_at: DateTime<Utc>,
    pub expired_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
    pub group_id: Option<String>,
    pub confidence: f64,
    pub source_episode_id: Option<i64>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEdge {
    pub id: i64,
    pub episode_id: i64,
    pub entity_id: String,
    pub role: String,
    pub created_at: DateTime<Utc>,
}

pub struct TemporalGraph {
    pub(crate) _db: libsql::Database,
    pub(crate) conn: libsql::Connection,
}

impl TemporalGraph {
    pub async fn open(path: &str) -> Result<Self> {
        let db = libsql::Builder::new_local(path).build().await?;
        let conn = db.connect()?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;").await?;
        let graph = Self { _db: db, conn };
        graph.run_migrations().await?;
        Ok(graph)
    }

    pub async fn open_in_memory() -> Result<Self> {
        let db = libsql::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        let graph = Self { _db: db, conn };
        graph.run_migrations().await?;
        Ok(graph)
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
            .execute(
                "ALTER TABLE entities_fts RENAME TO rql_entities_fts",
                (),
            )
            .await;
        // Indexes — vector + group_id. Both are CREATE INDEX IF NOT EXISTS
        // downstream, so on failure (missing index) the re-create path
        // covers them.
        let _ = conn
            .execute(
                "DROP INDEX IF EXISTS entities_vec_idx",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "DROP INDEX IF EXISTS idx_entities_group",
                (),
            )
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
                    created_at TEXT NOT NULL,
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
                    created_at TEXT NOT NULL,
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
                    created_at TEXT NOT NULL,
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
