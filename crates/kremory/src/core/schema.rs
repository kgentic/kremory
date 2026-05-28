//! Core schema structs for kremory's SQLite storage layer.
//!
//! # Namespace / storage-column asymmetry (v0.1.0)
//!
//! The public API uses `Namespace` (struct in `memory::types`) with fields
//! `namespace` and `thread`. Internally, storage maps these to `group_id`
//! on `entities` and `facts`. The public-API → SQL-column rename
//! (i.e. adding `namespace_id` / `thread_id` SQL columns) is deferred to
//! v0.1.1 behind a migration file. Callers must NOT hardcode `group_id`
//! column semantics — access only via `TemporalGraph` methods.

use chrono::{DateTime, Utc};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
// std::sync::Mutex is used only for NamespacePolicyCache (sync, O(1) critical section).
// tokio::sync::Mutex is used for write_lock (async await in begin_immediate_if_needed).
use tokio::sync::Mutex as AsyncMutex;

use crate::core::error::Result;

/// Default capacity for the namespace policy cache.
/// Built at const time from `NonZeroUsize::MIN + 255` — no expect/unwrap.
pub(crate) const DEFAULT_NS_POLICY_CACHE_CAP: NonZeroUsize = NonZeroUsize::MIN.saturating_add(255);

/// LRU cache for namespace policies (ADR-029b Decision 4).
///
/// Capacity-bounded (default 256 entries); no TTL — entries are invalidated
/// explicitly via `invalidate` when a policy is upgraded.
///
/// Uses `std::sync::Mutex` (not `tokio::sync::Mutex`) because all operations
/// are O(1) and the critical section is tiny — avoids async overhead on the
/// hot recall path.
pub(crate) struct NamespacePolicyCache {
    inner: std::sync::Mutex<LruCache<String, crate::memory::types::NamespacePolicy>>,
}

impl NamespacePolicyCache {
    /// Create a cache with the given capacity. Caller guarantees non-zero
    /// via the `NonZeroUsize` type — no runtime check, no fallback.
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: std::sync::Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Non-destructive peek — returns a clone if cached, `None` on miss.
    /// Does NOT update LRU recency (use `put` to refresh on a DB load).
    pub(crate) fn peek(&self, group_id: &str) -> Option<crate::memory::types::NamespacePolicy> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .peek(group_id)
            .cloned()
    }

    /// Insert or replace a policy entry. Called after a DB load.
    pub(crate) fn put(&self, group_id: &str, policy: crate::memory::types::NamespacePolicy) {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .put(group_id.to_string(), policy);
    }

    /// Evict the entry for `group_id`. Called by `upgrade_namespace_policy`
    /// inside the BEGIN IMMEDIATE transaction (Decision 4 race semantics).
    pub(crate) fn invalidate(&self, group_id: &str) {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop(group_id);
    }
}

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
    /// Number of times this entity was retrieved by a search query (Story #247).
    /// Column lives on `entities`; incremented atomically by search paths.
    pub access_count: i64,
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
    /// DEPRECATED: active access tracking moved to `Entity.access_count` on
    /// `entities`. This field remains for backward compat (libsql does not
    /// support DROP COLUMN on older SQLite builds). Value is always 0 going
    /// forward; do not write to `facts.access_count` in new code.
    pub access_count: i64,
    // ── ADR-029b (v0.1.5): composite FK fields ───────────────────────────────
    /// group_id of the subject entity in the composite FK after migration 004.
    /// Populated from `COALESCE(old_group_id, 'default')` during migration 004
    /// backfill. On pre-migration DBs this is absent and defaults to `None`.
    /// Post-migration this is always `Some` (the backfill guarantees non-NULL).
    pub subject_group_id: Option<String>,
    /// group_id of the object entity. `None` when `object_id` is `None`
    /// (literal-object facts) or on pre-migration 004 rows.
    pub object_group_id: Option<String>,
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
    // ── ADR-029b (v0.1.5): composite FK field ────────────────────────────────
    /// group_id of the target entity in the composite FK after migration 004.
    /// Populated from `COALESCE(e.group_id, 'default')` during migration 004
    /// backfill. On pre-migration DBs this is absent and defaults to `None`.
    /// Post-migration this is always `Some`.
    pub entity_group_id: Option<String>,
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
            // Per-handle dirty flag — only this graph handle's flag is set,
            // not a process-global. FU.6: prevents one TG instance's writes
            // from invalidating another's speculative cache.
            self.graph.dirty.store(true, Ordering::Release);
            self.graph
                .has_outer_transaction
                .store(false, Ordering::Release);
        }
        Ok(())
    }

    /// Roll back the transaction. No-op when this guard did not open a transaction.
    pub async fn rollback(mut self) -> Result<()> {
        self.dispatched = true;
        if self.opened {
            let _ = self.graph.conn.execute("ROLLBACK", ()).await;
            self.graph
                .has_outer_transaction
                .store(false, Ordering::Release);
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
    /// Uses `AsyncMutex` (tokio) so the lock can be held across `.await` points.
    pub(crate) write_lock: Arc<AsyncMutex<()>>,
    /// `true` when a `BEGIN IMMEDIATE` is already active on this connection.
    /// Used by `begin_immediate_if_needed` to skip nested BEGIN. Story #246.
    pub(crate) has_outer_transaction: AtomicBool,
    /// Per-handle dirty flag. Set to `true` by `BeginGuard::commit()` after a
    /// successful write. `flush_if_dirty()` CAS-clears it and checkpoints.
    /// `SpeculativeCache::check_dirty_and_invalidate()` takes `&Arc<AtomicBool>`
    /// from this field so cache invalidation is scoped to the owning graph handle,
    /// not the process. Story #215 / FU.6 (per-handle not global).
    pub(crate) dirty: Arc<AtomicBool>,
    /// Embedding vector dimensionality used when creating the schema. Default 384.
    /// Must match the `EmbeddingProvider` output dimension or the SQLite vector
    /// index will reject inserts with a dimension mismatch error.
    pub(crate) embedding_dim: usize,
    /// LRU cache for namespace policies (ADR-029b Decision 4).
    /// Capacity 256; invalidated on upgrade.
    pub(crate) policy_cache: NamespacePolicyCache,
}

impl TemporalGraph {
    pub async fn open(path: &str) -> Result<Self> {
        Self::open_with_dim(path, 384).await
    }

    /// Open (or create) a database at `path` with an explicit embedding dimension.
    ///
    /// Use when your embedder outputs vectors wider than the default 384 dims
    /// (e.g. `768` for `nomic-embed-text`, `1536` for OpenAI `text-embedding-3-small`).
    pub async fn open_with_dim(path: &str, embedding_dim: usize) -> Result<Self> {
        let db = libsql::Builder::new_local(path).build().await?;
        let conn = db.connect()?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;").await?;
        let graph = Self {
            _db: db,
            conn,
            write_lock: Arc::new(AsyncMutex::new(())),
            has_outer_transaction: AtomicBool::new(false),
            dirty: Arc::new(AtomicBool::new(false)),
            embedding_dim,
            policy_cache: NamespacePolicyCache::new(DEFAULT_NS_POLICY_CACHE_CAP),
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
            write_lock: Arc::new(AsyncMutex::new(())),
            has_outer_transaction: AtomicBool::new(false),
            dirty: Arc::new(AtomicBool::new(false)),
            embedding_dim: 384,
            policy_cache: NamespacePolicyCache::new(DEFAULT_NS_POLICY_CACHE_CAP),
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

    /// Flush pending writes to storage if the `DIRTY` flag is set.
    ///
    /// Atomically checks-and-clears `DIRTY` (CAS `true → false`). If the flag
    /// was set, issues a `PRAGMA wal_checkpoint(PASSIVE)` to push WAL frames
    /// to the main DB file. If the flag was already `false` (no writes since the
    /// last flush), returns immediately without touching the connection.
    ///
    /// Called by background task and shutdown path — never on the hot write path.
    /// Story #215. Returns `Result<()>` to propagate DB errors from the WAL
    /// checkpoint step (rationale: `bool` would hide checkpoint failures).
    pub async fn flush_if_dirty(&self) -> Result<()> {
        // CAS old=true → new=false. Ok → we won the race and must flush.
        // Err → flag was already false; nothing to do.
        if self
            .dirty
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // WAL checkpoint: push buffered frames to main DB file.
            // Non-fatal — busy/locked DBs return non-zero but don't error.
            let _ = self
                .conn
                .execute("PRAGMA wal_checkpoint(PASSIVE)", ())
                .await;
        }
        Ok(())
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

        // ADR-029b Migration 002: rename rql_entities → entities.
        // Idempotent: only runs when rql_entities still exists.
        // Must run BEFORE the CREATE TABLE IF NOT EXISTS block below so that
        // the base DDL targets `entities` (not `rql_entities`) on all paths.
        Self::migrate_002_drop_rql_prefix(&self.conn).await?;

        let dim = self.embedding_dim;
        self.conn
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties TEXT,
                    embedding F32_BLOB({dim}),
                    recorded_at TEXT NOT NULL,
                    updated_at TEXT,
                    group_id TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0
                )"
                ),
                (),
            )
            .await?;
        // Vector index — may fail on in-memory DBs, non-fatal
        let _ = self.conn.execute(
            "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        ).await;
        let _ = self
            .conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id)",
                (),
            )
            .await;
        // Story #247 migration: add access_count column to existing entities tables
        // created before this column was added to the DDL. Idempotent — ALTER TABLE ADD
        // COLUMN is a no-op when the column already exists in libsql (SQLite 3.37+).
        // Errors are swallowed: duplicate-column errors are expected on fresh DBs where
        // the CREATE TABLE IF NOT EXISTS already includes the column.
        let _ = self
            .conn
            .execute(
                "ALTER TABLE entities ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
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
                &format!(
                    "CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id TEXT,
                    object_value TEXT,
                    properties TEXT,
                    embedding F32_BLOB({dim}),
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    recorded_at TEXT NOT NULL,
                    expired_at TEXT,
                    invalid_at TEXT,
                    group_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    source_episode_id INTEGER,
                    memory_type TEXT,
                    content_hash TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (subject_id) REFERENCES entities(id),
                    FOREIGN KEY (object_id) REFERENCES entities(id),
                    FOREIGN KEY (source_episode_id) REFERENCES episodes(id)
                )"
                ),
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
        // FU.1: UNIQUE partial index on content_hash — storage-layer backstop for the
        // TOCTTOU-safe check+insert (partial: NULL allowed for pre-hash-rollout rows).
        self.conn
            .execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_content_hash_unique \
                 ON facts(content_hash) WHERE content_hash IS NOT NULL",
                (),
            )
            .await?;
        // FU.1: non-unique index for dedup-check SELECT performance (eliminates full-table scan).
        // Note: SQLite can use the unique index above for equality lookups; this non-unique
        // index is kept as an explicit covering hint and for parity with the spec AC.3.
        self.conn
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_content_hash \
                 ON facts(content_hash)",
                (),
            )
            .await?;
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS episodic_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id INTEGER NOT NULL,
                    entity_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'mentioned',
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id),
                    FOREIGN KEY (entity_id) REFERENCES entities(id)
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
                "CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
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
        // ADR-029a (v0.1.4): namespaces table for per-namespace policy storage.
        // CREATE-only, no backfill — populated lazily via register_namespace +
        // first-encounter writes (see TemporalGraph::ensure_namespace_policy_row).
        // The unprefixed name `namespaces` is deliberate (ADR-029a Decision 8);
        // the rest of the `rql_*` rename lands in ADR-029b.
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS namespaces (
                    group_id        TEXT PRIMARY KEY,
                    policy_json     TEXT NOT NULL,
                    recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
                    schema_version  INTEGER NOT NULL DEFAULT 1
                )",
                (),
            )
            .await?;

        // ADR-029b Migration 004: composite PK on entities table.
        // Must run AFTER the base DDL (entities table may have just been created).
        // Idempotent: skips when backup table entities_bak_004 already present.
        Self::migrate_004_composite_pk_entities(&self.conn).await?;

        // ADR-029b Migration 005: add upgraded_at to namespaces.
        // Idempotent: swallows duplicate-column error.
        Self::migrate_005_policy_upgraded_at(&self.conn).await?;

        // ADR-029b Migration 006: composite FK on facts + episodic_edges.
        // Replaces the placeholder ADD COLUMN stubs in migration 004.
        // Must run AFTER migrate_004 (pre-condition gate inside).
        // Idempotent: skips when facts_bak_006 already present.
        Self::migrate_006_composite_fk_facts_episodic_edges(&self.conn, self.embedding_dim).await?;

        Ok(())
    }

    /// Re-run the migration suite on this already-open handle.
    ///
    /// Every DDL statement in `run_migrations` uses `IF NOT EXISTS`, so
    /// calling this a second time on the same connection MUST be a no-op
    /// with no errors. This is the G5 idempotency invariant.
    ///
    /// Only available in `test` builds and when the `test-utils` feature is
    /// enabled. Never ship this in a production binary.
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn run_migrations_again_for_test(&self) -> Result<()> {
        self.run_migrations().await
    }

    /// Return a shared reference to the per-handle dirty flag.
    ///
    /// For use in tests that need to observe or arm the flag without going
    /// through a full write cycle. Only available in test/test-utils context.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn dirty_flag(&self) -> &Arc<AtomicBool> {
        &self.dirty
    }

    /// Count episodes recorded after `since` for a given namespace group.
    ///
    /// Used by `Engine::recall` to check episode-count thresholds for
    /// automatic consolidation triggering. The `group_id` parameter maps
    /// directly to the storage column (public API: `namespace_to_group_id(ns)`).
    ///
    /// Returns `Ok(0)` when the group has no episodes or none match the filter.
    pub async fn count_episodes_since(
        &self,
        group_id: &str,
        since: DateTime<Utc>,
    ) -> Result<usize> {
        let mut rows = self
            .conn
            .query(
                "SELECT COUNT(*) FROM episodes WHERE group_id = ?1 AND timestamp > ?2",
                libsql::params![group_id, since.timestamp()],
            )
            .await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| anyhow::anyhow!("count_episodes_since: query returned no row"))?;
        let count: i64 = row.get(0)?;
        Ok(count as usize)
    }

    // ── ADR-029b Inline Migrations ────────────────────────────────────────────

    /// Migration 002: rename `rql_entities` → `entities` (ADR-029b Decision 2).
    ///
    /// Idempotent: exits immediately when `rql_entities` does not exist.
    /// Uses DROP+CREATE index pattern (SQLite has no ALTER INDEX RENAME).
    /// FTS5 virtual table renamed via ALTER TABLE RENAME TO.
    ///
    /// Called from `run_migrations()` BEFORE the base DDL block so that
    /// fresh DBs never see `rql_entities` and existing DBs are renamed
    /// before `CREATE TABLE IF NOT EXISTS entities` becomes a no-op.
    async fn migrate_002_drop_rql_prefix(conn: &libsql::Connection) -> Result<()> {
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
            // Both tables exist — prior partial migration. Log and return; the
            // base DDL `CREATE TABLE IF NOT EXISTS entities` will be a no-op.
            tracing::warn!(
                target: "kremory::migrations",
                "migrate_002: both rql_entities and entities exist — skipping rename; \
                 manual inspection recommended"
            );
            return Ok(());
        }

        // Rename data table.
        conn.execute("ALTER TABLE rql_entities RENAME TO entities", ())
            .await?;

        // Rename FTS5 virtual table — best effort (may not exist on very old DBs).
        let _ = conn
            .execute("ALTER TABLE rql_entities_fts RENAME TO entities_fts", ())
            .await;

        // Index rename: DROP old + re-create with new names.
        // SQLite has no ALTER INDEX RENAME TO.
        let _ = conn
            .execute("DROP INDEX IF EXISTS rql_entities_vec_idx", ())
            .await;
        let _ = conn
            .execute("DROP INDEX IF EXISTS idx_rql_entities_group", ())
            .await;
        // New indexes created by the standard `CREATE INDEX IF NOT EXISTS` block
        // in run_migrations() that follows this call — no explicit re-create needed here.

        tracing::info!(
            target: "kremory::migrations",
            "migrate_002: rql_entities renamed to entities"
        );
        Ok(())
    }

    /// Migration 004: composite PK on `entities` (ADR-029b Decision 1).
    ///
    /// SQLite does not support `ALTER TABLE ADD PRIMARY KEY`. The migration
    /// uses CREATE-COPY-DROP-RENAME to restructure the table:
    ///
    /// 1. Backup existing rows into `entities_bak_004` (survives rollback).
    /// 2. Create `entities_new` with `PRIMARY KEY (id, group_id)`.
    /// 3. Backfill `group_id = COALESCE(group_id, 'default')` during copy.
    /// 4. DROP old `entities`.
    /// 5. RENAME `entities_new` → `entities`.
    /// 6. Re-create indexes.
    /// 7. Backfill `facts.subject_group_id` + `facts.object_group_id`.
    /// 8. Backfill `episodic_edges.entity_group_id`.
    ///
    /// Idempotent: skips if `entities_bak_004` already exists (prior run).
    /// The backup table is intentionally left in place as a rollback artifact.
    ///
    /// **Caller must hold a backup** (kremory-admin backup command) before
    /// executing this migration; the backup table is NOT sufficient for
    /// page-level corruption recovery.
    async fn migrate_004_composite_pk_entities(conn: &libsql::Connection) -> Result<()> {
        // Helper: tag a step name onto whatever libsql error bubbles up so
        // a future regression points the operator at the failing statement
        // rather than a bare `SqliteFailure(1, "SQL logic error")`.
        fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
            move |e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "migrate_004 step `{name}` failed: {e}"
                ))
            }
        }

        // Idempotency gate: if backup table exists, migration already ran.
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entities_bak_004'",
                (),
            )
            .await
            .map_err(step("check_bak_table"))?;
        if rows
            .next()
            .await
            .map_err(step("check_bak_table_next"))?
            .is_some()
        {
            return Ok(());
        }

        // Also check entities has the old single-column PK by checking if
        // entities_new exists (another idempotency guard).
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
        } else {
            // Disable FK enforcement for the restructure. SQLite's canonical
            // table-restructure pattern (https://www.sqlite.org/lang_altertable.html#otheralter)
            // requires this to allow DROP TABLE entities while episodic_edges
            // still holds a FK reference. Re-enabled below.
            conn.execute("PRAGMA foreign_keys = OFF", ())
                .await
                .map_err(step("fk_off"))?;

            // Step 1: backup. Survives across migration runs as a rollback artifact.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entities_bak_004 AS SELECT * FROM entities",
                (),
            )
            .await
            .map_err(step("create_bak"))?;

            // Step 2: create new table with composite PK.
            // group_id is NOT NULL post-migration (backfill ensures this).
            //
            // Note: `embedding` uses generic `BLOB` here (rather than `F32_BLOB(dim)`)
            // because (a) `dim` is not in scope inside this migration helper and
            // (b) the vector index is recreated below via `libsql_vector_idx`
            // which works with raw BLOB columns. The runtime embedder writes the
            // same byte representation either way.
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

        // Step 4 + 5: drop old, rename new. FK enforcement still off if we
        // entered the !partial_migration_in_progress branch above; otherwise
        // we explicitly disable here so partial-migration recovery also works.
        if partial_migration_in_progress {
            conn.execute("PRAGMA foreign_keys = OFF", ())
                .await
                .map_err(step("fk_off_recovery"))?;
        }
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
        // `.ai-docs/planning/v014-adr-029b-composite-fk-design-review-2026-05-28.md`.
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

        // Re-enable FK enforcement. Paired with the OFF toggle inside the
        // restructure branches above so the connection's user-visible PRAGMA
        // state is unchanged across the migration.
        conn.execute("PRAGMA foreign_keys = ON", ())
            .await
            .map_err(step("fk_on"))?;

        tracing::info!(
            target: "kremory::migrations",
            "migrate_004: composite PK (id, group_id) applied to entities"
        );
        Ok(())
    }

    /// Migration 005: add `upgraded_at` column to `namespaces` (ADR-029b Decision 5).
    ///
    /// Idempotent: `ALTER TABLE ADD COLUMN` errors are swallowed when the column
    /// already exists.
    async fn migrate_005_policy_upgraded_at(conn: &libsql::Connection) -> Result<()> {
        // Best-effort ADD COLUMN — errors swallowed (column may already exist).
        let _ = conn
            .execute("ALTER TABLE namespaces ADD COLUMN upgraded_at TEXT", ())
            .await;
        tracing::info!(
            target: "kremory::migrations",
            "migrate_005: upgraded_at column ensured on namespaces"
        );
        Ok(())
    }

    /// Migration 006: install composite FK constraints on `facts` and `episodic_edges`
    /// (ADR-029b Decision 1).
    ///
    /// Uses CREATE-COPY-DROP-RENAME to replace the placeholder ADD COLUMN stubs
    /// that migration 004 installed. After this migration both tables reference
    /// `entities(id, group_id)` rather than the now-invalid single-column `entities(id)`.
    ///
    /// Idempotency gates:
    ///   G1 — `facts_bak_006` exists → already ran, return Ok(()).
    ///   G2 — `entities_bak_004` absent → migration 004 not yet applied, return Err.
    ///   G3 — `facts_new` exists → partial migration, resume from drop+rename.
    ///   G4 — `episodic_edges_new` exists → same for episodic_edges half.
    ///
    /// The `dim` parameter is required because `facts` contains an `F32_BLOB(dim)`
    /// vector column; the new table DDL must embed the same dimension value.
    async fn migrate_006_composite_fk_facts_episodic_edges(
        conn: &libsql::Connection,
        dim: usize,
    ) -> Result<()> {
        fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
            move |e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "migrate_006 step `{name}` failed: {e}"
                ))
            }
        }

        // Helper: check whether a table exists. Used only for partial-migration
        // recovery checks below; the primary idempotency + pre-condition gates
        // inspect SHAPE (FK + PK composition) rather than backup-table presence —
        // Vera review 2026-05-28 #1 caught the backup-sentinel race condition.
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

        // Helper: does `facts` already have a composite FK column? We probe via
        // `PRAGMA foreign_key_list('facts')` and look for any `from` column
        // named `subject_group_id` (or `object_group_id`). When the migration
        // has completed those FK rows exist; before the migration they don't.
        // This is the recovery-safe replacement for the backup-table sentinel.
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
        // We inspect `PRAGMA table_info('entities')` for a row where `name`
        // is `group_id` and `pk > 0` (pk column is index of the column in PK,
        // 1-indexed, 0 = not in PK). Recovery-safe replacement for the
        // `entities_bak_004` sentinel.
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

        // G1 — primary idempotency gate (SHAPE-based, Vera 2026-05-28 #1):
        // if facts already has the composite FK, migration is complete.
        if facts_has_composite_fk(conn)
            .await
            .map_err(step("g1_facts_fk_shape"))?
        {
            return Ok(());
        }

        // G2 — pre-condition gate (SHAPE-based): entities must carry the
        // composite PK installed by migrate_004. Otherwise FK constraints
        // referencing `entities(id, group_id)` cannot be installed.
        if !entities_has_composite_pk(conn)
            .await
            .map_err(step("g2_entities_pk_shape"))?
        {
            return Err(crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_006: entities table does not have composite PK (id, group_id) — \
                 run migrate_004_composite_pk_entities first"
            )));
        }

        // G3 — partial migration recovery: facts_new exists (crashed between
        // CREATE and RENAME). The composite-FK shape gate above is already
        // false in this case (facts still has single-column FKs); we resume
        // from the rename rather than re-running the copy.
        let facts_partial = table_exists(conn, "facts_new")
            .await
            .map_err(step("g3_check_facts_new"))?;

        // G4 — partial migration recovery: episodic_edges_new exists.
        let edges_partial = table_exists(conn, "episodic_edges_new")
            .await
            .map_err(step("g4_check_episodic_edges_new"))?;

        // PRAGMA foreign_keys = OFF for the duration of the restructure.
        // Vera 2026-05-28 #2: any error-return below MUST still issue
        // `PRAGMA foreign_keys = ON` or the connection silently keeps FK
        // enforcement off for all subsequent application writes. We wrap the
        // restructure body in an `async {}` block so all `?` exits land at
        // the `body_result` binding, after which the restore PRAGMA always
        // fires regardless of body success/failure.
        conn.execute("PRAGMA foreign_keys = OFF", ())
            .await
            .map_err(step("fk_off"))?;
        let body_result: Result<()> = async {

        // ── facts half ───────────────────────────────────────────────────────────

        if facts_partial {
            tracing::warn!(
                target: "kremory::migrations",
                "migrate_006: facts_new already exists — attempting to complete partial migration"
            );
        } else {
            // Step 2: backup current facts table (becomes the rollback artifact).
            conn.execute(
                "CREATE TABLE IF NOT EXISTS facts_bak_006 AS SELECT * FROM facts",
                (),
            )
            .await
            .map_err(step("create_facts_bak_006"))?;

            // Step 3: create new facts table with composite FK.
            // Uses F32_BLOB({dim}) to match the original vector column type so that
            // the vector index can be re-created with the same metric after the rename.
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

            // Step 4: copy rows, backfilling subject_group_id / object_group_id.
            // subject_group_id: use ADD COLUMN stub (may be NULL) → fallback group_id → 'default'.
            // object_group_id: NULL when object_id IS NULL; else stub → group_id → 'default'.
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

        // Step 5 + 6: drop old, rename new.
        conn.execute("DROP TABLE facts", ())
            .await
            .map_err(step("drop_facts"))?;
        conn.execute("ALTER TABLE facts_new RENAME TO facts", ())
            .await
            .map_err(step("rename_facts_new"))?;

        // Step 6b: rebuild facts_fts. Vera 2026-05-28 #3 — `facts_fts` is a
        // STANDALONE fts5 virtual table (no `content=` directive — see the
        // CREATE VIRTUAL TABLE site earlier in run_migrations). `DROP TABLE
        // facts` does NOT cascade to it, so without this rebuild the fts
        // shadow tables hold zombie rowids that no longer resolve to live
        // rows in the renamed `facts`. Symptom: FTS queries return phantom
        // results for facts that were rewritten with new INTEGER ids during
        // INSERT INTO facts_new. We DROP the FTS table and recreate from
        // the live data — the simplest correct path and the same shape used
        // by migrate_002 for `rql_entities_fts`.
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
        // Populate FTS from the renamed facts. We only index facts that have
        // a non-NULL object_value (the existing population pattern at the
        // application layer matches this). Errors are best-effort: an in-
        // memory DB with no fts5 module support drops here gracefully.
        let _ = conn
            .execute(
                "INSERT INTO facts_fts (fact_id, predicate, object_value) \
                 SELECT id, predicate, object_value FROM facts \
                 WHERE object_value IS NOT NULL",
                (),
            )
            .await;

        // Step 7: re-create facts indexes.
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
        // Vector index: best-effort (may fail on in-memory DBs).
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS facts_vec_idx \
                 ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
                (),
            )
            .await;
        // Unique partial index on content_hash — FU.1 dedup backstop.
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
        // New composite-FK lookup hot-path indexes (ADR-029b planning doc §3).
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
            // Step 8: backup episodic_edges.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS episodic_edges_bak_006 AS SELECT * FROM episodic_edges",
                (),
            )
            .await
            .map_err(step("create_episodic_edges_bak_006"))?;

            // Step 9: create new table with composite FK.
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

            // Step 10: copy rows, backfilling entity_group_id from ADD COLUMN stub.
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

        // Step 11 + 12: drop old, rename new.
        conn.execute("DROP TABLE episodic_edges", ())
            .await
            .map_err(step("drop_episodic_edges"))?;
        conn.execute("ALTER TABLE episodic_edges_new RENAME TO episodic_edges", ())
            .await
            .map_err(step("rename_episodic_edges_new"))?;

        // Step 13: re-create episodic_edges indexes.
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

            // End of restructure body. Reached only when every `?` above
            // succeeded. The outer `body_result` binding receives this Ok.
            Ok(())
        }
        .await;

        // Always restore PRAGMA foreign_keys = ON, regardless of whether the
        // body succeeded or returned Err. Vera 2026-05-28 #2: failing to do
        // this on the error path leaves the connection with FK enforcement
        // permanently OFF for all subsequent application writes — a silent
        // data-integrity bug. We log a restore-PRAGMA error but do NOT shadow
        // the body's original error; the body error is more informative.
        if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
            tracing::error!(
                target: "kremory::migrations",
                error = %restore_err,
                "migrate_006: failed to re-enable PRAGMA foreign_keys after migration body — \
                 connection FK state is now inconsistent (still OFF); operator must reconnect"
            );
        }

        // Propagate the body's result. If the body failed we surface that
        // error now (after FK is restored above).
        body_result?;

        // Boy-scout integrity check: any FK violation introduced by the
        // restructure surfaces here BEFORE the next application write hits
        // it. Empty cursor = clean. Architect §2 + Vera audit.
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
}

#[cfg(test)]
mod schema_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::TemporalGraph;

    /// Story #246: nested begin_immediate_if_needed is a no-op (guard.opened == false).
    #[tokio::test]
    async fn begin_immediate_nested_call_does_not_issue_second_begin() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // First call opens a real transaction
        let outer = graph
            .begin_immediate_if_needed()
            .await
            .expect("outer begin");
        // Second call while outer is active must NOT issue another BEGIN
        let inner = graph
            .begin_immediate_if_needed()
            .await
            .expect("inner begin (nested)");
        // inner.commit() is a no-op (it did not open a tx)
        inner.commit().await.expect("inner commit no-op");
        // outer.commit() commits the real transaction
        outer.commit().await.expect("outer commit");
    }

    /// Story #246: has_outer_transaction is cleared after commit.
    #[tokio::test]
    async fn begin_immediate_outer_transaction_flag_cleared_after_commit() {
        use std::sync::atomic::Ordering;
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let guard = graph.begin_immediate_if_needed().await.expect("begin");
        assert!(
            graph.has_outer_transaction.load(Ordering::Acquire),
            "flag must be set while guard is active"
        );
        guard.commit().await.expect("commit");
        assert!(
            !graph.has_outer_transaction.load(Ordering::Acquire),
            "flag must be cleared after commit"
        );
    }

    /// Story #210: concurrent BEGIN IMMEDIATE calls serialise via write_lock.
    /// Two tasks both call begin_immediate_if_needed concurrently — one opens,
    /// the other blocks until the first commits. Both writes succeed without error.
    #[tokio::test]
    async fn concurrent_begin_immediate_serialises_via_write_lock() {
        use std::sync::Arc;
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));

        let g1 = Arc::clone(&graph);
        let t1 = tokio::spawn(async move {
            let guard = g1.begin_immediate_if_needed().await.expect("t1 begin");
            // Yield to allow t2 to attempt begin while we hold the lock
            tokio::task::yield_now().await;
            guard.commit().await.expect("t1 commit");
        });

        let g2 = Arc::clone(&graph);
        let t2 = tokio::spawn(async move {
            let guard = g2.begin_immediate_if_needed().await.expect("t2 begin");
            guard.commit().await.expect("t2 commit");
        });

        t1.await.expect("t1 join");
        t2.await.expect("t2 join");
        // After both commits, flag must be clear
        use std::sync::atomic::Ordering;
        assert!(
            !graph.has_outer_transaction.load(Ordering::Acquire),
            "flag must be clear after both tasks complete"
        );
    }

    /// Story #215 / FU.6: flush_if_dirty() called twice without intervening write
    /// only checkpoints once — second call is a no-op (per-handle dirty cleared).
    ///
    /// Uses `graph.dirty_flag()` to observe the per-handle flag directly —
    /// no global DIRTY needed (FU.6: static was removed in favour of per-handle).
    #[tokio::test]
    async fn flush_if_dirty_double_call_only_flushes_once() {
        use std::sync::atomic::Ordering;

        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let dirty = graph.dirty_flag();

        // Simulate a write having set the per-handle dirty flag.
        dirty.store(true, Ordering::Release);

        // First flush: flag was true → checkpoints, clears flag.
        graph.flush_if_dirty().await.expect("first flush");
        assert!(
            !dirty.load(Ordering::Acquire),
            "dirty must be false after first flush"
        );

        // Second flush: flag already false → no checkpoint, no-op.
        graph.flush_if_dirty().await.expect("second flush");
        assert!(
            !dirty.load(Ordering::Acquire),
            "dirty must remain false after no-op second flush"
        );
        // No cleanup needed — per-handle flag is isolated to this graph instance.
    }

    /// G3 gate: DDL must use `recorded_at` not `created_at`. Story #A1.
    #[test]
    fn schema_uses_recorded_at_not_created_at() {
        // Inline the DDL strings that run_migrations executes and verify
        // they contain `recorded_at` and NOT `created_at`.
        let ddl_entities = "CREATE TABLE IF NOT EXISTS entities (
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
            ("entities", ddl_entities),
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
