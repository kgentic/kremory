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

/// Identity of whoever currently owns the open `BEGIN IMMEDIATE` transaction.
///
/// DUR-1 (V1-CANONICAL §4.1) exists because `begin_immediate_if_needed` could not
/// tell **re-entrancy** (the same logical operation nesting, which must be a no-op)
/// from **concurrency** (a different writer, which must wait). Both looked identical:
/// `has_outer_transaction == true`. The concurrent writer was handed a nested no-op
/// guard, wrote into the owner's transaction, and its `commit()` returned `Ok(())`
/// — so an owner rollback destroyed data the caller was told had been committed.
///
/// Task id is the correct key when one exists, because a task keeps its id across
/// `.await` points even if the scheduler migrates it to another worker thread.
/// `try_id()` returns `None` outside a spawned task (notably `Runtime::block_on`,
/// which is what `#[tokio::test]` bodies run on), and there we fall back to the
/// thread id — sound precisely because `block_on` does not migrate. The two
/// variants never compare equal, which is correct: a `block_on` caller and a
/// spawned task are genuinely different writers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WriterId {
    Task(tokio::task::Id),
    Thread(std::thread::ThreadId),
}

impl WriterId {
    fn current() -> Self {
        match tokio::task::try_id() {
            Some(id) => Self::Task(id),
            None => Self::Thread(std::thread::current().id()),
        }
    }
}

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
    /// Type label resolved from `entity_types.name` at query time (via LEFT JOIN).
    ///
    /// Phase 1 (Migration 008) retained the `label` column on `entities` for
    /// backward compat.  Phase 2 (Migration 009) drops the column.  After
    /// Migration 009 this field is populated by LEFT JOIN with `entity_types` on
    /// `(group_id, entity_type_id)`.  Defaults to "Entity" when entity_type_id=0
    /// or the join produces NULL.
    pub label: String,
    /// Integer entity-type id within the owning namespace (group_id).
    ///
    /// Maps to `entity_types(group_id, id)`.  id=0 = "Entity" catch-all sentinel.
    /// Introduced by Migration 008; populated by Phase 2 ingest path.
    /// Use [`EntityTypeRegistry::id_to_name`] to resolve to a display string.
    pub entity_type_id: u32,
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
    /// When this row was recorded in the database (audit timestamp).
    /// Matches the `recorded_at` column on episodes table (SQL DEFAULT
    /// `datetime('now')`). `Option` because pre-G2 episode rows pre-date
    /// the column existing. Added v0.1.6 per Quinn cycle-1 G2 review —
    /// makes the Rust model consistent with the schema (mirrors
    /// `EpisodicEdge.recorded_at` pattern).
    pub recorded_at: Option<String>,
    /// Opaque caller-supplied identifier for the source document/conversation
    /// (e.g. a slug or UUID). Matches the `source_id` column added by
    /// Migration 007. `None` for episodes ingested before the column existed.
    /// TD-003 Phase G — closes struct ↔ table column asymmetry.
    pub source_id: Option<String>,
    /// Optional URI pointing to the original source artifact (URL, file path,
    /// etc.). Matches `source_uri` column added by Migration 007.
    /// TD-003 Phase G — closes struct ↔ table column asymmetry.
    pub source_uri: Option<String>,
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
    /// The write-serialiser, held for the **whole life of the transaction** when
    /// this guard opened it (`None` for a nested guard, which is covered by the
    /// outer guard's lock).
    ///
    /// DUR-1 fix: previously `begin_immediate_if_needed` released the serialiser
    /// when it *returned*, so it only serialised the `BEGIN` statement and not the
    /// transaction. A second writer therefore proceeded immediately, lost the CAS,
    /// and silently joined the first writer's transaction. Holding the lock here
    /// is what makes that second writer **wait** — which is the behaviour
    /// `concurrent_begin_immediate_serialises_via_write_lock` already claimed in
    /// prose while asserting nothing that could detect its absence.
    _lock: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl<'a> BeginGuard<'a> {
    /// Guard for a transaction **this** call opened. Owns the serialiser.
    pub(crate) fn owning(
        graph: &'a TemporalGraph,
        lock: tokio::sync::OwnedMutexGuard<()>,
    ) -> Self {
        Self {
            graph,
            opened: true,
            dispatched: false,
            _lock: Some(lock),
        }
    }

    /// Guard for a re-entrant call nesting under a transaction the **same writer**
    /// already owns. Holds no lock; commit/rollback are no-ops.
    pub(crate) fn nested(graph: &'a TemporalGraph) -> Self {
        Self {
            graph,
            opened: false,
            dispatched: false,
            _lock: None,
        }
    }

    /// `true` when this guard opened the transaction (so `commit()` performs the
    /// real `COMMIT` here); `false` for a nested call whose durable commit belongs
    /// to the outer transaction. Callers use this to place post-commit
    /// observability correctly — a counter summarising a durable write must only
    /// fire once the write is actually committed (never mid-txn).
    pub(crate) fn opened(&self) -> bool {
        self.opened
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
            self.graph.release_transaction_ownership();
        }
        Ok(())
    }

    /// Roll back the transaction. No-op when this guard did not open a transaction.
    pub async fn rollback(mut self) -> Result<()> {
        self.dispatched = true;
        if self.opened {
            let _ = self.graph.conn.execute("ROLLBACK", ()).await;
            self.graph.release_transaction_ownership();
        }
        Ok(())
    }
}

impl Drop for BeginGuard<'_> {
    fn drop(&mut self) {
        // Release ownership on EVERY opened-guard drop, including the paths where
        // `commit()`/`rollback()` returned early via `?` after setting
        // `dispatched = true`. Without this, a failed COMMIT would leave a stale
        // owner recorded, and the next writer — which correctly waits for the
        // serialiser — would then be mis-identified as re-entrant and handed a
        // no-op guard, reintroducing DUR-1 on the error path. Idempotent.
        if self.opened {
            self.graph.release_transaction_ownership();
        }
        if !self.dispatched && self.opened {
            tracing::warn!(
                target: "kremory::db",
                "BeginGuard dropped without explicit commit or rollback — rolling back defensively"
            );
            metrics::counter!("rql.db.begin_guard_drop_without_explicit_commit").increment(1);
            // Real defensive ROLLBACK (fixes a connection-poisoning bug found via
            // the LoCoMo benchmark smoke, 2026-07-17): a `BEGIN IMMEDIATE`
            // transaction stays open at the SQLite engine level until an
            // explicit COMMIT/ROLLBACK executes. Drop cannot `.await`, so this
            // previously ONLY reset the Rust-side `has_outer_transaction` flag
            // and left a comment claiming "the DB connection will auto-rollback
            // on drop/reuse anyway" — false for libsql/SQLite. When an in-flight
            // request is cancelled mid-transaction (e.g. an HTTP client
            // disconnects/times out while kremory-http's `ingest()` is still
            // running), the guard drops, the flag resets to `false`, but the
            // real transaction is STILL OPEN on the connection. The next
            // `begin_immediate_if_needed()` call then sees `false` and issues a
            // fresh `BEGIN IMMEDIATE` on a connection SQLite still considers
            // mid-transaction, failing every subsequent request with
            // `cannot start a transaction within a transaction` until process
            // restart. Fix: spawn a detached task that issues a real `ROLLBACK`
            // against the SAME underlying connection (`libsql::Connection`
            // clones share the same `Arc<dyn Conn>`) and only clears the flag
            // once that rollback completes — closing the race where a new
            // `BEGIN IMMEDIATE` could land before the old transaction is
            // actually closed.
            let conn = self.graph.conn.clone();
            let flag = Arc::clone(&self.graph.has_outer_transaction);
            // DUR-1: MOVE the write-serialiser into the spawned task rather than
            // letting it drop when this function returns. Struct fields drop after
            // the body, so without this the lock would be released while the
            // deferred ROLLBACK is still queued — letting the next writer acquire
            // the serialiser and issue `BEGIN IMMEDIATE` on a connection SQLite
            // still considers mid-transaction, which is the exact
            // "cannot start a transaction within a transaction" poisoning this
            // defensive rollback exists to prevent. Holding the lock until the
            // ROLLBACK completes makes the waiting writer's `BEGIN` safe by
            // construction instead of by timing.
            let held_lock = self._lock.take();
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    handle.spawn(async move {
                        if let Err(e) = conn.execute("ROLLBACK", ()).await {
                            tracing::error!(
                                target: "kremory::db",
                                error = %e,
                                "defensive ROLLBACK after dropped BeginGuard failed — \
                                 connection may be poisoned for subsequent requests"
                            );
                        }
                        flag.store(false, Ordering::Release);
                        // Released only now that the connection is genuinely idle.
                        drop(held_lock);
                    });
                }
                Err(_) => {
                    // No tokio runtime available (e.g. a synchronous Drop outside
                    // any async context) — can't issue the ROLLBACK. Best effort:
                    // clear the flag so we don't deadlock on a stale outer-tx
                    // bookkeeping state, but the underlying SQLite transaction may
                    // still be open; log loudly so it isn't silently swallowed.
                    tracing::error!(
                        target: "kremory::db",
                        "BeginGuard dropped with no tokio runtime available — \
                         cannot issue defensive ROLLBACK; connection may be poisoned"
                    );
                    flag.store(false, Ordering::Release);
                }
            }
        }
    }
}

pub struct TemporalGraph {
    pub(crate) _db: libsql::Database,
    /// Raw libsql connection. `pub` so integration tests compiled with
    /// `features = ["test-utils"]` can issue PRAGMA queries directly.
    /// Not part of the stable public API — use the typed methods instead.
    pub conn: libsql::Connection,
    /// Write-serialiser mutex (ADR-022). Acquired first on every write path.
    /// Prevents concurrent `BEGIN IMMEDIATE` races on a single libsql connection.
    /// Uses `AsyncMutex` (tokio) so the lock can be held across `.await` points.
    pub(crate) write_lock: Arc<AsyncMutex<()>>,
    /// `true` when a `BEGIN IMMEDIATE` is already active on this connection.
    /// Used by `begin_immediate_if_needed` to skip nested BEGIN. Story #246.
    /// `Arc`-wrapped (like `dirty` below) so `BeginGuard::drop`'s spawned
    /// defensive-rollback task can hold its own clone without borrowing
    /// `&'a TemporalGraph` past the guard's lifetime.
    pub(crate) has_outer_transaction: Arc<AtomicBool>,
    /// Which writer owns the currently-open `BEGIN IMMEDIATE`, if any (DUR-1).
    ///
    /// Read on every `begin_immediate_if_needed` to answer the question the old
    /// code could not: *is this call re-entrant, or is it a different writer?*
    /// `std::sync::Mutex` (not async) because the critical section is a single
    /// compare and holds across no `.await`.
    pub(crate) transaction_owner: Arc<std::sync::Mutex<Option<WriterId>>>,
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
            has_outer_transaction: Arc::new(AtomicBool::new(false)),
            transaction_owner: Arc::new(std::sync::Mutex::new(None)),
            dirty: Arc::new(AtomicBool::new(false)),
            embedding_dim,
            policy_cache: NamespacePolicyCache::new(DEFAULT_NS_POLICY_CACHE_CAP),
        };
        graph.run_migrations().await?;
        graph.check_integrity().await?;
        Ok(graph)
    }

    pub async fn open_in_memory() -> Result<Self> {
        let db = libsql::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        let graph = Self {
            _db: db,
            conn,
            write_lock: Arc::new(AsyncMutex::new(())),
            has_outer_transaction: Arc::new(AtomicBool::new(false)),
            transaction_owner: Arc::new(std::sync::Mutex::new(None)),
            dirty: Arc::new(AtomicBool::new(false)),
            embedding_dim: 384,
            policy_cache: NamespacePolicyCache::new(DEFAULT_NS_POLICY_CACHE_CAP),
        };
        graph.run_migrations().await?;
        graph.check_integrity().await?;
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
        let me = WriterId::current();

        // Re-entrancy check FIRST, and it must not touch the serialiser: this
        // writer already holds the lock via its outer guard, so waiting on it
        // here would deadlock against itself.
        if self.owns_open_transaction(me) {
            return Ok(BeginGuard::nested(self));
        }

        // A DIFFERENT writer either holds the transaction or is about to. Wait.
        // The guard returned below keeps this lock for the transaction's whole
        // life, so the next writer blocks here until we COMMIT or ROLLBACK
        // rather than silently joining our transaction (DUR-1).
        let lock = Arc::clone(&self.write_lock).lock_owned().await;

        // Holding the serialiser means no other writer can be mid-transaction,
        // so this is now an unconditional BEGIN rather than a CAS race.
        self.conn.execute("BEGIN IMMEDIATE", ()).await?;
        self.claim_transaction_ownership(me);
        Ok(BeginGuard::owning(self, lock))
    }

    /// `true` when `writer` is the current owner of an open transaction — i.e. this
    /// call is genuine re-entrancy and must nest rather than open a second BEGIN.
    fn owns_open_transaction(&self, writer: WriterId) -> bool {
        let owner = self
            .transaction_owner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *owner == Some(writer)
    }

    /// Record `writer` as the transaction owner and raise the open-transaction flag.
    fn claim_transaction_ownership(&self, writer: WriterId) {
        *self
            .transaction_owner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(writer);
        self.has_outer_transaction.store(true, Ordering::Release);
    }

    /// Clear the transaction owner and lower the open-transaction flag. Idempotent —
    /// called from `commit`, `rollback` and `Drop` so no error path can strand
    /// ownership (see `BeginGuard::drop`).
    pub(crate) fn release_transaction_ownership(&self) {
        *self
            .transaction_owner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.has_outer_transaction.store(false, Ordering::Release);
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

    async fn run_migrations(&self) -> Result<()> {
        // Backward migration (S5.C P1.F1, 2026-05-19): pre-rename dev DBs
        // hold the rql graph table at bare name `entities`. The workspace
        // P1 amend installed a colliding workspace `entities` on the same
        // file. Rename the legacy rql table out of the way before installing
        // the canonical `rql_entities` shape. Idempotent: only renames when
        // a label-shaped legacy table exists AND the new name is free.
        crate::core::migrations::migrate_legacy_rql_entities_table(&self.conn).await?;

        // ADR-029b Migration 002: rename rql_entities → entities.
        // Idempotent: only runs when rql_entities still exists.
        // Must run BEFORE the CREATE TABLE IF NOT EXISTS block below so that
        // the base DDL targets `entities` (not `rql_entities`) on all paths.
        crate::core::migrations::migrate_002_drop_rql_prefix(&self.conn).await?;

        // Migration 024 (TD-117, Vera M2): hard-error on embedding_dim mismatch
        // against a persisted registry row or pre-existing stored vector byte
        // lengths — BEFORE any table below is created/rebuilt using `dim`.
        // See `defs_k.rs` doc comment for the two independent guards.
        crate::core::migrations::migrate_024_verify_embedding_dim(&self.conn, self.embedding_dim)
            .await?;

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
        // Vector index — may legitimately fail on in-memory DBs (non-fatal), but a
        // real (file-backed, correctly-typed) DB failing here silently regressed to
        // brute-force search for months (TD-115) because this result was discarded.
        // Rule 19: capture + count + warn on failure; still non-fatal here (the
        // authoritative, LOUD create lives in migrate_023, which runs once the
        // column is guaranteed to be `F32_BLOB(dim)`).
        if let Err(e) = self.conn.execute(
            "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        ).await {
            metrics::counter!(
                "kremory.search.vector_index_create_failed",
                "table" => "entities",
            )
            .increment(1);
            tracing::warn!(
                target: "kremory::db",
                error = %e,
                table = "entities",
                "kremory.search.vector_index_create_failed — entities_vec_idx did not create; \
                 vector search will fall back to brute-force until TD-115's migrate_023 runs"
            );
        }
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
                    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
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
        // Vector index for fact embeddings — may legitimately fail on in-memory DBs
        // (non-fatal), but TD-115: a real DB whose `embedding` column is a plain
        // `BLOB` (not `F32_BLOB(dim)`) fails here silently too. Rule 19: capture +
        // count + warn. Non-fatal here — migrate_023 is the LOUD authoritative fix.
        if let Err(e) = self.conn.execute(
            "CREATE INDEX IF NOT EXISTS facts_vec_idx ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        ).await {
            metrics::counter!(
                "kremory.search.vector_index_create_failed",
                "table" => "facts",
            )
            .increment(1);
            tracing::warn!(
                target: "kremory::db",
                error = %e,
                table = "facts",
                "kremory.search.vector_index_create_failed — facts_vec_idx did not create; \
                 vector search will fall back to brute-force until TD-115's migrate_023 runs"
            );
        }
        // FU.1: UNIQUE partial index on content_hash — storage-layer backstop for the
        // TOCTTOU-safe check+insert (partial: NULL allowed for pre-hash-rollout rows).
        // TD-133 B2: scoped to ACTIVE (non-expired) rows so the index predicate matches
        // the dedup pre-check SELECT (`graph/facts.rs`: `... AND expired_at IS NULL`).
        // Without this, re-asserting a superseded/expired triple (a legitimate
        // bi-temporal assert→expire→re-assert, ADR-003) collides on the stale expired
        // row → `UNIQUE constraint failed` → the re-assertion is silently lost. Existing
        // DBs are migrated by `migrate_025_fact_dedup_expired_partial` (runs last).
        self.conn
            .execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_content_hash_unique \
                 ON facts(content_hash) WHERE content_hash IS NOT NULL AND expired_at IS NULL",
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
        // Idempotent: PRAGMA-shape gate — skips when group_id is already in the PK.
        crate::core::migrations::migrate_004_composite_pk_entities(&self.conn).await?;

        // ADR-029b Migration 005: add upgraded_at to namespaces.
        // Idempotent: swallows duplicate-column error.
        crate::core::migrations::migrate_005_policy_upgraded_at(&self.conn).await?;

        // ADR-029b Migration 006: composite FK on facts + episodic_edges.
        // Replaces the placeholder ADD COLUMN stubs in migration 004.
        // Must run AFTER migrate_004 (pre-condition gate inside).
        // Idempotent: skips when facts_bak_006 already present.
        crate::core::migrations::migrate_006_composite_fk_facts_episodic_edges(
            &self.conn,
            self.embedding_dim,
        )
        .await?;

        // v0.1.6 G1 Migration 007: source_id + source_uri columns on episodes.
        // Additive ALTER TABLE — no restructure, no FK changes.
        // Idempotent: PRAGMA table_info gate skips when both columns already present.
        crate::core::migrations::migrate_007_source_id_source_uri(&self.conn).await?;

        // TD-013 Migration 008: entity_types registry + entity_type_id on entities.
        // Creates entity_types table, seeds per-group_id catch-all (id=0 "Entity")
        // + observed labels, adds entity_type_id column backfilled from label.
        // Idempotent: PRAGMA table_info gate (G1) + IF NOT EXISTS + INSERT OR IGNORE.
        crate::core::migrations::migrate_008_entity_types(&self.conn).await?;

        // TD-013 Migration 009 (Phase 2): DROP entities.label column.
        // All INSERT/SELECT paths now use entity_type_id + LEFT JOIN entity_types
        // to resolve the label at query time.  Must run AFTER migrate_008 which
        // guarantees entity_type_id is populated.
        // Idempotent: PRAGMA table_info gate — skips when label column is already absent.
        crate::core::migrations::migrate_009_drop_label_column(&self.conn).await?;

        // TD-013 Migration 010: seed default entity_types vocabulary per group_id.
        // Backfills Migration 008's gap: 008 only seeded id=0 for groups with
        // pre-existing entities. 010 ensures every observed group_id has the
        // full default OntoNotes vocabulary (Entity+Person+Organisation+Location
        // +Date+Time+Money+Quantity+Event+Concept). Without this, fresh DBs
        // produce empty L2 prompts and LLM returns zero entities. The override
        // mechanism (SourceParams.entity_types_override) augments these defaults.
        // Idempotent: ensure_default_types_seeded no-ops when group_id already has rows.
        crate::core::migrations::migrate_010_default_entity_types(&self.conn).await?;

        // TD-003 Migration 011 (Phase G, ADR-042): add content_hash column to
        // episodes with SHA-256 backfill. Idempotent: PRAGMA table_info gate
        // skips the ALTER TABLE when the column already exists.
        crate::core::migrations::migrate_011_episodes_content_hash(&self.conn).await?;

        // ADR-045 §2 Migration 012 (v0.1.1): add entity_type_source / entity_type_assigned_at /
        // ner_confidence columns to `entities`; create v_entity_drift_candidates view;
        // backfill legacy NULL rows to 'Phase1Ner'. Idempotent: PRAGMA table_info guards.
        crate::core::migrations::migrate_012_source_tier_columns(&self.conn).await?;

        // ADR-037 §9.5 Migration 014 (v0.1.1, Dream Pass 0): add provenance columns
        // to `entity_types` — discovered_at / discovered_by / evidence_count / confidence.
        // Backfills pre-existing seed types with discovered_by = 'seed'.
        // Idempotent: PRAGMA table_info gate per column.
        crate::core::migrations::migrate_014_entity_types_provenance(&self.conn).await?;

        // ADR-047 Migration 013 (v0.1.2, Phase B): extend entity_type_source CHECK with
        // 'DreamPass4' + create dream_pass4_audit table.
        // Idempotent: sqlite_master DDL gate (checks for 'DreamPass4' in entities DDL).
        crate::core::migrations::migrate_013_pass4_source_tier(&self.conn).await?;

        // ADR-051 Migration 015a (v0.2.2, Phase 1): add episode_processing_status column
        // to episodes for the ADR-051 async extraction gate state machine.
        // Values: Pending → Extracting → Verified | Failed.
        // Backfill: episodes with episodic_edges are already Verified (R-02 mitigation).
        // Idempotent: PRAGMA table_info gate + IF NOT EXISTS index.
        crate::core::migrations::migrate_015a_episode_processing_status(&self.conn).await?;

        // ADR-050 Migration 016 (v0.2.4, Phase 1): crash-safety schema cluster.
        // Adds dream_idempotency_keys, op_checkpoints, dream_pass_budget_usage tables
        // and is_dream_generated columns on entities + facts.
        // Idempotent: CREATE TABLE/INDEX IF NOT EXISTS + PRAGMA-guarded ADD COLUMN.
        // Emergency downgrade: migrate_015b_downgrade_crash_safety_schema (not called here).
        crate::core::migrations::migrate_016_crash_safety_schema(&self.conn).await?;

        // Migration 017: presence-uniqueness invariant on episodic_edges —
        // ≤1 row per (episode_id, entity_id, entity_group_id). Dedups any
        // pre-existing duplicate presence edges (kremory ≤ 0.3.1 had no such
        // constraint) then installs a UNIQUE index. Idempotent: DELETE is a no-op
        // on a clean db + CREATE UNIQUE INDEX IF NOT EXISTS is re-runnable.
        crate::core::migrations::migrate_017_episodic_edges_presence_unique(&self.conn).await?;

        // Migration 018 (ADR-063 spec §5.0 + §5.1): identity-verdict / write-gate
        // prerequisites — `idx_facts_subject` (subject-side index mirroring
        // idx_facts_object; needed for Site #5's cooccurs_in_graph query, resolves
        // RISK-002) + the `identity_verdict_audit` table (Site #5/#3 adjudication
        // audit trail). Idempotent: CREATE INDEX/TABLE IF NOT EXISTS.
        crate::core::migrations::migrate_018_identity_verdict_prereqs(&self.conn).await?;

        // Migration 019 (ADR-066 spec §2): dream CONSOLIDATION sub-phase substrate
        // — `facts_archive` (append-only P2 archive history), `entity_communities`
        // + `community_summaries` (P4 deterministic label-propagation membership).
        // OPTIONAL feature tables: intentionally NOT in `CRITICAL_TABLES` — absence
        // must degrade gracefully (op finds no rows), not raise CorruptStore.
        // Idempotent: CREATE TABLE/INDEX IF NOT EXISTS.
        crate::core::migrations::migrate_019_consolidation_substrate(&self.conn).await?;

        // Migration 020 (ADR-067 V1): facts.corroboration_inert — provenance-anchored
        // corroboration column, the convergence fix for cross_episode_merges (P3).
        // Additive, PRAGMA-guarded, mirrors migration 016's is_dream_generated idiom.
        // Idempotent: PRAGMA-guarded ADD COLUMN.
        crate::core::migrations::migrate_020_facts_corroboration_inert(&self.conn).await?;

        // Migration 021 (reversible-graph-mutations arch-spec §2.1/§2.2/§8.4):
        // provenance + anti-re-merge substrate — `graph_mutation_log` (kind-tagged
        // in-txn snapshot log) + `merge_nogood` (sorted-pair anti-re-merge marker).
        // FOUNDATION ONLY: schema install; snapshot capture + undo replay are wired
        // in later sub-phases. Idempotent: CREATE TABLE/INDEX IF NOT EXISTS.
        crate::core::migrations::migrate_021_graph_mutation_log(&self.conn).await?;

        // Migration 022 (ADR-072 seq1 impl-spec §1): `episodes_fts` — BM25/FTS5
        // content-search substrate over raw `episodes.content`. Feature-gated:
        // only compiled + called behind `content-search`; default build is
        // unaffected (two-lever gating model, ADR-072 §6 Finding 3).
        // Idempotent: CREATE VIRTUAL TABLE IF NOT EXISTS + guarded backfill.
        #[cfg(feature = "content-search")]
        crate::core::migrations::migrate_022_episodes_content_recall(&self.conn).await?;

        // Migration 023 (TD-115): rebuild entities/facts with a real
        // `embedding F32_BLOB(dim)` column when the current declared type is
        // a generic `BLOB` — the DiskANN vector index rejects BLOB columns,
        // so every migrated DB has been silently falling back to
        // vector_search_brute_force. Detection-gated per table (no-op when
        // already F32_BLOB); the vector-index CREATE is LOUD (propagates
        // Err), unlike every prior migration's best-effort index create.
        crate::core::migrations::migrate_023_vector_index_column_type(&self.conn, dim).await?;

        // Migration 025 (TD-133 B2): make idx_facts_content_hash_unique a partial index
        // over ACTIVE (non-expired) rows only, matching the dedup SELECT predicate.
        // MUST run LAST — after migrate_023 (defs_j) recreates the old-form index —
        // else the un-partitioned form would be re-introduced. Idempotent + conditional.
        crate::core::migrations::migrate_025_fact_dedup_expired_partial(&self.conn).await?;

        // Migration 026 (TD-136): dense episode retrieval substrate — add an
        // `embedding F32_BLOB(dim)` column + `episodes_vec_idx` (libsql cosine
        // vector index) to `episodes`, mirroring `entities`/`facts`. Feature-
        // gated behind `content-search` (same two-lever gate as Migration 022's
        // `episodes_fts`) — the default build never touches `episodes` and stays
        // byte-identical. Idempotent: PRAGMA table_info column gate + CREATE
        // INDEX IF NOT EXISTS. Runs AFTER migrate_023 (whose F32_BLOB rebuild
        // is unrelated to episodes) so `dim` is the same authoritative value.
        #[cfg(feature = "content-search")]
        crate::core::migrations::migrate_026_episodes_embedding(&self.conn, dim).await?;

        Ok(())
    }

    /// Verify that the open store contains all critical tables required for
    /// operation.
    ///
    /// # Integrity contract (T1.8, v0.2.0 Phase B-prep)
    ///
    /// Called by `open_with_dim` and `open_in_memory` **after** `run_migrations`
    /// succeeds. If any check fails the method returns
    /// [`crate::core::error::Error::CorruptStore`] and the caller receives `Err`
    /// before the `TemporalGraph` is returned — refusing-to-operate is always
    /// safer than silently degrading on a corrupt schema.
    ///
    /// ## What is checked
    ///
    /// Critical tables: `episodes`, `entities`, `entity_types`, `facts`.
    /// These four tables are the minimum required for any kremory operation
    /// (ingest, recall, dream). Loss of any one of them indicates a truncated
    /// migration run, a manual `DROP TABLE`, or filesystem corruption.
    ///
    /// ## What is NOT checked (and why)
    ///
    /// - **`PRAGMA user_version`** — `TemporalGraph::run_migrations` does not
    ///   write `user_version` (only `MigrationRunner::bump_version` does, via a
    ///   separate path). Checking it here would always read 0 on a fresh DB and
    ///   produce false positives.
    /// - **Column-level schema** — deferred; the migration suite's PRAGMA-gated
    ///   idempotency already provides that guarantee.
    /// - **Forward / backward version compat** — both directions refused until an
    ///   explicit ADR relaxes them. Strict table-presence semantics for now.
    ///
    /// ## Observability (Rule 19)
    ///
    /// Emits `kremory.store.integrity_check_total{outcome="ok"|"corrupt"}`.
    /// On corruption emits `tracing::error!` naming the specific missing table.
    async fn check_integrity(&self) -> Result<()> {
        // Quinn LOW-01 fix: extended from 4 → 6 tables to include `episodic_edges`
        // (load-bearing for graph traversal — episodes are joined to entities via this
        // edge table) and `dream_pass4_audit` (ADR-049 audit trail; consistency_check
        // writes here on every correction). A missing `episodic_edges` would silently
        // produce empty disambiguation candidate sets; a missing `dream_pass4_audit`
        // would break verify_batch on first Pass 4 correction.
        //
        // Note on `*_bak_*` survivors: tables like `entities_bak_004`, `facts_bak_006`,
        // `episodes_bak_015a` are intentional pre-ALTER row-level backups left by
        // up-migrations as recovery surface. They are NOT in `CRITICAL_TABLES`
        // (presence-check semantics), and check_integrity does NOT flag their
        // presence in sqlite_master as anomalous. Schema auditors querying
        // sqlite_master directly should expect to see them.
        const CRITICAL_TABLES: &[&str] = &[
            "episodes",
            "entities",
            "entity_types",
            "facts",
            "episodic_edges",
            "dream_pass4_audit",
        ];

        for table in CRITICAL_TABLES {
            let mut rows = self
                .conn
                .query(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1",
                    libsql::params![*table],
                )
                .await?;

            if rows.next().await?.is_none() {
                let reason = format!("missing table: {table}");
                tracing::error!(
                    target: "kremory.store.integrity",
                    missing_table = %table,
                    "store integrity check failed — {reason}"
                );
                metrics::counter!(
                    "kremory.store.integrity_check_total",
                    "outcome" => "corrupt"
                )
                .increment(1);
                return Err(crate::core::error::Error::CorruptStore { reason });
            }
        }

        metrics::counter!(
            "kremory.store.integrity_check_total",
            "outcome" => "ok"
        )
        .increment(1);
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

    /// Invoke `migrate_023_vector_index_column_type` in isolation against
    /// this already-open handle.
    ///
    /// `migrate_023_vector_index_column_type` is `pub(crate)` (it is called
    /// exactly once, internally, from `run_migrations`) — this wrapper is
    /// the only way an integration test can drive it directly, which is
    /// required for the crash-window regression tests
    /// (`.ai-docs/architecture-review/vera-adr-074-migrate-023-review-2026-07-14.md`
    /// H3/H4): those tests manually construct a mid-crash DB state (e.g. a
    /// populated `facts_bak_023` + an empty `facts_new_023` + a dropped live
    /// `facts` table) and must re-invoke the migration on exactly that
    /// state, not on a fresh `open_with_dim`.
    ///
    /// Only available in `test` builds and when the `test-utils` feature is
    /// enabled. Never ship this in a production binary.
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn run_migrate_023_for_test(&self) -> Result<()> {
        crate::core::migrations::migrate_023_vector_index_column_type(
            &self.conn,
            self.embedding_dim,
        )
        .await
    }

    /// Return a shared reference to the per-handle dirty flag.
    ///
    /// For use in tests that need to observe or arm the flag without going
    /// through a full write cycle. Only available in test/test-utils context.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn dirty_flag(&self) -> &Arc<AtomicBool> {
        &self.dirty
    }

    /// Return a shared reference to the underlying libsql connection.
    ///
    /// For use in integration tests that need direct PRAGMA / DDL access
    /// without going through the facade builders. Only available in
    /// test/test-utils context. Never ship this in a production binary.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn conn_for_test(&self) -> &libsql::Connection {
        &self.conn
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
}

#[cfg(test)]
mod schema_tests {
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

    /// Story #210: concurrent `BEGIN IMMEDIATE` calls serialise via `write_lock` —
    /// one opens, the other **blocks until the first commits**.
    ///
    /// REWRITTEN 2026-08-03 (V1-CANONICAL §3.2). The previous version of this test
    /// asserted only that `has_outer_transaction` was clear once both tasks had
    /// finished. That is true whether or not serialisation happens — a flag set and
    /// cleared by the *winner* alone reads identically when the loser silently
    /// joined the winner's transaction. It therefore documented blocking that did
    /// not occur, and passed for as long as the DUR-1 defect existed.
    ///
    /// This version measures the property the doc comment claims: the **maximum
    /// number of writers simultaneously inside a transaction**. Serialised ⇒ 1.
    /// Verified sensitive in both directions — restoring the old
    /// `has_outer_transaction`-only re-entrancy check makes this test fail with
    /// `max_concurrent == 2`.
    #[tokio::test]
    async fn concurrent_begin_immediate_serialises_via_write_lock() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let inside = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let spawn_writer = |graph: Arc<TemporalGraph>,
                            inside: Arc<AtomicUsize>,
                            max_concurrent: Arc<AtomicUsize>,
                            label: &'static str| {
            tokio::spawn(async move {
                let guard = graph
                    .begin_immediate_if_needed()
                    .await
                    .unwrap_or_else(|e| panic!("{label} begin: {e}"));

                let now_inside = inside.fetch_add(1, Ordering::AcqRel) + 1;
                max_concurrent.fetch_max(now_inside, Ordering::AcqRel);
                // Hand the scheduler a chance to run the other writer while this
                // transaction is open — the interleaving the old test invited and
                // then failed to observe.
                tokio::task::yield_now().await;
                inside.fetch_sub(1, Ordering::AcqRel);

                guard
                    .commit()
                    .await
                    .unwrap_or_else(|e| panic!("{label} commit: {e}"));
            })
        };

        let t1 = spawn_writer(
            Arc::clone(&graph),
            Arc::clone(&inside),
            Arc::clone(&max_concurrent),
            "t1",
        );
        let t2 = spawn_writer(
            Arc::clone(&graph),
            Arc::clone(&inside),
            Arc::clone(&max_concurrent),
            "t2",
        );

        t1.await.expect("t1 join");
        t2.await.expect("t2 join");

        assert_eq!(
            max_concurrent.load(Ordering::Acquire),
            1,
            "two writers were inside a transaction at once — the write_lock is \
             serialising the BEGIN statement but not the transaction (DUR-1)"
        );
        assert!(
            !graph.has_outer_transaction.load(Ordering::Acquire),
            "flag must be clear after both tasks complete"
        );
    }

    /// DUR-1 (V1-CANONICAL §4.1): two CONCURRENT writers on one `TemporalGraph`
    /// must not silently share a transaction.
    ///
    /// Today the loser of the `has_outer_transaction` CAS receives a *nested*
    /// no-op guard — the same guard shape used for legitimate re-entrancy — so it
    /// writes into the winner's transaction and its `commit()` returns `Ok(())`
    /// having committed nothing of its own. When the winner then rolls back, the
    /// loser's rows are destroyed **after it was told it succeeded**.
    ///
    /// This test writes real rows inside both transactions, which is precisely
    /// what `concurrent_begin_immediate_serialises_via_write_lock` fails to do —
    /// that test observes only the bookkeeping flag and therefore cannot see the
    /// loss (V1-CANONICAL §3.2, "guards that pass while asserting properties the
    /// code lacks").
    ///
    /// FAILS before the fix: t2's row count is 0 because t1's ROLLBACK took it.
    #[tokio::test]
    async fn concurrent_writers_do_not_share_a_transaction() {
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::sync::Notify;

        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        graph
            .conn
            .execute(
                "CREATE TABLE IF NOT EXISTS dur1_probe (who TEXT NOT NULL)",
                (),
            )
            .await
            .expect("create probe table");

        // t1 has opened its transaction and written its row.
        let t1_holding = Arc::new(Notify::new());
        // t2 believes it has durably committed.
        let t2_committed = Arc::new(Notify::new());

        let g1 = Arc::clone(&graph);
        let h1 = Arc::clone(&t1_holding);
        let c1 = Arc::clone(&t2_committed);
        let t1 = tokio::spawn(async move {
            let guard = g1.begin_immediate_if_needed().await.expect("t1 begin");
            g1.conn
                .execute("INSERT INTO dur1_probe (who) VALUES ('t1')", ())
                .await
                .expect("t1 insert");
            h1.notify_one();
            // Give t2 a bounded window to interleave. Once the fix lands t2
            // blocks here instead, so this timeout expiring is the healthy path.
            let _ = tokio::time::timeout(Duration::from_millis(250), c1.notified()).await;
            // t1 abandons its work — a rollback must destroy ONLY t1's row.
            guard.rollback().await.expect("t1 rollback");
        });

        let g2 = Arc::clone(&graph);
        let h2 = Arc::clone(&t1_holding);
        let c2 = Arc::clone(&t2_committed);
        let t2 = tokio::spawn(async move {
            h2.notified().await;
            let guard = g2.begin_immediate_if_needed().await.expect("t2 begin");
            g2.conn
                .execute("INSERT INTO dur1_probe (who) VALUES ('t2')", ())
                .await
                .expect("t2 insert");
            // Returns Ok(()) today even though it committed nothing of its own.
            guard.commit().await.expect("t2 commit");
            c2.notify_one();
        });

        t1.await.expect("t1 join");
        t2.await.expect("t2 join");

        let mut rows = graph
            .conn
            .query("SELECT COUNT(*) FROM dur1_probe WHERE who = 't2'", ())
            .await
            .expect("count t2 rows");
        let row = rows.next().await.expect("row result").expect("one row");
        let t2_rows: i64 = row.get(0).expect("count column");

        assert_eq!(
            t2_rows, 1,
            "DUR-1: t2's commit() returned Ok(()) so its row MUST be durable, but \
             t1's rollback destroyed it — the two writers shared one transaction"
        );

        let mut rows = graph
            .conn
            .query("SELECT COUNT(*) FROM dur1_probe WHERE who = 't1'", ())
            .await
            .expect("count t1 rows");
        let row = rows.next().await.expect("row result").expect("one row");
        let t1_rows: i64 = row.get(0).expect("count column");
        assert_eq!(
            t1_rows, 0,
            "t1 rolled back, so its own row must be gone — if this fires the \
             rollback itself is broken, not the isolation"
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

    // ── migrate_006 test suite (D3, ADR-029b planning doc §7) ──────────────────

    // Helper: open a raw in-memory libsql connection (no TemporalGraph scaffolding).
    // Used for tests that need to call migration fns directly without running
    // the full schema bootstrap.
    async fn raw_in_memory_conn() -> libsql::Connection {
        libsql::Builder::new_local(":memory:")
            .build()
            .await
            .expect("in-memory db build")
            .connect()
            .expect("connect")
    }

    /// Verify PRAGMA foreign_key_list('facts') has composite FK to entities(id, group_id).
    /// Also verifies episodic_edges has composite FK for (entity_id, entity_group_id).
    #[tokio::test]
    async fn migrate_006_fresh_db_facts_has_composite_fk() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        // Collect the FK 'from' column names for facts.
        let mut rows = conn
            .query("PRAGMA foreign_key_list('facts')", ())
            .await
            .expect("pragma fk_list facts");
        let mut fk_from_cols: Vec<String> = Vec::new();
        while let Some(r) = rows.next().await.expect("next") {
            let from: String = r.get(3).expect("col from");
            let to_col: String = r.get(5).expect("col to");
            fk_from_cols.push(format!("{from}->{to_col}"));
        }
        assert!(
            fk_from_cols.iter().any(|s| s.contains("subject_group_id")),
            "facts must have composite FK column subject_group_id; got: {fk_from_cols:?}"
        );
        assert!(
            fk_from_cols.iter().any(|s| s.contains("object_group_id")),
            "facts must have composite FK column object_group_id; got: {fk_from_cols:?}"
        );

        // Verify episodic_edges has composite FK for entity_group_id.
        let mut rows2 = conn
            .query("PRAGMA foreign_key_list('episodic_edges')", ())
            .await
            .expect("pragma fk_list episodic_edges");
        let mut ee_fk_from: Vec<String> = Vec::new();
        while let Some(r) = rows2.next().await.expect("next") {
            let from: String = r.get(3).expect("col from");
            ee_fk_from.push(from);
        }
        assert!(
            ee_fk_from.iter().any(|s| s == "entity_group_id"),
            "episodic_edges must have composite FK column entity_group_id; got: {ee_fk_from:?}"
        );
    }

    /// Verify all expected indexes on facts and episodic_edges exist after migration.
    #[tokio::test]
    async fn migrate_006_fresh_db_indexes_recreated() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        let required_facts_indexes = [
            "idx_facts_temporal",
            "idx_facts_predicate",
            "idx_facts_object",
            "idx_facts_group",
            "idx_facts_content_hash_unique",
            "idx_facts_content_hash",
        ];
        let required_edges_indexes = ["idx_episodic_edges_entity", "idx_episodic_edges_episode"];

        for idx_name in required_facts_indexes {
            let mut rows = conn
                .query(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?1",
                    libsql::params![idx_name],
                )
                .await
                .expect("sqlite_master query");
            assert!(
                rows.next().await.expect("next").is_some(),
                "expected facts index `{idx_name}` to exist after migration"
            );
        }
        for idx_name in required_edges_indexes {
            let mut rows = conn
                .query(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?1",
                    libsql::params![idx_name],
                )
                .await
                .expect("sqlite_master query");
            assert!(
                rows.next().await.expect("next").is_some(),
                "expected episodic_edges index `{idx_name}` to exist after migration"
            );
        }
    }

    /// Verify facts_fts and entities_fts tables exist and FTS is functional
    /// after migration 006.
    #[tokio::test]
    async fn migrate_006_fresh_db_fts_tables_intact() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        // Both FTS virtual tables must exist.
        for tbl in ["facts_fts", "entities_fts"] {
            let mut rows = conn
                .query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                    libsql::params![tbl],
                )
                .await
                .expect("sqlite_master");
            assert!(
                rows.next().await.expect("next").is_some(),
                "expected `{tbl}` to exist after migration"
            );
        }

        // Insert an entity + fact with object_value = 'hello', then verify FTS query.
        // Migration 009 dropped entities.label — insert without it.
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES ('ent1', 0, '2026-01-01T00:00:00Z', 'default')",
            (),
        )
        .await
        .expect("insert entity");

        conn.execute(
            "INSERT INTO facts \
             (subject_id, subject_group_id, predicate, valid_from, recorded_at, group_id, object_value) \
             VALUES ('ent1', 'default', 'says', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'default', 'hello')",
            (),
        )
        .await
        .expect("insert fact");

        conn.execute(
            "INSERT INTO facts_fts (fact_id, predicate, object_value) \
             SELECT id, predicate, object_value FROM facts WHERE object_value IS NOT NULL",
            (),
        )
        .await
        .expect("populate fts");

        let mut fts_rows = conn
            .query(
                "SELECT fact_id FROM facts_fts WHERE object_value MATCH 'hello'",
                (),
            )
            .await
            .expect("fts query");
        assert!(
            fts_rows.next().await.expect("fts next").is_some(),
            "FTS query for 'hello' must return at least 1 result"
        );
    }

    /// Running run_migrations twice must be a no-op (G5 idempotency).
    /// Specifically for migrate_006: facts_bak_006 must still exist and
    /// facts schema must be unchanged on the second run.
    #[tokio::test]
    async fn migrate_006_idempotent_on_already_migrated_db() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // Second run must not error.
        graph
            .run_migrations_again_for_test()
            .await
            .expect("second run must be no-op");

        let conn = &graph.conn;

        // facts_bak_006 must still exist (not dropped by second run).
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_bak_006'",
                (),
            )
            .await
            .expect("sqlite_master");
        assert!(
            rows.next().await.expect("next").is_some(),
            "facts_bak_006 must persist after second migration run"
        );

        // facts table must still have composite FK (schema unchanged).
        let mut fk_rows = conn
            .query("PRAGMA foreign_key_list('facts')", ())
            .await
            .expect("pragma fk_list");
        let mut found_composite = false;
        while let Some(r) = fk_rows.next().await.expect("next") {
            let from: String = r.get(3).expect("col from");
            if from == "subject_group_id" {
                found_composite = true;
                break;
            }
        }
        assert!(
            found_composite,
            "facts must still have composite FK after second migration run"
        );
    }

    /// Migration 020 (ADR-067 V1): `facts.corroboration_inert` must be present on a
    /// FRESH in-memory DB (open runs the full migration chain including 020) AND
    /// survive a re-run of the whole chain unchanged (idempotent PRAGMA-guarded ADD
    /// COLUMN — impl-spec §C0 DoD: "a fresh in-memory DB + a migrated-from-prior DB
    /// both end with the column present").
    #[tokio::test]
    async fn migrate_020_facts_corroboration_inert_present_and_idempotent() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // Fresh DB: column present with default 0.
        let mut rows = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM pragma_table_info('facts') WHERE name = 'corroboration_inert'",
                (),
            )
            .await
            .expect("pragma_table_info query");
        let count: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("present")
            .get(0)
            .expect("count col");
        assert_eq!(
            count, 1,
            "facts.corroboration_inert must be present on a fresh in-memory DB"
        );

        // Re-run the whole migration chain (idempotent PRAGMA-guarded ADD COLUMN) —
        // must not error and the column must still be present exactly once.
        graph
            .run_migrations_again_for_test()
            .await
            .expect("second run must be no-op");

        let mut rows2 = graph
            .conn
            .query(
                "SELECT COUNT(*) FROM pragma_table_info('facts') WHERE name = 'corroboration_inert'",
                (),
            )
            .await
            .expect("pragma_table_info query (2nd)");
        let count2: i64 = rows2
            .next()
            .await
            .expect("row")
            .expect("present")
            .get(0)
            .expect("count col");
        assert_eq!(
            count2, 1,
            "facts.corroboration_inert must remain present exactly once after re-run \
             (idempotent — no duplicate-column error)"
        );
    }

    /// When facts_bak_006 is present, calling migrate_006 directly must return
    /// Ok(()) immediately (G1 shape gate) without touching the manually inserted row.
    #[tokio::test]
    async fn migrate_006_skips_when_bak_table_present() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        // Confirm migration already ran (facts_bak_006 exists).
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_bak_006'",
                (),
            )
            .await
            .expect("sqlite_master");
        assert!(
            rows.next().await.expect("next").is_some(),
            "pre-condition: facts_bak_006 must exist after open_in_memory"
        );

        // Insert a sentinel row. Migration 009 dropped entities.label — omit it.
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES ('skip_ent', 0, '2026-01-01T00:00:00Z', 'default')",
            (),
        )
        .await
        .expect("insert entity");
        conn.execute(
            "INSERT INTO facts (subject_id, subject_group_id, predicate, valid_from, recorded_at, group_id) \
             VALUES ('skip_ent', 'default', 'skipped', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'default')",
            (),
        )
        .await
        .expect("insert sentinel fact");

        // Call migrate_006 directly — must be a no-op (G1 gate fires based on shape).
        crate::core::migrations::migrate_006_composite_fk_facts_episodic_edges(
            conn,
            graph.embedding_dim,
        )
        .await
        .expect("direct migrate_006 call must return Ok");

        // Sentinel row must still be present.
        let mut rows = conn
            .query("SELECT id FROM facts WHERE predicate = 'skipped'", ())
            .await
            .expect("select sentinel");
        assert!(
            rows.next().await.expect("next").is_some(),
            "sentinel fact must still exist after idempotent migrate_006 call"
        );
    }

    /// Calling migrate_006 on a DB without migrate_004 applied must return
    /// Err containing "migrate_004 not yet applied" (G2 pre-condition gate).
    #[tokio::test]
    async fn migrate_006_errors_without_migration_004() {
        // Raw connection: no TemporalGraph, no migration suite.
        let conn = raw_in_memory_conn().await;

        // Minimal schema: entities with single-column PK (pre-004 shape).
        conn.execute_batch(
            // IF NOT EXISTS guards satisfy the no-bare-create meta-test
            // (migrations.rs:28-31 idempotency invariant). Test runs on fresh
            // in-memory connection so guards are no-ops; pre-004 schema shape
            // simulation intent preserved.
            "CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, label TEXT NOT NULL, recorded_at TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, timestamp TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS facts (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 subject_id TEXT NOT NULL,
                 predicate TEXT NOT NULL,
                 valid_from TEXT NOT NULL,
                 recorded_at TEXT NOT NULL,
                 group_id TEXT
             );
             CREATE TABLE IF NOT EXISTS episodic_edges (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 episode_id INTEGER NOT NULL,
                 entity_id TEXT NOT NULL,
                 role TEXT NOT NULL DEFAULT 'mentioned',
                 recorded_at TEXT NOT NULL
             );",
        )
        .await
        .expect("create minimal schema");

        // migrate_006 must fail with G2 error (entities lacks composite PK).
        let result =
            crate::core::migrations::migrate_006_composite_fk_facts_episodic_edges(&conn, 384)
                .await;
        assert!(
            result.is_err(),
            "must return Err when migrate_004 not applied"
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("migrate_004") || msg.contains("composite PK"),
            "error must mention migrate_004 or composite PK; got: {msg}"
        );
    }

    /// After migration 006, INSERT INTO facts with a non-existent (subject_id, subject_group_id)
    /// must return an FK constraint violation error.
    #[tokio::test]
    async fn migrate_006_composite_fk_enforced_post_migration() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        // Enable FK enforcement (may be OFF after migration body restore).
        conn.execute("PRAGMA foreign_keys = ON", ())
            .await
            .expect("fk on");

        // INSERT a fact referencing a non-existent entity — must fail.
        let result = conn
            .execute(
                "INSERT INTO facts (subject_id, subject_group_id, predicate, valid_from, recorded_at, group_id) \
                 VALUES ('nonexistent_entity', 'ns1', 'knows', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'ns1')",
                (),
            )
            .await;
        assert!(
            result.is_err(),
            "INSERT with non-existent (subject_id, subject_group_id) must violate composite FK"
        );
        let err_msg = format!("{}", result.unwrap_err());
        // libsql 0.9.30 emits: "FOREIGN KEY constraint failed" (SQLite
        // SQLITE_CONSTRAINT_FOREIGNKEY, extended code 787). The `.contains`
        // heuristic catches both the standard message and any future libsql
        // rephrasing. If this assertion fails, check the actual `err_msg` —
        // libsql may have changed the error text in a newer version.
        assert!(
            err_msg.to_lowercase().contains("foreign key")
                || err_msg.to_lowercase().contains("constraint"),
            "error must be an FK constraint violation; got: {err_msg}"
        );
    }

    /// After migration 006, INSERT INTO facts referencing an existing entity must succeed.
    #[tokio::test]
    async fn migrate_006_valid_composite_fk_insert_succeeds() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        conn.execute("PRAGMA foreign_keys = ON", ())
            .await
            .expect("fk on");

        // INSERT the entity first. Migration 009 dropped entities.label — omit it.
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
             VALUES ('alice', 0, '2026-01-01T00:00:00Z', 'ns1')",
            (),
        )
        .await
        .expect("insert entity alice");

        // INSERT fact referencing (alice, ns1) — must succeed.
        conn.execute(
            "INSERT INTO facts (subject_id, subject_group_id, predicate, valid_from, recorded_at, group_id) \
             VALUES ('alice', 'ns1', 'knows', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'ns1')",
            (),
        )
        .await
        .expect("INSERT fact with valid composite FK must succeed");
    }

    /// Simulate a crash between facts_new create (step 3) and facts rename (step 6):
    /// facts_new exists but facts does not. Calling migrate_006 must resume via G3
    /// and complete successfully.
    #[tokio::test]
    async fn migrate_006_partial_recovery_resumes_from_facts_new() {
        // Run full migration suite to get a post-004, pre-006 state.
        // Then simulate the crash state: facts_bak_006 + facts_new exist, facts does not.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = &graph.conn;

        // Post-migration: facts already has composite FK (migrate_006 ran).
        // We need to manufacture the partial-migration state.
        // Strategy: DROP facts, RENAME facts_bak_006 → facts (roll back to pre-006 shape
        // but with entities still having composite PK), then CREATE facts_new manually.
        conn.execute("PRAGMA foreign_keys = OFF", ())
            .await
            .expect("fk off");

        // Drop current facts (post-006).
        conn.execute("DROP TABLE IF EXISTS facts", ())
            .await
            .expect("drop facts");
        // facts_bak_006 holds pre-006 facts; rename it back to facts.
        conn.execute("ALTER TABLE facts_bak_006 RENAME TO facts", ())
            .await
            .expect("rename bak to facts");

        // Simulate crash state: create facts_new (composite FK DDL) but do NOT rename.
        // IF NOT EXISTS satisfies the no-bare-create meta-test (idempotency invariant
        // per migrations.rs:28-31); on this fresh in-memory connection facts_new
        // cannot pre-exist, so the guard is a no-op and crash-simulation intent
        // is preserved.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts_new (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id       TEXT NOT NULL,
                subject_group_id TEXT NOT NULL DEFAULT 'default',
                predicate        TEXT NOT NULL,
                object_id        TEXT,
                object_group_id  TEXT,
                object_value     TEXT,
                properties       TEXT,
                embedding        BLOB,
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
            )",
            (),
        )
        .await
        .expect("create facts_new (crash state)");

        conn.execute("PRAGMA foreign_keys = ON", ())
            .await
            .expect("fk on");

        // Now confirm the crash state: facts_new exists.
        // SCOPED block: libsql `Rows` cursor holds a connection lock until dropped.
        // If the iterator is alive when migrate_006's DROP TABLE fires below, SQLite
        // returns `database table is locked`. Explicit block-scope releases the lock
        // before the migration call.
        {
            let mut rows = conn
                .query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_new'",
                    (),
                )
                .await
                .expect("sqlite_master");
            assert!(
                rows.next().await.expect("next").is_some(),
                "pre-condition: facts_new must exist before resume test"
            );
        }

        // Call migrate_006 — G3 gate must fire and complete the partial migration.
        crate::core::migrations::migrate_006_composite_fk_facts_episodic_edges(
            conn,
            graph.embedding_dim,
        )
        .await
        .expect("migrate_006 must resume from partial state via G3 gate");

        // facts table must exist with composite FK.
        let mut fk_rows = conn
            .query("PRAGMA foreign_key_list('facts')", ())
            .await
            .expect("pragma fk_list");
        let mut found = false;
        while let Some(r) = fk_rows.next().await.expect("next") {
            let from: String = r.get(3).expect("col from");
            if from == "subject_group_id" {
                found = true;
                break;
            }
        }
        assert!(found, "facts must have composite FK after G3 resume");
    }

    // ── T1.8 integrity-gate tests (v0.2.0 Phase B-prep) ──────────────────────

    /// T1.8 happy path: a fresh in-memory DB opened via the normal path must
    /// pass `check_integrity()` and return `Ok`. Verifies the guard does NOT
    /// reject a legitimately-initialised store.
    #[tokio::test]
    async fn temporal_graph_open_succeeds_on_valid_db() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory must succeed on a fresh DB");

        // Belt-and-suspenders: confirm the four critical tables are present.
        for table in ["episodes", "entities", "entity_types", "facts"] {
            let mut rows = graph
                .conn
                .query(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1",
                    libsql::params![table],
                )
                .await
                .unwrap_or_else(|e| panic!("query failed for table {table}: {e}"));
            assert!(
                rows.next().await.expect("rows.next").is_some(),
                "critical table '{table}' must exist after normal open"
            );
        }
    }

    /// T1.8 corruption path: build a graph manually (bypassing `open_in_memory`
    /// so we can drop a table after migrations), then call `check_integrity()`
    /// directly. Must return `Err(CorruptStore { reason })` naming the missing
    /// table.
    #[tokio::test]
    async fn temporal_graph_open_refuses_corrupt_db() {
        use super::{NamespacePolicyCache, DEFAULT_NS_POLICY_CACHE_CAP};
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use tokio::sync::Mutex as AsyncMutex;

        let db = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .expect("build db");
        let conn = db.connect().expect("connect");
        let graph = TemporalGraph {
            _db: db,
            conn,
            write_lock: Arc::new(AsyncMutex::new(())),
            has_outer_transaction: Arc::new(AtomicBool::new(false)),
            transaction_owner: Arc::new(std::sync::Mutex::new(None)),
            dirty: Arc::new(AtomicBool::new(false)),
            embedding_dim: 384,
            policy_cache: NamespacePolicyCache::new(DEFAULT_NS_POLICY_CACHE_CAP),
        };
        // Run migrations so all tables exist — then corrupt one.
        graph.run_migrations().await.expect("run_migrations");
        // TD-136: under `content-search`, Migration 026 installs the DiskANN
        // `episodes_vec_idx` on `episodes`; libsql refuses `DROP TABLE` while its
        // vector index survives, so drop the index first (mirrors defs_j.rs's
        // `DROP INDEX IF EXISTS entities_vec_idx` before table restructure). IF
        // EXISTS makes this a harmless no-op on the default (no-index) build.
        graph
            .conn
            .execute("DROP INDEX IF EXISTS episodes_vec_idx", ())
            .await
            .expect("DROP INDEX episodes_vec_idx");
        graph
            .conn
            .execute("DROP TABLE episodes", ())
            .await
            .expect("DROP TABLE episodes");

        let result = graph.check_integrity().await;
        match result {
            Err(crate::core::error::Error::CorruptStore { reason }) => {
                assert!(
                    reason.contains("episodes"),
                    "CorruptStore reason must name the missing table; got: {reason}"
                );
            }
            other => panic!("expected Err(CorruptStore), got {other:?}"),
        }
    }
}
