//! Migration 022 (ADR-072 seq1 impl-spec §1): `episodes_fts` — BM25/FTS5
//! content-recall substrate over raw `episodes.content`.
//!
//! Feature-gated behind `content-search` (two-lever gating model, ADR-072 §6
//! Finding 3 — seq1 minimum is the compile feature-gate; a per-episode
//! runtime `index_content` toggle is a later increment). The entire function
//! is compiled out of the default build — `run_migrations()`'s call site in
//! `core/schema.rs` is gated identically, so a default (no-feature) build
//! never creates `episodes_fts` and behaves byte-identically to pre-change.
//!
//! ## Design
//!
//! `episodes_fts` is an **external-content** FTS5 table
//! (`content='episodes', content_rowid='id'`) — it shadows `episodes.content`
//! keyed by the episode's own `INTEGER PRIMARY KEY AUTOINCREMENT` rowid,
//! rather than storing a separate `episode_id` data column the way
//! `entities_fts`/`facts_fts` do (those predate external-content and carry
//! their own `entity_id`/`fact_id` UNINDEXED columns — schema.rs:599,609).
//! External-content keeps the indexed text as the SOLE column, which is what
//! lets `snippet()`/`bm25()`/the `rank` hidden column all work directly
//! against `episodes_fts` with zero duplication of `episodes.content`.
//!
//! ## Idempotency
//!
//! `CREATE VIRTUAL TABLE IF NOT EXISTS` (mirrors migration 019/020/021 idiom).
//! The backfill is a `WHERE NOT EXISTS` guarded INSERT — safe to re-run on an
//! already-migrated db (only ever inserts episodes missing their shadow row).
//! This guard also self-heals a partial write: `insert_episode`/
//! `insert_episode_with_group` populate `episodes_fts` as a same-call
//! (non-transactional) follow-up INSERT (see `core/graph/episodes.rs`); if a
//! crash lands between the two INSERTs, the NEXT `TemporalGraph::open*` call
//! re-runs this migration and the backfill guard repairs the missing shadow
//! row — the same safety net every idempotent-backfill migration in this
//! crate already relies on.
#[cfg(feature = "content-search")]
use crate::core::error::Result;

#[cfg(feature = "content-search")]
pub(crate) async fn migrate_022_episodes_content_recall(
    conn: &libsql::Connection,
) -> Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_022 step `{name}` failed: {e}"
            ))
        }
    }

    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5( \
             content, \
             content='episodes', \
             content_rowid='id' \
         )",
        (),
    )
    .await
    .map_err(step("create_episodes_fts"))?;

    // One-time guarded backfill of pre-existing episodes (pure SQL, no
    // embedder — `episodes.content` is stored verbatim + immutable per
    // ADR-042, so there is nothing to re-derive). `NOT EXISTS` makes this
    // idempotent AND self-healing across restarts (see module doc).
    let backfilled = conn
        .execute(
            "INSERT INTO episodes_fts(rowid, content) \
             SELECT e.id, e.content FROM episodes e \
             WHERE NOT EXISTS (SELECT 1 FROM episodes_fts f WHERE f.rowid = e.id)",
            (),
        )
        .await
        .map_err(step("backfill_episodes_fts"))?;

    tracing::info!(
        target: "kremory::migrations",
        migration = "022",
        backfilled,
        "migrate_022: episodes_fts (ADR-072 seq1 content-search substrate) installed"
    );
    Ok(())
}
