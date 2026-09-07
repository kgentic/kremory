//! Migration 026 (dense episode retrieval): add an `embedding`
//! column + DiskANN vector index to `episodes`, mirroring the
//! `entities`/`facts` vector-search substrate.
//!
//! ## Why
//!
//! Episodes are retrieved by FTS5-BM25 ONLY (Migration 022 `episodes_fts`) —
//! the table has no embedding column and no vector index, so
//! paraphrase/pronoun questions never lexically match their answering episode
//! (the primary recall-breadth gap). This migration
//! installs the dense arm's storage: an `F32_BLOB(dim)` column + a
//! `libsql_vector_idx` cosine index, the EXACT shape `entities_vec_idx` /
//! `facts_vec_idx` use (`core/schema.rs`, `migrations/defs_j.rs`).
//!
//! Feature-gated behind `content-search` (two-lever gating model — same gate
//! as Migration 022's `episodes_fts`): the whole
//! function is compiled out of the default build, and the `run_migrations()`
//! call site in `core/schema.rs` is gated identically, so a default
//! (no-feature) build never touches `episodes` and behaves byte-identically to
//! pre-change.
//!
//! ## Idempotency
//!
//! - Column: PRAGMA `table_info(episodes)` gate — `ALTER TABLE ... ADD COLUMN`
//!   only fires when `embedding` is absent (mirrors the PRAGMA-guarded ADD
//!   COLUMN idiom in Migration 007/011/016/020).
//! - Index: `CREATE INDEX IF NOT EXISTS episodes_vec_idx`. LOUD — the create
//!   propagates `Err` (unlike the best-effort `episodes`-less `schema.rs`
//!   index creates), so a genuine type mismatch (column not `F32_BLOB`) is
//!   never swallowed into a silent brute-force regression — the exact
//!   failure mode `migrate_023` was written to prevent.
//!
//! ## Backfill
//!
//! Intentionally NOT backfilled here: episode embeddings require the runtime
//! embedder (unlike `episodes_fts`, whose content is stored verbatim). The
//! existing corpus is populated via the explicit
//! `Memory::backfill_episode_embeddings()` maintenance entrypoint
//! (`facade/mod.rs`); new episodes are embedded at ingest when the dense arm
//! is enabled (`core/ingest/pipeline`). A NULL `embedding` is simply invisible
//! to `vector_search_episodes` (its `WHERE embedding IS NOT NULL` / index
//! membership), never an error.
#[cfg(feature = "content-search")]
use crate::core::error::Result;

#[cfg(feature = "content-search")]
pub(crate) async fn migrate_026_episodes_embedding(
    conn: &libsql::Connection,
    dim: usize,
) -> Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_026 step `{name}` failed: {e}"
            ))
        }
    }

    // Column presence gate (PRAGMA table_info) — ADD COLUMN only when absent.
    // ALTER TABLE ADD COLUMN cannot change an existing column's type, so a
    // second run must NOT re-attempt it; the gate makes this a clean no-op.
    let mut has_embedding = false;
    let mut rows = conn
        .query("PRAGMA table_info(episodes)", ())
        .await
        .map_err(step("pragma_table_info_episodes"))?;
    while let Some(row) = rows.next().await.map_err(step("pragma_row"))? {
        let name: String = row.get(1).map_err(step("pragma_col_name"))?;
        if name == "embedding" {
            has_embedding = true;
            break;
        }
    }

    let added = if has_embedding {
        false
    } else {
        // `F32_BLOB(dim)` is the declared type the DiskANN vector index requires
        // (a plain `BLOB` is rejected). A fresh ADD COLUMN can declare
        // it directly (no BLOB→F32_BLOB rebuild dance needed, unlike defs_j's
        // pre-existing-column conversion).
        conn.execute(
            &format!("ALTER TABLE episodes ADD COLUMN embedding F32_BLOB({dim})"),
            (),
        )
        .await
        .map_err(step("add_embedding_column"))?;
        true
    };

    // LOUD vector index create (idempotent via IF NOT EXISTS). Mirrors
    // `facts_vec_idx` / `entities_vec_idx` DDL exactly. Propagates Err on a
    // genuine type mismatch so a silent brute-force regression can't ship.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS episodes_vec_idx \
         ON episodes(libsql_vector_idx(embedding, 'metric=cosine'))",
        (),
    )
    .await
    .map_err(step("episodes_vec_idx_create_must_succeed"))?;

    tracing::info!(
        target: "kremory::migrations",
        migration = "026",
        column_added = added,
        dim,
        "migrate_026: episodes.embedding + episodes_vec_idx (dense episode arm) installed"
    );
    Ok(())
}
