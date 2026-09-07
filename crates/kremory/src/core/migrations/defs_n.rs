//! Migration 027: add `tokenize='porter unicode61'` to `entities_fts` / `facts_fts`
//! / `episodes_fts`, so a stored "painted" matches a query for "painting".
//!
//! ## Why a new migration, not editing the original `CREATE VIRTUAL TABLE`s
//!
//! FTS5's tokenizer is fixed at `CREATE VIRTUAL TABLE` time — there is no
//! `ALTER` for it, and every existing `CREATE VIRTUAL TABLE IF NOT EXISTS` is
//! a no-op on an already-migrated db regardless of what the SQL literal says.
//! Editing the historical migrations would silently do nothing for anyone who
//! already has the table; a new migration that drops, recreates with the new
//! tokenizer, and backfills is the only way to reach existing databases.
//!
//! Confirmed BEFORE writing this migration, not assumed: a throwaway
//! compile-spike against the exact `libsql = "=0.9.30"` this crate pins
//! confirmed `porter` is available and that `MATCH 'painting'` finds a row
//! stored as `'painted the fence yesterday'`.
//!
//! ## Idempotency
//!
//! Each table's own `sqlite_master.sql` is queried and checked for the
//! literal `porter` before doing anything — cheaper and more direct than a
//! PRAGMA/marker-table scheme, and it is inherently self-describing: the
//! table's own DDL text IS the fact being tested. `entities_fts` / `facts_fts`
//! are unconditional (base schema, no feature gate — mirrors their own
//! `CREATE VIRTUAL TABLE IF NOT EXISTS` at `core/schema.rs`). `episodes_fts`
//! only exists behind `content-search` (Migration 022's own gate), so its
//! porter step is gated identically — a default build never touches it and
//! stays byte-identical to pre-change.
//!
//! ## Backfill
//!
//! `entities_fts` / `facts_fts` are standalone (non-external-content) FTS5
//! tables that store their own copy of the searchable text — dropping the
//! table discards that copy, so the backfill re-derives it from the base
//! tables, mirroring the exact shape every live INSERT site already writes
//! (`core/graph/entities.rs:105`, `core/graph/facts.rs:525`, and the
//! provenance/dream write paths): `entities_fts.label` is always `''` (dead
//! since Migration 009 dropped `entities.label`; kept only because the FTS5
//! column list is fixed at creation), `properties` mirrors
//! `entities.properties` verbatim; `facts_fts` only ever gets a row when
//! `object_value IS NOT NULL` (the same guard `insert_fact_with_group` uses),
//! so the backfill applies the identical `WHERE` clause. `episodes_fts` is
//! external-content, so its backfill is the same `INSERT ... SELECT`
//! Migration 022 already uses.
use crate::core::error::Result;

async fn table_already_has_porter(conn: &libsql::Connection, table: &str) -> Result<bool> {
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?1",
            libsql::params![table],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_027 step `check_{table}_tokenizer` failed: {e}"
            ))
        })?;
    match rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_027 step `read_{table}_sql` failed: {e}"
        ))
    })? {
        None => Ok(false), // table doesn't exist yet — treated as "no porter", the
        // subsequent CREATE VIRTUAL TABLE IF NOT EXISTS below is
        // then the FIRST create and gets porter from day one.
        Some(row) => {
            let sql: String = row.get(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "migrate_027 step `read_{table}_sql_col` failed: {e}"
                ))
            })?;
            Ok(sql.contains("porter"))
        }
    }
}

fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
    move |e| crate::core::error::Error::Other(anyhow::anyhow!("migrate_027 step `{name}` failed: {e}"))
}

pub(crate) async fn migrate_027_fts5_porter_stemmer(conn: &libsql::Connection) -> Result<()> {
    // ── entities_fts (unconditional — base schema) ──────────────────────────
    if !table_already_has_porter(conn, "entities_fts").await? {
        conn.execute("DROP TABLE IF EXISTS entities_fts", ())
            .await
            .map_err(step("drop_entities_fts"))?;
        conn.execute(
            "CREATE VIRTUAL TABLE entities_fts USING fts5(
                entity_id UNINDEXED,
                label,
                properties,
                tokenize='porter unicode61'
            )",
            (),
        )
        .await
        .map_err(step("create_entities_fts_porter"))?;
        let n = conn
            .execute(
                "INSERT INTO entities_fts(entity_id, label, properties) \
                 SELECT id, '', properties FROM entities",
                (),
            )
            .await
            .map_err(step("backfill_entities_fts"))?;
        tracing::info!(
            target: "kremory::migrations",
            migration = "027",
            table = "entities_fts",
            backfilled = n,
            "migrate_027: entities_fts rebuilt with porter unicode61 tokenizer"
        );
    }

    // ── facts_fts (unconditional — base schema) ─────────────────────────────
    if !table_already_has_porter(conn, "facts_fts").await? {
        conn.execute("DROP TABLE IF EXISTS facts_fts", ())
            .await
            .map_err(step("drop_facts_fts"))?;
        conn.execute(
            "CREATE VIRTUAL TABLE facts_fts USING fts5(
                fact_id UNINDEXED,
                predicate,
                object_value,
                tokenize='porter unicode61'
            )",
            (),
        )
        .await
        .map_err(step("create_facts_fts_porter"))?;
        let n = conn
            .execute(
                "INSERT INTO facts_fts(fact_id, predicate, object_value) \
                 SELECT id, predicate, object_value FROM facts WHERE object_value IS NOT NULL",
                (),
            )
            .await
            .map_err(step("backfill_facts_fts"))?;
        tracing::info!(
            target: "kremory::migrations",
            migration = "027",
            table = "facts_fts",
            backfilled = n,
            "migrate_027: facts_fts rebuilt with porter unicode61 tokenizer"
        );
    }

    // ── episodes_fts (content-search only — Migration 022's own gate) ───────
    #[cfg(feature = "content-search")]
    if !table_already_has_porter(conn, "episodes_fts").await? {
        conn.execute("DROP TABLE IF EXISTS episodes_fts", ())
            .await
            .map_err(step("drop_episodes_fts"))?;
        conn.execute(
            "CREATE VIRTUAL TABLE episodes_fts USING fts5(
                content,
                content='episodes',
                content_rowid='id',
                tokenize='porter unicode61'
            )",
            (),
        )
        .await
        .map_err(step("create_episodes_fts_porter"))?;
        let n = conn
            .execute(
                "INSERT INTO episodes_fts(rowid, content) SELECT e.id, e.content FROM episodes e",
                (),
            )
            .await
            .map_err(step("backfill_episodes_fts"))?;
        tracing::info!(
            target: "kremory::migrations",
            migration = "027",
            table = "episodes_fts",
            backfilled = n,
            "migrate_027: episodes_fts rebuilt with porter unicode61 tokenizer"
        );
    }

    Ok(())
}
