// ─── Migration 023 ─────────────────────────────────────────────────────────

/// Migration 023 (TD-115, `.ai-docs/tech-debt/tech-debt-register.md` session
/// 2026-07-14): rebuild `entities` / `facts` with a real `embedding
/// F32_BLOB(dim)` column when the current declared type is a generic `BLOB`.
///
/// ## The bug
///
/// libSQL's DiskANN vector index (`libsql_vector_idx`) REJECTS a plain `BLOB`
/// column (`SqliteFailure(1, "vector index: unexpected vector column type:
/// BLOB")`). The `entities` table-rebuild migrations (`migrate_004` in
/// `defs_a.rs:212`, `migrate_013` in `defs_d.rs:169`) recreate `entities` with
/// `embedding BLOB` — `dim` was not threaded through those rebuild helpers, so
/// they fell back to the generic type. The vector-index `CREATE INDEX` in
/// `schema.rs` was wrapped in `let _ = ...` (now fixed to be observable, see
/// the `kremory.search.vector_index_create_failed` counter added alongside
/// this migration), so the failure was silent: `vector_top_k('entities_vec_idx',
/// ...)` finds no index, `vector_search_with_index` errors, and every recall
/// falls back to `vector_search_brute_force` (correct, but O(n)).
///
/// ## Detection-gated, per table (empirically verified before writing this)
///
/// A throwaway spike against a freshly-migrated file DB confirmed:
/// `entities.embedding` is declared `BLOB` (index create → `Err`);
/// `facts.embedding` is ALREADY `F32_BLOB(384)` (index create → `Ok`) because
/// `migrate_006` (`defs_h.rs`) *does* thread `dim` through its rebuild. So on
/// the normal forward-migration path only `entities` needs repair. `facts` is
/// checked independently anyway (`PRAGMA table_info`, not a blanket rebuild)
/// so this migration also defensively repairs a DB that reached a `BLOB`
/// `facts.embedding` via the emergency downgrade tooling
/// (`migrate_015b_downgrade_crash_safety_schema`, `defs_g2.rs` — not on the
/// forward path, but this migration must not assume it was never invoked).
///
/// ## Embedding preservation (spike-verified, load-bearing)
///
/// The rebuild copies `embedding` via a RAW `INSERT ... SELECT embedding` —
/// no re-wrap through `vector(...)`. A throwaway spike proved this preserves
/// the binary vector exactly: a `BLOB`-column table seeded via
/// `vector('[...]')`, raw-copied into an `F32_BLOB(dim)` table, indexed, and
/// queried via `vector_top_k` returned the correct nearest neighbours (the
/// exact match and a near-duplicate ranked ahead of an orthogonal vector).
/// SQLite column type affinity is a storage-class / index-eligibility hint,
/// not a byte-level transform — copying the same blob into a
/// differently-typed column preserves the bytes bit-for-bit.
///
/// ## Column shape (locked to the current schema; verified via
/// `PRAGMA table_info` against a DB that ran migrations 002-022)
///
/// `entities`: id, properties, embedding, recorded_at, updated_at, group_id,
/// access_count, entity_type_id, entity_type_source (CHECK constraint from
/// migration 013), entity_type_assigned_at, ner_confidence,
/// is_dream_generated (migration 016). `PRIMARY KEY (id, group_id)`.
///
/// `facts`: id, subject_id, subject_group_id, predicate, object_id,
/// object_group_id, object_value, properties, embedding, valid_from,
/// valid_to, recorded_at, expired_at, invalid_at, group_id, confidence,
/// source_episode_id, memory_type, content_hash, access_count,
/// is_dream_generated (migration 016), corroboration_inert (migration 020).
/// Composite FKs to `entities(id, group_id)` (subject + object) and
/// `episodes(id)` (source_episode_id) — mirrors `migrate_006`.
///
/// ## LOUD index creation (Rule 19 / the actual TD-115 fix)
///
/// Unlike every other migration in this module, the vector-index
/// `CREATE INDEX` here is NOT wrapped in `let _ = ...`. A failure here means
/// the fix itself didn't work (the column still isn't a real vector column),
/// which MUST surface as a hard migration error — silently continuing would
/// reproduce the exact bug this migration exists to close.
///
/// ## Crash-recovery hardening (Vera H3/H4, `.ai-docs/architecture-review/`
/// `vera-adr-074-migrate-023-review-2026-07-14.md`)
///
/// This migration runs unconditionally against every already-published
/// consumer's existing, populated database — a crash mid-run must never
/// cause silent data loss or permanent under-indexing. Two invariants:
///
/// - **Content-based resume (H3).** A leftover `X_new_023` scratch table is
///   NOT trusted just because it exists — a crash between `CREATE TABLE
///   X_new_023` and the `INSERT ... SELECT` copy leaves it existing but
///   empty/partial. Resume compares `COUNT(X_new_023)` against
///   `COUNT(X_bak_023)` (the immutable pre-migration snapshot) and
///   repopulates `X_new_023` from `X_bak_023` — never from the live `X`,
///   which may already have been dropped by a prior crashed run — before
///   proceeding to `DROP TABLE IF EXISTS X; RENAME`. If `X_bak_023` itself
///   is missing (crash before the snapshot was ever taken), it is
///   materialised from the live `X` table, which is guaranteed not to have
///   been dropped yet at that point in the control flow. `DROP TABLE X` is
///   `IF EXISTS` so a crash between the drop and the rename resumes cleanly
///   instead of erroring on the missing table. The rebuild gate itself
///   (`entities_needs_rebuild` / `facts_needs_rebuild`) fires on `X_new_023`
///   existing even when `X`'s declared column type can no longer be read
///   (because `X` was already dropped) — otherwise that exact crash window
///   would never re-enter this code path on restart.
/// - **Index completeness independent of the rebuild gate (H4).** The
///   column-type gate (`embedding` is `F32_BLOB`) is satisfied as soon as
///   `RENAME` completes, but ~10 more `CREATE INDEX`/`CREATE VIRTUAL TABLE`
///   statements follow it. A crash after rename but before all indexes
///   complete used to be permanently invisible: on the next open the column
///   already reads `F32_BLOB`, so the whole block — including the missed
///   indexes — was skipped forever. Index presence (`entities_vec_idx` /
///   `facts_vec_idx` in `sqlite_master`) is now checked independently of the
///   column-type gate; a table whose column is already `F32_BLOB` but whose
///   vector index is missing gets its full index set re-ensured (all
///   `CREATE INDEX`/`CREATE VIRTUAL TABLE` statements are `IF NOT EXISTS`,
///   safe no-ops when already present) without re-running the destructive
///   table-rebuild dance.
///
/// ## Idempotency
///
/// SHAPE-based gate per table: `PRAGMA table_info` reports the `embedding`
/// column's declared type; if it is already NOT the literal string `BLOB`
/// (i.e. already `F32_BLOB(...)`) AND there is no leftover `X_new_023`
/// scratch table AND the vector index already exists, that table is left
/// entirely untouched. Running this migration twice on an already-fully-
/// migrated, already-indexed DB is a clean no-op with no error.
pub(crate) async fn migrate_023_vector_index_column_type(
    conn: &libsql::Connection,
    dim: usize,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_023 step `{name}` failed: {e}"
            ))
        }
    }

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
        Ok(rows.next().await?.is_some())
    }

    /// H4: independent index-presence check — the rebuild gate (column
    /// type) cannot see this, since it's satisfied the instant `RENAME`
    /// completes, well before the index statements that follow it.
    async fn index_exists(
        conn: &libsql::Connection,
        name: &str,
    ) -> std::result::Result<bool, libsql::Error> {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?1",
                libsql::params![name],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    }

    /// H3: content-based completeness check. `COUNT(*)` always returns
    /// exactly one row for a real table; the `None` arm is unreachable in
    /// practice but handled without panicking (Rule 8 — no `.expect()` in
    /// src) rather than trusting that invariant blindly.
    async fn row_count(
        conn: &libsql::Connection,
        table: &str,
    ) -> std::result::Result<i64, libsql::Error> {
        let mut rows = conn
            .query(&format!("SELECT COUNT(*) FROM {table}"), ())
            .await?;
        match rows.next().await? {
            Some(row) => row.get::<i64>(0),
            None => Ok(0),
        }
    }

    /// Declared SQL type of `table.column` per `PRAGMA table_info`, e.g.
    /// `"BLOB"` or `"F32_BLOB(384)"`. `None` if the column (or table) doesn't
    /// exist.
    async fn column_type(
        conn: &libsql::Connection,
        table: &str,
        column: &str,
    ) -> std::result::Result<Option<String>, libsql::Error> {
        let mut rows = conn
            .query(&format!("PRAGMA table_info({table})"), ())
            .await?;
        while let Some(row) = rows.next().await? {
            let name: String = row.get(1)?;
            if name == column {
                let ty: String = row.get(2)?;
                return Ok(Some(ty));
            }
        }
        Ok(None)
    }

    /// H4: ensure `entities`' btree indexes + the LOUD vector index exist.
    /// Called both after a full rebuild (rename just completed) and,
    /// independently, when the column is already `F32_BLOB` but the vector
    /// index alone is missing (crash between a prior rename and indexing).
    /// Every statement is `IF NOT EXISTS` — safe, cheap no-op when already
    /// present.
    async fn create_entities_indexes(conn: &libsql::Connection) -> crate::core::error::Result<()> {
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id)",
            (),
        )
        .await
        .map_err(step("idx_entities_group"))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type_id \
             ON entities(group_id, entity_type_id)",
            (),
        )
        .await
        .map_err(step("idx_entities_type_id"))?;

        // LOUD — the whole point of this migration. Must NOT be swallowed.
        // `IF NOT EXISTS` makes this idempotent (H4) without weakening the
        // failure propagation: libsql still errors on a genuine type
        // mismatch (e.g. the column is still declared BLOB) regardless of
        // the clause — `IF NOT EXISTS` only short-circuits on a name
        // collision, not on a validation failure.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS entities_vec_idx \
             ON entities(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        )
        .await
        .map_err(step("entities_vec_idx_create_must_succeed"))?;
        Ok(())
    }

    /// H4: ensure `facts`' FTS shadow table + btree indexes + the LOUD
    /// vector index exist. Same calling contract as
    /// `create_entities_indexes`.
    async fn create_facts_indexes(conn: &libsql::Connection) -> crate::core::error::Result<()> {
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

        // Idempotent backfill guard: `fact_id` is UNINDEXED on the FTS5
        // virtual table (no unique constraint it can enforce), so an
        // unguarded re-run of the backfill on an already-populated
        // `facts_fts` would duplicate every row. Only backfill when empty.
        let facts_fts_rows = row_count(conn, "facts_fts")
            .await
            .map_err(step("count_facts_fts"))?;
        if facts_fts_rows == 0 {
            let _ = conn
                .execute(
                    "INSERT INTO facts_fts (fact_id, predicate, object_value) \
                     SELECT id, predicate, object_value FROM facts \
                     WHERE object_value IS NOT NULL",
                    (),
                )
                .await;
        }

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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_group ON facts(group_id)",
            (),
        )
        .await
        .map_err(step("idx_facts_group"))?;
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject_group \
             ON facts(subject_id, subject_group_id)",
            (),
        )
        .await
        .map_err(step("idx_facts_subject_group"))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_object_group \
             ON facts(object_id, object_group_id) WHERE object_id IS NOT NULL",
            (),
        )
        .await
        .map_err(step("idx_facts_object_group"))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id, expired_at)",
            (),
        )
        .await
        .map_err(step("idx_facts_subject"))?;

        // LOUD — the whole point of this migration. Must NOT be swallowed.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS facts_vec_idx \
             ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        )
        .await
        .map_err(step("facts_vec_idx_create_must_succeed"))?;
        Ok(())
    }

    // A table needs rebuilding iff its embedding column is declared exactly
    // `BLOB` (SQLite preserves the declared type text verbatim; compare
    // case-insensitively defensively) OR a leftover `X_new_023` scratch
    // table exists (H3: a prior run crashed mid-rebuild — resume is
    // required regardless of what `X` currently reports, including the
    // window where `X` was already dropped and `column_type` therefore
    // returns `None`).
    let entities_embedding_ty = column_type(conn, "entities", "embedding")
        .await
        .map_err(step("check_entities_embedding_type"))?;
    let facts_embedding_ty = column_type(conn, "facts", "embedding")
        .await
        .map_err(step("check_facts_embedding_type"))?;

    let entities_declared_blob = entities_embedding_ty
        .as_deref()
        .is_some_and(|ty| ty.eq_ignore_ascii_case("BLOB"));
    let facts_declared_blob = facts_embedding_ty
        .as_deref()
        .is_some_and(|ty| ty.eq_ignore_ascii_case("BLOB"));

    let entities_partial = table_exists(conn, "entities_new_023")
        .await
        .map_err(step("check_entities_new_023"))?;
    let facts_partial = table_exists(conn, "facts_new_023")
        .await
        .map_err(step("check_facts_new_023"))?;

    let entities_needs_rebuild = entities_declared_blob || entities_partial;
    let facts_needs_rebuild = facts_declared_blob || facts_partial;

    // H4: a table can already be fully rebuilt (column is F32_BLOB, no
    // leftover scratch table) yet still be missing its vector index if a
    // prior run crashed between RENAME and the ~10 CREATE INDEX statements
    // that follow it. The column-type gate alone can never re-detect this —
    // check index presence independently and, if missing, ensure the full
    // index set without re-running the (unnecessary, and on a live table,
    // wasteful) table-rebuild dance.
    let entities_vec_idx_present = index_exists(conn, "entities_vec_idx")
        .await
        .map_err(step("check_entities_vec_idx"))?;
    let facts_vec_idx_present = index_exists(conn, "facts_vec_idx")
        .await
        .map_err(step("check_facts_vec_idx"))?;

    let entities_needs_index_ensure = !entities_needs_rebuild && !entities_vec_idx_present;
    let facts_needs_index_ensure = !facts_needs_rebuild && !facts_vec_idx_present;

    if !entities_needs_rebuild
        && !facts_needs_rebuild
        && !entities_needs_index_ensure
        && !facts_needs_index_ensure
    {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_023: entities.embedding and facts.embedding are already \
             F32_BLOB and both vector indexes exist — no work needed"
        );
        return Ok(());
    }

    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        // ── entities half ────────────────────────────────────────────────────
        if entities_needs_rebuild {
            if entities_partial {
                tracing::warn!(
                    target: "kremory::migrations",
                    "migrate_023: entities_new_023 already exists — verifying completeness \
                     before resuming (H3: existence alone is not trusted)"
                );

                let bak_exists = table_exists(conn, "entities_bak_023")
                    .await
                    .map_err(step("check_entities_bak_023"))?;
                if !bak_exists {
                    // Crash happened before the snapshot was ever taken. `entities`
                    // must still be live here — the destructive `DROP TABLE`
                    // only runs further down this same block — so materialise
                    // the immutable recovery snapshot from it now, before
                    // proceeding. If `entities` no longer exists either (a
                    // deeper double-failure), this errors loudly rather than
                    // silently trusting the empty `entities_new_023`.
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS entities_bak_023 AS SELECT * FROM entities",
                        (),
                    )
                    .await
                    .map_err(step("create_entities_bak_023_late"))?;
                }

                let bak_count = row_count(conn, "entities_bak_023")
                    .await
                    .map_err(step("count_entities_bak_023"))?;
                let new_count = row_count(conn, "entities_new_023")
                    .await
                    .map_err(step("count_entities_new_023"))?;

                if new_count != bak_count {
                    tracing::warn!(
                        target: "kremory::migrations",
                        bak_count,
                        new_count,
                        "migrate_023: entities_new_023 is incomplete (crash mid-copy) — \
                         repopulating from entities_bak_023, never from live entities"
                    );
                    conn.execute("DELETE FROM entities_new_023", ())
                        .await
                        .map_err(step("clear_entities_new_023"))?;
                    conn.execute(
                        "INSERT INTO entities_new_023 (
                            id, properties, embedding, recorded_at, updated_at, group_id,
                            access_count, entity_type_id, entity_type_source,
                            entity_type_assigned_at, ner_confidence, is_dream_generated
                         )
                         SELECT
                            id, properties, embedding, recorded_at, updated_at, group_id,
                            access_count, entity_type_id, entity_type_source,
                            entity_type_assigned_at, ner_confidence, is_dream_generated
                         FROM entities_bak_023",
                        (),
                    )
                    .await
                    .map_err(step("recopy_entities_from_bak"))?;

                    let recopied_count = row_count(conn, "entities_new_023")
                        .await
                        .map_err(step("count_entities_new_023_after_recopy"))?;
                    if recopied_count != bak_count {
                        return Err(crate::core::error::Error::Other(anyhow::anyhow!(
                            "migrate_023: entities_new_023 recopy from entities_bak_023 still \
                             mismatched ({recopied_count} vs {bak_count} rows) — refusing to \
                             proceed; inspect entities_bak_023 for recovery"
                        )));
                    }
                }
            } else {
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS entities_bak_023 AS SELECT * FROM entities",
                    (),
                )
                .await
                .map_err(step("create_entities_bak_023"))?;

                conn.execute(
                    &format!(
                        "CREATE TABLE IF NOT EXISTS entities_new_023 (
                            id                      TEXT NOT NULL,
                            properties              TEXT,
                            embedding               F32_BLOB({dim}),
                            recorded_at             TEXT NOT NULL,
                            updated_at              TEXT,
                            group_id                TEXT NOT NULL DEFAULT 'default',
                            access_count            INTEGER NOT NULL DEFAULT 0,
                            entity_type_id          INTEGER NOT NULL DEFAULT 0,
                            entity_type_source      TEXT CHECK (entity_type_source IN (
                                'Phase1Ner', 'Phase2Llm', 'DreamPass0',
                                'DreamPass1', 'ConsumerPinned', 'DreamPass4'
                            )),
                            entity_type_assigned_at TEXT,
                            ner_confidence          REAL,
                            is_dream_generated      INTEGER NOT NULL DEFAULT 0,
                            PRIMARY KEY (id, group_id)
                        )"
                    ),
                    (),
                )
                .await
                .map_err(step("create_entities_new_023"))?;

                // RAW copy — `embedding` is passed through untouched (spike-verified:
                // preserves the vector bit-for-bit; no re-wrap through `vector(...)`).
                conn.execute(
                    "INSERT INTO entities_new_023 (
                        id, properties, embedding, recorded_at, updated_at, group_id,
                        access_count, entity_type_id, entity_type_source,
                        entity_type_assigned_at, ner_confidence, is_dream_generated
                     )
                     SELECT
                        id, properties, embedding, recorded_at, updated_at, group_id,
                        access_count, entity_type_id, entity_type_source,
                        entity_type_assigned_at, ner_confidence, is_dream_generated
                     FROM entities",
                    (),
                )
                .await
                .map_err(step("copy_entities"))?;
            }

            let _ = conn
                .execute("DROP INDEX IF EXISTS entities_vec_idx", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_entities_group", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_entities_type_id", ())
                .await;

            // H3: `IF EXISTS` — a prior run may have already dropped `entities`
            // before crashing (the window between DROP and RENAME). Without
            // this, resuming here would error on the missing table instead of
            // proceeding straight to the rename.
            conn.execute("DROP TABLE IF EXISTS entities", ())
                .await
                .map_err(step("drop_entities"))?;
            conn.execute("ALTER TABLE entities_new_023 RENAME TO entities", ())
                .await
                .map_err(step("rename_entities_new_023"))?;

            create_entities_indexes(conn).await?;
        } else if entities_needs_index_ensure {
            // H4: column already F32_BLOB, no leftover scratch table — the
            // table itself is fine, only its vector index (or a sibling
            // index/FTS statement) is missing from a prior crash between
            // rename and indexing. Ensure the full index set; no table
            // manipulation needed.
            create_entities_indexes(conn).await?;
        }

        // ── facts half ───────────────────────────────────────────────────────
        if facts_needs_rebuild {
            if facts_partial {
                tracing::warn!(
                    target: "kremory::migrations",
                    "migrate_023: facts_new_023 already exists — verifying completeness \
                     before resuming (H3: existence alone is not trusted)"
                );

                let bak_exists = table_exists(conn, "facts_bak_023")
                    .await
                    .map_err(step("check_facts_bak_023"))?;
                if !bak_exists {
                    // See entities half: `facts` must still be live here if it
                    // exists at all — materialise the snapshot before the
                    // destructive drop further down this block.
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS facts_bak_023 AS SELECT * FROM facts",
                        (),
                    )
                    .await
                    .map_err(step("create_facts_bak_023_late"))?;
                }

                let bak_count = row_count(conn, "facts_bak_023")
                    .await
                    .map_err(step("count_facts_bak_023"))?;
                let new_count = row_count(conn, "facts_new_023")
                    .await
                    .map_err(step("count_facts_new_023"))?;

                if new_count != bak_count {
                    tracing::warn!(
                        target: "kremory::migrations",
                        bak_count,
                        new_count,
                        "migrate_023: facts_new_023 is incomplete (crash mid-copy) — \
                         repopulating from facts_bak_023, never from live facts"
                    );
                    conn.execute("DELETE FROM facts_new_023", ())
                        .await
                        .map_err(step("clear_facts_new_023"))?;
                    conn.execute(
                        "INSERT INTO facts_new_023 (
                            id, subject_id, subject_group_id, predicate,
                            object_id, object_group_id, object_value, properties, embedding,
                            valid_from, valid_to, recorded_at, expired_at, invalid_at,
                            group_id, confidence, source_episode_id, memory_type, content_hash,
                            access_count, is_dream_generated, corroboration_inert
                         )
                         SELECT
                            id, subject_id, subject_group_id, predicate,
                            object_id, object_group_id, object_value, properties, embedding,
                            valid_from, valid_to, recorded_at, expired_at, invalid_at,
                            group_id, confidence, source_episode_id, memory_type, content_hash,
                            access_count, is_dream_generated, corroboration_inert
                         FROM facts_bak_023",
                        (),
                    )
                    .await
                    .map_err(step("recopy_facts_from_bak"))?;

                    let recopied_count = row_count(conn, "facts_new_023")
                        .await
                        .map_err(step("count_facts_new_023_after_recopy"))?;
                    if recopied_count != bak_count {
                        return Err(crate::core::error::Error::Other(anyhow::anyhow!(
                            "migrate_023: facts_new_023 recopy from facts_bak_023 still \
                             mismatched ({recopied_count} vs {bak_count} rows) — refusing to \
                             proceed; inspect facts_bak_023 for recovery"
                        )));
                    }
                }
            } else {
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS facts_bak_023 AS SELECT * FROM facts",
                    (),
                )
                .await
                .map_err(step("create_facts_bak_023"))?;

                conn.execute(
                    &format!(
                        "CREATE TABLE IF NOT EXISTS facts_new_023 (
                            id                INTEGER PRIMARY KEY AUTOINCREMENT,
                            subject_id        TEXT NOT NULL,
                            subject_group_id  TEXT NOT NULL DEFAULT 'default',
                            predicate         TEXT NOT NULL,
                            object_id         TEXT,
                            object_group_id   TEXT,
                            object_value      TEXT,
                            properties        TEXT,
                            embedding         F32_BLOB({dim}),
                            valid_from        TEXT NOT NULL,
                            valid_to          TEXT,
                            recorded_at       TEXT NOT NULL,
                            expired_at        TEXT,
                            invalid_at        TEXT,
                            group_id          TEXT NOT NULL DEFAULT 'default',
                            confidence        REAL DEFAULT 1.0,
                            source_episode_id INTEGER,
                            memory_type       TEXT,
                            content_hash      TEXT,
                            access_count      INTEGER NOT NULL DEFAULT 0,
                            is_dream_generated  INTEGER NOT NULL DEFAULT 0,
                            corroboration_inert INTEGER NOT NULL DEFAULT 0,
                            FOREIGN KEY (subject_id, subject_group_id) REFERENCES entities(id, group_id),
                            FOREIGN KEY (object_id,  object_group_id)  REFERENCES entities(id, group_id),
                            FOREIGN KEY (source_episode_id)            REFERENCES episodes(id)
                        )"
                    ),
                    (),
                )
                .await
                .map_err(step("create_facts_new_023"))?;

                // RAW copy — same rationale as entities: `embedding` passed through
                // untouched, spike-verified to preserve the vector bit-for-bit.
                conn.execute(
                    "INSERT INTO facts_new_023 (
                        id, subject_id, subject_group_id, predicate,
                        object_id, object_group_id, object_value, properties, embedding,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at,
                        group_id, confidence, source_episode_id, memory_type, content_hash,
                        access_count, is_dream_generated, corroboration_inert
                     )
                     SELECT
                        id, subject_id, subject_group_id, predicate,
                        object_id, object_group_id, object_value, properties, embedding,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at,
                        group_id, confidence, source_episode_id, memory_type, content_hash,
                        access_count, is_dream_generated, corroboration_inert
                     FROM facts",
                    (),
                )
                .await
                .map_err(step("copy_facts"))?;
            }

            let _ = conn.execute("DROP TABLE IF EXISTS facts_fts", ()).await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_temporal", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_predicate", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_object", ())
                .await;
            let _ = conn.execute("DROP INDEX IF EXISTS idx_facts_group", ()).await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS facts_vec_idx", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_content_hash_unique", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_content_hash", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_subject_group", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_object_group", ())
                .await;
            let _ = conn
                .execute("DROP INDEX IF EXISTS idx_facts_subject", ())
                .await;

            // H3: `IF EXISTS` — see entities half.
            conn.execute("DROP TABLE IF EXISTS facts", ())
                .await
                .map_err(step("drop_facts"))?;
            conn.execute("ALTER TABLE facts_new_023 RENAME TO facts", ())
                .await
                .map_err(step("rename_facts_new_023"))?;

            create_facts_indexes(conn).await?;
        } else if facts_needs_index_ensure {
            // H4: see entities half — column already F32_BLOB, only the
            // index set needs completing.
            create_facts_indexes(conn).await?;
        }

        Ok(())
    }
    .await;

    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_023: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    body_result?;

    // Boy-scout integrity check — mirrors migrate_004/migrate_006.
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
            "migrate_023: PRAGMA foreign_key_check reported violations after rebuild — \
             refusing to proceed; inspect entities_bak_023 / facts_bak_023 for recovery"
        )));
    }

    // M1: row-count observability, sourced from the immutable `_bak_023`
    // snapshot (before) vs. the live, post-rebuild table (after) — captured
    // BEFORE the backups are dropped below. Only meaningful for tables
    // actually rebuilt this run (`0` otherwise; see `entities_rebuilt` /
    // `facts_rebuilt` in the same log event).
    let entities_before = if entities_needs_rebuild {
        row_count(conn, "entities_bak_023")
            .await
            .map_err(step("count_entities_bak_023_final"))?
    } else {
        0
    };
    let entities_after = if entities_needs_rebuild {
        row_count(conn, "entities")
            .await
            .map_err(step("count_entities_final"))?
    } else {
        0
    };
    let facts_before = if facts_needs_rebuild {
        row_count(conn, "facts_bak_023")
            .await
            .map_err(step("count_facts_bak_023_final"))?
    } else {
        0
    };
    let facts_after = if facts_needs_rebuild {
        row_count(conn, "facts")
            .await
            .map_err(step("count_facts_final"))?
    } else {
        0
    };

    // M5: `_bak_023` scratch tables are the recovery snapshot for the
    // duration of this run only — drop them now that the FK check confirms
    // success, mirroring the sibling-migration convention (`defs_f.rs:333`,
    // `defs_g2.rs:231,339`). Without this they persist forever, permanently
    // duplicating the pre-migration table size on disk.
    //
    // UNCONDITIONAL `IF EXISTS` (Quinn NEW-1) — NOT gated on `*_needs_rebuild`.
    // A crash after the RENAME but before all indexes finish resumes via the
    // H4 index-ensure path with `needs_rebuild = false` (column is already
    // `F32_BLOB`), yet a `_bak_023` snapshot from the crashed run still exists.
    // Gating the drop on `needs_rebuild` would orphan that snapshot forever, so
    // the H4 and M5 fixes would leak against each other. `DROP TABLE IF EXISTS`
    // is a harmless no-op when the table was never created (fresh/already-clean
    // DBs), so an unconditional drop is safe on every path.
    conn.execute("DROP TABLE IF EXISTS entities_bak_023", ())
        .await
        .map_err(step("drop_entities_bak_023"))?;
    conn.execute("DROP TABLE IF EXISTS facts_bak_023", ())
        .await
        .map_err(step("drop_facts_bak_023"))?;

    tracing::info!(
        target: "kremory::migrations",
        migration = "023",
        entities_rebuilt = entities_needs_rebuild,
        facts_rebuilt = facts_needs_rebuild,
        entities_before,
        entities_after,
        facts_before,
        facts_after,
        "migrate_023: embedding column(s) converted to F32_BLOB(dim); vector index(es) \
         created (foreign_key_check clean)"
    );
    Ok(())
}
