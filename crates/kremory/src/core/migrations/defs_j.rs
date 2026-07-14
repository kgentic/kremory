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
/// ## Idempotency
///
/// SHAPE-based gate per table: `PRAGMA table_info` reports the `embedding`
/// column's declared type; if it is already NOT the literal string `BLOB`
/// (i.e. already `F32_BLOB(...)`), that table's rebuild is skipped entirely.
/// Partial-migration recovery mirrors `migrate_004`/`migrate_006`: a leftover
/// `entities_new_023` / `facts_new_023` scratch table from a crashed prior run
/// is detected and the migration resumes from drop+rename rather than
/// re-running the backup+copy step.
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

    // A table needs rebuilding iff its embedding column is declared exactly
    // `BLOB` (SQLite preserves the declared type text verbatim; compare
    // case-insensitively defensively).
    let entities_needs_rebuild = column_type(conn, "entities", "embedding")
        .await
        .map_err(step("check_entities_embedding_type"))?
        .is_some_and(|ty| ty.eq_ignore_ascii_case("BLOB"));
    let facts_needs_rebuild = column_type(conn, "facts", "embedding")
        .await
        .map_err(step("check_facts_embedding_type"))?
        .is_some_and(|ty| ty.eq_ignore_ascii_case("BLOB"));

    if !entities_needs_rebuild && !facts_needs_rebuild {
        tracing::debug!(
            target: "kremory::migrations",
            "migrate_023: entities.embedding and facts.embedding are already \
             F32_BLOB — no rebuild needed"
        );
        return Ok(());
    }

    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;

    let body_result: crate::core::error::Result<()> = async {
        // ── entities half ────────────────────────────────────────────────────
        if entities_needs_rebuild {
            let partial = table_exists(conn, "entities_new_023")
                .await
                .map_err(step("check_entities_new_023"))?;

            if partial {
                tracing::warn!(
                    target: "kremory::migrations",
                    "migrate_023: entities_new_023 already exists — resuming partial migration"
                );
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

            conn.execute("DROP TABLE entities", ())
                .await
                .map_err(step("drop_entities"))?;
            conn.execute("ALTER TABLE entities_new_023 RENAME TO entities", ())
                .await
                .map_err(step("rename_entities_new_023"))?;

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
            conn.execute(
                "CREATE INDEX entities_vec_idx \
                 ON entities(libsql_vector_idx(embedding, 'metric=cosine'))",
                (),
            )
            .await
            .map_err(step("entities_vec_idx_create_must_succeed"))?;
        }

        // ── facts half ───────────────────────────────────────────────────────
        if facts_needs_rebuild {
            let partial = table_exists(conn, "facts_new_023")
                .await
                .map_err(step("check_facts_new_023"))?;

            if partial {
                tracing::warn!(
                    target: "kremory::migrations",
                    "migrate_023: facts_new_023 already exists — resuming partial migration"
                );
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

            conn.execute("DROP TABLE facts", ())
                .await
                .map_err(step("drop_facts"))?;
            conn.execute("ALTER TABLE facts_new_023 RENAME TO facts", ())
                .await
                .map_err(step("rename_facts_new_023"))?;

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
                "CREATE INDEX facts_vec_idx \
                 ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
                (),
            )
            .await
            .map_err(step("facts_vec_idx_create_must_succeed"))?;
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

    tracing::info!(
        target: "kremory::migrations",
        migration = "023",
        entities_rebuilt = entities_needs_rebuild,
        facts_rebuilt = facts_needs_rebuild,
        "migrate_023: embedding column(s) converted to F32_BLOB(dim); vector index(es) \
         created (foreign_key_check clean)"
    );
    Ok(())
}
