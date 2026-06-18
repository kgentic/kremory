// ─── Migration 006 ─────────────────────────────────────────────────────────

/// Migration 006: install composite FK constraints on `facts` and `episodic_edges`
/// (ADR-029b Decision 1).
///
/// Uses CREATE-COPY-DROP-RENAME to replace the placeholder ADD COLUMN stubs
/// that migration 004 installed. After this migration both tables reference
/// `entities(id, group_id)` rather than the now-invalid single-column `entities(id)`.
///
/// Idempotency gates:
///   G1 — facts already has composite FK shape → already ran, return Ok(()).
///   G2 — entities does NOT have composite PK → migration 004 not yet applied, return Err.
///   G3 — `facts_new` exists → partial migration, resume from drop+rename.
///   G4 — `episodic_edges_new` exists → same for episodic_edges half.
///
/// The `dim` parameter is required because `facts` contains an `F32_BLOB(dim)`
/// vector column; the new table DDL must embed the same dimension value.
pub(crate) async fn migrate_006_composite_fk_facts_episodic_edges(
    conn: &libsql::Connection,
    dim: usize,
) -> crate::core::error::Result<()> {
    fn step<E: std::fmt::Display>(name: &str) -> impl Fn(E) -> crate::core::error::Error + '_ {
        move |e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "migrate_006 step `{name}` failed: {e}"
            ))
        }
    }

    // Helper: check whether a table exists.
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

    // Helper: does `facts` already have a composite FK column?
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

    // G1 — primary idempotency gate (SHAPE-based):
    if facts_has_composite_fk(conn)
        .await
        .map_err(step("g1_facts_fk_shape"))?
    {
        return Ok(());
    }

    // G2 — pre-condition gate (SHAPE-based): entities must carry the
    // composite PK installed by migrate_004.
    if !entities_has_composite_pk(conn)
        .await
        .map_err(step("g2_entities_pk_shape"))?
    {
        return Err(crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_006: entities table does not have composite PK (id, group_id) — \
             run migrate_004_composite_pk_entities first"
        )));
    }

    // G3 — partial migration recovery: facts_new exists.
    let facts_partial = table_exists(conn, "facts_new")
        .await
        .map_err(step("g3_check_facts_new"))?;

    // G4 — partial migration recovery: episodic_edges_new exists.
    let edges_partial = table_exists(conn, "episodic_edges_new")
        .await
        .map_err(step("g4_check_episodic_edges_new"))?;

    // PRAGMA foreign_keys = OFF for the duration of the restructure.
    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .map_err(step("fk_off"))?;
    let body_result: crate::core::error::Result<()> = async {

    // ── facts half ───────────────────────────────────────────────────────────

    if facts_partial {
        tracing::warn!(
            target: "kremory::migrations",
            "migrate_006: facts_new already exists — attempting to complete partial migration"
        );
    } else {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts_bak_006 AS SELECT * FROM facts",
            (),
        )
        .await
        .map_err(step("create_facts_bak_006"))?;

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

    conn.execute("DROP TABLE facts", ())
        .await
        .map_err(step("drop_facts"))?;
    conn.execute("ALTER TABLE facts_new RENAME TO facts", ())
        .await
        .map_err(step("rename_facts_new"))?;

    // Rebuild facts_fts (Vera 2026-05-28 #3): standalone FTS5 table; DROP TABLE facts
    // does NOT cascade. Rebuild from live data to prevent phantom FTS results.
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
    let _ = conn
        .execute(
            "INSERT INTO facts_fts (fact_id, predicate, object_value) \
             SELECT id, predicate, object_value FROM facts \
             WHERE object_value IS NOT NULL",
            (),
        )
        .await;

    // Re-create facts indexes.
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
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS facts_vec_idx \
             ON facts(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        )
        .await;
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
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodic_edges_bak_006 AS SELECT * FROM episodic_edges",
            (),
        )
        .await
        .map_err(step("create_episodic_edges_bak_006"))?;

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

    conn.execute("DROP TABLE episodic_edges", ())
        .await
        .map_err(step("drop_episodic_edges"))?;
    conn.execute("ALTER TABLE episodic_edges_new RENAME TO episodic_edges", ())
        .await
        .map_err(step("rename_episodic_edges_new"))?;

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

        Ok(())
    }
    .await;

    // Always restore PRAGMA foreign_keys = ON, regardless of body success/failure.
    if let Err(restore_err) = conn.execute("PRAGMA foreign_keys = ON", ()).await {
        tracing::error!(
            target: "kremory::migrations",
            error = %restore_err,
            "migrate_006: failed to re-enable PRAGMA foreign_keys after migration body — \
             connection FK state is now inconsistent (still OFF); operator must reconnect"
        );
    }

    body_result?;

    // Boy-scout integrity check: any FK violation introduced by the restructure
    // surfaces here BEFORE the next application write hits it.
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
