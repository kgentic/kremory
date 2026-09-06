use chrono::{DateTime, Utc};
use metrics::{counter, histogram};
use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::{EpisodicEdge, TemporalGraph};

use super::parse_dt;

/// Bundled parameters for [`TemporalGraph::insert_episode_with_group`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments). Construct
/// via [`EpisodeInsert::new`] + chainable setters. `group_id` is intentionally
/// NOT a field: the `*_with_group` variant takes it as a sibling arg so the
/// namespace capability stays explicit to that method (mirrors `FactInsert`).
pub struct EpisodeInsert<'a> {
    pub content: &'a str,
    pub timestamp: DateTime<Utc>,
    pub source_type: Option<&'a str>,
    pub metadata: Option<serde_json::Value>,
    pub saga_id: Option<&'a str>,
    pub sequence_number: Option<i64>,
    pub source_id: Option<&'a str>,
    pub source_uri: Option<&'a str>,
    pub recorded_at: Option<DateTime<Utc>>,
}

impl<'a> EpisodeInsert<'a> {
    /// Required fields; every optional defaults to `None`.
    pub fn new(content: &'a str, timestamp: DateTime<Utc>) -> Self {
        Self {
            content,
            timestamp,
            source_type: None,
            metadata: None,
            saga_id: None,
            sequence_number: None,
            source_id: None,
            source_uri: None,
            recorded_at: None,
        }
    }

    #[must_use]
    pub fn source_type(mut self, source_type: &'a str) -> Self {
        self.source_type = Some(source_type);
        self
    }

    #[must_use]
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    #[must_use]
    pub fn saga_id(mut self, saga_id: &'a str) -> Self {
        self.saga_id = Some(saga_id);
        self
    }

    #[must_use]
    pub fn sequence_number(mut self, sequence_number: i64) -> Self {
        self.sequence_number = Some(sequence_number);
        self
    }

    #[must_use]
    pub fn source_id(mut self, source_id: &'a str) -> Self {
        self.source_id = Some(source_id);
        self
    }

    #[must_use]
    pub fn source_uri(mut self, source_uri: &'a str) -> Self {
        self.source_uri = Some(source_uri);
        self
    }

    #[must_use]
    pub fn recorded_at(mut self, recorded_at: DateTime<Utc>) -> Self {
        self.recorded_at = Some(recorded_at);
        self
    }
}

/// Bundled parameters for [`TemporalGraph::insert_episode`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). Distinct from [`EpisodeInsert`]
/// (the builder for the `_with_group` variant) — this is the minimal four-column
/// insert path.
pub struct InsertEpisodeParams<'a> {
    pub content: &'a str,
    pub timestamp: DateTime<Utc>,
    pub source_type: Option<&'a str>,
    pub metadata: Option<serde_json::Value>,
}

/// Bundled parameters for [`TemporalGraph::prior_episodes_for_source`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
///
/// The thread key is `episodes.source_id`, which the facade writes from
/// [`SourceRef::id`](crate::memory::types::SourceRef) — i.e. exactly what
/// `remember(..).from_chat(id)` / `.from_document(id)` / `.from_source(id, kind)`
/// set. No new public surface is needed to thread a conversation: `source_id` is
/// already the documented threading primitive (`docs/api.md` §5.1) and already
/// carries an index (`idx_episodes_source_id`, Migration 007).
pub struct PriorEpisodesParams<'a> {
    /// Thread key — matched against `episodes.source_id`.
    pub source_id: &'a str,
    /// Namespace scope. `None` matches the `NULL` group, mirroring how
    /// [`TemporalGraph::insert_episode_with_group`] binds `group_id`.
    pub group_id: Option<&'a str>,
    /// Exclusive upper bound on episode id. Pass the id of the episode being
    /// ingested so it can never replay itself. Episode ids are `INTEGER PRIMARY
    /// KEY AUTOINCREMENT`, so id order is insertion order.
    pub before_id: i64,
    /// Maximum number of prior episodes to return. `0` short-circuits without
    /// touching the database.
    pub limit: usize,
}

/// Bundled parameters for [`TemporalGraph::insert_episodic_edge`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct InsertEpisodicEdgeParams<'a> {
    pub episode_id: i64,
    pub entity_id: &'a str,
    pub entity_group_id: Option<&'a str>,
    pub role: &'a str,
}

impl TemporalGraph {
    /// ADR-072 seq1 impl-spec §1: index one episode's verbatim content into
    /// `episodes_fts` (Migration 022, external-content FTS5 shadow of
    /// `episodes.content`, `content_rowid='id'`).
    ///
    /// Deliberately NOT transactionally coupled to the `episodes` INSERT
    /// (unlike `entities`+`entities_fts`, `core/graph/entities.rs:51`): the
    /// migration's backfill step is `WHERE NOT EXISTS`-guarded and re-runs on
    /// every `TemporalGraph::open*` call, so a crash between the two INSERTs
    /// self-heals on next open (see `migrations/defs_i.rs` module doc) — the
    /// same safety net every idempotent-backfill migration in this crate
    /// already relies on. Errors are NOT swallowed (Rule 19/21 — no silent
    /// `let _ = ...`): a real SQL failure here means the episode landed in the
    /// graph but is unsearchable via content-search until next reopen, which
    /// is a real consistency signal worth propagating.
    #[cfg(feature = "content-search")]
    async fn index_episode_content(&self, episode_id: i64, content: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO episodes_fts(rowid, content) VALUES (?1, ?2)",
                libsql::params![episode_id, content],
            )
            .await?;
        counter!("kremory.content_index.episode_indexed_total").increment(1);
        tracing::debug!(episode_id, "kremory.content_index.episode_indexed");
        Ok(())
    }

    pub async fn insert_episode(&self, params: InsertEpisodeParams<'_>) -> Result<i64> {
        let InsertEpisodeParams {
            content,
            timestamp,
            source_type,
            metadata,
        } = params;
        let _db_start = Instant::now();
        let ts_str = timestamp.to_rfc3339();
        let meta_str = metadata.as_ref().map(serde_json::to_string).transpose()?;
        self.conn
            .execute(
                "INSERT INTO episodes (content, timestamp, source_type, metadata) VALUES (?1, ?2, ?3, ?4)",
                libsql::params![content, ts_str, source_type, meta_str],
            )
            .await?;
        let mut rows = self.conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows
            .next()
            .await?
            .ok_or(crate::core::error::Error::InsertReturnedNoRowId {
                operation: "insert_episode",
            })?;
        let episode_id = row.get::<i64>(0)?;
        #[cfg(feature = "content-search")]
        self.index_episode_content(episode_id, content).await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_episode_ms").record(_ms);
        tracing::info!(_ms, episode_id, "kremory.db.insert_episode");
        Ok(episode_id)
    }

    // === Extended Insert Methods (group/saga/sequence variants) ===

    /// Insert an episode with optional group_id, saga_id, sequence_number,
    /// source_id, source_uri, and recorded_at (Migration 007 columns).
    ///
    /// `group_id` is a sibling arg (not an [`EpisodeInsert`] field) so the
    /// namespace capability stays explicit to this method — mirrors
    /// `insert_fact_with_group`.
    pub async fn insert_episode_with_group(
        &self,
        episode: EpisodeInsert<'_>,
        group_id: Option<&str>,
    ) -> Result<i64> {
        let EpisodeInsert {
            content,
            timestamp,
            source_type,
            metadata,
            saga_id,
            sequence_number,
            source_id,
            source_uri,
            recorded_at,
        } = episode;
        let _db_start = Instant::now();
        let ts_str = timestamp.to_rfc3339();
        let meta_str = metadata.as_ref().map(serde_json::to_string).transpose()?;
        // `recorded_at` is NOT NULL DEFAULT (datetime('now')) in the schema.
        // When the caller does not supply a value we default to Utc::now() so the
        // explicit column reference never sends NULL and bypasses the SQL DEFAULT.
        let recorded_at_str = recorded_at.unwrap_or_else(Utc::now).to_rfc3339();
        // TD-003 Phase G (ADR-042): compute SHA-256 content hash at insert time.
        // Migration 011 backfills existing rows; new rows are hashed here so the
        // column is always populated from v0.1.7 onwards.
        let content_hash = format!("{:x}", Sha256::digest(content.as_bytes()));
        self.conn
            .execute(
                "INSERT INTO episodes \
                 (content, timestamp, source_type, metadata, group_id, saga_id, sequence_number, \
                  source_id, source_uri, recorded_at, content_hash) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                libsql::params![
                    content,
                    ts_str,
                    source_type,
                    meta_str,
                    group_id,
                    saga_id,
                    sequence_number,
                    source_id,
                    source_uri,
                    recorded_at_str,
                    content_hash
                ],
            )
            .await?;
        let mut rows = self.conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows
            .next()
            .await?
            .ok_or(crate::core::error::Error::InsertReturnedNoRowId {
                operation: "insert_episode_with_group",
            })?;
        let episode_id = row.get::<i64>(0)?;
        #[cfg(feature = "content-search")]
        self.index_episode_content(episode_id, content).await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_episode_with_group_ms").record(_ms);
        tracing::info!(_ms, episode_id, "kremory.db.insert_episode_with_group");
        Ok(episode_id)
    }

    /// TD-136 (dense episode retrieval): set/update the embedding vector for
    /// one episode row, mirroring [`TemporalGraph::set_entity_embedding`] /
    /// `set_fact_embedding` exactly (JSON-array `vector()` UPDATE keyed by the
    /// episode's `INTEGER PRIMARY KEY`). Feature-gated behind `content-search`
    /// — the `episodes.embedding` column only exists in that build
    /// (Migration 026). Called at ingest (when the dense arm is enabled) and by
    /// the `Memory::{backfill_episode_embeddings, reembed_all_episode_embeddings}`
    /// maintenance paths (TD-136 gap-fill and TD-143 full re-embed, respectively).
    #[cfg(feature = "content-search")]
    pub async fn set_episode_embedding(&self, episode_id: i64, embedding: &[f32]) -> Result<()> {
        let _db_start = Instant::now();
        // Convert f32 slice to JSON array string for the vector() SQL function
        // (identical shape to set_entity_embedding / set_fact_embedding).
        let vec_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        self.conn
            .execute(
                "UPDATE episodes SET embedding = vector(?1) WHERE id = ?2",
                libsql::params![vec_str, episode_id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.set_episode_embedding_ms").record(_ms);
        tracing::debug!(_ms, episode_id, "kremory.db.set_episode_embedding");
        Ok(())
    }

    /// TD-136: select up to `limit` episodes whose `embedding` is still NULL,
    /// for the `Memory::backfill_episode_embeddings` maintenance loop. Returns
    /// `(episode_id, content)` pairs. Feature-gated behind `content-search`
    /// (the column only exists there).
    ///
    /// Selects real data columns (`id`, `content`) — NOT `COUNT(*)`/rowid-only —
    /// so the libsql DiskANN vector-index `COUNT(*)`-returns-0 trap
    /// (SYSTEM-PRIMER §2; `episodes` gains `episodes_vec_idx` in Migration 026)
    /// cannot silently zero out the backfill set. The `WHERE embedding IS NULL`
    /// predicate makes this converge: each backfilled episode drops out of the
    /// next page.
    #[cfg(feature = "content-search")]
    pub async fn episodes_missing_embedding(&self, limit: usize) -> Result<Vec<(i64, String)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content FROM episodes \
                 WHERE embedding IS NULL \
                 ORDER BY id ASC \
                 LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get::<i64>(0)?;
            let content: String = row.get::<String>(1)?;
            out.push((id, content));
        }
        Ok(out)
    }

    /// TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143): select up
    /// to `limit` episodes with `id > after_id`, ordered by `id` ASC, for the
    /// `Memory::reembed_all_episode_embeddings` maintenance loop. Returns
    /// `(episode_id, content)` pairs.
    ///
    /// Unlike [`episodes_missing_embedding`](Self::episodes_missing_embedding),
    /// this is NOT filtered by embedding state — it pages through EVERY
    /// episode row, including ones that already carry an embedding. That is
    /// the whole point: `episodes_missing_embedding`'s `WHERE embedding IS
    /// NULL` predicate can only ever fill a gap, never overwrite an existing
    /// vector, so it cannot serve a full re-embed (e.g. after flipping
    /// [`SearchConfig::embed_task_prefix_enabled`](crate::core::config::SearchConfig::embed_task_prefix_enabled)
    /// or swapping the embedder/dimension — see TD-112 for the sibling
    /// entity-embedding staleness problem).
    ///
    /// Because there is no filter for the caller's loop to self-consume, the
    /// caller MUST advance `after_id` to the last id in the returned page
    /// (an id-cursor) rather than re-issuing the same `LIMIT` — a fixed
    /// `LIMIT` with no cursor would fetch the same first page forever, since
    /// re-embedding a row doesn't remove it from an unfiltered result set.
    /// `after_id = 0` starts from the beginning (episode ids are `INTEGER
    /// PRIMARY KEY`, always `>= 1`). An empty result means the cursor has
    /// reached the end of the table.
    #[cfg(feature = "content-search")]
    pub async fn episodes_after_id(
        &self,
        after_id: i64,
        limit: usize,
    ) -> Result<Vec<(i64, String)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content FROM episodes \
                 WHERE id > ?1 \
                 ORDER BY id ASC \
                 LIMIT ?2",
                libsql::params![after_id, limit as i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get::<i64>(0)?;
            let content: String = row.get::<String>(1)?;
            out.push((id, content));
        }
        Ok(out)
    }

    /// Fetch the contents of the most recent episodes sharing `source_id`
    /// within `group_id`, strictly before `before_id`, in CHRONOLOGICAL order.
    ///
    /// This is the read half of prior-turn replay (ADR-080): when ingesting turn
    /// N of a conversation, the extractor is shown the preceding turns so that
    /// references resolve ("I prefer that one" has nothing to resolve against on
    /// its own). Both live competitors do this and converged on N=10 —
    /// mem0 replays the last 10 rows of its `messages` table for the session
    /// (`mem0/memory/main.py:920`), Graphiti the last 10 episodes for the group
    /// (`graphiti_core/graphiti.py:1086`). See
    /// `.ai-docs/research/competitive-landscape/write-path-teardown-2026-09-06.md`.
    ///
    /// # Ordering
    ///
    /// Selected `ORDER BY id DESC LIMIT n` (the *latest* n) then reversed in
    /// Rust to ascending, so the extractor reads them oldest-first. This mirrors
    /// mem0's `created_at DESC` + re-sort ASC (`storage.py:298-313`). `id` is
    /// used rather than `timestamp` because `timestamp` is the caller-supplied
    /// world clock and may be absent, equal, or non-monotonic across turns,
    /// whereas `id` is `INTEGER PRIMARY KEY AUTOINCREMENT` and therefore always
    /// reflects true insertion order.
    ///
    /// # Namespace scoping
    ///
    /// `group_id` uses `IS` rather than `=` so `None` matches the `NULL` group.
    /// A plain `=` would silently return zero rows for un-namespaced ingests,
    /// because `NULL = NULL` is `NULL` in SQL, not true.
    ///
    /// Returns an empty vec when `limit == 0` without issuing a query, which is
    /// the off switch for the whole feature
    /// (`PipelineConfig::prior_turn_replay_depth = 0`).
    pub async fn prior_episodes_for_source(
        &self,
        params: PriorEpisodesParams<'_>,
    ) -> Result<Vec<String>> {
        let PriorEpisodesParams {
            source_id,
            group_id,
            before_id,
            limit,
        } = params;
        if limit == 0 || source_id.is_empty() {
            return Ok(Vec::new());
        }
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT content FROM episodes \
                 WHERE source_id = ?1 \
                   AND group_id IS ?2 \
                   AND id < ?3 \
                 ORDER BY id DESC \
                 LIMIT ?4",
                libsql::params![source_id, group_id, before_id, limit as i64],
            )
            .await?;
        let mut out: Vec<String> = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row.get::<String>(0)?);
        }
        // DESC-then-reverse: the query takes the LATEST `limit`, the reverse
        // hands them to the extractor oldest-first.
        out.reverse();
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.prior_episodes_for_source_ms").record(_ms);
        counter!("kremory.replay.prior_episodes_fetched_total").increment(out.len() as u64);
        tracing::debug!(
            _ms,
            source_id,
            before_id,
            returned = out.len(),
            "kremory.db.prior_episodes_for_source"
        );
        Ok(out)
    }

    /// Resolve the `source_id` (conversation thread key) of one episode.
    ///
    /// Exists for the DEFERRED (background) ingest path, whose
    /// `IngestDeferredParams` carries `episode_id` but not the source — see
    /// that struct's doc comment, which records the same plumbing gap for
    /// `declared_reference_time`. Resolving it here keeps prior-turn replay
    /// (ADR-080) behaving IDENTICALLY inline and in background mode; a silent
    /// difference between the two would be worse than the missing plumbing,
    /// because it would make extraction quality depend on which path a caller
    /// happened to take.
    ///
    /// Returns `None` for a missing episode or a NULL `source_id`.
    pub async fn source_id_for_episode(&self, episode_id: i64) -> Result<Option<String>> {
        let mut rows = self
            .conn
            .query(
                "SELECT source_id FROM episodes WHERE id = ?1",
                libsql::params![episode_id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(row.get::<Option<String>>(0)?),
            None => Ok(None),
        }
    }

    // === Episodic Edge Methods ===

    /// Insert an episodic edge (MENTIONS link from episode to entity).
    ///
    /// `entity_group_id` MUST be the namespace the referenced entity was stored
    /// under (`None` ⇒ `'default'`). Migration 006 added the composite FK
    /// `(entity_id, entity_group_id) REFERENCES entities(id, group_id)`; the
    /// `entity_group_id` column defaults to `'default'`. Omitting the namespace
    /// for a non-`'default'` entity makes the row reference `(entity_id,
    /// 'default')`, for which no parent row exists — the FK then VIOLATES and the
    /// edge silently fails to insert in namespaced mode. Threading the entity's
    /// real namespace here lets the composite FK resolve so edges persist (and
    /// `on_edge_added` fires) for namespaced ingests too.
    pub async fn insert_episodic_edge(&self, params: InsertEpisodicEdgeParams<'_>) -> Result<i64> {
        let InsertEpisodicEdgeParams {
            episode_id,
            entity_id,
            entity_group_id,
            role,
        } = params;
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        // None ⇒ 'default', matching the entities-table convention
        // (`insert_entity_with_group`) so the composite FK lines up.
        let effective_group_id = entity_group_id.unwrap_or("default");
        // Presence-uniqueness invariant (migrate_017): UNIQUE(episode_id,
        // entity_id, entity_group_id). `INSERT OR IGNORE` makes the write
        // idempotent — re-asserting that an entity appears in an episode is a
        // no-op, not an error. The inline path already keeps presence
        // single-owned via `entity_loop_ids` (so this should rarely fire there);
        // it is the correct mechanism for the data-dependent overlaps the deferred
        // path and canonicalization merges can produce.
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO episodic_edges (episode_id, entity_id, entity_group_id, role, recorded_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![episode_id, entity_id, effective_group_id, role, now],
        ).await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_episodic_edge_ms").record(_ms);
        if changed == 0 {
            // Duplicate presence edge suppressed by the UNIQUE constraint. Emit an
            // observable signal (R19) — this counter should read ~zero on the
            // inline path; sustained increments mean a writer is re-asserting
            // presence that another writer already owns. Return the EXISTING edge
            // id (not a stale last_insert_rowid) so callers keep a valid handle.
            counter!("kremory.ingest.episodic_edge_dup_suppressed_total").increment(1);
            let mut rows = self.conn.query(
                "SELECT id FROM episodic_edges WHERE episode_id = ?1 AND entity_id = ?2 AND entity_group_id = ?3 LIMIT 1",
                libsql::params![episode_id, entity_id, effective_group_id],
            ).await?;
            if let Some(row) = rows.next().await? {
                let existing_id = row.get::<i64>(0)?;
                tracing::debug!(
                    _ms,
                    edge_id = existing_id,
                    "kremory.db.insert_episodic_edge.deduped"
                );
                return Ok(existing_id);
            }
            // changed==0 means the UNIQUE constraint suppressed the insert, so the
            // conflicting row MUST exist — a missing row is a structural violation
            // (constraint changed under us, or a race). Surface it loudly rather
            // than fall through to `last_insert_rowid()`, which would return the
            // stale rowid of a prior, unrelated insert on this connection.
            return Err(crate::core::error::Error::Other(anyhow::anyhow!(
                "insert_episodic_edge: INSERT OR IGNORE suppressed a row but no \
                 existing (episode_id={episode_id}, entity_id={entity_id}, \
                 entity_group_id={effective_group_id}) edge was found"
            )));
        }
        let edge_id = self.conn.last_insert_rowid();
        tracing::info!(_ms, edge_id, "kremory.db.insert_episodic_edge");
        Ok(edge_id)
    }

    /// Get all episodic edges for an entity.
    pub async fn episodic_edges_for_entity(&self, entity_id: &str) -> Result<Vec<EpisodicEdge>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, episode_id, entity_id, role, recorded_at FROM episodic_edges WHERE entity_id = ?1",
                libsql::params![entity_id],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get::<i64>(0)?;
            let episode_id: i64 = row.get::<i64>(1)?;
            let eid: String = row.get::<String>(2)?;
            let role: String = row.get::<String>(3)?;
            let recorded_str: String = row.get::<String>(4)?;
            let recorded_at = parse_dt(&recorded_str)?;
            edges.push(EpisodicEdge {
                id,
                episode_id,
                entity_id: eid,
                role,
                recorded_at,
                // ADR-029b: composite FK field — absent on pre-migration-004 rows.
                entity_group_id: None,
            });
        }
        Ok(edges)
    }
}
