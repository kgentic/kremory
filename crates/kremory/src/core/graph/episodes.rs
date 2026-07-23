use chrono::{DateTime, Utc};
use metrics::{counter, histogram};
use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::{Episode, EpisodicEdge, TemporalGraph};

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
    /// the `Memory::backfill_episode_embeddings` maintenance path.
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

    // === Extended Fact Methods ===

    /// Return all episodes in a given `group_id`, ordered by timestamp ascending.
    ///
    /// Used by the L7 reclassification pass to build the contextual evidence
    /// set for an entity.  Returns an empty `Vec` when the group has no episodes.
    pub async fn get_episodes_in_group(&self, group_id: &str) -> Result<Vec<Episode>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                // TD-003 Phase G: added source_id (idx 10) + source_uri (idx 11) to
                // close Episode struct ↔ table column asymmetry.
                "SELECT id, content, timestamp, source_type, metadata, group_id, saga_id,
                        sequence_number, content_hash, recorded_at, source_id, source_uri
                 FROM episodes
                 WHERE group_id = ?1
                 ORDER BY timestamp ASC",
                libsql::params![group_id],
            )
            .await?;
        let mut episodes = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get::<i64>(0)?;
            let content: String = row.get::<String>(1)?;
            let ts_str: String = row.get::<String>(2)?;
            let timestamp = parse_dt(&ts_str)?;
            let source_type: Option<String> = row.get::<Option<String>>(3)?;
            let meta_str: Option<String> = row.get::<Option<String>>(4)?;
            let metadata: Option<serde_json::Value> = meta_str
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok());
            let gid: Option<String> = row.get::<Option<String>>(5)?;
            let saga_id: Option<String> = row.get::<Option<String>>(6)?;
            let seq: Option<i64> = row.get::<Option<i64>>(7)?;
            let content_hash: Option<String> = row.get::<Option<String>>(8)?;
            let recorded_at: Option<String> = row.get::<Option<String>>(9)?;
            let source_id: Option<String> = row.get::<Option<String>>(10)?;
            let source_uri: Option<String> = row.get::<Option<String>>(11)?;
            episodes.push(Episode {
                id,
                content,
                timestamp,
                source_type,
                metadata,
                group_id: gid,
                saga_id,
                sequence_number: seq,
                content_hash,
                recorded_at,
                source_id,
                source_uri,
            });
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let count = episodes.len();
        histogram!("rql.db.get_episodes_in_group_ms").record(_ms);
        tracing::info!(_ms, count, group_id, "kremory.db.get_episodes_in_group");
        Ok(episodes)
    }
}
