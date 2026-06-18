use chrono::{DateTime, Utc};
use metrics::histogram;
use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::{Episode, EpisodicEdge, TemporalGraph};

use super::parse_dt;

impl TemporalGraph {
    pub async fn insert_episode(
        &self,
        content: &str,
        timestamp: DateTime<Utc>,
        source_type: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<i64> {
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
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_episode_ms").record(_ms);
        tracing::info!(_ms, episode_id, "kremory.db.insert_episode");
        Ok(episode_id)
    }

    // === Extended Insert Methods (group/saga/sequence variants) ===

    /// Insert an episode with optional group_id, saga_id, sequence_number,
    /// source_id, source_uri, and recorded_at (Migration 007 columns).
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_episode_with_group(
        &self,
        content: &str,
        timestamp: DateTime<Utc>,
        source_type: Option<&str>,
        metadata: Option<serde_json::Value>,
        group_id: Option<&str>,
        saga_id: Option<&str>,
        sequence_number: Option<i64>,
        source_id: Option<&str>,
        source_uri: Option<&str>,
        recorded_at: Option<DateTime<Utc>>,
    ) -> Result<i64> {
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
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_episode_with_group_ms").record(_ms);
        tracing::info!(_ms, episode_id, "kremory.db.insert_episode_with_group");
        Ok(episode_id)
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
    pub async fn insert_episodic_edge(
        &self,
        episode_id: i64,
        entity_id: &str,
        entity_group_id: Option<&str>,
        role: &str,
    ) -> Result<i64> {
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        // None ⇒ 'default', matching the entities-table convention
        // (`insert_entity_with_group`) so the composite FK lines up.
        let effective_group_id = entity_group_id.unwrap_or("default");
        self.conn.execute(
            "INSERT INTO episodic_edges (episode_id, entity_id, entity_group_id, role, recorded_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![episode_id, entity_id, effective_group_id, role, now],
        ).await?;
        let edge_id = self.conn.last_insert_rowid();
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_episodic_edge_ms").record(_ms);
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
