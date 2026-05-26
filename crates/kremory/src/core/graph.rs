use chrono::{DateTime, Utc};
use metrics::histogram;
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use tracing;

use crate::core::error::Result;
use crate::core::schema::{Entity, EpisodicEdge, Fact, TemporalGraph};

/// Compute a hex-encoded SHA-256 content hash for a fact triple.
/// Hash input: `"{subject_id}\x00{predicate}\x00{object_key}"` where
/// `object_key` is `object_id` if set, otherwise `object_value`, otherwise `""`.
/// Story #209.
fn fact_content_hash(
    subject_id: &str,
    predicate: &str,
    object_id: Option<&str>,
    object_value: Option<&str>,
) -> String {
    let object_key = object_id.or(object_value).unwrap_or("");
    let mut hasher = Sha256::new();
    hasher.update(subject_id.as_bytes());
    hasher.update(b"\x00");
    hasher.update(predicate.as_bytes());
    hasher.update(b"\x00");
    hasher.update(object_key.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone)]
pub struct SubGraph {
    pub entities: Vec<Entity>,
    pub facts: Vec<Fact>,
}

fn parse_dt(s: &str) -> anyhow::Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(s)
        .map_err(|e| anyhow::anyhow!("bad timestamp '{}': {}", s, e))?
        .with_timezone(&Utc))
}

#[allow(dead_code)]
fn parse_dt_opt(s: Option<String>) -> anyhow::Result<Option<DateTime<Utc>>> {
    match s {
        None => Ok(None),
        Some(v) => Ok(Some(parse_dt(&v)?)),
    }
}

fn row_to_entity(row: &libsql::Row) -> anyhow::Result<Entity> {
    let id: String = row.get::<String>(0)?;
    let label: String = row.get::<String>(1)?;
    let props_str: Option<String> = row.get::<Option<String>>(2)?;
    let created_str: String = row.get::<String>(3)?;
    let updated_str: Option<String> = row.get::<Option<String>>(4)?;
    let group_id: Option<String> = row.get::<Option<String>>(5)?;

    let properties: serde_json::Value = props_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);
    let recorded_at = parse_dt(&created_str)?;
    let updated_at = updated_str.as_deref().map(parse_dt).transpose()?;

    Ok(Entity {
        id,
        label,
        properties,
        recorded_at,
        updated_at,
        group_id,
    })
}

fn row_to_fact(row: &libsql::Row) -> anyhow::Result<Fact> {
    let id: i64 = row.get::<i64>(0)?;
    let subject_id: String = row.get::<String>(1)?;
    let predicate: String = row.get::<String>(2)?;
    let object_id: Option<String> = row.get::<Option<String>>(3)?;
    let object_value: Option<String> = row.get::<Option<String>>(4)?;
    let props_str: Option<String> = row.get::<Option<String>>(5)?;
    let valid_from_str: String = row.get::<String>(6)?;
    let valid_to_str: Option<String> = row.get::<Option<String>>(7)?;
    let recorded_str: String = row.get::<String>(8)?;
    let expired_str: Option<String> = row.get::<Option<String>>(9)?;
    let invalid_str: Option<String> = row.get::<Option<String>>(10)?;
    let group_id: Option<String> = row.get::<Option<String>>(11)?;
    let confidence: f64 = row.get::<f64>(12)?;
    let source_episode_id: Option<i64> = row.get::<Option<i64>>(13)?;
    let memory_type_str: Option<String> = row.get::<Option<String>>(14)?;
    let content_hash: Option<String> = row.get::<Option<String>>(15)?;
    let access_count: i64 = row.get::<i64>(16)?;

    let properties: Option<serde_json::Value> = props_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let valid_from = parse_dt(&valid_from_str)?;
    let valid_to = valid_to_str.as_deref().map(parse_dt).transpose()?;
    let recorded_at = parse_dt(&recorded_str)?;
    let expired_at = expired_str.as_deref().map(parse_dt).transpose()?;
    let invalid_at = invalid_str.as_deref().map(parse_dt).transpose()?;
    let memory_type = memory_type_str
        .as_deref()
        .and_then(|s| serde_json::from_str(&format!("\"{s}\"")).ok());

    Ok(Fact {
        id,
        subject_id,
        predicate,
        object_id,
        object_value,
        properties,
        valid_from,
        valid_to,
        recorded_at,
        expired_at,
        invalid_at,
        group_id,
        confidence,
        source_episode_id,
        memory_type,
        content_hash,
        access_count,
    })
}

impl TemporalGraph {
    // === Entity CRUD ===

    pub async fn insert_entity(
        &self,
        id: &str,
        label: &str,
        properties: serde_json::Value,
    ) -> Result<()> {
        let _db_start = Instant::now();
        let props_str = serde_json::to_string(&properties)?;
        let now = Utc::now().to_rfc3339();
        // Transaction guards the invariant: row in `entities` ⟹ row in `entities_fts`.
        // Without a transaction, a crash or FTS error between the two INSERTs leaves an
        // entity permanently invisible to FTS search (HIGH-2 remediation, 2026-05-15).
        self.conn.execute("BEGIN", ()).await?;
        let inner: Result<()> = async {
            self.conn
                .execute(
                    "INSERT INTO rql_entities (id, label, properties, recorded_at) VALUES (?1, ?2, ?3, ?4)",
                    libsql::params![id, label, props_str.clone(), now],
                )
                .await?;
            self.conn
                .execute(
                    "INSERT INTO rql_entities_fts(entity_id, label, properties) VALUES (?1, ?2, ?3)",
                    libsql::params![id, label, props_str],
                )
                .await?;
            Ok(())
        }
        .await;
        match inner {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                // Best-effort rollback; propagate the original error regardless.
                let _ = self.conn.execute("ROLLBACK", ()).await;
                return Err(e);
            }
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_entity_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.insert_entity");
        Ok(())
    }

    /// Store a pre-computed embedding vector for an entity.
    ///
    /// The embedding is stored in the `embedding` column (F32_BLOB(384)) and
    /// indexed by the `rql_entities_vec_idx` for cosine-similarity search.
    pub async fn update_entity_embedding(&self, id: &str, embedding: &[f32]) -> Result<()> {
        let _db_start = Instant::now();
        let vec_str = format!(
            "vector32('[{}]')",
            embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        self.conn
            .execute(
                &format!(
                    "UPDATE rql_entities SET embedding = {} WHERE id = ?1",
                    vec_str
                ),
                libsql::params![id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.update_entity_embedding_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.update_entity_embedding");
        Ok(())
    }

    pub async fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT id, label, properties, recorded_at, updated_at, group_id FROM rql_entities WHERE id = ?1",
                libsql::params![id],
            )
            .await?;
        let result = match rows.next().await? {
            None => Ok(None),
            Some(row) => Ok(Some(row_to_entity(&row)?)),
        };
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.get_entity_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.get_entity");
        result
    }

    pub async fn update_entity(&self, id: &str, properties: serde_json::Value) -> Result<()> {
        let _db_start = Instant::now();
        let props_str = serde_json::to_string(&properties)?;
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "UPDATE rql_entities SET properties = ?1, updated_at = ?2 WHERE id = ?3",
                libsql::params![props_str, now, id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.update_entity_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.update_entity");
        Ok(())
    }

    /// Update an entity's group_id and properties.
    /// Used when re-indexing a file into a different scope.
    pub async fn update_entity_group(
        &self,
        id: &str,
        group_id: Option<&str>,
        properties: serde_json::Value,
    ) -> Result<()> {
        let _db_start = Instant::now();
        let props_str = serde_json::to_string(&properties)?;
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "UPDATE rql_entities SET group_id = ?1, properties = ?2, updated_at = ?3 WHERE id = ?4",
                libsql::params![group_id, props_str, now, id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.update_entity_group_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.update_entity_group");
        Ok(())
    }

    /// Set an entity's `group_id` WITHOUT touching `properties`.
    ///
    /// Used by the ADR-A dual-store coupling co-mutation path in
    /// `the-host-application::repo::sqlite_entities::entity_move`. The entity_move
    /// flow rewrites scope membership for every chunk of a moved entity
    /// — properties (the chunk text + source metadata) must stay intact;
    /// only the scope filter (`group_id`) changes.
    ///
    /// Returns the number of rows updated (`0` when `id` doesn't exist
    /// in `rql_entities` — caller decides whether that is a legitimate
    /// no-op or an integrity failure).
    pub async fn set_entity_group_only(&self, id: &str, group_id: Option<&str>) -> Result<u64> {
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        let n = self
            .conn
            .execute(
                "UPDATE rql_entities SET group_id = ?1, updated_at = ?2 WHERE id = ?3",
                libsql::params![group_id, now, id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.set_entity_group_only_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.set_entity_group_only");
        Ok(n)
    }

    /// Delete a list of entities by exact ID, cleaning both `entities` and `entities_fts`.
    ///
    /// Used for cross-DB cleanup of UUID-v7 chunk entities tracked in workspace.db's
    /// `entity_chunks` join table — where the chunk entity ID has no prefix relationship
    /// to the parent entity ID, so `delete_entities_by_prefix` does not catch them.
    ///
    /// `entities_fts` is a standalone FTS5 virtual table (no `content=` parameter).
    /// SQLite does NOT auto-cascade deletes from `entities` into it. This function
    /// deletes from `entities_fts` first, then from `entities`, to ensure no zombie
    /// FTS rows remain after re-index (HIGH-1 remediation, 2026-05-15).
    ///
    /// Batches deletes at 500 IDs to stay well below SQLite's 32766-variable limit.
    pub async fn delete_entities_by_ids(&self, ids: &[String]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        const BATCH_SIZE: usize = 500;
        let mut total: u64 = 0;
        for batch in ids.chunks(BATCH_SIZE) {
            let placeholders = vec!["?"; batch.len()].join(",");
            let params: Vec<libsql::Value> = batch
                .iter()
                .map(|s| libsql::Value::Text(s.clone()))
                .collect();
            // FTS first — mirrors delete_entities_by_prefix idiom (see comment there).
            // Standalone FTS5 has no FK, but deleting FTS before entities prevents any
            // future trigger-based coupling from reversing the order unexpectedly.
            self.conn
                .execute(
                    &format!("DELETE FROM rql_entities_fts WHERE entity_id IN ({placeholders})"),
                    params.clone(),
                )
                .await?;
            let n = self
                .conn
                .execute(
                    &format!("DELETE FROM rql_entities WHERE id IN ({placeholders})"),
                    params,
                )
                .await?;
            total += n;
        }
        Ok(total)
    }

    /// Delete all entities whose ID starts with `prefix`.
    ///
    /// Removes from both `entities` (+ vector index) and `entities_fts`.
    /// Used to clear stale chunks when re-indexing a document.
    pub async fn delete_entities_by_prefix(&self, prefix: &str) -> Result<u64> {
        let _db_start = Instant::now();
        let pattern = format!("{prefix}%");

        // FTS first — references entity_id which must still exist for the join.
        self.conn
            .execute(
                "DELETE FROM rql_entities_fts WHERE entity_id LIKE ?1",
                libsql::params![pattern.clone()],
            )
            .await?;

        let deleted = self
            .conn
            .execute(
                "DELETE FROM rql_entities WHERE id LIKE ?1",
                libsql::params![pattern],
            )
            .await?;

        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.delete_entities_by_prefix_ms").record(_ms);
        tracing::info!(_ms, deleted, "kremory.db.delete_entities_by_prefix");
        Ok(deleted)
    }

    pub async fn list_entities(&self) -> Result<Vec<Entity>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT id, label, properties, recorded_at, updated_at, group_id FROM rql_entities",
                (),
            )
            .await?;
        let mut entities = Vec::new();
        while let Some(row) = rows.next().await? {
            entities.push(row_to_entity(&row)?);
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let count = entities.len();
        histogram!("rql.db.list_entities_count").record(count as f64);
        histogram!("rql.db.list_entities_ms").record(_ms);
        tracing::info!(_ms, count, "kremory.db.list_entities");
        Ok(entities)
    }

    // === Fact CRUD ===

    #[allow(clippy::too_many_arguments)]
    pub async fn insert_fact(
        &self,
        subject_id: &str,
        predicate: &str,
        object_id: Option<&str>,
        object_value: Option<&str>,
        valid_from: DateTime<Utc>,
        confidence: f64,
        source_episode_id: Option<i64>,
        embedding: Option<&[f32]>,
    ) -> Result<i64> {
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        let valid_from_str = valid_from.to_rfc3339();
        // Story #209: compute SHA-256 content hash for dedup
        let hash = fact_content_hash(subject_id, predicate, object_id, object_value);
        // Check for existing non-expired fact with same content hash
        let mut dup_check = self
            .conn
            .query(
                "SELECT id FROM facts WHERE content_hash = ?1 AND expired_at IS NULL LIMIT 1",
                libsql::params![hash.clone()],
            )
            .await?;
        if dup_check.next().await?.is_some() {
            return Err(crate::core::error::Error::Duplicate { content_hash: hash });
        }
        let vec_str = embedding.map(|e| {
            format!(
                "[{}]",
                e.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        });
        self.conn
            .execute(
                "INSERT INTO facts (subject_id, predicate, object_id, object_value, embedding, valid_from, recorded_at, confidence, source_episode_id, content_hash)
                 VALUES (?1, ?2, ?3, ?4, CASE WHEN ?5 IS NULL THEN NULL ELSE vector(?5) END, ?6, ?7, ?8, ?9, ?10)",
                libsql::params![
                    subject_id,
                    predicate,
                    object_id,
                    object_value,
                    vec_str,
                    valid_from_str,
                    now,
                    confidence,
                    source_episode_id,
                    hash,
                ],
            )
            .await?;
        let mut rows = self.conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows
            .next()
            .await?
            .ok_or(crate::core::error::Error::InsertReturnedNoRowId {
                operation: "insert_fact",
            })?;
        let fact_id = row.get::<i64>(0)?;
        if let Some(ov) = object_value {
            self.conn
                .execute(
                    "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
                    libsql::params![fact_id, predicate, ov],
                )
                .await?;
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_fact_ms").record(_ms);
        tracing::info!(_ms, fact_id, "kremory.db.insert_fact");
        Ok(fact_id)
    }

    pub async fn invalidate_fact(&self, fact_id: i64, at: DateTime<Utc>) -> Result<()> {
        let _db_start = Instant::now();
        let at_str = at.to_rfc3339();
        self.conn
            .execute(
                "UPDATE facts SET expired_at = ?1 WHERE id = ?2",
                libsql::params![at_str, fact_id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.invalidate_fact_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.invalidate_fact");
        Ok(())
    }

    // === Temporal Queries ===

    pub async fn facts_at(&self, time: DateTime<Utc>) -> Result<Vec<Fact>> {
        let _db_start = Instant::now();
        let t = time.to_rfc3339();
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE valid_from <= ?1
                   AND (valid_to IS NULL OR valid_to > ?1)
                   AND expired_at IS NULL",
                libsql::params![t],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let count = facts.len();
        histogram!("rql.db.facts_at_count").record(count as f64);
        histogram!("rql.db.facts_at_ms").record(_ms);
        tracing::info!(_ms, count, "kremory.db.facts_at");
        Ok(facts)
    }

    pub async fn entity_facts_at(&self, entity_id: &str, time: DateTime<Utc>) -> Result<Vec<Fact>> {
        let t = time.to_rfc3339();
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE subject_id = ?1
                   AND valid_from <= ?2
                   AND (valid_to IS NULL OR valid_to > ?2)
                   AND expired_at IS NULL",
                libsql::params![entity_id, t],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        Ok(facts)
    }

    pub async fn entity_history(&self, entity_id: &str) -> Result<Vec<Fact>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE subject_id = ?1
                 ORDER BY valid_from, recorded_at",
                libsql::params![entity_id],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let count = facts.len();
        histogram!("rql.db.entity_history_count").record(count as f64);
        histogram!("rql.db.entity_history_ms").record(_ms);
        tracing::info!(_ms, count, "kremory.db.entity_history");
        Ok(facts)
    }

    // === Graph Traversal ===

    pub async fn get_neighbours(&self, entity_id: &str, hops: u32) -> Result<SubGraph> {
        let _db_start = Instant::now();
        let mut visited_entities: HashSet<String> = HashSet::new();
        let mut collected_facts: Vec<Fact> = Vec::new();
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();

        visited_entities.insert(entity_id.to_string());
        queue.push_back((entity_id.to_string(), 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }

            // Find all non-expired facts where this entity is subject or object
            let current_id_str = current_id.clone();
            let mut rows = self
                .conn
                .query(
                    "SELECT id, subject_id, predicate, object_id, object_value, properties,
                            valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                     FROM facts
                     WHERE (subject_id = ?1 OR object_id = ?1)
                       AND expired_at IS NULL",
                    libsql::params![current_id_str],
                )
                .await?;

            let mut batch: Vec<Fact> = Vec::new();
            while let Some(row) = rows.next().await? {
                batch.push(row_to_fact(&row)?);
            }

            for fact in batch {
                // Collect connected entity IDs we haven't visited
                let neighbour_id = if fact.subject_id == current_id {
                    fact.object_id.clone()
                } else {
                    Some(fact.subject_id.clone())
                };

                collected_facts.push(fact);

                if let Some(nid) = neighbour_id {
                    if !visited_entities.contains(&nid) {
                        visited_entities.insert(nid.clone());
                        queue.push_back((nid, depth + 1));
                    }
                }
            }
        }

        // Deduplicate facts by id
        collected_facts.sort_by_key(|f| f.id);
        collected_facts.dedup_by_key(|f| f.id);

        // Load all discovered entities
        let mut entities = Vec::new();
        for eid in &visited_entities {
            if let Some(entity) = self.get_entity(eid).await? {
                entities.push(entity);
            }
        }

        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let entity_count = entities.len();
        let fact_count = collected_facts.len();
        histogram!("rql.db.get_neighbours_entities").record(entity_count as f64);
        histogram!("rql.db.get_neighbours_facts").record(fact_count as f64);
        histogram!("rql.db.get_neighbours_ms").record(_ms);
        tracing::info!(_ms, entity_count, fact_count, "kremory.db.get_neighbours");
        Ok(SubGraph {
            entities,
            facts: collected_facts,
        })
    }

    // === Embedding ===

    /// Set or update the embedding vector for an entity.
    /// `embedding` is a 384-dimensional f32 vector.
    pub async fn set_entity_embedding(&self, id: &str, embedding: &[f32]) -> Result<()> {
        let _db_start = Instant::now();
        // Convert f32 slice to JSON array string for the vector() SQL function
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
                "UPDATE rql_entities SET embedding = vector(?1) WHERE id = ?2",
                libsql::params![vec_str, id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.set_entity_embedding_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.set_entity_embedding");
        Ok(())
    }

    /// Set or update the embedding vector for a fact.
    /// `embedding` is a 384-dimensional f32 vector.
    pub async fn set_fact_embedding(&self, fact_id: i64, embedding: &[f32]) -> Result<()> {
        let _db_start = Instant::now();
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
                "UPDATE facts SET embedding = vector(?1) WHERE id = ?2",
                libsql::params![vec_str, fact_id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.set_fact_embedding_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.set_fact_embedding");
        Ok(())
    }

    // === Episode CRUD ===

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

    /// Insert an entity with an optional group_id.
    /// Existing tests use `insert_entity`; this variant is for new code that needs group scoping.
    pub async fn insert_entity_with_group(
        &self,
        id: &str,
        label: &str,
        properties: serde_json::Value,
        group_id: Option<&str>,
    ) -> Result<()> {
        let _db_start = Instant::now();
        let props_str = serde_json::to_string(&properties)?;
        let now = Utc::now().to_rfc3339();
        // Transaction guards the invariant: row in `entities` ⟹ row in `entities_fts`.
        // Without a transaction, a crash or FTS error between the two INSERTs leaves an
        // entity permanently invisible to FTS search (HIGH-2 remediation, 2026-05-15).
        self.conn.execute("BEGIN", ()).await?;
        let inner: Result<()> = async {
            self.conn
                .execute(
                    "INSERT INTO rql_entities (id, label, properties, recorded_at, group_id) VALUES (?1, ?2, ?3, ?4, ?5)",
                    libsql::params![id, label, props_str.clone(), now, group_id],
                )
                .await?;
            self.conn
                .execute(
                    "INSERT INTO rql_entities_fts(entity_id, label, properties) VALUES (?1, ?2, ?3)",
                    libsql::params![id, label, props_str],
                )
                .await?;
            Ok(())
        }
        .await;
        match inner {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                // Best-effort rollback; propagate the original error regardless.
                let _ = self.conn.execute("ROLLBACK", ()).await;
                return Err(e);
            }
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_entity_with_group_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.insert_entity_with_group");
        Ok(())
    }

    /// Insert a fact with an optional group_id.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_fact_with_group(
        &self,
        subject_id: &str,
        predicate: &str,
        object_id: Option<&str>,
        object_value: Option<&str>,
        valid_from: DateTime<Utc>,
        confidence: f64,
        source_episode_id: Option<i64>,
        group_id: Option<&str>,
        embedding: Option<&[f32]>,
    ) -> Result<i64> {
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        let valid_from_str = valid_from.to_rfc3339();
        // Story #209: compute SHA-256 content hash for dedup
        let hash = fact_content_hash(subject_id, predicate, object_id, object_value);
        let mut dup_check = self
            .conn
            .query(
                "SELECT id FROM facts WHERE content_hash = ?1 AND expired_at IS NULL LIMIT 1",
                libsql::params![hash.clone()],
            )
            .await?;
        if dup_check.next().await?.is_some() {
            return Err(crate::core::error::Error::Duplicate { content_hash: hash });
        }
        let vec_str = embedding.map(|e| {
            format!(
                "[{}]",
                e.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        });
        self.conn
            .execute(
                "INSERT INTO facts (subject_id, predicate, object_id, object_value, embedding, valid_from, recorded_at, confidence, source_episode_id, group_id, content_hash)
                 VALUES (?1, ?2, ?3, ?4, CASE WHEN ?5 IS NULL THEN NULL ELSE vector(?5) END, ?6, ?7, ?8, ?9, ?10, ?11)",
                libsql::params![
                    subject_id,
                    predicate,
                    object_id,
                    object_value,
                    vec_str,
                    valid_from_str,
                    now,
                    confidence,
                    source_episode_id,
                    group_id,
                    hash,
                ],
            )
            .await?;
        let mut rows = self.conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows
            .next()
            .await?
            .ok_or(crate::core::error::Error::InsertReturnedNoRowId {
                operation: "insert_fact_with_group",
            })?;
        let fact_id = row.get::<i64>(0)?;
        if let Some(ov) = object_value {
            self.conn
                .execute(
                    "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
                    libsql::params![fact_id, predicate, ov],
                )
                .await?;
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_fact_with_group_ms").record(_ms);
        tracing::info!(_ms, fact_id, "kremory.db.insert_fact_with_group");
        Ok(fact_id)
    }

    /// Insert an episode with optional group_id, saga_id, and sequence_number.
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
    ) -> Result<i64> {
        let _db_start = Instant::now();
        let ts_str = timestamp.to_rfc3339();
        let meta_str = metadata.as_ref().map(serde_json::to_string).transpose()?;
        self.conn
            .execute(
                "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, saga_id, sequence_number)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                libsql::params![content, ts_str, source_type, meta_str, group_id, saga_id, sequence_number],
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
    pub async fn insert_episodic_edge(
        &self,
        episode_id: i64,
        entity_id: &str,
        role: &str,
    ) -> Result<i64> {
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO episodic_edges (episode_id, entity_id, role, recorded_at) VALUES (?1, ?2, ?3, ?4)",
            libsql::params![episode_id, entity_id, role, now],
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
            });
        }
        Ok(edges)
    }

    // === Extended Fact Methods ===

    /// Invalidate a fact with both system-level (expired_at) and domain-level (invalid_at) timestamps.
    pub async fn invalidate_fact_with_reason(
        &self,
        fact_id: i64,
        expired_at: DateTime<Utc>,
        invalid_at: DateTime<Utc>,
    ) -> Result<()> {
        let _db_start = Instant::now();
        self.conn
            .execute(
                "UPDATE facts SET expired_at = ?1, invalid_at = ?2 WHERE id = ?3",
                libsql::params![expired_at.to_rfc3339(), invalid_at.to_rfc3339(), fact_id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.invalidate_fact_with_reason_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.invalidate_fact_with_reason");
        Ok(())
    }

    /// Get active (non-expired) facts for a subject + predicate combination.
    pub async fn get_facts_by_subject_predicate(
        &self,
        subject_id: &str,
        predicate: &str,
    ) -> Result<Vec<Fact>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE subject_id = ?1 AND predicate = ?2 AND expired_at IS NULL",
                libsql::params![subject_id, predicate],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        Ok(facts)
    }

    // === petgraph export ===

    pub async fn to_petgraph(&self) -> Result<petgraph::graph::DiGraph<String, (String, i64)>> {
        use petgraph::graph::DiGraph;
        use std::collections::HashMap;

        let entities = self.list_entities().await?;
        let mut graph: DiGraph<String, (String, i64)> = DiGraph::new();
        let mut node_index: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

        for entity in &entities {
            let idx = graph.add_node(entity.id.clone());
            node_index.insert(entity.id.clone(), idx);
        }

        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE object_id IS NOT NULL
                   AND expired_at IS NULL",
                (),
            )
            .await?;

        while let Some(row) = rows.next().await? {
            let fact = row_to_fact(&row)?;
            if let Some(ref oid) = fact.object_id {
                let src = node_index.get(&fact.subject_id);
                let dst = node_index.get(oid);
                if let (Some(&s), Some(&d)) = (src, dst) {
                    graph.add_edge(s, d, (fact.predicate.clone(), fact.id));
                }
            }
        }

        Ok(graph)
    }

    /// Find all facts that have no embedding and return their IDs + object_value.
    ///
    /// # SQLite-first ordering invariant (Story #214)
    ///
    /// SQLite commit MUST precede vector write (Story #214). If a crash occurs
    /// between SQLite commit and vector write, all facts with NULL embedding can
    /// be identified via this function and re-embedded by the caller. The reverse
    /// ordering (vector-first) has NO recovery path — this is the entire
    /// justification for the SQLite-first contract.
    pub async fn facts_missing_embeddings(&self) -> Result<Vec<(i64, String, String, Option<String>)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_value FROM facts WHERE embedding IS NULL AND expired_at IS NULL",
                (),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let subject_id: String = row.get(1)?;
            let predicate: String = row.get(2)?;
            let object_value: Option<String> = row.get(3)?;
            out.push((id, subject_id, predicate, object_value));
        }
        Ok(out)
    }

    /// Backfill a vector embedding for a specific fact by ID.
    ///
    /// Called after crash-recovery when SQLite has the row but the vector index
    /// did not receive the write. Story #214.
    pub async fn backfill_fact_embedding(&self, fact_id: i64, embedding: &[f32]) -> Result<()> {
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
                "UPDATE facts SET embedding = CASE WHEN ?1 IS NULL THEN NULL ELSE vector(?1) END WHERE id = ?2",
                libsql::params![vec_str, fact_id],
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::TemporalGraph;
    use chrono::Duration;

    // === Entity CRUD ===

    #[tokio::test]
    async fn test_insert_and_get_entity() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        let entity = g.get_entity("alice").await.unwrap().unwrap();
        assert_eq!(entity.id, "alice");
        assert_eq!(entity.label, "Person");
        assert_eq!(entity.properties["role"], "engineer");
        assert!(entity.updated_at.is_none());
    }

    #[tokio::test]
    async fn test_get_nonexistent_entity_returns_none() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let result = g.get_entity("nobody").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_update_entity_properties() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.update_entity("alice", serde_json::json!({"role": "manager"}))
            .await
            .unwrap();
        let entity = g.get_entity("alice").await.unwrap().unwrap();
        assert_eq!(entity.properties["role"], "manager");
        assert!(entity.updated_at.is_some());
    }

    #[tokio::test]
    async fn test_list_entities() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let entities = g.list_entities().await.unwrap();
        assert_eq!(entities.len(), 3);
        let mut ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec!["acme", "alice", "bob"]);
    }

    // === Fact CRUD ===

    #[tokio::test]
    async fn test_insert_fact_returns_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let id = g
            .insert_fact("alice", "exists", None, None, Utc::now(), 1.0, None, None)
            .await
            .unwrap();
        assert!(id > 0);
    }

    #[tokio::test]
    async fn test_insert_fact_with_object_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let id = g
            .insert_fact(
                "alice",
                "works_at",
                Some("acme"),
                None,
                Utc::now(),
                1.0,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(id > 0);
        let facts = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "works_at");
        assert_eq!(facts[0].subject_id, "alice");
        assert_eq!(facts[0].object_id.as_deref(), Some("acme"));
    }

    #[tokio::test]
    async fn test_insert_fact_with_object_value() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let id = g
            .insert_fact(
                "alice",
                "has_title",
                None,
                Some("PM"),
                Utc::now(),
                1.0,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(id > 0);
        let facts = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "has_title");
        assert_eq!(facts[0].object_value.as_deref(), Some("PM"));
        assert!(facts[0].object_id.is_none());
    }

    #[tokio::test]
    async fn test_invalidate_fact() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::days(1);
        let id = g
            .insert_fact("alice", "has_title", None, Some("PM"), t0, 1.0, None, None)
            .await
            .unwrap();
        let facts_before = g.facts_at(Utc::now()).await.unwrap();
        assert_eq!(facts_before.len(), 1);
        g.invalidate_fact(id, Utc::now()).await.unwrap();
        let facts_after = g.facts_at(Utc::now() + Duration::seconds(1)).await.unwrap();
        assert_eq!(facts_after.len(), 0);
    }

    // === Temporal Queries ===

    #[tokio::test]
    async fn test_facts_at_filters_by_valid_from() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now();
        g.insert_fact("alice", "has_title", None, Some("PM"), t0, 1.0, None, None)
            .await
            .unwrap();
        // query just before valid_from: should return nothing
        let facts_before = g.facts_at(t0 - Duration::seconds(1)).await.unwrap();
        assert_eq!(facts_before.len(), 0);
        // query after valid_from: should return the fact
        let facts_after = g.facts_at(t0 + Duration::seconds(1)).await.unwrap();
        assert_eq!(facts_after.len(), 1);
        assert_eq!(facts_after[0].predicate, "has_title");
    }

    #[tokio::test]
    async fn test_facts_at_excludes_expired() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::days(2);
        let id = g
            .insert_fact("alice", "has_title", None, Some("PM"), t0, 1.0, None, None)
            .await
            .unwrap();
        g.invalidate_fact(id, Utc::now() - Duration::days(1))
            .await
            .unwrap();
        let facts = g.facts_at(Utc::now()).await.unwrap();
        assert_eq!(facts.len(), 0);
    }

    #[tokio::test]
    async fn test_entity_facts_at_filters_by_entity() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::hours(1);
        g.insert_fact("alice", "has_title", None, Some("PM"), t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact("bob", "has_title", None, Some("Eng"), t0, 1.0, None, None)
            .await
            .unwrap();
        let alice_facts = g.entity_facts_at("alice", Utc::now()).await.unwrap();
        assert_eq!(alice_facts.len(), 1);
        assert_eq!(alice_facts[0].subject_id, "alice");
        assert_eq!(alice_facts[0].object_value.as_deref(), Some("PM"));
    }

    #[tokio::test]
    async fn test_entity_history_includes_expired() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::days(2);
        let id = g
            .insert_fact("alice", "has_title", None, Some("PM"), t0, 1.0, None, None)
            .await
            .unwrap();
        g.invalidate_fact(id, Utc::now() - Duration::days(1))
            .await
            .unwrap();
        // expired fact should NOT appear in facts_at
        let live = g.facts_at(Utc::now()).await.unwrap();
        assert_eq!(live.len(), 0);
        // but it SHOULD appear in entity_history
        let history = g.entity_history("alice").await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].predicate, "has_title");
        assert!(history[0].expired_at.is_some());
    }

    #[tokio::test]
    async fn test_point_in_time_temporal_evolution() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("newco", "Company", serde_json::json!({}))
            .await
            .unwrap();

        let t0 = Utc::now() - Duration::days(10);
        let t1 = Utc::now() - Duration::days(5);

        // alice works_at acme from t0
        let fact1_id = g
            .insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();

        // at t0+1d: acme fact is visible
        let facts_t0 = g.facts_at(t0 + Duration::days(1)).await.unwrap();
        assert_eq!(facts_t0.len(), 1);
        assert_eq!(facts_t0[0].object_id.as_deref(), Some("acme"));

        // administratively retract acme fact at t1, add newco fact from t1
        g.invalidate_fact(fact1_id, t1).await.unwrap();
        g.insert_fact(
            "alice",
            "works_at",
            Some("newco"),
            None,
            t1,
            1.0,
            None,
            None,
        )
        .await
        .unwrap();

        // at t1+1d: only newco fact is visible
        let facts_t1 = g.facts_at(t1 + Duration::days(1)).await.unwrap();
        assert_eq!(facts_t1.len(), 1);
        assert_eq!(facts_t1[0].object_id.as_deref(), Some("newco"));

        // history should show both facts for alice
        let history = g.entity_history("alice").await.unwrap();
        assert_eq!(history.len(), 2);
        let predicates: Vec<&str> = history.iter().map(|f| f.predicate.as_str()).collect();
        assert!(predicates.iter().all(|&p| p == "works_at"));
        let object_ids: Vec<Option<&str>> =
            history.iter().map(|f| f.object_id.as_deref()).collect();
        assert!(object_ids.contains(&Some("acme")));
        assert!(object_ids.contains(&Some("newco")));
    }

    // === Graph Traversal ===

    #[tokio::test]
    async fn test_get_neighbours_one_hop() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::hours(1);
        g.insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact("alice", "manages", Some("bob"), None, t0, 1.0, None, None)
            .await
            .unwrap();

        let subgraph = g.get_neighbours("alice", 1).await.unwrap();
        let mut entity_ids: Vec<&str> = subgraph.entities.iter().map(|e| e.id.as_str()).collect();
        entity_ids.sort();
        assert!(entity_ids.contains(&"alice"));
        assert!(entity_ids.contains(&"acme"));
        assert!(entity_ids.contains(&"bob"));
        assert_eq!(subgraph.facts.len(), 2);
    }

    #[tokio::test]
    async fn test_get_neighbours_two_hops() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::hours(1);
        g.insert_fact("alice", "manages", Some("bob"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact("bob", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();

        let subgraph = g.get_neighbours("alice", 2).await.unwrap();
        let entity_ids: Vec<&str> = subgraph.entities.iter().map(|e| e.id.as_str()).collect();
        assert!(entity_ids.contains(&"alice"));
        assert!(entity_ids.contains(&"bob"));
        assert!(entity_ids.contains(&"acme"));
        assert_eq!(subgraph.facts.len(), 2);
    }

    #[tokio::test]
    async fn test_get_neighbours_excludes_expired_facts() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::days(2);
        let id = g
            .insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.invalidate_fact(id, Utc::now() - Duration::days(1))
            .await
            .unwrap();

        let subgraph = g.get_neighbours("alice", 1).await.unwrap();
        // Only alice; acme should not be reached via the expired fact
        assert_eq!(subgraph.facts.len(), 0);
        assert_eq!(subgraph.entities.len(), 1);
        assert_eq!(subgraph.entities[0].id, "alice");
    }

    #[tokio::test]
    async fn test_get_neighbours_zero_hops() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::hours(1);
        g.insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();

        let subgraph = g.get_neighbours("alice", 0).await.unwrap();
        assert_eq!(subgraph.facts.len(), 0);
        assert_eq!(subgraph.entities.len(), 1);
        assert_eq!(subgraph.entities[0].id, "alice");
    }

    // === petgraph export ===

    #[tokio::test]
    async fn test_to_petgraph_nodes_and_edges() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::hours(1);
        g.insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact("bob", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();

        let pg = g.to_petgraph().await.unwrap();
        assert_eq!(pg.node_count(), 3);
        assert_eq!(pg.edge_count(), 2);
    }

    #[tokio::test]
    async fn test_to_petgraph_excludes_expired() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::days(2);
        let id = g
            .insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.invalidate_fact(id, Utc::now() - Duration::days(1))
            .await
            .unwrap();

        let pg = g.to_petgraph().await.unwrap();
        assert_eq!(pg.node_count(), 2);
        assert_eq!(pg.edge_count(), 0);
    }

    // === Episode CRUD ===

    #[tokio::test]
    async fn test_insert_episode() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let id = g
            .insert_episode(
                "Alice joined the meeting.",
                Utc::now(),
                Some("transcript"),
                Some(serde_json::json!({"speaker": "alice"})),
            )
            .await
            .unwrap();
        assert!(id > 0);
    }

    // === New E1.S3 Tests ===

    #[tokio::test]
    async fn test_insert_episodic_edge() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let ep_id = g
            .insert_episode("Alice joined.", Utc::now(), Some("transcript"), None)
            .await
            .unwrap();
        let edge_id = g
            .insert_episodic_edge(ep_id, "alice", "mentioned")
            .await
            .unwrap();
        assert!(edge_id > 0);

        let edges = g.episodic_edges_for_entity("alice").await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].entity_id, "alice");
        assert_eq!(edges[0].role, "mentioned");
        assert_eq!(edges[0].episode_id, ep_id);
    }

    #[tokio::test]
    async fn test_invalidate_fact_with_reason() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::days(2);
        let id = g
            .insert_fact("alice", "has_title", None, Some("PM"), t0, 1.0, None, None)
            .await
            .unwrap();
        let expired = Utc::now() - Duration::days(1);
        let invalid = Utc::now() - Duration::hours(12);
        g.invalidate_fact_with_reason(id, expired, invalid)
            .await
            .unwrap();

        // Fact should now be absent from active facts
        let facts = g.facts_at(Utc::now()).await.unwrap();
        assert_eq!(facts.len(), 0);

        // Full history should still show the fact with both timestamps set
        let history = g.entity_history("alice").await.unwrap();
        assert_eq!(history.len(), 1);
        assert!(history[0].expired_at.is_some());
        assert!(history[0].invalid_at.is_some());
    }

    #[tokio::test]
    async fn test_get_facts_by_subject_predicate() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now() - Duration::hours(1);
        g.insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact(
            "alice",
            "has_title",
            None,
            Some("Engineer"),
            t0,
            1.0,
            None,
            None,
        )
        .await
        .unwrap();

        let works_at_facts = g
            .get_facts_by_subject_predicate("alice", "works_at")
            .await
            .unwrap();
        assert_eq!(works_at_facts.len(), 1);
        assert_eq!(works_at_facts[0].object_id.as_deref(), Some("acme"));

        let title_facts = g
            .get_facts_by_subject_predicate("alice", "has_title")
            .await
            .unwrap();
        assert_eq!(title_facts.len(), 1);
        assert_eq!(title_facts[0].object_value.as_deref(), Some("Engineer"));
    }

    #[tokio::test]
    async fn test_insert_entity_with_group() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity_with_group(
            "alice",
            "Person",
            serde_json::json!({"name": "Alice"}),
            Some("group-abc"),
        )
        .await
        .unwrap();

        let entity = g.get_entity("alice").await.unwrap().unwrap();
        assert_eq!(entity.id, "alice");
        assert_eq!(entity.group_id.as_deref(), Some("group-abc"));
    }

    /// Stream 3 A.4.5 contract test: Lane A indexer call sites pass
    /// `group_id=None` (folder_id is always NULL pre-Lane-B). Resulting RQL
    /// entity must surface NULL group_id — not an empty string, not a
    /// "default" sentinel — so retrieval skips the group scope filter
    /// entirely. The follow-up Lane B work re-introduces non-null group_id
    /// when the create-folder UX populates entity.folder_id.
    #[tokio::test]
    async fn lane_a_indexer_writes_null_group_id_when_folder_id_absent() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        let group_id: Option<&str> = None;
        g.insert_entity_with_group(
            "doc:welcome:chunk_0",
            "Welcome to the host application (part 1)",
            serde_json::json!({
                "text": "the host application captures meetings and surfaces insights.",
                "source": "doc:welcome",
                "source_type": "document",
            }),
            group_id,
        )
        .await
        .unwrap();

        let entity = g.get_entity("doc:welcome:chunk_0").await.unwrap().unwrap();
        assert!(
            entity.group_id.is_none(),
            "Lane A indexer must persist NULL group_id (got {:?})",
            entity.group_id,
        );
    }

    #[tokio::test]
    async fn test_update_entity_group_changes_scope() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entity in group-a
        g.insert_entity_with_group(
            "doc_chunk_0",
            "Document (part 1)",
            serde_json::json!({"text": "original content"}),
            Some("group-a"),
        )
        .await
        .unwrap();

        let entity = g.get_entity("doc_chunk_0").await.unwrap().unwrap();
        assert_eq!(entity.group_id.as_deref(), Some("group-a"));

        // Re-scope to group-b with updated properties
        g.update_entity_group(
            "doc_chunk_0",
            Some("group-b"),
            serde_json::json!({"text": "updated content"}),
        )
        .await
        .unwrap();

        let updated = g.get_entity("doc_chunk_0").await.unwrap().unwrap();
        assert_eq!(updated.group_id.as_deref(), Some("group-b"));
        let props: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&updated.properties).unwrap()).unwrap();
        assert_eq!(props["text"], "updated content");
    }

    #[tokio::test]
    async fn test_update_entity_group_to_none() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group("e1", "Test", serde_json::json!({}), Some("scoped"))
            .await
            .unwrap();

        g.update_entity_group("e1", None, serde_json::json!({}))
            .await
            .unwrap();

        let entity = g.get_entity("e1").await.unwrap().unwrap();
        assert!(entity.group_id.is_none());
    }

    // === HIGH-2 RED→GREEN: insert_entity atomicity ===
    //
    // Verifies that if the FTS insert fails (simulated by dropping entities_fts
    // before the call), the entities row is NOT committed — i.e., the transaction
    // rolls back cleanly and does not leave a partially-indexed entity.

    #[tokio::test]
    async fn insert_entity_rolls_back_entities_row_when_fts_fails() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Drop entities_fts to force the second INSERT inside insert_entity to fail.
        g.conn
            .execute("DROP TABLE rql_entities_fts", ())
            .await
            .expect("DROP TABLE rql_entities_fts must succeed on fresh in-memory DB");

        let result = g
            .insert_entity(
                "test-atomic-1",
                "Test",
                serde_json::json!({"text": "hello"}),
            )
            .await;
        assert!(
            result.is_err(),
            "insert_entity must return Err when FTS table is missing (no partial commit)"
        );

        // The entities row must have been rolled back — zero rows for the attempted ID.
        let mut rows = g
            .conn
            .query(
                "SELECT COUNT(*) FROM rql_entities WHERE id = 'test-atomic-1'",
                (),
            )
            .await
            .expect("SELECT on rql_entities must succeed even after FTS drop");
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        assert_eq!(
            count, 0,
            "entities row must be rolled back when FTS insert fails (HIGH-2)"
        );
    }

    #[tokio::test]
    async fn insert_entity_with_group_rolls_back_entities_row_when_fts_fails() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Drop entities_fts to force the second INSERT to fail.
        g.conn
            .execute("DROP TABLE rql_entities_fts", ())
            .await
            .expect("DROP TABLE rql_entities_fts must succeed on fresh in-memory DB");

        let result = g
            .insert_entity_with_group(
                "test-atomic-grp-1",
                "Test",
                serde_json::json!({"text": "hello"}),
                Some("grp-a"),
            )
            .await;
        assert!(
            result.is_err(),
            "insert_entity_with_group must return Err when FTS table is missing"
        );

        let mut rows = g
            .conn
            .query(
                "SELECT COUNT(*) FROM rql_entities WHERE id = 'test-atomic-grp-1'",
                (),
            )
            .await
            .expect("SELECT on rql_entities must succeed even after FTS drop");
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        assert_eq!(
            count, 0,
            "entities row must be rolled back when FTS insert fails (HIGH-2, insert_entity_with_group)"
        );
    }

    // === SHA-256 dedup (Story #209) ===

    #[tokio::test]
    async fn insert_fact_duplicate_returns_error() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        // First insert succeeds
        g.insert_fact("alice", "likes", None, Some("coffee"), t, 1.0, None, None)
            .await
            .unwrap();
        // Second insert with same triple must return Duplicate error
        let err = g
            .insert_fact("alice", "likes", None, Some("coffee"), t, 0.9, None, None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::core::error::Error::Duplicate { .. }),
            "expected Duplicate error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn insert_fact_different_triples_both_succeed() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        g.insert_fact("alice", "likes", None, Some("coffee"), t, 1.0, None, None)
            .await
            .unwrap();
        // Different object_value → different hash → succeeds
        g.insert_fact("alice", "likes", None, Some("tea"), t, 1.0, None, None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn fact_content_hash_deterministic() {
        let h1 = fact_content_hash("alice", "likes", None, Some("coffee"));
        let h2 = fact_content_hash("alice", "likes", None, Some("coffee"));
        assert_eq!(h1, h2, "hash must be deterministic");
        let h3 = fact_content_hash("alice", "likes", None, Some("tea"));
        assert_ne!(h1, h3, "different facts must have different hashes");
        // SHA-256 produces 64 hex chars
        assert_eq!(h1.len(), 64);
    }

    // === SQLite-first vector ordering backfill (Story #214) ===

    #[tokio::test]
    async fn facts_missing_embeddings_returns_all_null_embedding_facts() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        // Insert a fact without embedding (None) — simulates SQLite-committed, vector-not-written
        let fact_id = g
            .insert_fact("alice", "works_at", None, Some("ACME"), t, 1.0, None, None)
            .await
            .unwrap();

        let missing = g.facts_missing_embeddings().await.unwrap();
        assert_eq!(missing.len(), 1, "one fact has no embedding");
        assert_eq!(missing[0].0, fact_id);

        // Backfill with a stub embedding
        let embedding: Vec<f32> = vec![0.1_f32; 384];
        g.backfill_fact_embedding(fact_id, &embedding).await.unwrap();

        // After backfill, no facts should be missing
        let still_missing = g.facts_missing_embeddings().await.unwrap();
        assert!(
            still_missing.is_empty(),
            "backfill must clear the missing-embedding list"
        );
    }

    #[tokio::test]
    async fn backfill_fact_embedding_is_idempotent() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        let fact_id = g
            .insert_fact("bob", "knows", None, Some("alice"), t, 1.0, None, None)
            .await
            .unwrap();
        let embedding: Vec<f32> = vec![0.2_f32; 384];
        // First backfill
        g.backfill_fact_embedding(fact_id, &embedding).await.unwrap();
        // Second backfill on same fact must not error
        g.backfill_fact_embedding(fact_id, &embedding).await.unwrap();
    }
}
