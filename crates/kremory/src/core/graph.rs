use chrono::{DateTime, Utc};
use metrics::histogram;
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use tracing;

use crate::core::error::Result;
use crate::core::schema::{Entity, Episode, EpisodicEdge, Fact, TemporalGraph};

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

/// Expected columns from the canonical entity SELECT (LEFT JOIN entity_types):
/// 0: e.id, 1: COALESCE(et.name,'Entity') as label, 2: e.properties,
/// 3: e.recorded_at, 4: e.updated_at, 5: e.group_id, 6: e.access_count,
/// 7: e.entity_type_id
fn row_to_entity(row: &libsql::Row) -> anyhow::Result<Entity> {
    let id: String = row.get::<String>(0)?;
    let label: String = row.get::<String>(1)?;
    let props_str: Option<String> = row.get::<Option<String>>(2)?;
    let created_str: String = row.get::<String>(3)?;
    let updated_str: Option<String> = row.get::<Option<String>>(4)?;
    let group_id: Option<String> = row.get::<Option<String>>(5)?;
    let access_count: i64 = row.get::<i64>(6)?;
    let entity_type_id_raw: i64 = row.get::<i64>(7)?;
    let entity_type_id: u32 = entity_type_id_raw.max(0) as u32;

    let properties: serde_json::Value = props_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);
    let recorded_at = parse_dt(&created_str)?;
    let updated_at = updated_str.as_deref().map(parse_dt).transpose()?;

    Ok(Entity {
        id,
        label,
        entity_type_id,
        properties,
        recorded_at,
        updated_at,
        group_id,
        access_count,
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
        // ADR-029b: composite FK fields — absent on pre-migration-004 rows;
        // populated by the migration 004 backfill. None on fresh rows until
        // the caller explicitly sets subject_group_id / object_group_id.
        subject_group_id: None,
        object_group_id: None,
    })
}

impl TemporalGraph {
    // === Entity CRUD ===

    pub async fn insert_entity(
        &self,
        id: &str,
        entity_type_id: u32,
        properties: serde_json::Value,
    ) -> Result<()> {
        let _db_start = Instant::now();
        let props_str = serde_json::to_string(&properties)?;
        let now = Utc::now().to_rfc3339();
        // Transaction guards the invariant: row in `entities` ⟹ row in `entities_fts`.
        // Without a transaction, a crash or FTS error between the two INSERTs leaves an
        // entity permanently invisible to FTS search (HIGH-2 remediation, 2026-05-15).
        // Uses BeginGuard so nested calls (caller already holds an outer txn) are a no-op
        // on BEGIN/COMMIT — required to avoid "transaction within a transaction" errors
        // when this method is called from inside the ingest pipeline's outer txn.
        let guard = self.begin_immediate_if_needed().await?;
        let inner: Result<()> = async {
            // ADR-045 §2 + migration-010-detail-spec §1.2: entity_type_source stamped
            // 'Phase1Ner' at INSERT time — all entity inserts via this method are Phase 1
            // (with_facts stub creation path). entity_type_assigned_at = now().
            self.conn
                .execute(
                    "INSERT INTO entities (id, entity_type_id, properties, recorded_at, \
                     entity_type_source, entity_type_assigned_at) \
                     VALUES (?1, ?2, ?3, ?4, 'Phase1Ner', ?4)",
                    libsql::params![id, entity_type_id as i64, props_str.clone(), now],
                )
                .await?;
            // FTS: label column in entities_fts is no longer populated (entities.label
            // was dropped in Migration 009). Insert empty string to satisfy the FTS schema.
            self.conn
                .execute(
                    "INSERT INTO entities_fts(entity_id, label, properties) VALUES (?1, '', ?2)",
                    libsql::params![id, props_str],
                )
                .await?;
            Ok(())
        }
        .await;
        match inner {
            Ok(()) => {
                guard.commit().await?;
            }
            Err(e) => {
                let _ = guard.rollback().await;
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
    /// indexed by the `entities_vec_idx` for cosine-similarity search.
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
                &format!("UPDATE entities SET embedding = {} WHERE id = ?1", vec_str),
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
                "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.properties, \
                        e.recorded_at, e.updated_at, e.group_id, e.access_count, e.entity_type_id \
                 FROM entities e \
                 LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id \
                 WHERE e.id = ?1",
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
                "UPDATE entities SET properties = ?1, updated_at = ?2 WHERE id = ?3",
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
        // ADR-029b: entities.group_id is NOT NULL post-migration-004. COALESCE maps
        // None → 'default' so callers using None-as-unscoped retain their semantics
        // while the storage constraint is satisfied.
        let effective_group_id = group_id.unwrap_or("default");
        self.conn
            .execute(
                "UPDATE entities SET group_id = ?1, properties = ?2, updated_at = ?3 WHERE id = ?4",
                libsql::params![effective_group_id, props_str, now, id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.update_entity_group_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.update_entity_group");
        Ok(())
    }

    /// Reassign an entity's `group_id` WITHOUT touching `properties`.
    ///
    /// **DANGEROUS** — callers MUST set `bypass_policy = false` in production.
    /// Only migration tooling (`kremory-admin`, Decision 7) may pass `true`.
    ///
    /// Composite-PK semantics (ADR-029b Decision 1): the entity is identified
    /// by `(id, old_group_id)`. The update is a single `UPDATE … WHERE id = ?
    /// AND group_id = ?` — if the row is absent (wrong `old_group_id` or
    /// entity not found), returns `Ok(0)`.
    ///
    /// Policy checks (ADR-029b Decision 5):
    /// - Source namespace (`old_group_id`) must not be `AppendOnly`.
    /// - Destination namespace (`new_group_id`) must not be `AppendOnly`.
    /// - Both checks are skipped when `bypass_policy = true`.
    ///
    /// Returns the number of rows updated.
    pub async fn reassign_entity_group_dangerous(
        &self,
        id: &str,
        old_group_id: &str,
        new_group_id: Option<&str>,
        bypass_policy: bool,
    ) -> Result<u64> {
        if !bypass_policy {
            // Check source policy.
            let source_policy = self
                .get_namespace_policy(old_group_id)
                .await?
                .unwrap_or_default();
            if source_policy.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                return Err(crate::core::error::Error::NamespacePolicyViolation {
                    namespace: old_group_id.to_string(),
                    operation: "entity_move_source".to_string(),
                    policy: source_policy,
                });
            }
            // Check destination policy.
            if let Some(new_gid) = new_group_id {
                let dest_policy = self
                    .get_namespace_policy(new_gid)
                    .await?
                    .unwrap_or_default();
                if dest_policy.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                    return Err(crate::core::error::Error::NamespacePolicyViolation {
                        namespace: new_gid.to_string(),
                        operation: "entity_move_dest".to_string(),
                        policy: dest_policy,
                    });
                }
            }
        }
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        let n = self
            .conn
            .execute(
                "UPDATE entities SET group_id = ?1, updated_at = ?2 WHERE id = ?3 AND group_id = ?4",
                libsql::params![new_group_id, now, id, old_group_id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.reassign_entity_group_dangerous_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.reassign_entity_group_dangerous");
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
                    &format!("DELETE FROM entities_fts WHERE entity_id IN ({placeholders})"),
                    params.clone(),
                )
                .await?;
            let n = self
                .conn
                .execute(
                    &format!("DELETE FROM entities WHERE id IN ({placeholders})"),
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
                "DELETE FROM entities_fts WHERE entity_id LIKE ?1",
                libsql::params![pattern.clone()],
            )
            .await?;

        let deleted = self
            .conn
            .execute(
                "DELETE FROM entities WHERE id LIKE ?1",
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
                "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.properties, \
                        e.recorded_at, e.updated_at, e.group_id, e.access_count, e.entity_type_id \
                 FROM entities e \
                 LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id",
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

    /// List all entities belonging to a specific namespace group.
    ///
    /// Used by `Engine::ingest_with` at the dedup sites (`core/ingest.rs`)
    /// to restrict entity matching to the caller's namespace, preventing
    /// cross-namespace entity collisions.
    ///
    /// The `group_id` parameter maps directly to the storage column; callers
    /// derive it from `Namespace` via `namespace_to_group_id(ns)`.
    pub async fn list_entities_in_group(&self, group_id: &str) -> Result<Vec<Entity>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.properties, \
                        e.recorded_at, e.updated_at, e.group_id, e.access_count, e.entity_type_id \
                 FROM entities e \
                 LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id \
                 WHERE e.group_id = ?1",
                libsql::params![group_id],
            )
            .await?;
        let mut entities = Vec::new();
        while let Some(row) = rows.next().await? {
            entities.push(row_to_entity(&row)?);
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let count = entities.len();
        histogram!("rql.db.list_entities_in_group_count").record(count as f64);
        histogram!("rql.db.list_entities_in_group_ms").record(_ms);
        tracing::info!(_ms, count, group_id, "kremory.db.list_entities_in_group");
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
        // FU.1: acquire BEGIN IMMEDIATE before the SELECT-check to serialise concurrent
        // writers and close the TOCTTOU window between the dup-check SELECT and the INSERT.
        let guard = self.begin_immediate_if_needed().await?;
        // Inner ops wrapped so we can explicitly rollback on Err — without this,
        // a raw `?` would drop the guard, leaking the BEGIN IMMEDIATE on the
        // libsql connection and causing "transaction within a transaction" on
        // the next call.
        let result: Result<i64> = async {
            let mut dup_check = self
                .conn
                .query(
                    "SELECT id FROM facts WHERE content_hash = ?1 AND expired_at IS NULL LIMIT 1",
                    libsql::params![hash.clone()],
                )
                .await?;
            if dup_check.next().await?.is_some() {
                return Err(crate::core::error::Error::Duplicate {
                    content_hash: hash.clone(),
                });
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
                        hash.clone(),
                    ],
                )
                .await?;
            let mut rows = self.conn.query("SELECT last_insert_rowid()", ()).await?;
            let row = rows.next().await?.ok_or(
                crate::core::error::Error::InsertReturnedNoRowId {
                    operation: "insert_fact",
                },
            )?;
            let fact_id = row.get::<i64>(0)?;
            if let Some(ov) = object_value {
                self.conn
                    .execute(
                        "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
                        libsql::params![fact_id, predicate, ov],
                    )
                    .await?;
            }
            Ok(fact_id)
        }
        .await;
        match result {
            Ok(fact_id) => {
                guard.commit().await?;
                let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
                histogram!("rql.db.insert_fact_ms").record(_ms);
                tracing::info!(_ms, fact_id, "kremory.db.insert_fact");
                Ok(fact_id)
            }
            Err(e) => {
                let _ = guard.rollback().await;
                Err(e)
            }
        }
    }

    /// `insert_fact` with `Err(Error::Duplicate)` swallowed to `Ok(None)`.
    ///
    /// Returns `Ok(Some(fact_id))` for a fresh insert, `Ok(None)` when a
    /// matching `content_hash` already exists, `Err` for any other failure.
    ///
    /// Used by the LLM Phase 2 extraction pipeline (`engine/ingest.rs`) and
    /// the `with_facts` caller-pin path (`memory/engine_handle.rs`) so that
    /// pre-pinned caller facts dedup cleanly against LLM-extracted ones
    /// without aborting the surrounding write. Mirrors the
    /// `disambiguation.rs:279` swallow pattern as a reusable helper per
    /// ADR-035 §5. Added v0.1.8.
    #[allow(clippy::too_many_arguments)]
    pub async fn try_insert_fact(
        &self,
        subject_id: &str,
        predicate: &str,
        object_id: Option<&str>,
        object_value: Option<&str>,
        valid_from: DateTime<Utc>,
        confidence: f64,
        source_episode_id: Option<i64>,
        embedding: Option<&[f32]>,
    ) -> Result<Option<i64>> {
        match self
            .insert_fact(
                subject_id,
                predicate,
                object_id,
                object_value,
                valid_from,
                confidence,
                source_episode_id,
                embedding,
            )
            .await
        {
            Ok(fact_id) => Ok(Some(fact_id)),
            Err(crate::core::error::Error::Duplicate { .. }) => {
                metrics::counter!(
                    "kremory.with_facts.deduped_total",
                    "axis" => "caller_vs_llm",
                    "fn" => "try_insert_fact"
                )
                .increment(1);
                tracing::debug!(
                    subject_id,
                    predicate,
                    "kremory.try_insert_fact.swallowed_duplicate"
                );
                Ok(None)
            }
            Err(e) => Err(e),
        }
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
                "UPDATE entities SET embedding = vector(?1) WHERE id = ?2",
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
    ///
    /// **Cross-namespace collision guard (ADR-029b §3.2 — bypass surface #2)**:
    /// If `id` already exists under a DIFFERENT `group_id`, this method returns
    /// `Err(CrossNamespaceCollision { … })` rather than silently collapsing
    /// the namespaces via the old single-PK UNIQUE constraint.
    ///
    /// With the composite PK `(id, group_id)` (migration 004), the same name
    /// CAN exist in two namespaces as independent rows. This pre-check guards
    /// against unintentional cross-namespace name reuse where an explicit error
    /// is safer than silently sharing a row.
    pub async fn insert_entity_with_group(
        &self,
        id: &str,
        entity_type_id: u32,
        properties: serde_json::Value,
        group_id: Option<&str>,
    ) -> Result<()> {
        let _db_start = Instant::now();
        // ADR-029b: entities.group_id is NOT NULL post-migration-004. COALESCE maps
        // None → 'default' so callers using None-as-unscoped retain their semantics
        // while the storage constraint is satisfied.
        let effective_group_id = group_id.unwrap_or("default");
        // ADR-029b §3.2 — bypass surface #2 guard:
        // Check if the same entity name already exists under a DIFFERENT group_id.
        // With composite PK, the insert WOULD succeed, but we want an explicit error
        // so callers know they are creating a cross-namespace name collision.
        //
        // Stubs (called from the forward-reference branch in ingest.rs) intentionally
        // skip this check — stubs use INSERT OR IGNORE semantics via the match block
        // in ingest.rs, so cross-namespace stub creation is allowed (the row is
        // independent under the composite PK).
        //
        // Uses effective_group_id (not the raw Option) so that None → 'default' is
        // resolved before comparison. Without this, `IS NOT NULL` would match ALL
        // non-null rows — triggering a spurious CrossNamespaceCollision when the same
        // name is written twice to 'default'.
        {
            let mut rows = self
                .conn
                .query(
                    "SELECT group_id FROM entities WHERE id = ?1 AND group_id != ?2 LIMIT 1",
                    libsql::params![id, effective_group_id],
                )
                .await?;
            if let Some(row) = rows.next().await? {
                let existing_ns: Option<String> = row.get(0).ok();
                let existing_str = existing_ns.as_deref().unwrap_or("<null>").to_string();
                let attempted_str = effective_group_id;
                tracing::error!(
                    target: "kremory.namespace.collision",
                    entity_name = %id,
                    existing_ns = %existing_str,
                    attempted_ns = %attempted_str,
                    "cross-namespace entity name collision detected (ADR-029b bypass surface #2)"
                );
                return Err(crate::core::error::Error::CrossNamespaceCollision {
                    name: id.to_string(),
                    existing_ns: existing_str,
                    attempted_ns: attempted_str.to_string(),
                });
            }
        }
        let props_str = serde_json::to_string(&properties)?;
        let now = Utc::now().to_rfc3339();
        // Transaction guards the invariant: row in `entities` ⟹ row in `entities_fts`.
        // Uses BeginGuard so nested calls (caller already holds an outer txn) skip the
        // inner BEGIN — required to avoid "transaction within a transaction" errors when
        // called from inside the ingest pipeline's outer txn.
        let guard = self.begin_immediate_if_needed().await?;
        let inner: Result<()> = async {
            // ADR-045 §2 + migration-010-detail-spec §1.2: all callers of this method
            // are Phase 1 entity inserts. Source-tier stamped 'Phase1Ner' at INSERT time.
            // ner_confidence populated separately via set_entity_ner_confidence when
            // a GLiNER span score is available (pipeline.rs Phase 1 entity loop).
            self.conn
                .execute(
                    "INSERT INTO entities \
                     (id, entity_type_id, properties, recorded_at, group_id, \
                      entity_type_source, entity_type_assigned_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, 'Phase1Ner', ?4)",
                    libsql::params![
                        id,
                        entity_type_id as i64,
                        props_str.clone(),
                        now,
                        effective_group_id
                    ],
                )
                .await?;
            // FTS: label column in entities_fts is no longer populated (entities.label
            // was dropped in Migration 009). Insert empty string to satisfy the FTS schema.
            self.conn
                .execute(
                    "INSERT INTO entities_fts(entity_id, label, properties) VALUES (?1, '', ?2)",
                    libsql::params![id, props_str],
                )
                .await?;
            Ok(())
        }
        .await;
        match inner {
            Ok(()) => {
                guard.commit().await?;
            }
            Err(e) => {
                let _ = guard.rollback().await;
                return Err(e);
            }
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.insert_entity_with_group_ms").record(_ms);
        tracing::info!(_ms, "kremory.db.insert_entity_with_group");
        Ok(())
    }

    /// Upserts an entity row. Behaves as INSERT on first call for this (id),
    /// and as UPDATE-in-place on subsequent calls (preserves row identity).
    /// The entities_fts shadow table is updated to match via DELETE + INSERT.
    ///
    /// Used by the v0.1.1 stub-entity promotion path: a stub row inserted via
    /// the pre-scan (forward reference) is upgraded to a real entity row when
    /// the same name is later extracted as a proper entity.
    pub async fn upsert_entity_with_group(
        &self,
        id: &str,
        entity_type_id: u32,
        properties: serde_json::Value,
        group_id: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let props_str = properties.to_string();
        // ADR-029b: entities.group_id is NOT NULL post-migration-004; composite PK is
        // (id, group_id). ON CONFLICT must target the composite PK.
        // None → 'default' so callers using None-as-unscoped retain their semantics.
        let effective_group_id = group_id.unwrap_or("default");
        let guard = self.begin_immediate_if_needed().await?;
        let inner: Result<()> = async {
            // ADR-045 §2 / spec §1.2: stamp Phase1Ner on INSERT; preserve existing
            // entity_type_source on conflict (stub-promotion must not overwrite a
            // ConsumerPinned or Phase2Llm tier that was already assigned).
            self.conn
                .execute(
                    "INSERT INTO entities \
                     (id, entity_type_id, properties, recorded_at, group_id, \
                      entity_type_source, entity_type_assigned_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, 'Phase1Ner', ?4) \
                     ON CONFLICT(id, group_id) DO UPDATE SET \
                       entity_type_id = excluded.entity_type_id, \
                       properties = excluded.properties, \
                       recorded_at = excluded.recorded_at, \
                       entity_type_source = COALESCE(entities.entity_type_source, excluded.entity_type_source), \
                       entity_type_assigned_at = COALESCE(entities.entity_type_assigned_at, excluded.entity_type_assigned_at)",
                    libsql::params![
                        id,
                        entity_type_id as i64,
                        props_str.clone(),
                        now,
                        effective_group_id
                    ],
                )
                .await?;
            // FTS shadow upsert via DELETE+INSERT (FTS5 idiomatic pattern).
            // label column in entities_fts is no longer populated (entities.label
            // was dropped in Migration 009); insert empty string for that column.
            self.conn
                .execute(
                    "DELETE FROM entities_fts WHERE entity_id = ?1",
                    libsql::params![id],
                )
                .await?;
            self.conn
                .execute(
                    "INSERT INTO entities_fts(entity_id, label, properties) VALUES (?1, '', ?2)",
                    libsql::params![id, props_str],
                )
                .await?;
            Ok(())
        }
        .await;
        match inner {
            Ok(()) => {
                guard.commit().await?;
                Ok(())
            }
            Err(e) => {
                let _ = guard.rollback().await;
                Err(e)
            }
        }
    }

    /// Set `ner_confidence` on an existing entity row.
    ///
    /// Called after GLiNER extraction inserts the entity row. Only writes when
    /// `confidence` is `Some` — no-op on `None` (LLM-only paths produce no score).
    ///
    /// Governing spec: migration-010-detail-spec §1.1 + ADR-045 §6 (Phase 1 GLiNER NER).
    pub async fn set_entity_ner_confidence(
        &self,
        id: &str,
        group_id: Option<&str>,
        confidence: f32,
    ) -> Result<()> {
        let effective_group_id = group_id.unwrap_or("default");
        self.conn
            .execute(
                "UPDATE entities SET ner_confidence = ?1 \
                 WHERE id = ?2 AND group_id = ?3",
                libsql::params![confidence as f64, id, effective_group_id],
            )
            .await
            .map_err(crate::core::error::Error::from)?;
        Ok(())
    }

    /// Overwrite `entity_type_source` and `entity_type_assigned_at` on an existing entity row.
    ///
    /// Used exclusively by the ConsumerPinned write path in `pipeline.rs` after a
    /// `try_insert_fact_with_group` success — stamps `'ConsumerPinned'` so the dream
    /// reclassify pass skips this entity.
    ///
    /// Governing spec: migration-010-detail-spec §1.2 + ADR-045 §3 (ConsumerPinned protection).
    pub async fn update_entity_source_tier(
        &self,
        id: &str,
        group_id: Option<&str>,
        source_tier: &str,
    ) -> Result<()> {
        let effective_group_id = group_id.unwrap_or("default");
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "UPDATE entities \
                 SET entity_type_source = ?1, entity_type_assigned_at = ?2 \
                 WHERE id = ?3 AND group_id = ?4",
                libsql::params![source_tier, now, id, effective_group_id],
            )
            .await
            .map_err(crate::core::error::Error::from)?;
        // Rule 19 §3: per-source UPDATE counter so tier-flip writes are observable
        // independently of INSERT counters (update_entity_source_tier is UPDATE-only;
        // it never fires `entity_persisted_total`).
        metrics::counter!(
            "kremory.ingest.entity_source_tier_updated_total",
            "source" => source_tier.to_string(),
            "namespace" => effective_group_id.to_string(),
        )
        .increment(1);
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
        // FU.1: acquire BEGIN IMMEDIATE before the SELECT-check to serialise concurrent
        // writers and close the TOCTTOU window between the dup-check SELECT and the INSERT.
        let guard = self.begin_immediate_if_needed().await?;
        // Inner ops wrapped so we can explicitly rollback on Err — see insert_fact rationale.
        let result: Result<i64> = async {
            let mut dup_check = self
                .conn
                .query(
                    "SELECT id FROM facts WHERE content_hash = ?1 AND expired_at IS NULL LIMIT 1",
                    libsql::params![hash.clone()],
                )
                .await?;
            if dup_check.next().await?.is_some() {
                return Err(crate::core::error::Error::Duplicate {
                    content_hash: hash.clone(),
                });
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
                        hash.clone(),
                    ],
                )
                .await?;
            let mut rows = self.conn.query("SELECT last_insert_rowid()", ()).await?;
            let row = rows.next().await?.ok_or(
                crate::core::error::Error::InsertReturnedNoRowId {
                    operation: "insert_fact_with_group",
                },
            )?;
            let fact_id = row.get::<i64>(0)?;
            if let Some(ov) = object_value {
                self.conn
                    .execute(
                        "INSERT INTO facts_fts(fact_id, predicate, object_value) VALUES (?1, ?2, ?3)",
                        libsql::params![fact_id, predicate, ov],
                    )
                    .await?;
            }
            Ok(fact_id)
        }
        .await;
        match result {
            Ok(fact_id) => {
                guard.commit().await?;
                let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
                histogram!("rql.db.insert_fact_with_group_ms").record(_ms);
                tracing::info!(_ms, fact_id, "kremory.db.insert_fact_with_group");
                Ok(fact_id)
            }
            Err(e) => {
                let _ = guard.rollback().await;
                Err(e)
            }
        }
    }

    /// `insert_fact_with_group` with `Err(Error::Duplicate)` swallowed to `Ok(None)`.
    ///
    /// Sibling helper to `try_insert_fact`. Use this when the caller has a
    /// `group_id` available (e.g. resolved from a `Namespace`). Same
    /// silent-dedup semantics + counter emission. Added v0.1.8 per ADR-035 §5.
    #[allow(clippy::too_many_arguments)]
    pub async fn try_insert_fact_with_group(
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
    ) -> Result<Option<i64>> {
        match self
            .insert_fact_with_group(
                subject_id,
                predicate,
                object_id,
                object_value,
                valid_from,
                confidence,
                source_episode_id,
                group_id,
                embedding,
            )
            .await
        {
            Ok(fact_id) => Ok(Some(fact_id)),
            Err(crate::core::error::Error::Duplicate { .. }) => {
                metrics::counter!(
                    "kremory.with_facts.deduped_total",
                    "axis" => "caller_vs_llm",
                    "fn" => "try_insert_fact_with_group"
                )
                .increment(1);
                tracing::debug!(
                    subject_id,
                    predicate,
                    group_id,
                    "kremory.try_insert_fact_with_group.swallowed_duplicate"
                );
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

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
                // ADR-029b: composite FK field — absent on pre-migration-004 rows.
                entity_group_id: None,
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

    /// Update the `entity_type_id` for an existing entity row.
    ///
    /// Used by the L7 dream-phase reclassification pass to promote an entity
    /// from the catch-all id=0 to a concrete registered type after enough
    /// contextual episodes have accumulated.
    ///
    /// Sets `updated_at` to the current wall-clock time so callers can detect
    /// the change via audit queries.  No-op (single UPDATE) — caller is
    /// responsible for validating `new_type_id` via `EntityTypeRegistry`
    /// before calling.
    pub async fn update_entity_type_id(&self, id: &str, new_type_id: u32) -> Result<()> {
        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "UPDATE entities SET entity_type_id = ?1, updated_at = ?2 WHERE id = ?3",
                libsql::params![new_type_id as i64, now, id],
            )
            .await?;
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.update_entity_type_id_ms").record(_ms);
        tracing::info!(_ms, id, new_type_id, "kremory.db.update_entity_type_id");
        Ok(())
    }

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

    /// Return all active (non-expired) `potential_alias` facts in `group_id`.
    ///
    /// Used by the L7 dream-phase `resolve_pending_aliases` pass to identify
    /// alias candidates for confirmation or revocation.
    pub async fn get_alias_facts_in_group(&self, group_id: &str) -> Result<Vec<Fact>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id,
                        confidence, source_episode_id, memory_type, content_hash, access_count
                 FROM facts
                 WHERE predicate = 'potential_alias'
                   AND group_id = ?1
                   AND expired_at IS NULL",
                libsql::params![group_id],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.get_alias_facts_in_group_ms").record(_ms);
        tracing::info!(_ms, group_id, "kremory.db.get_alias_facts_in_group");
        Ok(facts)
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
    /// Return all non-expired facts that are missing a vector embedding.
    ///
    /// Tuple layout: `(fact_id, subject_id, predicate, object_value, object_id)`.
    ///
    /// Both `object_value` and `object_id` are included so callers can build the
    /// text-to-embed with the best available object representation:
    /// prefer `object_id` (entity reference) over `object_value` (literal string)
    /// when constructing the embedding input. Story #214 (FU.8).
    pub async fn facts_missing_embeddings(
        &self,
    ) -> Result<Vec<(i64, String, String, Option<String>, Option<String>)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_value, object_id FROM facts WHERE embedding IS NULL AND expired_at IS NULL",
                (),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let subject_id: String = row.get(1)?;
            let predicate: String = row.get(2)?;
            let object_value: Option<String> = row.get(3)?;
            let object_id: Option<String> = row.get(4)?;
            out.push((id, subject_id, predicate, object_value, object_id));
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

    /// Delete an entity and ALL dependent rows atomically. Story #216.
    ///
    /// Deletes in dependency order within a single `BEGIN IMMEDIATE` transaction:
    /// 1. `entities_fts` — standalone FTS5 virtual table; no FK cascade.
    /// 2. `episodic_edges` — FK → `entities(id)` (no ON DELETE CASCADE).
    /// 3. `facts` — FK → `entities(id)` as subject/object (no ON DELETE CASCADE).
    /// 4. `entities` — parent row.
    ///
    /// If any step fails the transaction is rolled back and no rows are changed.
    ///
    /// Returns `true` when the entity existed and was deleted; `false` when the
    /// entity was not found (idempotent — not an error).
    pub async fn forget_entity(&self, entity_id: &str) -> Result<bool> {
        let id = libsql::Value::Text(entity_id.to_owned());
        let guard = self.begin_immediate_if_needed().await?;

        // 1. FTS — standalone FTS5; no FK cascade, must delete first.
        let fts_result = self
            .conn
            .execute(
                "DELETE FROM entities_fts WHERE entity_id = ?1",
                libsql::params![id.clone()],
            )
            .await;

        // 2. Episodic edges referencing this entity.
        let edges_result = if fts_result.is_ok() {
            self.conn
                .execute(
                    "DELETE FROM episodic_edges WHERE entity_id = ?1",
                    libsql::params![id.clone()],
                )
                .await
        } else {
            fts_result
        };

        // 3. Facts where this entity is subject or object.
        let facts_result = if edges_result.is_ok() {
            self.conn
                .execute(
                    "DELETE FROM facts WHERE subject_id = ?1 OR object_id = ?1",
                    libsql::params![id.clone()],
                )
                .await
        } else {
            edges_result
        };

        // 4. Entity row itself.
        if let Err(e) = facts_result {
            guard.rollback().await?;
            return Err(e.into());
        }

        let deleted = self
            .conn
            .execute("DELETE FROM entities WHERE id = ?1", libsql::params![id])
            .await;

        match deleted {
            Err(e) => {
                guard.rollback().await?;
                Err(e.into())
            }
            Ok(n) => {
                let found = n > 0;
                guard.commit().await?;
                tracing::info!(entity_id, found, "kremory.db.forget_entity");
                Ok(found)
            }
        }
    }

    /// Delete up to 250 entities in transactional 100-item chunks. Story #217.
    ///
    /// Each chunk of up to 100 IDs is wrapped in its own `BEGIN IMMEDIATE`
    /// transaction. Deletion order per chunk: FTS → episodic_edges → facts →
    /// entities.
    ///
    /// Returns the total number of entity rows deleted across all chunks.
    pub async fn batch_forget(&self, entity_ids: &[String]) -> Result<u64> {
        const CHUNK_SIZE: usize = 100;
        let mut total_deleted: u64 = 0;

        for chunk in entity_ids.chunks(CHUNK_SIZE) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let params: Vec<libsql::Value> = chunk
                .iter()
                .map(|s| libsql::Value::Text(s.clone()))
                .collect();

            let guard = self.begin_immediate_if_needed().await?;

            // Execute the four DELETE statements; collect the first error.
            macro_rules! try_delete {
                ($sql:expr, $p:expr) => {
                    match self.conn.execute(&$sql, $p).await {
                        Ok(n) => n,
                        Err(e) => {
                            guard.rollback().await?;
                            return Err(e.into());
                        }
                    }
                };
            }

            // 1. FTS.
            try_delete!(
                format!("DELETE FROM entities_fts WHERE entity_id IN ({placeholders})"),
                params.clone()
            );

            // 2. Episodic edges.
            try_delete!(
                format!("DELETE FROM episodic_edges WHERE entity_id IN ({placeholders})"),
                params.clone()
            );

            // 3. Facts (subject or object).
            //    Two IN clauses → params must be doubled.
            let sql_facts = format!(
                "DELETE FROM facts WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})"
            );
            let mut doubled = params.clone();
            doubled.extend_from_slice(&params);
            try_delete!(sql_facts, doubled);

            // 4. Entity rows.
            let n = try_delete!(
                format!("DELETE FROM entities WHERE id IN ({placeholders})"),
                params
            );

            guard.commit().await?;
            total_deleted += n;
        }

        tracing::info!(
            count = entity_ids.len(),
            deleted = total_deleted,
            "kremory.db.batch_forget"
        );
        Ok(total_deleted)
    }

    // ── Namespace policy (ADR-029a, v0.1.4) ──────────────────────────────────
    //
    // All three methods are `pub(crate)` per Vera cycle-1 MED-6 — only the
    // `facade` layer (kremory::Memory::register_namespace) and the lazy-population
    // wiring should call them. External consumers go through the facade and get
    // validation + idempotency + tracing + race-safety.

    /// Read the persisted [`crate::memory::types::NamespacePolicy`] for a
    /// `group_id`, or `None` if the namespace has not been observed.
    ///
    /// Used by `register_namespace` to detect idempotency vs immutable-conflict.
    /// Substrate-only (ADR-029a Decision 8).
    pub(crate) async fn get_namespace_policy(
        &self,
        group_id: &str,
    ) -> Result<Option<crate::memory::types::NamespacePolicy>> {
        let started = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT policy_json FROM namespaces WHERE group_id = ? LIMIT 1",
                libsql::params![group_id.to_string()],
            )
            .await?;
        let policy = if let Some(row) = rows.next().await? {
            let json: String = row.get(0)?;
            let parsed: crate::memory::types::NamespacePolicy = serde_json::from_str(&json)?;
            Some(parsed)
        } else {
            None
        };
        histogram!("kremory_core_namespace_policy_get_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(policy)
    }

    /// Write a `NamespacePolicy` for a `group_id` if not already present.
    ///
    /// Substrate-only. Caller (`register_namespace`) holds the BEGIN IMMEDIATE
    /// guard for race safety; this method does NOT manage its own transaction.
    /// `ON CONFLICT DO NOTHING` keeps the operation idempotent at the SQL level
    /// when called concurrently (ADR-029a Decision 8).
    pub(crate) async fn set_namespace_policy(
        &self,
        group_id: &str,
        policy: &crate::memory::types::NamespacePolicy,
    ) -> Result<()> {
        let started = Instant::now();
        let json = serde_json::to_string(policy)?;
        self.conn
            .execute(
                "INSERT INTO namespaces (group_id, policy_json, schema_version) \
                 VALUES (?, ?, 1) \
                 ON CONFLICT(group_id) DO NOTHING",
                libsql::params![group_id.to_string(), json],
            )
            .await?;
        histogram!("kremory_core_namespace_policy_set_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Write an updated `NamespacePolicy` for an existing `group_id`, stamping
    /// `upgraded_at = now()`. Used by `Memory::upgrade_namespace_policy` for
    /// the monotonic Mutable → AppendOnly upgrade (ADR-029b Decision 5).
    ///
    /// Caller holds the `BEGIN IMMEDIATE` guard. This method does NOT open a
    /// transaction — it is meant to be called inside the caller's atomic block.
    ///
    /// Sets `upgraded_at` to the current UTC time in RFC3339 format.
    pub(crate) async fn set_namespace_policy_with_upgraded_at(
        &self,
        group_id: &str,
        policy: &crate::memory::types::NamespacePolicy,
    ) -> Result<()> {
        let started = Instant::now();
        let json = serde_json::to_string(policy)?;
        let now = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO namespaces (group_id, policy_json, upgraded_at, schema_version) \
                 VALUES (?, ?, ?, 1) \
                 ON CONFLICT(group_id) DO UPDATE SET \
                   policy_json = excluded.policy_json, \
                   upgraded_at = excluded.upgraded_at",
                libsql::params![group_id.to_string(), json, now],
            )
            .await?;
        histogram!("kremory_core_namespace_policy_upgrade_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Ensure a default-policy row exists for `group_id` on implicit
    /// observation. Called by `remember`/`recall`/`forget`/`dream` on first
    /// encounter with a previously-unregistered namespace.
    ///
    /// # Race safety (Vera cycle-2 ASMP-001)
    ///
    /// This method opens its own `BEGIN IMMEDIATE` guard via
    /// `begin_immediate_if_needed`, which is a no-op when nested under an
    /// existing outer transaction. The wrapping serializes against concurrent
    /// `register_namespace` calls. `INSERT OR IGNORE` is itself idempotent —
    /// the guard ensures the surrounding read-decide sequence in
    /// `register_namespace` stays consistent.
    pub(crate) async fn ensure_namespace_policy_row(&self, group_id: &str) -> Result<()> {
        let started = Instant::now();
        // Default policy serialized inline — matches NamespacePolicy::default().
        const DEFAULT_POLICY_JSON: &str =
            r#"{"immutability":"mutable","forgettable":true,"dream_eligible":true}"#;
        let guard = self.begin_immediate_if_needed().await?;
        let result = self
            .conn
            .execute(
                "INSERT OR IGNORE INTO namespaces (group_id, policy_json, schema_version) \
                 VALUES (?, ?, 1)",
                libsql::params![group_id.to_string(), DEFAULT_POLICY_JSON.to_string()],
            )
            .await;
        match result {
            Ok(_) => {
                guard.commit().await?;
                histogram!("kremory_core_namespace_policy_ensure_seconds")
                    .record(started.elapsed().as_secs_f64());
                Ok(())
            }
            Err(e) => {
                guard.rollback().await?;
                Err(e.into())
            }
        }
    }

    /// Read the namespace policy using the per-handle LRU cache (ADR-029b Decision 4).
    ///
    /// Cache hit: returns the cached policy immediately (no DB read).
    /// Cache miss: reads from `namespaces` table, populates cache, returns result.
    ///
    /// The cache is a per-handle LRU (capacity 256) so different `TemporalGraph`
    /// instances have independent caches — no cross-handle invalidation needed.
    pub(crate) async fn get_namespace_policy_cached(
        &self,
        group_id: &str,
    ) -> Result<Option<crate::memory::types::NamespacePolicy>> {
        // Cache hit — peek does not update LRU recency on a miss path.
        if let Some(policy) = self.policy_cache.peek(group_id) {
            return Ok(Some(policy));
        }
        // Cache miss — read from DB and populate.
        let policy = self.get_namespace_policy(group_id).await?;
        if let Some(ref p) = policy {
            self.policy_cache.put(group_id, p.clone());
        }
        Ok(policy)
    }

    /// Evict the policy cache entry for `group_id`.
    ///
    /// Called by `upgrade_namespace_policy` after committing the upgrade so
    /// subsequent reads reflect the new AppendOnly policy.
    pub(crate) fn invalidate_policy_cache(&self, group_id: &str) {
        self.policy_cache.invalidate(group_id);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::schema::TemporalGraph;
    use chrono::Duration;

    // === Entity CRUD ===

    #[tokio::test]
    async fn test_insert_and_get_entity() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        let entity = g.get_entity("alice").await.unwrap().unwrap();
        assert_eq!(entity.id, "alice");
        // entity_type_id=0 (catch-all) → label resolves to "Entity" via LEFT JOIN COALESCE.
        assert_eq!(entity.entity_type_id, 0);
        assert_eq!(entity.label, "Entity");
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
        g.insert_entity("alice", 0, serde_json::json!({"role": "engineer"}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let id = g
            .insert_fact("alice", "exists", None, None, Utc::now(), 1.0, None, None)
            .await
            .unwrap();
        assert!(id > 0);
    }

    /// ADR-035 §5 Option A: `try_insert_fact` returns `Ok(Some(id))` on fresh insert.
    #[tokio::test]
    async fn test_try_insert_fact_returns_some_on_fresh_insert() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let id = g
            .try_insert_fact("alice", "exists", None, None, Utc::now(), 1.0, None, None)
            .await
            .unwrap();
        assert!(id.is_some(), "fresh insert must return Some(id)");
        assert!(id.unwrap() > 0);
    }

    /// ADR-035 §5 Option A: `try_insert_fact` returns `Ok(None)` on content_hash collision.
    #[tokio::test]
    async fn test_try_insert_fact_returns_none_on_duplicate() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now();
        let _first = g
            .insert_fact("alice", "exists", None, None, t0, 1.0, None, None)
            .await
            .unwrap();
        let dup = g
            .try_insert_fact("alice", "exists", None, None, t0, 1.0, None, None)
            .await
            .unwrap();
        assert!(
            dup.is_none(),
            "duplicate triple must return None (silent dedup)"
        );
    }

    /// ADR-035 §5 Option A: `try_insert_fact_with_group` returns `Ok(Some(id))` on fresh insert.
    #[tokio::test]
    async fn test_try_insert_fact_with_group_returns_some_on_fresh_insert() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        // Entities are global (FK on entity id only, not (id, group_id)) — pattern
        // mirrors search.rs:1700 + 1704 successful with_group tests.
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let id = g
            .try_insert_fact_with_group(
                "alice",
                "exists",
                None,
                None,
                Utc::now(),
                1.0,
                None,
                Some("g1"),
                None,
            )
            .await
            .unwrap();
        assert!(id.is_some(), "fresh insert must return Some(id)");
    }

    /// ADR-035 finding: `content_hash` does NOT include `group_id` — same triple
    /// in different groups still collides. This test pins the invariant that
    /// caller's `insert_fact_with_group(group=X)` and engine's `insert_fact(no group)`
    /// will dedup against each other (the basis of Path X caller-wins semantics).
    #[tokio::test]
    async fn test_try_insert_fact_with_group_dedups_cross_variant() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let t0 = Utc::now();
        // First: insert via plain insert_fact (no group_id) — simulates LLM Phase 2.
        let _first = g
            .insert_fact("alice", "exists", None, None, t0, 1.0, None, None)
            .await
            .unwrap();
        // Second: try_insert_fact_with_group (with group_id) — simulates caller pin.
        // Expect None: content_hash collides regardless of group_id.
        let dup = g
            .try_insert_fact_with_group(
                "alice",
                "exists",
                None,
                None,
                t0,
                1.0,
                None,
                Some("g1"),
                None,
            )
            .await
            .unwrap();
        assert!(
            dup.is_none(),
            "with_group + no_group SAME triple must collide (content_hash is group-agnostic)"
        );
    }

    #[tokio::test]
    async fn test_insert_fact_with_object_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("newco", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
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
            0,
            serde_json::json!({"name": "Alice"}),
            Some("group-abc"),
        )
        .await
        .unwrap();

        let entity = g.get_entity("alice").await.unwrap().unwrap();
        assert_eq!(entity.id, "alice");
        assert_eq!(entity.group_id.as_deref(), Some("group-abc"));
    }

    /// Stream 3 A.4.5 contract test — updated for ADR-029b:
    /// Lane A indexer call sites pass `group_id=None` (folder_id absent pre-Lane-B).
    /// Post-ADR-029b, None maps to 'default' (entities.group_id is NOT NULL).
    /// The entity is stored in the default namespace and is visible under
    /// unscoped queries or queries scoped to 'default'.
    #[tokio::test]
    async fn lane_a_indexer_writes_null_group_id_when_folder_id_absent() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        let group_id: Option<&str> = None;
        g.insert_entity_with_group(
            "doc:welcome:chunk_0",
            0,
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
        // ADR-029b: None maps to 'default' — not NULL (NOT NULL constraint enforced).
        assert_eq!(
            entity.group_id.as_deref(),
            Some("default"),
            "Lane A indexer: None group_id must persist as 'default' post-ADR-029b (got {:?})",
            entity.group_id,
        );
    }

    #[tokio::test]
    async fn test_update_entity_group_changes_scope() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entity in group-a
        g.insert_entity_with_group(
            "doc_chunk_0",
            0,
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
        // ADR-029b: entities.group_id is NOT NULL post-migration-004.
        // Passing None to update_entity_group maps to 'default' (not NULL).
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group("e1", 0, serde_json::json!({}), Some("scoped"))
            .await
            .unwrap();

        g.update_entity_group("e1", None, serde_json::json!({}))
            .await
            .unwrap();

        let entity = g.get_entity("e1").await.unwrap().unwrap();
        // None maps to 'default' — entity is now in the default namespace.
        assert_eq!(
            entity.group_id.as_deref(),
            Some("default"),
            "None group_id must map to 'default' post-ADR-029b"
        );
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
            .execute("DROP TABLE entities_fts", ())
            .await
            .expect("DROP TABLE entities_fts must succeed on fresh in-memory DB");

        let result = g
            .insert_entity("test-atomic-1", 0, serde_json::json!({"text": "hello"}))
            .await;
        assert!(
            result.is_err(),
            "insert_entity must return Err when FTS table is missing (no partial commit)"
        );

        // The entities row must have been rolled back — zero rows for the attempted ID.
        let mut rows = g
            .conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE id = 'test-atomic-1'",
                (),
            )
            .await
            .expect("SELECT on entities must succeed even after FTS drop");
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
            .execute("DROP TABLE entities_fts", ())
            .await
            .expect("DROP TABLE entities_fts must succeed on fresh in-memory DB");

        let result = g
            .insert_entity_with_group(
                "test-atomic-grp-1",
                0,
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
                "SELECT COUNT(*) FROM entities WHERE id = 'test-atomic-grp-1'",
                (),
            )
            .await
            .expect("SELECT on entities must succeed even after FTS drop");
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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
        g.insert_entity("alice", 0, serde_json::json!({}))
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

    /// FU.8: facts_missing_embeddings returns (id, subject_id, predicate,
    /// object_value, object_id) — the 5-tuple now includes object_id so callers
    /// can prefer the entity reference over the literal string.
    #[tokio::test]
    async fn facts_missing_embeddings_returns_all_null_embedding_facts() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        // Insert a fact with object_value only — simulates SQLite-committed, vector-not-written
        let fact_id = g
            .insert_fact("alice", "works_at", None, Some("ACME"), t, 1.0, None, None)
            .await
            .unwrap();

        let missing = g.facts_missing_embeddings().await.unwrap();
        assert_eq!(missing.len(), 1, "one fact has no embedding");
        let (id, _sub, _pred, object_value, object_id) = &missing[0];
        assert_eq!(*id, fact_id);
        assert_eq!(object_value.as_deref(), Some("ACME"));
        assert!(
            object_id.is_none(),
            "object_id must be None for literal-value fact"
        );

        // Backfill with a stub embedding
        let embedding: Vec<f32> = vec![0.1_f32; 384];
        g.backfill_fact_embedding(fact_id, &embedding)
            .await
            .unwrap();

        // After backfill, no facts should be missing
        let still_missing = g.facts_missing_embeddings().await.unwrap();
        assert!(
            still_missing.is_empty(),
            "backfill must clear the missing-embedding list"
        );
    }

    /// FU.8: facts with object_id (entity reference) expose the object_id in the
    /// 5-tuple so callers can build the embedding text from the entity name rather
    /// than a missing literal.
    #[tokio::test]
    async fn facts_missing_embeddings_includes_object_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        // Insert a fact with object_id (entity reference) — no object_value.
        let fact_id = g
            .insert_fact("alice", "works_at", Some("acme"), None, t, 1.0, None, None)
            .await
            .unwrap();

        let missing = g.facts_missing_embeddings().await.unwrap();
        let entry = missing
            .iter()
            .find(|(id, _, _, _, _)| *id == fact_id)
            .expect("fact must appear in missing list");
        let (_id, _sub, _pred, object_value, object_id) = entry;
        assert!(
            object_value.is_none(),
            "object_value must be None for entity-ref fact"
        );
        assert_eq!(
            object_id.as_deref(),
            Some("acme"),
            "object_id must be returned so callers can build the embedding text"
        );
    }

    #[tokio::test]
    async fn backfill_fact_embedding_is_idempotent() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        g.insert_entity("bob", 0, serde_json::json!({}))
            .await
            .unwrap();
        let t = Utc::now();
        let fact_id = g
            .insert_fact("bob", "knows", None, Some("alice"), t, 1.0, None, None)
            .await
            .unwrap();
        let embedding: Vec<f32> = vec![0.2_f32; 384];
        // First backfill
        g.backfill_fact_embedding(fact_id, &embedding)
            .await
            .unwrap();
        // Second backfill on same fact must not error
        g.backfill_fact_embedding(fact_id, &embedding)
            .await
            .unwrap();
    }

    /// FU.1: N concurrent callers racing to insert_fact with the same content must
    /// produce exactly 1 Ok(id) and N-1 Err(Duplicate). Only 1 row must exist.
    ///
    /// AC from Story #209 / FU.1: insert_fact_concurrent_dedup_exactly_one_wins.
    #[tokio::test]
    async fn insert_fact_concurrent_dedup_exactly_one_wins() {
        use std::sync::Arc;
        use tokio::sync::Barrier;

        const N: usize = 4;

        let g = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(N));
        let valid_from = Utc::now();

        let handles: Vec<_> = (0..N)
            .map(|_| {
                let g = Arc::clone(&g);
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    // All tasks reach the barrier before any begins — maximises race window.
                    barrier.wait().await;
                    g.insert_fact(
                        "alice",
                        "concurrent_pred",
                        None,
                        Some("same_value"),
                        valid_from,
                        1.0,
                        None,
                        None,
                    )
                    .await
                })
            })
            .collect();

        let mut ok_count = 0usize;
        let mut dup_count = 0usize;
        for h in handles {
            match h.await.expect("task panicked") {
                Ok(_) => ok_count += 1,
                Err(crate::core::error::Error::Duplicate { .. }) => dup_count += 1,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }

        assert_eq!(ok_count, 1, "exactly 1 insert must succeed");
        assert_eq!(
            dup_count,
            N - 1,
            "all other N-1 callers must return Duplicate"
        );

        // Verify exactly 1 row with this content_hash in facts
        let mut rows = g
            .conn
            .query(
                "SELECT COUNT(*) FROM facts WHERE predicate = 'concurrent_pred' AND expired_at IS NULL",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        assert_eq!(
            count, 1,
            "exactly 1 row must exist in facts after concurrent inserts"
        );
    }

    // === forget_entity — cascade delete atomicity (Story #216) ===

    #[tokio::test]
    async fn forget_entity_removes_entity_facts_and_edges() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entity + fact + episodic edge referencing it.
        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();
        let now = chrono::Utc::now();
        // insert_fact: (subject_id, predicate, object_id, object_value,
        //               valid_from, confidence, source_episode_id, embedding)
        // Use object_value (not object_id) to avoid FK on entities for "bob".
        g.insert_fact("alice", "knows", None, Some("bob"), now, 0.9, None, None)
            .await
            .unwrap();
        // insert_episodic_edge: (episode_id: i64, entity_id, role)
        // Must have a valid episode first (FK constraint).
        let ep_id = g
            .insert_episode("test episode", now, None, None)
            .await
            .unwrap();
        g.insert_episodic_edge(ep_id, "alice", "subject")
            .await
            .unwrap();

        // Verify setup.
        assert!(g.get_entity("alice").await.unwrap().is_some());

        // Forget should return true (entity was present).
        let found = g.forget_entity("alice").await.unwrap();
        assert!(found, "forget_entity must return true when entity existed");

        // Entity gone.
        assert!(g.get_entity("alice").await.unwrap().is_none());

        // Facts with alice as subject should be gone.
        let facts = g.entity_history("alice").await.unwrap();
        assert!(facts.is_empty(), "facts referencing alice must be deleted");

        // Episodic edges gone.
        let edges = g.episodic_edges_for_entity("alice").await.unwrap();
        assert!(edges.is_empty(), "episodic_edges for alice must be deleted");
    }

    #[tokio::test]
    async fn forget_entity_returns_false_for_nonexistent() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let found = g.forget_entity("ghost").await.unwrap();
        assert!(!found, "forget_entity must return false when entity absent");
    }

    // === batch_forget — 100-item chunk deletion (Story #217) ===

    #[tokio::test]
    async fn batch_forget_250_entities_deleted_cleanly() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert 250 entities.
        let ids: Vec<String> = (0..250).map(|i| format!("ent-{i:04}")).collect();
        for id in &ids {
            g.insert_entity(id, 0, serde_json::json!({})).await.unwrap();
        }

        // Verify setup.
        let before = g.list_entities().await.unwrap();
        assert_eq!(before.len(), 250);

        let deleted = g.batch_forget(&ids).await.unwrap();
        assert_eq!(deleted, 250, "batch_forget must delete all 250 entities");

        let after = g.list_entities().await.unwrap();
        assert!(
            after.is_empty(),
            "no entities must remain after batch_forget"
        );
    }

    #[tokio::test]
    async fn batch_forget_empty_slice_is_noop() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let deleted = g.batch_forget(&[]).await.unwrap();
        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn upsert_entity_with_group_inserts_then_updates() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = format!("{}/test.db", tmp.path().display());
        let g = TemporalGraph::open(&db).await.expect("open");

        // First call inserts (entity_type_id=0 = catch-all).
        g.upsert_entity_with_group("alice", 0, serde_json::json!({"context": "v1"}), None)
            .await
            .expect("first upsert");

        // After Phase 2 (Migration 009), entities.label is gone.
        // Verify via entity_type_id + properties columns.
        let mut rows = g
            .conn
            .query(
                "SELECT entity_type_id, properties FROM entities WHERE id = ?1",
                libsql::params!["alice"],
            )
            .await
            .expect("query");
        let r1 = rows.next().await.expect("row").expect("some");
        let etype1: i64 = r1.get(0).expect("entity_type_id");
        let props1: String = r1.get(1).expect("props");
        assert_eq!(etype1, 0, "first upsert entity_type_id must be 0");
        assert!(props1.contains("v1"));

        // Second call updates in place — same id, no duplicate row.
        // entity_type_id changes from 0 to 1 to verify ON CONFLICT updates it.
        g.upsert_entity_with_group("alice", 1, serde_json::json!({"context": "v2"}), None)
            .await
            .expect("second upsert");

        let mut count_rows = g
            .conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE id = ?1",
                libsql::params!["alice"],
            )
            .await
            .expect("count");
        let cr = count_rows.next().await.expect("cnt").expect("some");
        let count: i64 = cr.get(0).expect("count");
        assert_eq!(count, 1, "upsert must not create duplicate");

        let mut rows2 = g
            .conn
            .query(
                "SELECT entity_type_id, properties FROM entities WHERE id = ?1",
                libsql::params!["alice"],
            )
            .await
            .expect("query2");
        let r2 = rows2.next().await.expect("row").expect("some");
        let etype2: i64 = r2.get(0).expect("entity_type_id");
        let props2: String = r2.get(1).expect("props");
        assert_eq!(
            etype2, 1,
            "entity_type_id must be updated by ON CONFLICT path"
        );
        assert!(props2.contains("v2"), "properties must be updated");

        // FTS shadow row consistency — exactly 1 fts row after upsert (DELETE+INSERT).
        let mut fts_count = g
            .conn
            .query(
                "SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?1",
                libsql::params!["alice"],
            )
            .await
            .expect("fts count");
        let fc = fts_count.next().await.expect("row").expect("some");
        let fts_n: i64 = fc.get(0).expect("n");
        assert_eq!(fts_n, 1, "FTS shadow must have exactly 1 row after upsert");
    }
}
