use chrono::Utc;
use metrics::histogram;
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::{Entity, TemporalGraph};

use super::row_to_entity;

/// Bundled parameters for [`TemporalGraph::insert_entity`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
pub struct InsertEntityParams<'a> {
    pub id: &'a str,
    pub entity_type_id: u32,
    pub properties: serde_json::Value,
}

/// Bundled parameters for [`TemporalGraph::update_entity_group`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments).
pub struct UpdateEntityGroupParams<'a> {
    pub id: &'a str,
    pub group_id: Option<&'a str>,
    pub properties: serde_json::Value,
}

/// Bundled parameters for [`TemporalGraph::reassign_entity_group_dangerous`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct ReassignEntityGroupDangerousParams<'a> {
    pub id: &'a str,
    pub old_group_id: &'a str,
    pub new_group_id: Option<&'a str>,
    pub bypass_policy: bool,
}

impl TemporalGraph {
    pub async fn insert_entity(&self, params: InsertEntityParams<'_>) -> Result<()> {
        let InsertEntityParams {
            id,
            entity_type_id,
            properties,
        } = params;
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
                // R2.2 §spec-td-085: error-path counter + paired warn (ADR D1).
                // Reason labels per pre-R2 enumeration table §2.
                let reason: &'static str = match &e {
                    crate::core::error::Error::Serialization(_) => "serialization",
                    crate::core::error::Error::Database(_) => "db_error",
                    _ => "other",
                };
                metrics::counter!(
                    "kremory.db.insert_entity_error_total",
                    "reason" => reason,
                )
                .increment(1);
                tracing::warn!(
                    error = %e,
                    reason = %reason,
                    "kremory.db.insert_entity failed"
                );
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
    pub async fn update_entity_group(&self, params: UpdateEntityGroupParams<'_>) -> Result<()> {
        let UpdateEntityGroupParams {
            id,
            group_id,
            properties,
        } = params;
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
        params: ReassignEntityGroupDangerousParams<'_>,
    ) -> Result<u64> {
        let ReassignEntityGroupDangerousParams {
            id,
            old_group_id,
            new_group_id,
            bypass_policy,
        } = params;
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
}
