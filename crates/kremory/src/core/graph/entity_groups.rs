use chrono::Utc;
use metrics::histogram;
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

impl TemporalGraph {
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
}
