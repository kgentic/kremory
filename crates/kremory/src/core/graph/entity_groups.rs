use chrono::Utc;
use metrics::histogram;
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

/// Bundled parameters for [`TemporalGraph::insert_entity_with_group`] —
/// args-as-object to satisfy a too-many-arguments lint.
pub struct InsertEntityWithGroupParams<'a> {
    pub id: &'a str,
    pub entity_type_id: u32,
    pub properties: serde_json::Value,
    pub group_id: Option<&'a str>,
}

/// Bundled parameters for [`TemporalGraph::upsert_entity_with_group`] —
/// args-as-object to satisfy a too-many-arguments lint.
pub struct UpsertEntityWithGroupParams<'a> {
    pub id: &'a str,
    pub entity_type_id: u32,
    pub properties: serde_json::Value,
    pub group_id: Option<&'a str>,
}

/// Bundled parameters for [`TemporalGraph::set_entity_ner_confidence`] —
/// args-as-object to satisfy a too-many-arguments lint.
pub struct SetEntityNerConfidenceParams<'a> {
    pub id: &'a str,
    pub group_id: Option<&'a str>,
    pub confidence: f32,
}

/// Bundled parameters for [`TemporalGraph::update_entity_source_tier`] —
/// args-as-object to satisfy a too-many-arguments lint.
pub struct UpdateEntitySourceTierParams<'a> {
    pub id: &'a str,
    pub group_id: Option<&'a str>,
    pub source_tier: &'a str,
}

impl TemporalGraph {
    /// Insert an entity with an optional group_id.
    /// Existing tests use `insert_entity`; this variant is for new code that needs group scoping.
    ///
    /// **Entity identity is per-namespace-open**: the composite PK `(id, group_id)`
    /// (migration 004) makes the same surface name in two namespaces two
    /// independent rows — the competitor-standard model (Graphiti/Zep/mem0/Neo4j
    /// all isolate per partition). This insert therefore SUCCEEDS across
    /// namespaces. A cross-namespace name reuse is emitted as a NON-BLOCKING
    /// observability signal (trace + `rql.entity.cross_namespace_collision_total`
    /// counter), not an error — so multi-tenant/multi-conversation ingest is not
    /// aborted by a generic recurring name ("the user", "Alice", "session 1").
    ///
    /// Strict global uniqueness (audit-grade single-tenant) is a NAMED-but-not-yet
    /// -built opt-in (`NamespacePolicy::with_entity_identity_scope`). Dangling
    /// `subject_id` references remain protected by the FK constraint
    /// `facts → entities` (`PRAGMA foreign_keys = ON`), a separate mechanism
    /// untouched here.
    pub async fn insert_entity_with_group(
        &self,
        params: InsertEntityWithGroupParams<'_>,
    ) -> Result<()> {
        let InsertEntityWithGroupParams {
            id,
            entity_type_id,
            properties,
            group_id,
        } = params;
        let _db_start = Instant::now();
        // entities.group_id is NOT NULL post-migration-004. COALESCE maps
        // None → 'default' so callers using None-as-unscoped retain their semantics
        // while the storage constraint is satisfied.
        let effective_group_id = group_id.unwrap_or("default");
        // Per-namespace-open: the same surface name in a DIFFERENT group_id is a
        // legitimate independent row under the composite PK — NOT an error. We
        // keep the detection purely as a NON-BLOCKING observability signal (a
        // consumer may still want to know "you wrote a name that also exists in
        // namespace X"), but the insert proceeds.
        //
        // Uses effective_group_id (not the raw Option) so None → 'default' is resolved
        // before comparison (so the same name written twice to 'default' is NOT flagged).
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
                // This signal had a log line but no metric. Add the counter so
                // cross-namespace name reuse is countable (e.g. to detect a
                // consumer that expected strict-global identity).
                metrics::counter!(
                    "rql.entity.cross_namespace_collision_total",
                    "existing_ns" => existing_str.clone(),
                    "attempted_ns" => attempted_str.to_string(),
                )
                .increment(1);
                tracing::warn!(
                    target: "kremory.namespace.collision",
                    entity_name = %id,
                    existing_ns = %existing_str,
                    attempted_ns = %attempted_str,
                    "cross-namespace entity name reuse (per-namespace-open, ADR-029d — permitted, not an error)"
                );
                // Per-namespace-open: NO `return Err` — the insert below proceeds.
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
            // All callers of this method are Phase 1 entity inserts. Source-tier
            // stamped 'Phase1Ner' at INSERT time. ner_confidence populated
            // separately via set_entity_ner_confidence when a GLiNER span score
            // is available (pipeline.rs Phase 1 entity loop).
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
        params: UpsertEntityWithGroupParams<'_>,
    ) -> Result<()> {
        let UpsertEntityWithGroupParams {
            id,
            entity_type_id,
            properties,
            group_id,
        } = params;
        let now = Utc::now().to_rfc3339();
        let props_str = properties.to_string();
        // entities.group_id is NOT NULL post-migration-004; composite PK is
        // (id, group_id). ON CONFLICT must target the composite PK.
        // None → 'default' so callers using None-as-unscoped retain their semantics.
        let effective_group_id = group_id.unwrap_or("default");
        let guard = self.begin_immediate_if_needed().await?;
        let inner: Result<()> = async {
            // Stamp Phase1Ner on INSERT; preserve existing entity_type_source on
            // conflict (stub-promotion must not overwrite a ConsumerPinned or
            // Phase2Llm tier that was already assigned).
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
    pub async fn set_entity_ner_confidence(
        &self,
        params: SetEntityNerConfidenceParams<'_>,
    ) -> Result<()> {
        let SetEntityNerConfidenceParams {
            id,
            group_id,
            confidence,
        } = params;
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
    pub async fn update_entity_source_tier(
        &self,
        params: UpdateEntitySourceTierParams<'_>,
    ) -> Result<()> {
        let UpdateEntitySourceTierParams {
            id,
            group_id,
            source_tier,
        } = params;
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
        // Per-source UPDATE counter so tier-flip writes are observable
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
