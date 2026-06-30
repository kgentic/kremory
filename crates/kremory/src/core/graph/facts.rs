use chrono::{DateTime, Utc};
use metrics::histogram;
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::{Fact, TemporalGraph};

use super::{fact_content_hash, row_to_fact, FactContentHashParams};

/// Bundled parameters for the `insert_fact*` family — args-as-object per TD-042
/// (rust-conventions §too_many_arguments). Construct via [`FactInsert::new`] +
/// chainable setters. `group_id` is intentionally NOT a field: the
/// `*_with_group` variants take it as a sibling arg so the group capability
/// stays explicit to those methods.
pub struct FactInsert<'a> {
    pub subject_id: &'a str,
    pub predicate: &'a str,
    pub object_id: Option<&'a str>,
    pub object_value: Option<&'a str>,
    pub valid_from: DateTime<Utc>,
    pub confidence: f64,
    pub source_episode_id: Option<i64>,
    pub embedding: Option<&'a [f32]>,
}

impl<'a> FactInsert<'a> {
    /// Required fields; optionals default to `None`, `confidence` to `1.0`.
    pub fn new(subject_id: &'a str, predicate: &'a str, valid_from: DateTime<Utc>) -> Self {
        Self {
            subject_id,
            predicate,
            object_id: None,
            object_value: None,
            valid_from,
            confidence: 1.0,
            source_episode_id: None,
            embedding: None,
        }
    }

    #[must_use]
    pub fn object_id(mut self, object_id: &'a str) -> Self {
        self.object_id = Some(object_id);
        self
    }

    #[must_use]
    pub fn object_value(mut self, object_value: &'a str) -> Self {
        self.object_value = Some(object_value);
        self
    }

    #[must_use]
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    #[must_use]
    pub fn source_episode_id(mut self, source_episode_id: i64) -> Self {
        self.source_episode_id = Some(source_episode_id);
        self
    }

    #[must_use]
    pub fn embedding(mut self, embedding: &'a [f32]) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Bundled parameters for [`TemporalGraph::invalidate_fact_with_reason`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct InvalidateFactWithReasonParams {
    pub fact_id: i64,
    pub expired_at: DateTime<Utc>,
    pub invalid_at: DateTime<Utc>,
}

impl TemporalGraph {
    pub async fn insert_fact(&self, fact: FactInsert<'_>) -> Result<i64> {
        // TD-080 #3 (2026-06-29, spec
        // `td-080-facade-fact-quality-and-namespace-clearance-sprint-2026-06-29.md` §P2):
        // delegate to the group-aware path. `insert_fact` is DEFAULT-NAMESPACE-ONLY by
        // contract; any namespaced caller MUST use `insert_fact_with_group(.., Some(ns))`
        // so the composite FK resolves (see the TD-080 #1 background-path fix).
        //
        // We pass `Some("default")` — NOT `None`. `facts.subject_group_id` is NOT NULL
        // (schema DEFAULT 'default' applies only when the column is omitted from the
        // INSERT list; explicitly binding NULL violates NOT NULL). The prior bespoke INSERT
        // omitted these columns so the schema DEFAULT took effect; this delegation makes
        // that intent explicit at the call site rather than relying on a silent DDL default.
        self.insert_fact_with_group(fact, Some("default")).await
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
    pub async fn try_insert_fact(&self, fact: FactInsert<'_>) -> Result<Option<i64>> {
        let subject_id = fact.subject_id;
        let predicate = fact.predicate;
        match self.insert_fact(fact).await {
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
            // P3: self-loop counter already fired in insert_fact_with_group; swallow here
            // so a self-loop in a batch doesn't abort the surrounding fact writes.
            Err(crate::core::error::Error::SelfLoop { .. }) => {
                tracing::debug!(
                    subject_id,
                    predicate,
                    "kremory.try_insert_fact.swallowed_self_loop"
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

    /// Insert a fact with an optional group_id.
    pub async fn insert_fact_with_group(
        &self,
        fact: FactInsert<'_>,
        group_id: Option<&str>,
    ) -> Result<i64> {
        let FactInsert {
            subject_id,
            predicate,
            object_id,
            object_value,
            valid_from,
            confidence,
            source_episode_id,
            embedding,
        } = fact;

        // P3 self-loop guard (TD-080 §P3): formal invariant check — no DB needed.
        // object_value facts have object_id=None and are never self-loops.
        // Counter fires PRE-swallow so try_* callers still feed P5 quality metrics.
        if object_id == Some(subject_id) {
            metrics::counter!("kremory.fact.rejected_total", "reason" => "self_loop").increment(1);
            // Dual-emit (ADR D1 / R1.2): warn paired with counter above.
            tracing::warn!(
                subject_id = %subject_id,
                object_id = ?object_id,
                predicate = ?predicate,
                "kremory.fact.rejected self_loop"
            );
            return Err(crate::core::error::Error::SelfLoop {
                subject_id: subject_id.to_owned(),
                predicate: predicate.to_owned(),
            });
        }

        let _db_start = Instant::now();
        let now = Utc::now().to_rfc3339();
        let valid_from_str = valid_from.to_rfc3339();
        // Story #209: compute SHA-256 content hash for dedup
        let hash = fact_content_hash(FactContentHashParams {
            subject_id,
            predicate,
            object_id,
            object_value,
        });
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
            // Composite-FK columns (schema.rs:1450 / defs_h): the facts table enforces
            // `(subject_id, subject_group_id) REFERENCES entities(id, group_id)` and the same
            // for object. Entities live in the episode's `group_id` namespace, so the fact
            // MUST stamp `subject_group_id`/`object_group_id` with that group — otherwise they
            // default to NULL/"default" and the composite FK fails for any non-"default"
            // namespace, silently dropping every fact on the facade path (which always passes
            // `group_id = Some(namespace)`). `object_group_id` is NULL for literal objects
            // (object_id = None → object FK is not enforced).
            let object_group_id = object_id.and(group_id);
            self.conn
                .execute(
                    "INSERT INTO facts (subject_id, predicate, object_id, object_value, embedding, valid_from, recorded_at, confidence, source_episode_id, group_id, subject_group_id, object_group_id, content_hash)
                     VALUES (?1, ?2, ?3, ?4, CASE WHEN ?5 IS NULL THEN NULL ELSE vector(?5) END, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
                        group_id,
                        object_group_id,
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
    pub async fn try_insert_fact_with_group(
        &self,
        fact: FactInsert<'_>,
        group_id: Option<&str>,
    ) -> Result<Option<i64>> {
        let subject_id = fact.subject_id;
        let predicate = fact.predicate;
        match self.insert_fact_with_group(fact, group_id).await {
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
            // P3: self-loop counter already fired in insert_fact_with_group; swallow here
            // so a self-loop in a batch doesn't abort the surrounding fact writes.
            Err(crate::core::error::Error::SelfLoop { .. }) => {
                tracing::debug!(
                    subject_id,
                    predicate,
                    group_id,
                    "kremory.try_insert_fact_with_group.swallowed_self_loop"
                );
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Invalidate a fact with both system-level (expired_at) and domain-level (invalid_at) timestamps.
    pub async fn invalidate_fact_with_reason(
        &self,
        params: InvalidateFactWithReasonParams,
    ) -> Result<()> {
        let InvalidateFactWithReasonParams {
            fact_id,
            expired_at,
            invalid_at,
        } = params;
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
