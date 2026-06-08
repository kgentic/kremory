use chrono::{DateTime, Utc};
use metrics::histogram;
use std::time::Instant;
use tracing;

use crate::core::error::Result;
use crate::core::schema::{Entity, Fact, TemporalGraph};

/// A search hit with BM25 relevance score.
#[derive(Debug, Clone)]
pub struct SearchHit<T> {
    pub item: T,
    pub score: f64, // BM25 rank (negative; lower = more relevant)
}

/// Sanitise a raw text string for use as an FTS5 MATCH expression.
///
/// FTS5 treats characters like `'`, `"`, `(`, `)`, `,`, `*`, `+`, `-`, `:`
/// as special syntax. Raw natural language text (e.g. ASR output) will cause
/// parse errors if passed directly. This function splits the input into words,
/// strips non-alphanumeric characters, and wraps each token in double quotes
/// so FTS5 treats them as literal terms.
///
/// Returns `None` if the sanitised query is empty (no usable tokens).
fn sanitise_fts5_query(raw: &str) -> Option<String> {
    let tokens: Vec<String> = raw
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{}\"", w))
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

impl TemporalGraph {
    /// Increment `access_count` for a batch of entity IDs (Story #247).
    ///
    /// Called by all entity-returning search paths immediately after the
    /// result set is collected. Each entity in the result gets `access_count
    /// += 1` atomically via a single UPDATE statement. Empty `ids` is a
    /// no-op. Errors are swallowed with a warning — an access-count failure
    /// must never cause the search call to fail.
    pub(crate) async fn increment_entity_access_counts(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        // Build `UPDATE entities SET access_count = access_count + 1
        // WHERE id IN (?1, ?2, ...)`. Each id is bound positionally.
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "UPDATE entities SET access_count = access_count + 1 WHERE id IN ({})",
            placeholders.join(", ")
        );
        let params: Vec<libsql::Value> = ids
            .iter()
            .map(|id| libsql::Value::from(id.clone()))
            .collect();
        if let Err(e) = self.conn.execute(&sql, params).await {
            tracing::warn!(
                entity_count = ids.len(),
                error = %e,
                "kremory.search.access_count_update_failed"
            );
        }
    }

    /// Full-text search entities by label/properties.
    /// Returns entities ranked by BM25 relevance, scoped by `filters.group_ids`.
    pub async fn fts_search_entities(
        &self,
        query: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let hits = self
            .fts_search_entities_no_count(query, limit, filters)
            .await?;
        // Story #247: increment access_count for every returned entity.
        let returned_ids: Vec<String> = hits.iter().map(|h| h.item.id.clone()).collect();
        self.increment_entity_access_counts(&returned_ids).await;
        Ok(hits)
    }

    /// Like [`fts_search_entities`] but does NOT increment `access_count`.
    /// Used by `contextualize()` which manages access-count increments itself
    /// after RRF composition (RISK-002: avoid double-increment when both FTS
    /// and vector paths surface the same entity).
    pub(crate) async fn fts_search_entities_no_count(
        &self,
        query: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let _search_start = Instant::now();
        let safe_query = match sanitise_fts5_query(query) {
            Some(q) => q,
            None => {
                let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
                histogram!("rql.search.fts_entities_hits").record(0.0);
                histogram!("rql.search.fts_entities_ms").record(_ms);
                tracing::info!(hits = 0u64, _ms, "kremory.search.fts_entities empty query");
                return Ok(vec![]);
            }
        };

        // Build group_id filter — params start at ?3 (after ?1=query, ?2=limit)
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "e", 3);

        let sql = format!(
            "SELECT fts.entity_id, fts.rank \
             FROM entities_fts AS fts \
             JOIN entities AS e ON e.id = fts.entity_id \
             WHERE entities_fts MATCH ?1{} \
             ORDER BY fts.rank LIMIT ?2",
            group_clause
        );

        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(safe_query),
            libsql::Value::from(limit as i64),
        ];
        params.extend(group_params);

        let mut rows = self.conn.query(&sql, params).await?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let entity_id = row.get::<String>(0)?;
            let score = row.get::<f64>(1)?;
            if let Some(entity) = self.get_entity(&entity_id).await? {
                hits.push(SearchHit {
                    item: entity,
                    score,
                });
            }
        }
        let hits_count = hits.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.search.fts_entities_hits").record(hits_count as f64);
        histogram!("rql.search.fts_entities_ms").record(_ms);
        tracing::info!(hits = hits_count, _ms, "kremory.search.fts_entities");
        Ok(hits)
    }

    /// Full-text search facts by predicate/object_value.
    /// Returns facts ranked by BM25 relevance. Only searches non-expired facts.
    /// Scoped by `filters.group_ids`.
    pub async fn fts_search_facts(
        &self,
        query: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Fact>>> {
        let _search_start = Instant::now();
        let safe_query = match sanitise_fts5_query(query) {
            Some(q) => q,
            None => {
                let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
                histogram!("rql.search.fts_facts_hits").record(0.0);
                histogram!("rql.search.fts_facts_ms").record(_ms);
                tracing::info!(hits = 0u64, _ms, "kremory.search.fts_facts empty query");
                return Ok(vec![]);
            }
        };

        // Build group_id filter — params start at ?3 (after ?1=query, ?2=limit)
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "f", 3);

        let sql = format!(
            "SELECT f.id, f.subject_id, f.predicate, f.object_id, f.object_value, f.properties,
                    f.valid_from, f.valid_to, f.recorded_at, f.expired_at, f.invalid_at, f.group_id,
                    f.confidence, f.source_episode_id,
                    f.memory_type, f.content_hash, f.access_count,
                    fts.rank
             FROM facts_fts AS fts
             JOIN facts AS f ON CAST(fts.fact_id AS INTEGER) = f.id
             WHERE facts_fts MATCH ?1
               AND f.expired_at IS NULL{}
             ORDER BY fts.rank
             LIMIT ?2",
            group_clause
        );

        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(safe_query),
            libsql::Value::from(limit as i64),
        ];
        params.extend(group_params);

        let mut rows = self.conn.query(&sql, params).await?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let fact = row_to_fact_from_row(&row)?;
            let score = row.get::<f64>(17)?;
            hits.push(SearchHit { item: fact, score });
        }
        let hits_count = hits.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.search.fts_facts_hits").record(hits_count as f64);
        histogram!("rql.search.fts_facts_ms").record(_ms);
        tracing::info!(hits = hits_count, _ms, "kremory.search.fts_facts");
        Ok(hits)
    }

    /// Vector similarity search on entity embeddings using cosine distance.
    /// Returns entities ordered by similarity (closest first).
    /// Tries DiskANN index (vector_top_k) first, falls back to brute-force.
    /// Scoped by `filters.group_ids`.
    pub async fn vector_search_entities(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let hits = self
            .vector_search_entities_no_count(query_embedding, limit, filters)
            .await?;
        // Story #247: increment access_count for every returned entity.
        let returned_ids: Vec<String> = hits.iter().map(|h| h.item.id.clone()).collect();
        self.increment_entity_access_counts(&returned_ids).await;
        Ok(hits)
    }

    /// Like [`vector_search_entities`] but does NOT increment `access_count`.
    /// Used by `contextualize()` which manages access-count increments itself
    /// after RRF composition (RISK-002: avoid double-increment when both FTS
    /// and vector paths surface the same entity).
    pub(crate) async fn vector_search_entities_no_count(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let _search_start = Instant::now();
        let vec_str = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Try DiskANN index first (vector_top_k)
        let result = self
            .vector_search_with_index(&vec_str, limit, filters)
            .await;

        let hits = match result {
            Ok(hits) => hits,
            Err(_) => {
                // Fall back to brute-force cosine distance
                self.vector_search_brute_force(&vec_str, limit, filters)
                    .await?
            }
        };
        let hits_count = hits.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.search.vector_entities_hits").record(hits_count as f64);
        histogram!("rql.search.vector_entities_ms").record(_ms);
        tracing::info!(hits = hits_count, _ms, "kremory.search.vector_entities");
        Ok(hits)
    }

    async fn vector_search_with_index(
        &self,
        vec_str: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> anyhow::Result<Vec<SearchHit<Entity>>> {
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit)
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "e", 3);

        let sql = format!(
            "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.properties, \
                    e.recorded_at, e.updated_at, e.group_id, e.access_count, e.entity_type_id, \
                    vector_distance_cos(e.embedding, vector(?1)) as distance \
             FROM vector_top_k('entities_vec_idx', vector(?1), ?2) AS v \
             JOIN entities AS e ON e.rowid = v.id \
             LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id \
             WHERE 1=1{} \
             ORDER BY distance ASC",
            group_clause
        );

        let clamped = effective_k(limit, usize::MAX);
        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(vec_str.to_owned()),
            libsql::Value::from(clamped as i64),
        ];
        params.extend(group_params);

        let mut rows = self.conn.query(&sql, params).await?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let entity = row_to_entity_from_row(&row)?;
            // vector_distance_cos returns NULL when either vector has zero magnitude;
            // skip those rows rather than propagating a "Null value" error.
            // distance is at col 8 (cols 0-7 are entity fields + entity_type_id).
            let Some(distance) = row.get::<Option<f64>>(8)? else {
                continue;
            };
            // Convert cosine distance to a score (negative distance so lower = closer, matching FTS convention)
            hits.push(SearchHit {
                item: entity,
                score: -distance,
            });
        }
        Ok(hits)
    }

    async fn vector_search_brute_force(
        &self,
        vec_str: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> anyhow::Result<Vec<SearchHit<Entity>>> {
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit).
        // Table is aliased as "e" in the query, so use "e" here.
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "e", 3);

        let sql = format!(
            "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.properties, \
                    e.recorded_at, e.updated_at, e.group_id, e.access_count, e.entity_type_id, \
                    vector_distance_cos(e.embedding, vector(?1)) as distance \
             FROM entities e \
             LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id \
             WHERE e.embedding IS NOT NULL{} \
             ORDER BY distance ASC \
             LIMIT ?2",
            group_clause
        );

        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(vec_str.to_owned()),
            libsql::Value::from(limit as i64),
        ];
        params.extend(group_params);

        let mut rows = self.conn.query(&sql, params).await?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let entity = row_to_entity_from_row(&row)?;
            // vector_distance_cos returns NULL when either vector has zero magnitude;
            // skip those rows rather than propagating a "Null value" error.
            // distance is at col 8 (cols 0-7 are entity fields + entity_type_id).
            let Some(distance) = row.get::<Option<f64>>(8)? else {
                continue;
            };
            hits.push(SearchHit {
                item: entity,
                score: -distance,
            });
        }
        Ok(hits)
    }

    /// Hybrid search: combines vector similarity + FTS5 BM25 using Reciprocal Rank Fusion.
    /// `query_text` is used for FTS5, `query_embedding` is used for vector search.
    /// Returns entities ranked by combined RRF score (higher = more relevant).
    /// Scoped by `filters.group_ids`.
    pub async fn hybrid_search_entities(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let _search_start = Instant::now();
        // Fetch more candidates from each source than the final limit
        // to give fusion enough data to work with
        let fetch_limit = limit * 3;

        // Run both searches with the same filters
        let vector_hits = self
            .vector_search_entities(query_embedding, fetch_limit, filters)
            .await?;
        let fts_hits = self
            .fts_search_entities(query_text, fetch_limit, filters)
            .await?;

        // Fuse with RRF
        let fused = rrf_fuse_entities(vector_hits, fts_hits, 60.0);

        // Return top `limit` results
        let results: Vec<SearchHit<Entity>> = fused.into_iter().take(limit).collect();
        let result_count = results.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.search.hybrid_entities_hits").record(result_count as f64);
        histogram!("rql.search.hybrid_entities_ms").record(_ms);
        tracing::info!(hits = result_count, _ms, "kremory.search.hybrid_entities");
        Ok(results)
    }

    /// Vector similarity search on fact embeddings using cosine distance.
    /// Returns facts ordered by similarity (closest first). Only searches non-expired facts.
    /// Tries DiskANN index (vector_top_k) first, falls back to brute-force.
    /// Scoped by `filters.group_ids`.
    pub async fn vector_search_facts(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Fact>>> {
        let _search_start = Instant::now();
        let vec_str = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Try DiskANN index first (vector_top_k)
        let result = self
            .vector_search_facts_with_index(&vec_str, limit, filters)
            .await;

        let hits = match result {
            Ok(hits) => hits,
            Err(_) => {
                // Fall back to brute-force cosine distance
                self.vector_search_facts_brute_force(&vec_str, limit, filters)
                    .await?
            }
        };
        let hits_count = hits.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.search.vector_facts_hits").record(hits_count as f64);
        histogram!("rql.search.vector_facts_ms").record(_ms);
        tracing::info!(hits = hits_count, _ms, "kremory.search.vector_facts");
        Ok(hits)
    }

    async fn vector_search_facts_with_index(
        &self,
        vec_str: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> anyhow::Result<Vec<SearchHit<Fact>>> {
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit)
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "f", 3);

        let sql = format!(
            "SELECT f.id, f.subject_id, f.predicate, f.object_id, f.object_value, f.properties,
                    f.valid_from, f.valid_to, f.recorded_at, f.expired_at, f.invalid_at, f.group_id,
                    f.confidence, f.source_episode_id,
                    f.memory_type, f.content_hash, f.access_count,
                    vector_distance_cos(f.embedding, vector(?1)) as distance
             FROM vector_top_k('facts_vec_idx', vector(?1), ?2) AS v
             JOIN facts AS f ON f.rowid = v.id
             WHERE f.expired_at IS NULL{}
             ORDER BY distance ASC",
            group_clause
        );

        let clamped = effective_k(limit, usize::MAX);
        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(vec_str.to_owned()),
            libsql::Value::from(clamped as i64),
        ];
        params.extend(group_params);

        let mut rows = self.conn.query(&sql, params).await?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let fact = row_to_fact_from_row(&row)?;
            // vector_distance_cos returns NULL when either vector has zero magnitude;
            // skip those rows rather than propagating a "Null value" error.
            let Some(distance) = row.get::<Option<f64>>(17)? else {
                continue;
            };
            hits.push(SearchHit {
                item: fact,
                score: -distance,
            });
        }
        Ok(hits)
    }

    async fn vector_search_facts_brute_force(
        &self,
        vec_str: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> anyhow::Result<Vec<SearchHit<Fact>>> {
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit)
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "facts", 3);

        let sql = format!(
            "SELECT id, subject_id, predicate, object_id, object_value, properties,
                    valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id,
                    confidence, source_episode_id,
                    memory_type, content_hash, access_count,
                    vector_distance_cos(embedding, vector(?1)) as distance
             FROM facts
             WHERE embedding IS NOT NULL
               AND expired_at IS NULL{}
             ORDER BY distance ASC
             LIMIT ?2",
            group_clause
        );

        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(vec_str.to_owned()),
            libsql::Value::from(limit as i64),
        ];
        params.extend(group_params);

        let mut rows = self.conn.query(&sql, params).await?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let fact = row_to_fact_from_row(&row)?;
            // vector_distance_cos returns NULL when either vector has zero magnitude;
            // skip those rows rather than propagating a "Null value" error.
            let Some(distance) = row.get::<Option<f64>>(17)? else {
                continue;
            };
            hits.push(SearchHit {
                item: fact,
                score: -distance,
            });
        }
        Ok(hits)
    }

    /// Hybrid search: combines vector similarity + FTS5 BM25 for facts using Reciprocal Rank Fusion.
    /// `query_text` is used for FTS5, `query_embedding` is used for vector search.
    /// Returns facts ranked by combined RRF score (higher = more relevant).
    /// Scoped by `filters.group_ids`.
    pub async fn hybrid_search_facts(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Fact>>> {
        let _search_start = Instant::now();
        // Fetch more candidates from each source than the final limit
        // to give fusion enough data to work with
        let fetch_limit = limit * 3;

        // Run both searches with the same filters
        let vector_hits = self
            .vector_search_facts(query_embedding, fetch_limit, filters)
            .await?;
        let fts_hits = self
            .fts_search_facts(query_text, fetch_limit, filters)
            .await?;

        // Fuse with RRF
        let fused = rrf_fuse_facts(vector_hits, fts_hits, 60.0);

        // Return top `limit` results
        let results: Vec<SearchHit<Fact>> = fused.into_iter().take(limit).collect();
        let result_count = results.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.search.hybrid_facts_hits").record(result_count as f64);
        histogram!("rql.search.hybrid_facts_ms").record(_ms);
        tracing::info!(hits = result_count, _ms, "kremory.search.hybrid_facts");
        Ok(results)
    }
}

/// Filters for search queries — group scoping and temporal bounds.
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Restrict results to specific groups (tenant / session scopes).
    /// Empty vec means no group filtering (all groups returned).
    pub group_ids: Vec<String>,
    /// Only return facts valid after this timestamp (inclusive).
    pub valid_after: Option<DateTime<Utc>>,
    /// Only return facts valid before this timestamp (exclusive).
    pub valid_before: Option<DateTime<Utc>>,
    /// When true (the default), expired facts are excluded from results.
    pub exclude_expired: bool,
}

impl SearchFilters {
    /// Returns a new `SearchFilters` with `exclude_expired` set to `true` and no group filter.
    pub fn new() -> Self {
        Self {
            group_ids: vec![],
            exclude_expired: true,
            ..Default::default()
        }
    }

    /// Returns a new `SearchFilters` scoped to one group.
    pub fn for_group(group_id: impl Into<String>) -> Self {
        Self {
            group_ids: vec![group_id.into()],
            exclude_expired: true,
            ..Default::default()
        }
    }

    /// Returns a new `SearchFilters` scoped to multiple groups.
    pub fn for_groups(group_ids: Vec<String>) -> Self {
        Self {
            group_ids,
            exclude_expired: true,
            ..Default::default()
        }
    }
}

/// Build a SQL `AND <prefix>.group_id IN (?, ?, ...)` clause and matching param values.
/// Returns an empty string and empty vec when `group_ids` is empty (no filtering).
/// `param_offset` is the 1-based index of the first placeholder to use (e.g. 3 → `?3, ?4, ...`).
fn build_group_id_clause(
    group_ids: &[String],
    column_prefix: &str,
    param_offset: usize,
) -> (String, Vec<libsql::Value>) {
    if group_ids.is_empty() {
        return (String::new(), vec![]);
    }
    let placeholders: Vec<String> = (0..group_ids.len())
        .map(|i| format!("?{}", param_offset + i))
        .collect();
    let clause = format!(
        " AND ({}.group_id IN ({}) OR {}.group_id IS NULL)",
        column_prefix,
        placeholders.join(", "),
        column_prefix,
    );
    let params: Vec<libsql::Value> = group_ids
        .iter()
        .map(|id| libsql::Value::from(id.clone()))
        .collect();
    (clause, params)
}

/// Reciprocal Rank Fusion: merge two ranked entity lists into one.
/// k = 60 is the standard constant (controls how much rank position matters).
/// Higher RRF score means more relevant.
///
/// # Composite key (ADR-029c Decision 1)
///
/// The accumulator keys on `(entity_id, group_id)` rather than bare `entity_id`.
/// After ADR-029b's composite-PK migration, entities in different namespaces share
/// no rows, so same-name entities from namespace-A and namespace-B map to distinct
/// keys and are correctly preserved as separate results.
///
/// `group_id: None` = legacy unkeyed entities that predate ADR-029b. They share
/// the `(entity_id, None)` bucket — correct deduplication within that bucket.
fn rrf_fuse_entities(
    vector_hits: Vec<SearchHit<Entity>>,
    fts_hits: Vec<SearchHit<Entity>>,
    k: f64,
) -> Vec<SearchHit<Entity>> {
    use std::collections::HashMap;

    // Key: (entity_id, group_id) — composite, matching the post-029b PK shape.
    // group_id is Option<String>; None arm = legacy unkeyed entities.
    let mut scores: HashMap<(String, Option<String>), (f64, Entity)> = HashMap::new();

    // Score vector results by rank position
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        let key = (hit.item.id.clone(), hit.item.group_id.clone());
        scores
            .entry(key)
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, hit.item));
    }

    // Score FTS results by rank position
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        let key = (hit.item.id.clone(), hit.item.group_id.clone());
        scores
            .entry(key)
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, hit.item));
    }

    // Sort by RRF score descending (higher = more relevant)
    let mut results: Vec<SearchHit<Entity>> = scores
        .into_values()
        .map(|(score, entity)| SearchHit {
            item: entity,
            score,
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Reciprocal Rank Fusion: merge two ranked fact lists into one.
/// k = 60 is the standard constant (controls how much rank position matters).
/// Higher RRF score means more relevant.
///
/// # Composite key (ADR-029c Decision 1)
///
/// The accumulator keys on `(fact_id, group_id)`. Fact ids are
/// `INTEGER PRIMARY KEY AUTOINCREMENT` and are globally unique by construction,
/// so the `group_id` component is redundant for deduplication today. Keying
/// consistently on `(id, group_id)` makes the multi-namespace fusion
/// correct-by-construction for any future change to fact id semantics.
///
/// `group_id: None` = legacy unkeyed facts. Same None-bucket semantics as
/// `rrf_fuse_entities`.
fn rrf_fuse_facts(
    vector_hits: Vec<SearchHit<Fact>>,
    fts_hits: Vec<SearchHit<Fact>>,
    k: f64,
) -> Vec<SearchHit<Fact>> {
    use std::collections::HashMap;

    // Key: (fact_id, group_id) — globally-unique INTEGER PK, but keyed
    // consistently for multi-namespace correctness.
    let mut scores: HashMap<(i64, Option<String>), (f64, Fact)> = HashMap::new();

    // Score vector results by rank position
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        let key = (hit.item.id, hit.item.group_id.clone());
        scores
            .entry(key)
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, hit.item));
    }

    // Score FTS results by rank position
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        let key = (hit.item.id, hit.item.group_id.clone());
        scores
            .entry(key)
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, hit.item));
    }

    // Sort by RRF score descending (higher = more relevant)
    let mut results: Vec<SearchHit<Fact>> = scores
        .into_values()
        .map(|(score, fact)| SearchHit { item: fact, score })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ── test-utils re-exports (ADR-029c 5-tier pyramid, Phase A) ─────────────────
//
// Property tests in `tests/properties_029bc.rs` call these private functions
// directly so they can assert composite-key dedup invariants without going
// through the full SQL round-trip. Gated behind `test-utils` feature + `test`
// cfg so they never appear in production builds.
#[cfg(any(test, feature = "test-utils"))]
pub fn rrf_fuse_entities_for_test(
    vector_hits: Vec<SearchHit<Entity>>,
    fts_hits: Vec<SearchHit<Entity>>,
    k: f64,
) -> Vec<SearchHit<Entity>> {
    rrf_fuse_entities(vector_hits, fts_hits, k)
}

#[cfg(any(test, feature = "test-utils"))]
pub fn rrf_fuse_facts_for_test(
    vector_hits: Vec<SearchHit<Fact>>,
    fts_hits: Vec<SearchHit<Fact>>,
    k: f64,
) -> Vec<SearchHit<Fact>> {
    rrf_fuse_facts(vector_hits, fts_hits, k)
}

/// Clamp a requested top-K to the actual number of available results.
///
/// Story #166 / turbovec prior-art pattern: prevents `vector_top_k` from
/// requesting more results than the index contains, which causes a runtime
/// panic on sparse allowlist queries.
///
/// `k` — caller-requested limit. `n_available` — how many candidates are
/// available (e.g. size of an allowlist or total indexed vectors). The
/// effective K is `k.min(n_available)` clamped to at least 1.
///
/// Callers that do not know `n_available` at call time pass `usize::MAX`
/// and only the lower bound (≥ 1) is enforced.
pub(crate) fn effective_k(k: usize, n_available: usize) -> usize {
    k.min(n_available).max(1)
}

/// Helper to extract an Entity from a query row.
/// Expected columns (canonical entity SELECT with LEFT JOIN entity_types):
///   id(0), label(1) [COALESCE(et.name,'Entity')], properties(2), recorded_at(3),
///   updated_at(4), group_id(5), access_count(6), entity_type_id(7).
fn row_to_entity_from_row(row: &libsql::Row) -> anyhow::Result<Entity> {
    use chrono::DateTime;
    let id = row.get::<String>(0)?;
    let label = row.get::<String>(1)?;
    let props_str = row.get::<Option<String>>(2)?;
    let created_str = row.get::<String>(3)?;
    let updated_str = row.get::<Option<String>>(4)?;
    let group_id = row.get::<Option<String>>(5)?;
    let access_count = row.get::<i64>(6)?;
    let entity_type_id_raw: i64 = row.get::<i64>(7)?;
    let entity_type_id: u32 = entity_type_id_raw.max(0) as u32;

    let parse_dt = |s: &str| -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
        Ok(DateTime::parse_from_rfc3339(s)
            .map_err(|e| anyhow::anyhow!("bad timestamp '{}': {}", s, e))?
            .with_timezone(&chrono::Utc))
    };

    let properties = props_str
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

/// Helper to extract a Fact from a query row (same column order as fts_search_facts query).
/// Columns: id, subject_id, predicate, object_id, object_value, properties,
///          valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence,
///          source_episode_id, memory_type, content_hash, access_count
fn row_to_fact_from_row(row: &libsql::Row) -> anyhow::Result<Fact> {
    use chrono::DateTime;

    let id = row.get::<i64>(0)?;
    let subject_id = row.get::<String>(1)?;
    let predicate = row.get::<String>(2)?;
    let object_id = row.get::<Option<String>>(3)?;
    let object_value = row.get::<Option<String>>(4)?;
    let props_str = row.get::<Option<String>>(5)?;
    let valid_from_str = row.get::<String>(6)?;
    let valid_to_str = row.get::<Option<String>>(7)?;
    let recorded_str = row.get::<String>(8)?;
    let expired_str = row.get::<Option<String>>(9)?;
    let invalid_str = row.get::<Option<String>>(10)?;
    let group_id = row.get::<Option<String>>(11)?;
    let confidence = row.get::<f64>(12)?;
    let source_episode_id = row.get::<Option<i64>>(13)?;
    let memory_type_str = row.get::<Option<String>>(14)?;
    let content_hash = row.get::<Option<String>>(15)?;
    let access_count = row.get::<i64>(16)?;

    let parse_dt = |s: &str| -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
        Ok(DateTime::parse_from_rfc3339(s)
            .map_err(|e| anyhow::anyhow!("bad timestamp '{}': {}", s, e))?
            .with_timezone(&chrono::Utc))
    };

    let properties = props_str
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
        // ADR-029b: composite FK fields — absent on pre-migration-004 rows.
        subject_group_id: None,
        object_group_id: None,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use chrono::{Duration, Utc};

    // === effective_k clamp (Story #166) ===

    #[test]
    fn effective_k_clamps_to_n_available() {
        // AC: k=100, n_allowed=3 → 3 results
        assert_eq!(effective_k(100, 3), 3);
    }

    #[test]
    fn effective_k_min_one_when_zero_available() {
        // Even with n_available=0, effective_k returns at least 1 (avoids LIMIT 0).
        assert_eq!(effective_k(10, 0), 1);
    }

    #[test]
    fn effective_k_no_clamp_when_k_lt_n_available() {
        assert_eq!(effective_k(5, 100), 5);
    }

    #[test]
    fn effective_k_passthrough_when_no_allowlist() {
        // Caller passes usize::MAX when n_available is unknown.
        assert_eq!(effective_k(10, usize::MAX), 10);
    }

    #[test]
    fn test_search_filters_default() {
        let f = SearchFilters::new();
        assert!(f.exclude_expired, "exclude_expired should default to true");
        assert!(f.group_ids.is_empty());
        assert!(f.valid_after.is_none());
        assert!(f.valid_before.is_none());
    }

    #[test]
    fn test_search_filters_for_group() {
        let f = SearchFilters::for_group("tenant-1");
        assert_eq!(f.group_ids, vec!["tenant-1"]);
        assert!(f.exclude_expired);
    }

    #[test]
    fn test_search_filters_for_groups() {
        let f = SearchFilters::for_groups(vec!["a".into(), "b".into()]);
        assert_eq!(f.group_ids, vec!["a", "b"]);
    }

    async fn setup_graph_with_data() -> TemporalGraph {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let t0 = Utc::now() - Duration::hours(1);

        g.insert_entity(
            "alice",
            0,
            serde_json::json!({"role": "engineer", "department": "platform"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "bob",
            0,
            serde_json::json!({"role": "manager", "department": "sales"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "acme",
            0,
            serde_json::json!({"industry": "technology", "size": "startup"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "budget_2025",
            0,
            serde_json::json!({"title": "Q1 Budget Review"}),
        )
        .await
        .unwrap();

        g.insert_fact("alice", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact("bob", "works_at", Some("acme"), None, t0, 1.0, None, None)
            .await
            .unwrap();
        g.insert_fact(
            "alice",
            "has_title",
            None,
            Some("Senior Engineer"),
            t0,
            1.0,
            None,
            None,
        )
        .await
        .unwrap();
        g.insert_fact(
            "bob",
            "has_title",
            None,
            Some("Sales Manager"),
            t0,
            1.0,
            None,
            None,
        )
        .await
        .unwrap();
        g.insert_fact(
            "alice",
            "discussed",
            None,
            Some("budget allocation for Q1"),
            t0,
            0.9,
            None,
            None,
        )
        .await
        .unwrap();

        g
    }

    #[tokio::test]
    async fn test_fts_search_entities_by_department() {
        // After Phase 2 (Migration 009), entities_fts.label is empty (entities.label
        // column was dropped). FTS searches on entity properties only.
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_entities("platform", 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "should find alice by 'platform' in properties"
        );
        assert_eq!(hits[0].item.id, "alice");
        // BM25 scores should be negative
        for hit in &hits {
            assert!(hit.score < 0.0, "BM25 rank should be negative");
        }
    }

    #[tokio::test]
    async fn test_fts_search_entities_by_properties() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_entities("engineer", 10, &no_filter)
            .await
            .unwrap();
        assert!(
            !hits.is_empty(),
            "should find entities with 'engineer' in properties"
        );
        assert_eq!(hits[0].item.id, "alice");
    }

    #[tokio::test]
    async fn test_fts_search_entities_no_results() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_entities("nonexistent_term_xyz", 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[tokio::test]
    async fn test_fts_search_entities_limit() {
        // setup_graph_with_data inserts alice (role/department), bob (role/sales),
        // acme (industry/technology), budget_2025 (title).
        // "role" appears in alice + bob properties — 2 potential matches.
        // Label is no longer in FTS after Phase 2 (entities.label column dropped).
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g.fts_search_entities("role", 1, &no_filter).await.unwrap();
        assert_eq!(hits.len(), 1, "limit should cap results");
    }

    // ── FTS5 sanitisation tests ─────────────────────────────────────────────

    #[test]
    fn test_sanitise_fts5_simple_word() {
        assert_eq!(sanitise_fts5_query("hello"), Some("\"hello\"".to_string()));
    }

    #[test]
    fn test_sanitise_fts5_multiple_words() {
        assert_eq!(
            sanitise_fts5_query("hello world"),
            Some("\"hello\" \"world\"".to_string())
        );
    }

    #[test]
    fn test_sanitise_fts5_strips_apostrophes_and_commas() {
        // "It's an honor, to be here" should not crash FTS5
        assert_eq!(
            sanitise_fts5_query("It's an honor, to be here"),
            Some("\"Its\" \"an\" \"honor\" \"to\" \"be\" \"here\"".to_string())
        );
    }

    #[test]
    fn test_sanitise_fts5_preserves_underscores() {
        assert_eq!(
            sanitise_fts5_query("has_title"),
            Some("\"has_title\"".to_string())
        );
    }

    #[test]
    fn test_sanitise_fts5_empty_after_stripping() {
        assert_eq!(sanitise_fts5_query("'',, ..."), None);
    }

    #[test]
    fn test_sanitise_fts5_empty_input() {
        assert_eq!(sanitise_fts5_query(""), None);
    }

    // ── FTS5 with natural language text (regression for the apostrophe crash) ──

    #[tokio::test]
    async fn test_fts_search_entities_with_punctuated_query() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        // This query contains apostrophes and commas — previously crashed FTS5
        let hits = g
            .fts_search_entities("Alice's work at Acme, Inc.", 10, &no_filter)
            .await
            .unwrap();
        // Should not crash; may or may not find results depending on data
        assert!(hits.len() <= 10);
    }

    #[tokio::test]
    async fn test_fts_search_facts_with_punctuated_query() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_facts("It's a test, isn't it?", 10, &no_filter)
            .await
            .unwrap();
        assert!(hits.len() <= 10);
    }

    #[tokio::test]
    async fn test_fts_search_facts_by_object_value() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_facts("Engineer", 10, &no_filter)
            .await
            .unwrap();
        assert!(
            !hits.is_empty(),
            "should find facts with 'Engineer' in object_value"
        );
        assert_eq!(hits[0].item.predicate, "has_title");
    }

    #[tokio::test]
    async fn test_fts_search_facts_by_predicate() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_facts("has_title", 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2, "should find 2 has_title facts");
    }

    #[tokio::test]
    async fn test_fts_search_facts_excludes_expired() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        // Get the "discussed budget" fact and expire it
        let facts = g.facts_at(Utc::now()).await.unwrap();
        let budget_fact = facts.iter().find(|f| f.predicate == "discussed").unwrap();
        g.invalidate_fact(budget_fact.id, Utc::now()).await.unwrap();

        let hits = g.fts_search_facts("budget", 10, &no_filter).await.unwrap();
        assert_eq!(hits.len(), 0, "expired facts should not appear in search");
    }

    #[tokio::test]
    async fn test_fts_search_facts_ranking() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g.fts_search_facts("budget", 10, &no_filter).await.unwrap();
        assert!(!hits.is_empty());
        // Results should be ordered by rank (ascending, since BM25 rank is negative)
        for i in 1..hits.len() {
            assert!(
                hits[i].score >= hits[i - 1].score,
                "results should be ranked by BM25"
            );
        }
    }

    // Helper to create a simple 384-dim embedding with a pattern
    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..384)
            .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
            .collect()
    }

    #[tokio::test]
    async fn test_set_and_search_entity_embedding() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("alice", 0, serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({"role": "manager"}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({"industry": "tech"}))
            .await
            .unwrap();

        // Set embeddings — alice and acme get similar embeddings, bob gets different
        let alice_emb = make_embedding(1.0);
        let bob_emb = make_embedding(5.0); // very different from alice
        let acme_emb = make_embedding(1.01); // very similar to alice

        g.set_entity_embedding("alice", &alice_emb).await.unwrap();
        g.set_entity_embedding("bob", &bob_emb).await.unwrap();
        g.set_entity_embedding("acme", &acme_emb).await.unwrap();

        // Search with alice's embedding — should find alice first, acme second (similar), bob last
        let hits = g
            .vector_search_entities(&alice_emb, 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 3, "should find all 3 entities with embeddings");
        assert_eq!(
            hits[0].item.id, "alice",
            "alice should be closest to her own embedding"
        );
    }

    #[tokio::test]
    async fn test_vector_search_limit() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("a", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("b", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("c", 0, serde_json::json!({}))
            .await
            .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("a", &emb).await.unwrap();
        g.set_entity_embedding("b", &make_embedding(2.0))
            .await
            .unwrap();
        g.set_entity_embedding("c", &make_embedding(3.0))
            .await
            .unwrap();

        let hits = g.vector_search_entities(&emb, 2, &no_filter).await.unwrap();
        assert_eq!(hits.len(), 2, "limit should cap results to 2");
    }

    #[tokio::test]
    async fn test_vector_search_empty_returns_empty() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();
        // No entities, no embeddings
        let emb = make_embedding(1.0);
        let hits = g
            .vector_search_entities(&emb, 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[tokio::test]
    async fn test_vector_search_skips_null_embeddings() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("with_emb", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("no_emb", 0, serde_json::json!({}))
            .await
            .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("with_emb", &emb).await.unwrap();
        // no_emb has no embedding set

        let hits = g
            .vector_search_entities(&emb, 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "should only find entities with embeddings");
        assert_eq!(hits[0].item.id, "with_emb");
    }

    #[tokio::test]
    async fn test_vector_search_cosine_similarity_ordering() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("close", 0, serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("far", 0, serde_json::json!({}))
            .await
            .unwrap();

        let query = make_embedding(1.0);
        let close_emb = make_embedding(1.05); // very similar
        let far_emb = make_embedding(10.0); // very different

        g.set_entity_embedding("close", &close_emb).await.unwrap();
        g.set_entity_embedding("far", &far_emb).await.unwrap();

        let hits = g
            .vector_search_entities(&query, 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].item.id, "close",
            "closer embedding should rank first"
        );
        assert_eq!(hits[1].item.id, "far");
    }

    #[tokio::test]
    async fn test_hybrid_search_combines_vector_and_fts() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        // Create entities with both text and embeddings
        g.insert_entity("alice", 0, serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.insert_entity("bob", 0, serde_json::json!({"role": "manager"}))
            .await
            .unwrap();
        g.insert_entity("acme", 0, serde_json::json!({"industry": "technology"}))
            .await
            .unwrap();

        let alice_emb = make_embedding(1.0);
        let bob_emb = make_embedding(5.0);
        let acme_emb = make_embedding(1.01); // similar to alice

        g.set_entity_embedding("alice", &alice_emb).await.unwrap();
        g.set_entity_embedding("bob", &bob_emb).await.unwrap();
        g.set_entity_embedding("acme", &acme_emb).await.unwrap();

        // Search for "engineer" with alice's embedding
        // Alice should rank highest: matches both FTS ("engineer" in properties) AND vector (own embedding)
        let hits = g
            .hybrid_search_entities("engineer", &alice_emb, 10, &no_filter)
            .await
            .unwrap();
        assert!(!hits.is_empty(), "hybrid search should return results");
        assert_eq!(
            hits[0].item.id, "alice",
            "alice should rank #1 (appears in both FTS and vector results)"
        );
    }

    #[tokio::test]
    async fn test_hybrid_search_rrf_boosts_dual_matches() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("both_match", 0, serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.insert_entity("fts_only", 0, serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.insert_entity("vec_only", 0, serde_json::json!({"industry": "finance"}))
            .await
            .unwrap();

        let query_emb = make_embedding(1.0);
        g.set_entity_embedding("both_match", &make_embedding(1.01))
            .await
            .unwrap(); // similar to query
                       // fts_only has no embedding — will only appear in FTS results
        g.set_entity_embedding("vec_only", &make_embedding(1.02))
            .await
            .unwrap(); // similar to query but won't match "engineer" FTS

        let hits = g
            .hybrid_search_entities("engineer", &query_emb, 10, &no_filter)
            .await
            .unwrap();

        // both_match should have highest RRF score (appears in both lists)
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].item.id, "both_match",
            "entity matching both FTS and vector should rank highest"
        );
    }

    #[tokio::test]
    async fn test_hybrid_search_limit() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("a", 0, serde_json::json!({"x": "y"}))
            .await
            .unwrap();
        g.insert_entity("b", 0, serde_json::json!({"x": "y"}))
            .await
            .unwrap();
        g.insert_entity("c", 0, serde_json::json!({"x": "y"}))
            .await
            .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("a", &emb).await.unwrap();
        g.set_entity_embedding("b", &make_embedding(2.0))
            .await
            .unwrap();
        g.set_entity_embedding("c", &make_embedding(3.0))
            .await
            .unwrap();

        let hits = g
            .hybrid_search_entities("Person", &emb, 2, &no_filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2, "limit should cap hybrid results");
    }

    #[tokio::test]
    async fn test_hybrid_search_rrf_scores_are_positive() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        // Set some embeddings on the existing test data
        g.set_entity_embedding("alice", &make_embedding(1.0))
            .await
            .unwrap();
        g.set_entity_embedding("bob", &make_embedding(2.0))
            .await
            .unwrap();

        let hits = g
            .hybrid_search_entities("Person", &make_embedding(1.0), 10, &no_filter)
            .await
            .unwrap();
        for hit in &hits {
            assert!(
                hit.score > 0.0,
                "RRF scores should be positive (higher = better)"
            );
        }
    }

    // ── group_id filtering tests ────────────────────────────────────────────

    /// ADR-029b: group_id is NOT NULL post-migration-004. None → 'default'.
    /// Scoped searches return only entities in the requested namespace.
    #[tokio::test]
    async fn test_fts_search_entities_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entities in different groups.
        // All entities share "employee" in their properties so FTS matches all 4;
        // label is no longer in FTS (Phase 2 dropped entities.label column —
        // entity types are resolved via JOIN on entity_types at query time).
        g.insert_entity_with_group(
            "alice",
            0,
            serde_json::json!({"category": "employee", "role": "engineer"}),
            Some("group-a"),
        )
        .await
        .unwrap();
        g.insert_entity_with_group(
            "bob",
            0,
            serde_json::json!({"category": "employee", "role": "manager"}),
            Some("group-b"),
        )
        .await
        .unwrap();
        g.insert_entity_with_group(
            "carol",
            0,
            serde_json::json!({"category": "employee", "role": "designer"}),
            Some("group-a"),
        )
        .await
        .unwrap();
        // dave: None → 'default' post-ADR-029b (no longer NULL = workspace-wide).
        g.insert_entity_with_group(
            "dave",
            0,
            serde_json::json!({"category": "employee", "role": "analyst"}),
            None,
        )
        .await
        .unwrap();

        // No filter: all 4 employee entities (search on properties term, not label).
        let no_filter = SearchFilters::new();
        let all = g
            .fts_search_entities("employee", 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(all.len(), 4, "no filter should return all entities");

        // Filter to group-a: alice + carol only (dave is in 'default', not 'group-a').
        let group_a = SearchFilters::for_group("group-a");
        let hits_a = g
            .fts_search_entities("employee", 10, &group_a)
            .await
            .unwrap();
        assert_eq!(
            hits_a.len(),
            2,
            "group-a should have exactly 2 scoped entities (alice + carol)"
        );
        let ids: Vec<&str> = hits_a.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids.contains(&"alice"));
        assert!(ids.contains(&"carol"));
        assert!(
            !ids.contains(&"dave"),
            "'default' namespace entity must NOT appear in group-a search"
        );

        // Filter to group-b: bob only.
        let group_b = SearchFilters::for_group("group-b");
        let hits_b = g
            .fts_search_entities("employee", 10, &group_b)
            .await
            .unwrap();
        assert_eq!(
            hits_b.len(),
            1,
            "group-b should have exactly 1 scoped entity (bob)"
        );
        let ids_b: Vec<&str> = hits_b.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids_b.contains(&"bob"));

        // Filter to 'default': dave only.
        let group_default = SearchFilters::for_group("default");
        let hits_default = g
            .fts_search_entities("employee", 10, &group_default)
            .await
            .unwrap();
        assert_eq!(hits_default.len(), 1, "'default' should return only dave");
        assert_eq!(hits_default[0].item.id, "dave");

        // Filter to non-existent group: empty (no workspace-wide entities post-ADR-029b).
        let group_x = SearchFilters::for_group("group-x");
        let hits_x = g
            .fts_search_entities("employee", 10, &group_x)
            .await
            .unwrap();
        assert_eq!(
            hits_x.len(),
            0,
            "non-existent group should return 0 entities post-ADR-029b"
        );
    }

    #[tokio::test]
    async fn test_fts_search_entities_filters_by_multiple_groups() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // "member" is a common term in all properties — label no longer in FTS after Phase 2.
        g.insert_entity_with_group(
            "alice",
            0,
            serde_json::json!({"kind": "member"}),
            Some("g1"),
        )
        .await
        .unwrap();
        g.insert_entity_with_group("bob", 0, serde_json::json!({"kind": "member"}), Some("g2"))
            .await
            .unwrap();
        g.insert_entity_with_group(
            "carol",
            0,
            serde_json::json!({"kind": "member"}),
            Some("g3"),
        )
        .await
        .unwrap();

        let filters = SearchFilters::for_groups(vec!["g1".into(), "g3".into()]);
        let hits = g.fts_search_entities("member", 10, &filters).await.unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids.contains(&"alice"));
        assert!(ids.contains(&"carol"));
    }

    #[tokio::test]
    async fn test_vector_search_entities_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group("alice", 0, serde_json::json!({}), Some("group-a"))
            .await
            .unwrap();
        g.insert_entity_with_group("bob", 0, serde_json::json!({}), Some("group-b"))
            .await
            .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("alice", &emb).await.unwrap();
        g.set_entity_embedding("bob", &make_embedding(1.01))
            .await
            .unwrap();

        // No filter: both
        let no_filter = SearchFilters::new();
        let all = g
            .vector_search_entities(&emb, 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        // Filter to group-a: alice only
        let group_a = SearchFilters::for_group("group-a");
        let hits = g.vector_search_entities(&emb, 10, &group_a).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item.id, "alice");
        assert_eq!(hits[0].item.group_id.as_deref(), Some("group-a"));
    }

    #[tokio::test]
    async fn test_fts_search_facts_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let t0 = Utc::now() - Duration::hours(1);

        g.insert_entity("alice", 0, serde_json::json!({}))
            .await
            .unwrap();

        g.insert_fact_with_group(
            "alice",
            "has_title",
            None,
            Some("Engineer"),
            t0,
            1.0,
            None,
            Some("group-a"),
            None,
        )
        .await
        .unwrap();
        g.insert_fact_with_group(
            "alice",
            "has_title",
            None,
            Some("Manager"),
            t0,
            1.0,
            None,
            Some("group-b"),
            None,
        )
        .await
        .unwrap();

        // No filter: both facts
        let no_filter = SearchFilters::new();
        let all = g
            .fts_search_facts("has_title", 10, &no_filter)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        // Filter to group-a: 1 fact
        let group_a = SearchFilters::for_group("group-a");
        let hits = g.fts_search_facts("has_title", 10, &group_a).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item.object_value.as_deref(), Some("Engineer"));
    }

    /// Regression test: entities indexed with wrong group_id then re-scoped
    /// must be findable under the new group_id (not the old one).
    #[tokio::test]
    async fn test_rescoped_entity_searchable_under_new_group() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Simulate file indexed at workspace level (group_id = "default")
        g.insert_entity_with_group(
            "pricing_chunk_0",
            0,
            serde_json::json!({"text": "Full Build price £3,000 + VAT"}),
            Some("default"),
        )
        .await
        .unwrap();

        // Chat queries with space scope — should NOT find it
        let space_filter = SearchFilters::for_group("space-abc");
        let hits = g
            .fts_search_entities("price", 10, &space_filter)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            0,
            "entity in 'default' group should not appear in space-abc"
        );

        // Re-scope entity to space-abc (simulates update_entity_group fix)
        g.update_entity_group(
            "pricing_chunk_0",
            Some("space-abc"),
            serde_json::json!({"text": "Full Build price £3,000 + VAT"}),
        )
        .await
        .unwrap();

        // Now chat queries with space scope — SHOULD find it
        let hits = g
            .fts_search_entities("price", 10, &space_filter)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "re-scoped entity should be findable under new group"
        );
        assert_eq!(hits[0].item.id, "pricing_chunk_0");
        assert_eq!(hits[0].item.group_id.as_deref(), Some("space-abc"));

        // Old group should NOT find it
        let old_filter = SearchFilters::for_group("default");
        let old_hits = g
            .fts_search_entities("price", 10, &old_filter)
            .await
            .unwrap();
        assert_eq!(
            old_hits.len(),
            0,
            "entity should no longer appear under old group"
        );
    }

    /// ADR-029b: Post-migration, entities.group_id is NOT NULL. The 'default' namespace
    /// is the workspace-wide namespace (equivalent to pre-ADR-029b NULL group_id).
    /// Scoped searches return only entities in the requested namespace;
    /// 'default' namespace entities are only visible in searches scoped to 'default'.
    #[tokio::test]
    async fn test_null_group_id_included_when_scoped() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entity WITHOUT group_id — maps to 'default' post-ADR-029b.
        g.insert_entity(
            "kb_doc",
            0,
            serde_json::json!({ "text": "revenue targets" }),
        )
        .await
        .unwrap();

        // Insert entity WITH group_id — scoped to space-1.
        g.insert_entity_with_group(
            "scoped_doc",
            0,
            serde_json::json!({ "text": "revenue analysis" }),
            Some("space-1"),
        )
        .await
        .unwrap();

        // Unscoped search → both entities found.
        let all = g
            .fts_search_entities("revenue", 10, &SearchFilters::new())
            .await
            .unwrap();
        assert_eq!(all.len(), 2, "unscoped should return both entities");

        // Scoped search for 'default' → only kb_doc (it lives in 'default').
        let default_scoped = g
            .fts_search_entities("revenue", 10, &SearchFilters::for_group("default"))
            .await
            .unwrap();
        assert_eq!(
            default_scoped.len(),
            1,
            "scoped to 'default' must return only the default-namespace entity"
        );
        let default_ids: Vec<&str> = default_scoped.iter().map(|h| h.item.id.as_str()).collect();
        assert!(
            default_ids.contains(&"kb_doc"),
            "'default' scoped search must include kb_doc"
        );

        // Scoped search for 'space-1' → only scoped_doc.
        let space_scoped = g
            .fts_search_entities("revenue", 10, &SearchFilters::for_group("space-1"))
            .await
            .unwrap();
        assert_eq!(
            space_scoped.len(),
            1,
            "scoped to 'space-1' must return only the space-1 entity"
        );
        let space_ids: Vec<&str> = space_scoped.iter().map(|h| h.item.id.as_str()).collect();
        assert!(
            space_ids.contains(&"scoped_doc"),
            "space-1 scoped search must include scoped_doc"
        );
    }

    /// Story #247 gate: access_count increments on every entity-returning search.
    ///
    /// AC: access_count starts at 0; after first search that returns the entity it
    /// must be 1; after second search it must be 2. Deterministic in-memory DB.
    #[tokio::test]
    async fn access_count_increments_on_search() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entity with a unique label so FTS will return it deterministically.
        g.insert_entity(
            "ac_entity_1",
            0,
            serde_json::json!({ "text": "kremory_ac_probe_term_unique" }),
        )
        .await
        .unwrap();

        // Baseline: access_count must be 0 before any search.
        let initial = g.get_entity("ac_entity_1").await.unwrap().unwrap();
        assert_eq!(
            initial.access_count, 0,
            "access_count must start at 0 before any search"
        );

        // First search — must return the entity and increment access_count to 1.
        let hits = g
            .fts_search_entities("kremory_ac_probe_term_unique", 10, &SearchFilters::new())
            .await
            .unwrap();
        assert!(!hits.is_empty(), "first search must return the entity");
        let after_first = g.get_entity("ac_entity_1").await.unwrap().unwrap();
        assert_eq!(
            after_first.access_count, 1,
            "access_count must be 1 after first search"
        );

        // Second search — must increment to 2.
        let hits2 = g
            .fts_search_entities("kremory_ac_probe_term_unique", 10, &SearchFilters::new())
            .await
            .unwrap();
        assert!(!hits2.is_empty(), "second search must return the entity");
        let after_second = g.get_entity("ac_entity_1").await.unwrap().unwrap();
        assert_eq!(
            after_second.access_count, 2,
            "access_count must be 2 after second search"
        );
    }

    /// Regression guard: empty group_ids = no filter (returns all entities).
    #[tokio::test]
    async fn test_empty_group_ids_returns_all() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group(
            "doc_a",
            0,
            serde_json::json!({ "text": "alpha project" }),
            Some("group-1"),
        )
        .await
        .unwrap();
        g.insert_entity_with_group(
            "doc_b",
            0,
            serde_json::json!({ "text": "alpha budget" }),
            Some("group-2"),
        )
        .await
        .unwrap();

        // Empty group_ids → no filter → both returned.
        let results = g
            .fts_search_entities("alpha", 10, &SearchFilters::new())
            .await
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "empty group_ids should return all entities"
        );
    }
}
