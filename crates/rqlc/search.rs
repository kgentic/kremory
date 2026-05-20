use chrono::{DateTime, Utc};
use metrics::histogram;
use std::time::Instant;

use crate::error::Result;
use crate::schema::{Entity, Fact, TemporalGraph};

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
    /// Full-text search entities by label/properties.
    /// Returns entities ranked by BM25 relevance, scoped by `filters.group_ids`.
    pub async fn fts_search_entities(
        &self,
        query: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let _search_start = Instant::now();
        let safe_query = match sanitise_fts5_query(query) {
            Some(q) => q,
            None => {
                histogram!("rql.search.fts_entities_hits").record(0.0);
                histogram!("rql.search.fts_entities_ms")
                    .record(_search_start.elapsed().as_secs_f64() * 1000.0);
                return Ok(vec![]);
            }
        };

        // Build group_id filter — params start at ?3 (after ?1=query, ?2=limit)
        let (group_clause, group_params) =
            build_group_id_clause(&filters.group_ids, "e", 3);

        let sql = format!(
            "SELECT fts.entity_id, fts.rank \
             FROM rql_entities_fts AS fts \
             JOIN rql_entities AS e ON e.id = fts.entity_id \
             WHERE rql_entities_fts MATCH ?1{} \
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
        histogram!("rql.search.fts_entities_hits").record(hits.len() as f64);
        histogram!("rql.search.fts_entities_ms")
            .record(_search_start.elapsed().as_secs_f64() * 1000.0);
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
                histogram!("rql.search.fts_facts_hits").record(0.0);
                histogram!("rql.search.fts_facts_ms")
                    .record(_search_start.elapsed().as_secs_f64() * 1000.0);
                return Ok(vec![]);
            }
        };

        // Build group_id filter — params start at ?3 (after ?1=query, ?2=limit)
        let (group_clause, group_params) =
            build_group_id_clause(&filters.group_ids, "f", 3);

        let sql = format!(
            "SELECT f.id, f.subject_id, f.predicate, f.object_id, f.object_value, f.properties,
                    f.valid_from, f.valid_to, f.created_at, f.expired_at, f.invalid_at, f.group_id,
                    f.confidence, f.source_episode_id,
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
            let score = row.get::<f64>(14)?;
            hits.push(SearchHit { item: fact, score });
        }
        histogram!("rql.search.fts_facts_hits").record(hits.len() as f64);
        histogram!("rql.search.fts_facts_ms")
            .record(_search_start.elapsed().as_secs_f64() * 1000.0);
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
        histogram!("rql.search.vector_entities_hits").record(hits.len() as f64);
        histogram!("rql.search.vector_entities_ms")
            .record(_search_start.elapsed().as_secs_f64() * 1000.0);
        Ok(hits)
    }

    async fn vector_search_with_index(
        &self,
        vec_str: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> anyhow::Result<Vec<SearchHit<Entity>>> {
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit)
        let (group_clause, group_params) =
            build_group_id_clause(&filters.group_ids, "e", 3);

        let sql = format!(
            "SELECT e.id, e.label, e.properties, e.created_at, e.updated_at, e.group_id,
                    vector_distance_cos(e.embedding, vector(?1)) as distance
             FROM vector_top_k('rql_entities_vec_idx', vector(?1), ?2) AS v
             JOIN rql_entities AS e ON e.rowid = v.id
             WHERE 1=1{}
             ORDER BY distance ASC",
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
            let distance = row.get::<f64>(6)?;
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
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit)
        let (group_clause, group_params) =
            build_group_id_clause(&filters.group_ids, "rql_entities", 3);

        let sql = format!(
            "SELECT id, label, properties, created_at, updated_at, group_id,
                    vector_distance_cos(embedding, vector(?1)) as distance
             FROM rql_entities
             WHERE embedding IS NOT NULL{}
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
            let entity = row_to_entity_from_row(&row)?;
            let distance = row.get::<f64>(6)?;
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
        histogram!("rql.search.hybrid_entities_hits").record(results.len() as f64);
        histogram!("rql.search.hybrid_entities_ms")
            .record(_search_start.elapsed().as_secs_f64() * 1000.0);
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
        histogram!("rql.search.vector_facts_hits").record(hits.len() as f64);
        histogram!("rql.search.vector_facts_ms")
            .record(_search_start.elapsed().as_secs_f64() * 1000.0);
        Ok(hits)
    }

    async fn vector_search_facts_with_index(
        &self,
        vec_str: &str,
        limit: usize,
        filters: &SearchFilters,
    ) -> anyhow::Result<Vec<SearchHit<Fact>>> {
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=limit)
        let (group_clause, group_params) =
            build_group_id_clause(&filters.group_ids, "f", 3);

        let sql = format!(
            "SELECT f.id, f.subject_id, f.predicate, f.object_id, f.object_value, f.properties,
                    f.valid_from, f.valid_to, f.created_at, f.expired_at, f.invalid_at, f.group_id,
                    f.confidence, f.source_episode_id,
                    vector_distance_cos(f.embedding, vector(?1)) as distance
             FROM vector_top_k('facts_vec_idx', vector(?1), ?2) AS v
             JOIN facts AS f ON f.rowid = v.id
             WHERE f.expired_at IS NULL{}
             ORDER BY distance ASC",
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
            let distance = row.get::<f64>(14)?;
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
        let (group_clause, group_params) =
            build_group_id_clause(&filters.group_ids, "facts", 3);

        let sql = format!(
            "SELECT id, subject_id, predicate, object_id, object_value, properties,
                    valid_from, valid_to, created_at, expired_at, invalid_at, group_id,
                    confidence, source_episode_id,
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
            let distance = row.get::<f64>(14)?;
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
        histogram!("rql.search.hybrid_facts_hits").record(results.len() as f64);
        histogram!("rql.search.hybrid_facts_ms")
            .record(_search_start.elapsed().as_secs_f64() * 1000.0);
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
fn rrf_fuse_entities(
    vector_hits: Vec<SearchHit<Entity>>,
    fts_hits: Vec<SearchHit<Entity>>,
    k: f64,
) -> Vec<SearchHit<Entity>> {
    use std::collections::HashMap;

    // Build a map of entity_id -> (rrf_score, Entity)
    let mut scores: HashMap<String, (f64, Entity)> = HashMap::new();

    // Score vector results by rank position
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        scores
            .entry(hit.item.id.clone())
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, hit.item));
    }

    // Score FTS results by rank position
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        scores
            .entry(hit.item.id.clone())
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
/// Higher RRF score means more relevant. Deduplicates by fact id.
fn rrf_fuse_facts(
    vector_hits: Vec<SearchHit<Fact>>,
    fts_hits: Vec<SearchHit<Fact>>,
    k: f64,
) -> Vec<SearchHit<Fact>> {
    use std::collections::HashMap;

    // Build a map of fact_id -> (rrf_score, Fact)
    let mut scores: HashMap<i64, (f64, Fact)> = HashMap::new();

    // Score vector results by rank position
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        scores
            .entry(hit.item.id)
            .and_modify(|(s, _)| *s += rrf_score)
            .or_insert((rrf_score, hit.item));
    }

    // Score FTS results by rank position
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f64 + 1.0);
        scores
            .entry(hit.item.id)
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

/// Helper to extract an Entity from a query row.
/// Expected columns: id(0), label(1), properties(2), created_at(3), updated_at(4), group_id(5).
fn row_to_entity_from_row(row: &libsql::Row) -> anyhow::Result<Entity> {
    use chrono::DateTime;
    let id = row.get::<String>(0)?;
    let label = row.get::<String>(1)?;
    let props_str = row.get::<Option<String>>(2)?;
    let created_str = row.get::<String>(3)?;
    let updated_str = row.get::<Option<String>>(4)?;
    let group_id = row.get::<Option<String>>(5)?;

    let parse_dt = |s: &str| -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
        Ok(DateTime::parse_from_rfc3339(s)
            .map_err(|e| anyhow::anyhow!("bad timestamp '{}': {}", s, e))?
            .with_timezone(&chrono::Utc))
    };

    let properties = props_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);
    let created_at = parse_dt(&created_str)?;
    let updated_at = updated_str.as_deref().map(parse_dt).transpose()?;

    Ok(Entity {
        id,
        label,
        properties,
        created_at,
        updated_at,
        group_id,
    })
}

/// Helper to extract a Fact from a query row (same column order as fts_search_facts query).
/// Columns: id, subject_id, predicate, object_id, object_value, properties,
///          valid_from, valid_to, created_at, expired_at, invalid_at, group_id, confidence, source_episode_id
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
    let created_str = row.get::<String>(8)?;
    let expired_str = row.get::<Option<String>>(9)?;
    let invalid_str = row.get::<Option<String>>(10)?;
    let group_id = row.get::<Option<String>>(11)?;
    let confidence = row.get::<f64>(12)?;
    let source_episode_id = row.get::<Option<i64>>(13)?;

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
    let created_at = parse_dt(&created_str)?;
    let expired_at = expired_str.as_deref().map(parse_dt).transpose()?;
    let invalid_at = invalid_str.as_deref().map(parse_dt).transpose()?;

    Ok(Fact {
        id,
        subject_id,
        predicate,
        object_id,
        object_value,
        properties,
        valid_from,
        valid_to,
        created_at,
        expired_at,
        invalid_at,
        group_id,
        confidence,
        source_episode_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

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
            "Person",
            serde_json::json!({"role": "engineer", "department": "platform"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "bob",
            "Person",
            serde_json::json!({"role": "manager", "department": "sales"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "acme",
            "Company",
            serde_json::json!({"industry": "technology", "size": "startup"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "budget_2025",
            "Document",
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
    async fn test_fts_search_entities_by_label() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g.fts_search_entities("Person", 10, &no_filter).await.unwrap();
        assert_eq!(hits.len(), 2, "should find 2 Person entities");
        // BM25 scores should be negative
        for hit in &hits {
            assert!(hit.score < 0.0, "BM25 rank should be negative");
        }
    }

    #[tokio::test]
    async fn test_fts_search_entities_by_properties() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g.fts_search_entities("engineer", 10, &no_filter).await.unwrap();
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
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g.fts_search_entities("Person", 1, &no_filter).await.unwrap();
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
        let hits = g.fts_search_facts("Engineer", 10, &no_filter).await.unwrap();
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
        let hits = g.fts_search_facts("has_title", 10, &no_filter).await.unwrap();
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

        g.insert_entity("alice", "Person", serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({"role": "manager"}))
            .await
            .unwrap();
        g.insert_entity("acme", "Company", serde_json::json!({"industry": "tech"}))
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
        let hits = g.vector_search_entities(&alice_emb, 10, &no_filter).await.unwrap();
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

        g.insert_entity("a", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("b", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("c", "Person", serde_json::json!({}))
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
        let hits = g.vector_search_entities(&emb, 10, &no_filter).await.unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[tokio::test]
    async fn test_vector_search_skips_null_embeddings() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("with_emb", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("no_emb", "Person", serde_json::json!({}))
            .await
            .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("with_emb", &emb).await.unwrap();
        // no_emb has no embedding set

        let hits = g.vector_search_entities(&emb, 10, &no_filter).await.unwrap();
        assert_eq!(hits.len(), 1, "should only find entities with embeddings");
        assert_eq!(hits[0].item.id, "with_emb");
    }

    #[tokio::test]
    async fn test_vector_search_cosine_similarity_ordering() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity("close", "Person", serde_json::json!({}))
            .await
            .unwrap();
        g.insert_entity("far", "Person", serde_json::json!({}))
            .await
            .unwrap();

        let query = make_embedding(1.0);
        let close_emb = make_embedding(1.05); // very similar
        let far_emb = make_embedding(10.0); // very different

        g.set_entity_embedding("close", &close_emb).await.unwrap();
        g.set_entity_embedding("far", &far_emb).await.unwrap();

        let hits = g.vector_search_entities(&query, 10, &no_filter).await.unwrap();
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
        g.insert_entity("alice", "Person", serde_json::json!({"role": "engineer"}))
            .await
            .unwrap();
        g.insert_entity("bob", "Person", serde_json::json!({"role": "manager"}))
            .await
            .unwrap();
        g.insert_entity(
            "acme",
            "Company",
            serde_json::json!({"industry": "technology"}),
        )
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

        g.insert_entity(
            "both_match",
            "Person",
            serde_json::json!({"role": "engineer"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "fts_only",
            "Person",
            serde_json::json!({"role": "engineer"}),
        )
        .await
        .unwrap();
        g.insert_entity(
            "vec_only",
            "Company",
            serde_json::json!({"industry": "finance"}),
        )
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

        g.insert_entity("a", "Person", serde_json::json!({"x": "y"}))
            .await
            .unwrap();
        g.insert_entity("b", "Person", serde_json::json!({"x": "y"}))
            .await
            .unwrap();
        g.insert_entity("c", "Person", serde_json::json!({"x": "y"}))
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

    #[tokio::test]
    async fn test_fts_search_entities_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entities in different groups
        g.insert_entity_with_group("alice", "Person", serde_json::json!({"role": "engineer"}), Some("group-a"))
            .await
            .unwrap();
        g.insert_entity_with_group("bob", "Person", serde_json::json!({"role": "manager"}), Some("group-b"))
            .await
            .unwrap();
        g.insert_entity_with_group("carol", "Person", serde_json::json!({"role": "designer"}), Some("group-a"))
            .await
            .unwrap();
        g.insert_entity_with_group("dave", "Person", serde_json::json!({"role": "analyst"}), None)
            .await
            .unwrap();

        // No filter: all 4 Person entities
        let no_filter = SearchFilters::new();
        let all = g.fts_search_entities("Person", 10, &no_filter).await.unwrap();
        assert_eq!(all.len(), 4, "no filter should return all entities");

        // Filter to group-a: alice + carol + dave (NULL = workspace-wide, visible in all scopes)
        let group_a = SearchFilters::for_group("group-a");
        let hits_a = g.fts_search_entities("Person", 10, &group_a).await.unwrap();
        assert_eq!(hits_a.len(), 3, "group-a should have 2 scoped + 1 NULL (workspace-wide)");
        let ids: Vec<&str> = hits_a.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids.contains(&"alice"));
        assert!(ids.contains(&"carol"));
        assert!(ids.contains(&"dave"), "NULL group_id entities must be visible in all scopes");

        // Filter to group-b: bob + dave (NULL = workspace-wide)
        let group_b = SearchFilters::for_group("group-b");
        let hits_b = g.fts_search_entities("Person", 10, &group_b).await.unwrap();
        assert_eq!(hits_b.len(), 2, "group-b should have 1 scoped + 1 NULL (workspace-wide)");
        let ids_b: Vec<&str> = hits_b.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids_b.contains(&"bob"));
        assert!(ids_b.contains(&"dave"));

        // Filter to non-existent group: dave only (NULL = workspace-wide)
        let group_x = SearchFilters::for_group("group-x");
        let hits_x = g.fts_search_entities("Person", 10, &group_x).await.unwrap();
        assert_eq!(hits_x.len(), 1, "non-existent group should still return NULL entities");
        assert_eq!(hits_x[0].item.id, "dave");
    }

    #[tokio::test]
    async fn test_fts_search_entities_filters_by_multiple_groups() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group("alice", "Person", serde_json::json!({}), Some("g1"))
            .await
            .unwrap();
        g.insert_entity_with_group("bob", "Person", serde_json::json!({}), Some("g2"))
            .await
            .unwrap();
        g.insert_entity_with_group("carol", "Person", serde_json::json!({}), Some("g3"))
            .await
            .unwrap();

        let filters = SearchFilters::for_groups(vec!["g1".into(), "g3".into()]);
        let hits = g.fts_search_entities("Person", 10, &filters).await.unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids.contains(&"alice"));
        assert!(ids.contains(&"carol"));
    }

    #[tokio::test]
    async fn test_vector_search_entities_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group("alice", "Person", serde_json::json!({}), Some("group-a"))
            .await
            .unwrap();
        g.insert_entity_with_group("bob", "Person", serde_json::json!({}), Some("group-b"))
            .await
            .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("alice", &emb).await.unwrap();
        g.set_entity_embedding("bob", &make_embedding(1.01)).await.unwrap();

        // No filter: both
        let no_filter = SearchFilters::new();
        let all = g.vector_search_entities(&emb, 10, &no_filter).await.unwrap();
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

        g.insert_entity("alice", "Person", serde_json::json!({}))
            .await
            .unwrap();

        g.insert_fact_with_group(
            "alice", "has_title", None, Some("Engineer"), t0, 1.0, None,
            Some("group-a"), None,
        )
        .await
        .unwrap();
        g.insert_fact_with_group(
            "alice", "has_title", None, Some("Manager"), t0, 1.0, None,
            Some("group-b"), None,
        )
        .await
        .unwrap();

        // No filter: both facts
        let no_filter = SearchFilters::new();
        let all = g.fts_search_facts("has_title", 10, &no_filter).await.unwrap();
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
            "pricing-options (part 1)",
            serde_json::json!({"text": "Full Build price £3,000 + VAT"}),
            Some("default"),
        )
        .await
        .unwrap();

        // Chat queries with space scope — should NOT find it
        let space_filter = SearchFilters::for_group("space-abc");
        let hits = g.fts_search_entities("price", 10, &space_filter).await.unwrap();
        assert_eq!(hits.len(), 0, "entity in 'default' group should not appear in space-abc");

        // Re-scope entity to space-abc (simulates update_entity_group fix)
        g.update_entity_group(
            "pricing_chunk_0",
            Some("space-abc"),
            serde_json::json!({"text": "Full Build price £3,000 + VAT"}),
        )
        .await
        .unwrap();

        // Now chat queries with space scope — SHOULD find it
        let hits = g.fts_search_entities("price", 10, &space_filter).await.unwrap();
        assert_eq!(hits.len(), 1, "re-scoped entity should be findable under new group");
        assert_eq!(hits[0].item.id, "pricing_chunk_0");
        assert_eq!(hits[0].item.group_id.as_deref(), Some("space-abc"));

        // Old group should NOT find it
        let old_filter = SearchFilters::for_group("default");
        let old_hits = g.fts_search_entities("price", 10, &old_filter).await.unwrap();
        assert_eq!(old_hits.len(), 0, "entity should no longer appear under old group");
    }

    /// NULL group_id = workspace-wide visibility. Scoped searches must include
    /// NULL entities alongside explicitly-scoped ones. Regression guard for the
    /// KGT-288 chat grounding bug where NULL entities were invisible.
    #[tokio::test]
    async fn test_null_group_id_included_when_scoped() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Insert entity WITHOUT group_id (NULL) — workspace-wide visibility.
        g.insert_entity("kb_doc", "Document", serde_json::json!({ "text": "revenue targets" }))
            .await
            .unwrap();

        // Insert entity WITH group_id — scoped to space-1.
        g.insert_entity_with_group(
            "scoped_doc", "Document",
            serde_json::json!({ "text": "revenue analysis" }),
            Some("space-1"),
        )
        .await
        .unwrap();

        // Unscoped search → both entities found.
        let all = g.fts_search_entities("revenue", 10, &SearchFilters::new()).await.unwrap();
        assert_eq!(all.len(), 2, "unscoped should return both entities");

        // Scoped search → both returned (NULL = visible in all scopes).
        let scoped = g.fts_search_entities("revenue", 10, &SearchFilters::for_group("space-1")).await.unwrap();
        assert_eq!(scoped.len(), 2, "scoped search must include NULL group_id (workspace-wide) entities");
        let ids: Vec<&str> = scoped.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids.contains(&"kb_doc"), "NULL group_id entity must be visible");
        assert!(ids.contains(&"scoped_doc"), "scoped entity must be visible");
    }

    /// Regression guard: empty group_ids = no filter (returns all entities).
    #[tokio::test]
    async fn test_empty_group_ids_returns_all() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group(
            "doc_a", "Document",
            serde_json::json!({ "text": "alpha project" }),
            Some("group-1"),
        )
        .await
        .unwrap();
        g.insert_entity_with_group(
            "doc_b", "Document",
            serde_json::json!({ "text": "alpha budget" }),
            Some("group-2"),
        )
        .await
        .unwrap();

        // Empty group_ids → no filter → both returned.
        let results = g.fts_search_entities("alpha", 10, &SearchFilters::new()).await.unwrap();
        assert_eq!(results.len(), 2, "empty group_ids should return all entities");
    }
}
