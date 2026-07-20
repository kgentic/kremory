use chrono::{DateTime, Utc};
use metrics::histogram;
use std::time::Instant;
use tracing;

use crate::core::error::Result;
use crate::core::schema::{Entity, Fact, TemporalGraph};
// ADR-072 seq1 impl-spec §2 ("Recall arm — content_search"): the content-RAG
// projection type + episode source-attribution kind. Both are themselves
// `#[cfg(feature = "content-search")]`-gated in `memory::types`.
#[cfg(feature = "content-search")]
use crate::memory::types::{ContentPassage, SourceKind, SourceRef};

/// A search hit with BM25 relevance score.
#[derive(Debug, Clone)]
pub struct SearchHit<T> {
    pub item: T,
    pub score: f64, // BM25 rank (negative; lower = more relevant)
}

/// Tokenise + sanitise raw text into FTS5-literal (double-quoted) tokens.
///
/// FTS5 treats characters like `'`, `"`, `(`, `)`, `,`, `*`, `+`, `-`, `:`
/// as special syntax. Raw natural language text (e.g. ASR output) will cause
/// parse errors if passed directly. This function splits the input into
/// words, strips non-alphanumeric characters, and wraps each token in double
/// quotes so FTS5 treats them as literal terms. Shared by
/// [`sanitise_fts5_query`] (AND-joined) and `content_search`'s AND/OR ladder
/// below — both need the identical token set, only the join operator
/// differs.
///
/// Returns an empty `Vec` if no usable tokens remain (e.g. all-punctuation
/// input) — callers treat that as "no query" (see [`sanitise_fts5_query`]'s
/// `None` return and `content_search`'s empty-tokens short-circuit).
fn fts5_tokens(raw: &str) -> Vec<String> {
    raw.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{}\"", w))
        .collect()
}

/// Sanitise a raw text string for use as an FTS5 MATCH expression, joining
/// tokens with FTS5's default (implicit-AND) operator — every token must
/// appear in the row for a match.
///
/// Used by [`TemporalGraph::fts_search_entities`]/[`TemporalGraph::fts_search_facts`],
/// whose queries are short extracted keyword phrases (entity names, fact
/// predicates/objects) — AND-joining gives precise matches for that shape of
/// query. `content_search` does NOT use this directly (see its own AND/OR
/// ladder) because its queries are full natural-language sentences, for
/// which AND-joining every token — including stopwords — makes a match
/// against a short conversational passage near-impossible.
///
/// Returns `None` if the sanitised query is empty (no usable tokens).
fn sanitise_fts5_query(raw: &str) -> Option<String> {
    let tokens = fts5_tokens(raw);
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

/// Bundled parameters for [`TemporalGraph::fts_search_entities`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments).
pub struct FtsSearchEntitiesParams<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::fts_search_entities_no_count`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub(crate) struct FtsSearchEntitiesNoCountParams<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::fts_search_facts`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments).
pub struct FtsSearchFactsParams<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::content_search`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). ADR-072 seq1 impl-spec
/// §2. Feature-gated behind `content-search` (mirrors the type it returns).
#[cfg(feature = "content-search")]
pub(crate) struct ContentSearchParams<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::run_content_match_query`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
/// `content_search`'s AND/OR fallback ladder (ADR-072 substrate fix) calls
/// this twice with different `match_query` values, so the shared shape
/// avoids duplicating the sql/limit/group_params plumbing per rung.
#[cfg(feature = "content-search")]
struct ContentMatchQueryParams<'a> {
    sql: &'a str,
    match_query: &'a str,
    limit: usize,
    group_params: &'a [libsql::Value],
}

/// Bundled parameters for [`TemporalGraph::vector_search_entities`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct VectorSearchEntitiesParams<'a> {
    pub query_embedding: &'a [f32],
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::vector_search_entities_no_count`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub(crate) struct VectorSearchEntitiesNoCountParams<'a> {
    pub query_embedding: &'a [f32],
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::vector_search_with_index`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
struct VectorSearchWithIndexParams<'a> {
    vec_str: &'a str,
    limit: usize,
    filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::vector_search_brute_force`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
struct VectorSearchBruteForceParams<'a> {
    vec_str: &'a str,
    limit: usize,
    filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::hybrid_search_entities`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct HybridSearchEntitiesParams<'a> {
    pub query_text: &'a str,
    pub query_embedding: &'a [f32],
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::vector_search_facts`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct VectorSearchFactsParams<'a> {
    pub query_embedding: &'a [f32],
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::vector_search_facts_with_index`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
struct VectorSearchFactsWithIndexParams<'a> {
    vec_str: &'a str,
    limit: usize,
    filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::vector_search_facts_brute_force`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
struct VectorSearchFactsBruteForceParams<'a> {
    vec_str: &'a str,
    limit: usize,
    filters: &'a SearchFilters,
}

/// Bundled parameters for [`TemporalGraph::hybrid_search_facts`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub struct HybridSearchFactsParams<'a> {
    pub query_text: &'a str,
    pub query_embedding: &'a [f32],
    pub limit: usize,
    pub filters: &'a SearchFilters,
}

/// TD-114: over-fetch plan for a filtered-ANN (`vector_top_k`) query.
///
/// `vector_top_k` exposes no predicate argument, so `group_id`/namespace
/// filtering is a POST-filter applied AFTER the index fetch. Fetching exactly
/// `limit` from a shared multi-namespace index then leaves fewer than `limit`
/// after the filter when the namespace is a small fraction of the DB — a silent
/// per-namespace recall shortfall. This plan scales the index fetch by the
/// estimated namespace selectivity so ~`limit` survive the post-filter.
struct IndexFetchPlan {
    /// `k` to request from `vector_top_k` (≥ the requested `limit`).
    fetch_k: usize,
    /// Row count of the target namespace(s); `None` when unfiltered or the count
    /// query failed. Used only to decide whether an under-fill is a true
    /// ANN-horizon shortfall (`namespace_rows ≥ limit`) vs a genuinely sparse
    /// namespace (not a degradation).
    namespace_rows: Option<i64>,
}

/// TD-114: args-as-object for [`TemporalGraph::plan_index_fetch`]
/// (rust-conventions §too_many_arguments; clippy.toml threshold 3).
struct IndexFetchQuery<'a> {
    /// Table to size the over-fetch against (`"entities"` | `"facts"`).
    table: &'a str,
    /// Requested top-k.
    limit: usize,
    filters: &'a SearchFilters,
}

/// TD-114: args-as-object for [`TemporalGraph::emit_index_shortfall`]
/// (rust-conventions §too_many_arguments).
struct IndexShortfall<'a> {
    /// Arm label for the metric (`"entities"` | `"facts"`).
    arm: &'a str,
    plan: &'a IndexFetchPlan,
    /// Requested top-k.
    limit: usize,
    /// Rows actually delivered after the post-filter.
    delivered: usize,
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

    /// TD-114: compute the filtered-ANN over-fetch plan for `vector_top_k`.
    ///
    /// When a `group_id`/namespace filter is present, estimate the namespace
    /// selectivity from indexed row counts and scale the index fetch so ~`limit`
    /// rows survive the post-filter: `k ≈ limit × total / namespace_rows`, capped
    /// at `total`. Unfiltered or whole-table namespaces short-circuit to `k =
    /// limit` — the "one DB per project" common case (see TD-114 escape hatch)
    /// pays nothing. Count-query failures degrade gracefully to `k = limit`
    /// (prior behaviour), never an error.
    async fn plan_index_fetch(&self, q: IndexFetchQuery<'_>) -> IndexFetchPlan {
        let IndexFetchQuery {
            table,
            limit,
            filters,
        } = q;
        let base = effective_k(limit, usize::MAX);
        if filters.group_ids.is_empty() {
            return IndexFetchPlan {
                fetch_k: base,
                namespace_rows: None,
            };
        }
        // Count rows in the target namespace(s) — backed by `idx_entities_group`
        // / `idx_facts_group` (schema.rs), so this is a cheap indexed count.
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, table, 1);
        let ns_rows = self
            .count_rows(
                &format!("SELECT COUNT(*) FROM {table} WHERE 1=1{group_clause}"),
                group_params,
            )
            .await;
        let total_rows = self
            .count_rows(&format!("SELECT COUNT(*) FROM {table}"), vec![])
            .await;
        let fetch_k = match (ns_rows, total_rows) {
            // Only over-fetch when the namespace is a strict fraction of the DB.
            (Some(ns), Some(total)) if ns > 0 && total > ns => {
                // Inverse-selectivity scale (u128 to avoid overflow), capped at
                // total rows — `vector_top_k` tolerates k > index size. Upper
                // bound widens to `base` when the caller asked for more than
                // exist (`limit > total`), keeping `clamp` valid (min ≤ max).
                let scaled = (base as u128).saturating_mul(total as u128) / (ns as u128);
                (scaled as usize).clamp(base, (total as usize).max(base))
            }
            _ => base,
        };
        IndexFetchPlan {
            fetch_k,
            namespace_rows: ns_rows,
        }
    }

    /// Run a `SELECT COUNT(*)`-shaped query and return the scalar. Any failure
    /// (query error, empty result, non-integer) degrades to `None` so callers
    /// can fall back to the un-scaled fetch rather than propagating.
    async fn count_rows(&self, sql: &str, params: Vec<libsql::Value>) -> Option<i64> {
        let mut rows = self.conn.query(sql, params).await.ok()?;
        let row = rows.next().await.ok()??;
        row.get::<i64>(0).ok()
    }

    /// TD-114: emit the per-namespace recall-shortfall signal for a filtered-ANN
    /// query. Fires only when the namespace held ENOUGH rows to satisfy the
    /// request (`namespace_rows ≥ limit`) yet the post-filtered index delivered
    /// fewer than `limit` — a true ANN-horizon loss, distinct from a genuinely
    /// sparse namespace (which is not a degradation and stays silent).
    fn emit_index_shortfall(&self, s: IndexShortfall<'_>) {
        let IndexShortfall {
            arm,
            plan,
            limit,
            delivered,
        } = s;
        if let Some(ns) = plan.namespace_rows {
            if ns as usize >= limit && delivered < limit {
                metrics::counter!(
                    "kremory.search.namespace_recall_shortfall_total",
                    "path" => "vector_index",
                    "arm" => arm.to_owned(),
                )
                .increment(1);
                tracing::warn!(
                    arm,
                    requested = limit,
                    delivered,
                    namespace_rows = ns,
                    fetch_k = plan.fetch_k,
                    "kremory.search.namespace_recall_shortfall filtered-ANN post-filter under-filled requested k"
                );
            }
        }
    }

    /// Full-text search entities by label/properties.
    /// Returns entities ranked by BM25 relevance, scoped by `filters.group_ids`.
    pub async fn fts_search_entities(
        &self,
        params: FtsSearchEntitiesParams<'_>,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let FtsSearchEntitiesParams {
            query,
            limit,
            filters,
        } = params;
        let hits = self
            .fts_search_entities_no_count(FtsSearchEntitiesNoCountParams {
                query,
                limit,
                filters,
            })
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
        params: FtsSearchEntitiesNoCountParams<'_>,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let FtsSearchEntitiesNoCountParams {
            query,
            limit,
            filters,
        } = params;
        let _search_start = Instant::now();
        let safe_query = match sanitise_fts5_query(query) {
            Some(q) => q,
            None => {
                let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
                histogram!("rql.search.fts_entities_hits").record(0.0);
                histogram!("rql.search.fts_entities_ms").record(_ms);
                // R2.2 §spec-td-085: empty-result counter + paired info (ADR D1).
                // arm label per pre-R2 enumeration table §4.
                metrics::counter!(
                    "kremory.search.empty_result_total",
                    "arm" => "fts_entities",
                )
                .increment(1);
                tracing::info!(
                    arm = "fts_entities",
                    _ms,
                    "kremory.search.empty_result sanitiser fired"
                );
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
             ORDER BY fts.rank, fts.entity_id LIMIT ?2",
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
        params: FtsSearchFactsParams<'_>,
    ) -> Result<Vec<SearchHit<Fact>>> {
        let FtsSearchFactsParams {
            query,
            limit,
            filters,
        } = params;
        let _search_start = Instant::now();
        let safe_query = match sanitise_fts5_query(query) {
            Some(q) => q,
            None => {
                let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
                histogram!("rql.search.fts_facts_hits").record(0.0);
                histogram!("rql.search.fts_facts_ms").record(_ms);
                // R2.2 §spec-td-085: empty-result counter + paired info (ADR D1).
                // arm label per pre-R2 enumeration table §4.
                metrics::counter!(
                    "kremory.search.empty_result_total",
                    "arm" => "fts_facts",
                )
                .increment(1);
                tracing::info!(
                    arm = "fts_facts",
                    _ms,
                    "kremory.search.empty_result sanitiser fired"
                );
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
             ORDER BY fts.rank, f.id
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

    /// Run one `episodes_fts MATCH` attempt for `content_search` and parse
    /// the rows into `ContentPassage`s. Pure query-execution + row-mapping —
    /// no metrics/logging (the caller, `content_search`, owns per-arm
    /// observability since it may call this twice, see the AND/OR ladder
    /// below).
    ///
    /// FTS5 MATCH syntax errors surface here (Rule 19 — never swallowed):
    /// `fts5_tokens` quotes every token so this should be rare in practice,
    /// but a genuine parse failure must be loud, not silent.
    #[cfg(feature = "content-search")]
    async fn run_content_match_query(
        &self,
        params: ContentMatchQueryParams<'_>,
    ) -> Result<Vec<ContentPassage>> {
        let ContentMatchQueryParams {
            sql,
            match_query,
            limit,
            group_params,
        } = params;
        let mut sql_params: Vec<libsql::Value> = vec![
            libsql::Value::from(match_query.to_string()),
            libsql::Value::from(limit as i64),
        ];
        sql_params.extend(group_params.iter().cloned());

        let mut rows = self.conn.query(sql, sql_params).await.map_err(|e| {
            metrics::counter!("kremory.recall.content_search_error_total").increment(1);
            tracing::warn!(
                error = %e,
                query = %match_query,
                "kremory.recall.content_search FTS5 MATCH query failed"
            );
            crate::core::error::Error::Search(format!("content_search MATCH query failed: {e}"))
        })?;

        let mut passages: Vec<ContentPassage> = Vec::new();
        while let Some(row) = rows.next().await? {
            let episode_id: i64 = row.get(0)?;
            let ts_str: String = row.get(1)?;
            let snippet: String = row.get(2)?;
            let score: f64 = row.get(3)?;
            let occurred_at: DateTime<Utc> = DateTime::parse_from_rfc3339(&ts_str)
                .map_err(|e| {
                    crate::core::error::Error::Parse(format!(
                        "content_search: episode {episode_id} has unparseable timestamp \
                         {ts_str:?}: {e}"
                    ))
                })?
                .with_timezone(&Utc);
            passages.push(ContentPassage {
                episode_id,
                snippet,
                score: score as f32,
                source_ref: SourceRef {
                    kind: SourceKind::Episode,
                    id: episode_id.to_string(),
                    occurred_at,
                    published_at: None,
                },
            });
        }
        Ok(passages)
    }

    /// ADR-072 seq1 impl-spec §2 — BM25-only full-text search over raw
    /// `episodes.content` via the `episodes_fts` external-content shadow
    /// table (Migration 022). **Parallel arm, NOT fused** into
    /// `rrf_fuse_entities`/`rrf_fuse_facts` below — kremory's RRF is
    /// pairwise per-result-type, not a generic N-list fuser (ADR-072 §6b);
    /// content passages are a third result *type*, returned as their own
    /// BM25-ranked stream via `.content()` (`facade::recall`).
    ///
    /// Scoped by `filters.group_ids` (same `build_group_id_clause` semantics
    /// as `fts_search_entities`/`fts_search_facts` — namespace-null legacy
    /// rows are included alongside the matched group).
    ///
    /// ## AND-first, OR-fallback ladder (content-search substrate fix)
    ///
    /// Unlike [`sanitise_fts5_query`] (used by `fts_search_entities`/
    /// `fts_search_facts`), this method does NOT always AND-join tokens.
    /// `content_search`'s consumers are full natural-language sentences
    /// (benchmark harness questions, chat-style queries) run against SHORT
    /// conversational passages (`episodes.content`) — AND-joining every
    /// token, including stopwords ("did", "the", "to", "go"), makes a match
    /// against any single passage near-impossible and silently zeroes out
    /// the entire BM25 arm for the vast majority of realistic queries (empty
    /// results, not an error — nothing was loud about it). Verified: the
    /// LoCoMo-benchmark query `"When did Caroline go to the LGBTQ support
    /// group?"` AND-joins to 9 required tokens and matches 0 of 29 ingested
    /// episodes; a 3-word keyword query like `"Postgres migration"` still
    /// matches correctly via AND.
    ///
    /// `crates/kremory/tests/content_recall_benchmark.rs` deliberately tests
    /// AND semantics (precision gate ≥0.80 for short keyword queries like
    /// `"Postgres migration"` / `"Acme revenue"`, explicitly excluding
    /// distractors that share only one term) — that benchmark's queries are
    /// short keyword phrases, not full sentences, so AND still fires first
    /// and wins for every one of them (the ladder never reaches the OR arm).
    /// The ladder therefore ADDS coverage for natural-language queries
    /// WITHOUT touching the tested AND-precision behaviour:
    ///
    /// 1. Try AND-joined tokens (FTS5 default operator) — precise; wins for
    ///    keyword-style queries and is returned immediately when non-empty.
    /// 2. If AND returns zero hits, retry OR-joined tokens — recall-oriented;
    ///    `bm25()` ranking still applies (its IDF component discounts common
    ///    terms), so the fallback is not a blunt "any word matches" scan, it
    ///    is a ranked BM25 stream biased toward the query's distinctive
    ///    terms.
    ///
    /// `fts_search_entities`/`fts_search_facts` are UNCHANGED (still
    /// AND-only via `sanitise_fts5_query`) — this ladder is scoped to
    /// `content_search` only, per the ADR-072 content-RAG substrate.
    #[cfg(feature = "content-search")]
    pub(crate) async fn content_search(
        &self,
        params: ContentSearchParams<'_>,
    ) -> Result<Vec<ContentPassage>> {
        let ContentSearchParams {
            query,
            limit,
            filters,
        } = params;
        let _search_start = Instant::now();

        let tokens = fts5_tokens(query);
        if tokens.is_empty() {
            let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
            histogram!("kremory.recall.content_search_ms").record(_ms);
            metrics::counter!(
                "kremory.search.empty_result_total",
                "arm" => "content",
            )
            .increment(1);
            tracing::debug!(
                arm = "content",
                _ms,
                "kremory.recall.content_search 0 hits (empty sanitised query)"
            );
            return Ok(vec![]);
        }

        // Build group_id filter — params start at ?3 (after ?1=query, ?2=limit).
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "e", 3);

        // Return the FULL episode text (`e.content`), not an FTS5 `snippet()`
        // excerpt. The 32-token snippet window (FTS5 caps `snippet()` at 64
        // tokens) was severing two load-bearing things measured by the
        // LoCoMo LLM-judge (Workstream A, 2026-07-20): (A) the answer itself
        // when it sat outside the ±window around the matched term, and (B) the
        // episode's leading `[Session N] [<time> on <date>]` header — needed to
        // resolve relative-date facts ("yesterday") to an absolute date — which
        // lives at the episode START, outside a mid-episode snippet window.
        // Full content carries both. `ContentPassage.snippet` now holds the full
        // episode text (field-name drift tracked for a follow-up rename); this
        // matches the Zep-style "return a full context block" shape the
        // recall-response-shape research recommends for LLM consumers.
        let sql = format!(
            "SELECT e.id, e.timestamp, e.content, \
                    episodes_fts.rank \
             FROM episodes_fts \
             JOIN episodes AS e ON e.id = episodes_fts.rowid \
             WHERE episodes_fts MATCH ?1{group_clause} \
             ORDER BY episodes_fts.rank, e.id \
             LIMIT ?2",
        );

        let and_query = tokens.join(" ");
        let mut passages = self
            .run_content_match_query(ContentMatchQueryParams {
                sql: &sql,
                match_query: &and_query,
                limit,
                group_params: &group_params,
            })
            .await?;
        let mut arm = "and";

        if passages.is_empty() {
            let or_query = tokens.join(" OR ");
            passages = self
                .run_content_match_query(ContentMatchQueryParams {
                    sql: &sql,
                    match_query: &or_query,
                    limit,
                    group_params: &group_params,
                })
                .await?;
            arm = if passages.is_empty() {
                "empty"
            } else {
                "or_fallback"
            };
        }

        let hits = passages.len();
        let _ms = _search_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("kremory.recall.content_search_ms").record(_ms);
        // Per-arm attribution (Rule 19 anti-pattern #9 — a plain success
        // counter would lie: "and" and "or_fallback" both look like generic
        // success, but "or_fallback" firing at a high rate is a signal the
        // AND-precise arm is systematically missing, worth watching).
        metrics::counter!("kremory.recall.content_search_total", "arm" => arm).increment(1);
        // Per-arm attribution (Rule 19 anti-pattern #3 — never a single
        // aggregate): content's contribution is visible against entity/fact
        // via the shared `arm` label, even though seq1 does not retrofit the
        // entity/fact paths with this same counter (out of scope — see
        // impl-spec DoD #3, entity/fact recall stays byte-identical).
        metrics::counter!("kremory.recall.results_total", "arm" => "content")
            .increment(hits as u64);
        if hits == 0 {
            tracing::debug!(_ms, "kremory.recall.content_search 0 hits");
        } else {
            tracing::info!(hits, _ms, arm, "kremory.recall.content_search");
        }
        // KREMORY_DEBUG=1: dump the raw MATCH query + returned passages
        // (diagnostic-by-env-switch, zero cost when off — mirrors the
        // existing extraction-payload KREMORY_DEBUG convention).
        if std::env::var("KREMORY_DEBUG").is_ok() {
            tracing::debug!(
                target: "kremory.recall.content_search",
                arm,
                and_query = %and_query,
                passages = ?passages,
                "[KREMORY_DEBUG] content_search raw match + results"
            );
        }
        Ok(passages)
    }

    /// Vector similarity search on entity embeddings using cosine distance.
    /// Returns entities ordered by similarity (closest first).
    /// Tries DiskANN index (vector_top_k) first, falls back to brute-force.
    /// Scoped by `filters.group_ids`.
    pub async fn vector_search_entities(
        &self,
        params: VectorSearchEntitiesParams<'_>,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let VectorSearchEntitiesParams {
            query_embedding,
            limit,
            filters,
        } = params;
        let hits = self
            .vector_search_entities_no_count(VectorSearchEntitiesNoCountParams {
                query_embedding,
                limit,
                filters,
            })
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
        params: VectorSearchEntitiesNoCountParams<'_>,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let VectorSearchEntitiesNoCountParams {
            query_embedding,
            limit,
            filters,
        } = params;
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
            .vector_search_with_index(VectorSearchWithIndexParams {
                vec_str: &vec_str,
                limit,
                filters,
            })
            .await;

        let hits = match result {
            Ok(hits) => hits,
            Err(e) => {
                // R2.2 §spec-td-085 FLAG-1: bind error before fallback — counter + paired warn (ADR D1).
                // arm/reason labels per pre-R2 enumeration table §5.
                metrics::counter!(
                    "kremory.search.error_total",
                    "arm" => "vector_entities",
                    "reason" => "index_fallback",
                )
                .increment(1);
                tracing::warn!(
                    error = %e,
                    arm = "vector_entities",
                    reason = "index_fallback",
                    "kremory.search.vector_entities index failed, falling back to brute-force"
                );
                // R2.2 FLAG-4: instrument brute-force failure path before propagation.
                match self
                    .vector_search_brute_force(VectorSearchBruteForceParams {
                        vec_str: &vec_str,
                        limit,
                        filters,
                    })
                    .await
                {
                    Ok(hits) => hits,
                    Err(e) => {
                        metrics::counter!(
                            "kremory.search.error_total",
                            "arm" => "vector_entities",
                            "reason" => "brute_force_failed",
                        )
                        .increment(1);
                        tracing::warn!(
                            error = %e,
                            arm = "vector_entities",
                            reason = "brute_force_failed",
                            "kremory.search.vector_entities brute-force failed"
                        );
                        return Err(e.into());
                    }
                }
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
        params: VectorSearchWithIndexParams<'_>,
    ) -> anyhow::Result<Vec<SearchHit<Entity>>> {
        let VectorSearchWithIndexParams {
            vec_str,
            limit,
            filters,
        } = params;
        // TD-114: `vector_top_k` has no predicate arg, so `group_id` is a
        // POST-filter (WHERE below). Over-fetch by estimated namespace
        // selectivity so ~`limit` survive, then cap with a real LIMIT.
        let base = effective_k(limit, usize::MAX);
        let plan = self
            .plan_index_fetch(IndexFetchQuery {
                table: "entities",
                limit,
                filters,
            })
            .await;
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=fetch_k).
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "e", 3);
        // Final LIMIT param sits after the variable-count group params.
        let limit_param = 3 + filters.group_ids.len();

        let sql = format!(
            "SELECT e.id, COALESCE(et.name, 'Entity') AS label, e.properties, \
                    e.recorded_at, e.updated_at, e.group_id, e.access_count, e.entity_type_id, \
                    vector_distance_cos(e.embedding, vector(?1)) as distance \
             FROM vector_top_k('entities_vec_idx', vector(?1), ?2) AS v \
             JOIN entities AS e ON e.rowid = v.id \
             LEFT JOIN entity_types et ON et.group_id = e.group_id AND et.id = e.entity_type_id \
             WHERE 1=1{group_clause} \
             ORDER BY distance ASC, e.id ASC \
             LIMIT ?{limit_param}"
        );

        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(vec_str.to_owned()),
            libsql::Value::from(plan.fetch_k as i64),
        ];
        params.extend(group_params);
        params.push(libsql::Value::from(base as i64));

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
        self.emit_index_shortfall(IndexShortfall {
            arm: "entities",
            plan: &plan,
            limit: base,
            delivered: hits.len(),
        });
        Ok(hits)
    }

    async fn vector_search_brute_force(
        &self,
        params: VectorSearchBruteForceParams<'_>,
    ) -> anyhow::Result<Vec<SearchHit<Entity>>> {
        let VectorSearchBruteForceParams {
            vec_str,
            limit,
            filters,
        } = params;
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
             ORDER BY distance ASC, e.id ASC \
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
        params: HybridSearchEntitiesParams<'_>,
    ) -> Result<Vec<SearchHit<Entity>>> {
        let HybridSearchEntitiesParams {
            query_text,
            query_embedding,
            limit,
            filters,
        } = params;
        let _search_start = Instant::now();
        // Fetch more candidates from each source than the final limit
        // to give fusion enough data to work with
        let fetch_limit = limit * 3;

        // Run both searches with the same filters
        let vector_hits = self
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding,
                limit: fetch_limit,
                filters,
            })
            .await?;
        let fts_hits = self
            .fts_search_entities(FtsSearchEntitiesParams {
                query: query_text,
                limit: fetch_limit,
                filters,
            })
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
        params: VectorSearchFactsParams<'_>,
    ) -> Result<Vec<SearchHit<Fact>>> {
        let VectorSearchFactsParams {
            query_embedding,
            limit,
            filters,
        } = params;
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
            .vector_search_facts_with_index(VectorSearchFactsWithIndexParams {
                vec_str: &vec_str,
                limit,
                filters,
            })
            .await;

        let hits = match result {
            Ok(hits) => hits,
            Err(e) => {
                // R2.2 §spec-td-085 FLAG-1: bind error before fallback — counter + paired warn (ADR D1).
                // arm/reason labels per pre-R2 enumeration table §5.
                metrics::counter!(
                    "kremory.search.error_total",
                    "arm" => "vector_facts",
                    "reason" => "index_fallback",
                )
                .increment(1);
                tracing::warn!(
                    error = %e,
                    arm = "vector_facts",
                    reason = "index_fallback",
                    "kremory.search.vector_facts index failed, falling back to brute-force"
                );
                // R2.2 FLAG-4: instrument brute-force failure path before propagation.
                match self
                    .vector_search_facts_brute_force(VectorSearchFactsBruteForceParams {
                        vec_str: &vec_str,
                        limit,
                        filters,
                    })
                    .await
                {
                    Ok(hits) => hits,
                    Err(e) => {
                        metrics::counter!(
                            "kremory.search.error_total",
                            "arm" => "vector_facts",
                            "reason" => "brute_force_failed",
                        )
                        .increment(1);
                        tracing::warn!(
                            error = %e,
                            arm = "vector_facts",
                            reason = "brute_force_failed",
                            "kremory.search.vector_facts brute-force failed"
                        );
                        return Err(e.into());
                    }
                }
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
        params: VectorSearchFactsWithIndexParams<'_>,
    ) -> anyhow::Result<Vec<SearchHit<Fact>>> {
        let VectorSearchFactsWithIndexParams {
            vec_str,
            limit,
            filters,
        } = params;
        // TD-114: `vector_top_k` post-filters `group_id` (no predicate arg), so
        // over-fetch by estimated namespace selectivity + cap with a real LIMIT.
        let base = effective_k(limit, usize::MAX);
        let plan = self
            .plan_index_fetch(IndexFetchQuery {
                table: "facts",
                limit,
                filters,
            })
            .await;
        // Build group_id filter — params start at ?3 (after ?1=vec, ?2=fetch_k).
        let (group_clause, group_params) = build_group_id_clause(&filters.group_ids, "f", 3);
        // Final LIMIT param sits after the variable-count group params.
        let limit_param = 3 + filters.group_ids.len();

        let sql = format!(
            "SELECT f.id, f.subject_id, f.predicate, f.object_id, f.object_value, f.properties,
                    f.valid_from, f.valid_to, f.recorded_at, f.expired_at, f.invalid_at, f.group_id,
                    f.confidence, f.source_episode_id,
                    f.memory_type, f.content_hash, f.access_count,
                    vector_distance_cos(f.embedding, vector(?1)) as distance
             FROM vector_top_k('facts_vec_idx', vector(?1), ?2) AS v
             JOIN facts AS f ON f.rowid = v.id
             WHERE f.expired_at IS NULL{group_clause}
             ORDER BY distance ASC, f.id ASC
             LIMIT ?{limit_param}"
        );

        let mut params: Vec<libsql::Value> = vec![
            libsql::Value::from(vec_str.to_owned()),
            libsql::Value::from(plan.fetch_k as i64),
        ];
        params.extend(group_params);
        params.push(libsql::Value::from(base as i64));

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
        self.emit_index_shortfall(IndexShortfall {
            arm: "facts",
            plan: &plan,
            limit: base,
            delivered: hits.len(),
        });
        Ok(hits)
    }

    async fn vector_search_facts_brute_force(
        &self,
        params: VectorSearchFactsBruteForceParams<'_>,
    ) -> anyhow::Result<Vec<SearchHit<Fact>>> {
        let VectorSearchFactsBruteForceParams {
            vec_str,
            limit,
            filters,
        } = params;
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
             ORDER BY distance ASC, id ASC
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
        params: HybridSearchFactsParams<'_>,
    ) -> Result<Vec<SearchHit<Fact>>> {
        let HybridSearchFactsParams {
            query_text,
            query_embedding,
            limit,
            filters,
        } = params;
        let _search_start = Instant::now();
        // Fetch more candidates from each source than the final limit
        // to give fusion enough data to work with
        let fetch_limit = limit * 3;

        // Run both searches with the same filters
        let vector_hits = self
            .vector_search_facts(VectorSearchFactsParams {
                query_embedding,
                limit: fetch_limit,
                filters,
            })
            .await?;
        let fts_hits = self
            .fts_search_facts(FtsSearchFactsParams {
                query: query_text,
                limit: fetch_limit,
                filters,
            })
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

    // Sort by RRF score descending (higher = more relevant). Ties (common —
    // e.g. two entities that only appear in one source list, at the same
    // rank position, score identically) were previously broken by
    // `HashMap`'s per-process-random iteration order, making result order
    // nondeterministic across restarts on otherwise-identical input (recall-
    // ranking nondeterminism bug). `(id, group_id)` is the same composite
    // key `scores` is keyed on above, so it's already a total order across
    // the deduplicated result set — deterministic secondary tie-break, no
    // relevance signal implied.
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
            .then_with(|| a.item.id.cmp(&b.item.id))
            .then_with(|| a.item.group_id.cmp(&b.item.group_id))
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

    // Sort by RRF score descending (higher = more relevant). Ties were
    // previously broken by `HashMap`'s per-process-random iteration order —
    // nondeterministic across restarts on identical input (recall-ranking
    // nondeterminism bug). `id` (`INTEGER PRIMARY KEY AUTOINCREMENT`, per the
    // doc comment above) is globally unique, so it alone is a sufficient
    // deterministic secondary tie-break — no relevance signal implied.
    let mut results: Vec<SearchHit<Fact>> = scores
        .into_values()
        .map(|(score, fact)| SearchHit { item: fact, score })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.item.id.cmp(&b.item.id))
    });
    results
}

// ─── TD-066 Change 2: graph-degree bonus (secondary/additive signal) ────────
//
// Grounding: `.ai-docs/research/prior-art-graph-recall-scoring-multi-hop-
// traversal--reranking-wave-1-substrate.md`. HippoRAG2 (arXiv 2502.14802)
// uses an additive pre-PPR fusion term at weight 0.05; GraphRAG/LightRAG use
// node degree as a SECONDARY sort key, never primary. kremory already has an
// UNSEEDED, GLOBAL `petgraph::algo::page_rank` in `speculative_cache.rs`
// (TD-071, dead code) — deliberately NOT reused here, because HippoRAG's own
// ablation shows un-seeded degree/PPR signals inherit hub bias. This bonus
// is instead scoped to QUERY-RELEVANT seed nodes only (HippoRAG's anti-hub
// mitigation, §3.2) and computed from the 1-hop neighbour count `context::
// contextualize` already fetches for its expansion — zero extra graph
// queries.

/// Degree value at which [`graph_degree_bonus`] saturates. Caps a single
/// highly-connected ("hub") entity's contribution instead of letting raw
/// degree grow unbounded.
pub(crate) const GRAPH_DEGREE_SATURATION: f32 = 10.0;

/// Small additive graph-degree bonus for a seed entity's own score (TD-066
/// Change 2). `degree` = the entity's 1-hop neighbour count; `weight` is the
/// axis weight (recall-v2 Phase 2a: the former `GRAPH_DEGREE_WEIGHT` const,
/// now `SearchConfig::graph_degree_weight`, default 0.05 — deliberately tiny
/// relative to the `[0, 1]` RRF-normalised score range so degree can only ever
/// nudge ranking among already-selected candidates, never dominate relevance).
/// Saturates at [`GRAPH_DEGREE_SATURATION`] and is bounded above by `weight`
/// — callers must still clamp the entity's TOTAL score (base + bonus) to
/// `[0, 1]` themselves, since this fn only bounds the bonus term. `weight <=
/// 0.0` makes the bonus a true no-op.
///
/// Callers MUST only pass the degree of a node RRF fusion already selected
/// as a search hit (a "seed") — never a degree computed from an unseeded/
/// global graph traversal (see module-level note above).
pub(crate) fn graph_degree_bonus(degree: usize, weight: f32) -> f32 {
    weight * (degree as f32 / GRAPH_DEGREE_SATURATION).min(1.0)
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
    use super::*;
    use crate::core::graph::{
        FactInsert, InsertEntityParams, InsertEntityWithGroupParams, UpdateEntityGroupParams,
    };
    use chrono::{Duration, Utc};

    // === TD-066 Change 2 / recall-v2 Phase 2a: graph_degree_bonus ===
    //
    // The weight is now a parameter (`SearchConfig::graph_degree_weight`), not
    // the removed `GRAPH_DEGREE_WEIGHT` const. `DEG_W` pins the shipped default
    // (0.05) so these tests still assert the live-config behaviour.
    const DEG_W: f32 = 0.05;

    #[test]
    fn graph_degree_bonus_zero_degree_is_zero() {
        assert_eq!(graph_degree_bonus(0, DEG_W), 0.0);
    }

    #[test]
    fn graph_degree_bonus_zero_weight_is_no_op() {
        // recall-v2 Phase 2a: weight <= 0.0 makes the axis a true no-op for any
        // degree — the mechanism by which a config can disable the boost.
        for degree in [0, 1, 10, 1_000] {
            assert_eq!(graph_degree_bonus(degree, 0.0), 0.0);
        }
    }

    #[test]
    fn graph_degree_bonus_saturates_at_ceiling() {
        // Degree far past GRAPH_DEGREE_SATURATION must not exceed the
        // saturated bonus — a single hub entity cannot keep growing its bonus.
        let saturated = graph_degree_bonus(GRAPH_DEGREE_SATURATION as usize, DEG_W);
        let hub = graph_degree_bonus(100_000, DEG_W);
        assert!(
            (hub - saturated).abs() < 1e-6,
            "degree far beyond saturation ({hub}) must equal the saturated bonus ({saturated})"
        );
    }

    #[test]
    fn graph_degree_bonus_never_exceeds_weight() {
        // Bounded above by the weight for any degree ("do not let degree swamp
        // relevance") — the bonus alone, before it's added to a base score,
        // must never exceed the configured weight.
        for degree in [0, 1, 5, 10, 50, 1_000] {
            let bonus = graph_degree_bonus(degree, DEG_W);
            assert!(
                bonus <= DEG_W + 1e-6,
                "degree={degree} produced bonus={bonus} > weight={DEG_W}"
            );
            assert!(bonus >= 0.0, "bonus must never be negative: {bonus}");
        }
    }

    #[test]
    fn graph_degree_bonus_monotonic_below_saturation() {
        // Secondary/tie-breaking signal: more-connected seeds get a bigger
        // (but still small) bonus than less-connected ones, up to saturation.
        assert!(graph_degree_bonus(5, DEG_W) > graph_degree_bonus(1, DEG_W));
        assert!(graph_degree_bonus(9, DEG_W) > graph_degree_bonus(5, DEG_W));
    }

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

    // === TD-114: filtered-ANN over-fetch plan ===

    async fn seed_two_namespaces() -> TemporalGraph {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        // 3 rows in "small", 9 in "big" → total 12, small selectivity 3/12.
        for i in 0..3 {
            g.insert_entity_with_group(InsertEntityWithGroupParams {
                id: &format!("s{i}"),
                entity_type_id: 0,
                properties: serde_json::json!({}),
                group_id: Some("small"),
            })
            .await
            .unwrap();
        }
        for i in 0..9 {
            g.insert_entity_with_group(InsertEntityWithGroupParams {
                id: &format!("b{i}"),
                entity_type_id: 0,
                properties: serde_json::json!({}),
                group_id: Some("big"),
            })
            .await
            .unwrap();
        }
        g
    }

    #[tokio::test]
    async fn plan_index_fetch_scales_by_inverse_selectivity_capped_at_total() {
        let g = seed_two_namespaces().await;
        // Small namespace: k = 4 × 12/3 = 16, clamped at total rows = 12.
        let plan = g
            .plan_index_fetch(IndexFetchQuery {
                table: "entities",
                limit: 4,
                filters: &SearchFilters::for_group("small"),
            })
            .await;
        assert_eq!(
            plan.fetch_k, 12,
            "over-fetch scales by inverse selectivity, capped at total rows"
        );
        assert_eq!(plan.namespace_rows, Some(3));

        // Regression: when the caller requests more than exist (`limit > total`),
        // `base > total` — the clamp upper bound must widen to `base` (min ≤ max),
        // not panic. fetch_k stays at base; the trailing LIMIT caps output.
        let plan_over = g
            .plan_index_fetch(IndexFetchQuery {
                table: "entities",
                limit: 100,
                filters: &SearchFilters::for_group("small"),
            })
            .await;
        assert_eq!(
            plan_over.fetch_k, 100,
            "limit > total → fetch_k = base, no panic"
        );
    }

    #[tokio::test]
    async fn plan_index_fetch_no_over_fetch_when_unfiltered_or_whole_table() {
        let g = seed_two_namespaces().await;
        // Unfiltered → base fetch, no namespace count.
        let unfiltered = g
            .plan_index_fetch(IndexFetchQuery {
                table: "entities",
                limit: 4,
                filters: &SearchFilters::new(),
            })
            .await;
        assert_eq!(unfiltered.fetch_k, 4);
        assert_eq!(unfiltered.namespace_rows, None);
        // Both namespaces (ns == total) → no over-fetch benefit.
        let whole = g
            .plan_index_fetch(IndexFetchQuery {
                table: "entities",
                limit: 4,
                filters: &SearchFilters::for_groups(vec!["small".into(), "big".into()]),
            })
            .await;
        assert_eq!(whole.fetch_k, 4, "ns == total → base fetch");
        assert_eq!(whole.namespace_rows, Some(12));
    }

    #[test]
    fn emit_index_shortfall_fires_only_on_true_ann_horizon_loss() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        fn shortfall_count(snapshotter: &metrics_util::debugging::Snapshotter) -> u64 {
            snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .filter(|(k, _, _, _)| {
                    k.key().name() == "kremory.search.namespace_recall_shortfall_total"
                })
                .filter_map(|(_, _, _, v)| match v {
                    DebugValue::Counter(c) => Some(c),
                    _ => None,
                })
                .sum()
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let g = TemporalGraph::open_in_memory().await.unwrap();

            let recorder = DebuggingRecorder::new();
            let snap = recorder.snapshotter();
            metrics::with_local_recorder(&recorder, || {
                // ns has ENOUGH rows (10 ≥ limit 5) but only 2 delivered → shortfall.
                g.emit_index_shortfall(IndexShortfall {
                    arm: "entities",
                    plan: &IndexFetchPlan {
                        fetch_k: 40,
                        namespace_rows: Some(10),
                    },
                    limit: 5,
                    delivered: 2,
                });
                assert_eq!(shortfall_count(&snap), 1, "ann-horizon loss must emit");

                // Genuinely sparse namespace (3 < limit 5): under-fill is expected,
                // NOT a degradation → silent.
                g.emit_index_shortfall(IndexShortfall {
                    arm: "entities",
                    plan: &IndexFetchPlan {
                        fetch_k: 5,
                        namespace_rows: Some(3),
                    },
                    limit: 5,
                    delivered: 3,
                });
                assert_eq!(
                    shortfall_count(&snap),
                    1,
                    "sparse namespace must stay silent"
                );

                // Request satisfied (delivered ≥ limit) → silent.
                g.emit_index_shortfall(IndexShortfall {
                    arm: "entities",
                    plan: &IndexFetchPlan {
                        fetch_k: 40,
                        namespace_rows: Some(10),
                    },
                    limit: 5,
                    delivered: 5,
                });
                assert_eq!(
                    shortfall_count(&snap),
                    1,
                    "satisfied request must stay silent"
                );
            });
        });
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

        g.insert_entity(InsertEntityParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "engineer", "department": "platform"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "manager", "department": "sales"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "acme",
            entity_type_id: 0,
            properties: serde_json::json!({"industry": "technology", "size": "startup"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "budget_2025",
            entity_type_id: 0,
            properties: serde_json::json!({"title": "Q1 Budget Review"}),
        })
        .await
        .unwrap();

        g.insert_fact(FactInsert::new("alice", "works_at", t0).object_id("acme"))
            .await
            .unwrap();
        g.insert_fact(FactInsert::new("bob", "works_at", t0).object_id("acme"))
            .await
            .unwrap();
        g.insert_fact(FactInsert::new("alice", "has_title", t0).object_value("Senior Engineer"))
            .await
            .unwrap();
        g.insert_fact(FactInsert::new("bob", "has_title", t0).object_value("Sales Manager"))
            .await
            .unwrap();
        g.insert_fact(
            FactInsert::new("alice", "discussed", t0)
                .object_value("budget allocation for Q1")
                .confidence(0.9),
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "platform",
                limit: 10,
                filters: &no_filter,
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "engineer",
                limit: 10,
                filters: &no_filter,
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "nonexistent_term_xyz",
                limit: 10,
                filters: &no_filter,
            })
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
        let hits = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "role",
                limit: 1,
                filters: &no_filter,
            })
            .await
            .unwrap();
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "Alice's work at Acme, Inc.",
                limit: 10,
                filters: &no_filter,
            })
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
            .fts_search_facts(FtsSearchFactsParams {
                query: "It's a test, isn't it?",
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert!(hits.len() <= 10);
    }

    #[tokio::test]
    async fn test_fts_search_facts_by_object_value() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_facts(FtsSearchFactsParams {
                query: "Engineer",
                limit: 10,
                filters: &no_filter,
            })
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
            .fts_search_facts(FtsSearchFactsParams {
                query: "has_title",
                limit: 10,
                filters: &no_filter,
            })
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

        let hits = g
            .fts_search_facts(FtsSearchFactsParams {
                query: "budget",
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 0, "expired facts should not appear in search");
    }

    #[tokio::test]
    async fn test_fts_search_facts_ranking() {
        let g = setup_graph_with_data().await;
        let no_filter = SearchFilters::new();
        let hits = g
            .fts_search_facts(FtsSearchFactsParams {
                query: "budget",
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
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

        g.insert_entity(InsertEntityParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "engineer"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "manager"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "acme",
            entity_type_id: 0,
            properties: serde_json::json!({"industry": "tech"}),
        })
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
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &alice_emb,
                limit: 10,
                filters: &no_filter,
            })
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

        g.insert_entity(InsertEntityParams {
            id: "a",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "b",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "c",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
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
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &emb,
                limit: 2,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 2, "limit should cap results to 2");
    }

    #[tokio::test]
    async fn test_vector_search_empty_returns_empty() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();
        // No entities, no embeddings
        let emb = make_embedding(1.0);
        let hits = g
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &emb,
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[tokio::test]
    async fn test_vector_search_skips_null_embeddings() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity(InsertEntityParams {
            id: "with_emb",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "no_emb",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();

        let emb = make_embedding(1.0);
        g.set_entity_embedding("with_emb", &emb).await.unwrap();
        // no_emb has no embedding set

        let hits = g
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &emb,
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "should only find entities with embeddings");
        assert_eq!(hits[0].item.id, "with_emb");
    }

    #[tokio::test]
    async fn test_vector_search_cosine_similarity_ordering() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let no_filter = SearchFilters::new();

        g.insert_entity(InsertEntityParams {
            id: "close",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "far",
            entity_type_id: 0,
            properties: serde_json::json!({}),
        })
        .await
        .unwrap();

        let query = make_embedding(1.0);
        let close_emb = make_embedding(1.05); // very similar
        let far_emb = make_embedding(10.0); // very different

        g.set_entity_embedding("close", &close_emb).await.unwrap();
        g.set_entity_embedding("far", &far_emb).await.unwrap();

        let hits = g
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &query,
                limit: 10,
                filters: &no_filter,
            })
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
        g.insert_entity(InsertEntityParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "engineer"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "manager"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "acme",
            entity_type_id: 0,
            properties: serde_json::json!({"industry": "technology"}),
        })
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
            .hybrid_search_entities(HybridSearchEntitiesParams {
                query_text: "engineer",
                query_embedding: &alice_emb,
                limit: 10,
                filters: &no_filter,
            })
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

        g.insert_entity(InsertEntityParams {
            id: "both_match",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "engineer"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "fts_only",
            entity_type_id: 0,
            properties: serde_json::json!({"role": "engineer"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "vec_only",
            entity_type_id: 0,
            properties: serde_json::json!({"industry": "finance"}),
        })
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
            .hybrid_search_entities(HybridSearchEntitiesParams {
                query_text: "engineer",
                query_embedding: &query_emb,
                limit: 10,
                filters: &no_filter,
            })
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

        g.insert_entity(InsertEntityParams {
            id: "a",
            entity_type_id: 0,
            properties: serde_json::json!({"x": "y"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "b",
            entity_type_id: 0,
            properties: serde_json::json!({"x": "y"}),
        })
        .await
        .unwrap();
        g.insert_entity(InsertEntityParams {
            id: "c",
            entity_type_id: 0,
            properties: serde_json::json!({"x": "y"}),
        })
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
            .hybrid_search_entities(HybridSearchEntitiesParams {
                query_text: "Person",
                query_embedding: &emb,
                limit: 2,
                filters: &no_filter,
            })
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
            .hybrid_search_entities(HybridSearchEntitiesParams {
                query_text: "Person",
                query_embedding: &make_embedding(1.0),
                limit: 10,
                filters: &no_filter,
            })
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
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({"category": "employee", "role": "engineer"}),
            group_id: Some("group-a"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({"category": "employee", "role": "manager"}),
            group_id: Some("group-b"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "carol",
            entity_type_id: 0,
            properties: serde_json::json!({"category": "employee", "role": "designer"}),
            group_id: Some("group-a"),
        })
        .await
        .unwrap();
        // dave: None → 'default' post-ADR-029b (no longer NULL = workspace-wide).
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "dave",
            entity_type_id: 0,
            properties: serde_json::json!({"category": "employee", "role": "analyst"}),
            group_id: None,
        })
        .await
        .unwrap();

        // No filter: all 4 employee entities (search on properties term, not label).
        let no_filter = SearchFilters::new();
        let all = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "employee",
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 4, "no filter should return all entities");

        // Filter to group-a: alice + carol only (dave is in 'default', not 'group-a').
        let group_a = SearchFilters::for_group("group-a");
        let hits_a = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "employee",
                limit: 10,
                filters: &group_a,
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "employee",
                limit: 10,
                filters: &group_b,
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "employee",
                limit: 10,
                filters: &group_default,
            })
            .await
            .unwrap();
        assert_eq!(hits_default.len(), 1, "'default' should return only dave");
        assert_eq!(hits_default[0].item.id, "dave");

        // Filter to non-existent group: empty (no workspace-wide entities post-ADR-029b).
        let group_x = SearchFilters::for_group("group-x");
        let hits_x = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "employee",
                limit: 10,
                filters: &group_x,
            })
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
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({"kind": "member"}),
            group_id: Some("g1"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({"kind": "member"}),
            group_id: Some("g2"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "carol",
            entity_type_id: 0,
            properties: serde_json::json!({"kind": "member"}),
            group_id: Some("g3"),
        })
        .await
        .unwrap();

        let filters = SearchFilters::for_groups(vec!["g1".into(), "g3".into()]);
        let hits = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "member",
                limit: 10,
                filters: &filters,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().map(|h| h.item.id.as_str()).collect();
        assert!(ids.contains(&"alice"));
        assert!(ids.contains(&"carol"));
    }

    #[tokio::test]
    async fn test_vector_search_entities_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some("group-a"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some("group-b"),
        })
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
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &emb,
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        // Filter to group-a: alice only
        let group_a = SearchFilters::for_group("group-a");
        let hits = g
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &emb,
                limit: 10,
                filters: &group_a,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item.id, "alice");
        assert_eq!(hits[0].item.group_id.as_deref(), Some("group-a"));
    }

    #[tokio::test]
    async fn test_fts_search_facts_filters_by_group_id() {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let t0 = Utc::now() - Duration::hours(1);

        // Composite FK (subject_id, subject_group_id) → entities(id, group_id)
        // (schema.rs:1450): a fact's subject must exist in the fact's namespace, and the
        // cross-namespace entity guard forbids one id spanning groups — so each group gets
        // its own subject entity.
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "alice",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some("group-a"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "bob",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some("group-b"),
        })
        .await
        .unwrap();

        g.insert_fact_with_group(
            FactInsert::new("alice", "has_title", t0).object_value("Engineer"),
            Some("group-a"),
        )
        .await
        .unwrap();
        g.insert_fact_with_group(
            FactInsert::new("bob", "has_title", t0).object_value("Manager"),
            Some("group-b"),
        )
        .await
        .unwrap();

        // No filter: both facts
        let no_filter = SearchFilters::new();
        let all = g
            .fts_search_facts(FtsSearchFactsParams {
                query: "has_title",
                limit: 10,
                filters: &no_filter,
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        // Filter to group-a: 1 fact
        let group_a = SearchFilters::for_group("group-a");
        let hits = g
            .fts_search_facts(FtsSearchFactsParams {
                query: "has_title",
                limit: 10,
                filters: &group_a,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].item.object_value.as_deref(), Some("Engineer"));
    }

    /// Regression test: entities indexed with wrong group_id then re-scoped
    /// must be findable under the new group_id (not the old one).
    #[tokio::test]
    async fn test_rescoped_entity_searchable_under_new_group() {
        let g = TemporalGraph::open_in_memory().await.unwrap();

        // Simulate file indexed at workspace level (group_id = "default")
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "pricing_chunk_0",
            entity_type_id: 0,
            properties: serde_json::json!({"text": "Full Build price £3,000 + VAT"}),
            group_id: Some("default"),
        })
        .await
        .unwrap();

        // Chat queries with space scope — should NOT find it
        let space_filter = SearchFilters::for_group("space-abc");
        let hits = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "price",
                limit: 10,
                filters: &space_filter,
            })
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            0,
            "entity in 'default' group should not appear in space-abc"
        );

        // Re-scope entity to space-abc (simulates update_entity_group fix)
        g.update_entity_group(UpdateEntityGroupParams {
            id: "pricing_chunk_0",
            group_id: Some("space-abc"),
            properties: serde_json::json!({"text": "Full Build price £3,000 + VAT"}),
        })
        .await
        .unwrap();

        // Now chat queries with space scope — SHOULD find it
        let hits = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "price",
                limit: 10,
                filters: &space_filter,
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "price",
                limit: 10,
                filters: &old_filter,
            })
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
        g.insert_entity(InsertEntityParams {
            id: "kb_doc",
            entity_type_id: 0,
            properties: serde_json::json!({ "text": "revenue targets" }),
        })
        .await
        .unwrap();

        // Insert entity WITH group_id — scoped to space-1.
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "scoped_doc",
            entity_type_id: 0,
            properties: serde_json::json!({ "text": "revenue analysis" }),
            group_id: Some("space-1"),
        })
        .await
        .unwrap();

        // Unscoped search → both entities found.
        let all = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "revenue",
                limit: 10,
                filters: &SearchFilters::new(),
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 2, "unscoped should return both entities");

        // Scoped search for 'default' → only kb_doc (it lives in 'default').
        let default_scoped = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "revenue",
                limit: 10,
                filters: &SearchFilters::for_group("default"),
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "revenue",
                limit: 10,
                filters: &SearchFilters::for_group("space-1"),
            })
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
        g.insert_entity(InsertEntityParams {
            id: "ac_entity_1",
            entity_type_id: 0,
            properties: serde_json::json!({ "text": "kremory_ac_probe_term_unique" }),
        })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "kremory_ac_probe_term_unique",
                limit: 10,
                filters: &SearchFilters::new(),
            })
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
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "kremory_ac_probe_term_unique",
                limit: 10,
                filters: &SearchFilters::new(),
            })
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

        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "doc_a",
            entity_type_id: 0,
            properties: serde_json::json!({ "text": "alpha project" }),
            group_id: Some("group-1"),
        })
        .await
        .unwrap();
        g.insert_entity_with_group(InsertEntityWithGroupParams {
            id: "doc_b",
            entity_type_id: 0,
            properties: serde_json::json!({ "text": "alpha budget" }),
            group_id: Some("group-2"),
        })
        .await
        .unwrap();

        // Empty group_ids → no filter → both returned.
        let results = g
            .fts_search_entities(FtsSearchEntitiesParams {
                query: "alpha",
                limit: 10,
                filters: &SearchFilters::new(),
            })
            .await
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "empty group_ids should return all entities"
        );
    }

    // === Recall-ranking nondeterminism fix: `rrf_fuse_entities`/`rrf_fuse_facts` ===
    //
    // `rrf_fuse_entities`/`rrf_fuse_facts` accumulate scores into a
    // `HashMap`, then sort the collected `Vec` by score descending. Two
    // distinct ids that each appear in only ONE of the two input lists, at
    // the same rank position, score EXACTLY the same RRF value (bit-for-bit
    // — IEEE754 addition is commutative, and here it's literally the same
    // single term on both sides: `1.0 / (k + rank + 1.0)`). Pre-fix, the
    // resulting tie was broken by `HashMap`'s iteration order — which
    // varies per `HashMap` instance (fresh `RandomState` seed on every
    // `HashMap::new()` call, even within the same process/thread) —
    // producing run-to-run ranking jitter on byte-identical input. These
    // tests force that exact tie and prove the fix's `.then_with(...)`
    // secondary key makes the result deterministic across repeated calls.

    fn test_entity(id: &str) -> Entity {
        Entity {
            id: id.to_owned(),
            label: id.to_owned(),
            entity_type_id: 0,
            properties: serde_json::Value::Null,
            recorded_at: Utc::now(),
            updated_at: None,
            group_id: None,
            access_count: 0,
        }
    }

    fn test_fact(id: i64) -> Fact {
        Fact {
            id,
            subject_id: "s".to_owned(),
            predicate: "p".to_owned(),
            object_id: None,
            object_value: Some("o".to_owned()),
            properties: None,
            valid_from: Utc::now(),
            valid_to: None,
            recorded_at: Utc::now(),
            expired_at: None,
            invalid_at: None,
            group_id: None,
            confidence: 1.0,
            source_episode_id: None,
            memory_type: None,
            content_hash: None,
            access_count: 0,
            subject_group_id: None,
            object_group_id: None,
        }
    }

    #[test]
    fn rrf_fuse_entities_deterministic_across_repeated_calls_on_genuine_tie() {
        // "tie_b" is FTS-only at rank 0; "tie_a" is vector-only at rank 0.
        // Under the shared helper's unweighted RRF (k=60), both score
        // exactly `1.0 / 61.0` — a genuine, bit-exact tie between two
        // DIFFERENT entities, forcing the HashMap-collection-then-sort path
        // to rely on the secondary tie-break to stay deterministic.
        let vector_hits = vec![SearchHit {
            item: test_entity("tie_a"),
            score: -0.01, // vector score is unused by rrf_fuse_entities (rank-based)
        }];
        let fts_hits = vec![SearchHit {
            item: test_entity("tie_b"),
            score: -1.0, // fts score is unused by rrf_fuse_entities (rank-based)
        }];

        let mut first_order: Option<Vec<String>> = None;
        for i in 0..20 {
            let out = rrf_fuse_entities(vector_hits.clone(), fts_hits.clone(), 60.0);
            assert_eq!(out.len(), 2, "run {i}: expected both tied entities present");
            assert!(
                (out[0].score - out[1].score).abs() < f64::EPSILON,
                "run {i}: expected a genuine bit-exact score tie, got {} vs {}",
                out[0].score,
                out[1].score
            );
            let order: Vec<String> = out.into_iter().map(|h| h.item.id).collect();
            match &first_order {
                None => first_order = Some(order),
                Some(expected) => assert_eq!(
                    &order, expected,
                    "run {i}: tied-score order differs from run 0 — ranking is \
                     nondeterministic across repeated calls"
                ),
            }
        }
        // The deterministic tie-break is id ascending: "tie_a" < "tie_b".
        assert_eq!(
            first_order.unwrap(),
            vec!["tie_a".to_owned(), "tie_b".to_owned()],
            "tied entities must order by id ascending, not HashMap iteration order"
        );
    }

    #[test]
    fn rrf_fuse_facts_deterministic_across_repeated_calls_on_genuine_tie() {
        // fact id 20 is FTS-only at rank 0; fact id 10 is vector-only at
        // rank 0 — same exact-tie construction as the entities test above,
        // but with numeric ids so the tie-break (ascending fact id) is
        // distinguishable from insertion/label order.
        let vector_hits = vec![SearchHit {
            item: test_fact(10),
            score: -0.01,
        }];
        let fts_hits = vec![SearchHit {
            item: test_fact(20),
            score: -1.0,
        }];

        let mut first_order: Option<Vec<i64>> = None;
        for i in 0..20 {
            let out = rrf_fuse_facts(vector_hits.clone(), fts_hits.clone(), 60.0);
            assert_eq!(out.len(), 2, "run {i}: expected both tied facts present");
            assert!(
                (out[0].score - out[1].score).abs() < f64::EPSILON,
                "run {i}: expected a genuine bit-exact score tie, got {} vs {}",
                out[0].score,
                out[1].score
            );
            let order: Vec<i64> = out.into_iter().map(|h| h.item.id).collect();
            match &first_order {
                None => first_order = Some(order),
                Some(expected) => assert_eq!(
                    &order, expected,
                    "run {i}: tied-score order differs from run 0 — ranking is \
                     nondeterministic across repeated calls"
                ),
            }
        }
        // The deterministic tie-break is id ascending: 10 < 20.
        assert_eq!(
            first_order.unwrap(),
            vec![10_i64, 20_i64],
            "tied facts must order by id ascending, not HashMap iteration order"
        );
    }
}
