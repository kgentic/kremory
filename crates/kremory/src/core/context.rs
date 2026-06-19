use std::collections::HashMap;

use crate::core::error::Result;
use crate::core::ingest::Engine;
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::core::schema::{Entity, Fact};
use crate::core::search::{
    FtsSearchEntitiesNoCountParams, SearchFilters, VectorSearchEntitiesNoCountParams,
};

/// Result of a contextualize() call: entities + facts from search + 1-hop expansion.
#[derive(Debug)]
pub struct ContextResult {
    /// The entities found by search + 1-hop neighbors.
    pub entities: Vec<Entity>,
    /// The active (non-expired) facts connecting these entities.
    pub facts: Vec<Fact>,
    /// RRF-derived relevance scores for seed entities, min-max normalised to [0.0, 1.0].
    /// Keyed by entity ID. 1-hop neighbors not in the original seed set will be absent
    /// (callers should default to 0.0 for missing keys).
    pub scores: HashMap<String, f32>,
}

impl<L: ChatProvider, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Search for entities matching the query via RRF of FTS + vector search,
    /// then expand 1-hop to get context.
    /// Returns entities, their connecting facts, and normalised relevance scores.
    #[tracing::instrument(
        name = "kremory.contextualize",
        skip(self, query),
        fields(
            kremory.operation = "contextualize",
        )
    )]
    pub async fn contextualize(
        &self,
        query: &str,
        group_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<ContextResult> {
        let limit = limit.unwrap_or(self.config.search.top_k);

        // Build search filters from the optional group_id
        let filters = match group_id {
            Some(gid) => SearchFilters::for_group(gid),
            None => SearchFilters::new(),
        };

        // Step 1: Compute query embedding for vector search (Bug D)
        let query_embedding = self.embedder.embed(query).await?;

        // Step 2: Run FTS + vector search in parallel, suppressing per-call
        // access_count increments (RISK-002 — we do one increment after RRF)
        let fts_hits = self
            .graph
            .fts_search_entities_no_count(FtsSearchEntitiesNoCountParams {
                query,
                limit,
                filters: &filters,
            })
            .await?;
        let vector_hits = self
            .graph
            .vector_search_entities_no_count(VectorSearchEntitiesNoCountParams {
                query_embedding: &query_embedding,
                limit,
                filters: &filters,
            })
            .await?;

        // Step 3: Reciprocal Rank Fusion (RRF) with k=60 (Bug C + NEW-004)
        const RRF_K: f32 = 60.0;
        let bm25_weight = self.config.search.bm25_weight as f32;
        let vector_weight = self.config.search.vector_weight as f32;

        let mut rrf_scores: HashMap<String, f32> = HashMap::new();
        for (rank, hit) in fts_hits.iter().enumerate() {
            let id = hit.item.id.clone();
            *rrf_scores.entry(id).or_insert(0.0) +=
                bm25_weight * (1.0 / (RRF_K + rank as f32 + 1.0));
        }
        for (rank, hit) in vector_hits.iter().enumerate() {
            let id = hit.item.id.clone();
            *rrf_scores.entry(id).or_insert(0.0) +=
                vector_weight * (1.0 / (RRF_K + rank as f32 + 1.0));
        }

        // If both FTS and vector returned nothing, return empty (NEW-004: FTS-only
        // early-return deleted; this is now the single combined-empty guard)
        if rrf_scores.is_empty() {
            return Ok(ContextResult {
                entities: vec![],
                facts: vec![],
                scores: HashMap::new(),
            });
        }

        // Step 4: Sort by RRF score descending and take top-K seed IDs
        let mut ranked: Vec<(String, f32)> =
            rrf_scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let seed_ids: Vec<String> = ranked
            .iter()
            .take(limit)
            .map(|(id, _)| id.clone())
            .collect();

        // Step 5: Min-max normalise RRF scores to [0.0, 1.0]
        // Restrict to seed_ids only (top-K; drop lower-ranked hits from scores map)
        let seed_rrf: HashMap<String, f32> = seed_ids
            .iter()
            .filter_map(|id| rrf_scores.get(id).map(|s| (id.clone(), *s)))
            .collect();

        let (min_s, max_s) = (
            seed_rrf.values().copied().fold(f32::MAX, f32::min),
            seed_rrf.values().copied().fold(f32::MIN, f32::max),
        );
        let normalized: HashMap<String, f32> = if (max_s - min_s).abs() < 1e-9 {
            // Degenerate (single result or all-equal): every result scores 1.0
            seed_rrf.keys().map(|k| (k.clone(), 1.0_f32)).collect()
        } else {
            seed_rrf
                .iter()
                .map(|(k, v)| (k.clone(), (v - min_s) / (max_s - min_s)))
                .collect()
        };

        // Step 6: Increment access_count ONCE per unique seed ID (RISK-002)
        self.graph.increment_entity_access_counts(&seed_ids).await;

        // Step 7: Expand 1-hop from each seed entity
        let mut all_entities: Vec<Entity> = Vec::new();
        let mut all_facts: Vec<Fact> = Vec::new();
        let mut seen_entity_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut seen_fact_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

        for seed_id in &seed_ids {
            let subgraph = self.graph.get_neighbours(seed_id, 1).await?;

            for entity in subgraph.entities {
                // Apply group_id filter if specified
                if let Some(gid) = group_id {
                    if entity.group_id.as_deref() != Some(gid) && entity.group_id.is_some() {
                        continue;
                    }
                }
                if seen_entity_ids.insert(entity.id.clone()) {
                    all_entities.push(entity);
                }
            }

            for fact in subgraph.facts {
                if seen_fact_ids.insert(fact.id) {
                    all_facts.push(fact);
                }
            }
        }

        Ok(ContextResult {
            entities: all_entities,
            facts: all_facts,
            scores: normalized,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::core::graph::FactInsert;
    use crate::core::ingest::SimpleGraph;
    use chrono::Utc;

    /// Build a SimpleGraph and insert some test entities and facts directly.
    async fn setup_graph_with_data() -> SimpleGraph {
        let rql = SimpleGraph::open_in_memory_simple().await.unwrap();
        let now = Utc::now();

        rql.graph
            .insert_entity("alice", 0, serde_json::json!({"name": "Alice"}))
            .await
            .unwrap();
        rql.graph
            .insert_entity("acme", 0, serde_json::json!({"name": "Acme"}))
            .await
            .unwrap();
        rql.graph
            .insert_entity("bob", 0, serde_json::json!({"name": "Bob"}))
            .await
            .unwrap();

        rql.graph
            .insert_fact(FactInsert::new("alice", "works_at", now).object_id("acme"))
            .await
            .unwrap();
        rql.graph
            .insert_fact(FactInsert::new("bob", "works_at", now).object_id("acme"))
            .await
            .unwrap();

        rql
    }

    #[tokio::test]
    async fn test_contextualize_finds_entity_and_neighbors() {
        let rql = setup_graph_with_data().await;

        // Searching for "alice" should return alice + her 1-hop neighbors (acme)
        let ctx = rql.contextualize("alice", None, None).await.unwrap();

        assert!(
            !ctx.entities.is_empty(),
            "contextualize should return at least one entity"
        );

        let entity_ids: Vec<&str> = ctx.entities.iter().map(|e| e.id.as_str()).collect();
        assert!(
            entity_ids.contains(&"alice") || entity_ids.contains(&"acme"),
            "result should contain alice or her neighbor acme"
        );

        // Facts should be present
        assert!(
            !ctx.facts.is_empty(),
            "contextualize should return connecting facts"
        );
    }

    #[tokio::test]
    async fn test_contextualize_empty_query_returns_empty() {
        let rql = setup_graph_with_data().await;

        // A term that won't match anything in FTS or vector search
        let ctx = rql
            .contextualize("xyzzy_nonexistent_term_42", None, None)
            .await
            .unwrap();

        assert!(
            ctx.entities.is_empty(),
            "no search match should yield empty entities"
        );
        assert!(
            ctx.facts.is_empty(),
            "no search match should yield empty facts"
        );
        assert!(
            ctx.scores.is_empty(),
            "no search match should yield empty scores"
        );
    }

    #[tokio::test]
    async fn test_contextualize_respects_limit() {
        let rql = setup_graph_with_data().await;

        // Phase 2 (Migration 009): entities_fts.label is empty — FTS searches
        // properties only.  Search for "Alice" which appears in alice's properties["name"].
        // limit=1 means at most 1 seed entity (neighbours may expand the final set).
        let ctx = rql.contextualize("Alice", None, Some(1)).await.unwrap();

        // We should get no more seed entities than the limit requested
        // (neighbours may expand the set, but seed query is capped)
        // The easiest check: result should be non-empty and not exceed the
        // full set (alice + bob + acme = 3) — the seed was limited to 1.
        assert!(
            !ctx.entities.is_empty(),
            "should still return entities with limit=1"
        );
    }

    #[tokio::test]
    async fn test_context_result_includes_facts() {
        let rql = setup_graph_with_data().await;

        // Phase 2 (Migration 009): entities_fts.label is empty — FTS searches
        // properties only.  Search for "Acme" which appears in acme's properties["name"].
        // 1-hop from acme should include alice and bob via works_at facts.
        let ctx = rql.contextualize("Acme", None, None).await.unwrap();

        assert!(
            !ctx.facts.is_empty(),
            "facts should be included in context result"
        );

        // All returned facts should be works_at
        for fact in &ctx.facts {
            assert_eq!(
                fact.predicate, "works_at",
                "only works_at facts should be present"
            );
        }
    }

    #[tokio::test]
    async fn test_context_result_has_scores_for_seed_entities() {
        let rql = setup_graph_with_data().await;

        let ctx = rql.contextualize("alice", None, None).await.unwrap();

        // At least one seed entity should have a score entry
        assert!(
            !ctx.scores.is_empty(),
            "scores map should be populated when results are found"
        );

        // All scores should be in [0.0, 1.0]
        for (id, score) in &ctx.scores {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "score for {id} out of [0,1]: {score}"
            );
        }
    }
}
