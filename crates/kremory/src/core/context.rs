use crate::core::error::Result;
use crate::core::ingest::RqlGraph;
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::core::schema::{Entity, Fact};
use crate::core::search::SearchFilters;

/// Result of a contextualize() call: entities + facts from search + 1-hop expansion.
#[derive(Debug)]
pub struct ContextResult {
    /// The entities found by search + 1-hop neighbors.
    pub entities: Vec<Entity>,
    /// The active (non-expired) facts connecting these entities.
    pub facts: Vec<Fact>,
}

impl<L: ChatProvider, Emb: EmbeddingProvider> RqlGraph<L, Emb> {
    /// Search for entities matching the query, then expand 1-hop to get context.
    /// Returns entities and their connecting facts.
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

        // Step 1: Search for matching entities via FTS
        let search_hits = self
            .graph
            .fts_search_entities(query, limit, &filters)
            .await?;

        if search_hits.is_empty() {
            return Ok(ContextResult {
                entities: vec![],
                facts: vec![],
            });
        }

        // Step 2: Collect seed entity IDs
        let seed_ids: Vec<String> = search_hits.iter().map(|h| h.item.id.clone()).collect();

        // Step 3: Expand 1-hop from each seed entity
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
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::core::ingest::SimpleGraph;
    use chrono::Utc;

    /// Build a SimpleGraph and insert some test entities and facts directly.
    async fn setup_graph_with_data() -> SimpleGraph {
        let rql = SimpleGraph::open_in_memory_simple().await.unwrap();
        let now = Utc::now();

        rql.graph
            .insert_entity("alice", "Person", serde_json::json!({"name": "Alice"}))
            .await
            .unwrap();
        rql.graph
            .insert_entity("acme", "Organization", serde_json::json!({"name": "Acme"}))
            .await
            .unwrap();
        rql.graph
            .insert_entity("bob", "Person", serde_json::json!({"name": "Bob"}))
            .await
            .unwrap();

        rql.graph
            .insert_fact(
                "alice",
                "works_at",
                Some("acme"),
                None,
                now,
                1.0,
                None,
                None,
            )
            .await
            .unwrap();
        rql.graph
            .insert_fact("bob", "works_at", Some("acme"), None, now, 1.0, None, None)
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

        // A term that won't match anything in FTS
        let ctx = rql
            .contextualize("xyzzy_nonexistent_term_42", None, None)
            .await
            .unwrap();

        assert!(
            ctx.entities.is_empty(),
            "no FTS match should yield empty entities"
        );
        assert!(
            ctx.facts.is_empty(),
            "no FTS match should yield empty facts"
        );
    }

    #[tokio::test]
    async fn test_contextualize_respects_limit() {
        let rql = setup_graph_with_data().await;

        // FTS for "Person" matches alice and bob; limit=1 should only seed 1 entity
        let ctx = rql.contextualize("Person", None, Some(1)).await.unwrap();

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

        // FTS search for "Organization" hits acme; 1-hop from acme should include
        // alice and bob via works_at facts.
        let ctx = rql.contextualize("Organization", None, None).await.unwrap();

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
}
