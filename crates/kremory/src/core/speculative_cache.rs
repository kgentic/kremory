use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::core::error::Result;
use crate::core::schema::{Entity, TemporalGraph};
use crate::core::search::SearchHit;

/// A cached entity with its pre-computed PageRank score and expiry time.
#[derive(Debug, Clone)]
struct CachedEntry {
    entity: Entity,
    /// PageRank score from the graph (higher = more important)
    pagerank: f64,
    /// When this entry was cached
    cached_at: Instant,
}

/// Speculative cache that pre-warms graph neighbours after searches.
///
/// After a search returns results, the cache traverses graph edges from
/// the result entities, ranks neighbours by PageRank importance, and
/// stores them for instant retrieval on subsequent searches.
///
/// Predictions affect LATENCY only, never RANKING.
pub struct SpeculativeCache {
    /// entity_id -> cached entry
    entries: Mutex<HashMap<String, CachedEntry>>,
    /// How long cached entries remain valid
    ttl: Duration,
    /// Max number of hops to traverse when finding neighbours
    prefetch_depth: u32,
    /// Max number of entries to cache per prefetch round
    max_prefetch: usize,
}

impl SpeculativeCache {
    pub fn new(ttl: Duration, prefetch_depth: u32, max_prefetch: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            prefetch_depth,
            max_prefetch,
        }
    }

    /// Check cache for an entity by ID. Returns Some if cached and not expired.
    pub fn get(&self, entity_id: &str) -> Option<(Entity, f64)> {
        let entries = self.entries.lock().unwrap();
        if let Some(entry) = entries.get(entity_id) {
            if entry.cached_at.elapsed() < self.ttl {
                return Some((entry.entity.clone(), entry.pagerank));
            }
        }
        None
    }

    /// Check cache for multiple entity IDs. Returns cached hits sorted by PageRank descending.
    pub fn get_many(&self, entity_ids: &[&str]) -> Vec<SearchHit<Entity>> {
        let entries = self.entries.lock().unwrap();
        let mut hits = Vec::new();
        for id in entity_ids {
            if let Some(entry) = entries.get(*id) {
                if entry.cached_at.elapsed() < self.ttl {
                    hits.push(SearchHit {
                        item: entry.entity.clone(),
                        score: entry.pagerank,
                    });
                }
            }
        }
        // Sort by pagerank descending
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    /// Pre-warm cache with graph neighbours of the given entity IDs.
    /// Traverses `prefetch_depth` hops, ranks by PageRank, caches top entries.
    /// Returns the number of entries cached.
    pub async fn prefetch(
        &self,
        graph: &TemporalGraph,
        seed_entity_ids: &[String],
    ) -> Result<usize> {
        // Build petgraph for PageRank computation
        let pg = graph.to_petgraph().await?;

        // Compute PageRank over the full graph
        let pageranks = petgraph::algo::page_rank(&pg, 0.85_f64, 20);

        // Build entity_id -> pagerank map indexed by node position
        let mut pr_map: HashMap<String, f64> = HashMap::new();
        for node_idx in pg.node_indices() {
            let entity_id = &pg[node_idx];
            pr_map.insert(entity_id.clone(), pageranks[node_idx.index()]);
        }

        // Traverse neighbours of seed entities
        let mut candidates: Vec<(Entity, f64)> = Vec::new();
        for seed_id in seed_entity_ids {
            let subgraph = graph.get_neighbours(seed_id, self.prefetch_depth).await?;
            for entity in subgraph.entities {
                if !seed_entity_ids.contains(&entity.id) {
                    // Don't cache the seeds themselves — they're already in the search results
                    let pr = pr_map.get(&entity.id).copied().unwrap_or(0.0);
                    candidates.push((entity, pr));
                }
            }
        }

        // Deduplicate by entity ID, keeping highest pagerank
        let mut deduped: HashMap<String, (Entity, f64)> = HashMap::new();
        for (entity, pr) in candidates {
            deduped
                .entry(entity.id.clone())
                .and_modify(|(_, existing_pr)| {
                    if pr > *existing_pr {
                        *existing_pr = pr;
                    }
                })
                .or_insert((entity, pr));
        }

        // Sort by PageRank descending, take top max_prefetch
        let mut sorted: Vec<(Entity, f64)> = deduped.into_values().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.max_prefetch);

        // Store in cache
        let now = Instant::now();
        let count = sorted.len();
        let mut entries = self.entries.lock().unwrap();

        // Evict expired entries first
        entries.retain(|_, entry| entry.cached_at.elapsed() < self.ttl);

        for (entity, pr) in sorted {
            entries.insert(
                entity.id.clone(),
                CachedEntry {
                    entity,
                    pagerank: pr,
                    cached_at: now,
                },
            );
        }

        Ok(count)
    }

    /// Evict all expired entries. Returns count of evicted entries.
    pub fn evict_expired(&self) -> usize {
        let mut entries = self.entries.lock().unwrap();
        let before = entries.len();
        entries.retain(|_, entry| entry.cached_at.elapsed() < self.ttl);
        before - entries.len()
    }

    /// Number of currently cached entries (including possibly expired ones).
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Returns true if there are no cached entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::TemporalGraph;
    use crate::core::search::SearchFilters;
    use chrono::{Duration as ChronoDuration, Utc};
    use std::time::Duration;

    /// Build a graph with enough structure for speculative cache testing:
    ///
    ///   project_alpha --has_member--> alice
    ///   project_alpha --has_member--> bob
    ///   alice --works_at--> acme
    ///   bob --works_at--> acme
    ///   alice --collaborates_with--> carol
    ///   carol --works_at--> startup_x
    ///   project_alpha --depends_on--> budget_q1
    ///   budget_q1 --owned_by--> dave
    ///
    async fn setup_graph() -> TemporalGraph {
        let g = TemporalGraph::open_in_memory().await.unwrap();
        let t0 = Utc::now() - ChronoDuration::hours(1);

        for (id, label) in &[
            ("project_alpha", "Project"),
            ("alice", "Person"),
            ("bob", "Person"),
            ("carol", "Person"),
            ("dave", "Person"),
            ("acme", "Company"),
            ("startup_x", "Company"),
            ("budget_q1", "Document"),
        ] {
            g.insert_entity(id, label, serde_json::json!({"name": id}))
                .await
                .unwrap();
        }

        let edges = vec![
            ("project_alpha", "has_member", "alice"),
            ("project_alpha", "has_member", "bob"),
            ("alice", "works_at", "acme"),
            ("bob", "works_at", "acme"),
            ("alice", "collaborates_with", "carol"),
            ("carol", "works_at", "startup_x"),
            ("project_alpha", "depends_on", "budget_q1"),
            ("budget_q1", "owned_by", "dave"),
        ];
        for (s, p, o) in edges {
            g.insert_fact(s, p, Some(o), None, t0, 1.0, None, None)
                .await
                .unwrap();
        }

        g
    }

    #[tokio::test]
    async fn test_prefetch_populates_cache() {
        let g = setup_graph().await;
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        let count = cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();
        assert!(count > 0, "should prefetch neighbours of project_alpha");
        assert!(cache.len() > 0, "cache should have entries");

        // alice and bob are 1-hop neighbours of project_alpha
        assert!(
            cache.get("alice").is_some(),
            "alice should be cached (1-hop from project_alpha)"
        );
        assert!(
            cache.get("bob").is_some(),
            "bob should be cached (1-hop from project_alpha)"
        );
    }

    #[tokio::test]
    async fn test_prefetch_reaches_two_hops() {
        let g = setup_graph().await;
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();

        // acme is 2 hops from project_alpha (project_alpha -> alice -> acme)
        assert!(
            cache.get("acme").is_some(),
            "acme should be cached (2-hop via alice)"
        );
    }

    #[tokio::test]
    async fn test_prefetch_excludes_seed_entities() {
        let g = setup_graph().await;
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();

        // project_alpha is the seed — should NOT be in cache (it's already in search results)
        assert!(
            cache.get("project_alpha").is_none(),
            "seed entity should not be cached"
        );
    }

    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        let g = setup_graph().await;
        // Very short TTL for testing
        let cache = SpeculativeCache::new(Duration::from_millis(1), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();
        assert!(cache.len() > 0);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Individual gets should return None
        assert!(
            cache.get("alice").is_none(),
            "expired entry should return None"
        );

        // Evict should clean up
        let evicted = cache.evict_expired();
        assert!(evicted > 0, "should evict expired entries");
        assert_eq!(cache.len(), 0, "cache should be empty after eviction");
    }

    #[tokio::test]
    async fn test_get_many_returns_sorted_by_pagerank() {
        let g = setup_graph().await;
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();

        let hits = cache.get_many(&["alice", "bob", "acme"]);
        assert!(!hits.is_empty());
        // Results should be sorted by pagerank descending
        for i in 1..hits.len() {
            assert!(
                hits[i].score <= hits[i - 1].score,
                "should be sorted by pagerank desc"
            );
        }
    }

    #[tokio::test]
    async fn test_max_prefetch_limits_cache_size() {
        let g = setup_graph().await;
        // Only allow 2 entries max
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 2);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();
        assert!(cache.len() <= 2, "cache should respect max_prefetch limit");
    }

    #[tokio::test]
    async fn test_cold_vs_cached_latency_difference() {
        let g = setup_graph().await;
        // Set embeddings for vector search
        let emb = |seed: f32| -> Vec<f32> {
            (0..384)
                .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
                .collect()
        };
        g.set_entity_embedding("alice", &emb(1.0)).await.unwrap();
        g.set_entity_embedding("bob", &emb(1.5)).await.unwrap();
        g.set_entity_embedding("acme", &emb(2.0)).await.unwrap();
        g.set_entity_embedding("carol", &emb(3.0)).await.unwrap();

        // Cold search
        let cold_start = Instant::now();
        let _cold_results = g.vector_search_entities(&emb(1.0), 5, &SearchFilters::new()).await.unwrap();
        let cold_duration = cold_start.elapsed();

        // Prefetch neighbours of alice
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);
        cache.prefetch(&g, &["alice".to_string()]).await.unwrap();

        // Cached lookup
        let cache_start = Instant::now();
        let cached_hits = cache.get_many(&["bob", "acme", "carol"]);
        let cache_duration = cache_start.elapsed();

        // Cache lookup should be significantly faster than cold search
        // (In practice: cold = DB query, cached = HashMap lookup)
        assert!(!cached_hits.is_empty(), "should have cached hits");
        assert!(
            cache_duration < cold_duration,
            "cache lookup ({:?}) should be faster than cold search ({:?})",
            cache_duration,
            cold_duration
        );
    }
}
