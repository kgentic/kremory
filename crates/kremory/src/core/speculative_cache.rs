//! # Three-Cache Separation
//!
//! kremory uses three distinct cache tiers with explicit invalidation contracts:
//!
//! ## Tier-1: Deterministic OnceLock Caches
//!
//! `static FOO: OnceLock<T>` for values computed exactly once from compile-time
//! constants. **No mutation path exists.** Examples: stop-word sets, sentence
//! terminator sets, minimum token lengths (`text_utils.rs`). These are computed
//! on first access and reused for the lifetime of the process.
//!
//! *Invalidation contract*: never invalidated.
//!
//! ## Tier-2: Mutable-Data-Derived Cache (this module)
//!
//! `SpeculativeCache` holds entity prefetch entries whose validity is tied to
//! the graph's write state. Entries are time-bounded (TTL) and evicted when
//! the `DIRTY` flag (set in `BeginGuard::commit()`) signals that the graph has
//! been mutated since the last cache fill.
//!
//! Consumers that observe `DIRTY == true` MUST call `SpeculativeCache::clear()`
//! or let TTL expiry discard stale entries before reading from the cache.
//!
//! *Invalidation contract*: entries expire after `ttl`; callers should call
//! `evict_expired()` periodically or check `DIRTY` before read.
//!
//! ## Tier-3: Per-Request HashSet / Vec
//!
//! Short-lived dedup structures allocated on the stack (or heap, but never
//! stored in `self`) for the duration of a single function call. Examples: the
//! `seen: HashSet<String>` in `scan_proper_nouns`, `extract_candidates`, and
//! `ingest_episode`. Dropped at end of call — zero inter-request leakage.
//!
//! *Invalidation contract*: automatic (scope drop).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::core::error::Result;
use crate::core::schema::{Entity, TemporalGraph};
use crate::core::search::SearchHit;

/// A cached entity with its pre-computed PageRank score and expiry time.
#[allow(dead_code)] // planned consumer: SpeculativeCache (wired in search hot-path)
#[derive(Debug, Clone)]
struct CachedEntry {
    entity: Entity,
    /// PageRank score from the graph (higher = more important)
    pagerank: f64,
    /// When this entry was cached
    cached_at: Instant,
}

/// Speculative cache that pre-warms graph neighbours after searches (Tier-2).
///
/// After a search returns results, the cache traverses graph edges from
/// the result entities, ranks neighbours by PageRank importance, and
/// stores them for instant retrieval on subsequent searches.
///
/// Predictions affect LATENCY only, never RANKING.
///
/// Invalidation: entries are TTL-bounded. Callers MUST call `evict_expired()`
/// or `clear()` when `DIRTY` is observed (see three-cache separation above).
///
/// Not yet wired into the search hot-path — integration pending.
/// Surgical exemption for the struct and all its methods until that wiring is done.
#[allow(dead_code)] // planned consumer: search hot-path integration
pub(crate) struct SpeculativeCache {
    /// entity_id -> cached entry
    entries: Mutex<HashMap<String, CachedEntry>>,
    /// How long cached entries remain valid
    ttl: Duration,
    /// Max number of hops to traverse when finding neighbours
    prefetch_depth: u32,
    /// Max number of entries to cache per prefetch round
    max_prefetch: usize,
}

#[allow(dead_code)] // planned consumer: search hot-path integration
impl SpeculativeCache {
    pub fn new(ttl: Duration, prefetch_depth: u32, max_prefetch: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            prefetch_depth,
            max_prefetch,
        }
    }

    /// Atomically check the per-handle `dirty` flag and, if set, clear all
    /// cached entries and reset the flag.
    ///
    /// Returns `true` if the cache was invalidated (dirty was true), `false`
    /// if it was already clean (no-op).
    ///
    /// Uses `swap(false, AcqRel)` so the read and the reset are one atomic
    /// operation — no TOCTOU window between observing `true` and clearing it.
    ///
    /// The `dirty` parameter is the per-handle `AtomicBool` from `TemporalGraph`.
    /// Passing it explicitly (rather than reading a global) ensures cache
    /// invalidation is scoped to the owning graph handle.
    ///
    /// Called automatically at the entry of every read method (`get`, `get_many`)
    /// so callers never need to check the dirty flag themselves.
    pub(crate) fn check_dirty_and_invalidate(&self, dirty: &AtomicBool) -> bool {
        // Atomic swap: set dirty = false and get the previous value.
        // AcqRel: the Acquire half ensures all prior writes (the graph mutation
        // that set dirty) are visible before we clear; the Release half ensures
        // the cache.clear() below is visible to all subsequent reads.
        let was_dirty = dirty.swap(false, Ordering::AcqRel);
        if was_dirty {
            self.entries
                .lock()
                .unwrap_or_else(|poisoned| {
                    panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
                })
                .clear();
        }
        was_dirty
    }

    /// Check cache for an entity by ID. Returns Some if cached and not expired.
    ///
    /// Calls `check_dirty_and_invalidate` at entry — if the per-handle dirty
    /// flag was set by a concurrent write, returns `None` without exposing stale data.
    ///
    /// `dirty` is the per-handle flag from the owning `TemporalGraph::dirty` field.
    pub fn get(&self, dirty: &AtomicBool, entity_id: &str) -> Option<(Entity, f64)> {
        if self.check_dirty_and_invalidate(dirty) {
            return None;
        }
        let entries = self.entries.lock().unwrap_or_else(|poisoned| {
            panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
        });
        if let Some(entry) = entries.get(entity_id) {
            if entry.cached_at.elapsed() < self.ttl {
                return Some((entry.entity.clone(), entry.pagerank));
            }
        }
        None
    }

    /// Check cache for multiple entity IDs. Returns cached hits sorted by PageRank descending.
    ///
    /// Calls `check_dirty_and_invalidate` at entry — if the per-handle dirty
    /// flag was set by a concurrent write, returns empty `Vec` without stale data.
    ///
    /// `dirty` is the per-handle flag from the owning `TemporalGraph::dirty` field.
    pub fn get_many(&self, dirty: &AtomicBool, entity_ids: &[&str]) -> Vec<SearchHit<Entity>> {
        if self.check_dirty_and_invalidate(dirty) {
            return Vec::new();
        }
        let entries = self.entries.lock().unwrap_or_else(|poisoned| {
            panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
        });
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
        let mut entries = self.entries.lock().unwrap_or_else(|poisoned| {
            panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
        });

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
        let mut entries = self.entries.lock().unwrap_or_else(|poisoned| {
            panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
        });
        let before = entries.len();
        entries.retain(|_, entry| entry.cached_at.elapsed() < self.ttl);
        before - entries.len()
    }

    /// Number of currently cached entries (including possibly expired ones).
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| {
                panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
            })
            .len()
    }

    /// Returns true if there are no cached entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| {
                panic!("invariant: SpeculativeCache entries mutex poisoned: {poisoned}")
            })
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{FactInsert, InsertEntityParams};
    use crate::core::schema::TemporalGraph;
    use crate::core::search::{SearchFilters, VectorSearchEntitiesParams};
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

        for id in &[
            "project_alpha",
            "alice",
            "bob",
            "carol",
            "dave",
            "acme",
            "startup_x",
            "budget_q1",
        ] {
            g.insert_entity(InsertEntityParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({"name": id}),
            })
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
            g.insert_fact(FactInsert::new(s, p, t0).object_id(o))
                .await
                .unwrap();
        }

        g
    }

    #[tokio::test]
    async fn test_prefetch_populates_cache() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        // Reset: insert_entity/insert_fact commits set dirty=true.
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        let count = cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();
        assert!(count > 0, "should prefetch neighbours of project_alpha");
        assert!(!cache.is_empty(), "cache should have entries");

        // Reset before reads so prefetch entries survive.
        dirty.store(false, Ordering::Release);
        // alice and bob are 1-hop neighbours of project_alpha
        assert!(
            cache.get(dirty, "alice").is_some(),
            "alice should be cached (1-hop from project_alpha)"
        );
        assert!(
            cache.get(dirty, "bob").is_some(),
            "bob should be cached (1-hop from project_alpha)"
        );
    }

    #[tokio::test]
    async fn test_prefetch_reaches_two_hops() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();

        // Reset before read so prefetch entries survive.
        dirty.store(false, Ordering::Release);
        // acme is 2 hops from project_alpha (project_alpha -> alice -> acme)
        assert!(
            cache.get(dirty, "acme").is_some(),
            "acme should be cached (2-hop via alice)"
        );
    }

    #[tokio::test]
    async fn test_prefetch_excludes_seed_entities() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();

        dirty.store(false, Ordering::Release);
        // project_alpha is the seed — should NOT be in cache (it's already in search results)
        assert!(
            cache.get(dirty, "project_alpha").is_none(),
            "seed entity should not be cached"
        );
    }

    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        dirty.store(false, Ordering::Release);
        // Very short TTL for testing
        let cache = SpeculativeCache::new(Duration::from_millis(1), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();
        assert!(!cache.is_empty());

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Reset dirty after sleep — we want the TTL path, not the dirty path.
        // Both return None for get(), but dirty clears before evict,
        // making evict_expired() return 0.
        dirty.store(false, Ordering::Release);

        // Individual gets should return None (TTL expired)
        assert!(
            cache.get(dirty, "alice").is_none(),
            "expired entry should return None"
        );

        // Evict should clean up (entries still present — only TTL expired, not cleared)
        let evicted = cache.evict_expired();
        assert!(evicted > 0, "should evict expired entries");
        assert_eq!(cache.len(), 0, "cache should be empty after eviction");
    }

    #[tokio::test]
    async fn test_get_many_returns_sorted_by_pagerank() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();

        // Reset before read so prefetch entries survive.
        dirty.store(false, Ordering::Release);
        let hits = cache.get_many(dirty, &["alice", "bob", "acme"]);
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
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        dirty.store(false, Ordering::Release);
        // Only allow 2 entries max
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 2);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .unwrap();
        assert!(cache.len() <= 2, "cache should respect max_prefetch limit");
    }

    /// Tier-2 cache must not return stale data
    /// after a write.
    ///
    /// Original flow (caller-manual eviction):
    ///   1. Prefetch → cache is warm.
    ///   2. Set DIRTY.
    ///   3. Caller manually calls clear() + resets DIRTY.
    ///   4. get() returns None.
    ///
    /// Auto-invalidation is now wired into get(). Step 3 is now handled
    /// internally — callers no longer need to check DIRTY before reading.
    /// This test retains the manual clear() path for backward-compat coverage;
    /// the tests below verify the no-manual-clear path.
    #[tokio::test]
    async fn tier2_cache_stale_data_evicted_after_dirty_write() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        // Reset: insert_entity/insert_fact commits set dirty=true;
        // reset so the warm-check is not consumed before the deliberate write.
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        // Step 1: warm the cache.
        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .expect("prefetch");
        // Reset before warm-check.
        dirty.store(false, Ordering::Release);
        assert!(
            cache.get(dirty, "alice").is_some(),
            "cache should be warm with alice before write"
        );

        // Step 2: simulate a write by setting the per-handle dirty flag.
        dirty.store(true, Ordering::Release);

        // Step 3: manual eviction path (backward-compat — still valid).
        // Auto-wiring means the next get() would also do this; manual
        // clear() here exercises the explicit eviction branch.
        cache.clear();
        dirty.store(false, Ordering::Release);

        // Step 4: cache must return None — no stale tier-2 read.
        assert!(
            cache.get(dirty, "alice").is_none(),
            "after write + eviction, cache must not return stale data"
        );
    }

    // -------------------------------------------------------------------------
    // check_dirty_and_invalidate + auto-wiring
    // -------------------------------------------------------------------------

    /// Setting dirty then calling get() returns None without any
    /// manual clear() call — auto-invalidation fires inside get().
    #[tokio::test]
    async fn cache_get_returns_none_when_dirty_set() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        // Reset: commits in setup_graph set dirty=true; reset so the
        // warm-check is not auto-invalidated before the deliberate write.
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .expect("prefetch");
        // Reset again before warm-check.
        dirty.store(false, Ordering::Release);
        assert!(
            cache.get(dirty, "alice").is_some(),
            "pre-condition: alice in cache before write"
        );

        // Simulate a graph write — caller sets dirty, cache wires the rest.
        dirty.store(true, Ordering::Release);

        // No manual clear(). get() must auto-invalidate and return None.
        assert!(
            cache.get(dirty, "alice").is_none(),
            "get() must return None when dirty was true, without a manual clear()"
        );
    }

    /// Atomic reset: after the first get() that observes dirty=true,
    /// the flag must be reset to false — no second invalidation needed.
    #[tokio::test]
    async fn cache_get_resets_dirty_atomically() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
        // Reset: commits in setup_graph set dirty=true; reset so
        // prefetch's warm-up is not immediately invalidated.
        dirty.store(false, Ordering::Release);
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .expect("prefetch");

        // Arm dirty immediately before the assertion sequence.
        dirty.store(true, Ordering::Release);
        // First get observes dirty and resets the flag.
        let _ = cache.get(dirty, "alice");

        assert!(
            !dirty.load(Ordering::Acquire),
            "dirty must be false after get() consumed the flag"
        );
    }

    /// Concurrent swap: N threads race to call check_dirty_and_invalidate()
    /// while dirty=true. Exactly one swap wins (sees `was_dirty=true`); all others
    /// see `was_dirty=false`. The test verifies the cache ends up consistently
    /// empty (no stale data) and dirty is false afterward.
    #[tokio::test]
    async fn cache_get_concurrent_dirty_set_at_most_one_invalidation() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let g = setup_graph().await;
        let dirty = Arc::clone(g.dirty_flag());
        // Reset: commits in setup_graph set dirty=true; reset so
        // prefetch entries survive into the concurrent phase.
        dirty.store(false, Ordering::Release);
        let cache = Arc::new(SpeculativeCache::new(Duration::from_secs(60), 2, 50));

        cache
            .prefetch(&g, &["project_alpha".to_string()])
            .await
            .expect("prefetch");

        // Arm the dirty flag before spawning threads.
        dirty.store(true, Ordering::Release);

        // Count how many threads observe `was_dirty=true` (i.e. win the swap).
        let winners = Arc::new(AtomicUsize::new(0));
        let n_threads = 8_usize;

        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let cache_clone = Arc::clone(&cache);
            let dirty_clone = Arc::clone(&dirty);
            let winners_clone = Arc::clone(&winners);
            handles.push(tokio::spawn(async move {
                let was_dirty = cache_clone.check_dirty_and_invalidate(&dirty_clone);
                if was_dirty {
                    winners_clone.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.await.expect("task panicked");
        }

        // Exactly one thread should have won the swap.
        assert_eq!(
            winners.load(Ordering::Relaxed),
            1,
            "exactly one concurrent caller should observe was_dirty=true"
        );
        // dirty must have been reset.
        assert!(
            !dirty.load(Ordering::Acquire),
            "dirty must be false after concurrent invalidation"
        );
        // Cache must be empty — the winning thread cleared it.
        assert_eq!(
            cache.len(),
            0,
            "cache must be empty after concurrent invalidation"
        );
    }

    #[tokio::test]
    async fn test_cold_vs_cached_latency_difference() {
        use std::sync::atomic::Ordering;

        let g = setup_graph().await;
        let dirty = g.dirty_flag();
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
        // Reset after all writes: commits set dirty=true; reset so prefetch
        // entries survive into the cached-lookup phase.
        dirty.store(false, Ordering::Release);

        // Cold search
        let cold_start = Instant::now();
        let _cold_results = g
            .vector_search_entities(VectorSearchEntitiesParams {
                query_embedding: &emb(1.0),
                limit: 5,
                filters: &SearchFilters::new(),
            })
            .await
            .unwrap();
        let cold_duration = cold_start.elapsed();

        // Prefetch neighbours of alice
        let cache = SpeculativeCache::new(Duration::from_secs(60), 2, 50);
        cache.prefetch(&g, &["alice".to_string()]).await.unwrap();

        // Reset before cached lookup.
        dirty.store(false, Ordering::Release);

        // Cached lookup
        let cache_start = Instant::now();
        let cached_hits = cache.get_many(dirty, &["bob", "acme", "carol"]);
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
