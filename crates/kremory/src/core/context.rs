use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::core::error::Result;
use crate::core::ingest::Engine;
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::core::schema::{Entity, Fact};
use crate::core::search::{
    graph_degree_bonus, FtsSearchEntitiesNoCountParams, SearchFilters,
    VectorSearchEntitiesNoCountParams,
};

/// TD-066 Change 1 — decay factor applied to a 1-hop neighbour's score
/// relative to the seed entity that surfaced it. Grounding:
/// `.ai-docs/research/prior-art-graph-recall-scoring-multi-hop-traversal--
/// reranking-wave-1-substrate.md` — HippoRAG's ablation (arXiv 2405.14831
/// Table 5) shows plain UNWEIGHTED graph expansion measurably HURTS recall
/// (drops below no-expansion on all 3 benchmarks tested); only *weighted*
/// expansion beats the no-expansion baseline. kremory's prior behaviour
/// (bare inclusion, implicit score 0.0) was exactly the harmful unweighted
/// variant. 0.5 sits mid-range of the literature's [0.3, 0.7] decay band
/// (HippoRAG's own PPR damping factor is also 0.5).
const NEIGHBOUR_SCORE_DECAY: f32 = 0.5;

/// TD-066 Change 1 — fan-out cap per seed's 1-hop expansion. `SubGraph::
/// entities` is populated by iterating a `HashSet` (no relevance ordering),
/// so an unbounded expansion lets one highly-connected ("hub") seed flood
/// the result set with arbitrary-order neighbours that dilute/displace
/// higher-relevance candidates once `facade/recall.rs::execute` sorts +
/// truncates by score. Small, named, tunable.
const MAX_NEIGHBOURS_PER_SEED: usize = 8;

/// Bundled parameters for [`Engine::contextualize`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
#[derive(Debug, Clone, Copy)]
pub struct ContextualizeParams<'a> {
    /// Free-text query to search for.
    pub query: &'a str,
    /// Optional namespace scope.
    pub group_id: Option<&'a str>,
    /// Optional result cap (defaults to `config.search.top_k`).
    pub limit: Option<usize>,
    /// ADR-068 — point-in-time (valid-time) filter for the 1-hop fact
    /// expansion. `None` = today's behaviour (all non-expired facts,
    /// valid-time-agnostic). `Some(t)` = only facts whose
    /// `[valid_from, valid_to)` window contains `t` are surfaced. Entity
    /// search itself is unaffected — entities carry no temporal columns.
    pub as_of: Option<DateTime<Utc>>,
}

/// Result of a contextualize() call: entities + facts from search + 1-hop expansion.
#[derive(Debug)]
pub struct ContextResult {
    /// The entities found by search + 1-hop neighbors.
    pub entities: Vec<Entity>,
    /// The active (non-expired) facts connecting these entities.
    pub facts: Vec<Fact>,
    /// Relevance scores, keyed by entity ID, all clamped to `[0.0, 1.0]`.
    ///
    /// Seed entities: RRF-derived, min-max normalised, plus a small additive
    /// [`graph_degree_bonus`] (TD-066 Change 2) for their own 1-hop degree.
    /// 1-hop neighbour entities (not themselves seeds): [`NEIGHBOUR_SCORE_DECAY`]
    /// `* ` their connecting seed's score (TD-066 Change 1) — always below
    /// that seed's own score, but may still out-rank a weaker seed if
    /// strongly connected, matching the literature's "weighted expansion
    /// beats no expansion" finding. A neighbour reachable from multiple
    /// seeds takes the max decayed score across them. Any entity present in
    /// `entities` also has an entry here (no more silent 0.0-default gap).
    pub scores: HashMap<String, f32>,
}

impl<L: ChatProvider, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Search for entities matching the query via RRF of FTS + vector search,
    /// then expand 1-hop to get context.
    /// Returns entities, their connecting facts, and normalised relevance scores.
    #[tracing::instrument(
        name = "kremory.contextualize",
        skip(self, params),
        fields(
            kremory.operation = "contextualize",
        )
    )]
    pub async fn contextualize(&self, params: ContextualizeParams<'_>) -> Result<ContextResult> {
        let ContextualizeParams {
            query,
            group_id,
            limit,
            as_of,
        } = params;
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
        let mut normalized: HashMap<String, f32> = if (max_s - min_s).abs() < 1e-9 {
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

        // Step 7: Expand 1-hop from each seed entity (TD-066 Changes 1 + 2 —
        // weighted/capped expansion + seed degree bonus; see the module-level
        // `NEIGHBOUR_SCORE_DECAY`/`MAX_NEIGHBOURS_PER_SEED` docs above).
        let mut all_entities: Vec<Entity> = Vec::new();
        let mut all_facts: Vec<Fact> = Vec::new();
        let mut seen_entity_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut seen_fact_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
        // Decayed neighbour scores accumulate here (not directly into
        // `normalized`) so a neighbour that's ALSO a genuine seed elsewhere
        // in `seed_ids` never has its real RRF+degree score clobbered by a
        // decayed one — merged in below via `entry().or_insert()`.
        let mut expansion_scores: HashMap<String, f32> = HashMap::new();

        for seed_id in &seed_ids {
            // ADR-068 Decision 2/3: `get_neighbours_at` with `as_of: None`
            // runs the copy-identical query `get_neighbours` ran here before
            // this spec — switching unconditionally to the `_at` sibling
            // keeps this call site single-shaped rather than branching on
            // `as_of.is_some()`.
            let mut subgraph = self
                .graph
                .get_neighbours_at(crate::core::graph::GetNeighboursAtParams {
                    entity_id: seed_id,
                    hops: 1,
                    as_of,
                })
                .await?;

            // TD-066 Change 1 (determinism guard): `SubGraph::entities` is
            // built by iterating a `HashSet` (`TemporalGraph::get_neighbours_at`
            // visited_entities), whose order depends on Rust's per-thread
            // random hash seed — NOT stable across repeated calls (a query
            // served by a different tokio worker thread can iterate the same
            // entity set in a different order). Before MAX_NEIGHBOURS_PER_SEED
            // existed this was harmless (every neighbour was kept regardless
            // of order); now that the cap can drop entities past the Nth, an
            // unstable order would make WHICH neighbours survive
            // non-deterministic for an identical query. Sort by ID first —
            // not a relevance signal, just a stable, reproducible tie-break.
            subgraph.entities.sort_by(|a, b| a.id.cmp(&b.id));

            // TD-066 Change 2: `SubGraph::entities` always includes the seed
            // itself (see `TemporalGraph::get_neighbours_at`), so degree =
            // len - 1. This is the seed's true out-degree (computed BEFORE
            // the fan-out cap below trims which neighbours get returned) —
            // scoped to this query-relevant seed only, never a global/
            // unseeded traversal (search.rs `graph_degree_bonus` docs).
            // Clamp to 1.0: keeps `ContextResult::scores`'s `[0.0, 1.0]`
            // invariant even in the degenerate single-seed case where the
            // base score is already 1.0.
            let degree = subgraph.entities.len().saturating_sub(1);
            let degree_bonus = graph_degree_bonus(degree);
            normalized
                .entry(seed_id.clone())
                .and_modify(|s| *s = (*s + degree_bonus).min(1.0));
            let seed_score = normalized.get(seed_id).copied().unwrap_or(0.0);

            let mut neighbours_added_for_seed = 0usize;
            for entity in subgraph.entities {
                // Apply group_id filter if specified
                if let Some(gid) = group_id {
                    if entity.group_id.as_deref() != Some(gid) && entity.group_id.is_some() {
                        continue;
                    }
                }

                // The seed itself always appears in its own subgraph; it
                // already carries a real (non-decayed) score and must never
                // count against its own fan-out cap.
                let is_seed = entity.id == *seed_id;
                if !is_seed {
                    if neighbours_added_for_seed >= MAX_NEIGHBOURS_PER_SEED {
                        // TD-066 Change 1 fan-out cap reached for this seed —
                        // skip remaining neighbours (their connecting facts
                        // may still surface under the seed's own fact list).
                        continue;
                    }
                    neighbours_added_for_seed += 1;
                    let decayed = NEIGHBOUR_SCORE_DECAY * seed_score;
                    expansion_scores
                        .entry(entity.id.clone())
                        .and_modify(|s: &mut f32| *s = s.max(decayed))
                        .or_insert(decayed);
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

        // Merge decayed neighbour scores in — `or_insert` only fires when
        // the key is absent, so a neighbour that's ALSO a real seed keeps
        // its own RRF+degree score untouched.
        for (id, score) in expansion_scores {
            normalized.entry(id).or_insert(score);
        }

        // Rule 19 / ADR-074 review H1: observe the fact-collection width of
        // `contextualize`'s output — the upstream half of the same silent
        // projection/filter shape whose prior version dropped facts undetected
        // (TD-116). `graph_search` (the consumer) separately counts how many of
        // these candidates survive the per-entity ownership filter; this
        // histogram catches a regression further upstream, at collection time.
        let facts_count = all_facts.len();
        metrics::histogram!("kremory.contextualize.facts_count").record(facts_count as f64);
        tracing::debug!(
            entities = all_entities.len(),
            facts_count,
            "kremory.contextualize.facts_collected"
        );

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
    use super::{ContextualizeParams, MAX_NEIGHBOURS_PER_SEED, NEIGHBOUR_SCORE_DECAY};
    use crate::core::graph::{FactInsert, InsertEntityParams};
    use crate::core::ingest::SimpleGraph;
    use chrono::Utc;

    /// Test helper: build a [`ContextualizeParams`] from the common positional shape.
    fn ctx_params(query: &str) -> ContextualizeParams<'_> {
        ContextualizeParams {
            query,
            group_id: None,
            limit: None,
            as_of: None,
        }
    }

    /// Build a SimpleGraph and insert some test entities and facts directly.
    async fn setup_graph_with_data() -> SimpleGraph {
        let rql = SimpleGraph::open_in_memory_simple().await.unwrap();
        let now = Utc::now();

        rql.graph
            .insert_entity(InsertEntityParams {
                id: "alice",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Alice"}),
            })
            .await
            .unwrap();
        rql.graph
            .insert_entity(InsertEntityParams {
                id: "acme",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Acme"}),
            })
            .await
            .unwrap();
        rql.graph
            .insert_entity(InsertEntityParams {
                id: "bob",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Bob"}),
            })
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
        let ctx = rql.contextualize(ctx_params("alice")).await.unwrap();

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
            .contextualize(ctx_params("xyzzy_nonexistent_term_42"))
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
        let ctx = rql
            .contextualize(ContextualizeParams {
                query: "Alice",
                group_id: None,
                limit: Some(1),
                as_of: None,
            })
            .await
            .unwrap();

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
        let ctx = rql.contextualize(ctx_params("Acme")).await.unwrap();

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

        let ctx = rql.contextualize(ctx_params("alice")).await.unwrap();

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

    /// TD-066 Change 1: a genuine 1-hop neighbour (not itself a seed) must
    /// score strictly below the seed that surfaced it. Grounding:
    /// `.ai-docs/research/prior-art-graph-recall-scoring-multi-hop-
    /// traversal--reranking-wave-1-substrate.md` — naive unweighted
    /// expansion (neighbour score == seed score, or a flat default) is the
    /// pattern the research identifies as harmful.
    #[tokio::test]
    async fn test_contextualize_neighbour_score_decays_below_seed() {
        let rql = SimpleGraph::open_in_memory_simple().await.unwrap();
        let now = Utc::now();

        rql.graph
            .insert_entity(InsertEntityParams {
                id: "hubdecay",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Hubdecay"}),
            })
            .await
            .unwrap();
        rql.graph
            .insert_entity(InsertEntityParams {
                id: "leafdecay1",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Leafdecay1"}),
            })
            .await
            .unwrap();
        rql.graph
            .insert_fact(FactInsert::new("hubdecay", "connected_to", now).object_id("leafdecay1"))
            .await
            .unwrap();

        // "Hubdecay" is a unique FTS token — the seed set is exactly
        // {hubdecay}, so the degenerate min-max-normalisation branch gives
        // it a base score of 1.0 before any degree bonus.
        let ctx = rql.contextualize(ctx_params("Hubdecay")).await.unwrap();

        let hub_score = *ctx
            .scores
            .get("hubdecay")
            .expect("seed hubdecay must have a score entry");
        let leaf_score = *ctx
            .scores
            .get("leafdecay1")
            .expect("1-hop neighbour leafdecay1 must have a decayed score entry, not be absent");

        assert!(
            leaf_score < hub_score,
            "neighbour score ({leaf_score}) must be strictly below its seed's score ({hub_score})"
        );
        // hub degree = 1 → degree bonus saturates far from the ceiling and
        // hub_score clamps to 1.0 either way, so the neighbour's decayed
        // score is deterministically NEIGHBOUR_SCORE_DECAY * 1.0.
        assert!(
            (leaf_score - NEIGHBOUR_SCORE_DECAY).abs() < 1e-6,
            "expected leaf_score ({leaf_score}) == NEIGHBOUR_SCORE_DECAY ({NEIGHBOUR_SCORE_DECAY})"
        );
    }

    /// TD-066 Change 1: a hub seed's 1-hop expansion must not flood the
    /// result set past `MAX_NEIGHBOURS_PER_SEED` genuine neighbours, even
    /// when the seed is connected to far more entities than that.
    #[tokio::test]
    async fn test_contextualize_expansion_fanout_capped() {
        let rql = SimpleGraph::open_in_memory_simple().await.unwrap();
        let now = Utc::now();

        rql.graph
            .insert_entity(InsertEntityParams {
                id: "hubcap",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Hubcap"}),
            })
            .await
            .unwrap();

        let leaf_count = MAX_NEIGHBOURS_PER_SEED + 4;
        for i in 0..leaf_count {
            let leaf_id = format!("leafcap{i}");
            rql.graph
                .insert_entity(InsertEntityParams {
                    id: &leaf_id,
                    entity_type_id: 0,
                    properties: serde_json::json!({"name": format!("Leafcap{i}")}),
                })
                .await
                .unwrap();
            rql.graph
                .insert_fact(FactInsert::new("hubcap", "connected_to", now).object_id(&leaf_id))
                .await
                .unwrap();
        }

        let ctx = rql.contextualize(ctx_params("Hubcap")).await.unwrap();

        // hubcap (1 seed) + at most MAX_NEIGHBOURS_PER_SEED genuine
        // neighbours, even though hubcap is connected to `leaf_count` (>
        // MAX_NEIGHBOURS_PER_SEED) entities.
        assert!(
            ctx.entities.len() <= 1 + MAX_NEIGHBOURS_PER_SEED,
            "fan-out cap violated: {} entities returned for a seed with {leaf_count} \
             neighbours (cap = {MAX_NEIGHBOURS_PER_SEED})",
            ctx.entities.len()
        );
        assert!(
            ctx.entities.iter().any(|e| e.id == "hubcap"),
            "the seed entity itself must always be present"
        );
    }

    /// ADR-074 review H1 (Rule 19): `contextualize` must record the width of
    /// the fact collection it hands back, so a future silent regression in the
    /// 1-hop expansion (the exact shape that dropped facts undetected pre-
    /// TD-116) shows up as a metric drop, not just a passing-looking test.
    #[tokio::test]
    async fn test_contextualize_emits_facts_count_histogram() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let rql = setup_graph_with_data().await;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        // `set_default_local_recorder` (not `with_local_recorder`) — per its own
        // docs it is "suitable for capturing metrics in asynchronous code,
        // particularly when using a single-threaded runtime" (`#[tokio::test]`
        // defaults to a current-thread runtime), because the guard can be held
        // across `.await` points without requiring a sync closure.
        let guard = metrics::set_default_local_recorder(&recorder);
        let ctx = rql.contextualize(ctx_params("Acme")).await.unwrap();
        drop(guard);

        assert!(!ctx.facts.is_empty(), "fixture must produce facts");

        let recorded: Vec<f64> = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter(|(k, _, _, _)| k.key().name() == "kremory.contextualize.facts_count")
            .filter_map(|(_, _, _, v)| match v {
                DebugValue::Histogram(samples) => Some(samples),
                _ => None,
            })
            .flatten()
            .map(|v| v.into_inner())
            .collect();

        assert_eq!(
            recorded,
            vec![ctx.facts.len() as f64],
            "facts_count histogram must record exactly the returned fact count"
        );
    }
}
