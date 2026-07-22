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

// recall-v2 Phase 4 (TD-056): the former module consts `NEIGHBOUR_SCORE_DECAY`
// and `MAX_NEIGHBOURS_PER_SEED` are now `SearchConfig::neighbour_score_decay`
// (default 0.5) and `SearchConfig::expansion_fan_out_cap` (default 8) — read per
// recall from `self.config.search`. Grounding for the decay (unchanged):
// `.ai-docs/research/prior-art-graph-recall-scoring-multi-hop-traversal--
// reranking-wave-1-substrate.md` — HippoRAG's ablation (arXiv 2405.14831 Table
// 5) shows plain UNWEIGHTED graph expansion measurably HURTS recall; only
// *weighted* expansion beats the no-expansion baseline. 0.5 is mid-range of the
// literature's [0.3, 0.7] band (HippoRAG's PPR damping is also 0.5).

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

        // TD-066 phase 1 (recall-v2 spec Decision 1, #77): classify query intent
        // up-front, BEFORE FTS/vector search, so downstream scoring axes can read
        // it for per-intent weight defaults. PHASE 1 CLASSIFIES + OBSERVES ONLY —
        // it does NOT yet alter scoring (phase 2 wires the per-axis weight-override
        // lookup). Emitting it now makes intent visible on every recall for the
        // pattern-tuning the spec (Decision 5) defers to build-time eval feedback
        // (observability-first-class).
        let intent = crate::core::intent::classify_intent(query);
        metrics::counter!("kremory.recall.intent_total", "intent" => intent.as_str())
            .increment(1);
        tracing::debug!(
            target: "kremory.recall.intent",
            intent = intent.as_str(),
            "recall query intent classified (TD-066 phase 1: observe-only)"
        );

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

        // Step 3: Reciprocal Rank Fusion (RRF) (Bug C + NEW-004).
        // recall-improvement-e2e-spec-2026-07-22 §S0-infra (D3): the RRF
        // constant is read from `config.search.rrf_k` (default 60), NOT a
        // hardcoded const, so the `KREMORY_RRF_K` boot override reaches this —
        // the entity-graph FTS+vector fusion — sweep site.
        let rrf_k = self.config.search.rrf_k as f32;
        let bm25_weight = self.config.search.bm25_weight as f32;
        let vector_weight = self.config.search.vector_weight as f32;

        let mut rrf_scores: HashMap<String, f32> = HashMap::new();
        for (rank, hit) in fts_hits.iter().enumerate() {
            let id = hit.item.id.clone();
            *rrf_scores.entry(id).or_insert(0.0) +=
                bm25_weight * (1.0 / (rrf_k + rank as f32 + 1.0));
        }
        for (rank, hit) in vector_hits.iter().enumerate() {
            let id = hit.item.id.clone();
            *rrf_scores.entry(id).or_insert(0.0) +=
                vector_weight * (1.0 / (rrf_k + rank as f32 + 1.0));
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

        // Step 4: Sort by RRF score descending and take top-K seed IDs.
        // Ties (an entity present in only one of fts_hits/vector_hits at the
        // same rank position as another entity in the other list scores
        // identically) were previously broken by `rrf_scores`'s `HashMap`
        // iteration order — nondeterministic across process restarts on
        // identical input (recall-ranking nondeterminism bug; same class of
        // fix as TD-066 Change 1's neighbour-sort determinism guard below).
        // Comparator extracted to `score_desc_id_asc` so its determinism is
        // directly unit-testable without a full async DB round trip.
        let mut ranked: Vec<(String, f32)> =
            rrf_scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
        ranked.sort_by(score_desc_id_asc);
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

        // recall-v2 Phase 5 (Decision 6): the single access-count increment
        // (RISK-002) MOVED from here (pre-boost) to AFTER the boost loop + floor
        // — so the floor gates on POST-boost scores and a seed dropped by the
        // floor is never counted as recalled. See the floor block near the end.

        // Step 7: Expand 1-hop from each seed entity (TD-066 Changes 1 + 2 —
        // weighted/capped expansion + seed degree bonus; see the module-level
        // config docs above for the neighbour-decay / fan-out-cap knobs).
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

        // recall-v2 Phase 2b (Decision 1/5): resolve the per-intent axis weights
        // over the config base. All intent multipliers are 1.0 for now, so this
        // returns the config base unchanged — intent is CONSULTED (the
        // `intent_total` counter above) but behaviourally neutral until Phase 7
        // calibrates the multipliers (spec risk R3 resolved with a VISIBLE
        // half-state: the reorder counters below read 0 until a weight is set).
        let weights = crate::core::scoring::weight_overrides_for(
            intent,
            crate::core::scoring::ScoringWeights::from_config(&self.config.search),
        );
        let now = Utc::now();
        let temporal_lambda = self.config.search.temporal_decay_lambda;
        // Per-seed axis contributions, captured for post-loop reorder
        // attribution (`kremory.search.<axis>_reorder_total`).
        let mut axis_contributions: Vec<crate::core::scoring::SeedAxisContribution> =
            Vec::with_capacity(seed_ids.len());

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
                    // recall-v2 Phase 4 (TD-056): config-driven hop bound (default
                    // 1 = today's behaviour) + MANDATORY in-BFS fan-out cap in the
                    // SAME call (spec R1 — widening hops without a cap reintroduces
                    // hub-explosion). At defaults (hops=1, cap=8) byte-identical.
                    hops: self.config.search.expansion_hop_bound,
                    as_of,
                    max_visited: Some(self.config.search.expansion_fan_out_cap),
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
            // recall-v2 Phase 2a: config-driven graph-degree bonus (default 0.05,
            // resolved through the per-intent weights above).
            let degree_bonus = graph_degree_bonus(degree, weights.graph_degree_weight);
            // recall-v2 Phase 2b: additive temporal-recency boost over the facts
            // already fetched for THIS seed's 1-hop expansion — no new query
            // (read-side-pure, spec RISK-003). Bounded `[0, weight]` like the
            // degree bonus, so base + degree + temporal shares ONE `.min(1.0)`
            // clamp (Fork-1 additive hybrid — no second normalization pass).
            let temporal_bonus = crate::core::scoring::temporal::temporal_boost(
                crate::core::scoring::temporal::TemporalBoostParams {
                    facts: &subgraph.facts,
                    weight: weights.temporal_weight,
                    lambda: temporal_lambda,
                    now,
                },
            );
            // Capture the base (pre-boost) score + per-axis deltas BEFORE applying
            // them, so `axis_reorders` can attribute output-order changes to each
            // axis honestly after the loop.
            let base_score = normalized.get(seed_id).copied().unwrap_or(0.0);
            normalized
                .entry(seed_id.clone())
                .and_modify(|s| *s = (*s + degree_bonus + temporal_bonus).min(1.0));
            let seed_score = normalized.get(seed_id).copied().unwrap_or(0.0);
            axis_contributions.push(crate::core::scoring::SeedAxisContribution {
                id: seed_id.clone(),
                base: base_score,
                degree_delta: degree_bonus,
                temporal_delta: temporal_bonus,
            });

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
                    if neighbours_added_for_seed >= self.config.search.expansion_fan_out_cap {
                        // recall-v2 Phase 4 (was TD-066 const): caller-side per-seed
                        // fan-out cap reached — skip remaining neighbours for
                        // SCORING (their connecting facts may still surface under
                        // the seed's own fact list). Complements the in-BFS
                        // `max_visited` cap (which bounds traversal cost at hops>=2).
                        continue;
                    }
                    neighbours_added_for_seed += 1;
                    let decayed = self.config.search.neighbour_score_decay * seed_score;
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

        // recall-v2 Phase 2b measurement discipline: per-axis reorder attribution
        // + axis-off counters. `axis_reorders` reports whether each axis actually
        // changed the score-descending OUTPUT order of the seed set (HONEST — a
        // non-zero boost that doesn't move the order reads `changed=false`, per
        // observability Rule 19 #9 "counters must not lie"). This is the cheap
        // gate the eval reads before spending an llm-judge run: `changed=true`
        // count 0 ⇒ the axis reordered nothing ⇒ the judge run measures nothing.
        let (degree_reordered, temporal_reordered) =
            crate::core::scoring::axis_reorders(&axis_contributions);
        metrics::counter!(
            "kremory.search.graph_degree_reorder_total",
            "changed" => if degree_reordered { "true" } else { "false" },
        )
        .increment(1);
        metrics::counter!(
            "kremory.search.temporal_reorder_total",
            "changed" => if temporal_reordered { "true" } else { "false" },
        )
        .increment(1);
        // Axis-off signal: how often each axis ran with a zero (neutral) weight.
        if weights.graph_degree_weight <= 0.0 {
            metrics::counter!("kremory.search.graph_degree_weight_zero_total").increment(1);
        }
        if weights.temporal_weight <= 0.0 {
            metrics::counter!("kremory.search.temporal_weight_zero_total").increment(1);
        }
        // Fork-2 truth-boost build-trigger instrument (spec Decision 3): the
        // fact-confidence distribution over this recall's facts. Today every
        // writer hardcodes `confidence = 1.0`, so this is a spike at 1.0 — WHEN
        // it stops being a spike, that's the mechanical signal to build the
        // truth-boost axis (tracked by the new per-fact-confidence TD). Instrument
        // now, don't half-ship the axis.
        for fact in &all_facts {
            metrics::histogram!("kremory.recall.fact_confidence").record(fact.confidence);
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

        // recall-v2 Phase 5 (Decision 6): floor threshold + the moved
        // access-count increment. Apply the floor to the POST-BOOST scores over
        // the seed set, drop floored seeds from the output, then increment
        // access counts on SURVIVORS ONLY. Because this runs AFTER the boost
        // loop, the gate sees boosted scores (a seed a boost lifted above the
        // floor survives even over a higher-RRF-but-unboosted one) — and a
        // dropped seed is neither returned nor counted as recalled. RISK-002
        // single-increment is preserved: exactly one increment per unique
        // surviving seed. `floor_threshold <= 0.0` (default) → no-op.
        let floor = self.config.search.floor_threshold;
        let surviving_seeds = floor_survivors(&seed_ids, &normalized, floor);
        if surviving_seeds.len() != seed_ids.len() {
            let seed_set: std::collections::HashSet<&String> = seed_ids.iter().collect();
            let survivor_set: std::collections::HashSet<&String> = surviving_seeds.iter().collect();
            // Drop floored SEEDS from the output. Neighbours are not seeds, so
            // `!seed_set.contains` keeps every neighbour untouched.
            all_entities.retain(|e| !seed_set.contains(&e.id) || survivor_set.contains(&e.id));
            normalized.retain(|id, _| !seed_set.contains(id) || survivor_set.contains(id));
            metrics::counter!("kremory.search.floor_dropped_total")
                .increment((seed_ids.len() - surviving_seeds.len()) as u64);
        }
        // Step 6 (moved): increment access_count ONCE per unique SURVIVING seed.
        self.graph
            .increment_entity_access_counts(&surviving_seeds)
            .await;

        Ok(ContextResult {
            entities: all_entities,
            facts: all_facts,
            scores: normalized,
        })
    }
}

/// Comparator for `contextualize()` Step 4's seed ranking: RRF score
/// descending, entity id ascending as a deterministic secondary tie-break.
///
/// Extracted as a standalone free function (rather than left as an inline
/// closure) so its determinism is directly unit-testable without a full
/// async DB round trip — see `mod tests` below. Two entities can score
/// identically (e.g. each present in only one of fts_hits/vector_hits, at
/// the same rank position, under the default equal bm25/vector weights);
/// `rrf_scores` is a `HashMap`, so without a total-order secondary key the
/// tied pair's relative order depended on `HashMap` iteration order — which
/// varies per `HashMap` instance (fresh `RandomState` per `HashMap::new()`
/// call), producing run-to-run ranking jitter on byte-identical input
/// (recall-ranking nondeterminism bug). `id` is unique per entity, so it
/// alone gives a total order — no relevance signal is implied by it.
fn score_desc_id_asc(a: &(String, f32), b: &(String, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

/// recall-v2 Phase 5 (Decision 6): partition seed ids by their POST-BOOST score
/// against `floor` — a seed survives iff `scores[seed] >= floor`.
///
/// Gates on the boosted score (the value in `normalized` AFTER the
/// graph-degree/temporal boosts have been applied), NOT the pre-boost RRF rank:
/// a seed a boost lifted above the floor survives even when a higher-RRF but
/// UNBOOSTED seed falls below it. `floor <= 0.0` (the default) keeps every seed
/// — a true no-op. Survivors preserve the input `seed_ids` order.
///
/// Extracted as a free fn so the post-boost-gating property is directly
/// unit-testable without a full async recall round-trip.
fn floor_survivors(seed_ids: &[String], scores: &HashMap<String, f32>, floor: f32) -> Vec<String> {
    if floor <= 0.0 {
        return seed_ids.to_vec();
    }
    seed_ids
        .iter()
        .filter(|id| scores.get(*id).copied().unwrap_or(0.0) >= floor)
        .cloned()
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{floor_survivors, score_desc_id_asc, ContextualizeParams};
    use crate::core::config::SearchConfig;
    use crate::core::graph::{FactInsert, InsertEntityParams};
    use crate::core::ingest::SimpleGraph;
    use chrono::Utc;

    // recall-v2 Phase 4: the fan-out cap + neighbour decay are config-driven now
    // (`SearchConfig`). `SimpleGraph::open_in_memory_simple` uses the DEFAULT
    // config, so these tests assert against the default values.
    fn default_fan_out_cap() -> usize {
        SearchConfig::default().expansion_fan_out_cap
    }
    fn default_neighbour_decay() -> f32 {
        SearchConfig::default().neighbour_score_decay
    }

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
        // score is deterministically `neighbour_score_decay * 1.0`.
        let decay = default_neighbour_decay();
        assert!(
            (leaf_score - decay).abs() < 1e-6,
            "expected leaf_score ({leaf_score}) == neighbour_score_decay ({decay})"
        );
    }

    /// TD-066 Change 1 / recall-v2 Phase 4: a hub seed's 1-hop expansion must not
    /// flood the result set past `expansion_fan_out_cap` genuine neighbours, even
    /// when the seed is connected to far more entities than that. Reads the cap
    /// from the default `SearchConfig` (Phase 4 promoted the const to config).
    #[tokio::test]
    async fn test_contextualize_expansion_fanout_capped() {
        let rql = SimpleGraph::open_in_memory_simple().await.unwrap();
        let now = Utc::now();
        let cap = default_fan_out_cap();

        rql.graph
            .insert_entity(InsertEntityParams {
                id: "hubcap",
                entity_type_id: 0,
                properties: serde_json::json!({"name": "Hubcap"}),
            })
            .await
            .unwrap();

        let leaf_count = cap + 4;
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

        // hubcap (1 seed) + at most `cap` genuine neighbours, even though hubcap
        // is connected to `leaf_count` (> cap) entities.
        assert!(
            ctx.entities.len() <= 1 + cap,
            "fan-out cap violated: {} entities returned for a seed with {leaf_count} \
             neighbours (cap = {cap})",
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

    /// recall-v2 Phase 2b DoD ("Counters live") + Quinn TEST-001: drive the real
    /// producer (`contextualize`) and assert the new scoring observability actually
    /// emits — a pure unit test of `axis_reorders` proves the LOGIC but not that
    /// `context.rs` WIRES it (feedback_test_the_producer_not_the_callback). At
    /// default config the temporal axis is off (weight 0) and graph-degree is live
    /// (0.05), which pins the two axes' weight-zero counters in opposite states.
    #[tokio::test]
    async fn test_contextualize_emits_recall_v2_scoring_counters() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let rql = setup_graph_with_data().await;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let guard = metrics::set_default_local_recorder(&recorder);
        // "Acme" matches acme's properties in FTS → a real seed → the boost loop
        // + post-loop counter block run.
        let _ctx = rql.contextualize(ctx_params("Acme")).await.unwrap();
        drop(guard);

        let snapshot = snapshotter.snapshot().into_vec();
        let counter_named = |name: &str| -> Option<(u64, Vec<(String, String)>)> {
            snapshot.iter().find_map(|(k, _, _, v)| {
                if k.key().name() == name {
                    if let DebugValue::Counter(c) = v {
                        let labels = k
                            .key()
                            .labels()
                            .map(|l| (l.key().to_string(), l.value().to_string()))
                            .collect();
                        return Some((*c, labels));
                    }
                }
                None
            })
        };

        // Default temporal_weight = 0.0 → axis OFF → its weight-zero counter fires.
        let (tw_zero, _) = counter_named("kremory.search.temporal_weight_zero_total")
            .expect("temporal_weight_zero_total must fire at default config (axis off)");
        assert_eq!(tw_zero, 1, "weight-zero counter fires once per recall");

        // Default graph_degree_weight = 0.05 (> 0) → axis LIVE → its weight-zero
        // counter must NOT fire (proves the default preserves the live axis).
        assert!(
            counter_named("kremory.search.graph_degree_weight_zero_total").is_none(),
            "graph_degree_weight_zero_total must NOT fire at default (0.05 > 0 = axis live)"
        );

        // The per-axis reorder counter must be wired into the producer, carrying a
        // boolean `changed` label (the honest pre-judge gate).
        let (_, degree_labels) = counter_named("kremory.search.graph_degree_reorder_total")
            .expect("graph_degree_reorder_total must be emitted by contextualize");
        let changed = degree_labels
            .iter()
            .find(|(k, _)| k == "changed")
            .expect("graph_degree_reorder_total must carry a `changed` label");
        assert!(
            changed.1 == "true" || changed.1 == "false",
            "`changed` label must be a boolean string, got {:?}",
            changed.1
        );
    }

    // === Recall-ranking nondeterminism fix: `score_desc_id_asc` determinism ===
    //
    // `contextualize()`'s Step 4 seed ranking builds `ranked` from
    // `rrf_scores: HashMap<String, f32>` — an entity present in only one of
    // fts_hits/vector_hits at the same rank position as another entity in
    // the other list (common under the default equal 0.5/0.5 bm25/vector
    // weights) scores identically. Pre-fix, `ranked.sort_by` had no
    // secondary key, so a tied pair's relative order depended on
    // `HashMap`'s iteration order — which varies per `HashMap` instance
    // (fresh `RandomState` on every `HashMap::new()` call), producing
    // run-to-run ranking jitter on byte-identical input even within a
    // single process. `SimpleGraph`'s `NullEmbeddingProvider` always embeds
    // to an all-zero vector regardless of query text, which makes
    // `vector_search` return zero hits (cosine distance against a
    // zero-magnitude vector is NULL, and NULL-distance rows are skipped) —
    // so a genuine cross-list RRF score tie cannot be forced through the
    // full async `contextualize()` round trip on this harness. Per the
    // task's explicit fallback, these tests instead prove the extracted
    // comparator itself (now the single source of truth `contextualize()`
    // sorts with) is a deterministic total order — score descending, id
    // ascending on ties — which is the property that makes the seed
    // ranking reproducible regardless of what order `HashMap` iteration
    // happens to hand it the tied pair.

    /// Distinct scores: the comparator must never fall through to the id
    /// tie-break — higher score always sorts first.
    #[test]
    fn score_desc_id_asc_orders_by_score_when_distinct() {
        let mut v = [
            ("zzz".to_owned(), 0.2_f32),
            ("aaa".to_owned(), 0.9_f32),
            ("mmm".to_owned(), 0.5_f32),
        ];
        v.sort_by(score_desc_id_asc);
        let ids: Vec<&str> = v.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["aaa", "mmm", "zzz"],
            "distinct scores must sort strictly by score descending, \
             independent of id"
        );
    }

    /// Equal scores: the comparator must fall back to id ascending — a
    /// stable, reproducible order, not an arbitrary one.
    #[test]
    fn score_desc_id_asc_breaks_ties_by_id_ascending() {
        let mut v = [
            ("zeta".to_owned(), 0.5_f32),
            ("alpha".to_owned(), 0.5_f32),
            ("mike".to_owned(), 0.5_f32),
        ];
        v.sort_by(score_desc_id_asc);
        let ids: Vec<&str> = v.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["alpha", "mike", "zeta"],
            "tied scores must break by entity id ascending"
        );
    }

    /// The property that actually fixes the bug: for a fixed *set* of
    /// (id, tied-score) pairs, sorting is independent of the *insertion
    /// order* the caller hands in — exactly what varies between calls when
    /// the input Vec is collected from a `HashMap` with a freshly-seeded
    /// `RandomState` (as `contextualize()` Step 4 does). Every permutation
    /// of the same tied set must sort to the identical id-ascending output.
    #[test]
    fn score_desc_id_asc_deterministic_across_input_permutations() {
        let base: Vec<(String, f32)> = vec![
            ("delta".to_owned(), 0.7_f32),
            ("bravo".to_owned(), 0.7_f32),
            ("foxtrot".to_owned(), 0.7_f32),
            ("charlie".to_owned(), 0.3_f32), // distinct, lower score
            ("alpha".to_owned(), 0.7_f32),
        ];
        let expected: Vec<&str> = vec!["alpha", "bravo", "delta", "foxtrot", "charlie"];

        // Simulate several distinct "HashMap iteration orders" by feeding in
        // several different permutations of the same underlying set.
        let permutations: Vec<Vec<(String, f32)>> = vec![
            base.clone(),
            {
                let mut p = base.clone();
                p.reverse();
                p
            },
            vec![
                base[2].clone(),
                base[0].clone(),
                base[4].clone(),
                base[3].clone(),
                base[1].clone(),
            ],
            vec![
                base[3].clone(),
                base[4].clone(),
                base[1].clone(),
                base[2].clone(),
                base[0].clone(),
            ],
        ];

        for (i, perm) in permutations.into_iter().enumerate() {
            let mut sorted = perm;
            sorted.sort_by(score_desc_id_asc);
            let ids: Vec<&str> = sorted.iter().map(|(id, _)| id.as_str()).collect();
            assert_eq!(
                ids, expected,
                "permutation {i} sorted to a different order — ranking is \
                 not deterministic w.r.t. input order"
            );
        }
    }

    // === recall-v2 Phase 5 (Decision 6): floor threshold + increment ordering ===

    /// STRUCTURAL ordering proof: the floor gates on the POST-boost score, not
    /// the pre-boost RRF rank. "promoted" has a LOWER base than "unpromoted" but
    /// a boost lifted its post-boost score above the floor, while "unpromoted"
    /// (higher base, no boost) stayed below — so the floor keeps "promoted" and
    /// drops the higher-base "unpromoted". The map passed in IS the post-boost
    /// `normalized` (contextualize applies the floor AFTER the boost loop).
    #[test]
    fn floor_survivors_gates_on_post_boost_score_not_base() {
        use std::collections::HashMap;
        let scores: HashMap<String, f32> = [
            ("promoted".to_owned(), 0.55_f32),
            ("unpromoted".to_owned(), 0.45_f32),
        ]
        .into();
        // Input order = descending RRF base (unpromoted ranked above promoted).
        let seeds = vec!["unpromoted".to_owned(), "promoted".to_owned()];
        assert_eq!(
            floor_survivors(&seeds, &scores, 0.5),
            vec!["promoted".to_owned()],
            "the boost-promoted seed must survive; the higher-base unboosted seed \
             must be dropped — floor gates on post-boost score"
        );
    }

    #[test]
    fn floor_survivors_non_positive_floor_keeps_all() {
        use std::collections::HashMap;
        let scores: HashMap<String, f32> =
            [("a".to_owned(), 0.1_f32), ("b".to_owned(), 0.9_f32)].into();
        let seeds = vec!["a".to_owned(), "b".to_owned()];
        assert_eq!(floor_survivors(&seeds, &scores, 0.0), seeds, "floor 0 = no-op");
        assert_eq!(
            floor_survivors(&seeds, &scores, -1.0),
            seeds,
            "negative floor = no-op"
        );
    }

    #[test]
    fn floor_survivors_above_all_drops_everything() {
        use std::collections::HashMap;
        let scores: HashMap<String, f32> =
            [("a".to_owned(), 0.9_f32), ("b".to_owned(), 1.0_f32)].into();
        let seeds = vec!["a".to_owned(), "b".to_owned()];
        assert!(
            floor_survivors(&seeds, &scores, 1.5).is_empty(),
            "a floor above every post-boost score drops all seeds"
        );
    }

    /// Single-increment spike (RISK-002): the access-count increment moved to
    /// AFTER the boost loop must still fire EXACTLY ONCE per recalled seed.
    #[tokio::test]
    async fn test_contextualize_default_floor_increments_surviving_seed_once() {
        let rql = setup_graph_with_data().await; // default config → floor 0 = no-op
        let _ = rql.contextualize(ctx_params("Acme")).await.unwrap();
        let acme = rql
            .graph
            .get_entity("acme")
            .await
            .unwrap()
            .expect("acme must exist");
        assert_eq!(
            acme.access_count, 1,
            "recalled seed incremented exactly once post-move (RISK-002 preserved)"
        );
    }

    /// Wiring proof: a floor above the max normalised score (1.0) drops the seed
    /// from BOTH the output entities and scores, AND — because increment now runs
    /// on survivors only — the dropped seed is NOT counted as recalled
    /// (access_count stays 0).
    #[tokio::test]
    async fn test_contextualize_high_floor_drops_seed_and_skips_increment() {
        let rql = SimpleGraph::open_in_memory_with_search_config(|s| s.floor_threshold = 1.5)
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

        let ctx = rql.contextualize(ctx_params("Acme")).await.unwrap();
        assert!(
            !ctx.entities.iter().any(|e| e.id == "acme"),
            "a seed scoring below the floor must be dropped from output entities"
        );
        assert!(
            !ctx.scores.contains_key("acme"),
            "a dropped seed must be absent from scores too"
        );
        let acme = rql
            .graph
            .get_entity("acme")
            .await
            .unwrap()
            .expect("acme row still exists in the graph");
        assert_eq!(
            acme.access_count, 0,
            "a floored seed must NOT be counted as recalled (survivors-only increment)"
        );
    }
}
