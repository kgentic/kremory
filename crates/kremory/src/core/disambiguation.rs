//! L4 graph-time entity disambiguation (Cognee pattern).
//!
//! Runs immediately after extraction and before DB insertion.  Each new entity
//! name is compared against existing entities in the same `group_id` using
//! cosine similarity on their embeddings.  Based on the similarity score one of
//! three outcomes is selected:
//!
//! | Threshold                              | Action                                        |
//! |----------------------------------------|-----------------------------------------------|
//! | `sim >= L4_MERGE_THRESHOLD` (0.95)     | **MERGE** — reuse existing entity_id          |
//! | `L4_POTENTIAL_ALIAS_THRESHOLD` ≤ sim < `L4_MERGE_THRESHOLD` (0.70–0.95) | **ALIAS** — insert new entity + `potential_alias` fact |
//! | `sim < L4_POTENTIAL_ALIAS_THRESHOLD`   | **NEW** — insert as completely new entity     |
//!
//! ## §3 — `potential_alias` reserved predicate
//!
//! `RESERVED_PREDICATE_POTENTIAL_ALIAS` is the predicate used for the meta-edge
//! that records a probable-alias relationship between two entities.  It must be
//! treated as a meta-edge during dream-phase consolidation (L7): the dream phase
//! resolves confirmed aliases into permanent merges and drops `potential_alias`
//! edges whose similarity has fallen below `L4_REVOKE_THRESHOLD` (0.50).
//!
//! ## Design rationale
//!
//! Pattern adapted from Cognee's `post_extraction_canonicalization` POC.
//! Key differences from a naïve dedup:
//!
//! - Operates on *embedding* similarity, not string equality — catches
//!   "Alice Johnson" vs "Alice J." style variants that string edit-distance misses.
//! - Uses `vector_search_entities_no_count` (doesn't inflate `access_count`).
//! - `potential_alias` is stored as a **fact** (not an episodic edge) so it
//!   participates in temporal validity and can be invalidated by the dream phase.
//! - Thresholds are empirically sourced from Cognee's research; adjustable via
//!   the named constants below.

use chrono::Utc;
use metrics::counter;
use tracing;

use crate::core::error::Result;
use crate::core::provider::EmbeddingProvider;
use crate::core::schema::TemporalGraph;
use crate::core::search::SearchFilters;

// ─── Threshold constants ──────────────────────────────────────────────────────

/// Cosine similarity at or above which a new entity is merged into an existing
/// one instead of being inserted as a new row.
///
/// Source: Cognee `post_extraction_canonicalization` POC (empirically derived).
pub const L4_MERGE_THRESHOLD: f32 = 0.95;

/// Cosine similarity at or above which a new entity is inserted but linked to
/// the most similar existing entity via a `potential_alias` fact edge.
///
/// Source: Cognee `post_extraction_canonicalization` POC (empirically derived).
pub const L4_POTENTIAL_ALIAS_THRESHOLD: f32 = 0.70;

/// Cosine similarity below which the dream phase (L7) will invalidate a
/// `potential_alias` fact as unconfirmed.
///
/// Kept here for co-location with the other L4 threshold constants; actually
/// consumed by the dream-phase reclassification logic (Phase 8 scope).
pub const L4_REVOKE_THRESHOLD: f32 = 0.50;

// ─── Reserved predicates ─────────────────────────────────────────────────────

/// Predicate for the meta-edge that records a probable alias relationship.
///
/// Stored as a `facts` row (not an episodic edge) so it can be invalidated
/// by the dream phase and participates in temporal validity.
///
/// ## Dream-phase awareness
///
/// During L7 dream-phase reclassification, edges with this predicate are
/// treated as meta-edges rather than domain facts:
/// - `sim >= L4_MERGE_THRESHOLD` confirmed by re-embedding → promote to hard merge.
/// - `sim < L4_REVOKE_THRESHOLD` → invalidate (fact `expired_at` set).
/// - Otherwise → retain as unresolved alias candidate.
pub const RESERVED_PREDICATE_POTENTIAL_ALIAS: &str = "potential_alias";

// ─── Disambiguation result ────────────────────────────────────────────────────

/// The outcome of a single entity disambiguation.
#[derive(Debug, Clone, PartialEq)]
pub enum DisambiguationOutcome {
    /// The new entity is sufficiently similar to an existing one to be treated
    /// as the same real-world entity.  The caller should use `existing_id`
    /// as the canonical entity id rather than inserting a new row.
    Merge {
        /// The existing entity's id that the new name maps to.
        existing_id: String,
        /// Cosine similarity score (0.0–1.0).
        similarity: f32,
    },
    /// The new entity is probably an alias of an existing one.
    /// The caller should insert a new entity row AND insert a `potential_alias`
    /// fact from `new_id → existing_id` with `confidence = similarity`.
    PotentialAlias {
        /// The most similar existing entity.
        existing_id: String,
        /// Cosine similarity score (0.0–1.0).
        similarity: f32,
    },
    /// No sufficiently similar entity found — insert as a completely new entity.
    New,
}

// ─── Core disambiguation function ────────────────────────────────────────────

/// Disambiguate a newly extracted entity name against existing entities in the
/// same `group_id`.
///
/// ## Algorithm
///
/// 1. Embed `entity_name` using the configured embedder.
/// 2. Vector-search the `entities` table for the nearest neighbour in `group_id`
///    (top-1, cosine similarity, `no_count` variant so `access_count` stays clean).
/// 3. Apply the threshold ladder:
///    - `sim >= L4_MERGE_THRESHOLD` → `Merge`
///    - `L4_POTENTIAL_ALIAS_THRESHOLD <= sim < L4_MERGE_THRESHOLD` → `PotentialAlias`
///    - `sim < L4_POTENTIAL_ALIAS_THRESHOLD` → `New`
///
/// ## No-op conditions
///
/// Returns `DisambiguationOutcome::New` immediately when:
/// - `group_id` is `None` (global namespace — disambiguation not applied).
/// - The embedder returns an empty vector (null embedder / test stub).
/// - No existing entities found in the group.
pub async fn disambiguate<Emb: EmbeddingProvider>(
    entity_name: &str,
    group_id: Option<&str>,
    graph: &TemporalGraph,
    embedder: &Emb,
) -> Result<DisambiguationOutcome> {
    // No-op: global namespace or empty name.
    let Some(gid) = group_id else {
        return Ok(DisambiguationOutcome::New);
    };
    if entity_name.trim().is_empty() {
        return Ok(DisambiguationOutcome::New);
    }

    // Step 1: embed the new entity name.
    let embedding = embedder.embed(entity_name).await?;
    if embedding.is_empty() {
        // Degenerate embedder returned a zero-length slice — skip disambiguation.
        return Ok(DisambiguationOutcome::New);
    }
    // Zero-magnitude vectors (e.g. NullEmbeddingProvider) produce NULL cosine
    // distance in libsql, which causes all search hits to be skipped in
    // `vector_search_with_index`.  The no-hits branch below correctly returns
    // `New` in that case, so no explicit zero-magnitude guard is needed here.

    // Step 2: nearest-neighbour search scoped to group_id.
    // `no_count` variant: does not increment `access_count` — disambiguation is
    // a read-only probe, not a recall event.
    let filters = SearchFilters::for_group(gid);
    let hits = graph
        .vector_search_entities_no_count(&embedding, 1, &filters)
        .await?;

    let Some(top_hit) = hits.into_iter().next() else {
        // No existing entities in this group — definitely new.
        return Ok(DisambiguationOutcome::New);
    };

    // Step 3: convert score → similarity.
    // `vector_search_with_index` stores `score = -distance` (cosine distance).
    // cosine_similarity = 1.0 - cosine_distance = 1.0 - (-score) = 1.0 + score.
    // Clamp to [0.0, 1.0] to defend against floating-point edge cases.
    let similarity = (1.0_f32 + top_hit.score as f32).clamp(0.0, 1.0);
    let existing_id = top_hit.item.id.clone();

    tracing::debug!(
        target: "kremory.l4",
        entity_name,
        existing_id = %existing_id,
        similarity,
        "kremory.l4.similarity_probe"
    );

    // Step 4: threshold ladder.
    if similarity >= L4_MERGE_THRESHOLD {
        counter!("kremory.l4.merge_total").increment(1);
        tracing::info!(
            target: "kremory.l4",
            entity_name,
            existing_id = %existing_id,
            similarity,
            "kremory.l4.merge"
        );
        Ok(DisambiguationOutcome::Merge {
            existing_id,
            similarity,
        })
    } else if similarity >= L4_POTENTIAL_ALIAS_THRESHOLD {
        counter!("kremory.l4.potential_alias_total").increment(1);
        tracing::info!(
            target: "kremory.l4",
            entity_name,
            existing_id = %existing_id,
            similarity,
            "kremory.l4.potential_alias"
        );
        Ok(DisambiguationOutcome::PotentialAlias {
            existing_id,
            similarity,
        })
    } else {
        counter!("kremory.l4.new_entity_total").increment(1);
        tracing::debug!(
            target: "kremory.l4",
            entity_name,
            similarity,
            "kremory.l4.new_entity"
        );
        Ok(DisambiguationOutcome::New)
    }
}

/// Provenance context for a `potential_alias` fact: the episode that triggered
/// the alias and the namespace (group_id) it belongs to.
///
/// Bundled into a single struct so `insert_potential_alias_fact` stays below
/// the workspace clippy `too_many_arguments` limit (≤5).
#[derive(Debug, Clone, Copy)]
pub struct AliasProvenance<'a> {
    /// Episode id that originated this alias discovery.  Forwarded to the
    /// `facts` row as `source_episode_id` for provenance tracking.
    pub source_episode_id: Option<i64>,
    /// Namespace the entities belong to.  Must match `group_id` on both
    /// entity rows so the fact is scoped to the same namespace.
    pub group_id: Option<&'a str>,
}

/// Insert a `potential_alias` fact edge from `new_entity_id` → `existing_id`
/// with `confidence = similarity`.
///
/// The edge is stored as a standard `facts` row so it participates in temporal
/// validity and can be invalidated by the dream phase (L7).
///
/// Returns the new `fact_id` on success.  A `Duplicate` error from
/// `insert_fact_with_group` is silently swallowed — the alias was already
/// recorded in a prior ingest and does not need to be duplicated.
pub async fn insert_potential_alias_fact(
    graph: &TemporalGraph,
    new_entity_id: &str,
    existing_id: &str,
    similarity: f32,
    provenance: AliasProvenance<'_>,
) -> Result<Option<i64>> {
    let now = Utc::now();
    let result = graph
        .insert_fact_with_group(
            new_entity_id,
            RESERVED_PREDICATE_POTENTIAL_ALIAS,
            Some(existing_id),
            None,
            now,
            f64::from(similarity),
            provenance.source_episode_id,
            provenance.group_id,
            None,
        )
        .await;

    match result {
        Ok(id) => {
            tracing::debug!(
                target: "kremory.l4",
                new_entity_id,
                existing_id,
                similarity,
                fact_id = id,
                "kremory.l4.potential_alias_fact_inserted"
            );
            Ok(Some(id))
        }
        Err(crate::core::error::Error::Duplicate { .. }) => {
            // Already recorded — idempotent.
            tracing::debug!(
                target: "kremory.l4",
                new_entity_id,
                existing_id,
                "kremory.l4.potential_alias_fact_duplicate_skipped"
            );
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

// ─── Dream-phase alias resolution (7.C) ──────────────────────────────────────

/// Resolve all pending `potential_alias` facts in `group_id` accumulated during
/// ingest-time L4 disambiguation.
///
/// For each non-expired `potential_alias` fact in the group the current cosine
/// similarity between the subject and object entity embeddings is recomputed.
/// Three outcomes are possible:
///
/// | Similarity                    | Action                                         |
/// |-------------------------------|------------------------------------------------|
/// | `>= L4_MERGE_THRESHOLD` (0.95) | **MERGE** — invalidate alias fact (confirmed, L5 will handle structural merge on next canonicalization pass). |
/// | `< L4_REVOKE_THRESHOLD` (0.50) | **REVOKE** — invalidate alias fact (false alarm). |
/// | otherwise                     | **KEEP** — leave the alias fact in place.      |
///
/// Returns the total number of alias facts that were resolved (merged or revoked).
///
/// ## No-op conditions
///
/// - Either endpoint entity has no embedding (`NULL` in DB) → skip the fact.
/// - `object_id` is `None` on the fact row → skip (malformed alias fact).
pub async fn resolve_pending_aliases(graph: &TemporalGraph, group_id: &str) -> Result<usize> {
    let alias_facts = graph.get_alias_facts_in_group(group_id).await?;
    if alias_facts.is_empty() {
        return Ok(0);
    }

    let now = Utc::now();
    let mut resolved = 0usize;

    for fact in &alias_facts {
        // Alias fact must have an object_id pointing to the canonical entity.
        let Some(ref object_id) = fact.object_id else {
            tracing::debug!(
                target: "kremory.l7",
                fact_id = fact.id,
                "kremory.l7.resolve_aliases.skip_no_object_id"
            );
            continue;
        };

        // Re-compute cosine similarity between subject and object embeddings via SQL.
        // `vector_distance_cos` returns NULL when either vector has zero magnitude.
        let mut rows = graph
            .conn
            .query(
                "SELECT vector_distance_cos(a.embedding, b.embedding) \
                 FROM entities a, entities b \
                 WHERE a.id = ?1 AND b.id = ?2",
                libsql::params![fact.subject_id.clone(), object_id.clone()],
            )
            .await?;

        let Some(row) = rows.next().await? else {
            // No row returned — at least one entity is missing.
            continue;
        };

        let distance_opt: Option<f64> = row.get(0)?;
        let Some(distance) = distance_opt else {
            // NULL from vector_distance_cos — zero-magnitude embedding, skip.
            tracing::debug!(
                target: "kremory.l7",
                fact_id = fact.id,
                subject_id = %fact.subject_id,
                object_id = %object_id,
                "kremory.l7.resolve_aliases.skip_null_distance"
            );
            continue;
        };

        let similarity = (1.0_f32 - distance as f32).clamp(0.0, 1.0);

        if similarity >= L4_MERGE_THRESHOLD {
            // Confirmed alias — invalidate the fact; L5 will merge structurally.
            graph.invalidate_fact(fact.id, now).await?;
            counter!("kremory.l7.resolve_aliases_total", "outcome" => "merged").increment(1);
            tracing::info!(
                target: "kremory.l7",
                fact_id = fact.id,
                subject_id = %fact.subject_id,
                object_id = %object_id,
                similarity,
                "kremory.l7.resolve_aliases.merged"
            );
            resolved += 1;
        } else if similarity < L4_REVOKE_THRESHOLD {
            // False alarm — revoke the alias fact.
            graph.invalidate_fact(fact.id, now).await?;
            counter!("kremory.l7.resolve_aliases_total", "outcome" => "revoked").increment(1);
            tracing::info!(
                target: "kremory.l7",
                fact_id = fact.id,
                subject_id = %fact.subject_id,
                object_id = %object_id,
                similarity,
                "kremory.l7.resolve_aliases.revoked"
            );
            resolved += 1;
        } else {
            // Mid-range similarity — keep for now.
            counter!("kremory.l7.resolve_aliases_total", "outcome" => "kept").increment(1);
            tracing::debug!(
                target: "kremory.l7",
                fact_id = fact.id,
                similarity,
                "kremory.l7.resolve_aliases.kept"
            );
        }
    }

    Ok(resolved)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::Result as KResult;
    #[cfg(any(test, feature = "test-utils"))]
    use crate::core::provider::NullEmbeddingProvider;
    use crate::core::schema::TemporalGraph;

    // ── Threshold constants ───────────────────────────────────────────────────

    #[test]
    fn l4_thresholds_are_ordered() {
        const {
            assert!(
                L4_REVOKE_THRESHOLD < L4_POTENTIAL_ALIAS_THRESHOLD,
                "revoke threshold must be below alias threshold"
            );
            assert!(
                L4_POTENTIAL_ALIAS_THRESHOLD < L4_MERGE_THRESHOLD,
                "alias threshold must be below merge threshold"
            );
            assert!(
                L4_MERGE_THRESHOLD <= 1.0,
                "merge threshold must be at most 1.0"
            );
        }
    }

    #[test]
    fn reserved_predicate_is_snake_case() {
        assert!(
            RESERVED_PREDICATE_POTENTIAL_ALIAS
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_'),
            "RESERVED_PREDICATE_POTENTIAL_ALIAS must be snake_case"
        );
        assert_eq!(RESERVED_PREDICATE_POTENTIAL_ALIAS, "potential_alias");
    }

    // ── Outcome classification from simulated scores ──────────────────────────

    /// Test helper: classify a raw similarity score using the threshold ladder.
    /// Mirrors the runtime logic without hitting the DB.
    fn classify(similarity: f32) -> DisambiguationOutcome {
        if similarity >= L4_MERGE_THRESHOLD {
            DisambiguationOutcome::Merge {
                existing_id: "existing".to_string(),
                similarity,
            }
        } else if similarity >= L4_POTENTIAL_ALIAS_THRESHOLD {
            DisambiguationOutcome::PotentialAlias {
                existing_id: "existing".to_string(),
                similarity,
            }
        } else {
            DisambiguationOutcome::New
        }
    }

    #[test]
    fn similarity_0_97_produces_merge() {
        let outcome = classify(0.97);
        assert!(
            matches!(outcome, DisambiguationOutcome::Merge { similarity, .. } if (similarity - 0.97).abs() < 1e-6),
            "sim=0.97 must produce Merge, got {outcome:?}"
        );
    }

    #[test]
    fn similarity_0_82_produces_potential_alias() {
        let outcome = classify(0.82);
        assert!(
            matches!(outcome, DisambiguationOutcome::PotentialAlias { similarity, .. } if (similarity - 0.82).abs() < 1e-6),
            "sim=0.82 must produce PotentialAlias, got {outcome:?}"
        );
    }

    #[test]
    fn similarity_0_45_produces_new() {
        let outcome = classify(0.45);
        assert!(
            matches!(outcome, DisambiguationOutcome::New),
            "sim=0.45 must produce New, got {outcome:?}"
        );
    }

    #[test]
    fn merge_threshold_boundary_exact_is_merge() {
        // Exactly at the merge threshold → Merge.
        let outcome = classify(L4_MERGE_THRESHOLD);
        assert!(
            matches!(outcome, DisambiguationOutcome::Merge { .. }),
            "sim == L4_MERGE_THRESHOLD must be Merge"
        );
    }

    #[test]
    fn alias_threshold_boundary_exact_is_alias() {
        // Exactly at the alias threshold → PotentialAlias (not New).
        let outcome = classify(L4_POTENTIAL_ALIAS_THRESHOLD);
        assert!(
            matches!(outcome, DisambiguationOutcome::PotentialAlias { .. }),
            "sim == L4_POTENTIAL_ALIAS_THRESHOLD must be PotentialAlias"
        );
    }

    // ── No-op conditions (null embedder path) ─────────────────────────────────

    /// With a NullEmbeddingProvider (zero-magnitude vectors) the cosine distance
    /// is NULL in libsql; all hits are skipped → no hits → disambiguate() returns New.
    #[tokio::test]
    async fn null_embedder_returns_new() -> KResult<()> {
        let graph = TemporalGraph::open_in_memory().await?;
        let embedder = NullEmbeddingProvider { dim: 384 };
        let outcome = disambiguate("Alice", Some("test-group"), &graph, &embedder).await?;
        assert_eq!(
            outcome,
            DisambiguationOutcome::New,
            "NullEmbeddingProvider must produce New (zero-vector cosine is NULL → no hits)"
        );
        Ok(())
    }

    #[tokio::test]
    async fn none_group_id_returns_new() -> KResult<()> {
        let graph = TemporalGraph::open_in_memory().await?;
        let embedder = NullEmbeddingProvider { dim: 384 };
        let outcome = disambiguate("Alice", None, &graph, &embedder).await?;
        assert_eq!(
            outcome,
            DisambiguationOutcome::New,
            "None group_id must skip disambiguation and return New"
        );
        Ok(())
    }

    #[tokio::test]
    async fn empty_name_returns_new() -> KResult<()> {
        let graph = TemporalGraph::open_in_memory().await?;
        let embedder = NullEmbeddingProvider { dim: 384 };
        let outcome = disambiguate("   ", Some("g"), &graph, &embedder).await?;
        assert_eq!(
            outcome,
            DisambiguationOutcome::New,
            "blank entity name must return New"
        );
        Ok(())
    }
}
