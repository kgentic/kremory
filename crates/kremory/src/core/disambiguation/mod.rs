//! L4 graph-time entity disambiguation (Cognee pattern).
//!
//! Runs immediately after extraction and before DB insertion.  Each new entity
//! name is compared against existing entities in the same `group_id` using
//! cosine similarity on their embeddings.  Based on the similarity score one of
//! three outcomes is selected:
//!
//! | Threshold + lexical gate (ADR-057 / TD-098)                            | Action |
//! |------------------------------------------------------------------------|--------|
//! | `sim >= L4_MERGE_THRESHOLD` (0.95) **AND** names lexically compatible   | **MERGE** — reuse existing entity_id |
//! | `L4_POTENTIAL_ALIAS_THRESHOLD` ≤ sim < 0.95 (0.70–0.95) **AND** names lexically compatible | **ALIAS** — insert new entity + `potential_alias` fact |
//! | cosine in the merge/alias band **but names lexically INCOMPATIBLE**     | **NEW** — anisotropy noise; no merge, no *persistent* false alias (TD-098: the alias arm was the missed 5th cosine-alone site of ADR-057's invariant) |
//! | `sim < L4_POTENTIAL_ALIAS_THRESHOLD`                                    | **NEW** — insert as completely new entity |
//!
//! The lexical gate (`lexical::names_lexically_compatible`, ADR-057 Jaccard 0.5) is
//! the deterministic, embedder-independent signal that makes BOTH the destructive
//! merge AND the (persistent) alias-creation robust to a weak/anisotropic consumer
//! embedder (BYOM). Cosine alone can authorize neither.
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

pub(crate) mod lexical;
pub(crate) use lexical::names_lexically_compatible;

use chrono::Utc;
use metrics::counter;
use tracing;

use crate::core::error::Result;
use crate::core::graph::FactInsert;
use crate::core::provider::EmbeddingProvider;
use crate::core::schema::TemporalGraph;
use crate::core::search::{SearchFilters, VectorSearchEntitiesNoCountParams};

// ─── Threshold constants ──────────────────────────────────────────────────────

/// Cosine similarity at or above which a new entity is merged into an existing
/// one instead of being inserted as a new row.
///
/// Source: Cognee `post_extraction_canonicalization` POC (empirically derived).
pub const L4_MERGE_THRESHOLD: f32 = 0.95;

/// Cosine similarity at or above which a new entity is inserted but linked to
/// the most similar existing entity via a `potential_alias` fact edge — **provided
/// the two names are also lexically compatible** (`names_lexically_compatible`,
/// TD-098 / Site #4 of ADR-063). A high-cosine but lexically-incompatible pair is
/// embedder anisotropy noise and is classified `New`, NOT a persistent alias.
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

/// Bundled non-generic parameters for `disambiguate`, args-as-object per TD-042
/// (rust-conventions §too_many_arguments). The generic `embedder: &Emb` stays a
/// lead positional argument.
#[derive(Clone, Copy)]
pub struct DisambiguateParams<'a> {
    /// The newly extracted entity name to disambiguate.
    pub entity_name: &'a str,
    /// Optional namespace scope (`None` = global namespace, disambiguation skipped).
    pub group_id: Option<&'a str>,
    /// The temporal graph to search against.
    pub graph: &'a TemporalGraph,
}

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
    params: DisambiguateParams<'_>,
    embedder: &Emb,
) -> Result<DisambiguationOutcome> {
    let DisambiguateParams {
        entity_name,
        group_id,
        graph,
    } = params;
    // No-op: global namespace or empty name.
    let Some(gid) = group_id else {
        return Ok(DisambiguationOutcome::New);
    };
    if entity_name.trim().is_empty() {
        return Ok(DisambiguationOutcome::New);
    }

    // Step 1: embed the entity name.
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
        .vector_search_entities_no_count(VectorSearchEntitiesNoCountParams {
            query_embedding: &embedding,
            limit: 1,
            filters: &filters,
        })
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

    // Step 4: threshold ladder — delegated to the pure `classify_pair` helper.
    // Metrics and tracing are re-derived here because `classify_pair` must
    // remain counter-free (the corpus benchmark calls it thousands of times
    // and must not touch production counters).
    //
    // ADR-057: `existing_id` is the existing entity's normalized name (entity ids
    // ARE normalized names — ingest_with.rs:1017), so it is the correct lexical
    // comparand AND the `existing_id` forwarded in Merge/PotentialAlias outcomes.
    let outcome = classify_pair(similarity, entity_name, &existing_id);

    match &outcome {
        DisambiguationOutcome::Merge { .. } => {
            counter!("kremory.l4.merge_total").increment(1);
            tracing::info!(
                target: "kremory.l4",
                entity_name,
                existing_id = %existing_id,
                similarity,
                "kremory.l4.merge"
            );
        }
        DisambiguationOutcome::PotentialAlias { .. } => {
            // Post-TD-098 (Site #4): the PotentialAlias arm now fires ONLY for
            // lexically-COMPATIBLE pairs in the [alias, merge) band. A high-cosine but
            // lexically-incompatible pair no longer downgrades to PotentialAlias — it
            // falls to `New` (see that arm's blocked-lexical attribution), so the old
            // `cosine >= L4_MERGE_THRESHOLD` "merge_blocked here" branch can no longer
            // arise in this arm and has moved to `New`.
            counter!("kremory.l4.potential_alias_total").increment(1);
            tracing::info!(
                target: "kremory.l4",
                entity_name,
                existing_id = %existing_id,
                similarity,
                "kremory.l4.potential_alias"
            );
        }
        DisambiguationOutcome::New => {
            // TD-098: `New` is now reachable by THREE routes. Attribute the two
            // lexically-blocked routes so a weak consumer embedder stays observable
            // (observability-first-class, Rule 19). `classify_pair` is counter-free by
            // design, so re-derive the band locally here (mirrors its thresholds).
            if similarity >= L4_MERGE_THRESHOLD {
                // Was ADR-057's "≥0.95 + incompatible → PotentialAlias" downgrade cell;
                // TD-098 discards it as `New` (no persistent false-alias fact). The
                // counter name is kept so existing merge-blocked observability + the
                // l4_l5_real_embedding regression assertion still fire.
                counter!("kremory.l4.merge_blocked_lexical_total").increment(1);
                tracing::info!(
                    target: "kremory.l4",
                    entity_name,
                    existing_id = %existing_id,
                    similarity,
                    "kremory.l4.merge_blocked_lexical"
                );
            } else if similarity >= L4_POTENTIAL_ALIAS_THRESHOLD {
                // TD-098 poisoning fix: a degenerate high-cosine alias-band pair whose
                // names are lexically incompatible is NOT recorded as a persistent
                // `potential_alias` fact — it becomes `New`. New counter distinguishes
                // this blocked case from a genuine low-cosine New.
                counter!("kremory.l4.alias_blocked_lexical_total").increment(1);
                tracing::info!(
                    target: "kremory.l4",
                    entity_name,
                    existing_id = %existing_id,
                    similarity,
                    "kremory.l4.alias_blocked_lexical"
                );
            } else {
                counter!("kremory.l4.new_entity_total").increment(1);
                tracing::debug!(
                    target: "kremory.l4",
                    entity_name,
                    similarity,
                    "kremory.l4.new_entity"
                );
            }
        }
    }

    Ok(outcome)
}

/// Classify a pre-computed cosine score through the L4 threshold ladder and
/// ADR-057 lexical floor. Pure: no metrics, no tracing, no I/O — so the B-on/off
/// benchmark exercises the identical decision path as production.
///
/// `name_b` is the existing entity's normalized name (entity ids ARE normalized
/// names — ingest_with.rs:1017). It is used as both the lexical comparand and the
/// `existing_id` forwarded in `Merge` and `PotentialAlias` outcomes.
///
/// Production `disambiguate` delegates its final decision to this function;
/// benchmarks call this directly — one implementation, no copy.
pub(crate) fn classify_pair(cosine: f32, name_a: &str, name_b: &str) -> DisambiguationOutcome {
    if cosine >= L4_MERGE_THRESHOLD && names_lexically_compatible(name_a, name_b) {
        DisambiguationOutcome::Merge {
            existing_id: name_b.to_owned(),
            similarity: cosine,
        }
    } else if cosine >= L4_POTENTIAL_ALIAS_THRESHOLD && names_lexically_compatible(name_a, name_b) {
        // TD-098 (Site #4 of ADR-063): the PotentialAlias arm was cosine-ALONE — the
        // 5th, missed site of ADR-057's "cosine never authorizes an identity write
        // alone" invariant. ADR-057 line 116 assumed non-destructive alias creation
        // could stay permissive because L7 self-heals; TD-098 proved that assumption
        // FALSE under embedder degeneracy: `nomic-embed-text` returns cosine ≈ 1.0 for
        // unrelated short names (`cos(Ria,Morocco)=1.0000`), so the cosine NEVER drops
        // below L4_REVOKE_THRESHOLD → L7 revocation never fires → a false alias fact
        // persists indefinitely, poisoning graph analyses. So a high-cosine but
        // lexically-incompatible pair is anisotropy noise and must NOT even become a
        // (persistent) PotentialAlias — it falls through to `New`. This supersedes
        // ADR-057's "≥0.95 + incompatible → PotentialAlias" downgrade cell (that ADR's
        // parenthetical about "preserving acronym-variant detection" is also superseded:
        // R1/R3 show L4 cosine does not reliably elevate acronym pairs anyway, and
        // ADR-063 Site #5 is the dedicated acronym/nickname mechanism). Uses the SAME
        // `names_lexically_compatible` helper (ADR-057 Jaccard 0.5) as the 4 sibling
        // sites — no new number (Phase 1 scope). The confidence reject-floor + forced
        // re-confirmation deadline (TD-098 parts 2/3) are spike-gated (S4/S5) → Phase 2.
        DisambiguationOutcome::PotentialAlias {
            existing_id: name_b.to_owned(),
            similarity: cosine,
        }
    } else {
        DisambiguationOutcome::New
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
/// Bundled parameters for [`insert_potential_alias_fact`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
#[derive(Clone, Copy)]
pub struct InsertPotentialAliasFactParams<'a> {
    /// The temporal graph to insert the alias fact into.
    pub graph: &'a TemporalGraph,
    /// The newly inserted entity (subject of the alias edge).
    pub new_entity_id: &'a str,
    /// The existing canonical entity (object of the alias edge).
    pub existing_id: &'a str,
    /// Cosine similarity score, stored as the fact `confidence`.
    pub similarity: f32,
    /// Episode + namespace provenance for the alias fact.
    pub provenance: AliasProvenance<'a>,
}

pub async fn insert_potential_alias_fact(
    params: InsertPotentialAliasFactParams<'_>,
) -> Result<Option<i64>> {
    let InsertPotentialAliasFactParams {
        graph,
        new_entity_id,
        existing_id,
        similarity,
        provenance,
    } = params;
    let now = Utc::now();
    let result = graph
        .insert_fact_with_group(
            FactInsert {
                subject_id: new_entity_id,
                predicate: RESERVED_PREDICATE_POTENTIAL_ALIAS,
                object_id: Some(existing_id),
                object_value: None,
                valid_from: now,
                confidence: f64::from(similarity),
                source_episode_id: provenance.source_episode_id,
                embedding: None,
            },
            provenance.group_id,
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

        // ADR-057: L7 alias-confirmation is the THIRD destructive-merge path (it
        // invalidates the alias fact so L5 performs the structural merge). It must
        // honour the SAME deterministic name gate as L4/L5: a high cosine between
        // lexically-incompatible names is embedder anisotropy, not a confirmed
        // identity. Without this, one dream cycle could re-merge `boston` into
        // `amazon robotics` and re-introduce the TD-080 #2 corruption.
        let names_compatible = names_lexically_compatible(&fact.subject_id, object_id);
        if similarity >= L4_MERGE_THRESHOLD && names_compatible {
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
        } else if similarity < L4_REVOKE_THRESHOLD
            || (similarity >= L4_MERGE_THRESHOLD && !names_compatible)
        {
            // Revoke: a genuine false alarm (low cosine) OR a high-cosine pair the
            // lexical gate rejects as anisotropy (never the same real-world entity).
            if similarity >= L4_MERGE_THRESHOLD && !names_compatible {
                counter!("kremory.l7.merge_blocked_lexical_total").increment(1);
            }
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

    // ── ADR-057: lexical-name compatibility gate ──────────────────────────────
    //
    // Truth table anchored on the VERIFIED anisotropy data
    // (tests/spike_td080_embedder_cosine.rs): every observed false-merge pair
    // shares zero significant tokens and MUST be blocked; true variants MUST pass.

    #[test]
    fn lexical_gate_blocks_unrelated_names() {
        // The exact pairs that nomic-embed-text rated cosine 0.95–1.00 — all unrelated.
        assert!(!names_lexically_compatible("Ria", "Morocco"));
        assert!(!names_lexically_compatible("Ria", "Amazon Robotics"));
        assert!(!names_lexically_compatible("Morocco", "Boston"));
        assert!(!names_lexically_compatible(
            "Northeastern University",
            "Amazon Robotics"
        ));
        // City vs firm sharing ONE token — the token-subset hole; Jaccard 1/3 < 0.5.
        assert!(!names_lexically_compatible(
            "Boston",
            "Boston Consulting Group"
        ));
        // Different orgs sharing one token.
        assert!(!names_lexically_compatible(
            "Amazon Robotics",
            "Amazon Web Services"
        ));
    }

    #[test]
    fn lexical_gate_allows_true_variants_and_idempotent_reingest() {
        // Idempotent re-ingest of the same name (the dominant real merge case).
        assert!(names_lexically_compatible("Ria", "Ria"));
        assert!(names_lexically_compatible(
            "Northeastern University",
            "Northeastern University"
        ));
        // Case / punctuation variants normalize-equal.
        assert!(names_lexically_compatible(
            "Amazon Robotics",
            "amazon robotics"
        ));
        assert!(names_lexically_compatible("Acme, Inc", "acme inc"));
        // True multi-token variant — Jaccard 2/3 = 0.67 ≥ 0.5.
        assert!(names_lexically_compatible(
            "Alice Johnson",
            "Alice Marie Johnson"
        ));
        // First-name → full-name (Jaccard 1/2 = 0.5).
        assert!(names_lexically_compatible("Ria Patel", "Ria"));
    }

    #[test]
    fn lexical_gate_drops_single_initial_noise() {
        // "j" is a single-char (insignificant) token; only "alice" is significant
        // on each side → equal significant-token sets → compatible.
        assert!(names_lexically_compatible("Alice J", "Alice J."));
        // But a bare differing initial cannot rescue unrelated names.
        assert!(!names_lexically_compatible("A Ria", "B Morocco"));
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
        let outcome = disambiguate(
            DisambiguateParams {
                entity_name: "Alice",
                group_id: Some("test-group"),
                graph: &graph,
            },
            &embedder,
        )
        .await?;
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
        let outcome = disambiguate(
            DisambiguateParams {
                entity_name: "Alice",
                group_id: None,
                graph: &graph,
            },
            &embedder,
        )
        .await?;
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
        let outcome = disambiguate(
            DisambiguateParams {
                entity_name: "   ",
                group_id: Some("g"),
                graph: &graph,
            },
            &embedder,
        )
        .await?;
        assert_eq!(
            outcome,
            DisambiguationOutcome::New,
            "blank entity name must return New"
        );
        Ok(())
    }

    // ── D3: classify_pair boundary tests ─────────────────────────────────────
    //
    // Spec: td080-b1-context-embedding-toggle-spec-2026-06-30.md §classify_pair
    // Four cases that must hold independent of embedder choice.

    #[test]
    fn classify_pair_at_merge_threshold_compatible_names_is_merge() {
        // Exactly at L4_MERGE_THRESHOLD with lexically-compatible names → Merge.
        // "alice johnson" / "alice johnson" normalize-equal → compatible.
        let outcome = classify_pair(L4_MERGE_THRESHOLD, "Alice Johnson", "alice johnson");
        assert!(
            matches!(
                outcome,
                DisambiguationOutcome::Merge { ref existing_id, similarity }
                if existing_id == "alice johnson"
                    && (similarity - L4_MERGE_THRESHOLD).abs() < f32::EPSILON
            ),
            "cosine == L4_MERGE_THRESHOLD + compatible names must produce Merge, got {outcome:?}"
        );
    }

    #[test]
    fn classify_pair_at_alias_threshold_is_potential_alias() {
        // Exactly at L4_POTENTIAL_ALIAS_THRESHOLD with lexically-COMPATIBLE names →
        // PotentialAlias (cosine < merge threshold so the merge arm cannot fire).
        // Post-TD-098 the alias arm ALSO requires lexical compatibility, so this test
        // uses a compatible pair ("alice johnson" ⊆ "alice marie johnson", Jaccard
        // 2/3 ≥ 0.5) to exercise the threshold ladder.
        let outcome = classify_pair(
            L4_POTENTIAL_ALIAS_THRESHOLD,
            "alice marie johnson",
            "alice johnson",
        );
        assert!(
            matches!(
                outcome,
                DisambiguationOutcome::PotentialAlias { ref existing_id, similarity }
                if existing_id == "alice johnson"
                    && (similarity - L4_POTENTIAL_ALIAS_THRESHOLD).abs() < f32::EPSILON
            ),
            "cosine == L4_POTENTIAL_ALIAS_THRESHOLD + compatible names must produce \
             PotentialAlias, got {outcome:?}"
        );
    }

    #[test]
    fn classify_pair_cosine_above_merge_threshold_incompatible_names_is_new() {
        // TD-098 (Site #4): cosine ≥ L4_MERGE_THRESHOLD but names lexically
        // incompatible → NOT Merge (ADR-057) AND NOT a persistent PotentialAlias
        // (TD-098) → New. "alice"/"bob" share zero tokens → incompatible. This
        // SUPERSEDES ADR-057's "downgrade to PotentialAlias" cell: a degenerate
        // high-cosine pair between unrelated names must not create a false alias fact
        // that L7 can never revoke (cosine stays ≈ 1.0 forever under a weak embedder).
        let cosine = (L4_MERGE_THRESHOLD + 0.01).min(1.0);
        let outcome = classify_pair(cosine, "alice", "bob");
        assert!(
            matches!(outcome, DisambiguationOutcome::New),
            "cosine >= L4_MERGE_THRESHOLD + incompatible names must produce New \
             (TD-098: no persistent false alias), got {outcome:?}"
        );
    }

    #[test]
    fn classify_pair_alias_band_incompatible_names_is_new() {
        // TD-098 core poisoning fix: a pair in the [alias, merge) band (0.70–0.95)
        // whose names are lexically incompatible is degenerate-embedding noise and
        // must be `New`, NOT a persistent `potential_alias` fact. Pre-TD-098 this arm
        // was cosine-alone and produced PotentialAlias. "alice"/"bob" → zero shared tokens.
        let mid_band = (L4_POTENTIAL_ALIAS_THRESHOLD + L4_MERGE_THRESHOLD) / 2.0;
        let outcome = classify_pair(mid_band, "alice", "bob");
        assert!(
            matches!(outcome, DisambiguationOutcome::New),
            "alias-band cosine + incompatible names must produce New (TD-098), got {outcome:?}"
        );
    }

    #[test]
    fn classify_pair_below_alias_threshold_is_new() {
        // Below L4_POTENTIAL_ALIAS_THRESHOLD → New regardless of names.
        let cosine = L4_POTENTIAL_ALIAS_THRESHOLD - 0.01;
        let outcome = classify_pair(cosine, "alice", "bob");
        assert!(
            matches!(outcome, DisambiguationOutcome::New),
            "cosine < L4_POTENTIAL_ALIAS_THRESHOLD must produce New, got {outcome:?}"
        );
    }

    // ── Phase A: corpus precision/recall measurement ─────────────────────────
    //
    // `names_lexically_compatible` is pub(crate) — integration tests cannot
    // reach it. This unit test loads the adversarial corpus
    // (crates/kremory-eval/fixtures/entity_pairs.jsonl), runs the lexical gate
    // over every pair, and records precision/recall/F1 + per-category breakdown.
    //
    // Capability gaps in token-Jaccard (categories where overall recall < 1.0):
    //
    //   acronym   — IBM / International Business Machines: zero shared tokens → Jaccard 0.
    //   nickname  — Bob / Robert: zero shared tokens.
    //   diacritic — café / cafe: unicode-aware `is_alphanumeric` preserves diacritics,
    //               so the token sets never collide.
    //
    // These are NOT accepted failures — they are gaps that a pure token-matching approach
    // cannot close without hardcoded lists. The correct fix is context-embedding (ADR-058
    // B1), which routes these pairs through a semantic signal rather than a name-token
    // heuristic. Adding a curated acronym blocklist here was explicitly rejected in ADR-058.
    //
    // Known FP (homonym — measured, documented, routes to B1 for resolution):
    //   Amazon / Amazon River: both names share the token "amazon" → Jaccard 1/2 = 0.5.
    //   Homonyms cannot be distinguished by name tokens alone; context-embedding resolves
    //   them by embedding `name + context`, where "Amazon River" context diverges from
    //   "Amazon" (company) context.
    //
    // Hard assertions (ADR-057 §contract):
    //   precision ≥ 0.95  — one false merge corrupts every fact with that subject
    //   trivial recall ≥ 0.90  — guards against a degenerate never-merge gate
    //                            (ADR-058 RISK-006)
    //
    // Overall recall is RECORDED but not asserted — gap categories are capability limits
    // of the token layer, addressed by the embedding layer (B1), not by this gate.
    #[test]
    fn corpus_precision_recall() {
        let corpus_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../kremory-eval/fixtures/entity_pairs.jsonl"
        );
        let content = std::fs::read_to_string(corpus_path)
            .unwrap_or_else(|e| panic!("failed to read corpus at {corpus_path}: {e}"));

        struct Row {
            name_a: String,
            name_b: String,
            should_merge: bool,
            category: String,
            source: String,
        }

        let rows: Vec<Row> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| {
                let v: serde_json::Value = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("corpus line {i}: JSON error: {e}"));
                Row {
                    name_a: v["name_a"].as_str().expect("name_a").to_string(),
                    name_b: v["name_b"].as_str().expect("name_b").to_string(),
                    should_merge: v["should_merge"].as_bool().expect("should_merge"),
                    category: v["category"].as_str().expect("category").to_string(),
                    source: v["source"].as_str().expect("source").to_string(),
                }
            })
            .collect();

        assert!(!rows.is_empty(), "corpus must not be empty");

        let gate: Vec<bool> = rows
            .iter()
            .map(|r| names_lexically_compatible(&r.name_a, &r.name_b))
            .collect();

        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        let mut tn = 0usize;
        for (r, &g) in rows.iter().zip(gate.iter()) {
            match (g, r.should_merge) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let precision = if tp + fp == 0 {
            1.0_f64
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            1.0_f64
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0_f64
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        let categories = [
            "trivial",
            "suffix",
            "abbrev",
            "acronym",
            "nickname",
            "diacritic",
            "unrelated",
            "same-type",
            "same-context-diff-entity",
            "dotted_initialism",
        ];

        println!("\n── corpus_precision_recall ─────────────────────────────────────────");
        println!("  total={} TP={tp} FP={fp} FN={fn_} TN={tn}", rows.len());
        println!("  precision={precision:.4}  recall={recall:.4}  F1={f1:.4}");
        println!("  per-category:");

        let mut trivial_tp = 0usize;
        let mut trivial_pos = 0usize;

        for cat in &categories {
            let cat_tp: usize = rows
                .iter()
                .zip(gate.iter())
                .filter(|(r, &g)| r.category == *cat && g && r.should_merge)
                .count();
            let cat_fp: usize = rows
                .iter()
                .zip(gate.iter())
                .filter(|(r, &g)| r.category == *cat && g && !r.should_merge)
                .count();
            let cat_fn: usize = rows
                .iter()
                .zip(gate.iter())
                .filter(|(r, &g)| r.category == *cat && !g && r.should_merge)
                .count();
            let cat_tn: usize = rows
                .iter()
                .zip(gate.iter())
                .filter(|(r, &g)| r.category == *cat && !g && !r.should_merge)
                .count();
            let cat_label_pos = cat_tp + cat_fn;
            let cat_rec: f64 = if cat_label_pos == 0 {
                f64::NAN
            } else {
                cat_tp as f64 / cat_label_pos as f64
            };
            println!(
                "    {cat:<32} tp={cat_tp} fp={cat_fp} fn={cat_fn} tn={cat_tn}  rec={cat_rec:.2}"
            );
            if *cat == "trivial" {
                trivial_tp = cat_tp;
                trivial_pos = cat_label_pos;
            }
        }

        let trivial_recall = if trivial_pos == 0 {
            1.0_f64
        } else {
            trivial_tp as f64 / trivial_pos as f64
        };

        // Control-subset numbers (source="human") for ADR-058 ASMP-005.
        let human_tp: usize = rows
            .iter()
            .zip(gate.iter())
            .filter(|(r, &g)| r.source == "human" && g && r.should_merge)
            .count();
        let human_gate_pos: usize = rows
            .iter()
            .zip(gate.iter())
            .filter(|(r, &g)| r.source == "human" && g)
            .count();
        let human_label_pos: usize = rows
            .iter()
            .filter(|r| r.source == "human" && r.should_merge)
            .count();
        let human_prec = if human_gate_pos == 0 {
            1.0_f64
        } else {
            human_tp as f64 / human_gate_pos as f64
        };
        let human_rec = if human_label_pos == 0 {
            1.0_f64
        } else {
            human_tp as f64 / human_label_pos as f64
        };
        println!("\n  control-subset (source=human):");
        println!("    gate_pos={human_gate_pos}  label_pos={human_label_pos}  tp={human_tp}");
        println!("    precision={human_prec:.4}  recall={human_rec:.4}");
        println!("────────────────────────────────────────────────────────────────────");

        // ── Hard assertions (ADR-057 §contract) ──────────────────────────────
        assert!(
            precision >= 0.95,
            "lexical gate precision {precision:.4} < 0.95 — a false merge corrupts \
             every fact with that subject. FP={fp}. Inspect per-category output above."
        );
        assert!(
            trivial_recall >= 0.90,
            "trivial-variant recall {trivial_recall:.4} < 0.90 (ADR-058 RISK-006 — \
             degenerate never-merge gate guard). Expected {trivial_pos} trivial pairs \
             to pass; got {trivial_tp}."
        );
    }
}
