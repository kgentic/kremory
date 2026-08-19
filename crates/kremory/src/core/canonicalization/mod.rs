//! L5 vector canonicalization (dream-phase post-pass).
//!
//! Periodic batch task that runs pairwise cosine similarity across all entity
//! embeddings within a `group_id`.  Pairs above [`L5_CANONICALIZATION_THRESHOLD`]
//! are merged: the entity with the longer `description` field (LightRAG heuristic)
//! is kept; the other is remapped and deleted.
//!
//! ## Design notes
//!
//! - Pairwise complexity is O(N²) in Rust but the inner similarity computation
//!   is done via libsql's `vector_distance_cos`; only pairs with embeddings are
//!   considered (NULL embeddings are excluded from the listing).
//! - Each merge is wrapped in a `BEGIN IMMEDIATE` transaction so partial failures
//!   cannot leave the graph in an inconsistent state.
//! - A `loser → keeper` map with chain resolution handles transitivity correctly
//!   when merging chains of similar entities.
//! - Threshold comparison is **strict** (`>`): a score of exactly
//!   `L5_CANONICALIZATION_THRESHOLD` does NOT trigger a merge.
//!
//! ## Caller
//!
//! Invoke from the dream-phase pipeline after L4 disambiguation has run.
//! The function is intentionally exposed as a stand-alone method on
//! [`crate::core::schema::TemporalGraph`] (via the `canonicalize_surface_forms`
//! free-function) so it can also be unit-tested without a full dream-phase harness.

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use metrics::counter;
use tracing;

use crate::core::dream::provenance::{
    EntityMergePreState, EpisodicEdgeCols, EpisodicEdgeSnapshot, FactEndpoint, KeeperPre,
    LoserEntityRow, MergeInputs, MergeSite, RepointedFact,
};
use crate::core::error::{Error, Result};
use crate::core::identity_verdict::WriteDecision;
use crate::core::provider::DynEmbeddingProvider;
use crate::core::schema::TemporalGraph;

mod adjudicate;

pub use adjudicate::L5Adjudicator;

// ─── Threshold constant ───────────────────────────────────────────────────────

/// Cosine similarity **above which** two entity embeddings are considered surface
/// variants of the same real-world entity and are merged.
///
/// Comparison is **strict** (`>`): a score equal to the threshold does NOT merge.
///
/// Source: LightRAG post-extraction canonicalization design (empirically derived).
pub const L5_CANONICALIZATION_THRESHOLD: f32 = 0.8;

// ─── Report struct ────────────────────────────────────────────────────────────

/// Summary returned by [`canonicalize_surface_forms`].
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalizationReport {
    /// The `group_id` that was canonicalized.
    pub group_id: String,
    /// Number of entity pairs examined (upper triangle of the similarity matrix).
    pub pairs_examined: usize,
    /// Number of merge operations applied (loser → keeper remappings).
    pub merges_applied: usize,
    /// The threshold value used for this run.
    pub threshold_used: f32,
}

// ─── Internal type ────────────────────────────────────────────────────────────

/// Minimal entity representation loaded by [`canonicalize_surface_forms`].
/// The fields needed for the similarity check, keeper selection, and — on the
/// adjudicated path — the LLM's context.
struct EntitySlot {
    id: String,
    /// The entity's `properties.description`, empty when absent. Carried so the
    /// adjudicator can separate a CATEGORY from an IDENTITY (`pottery class` vs
    /// `pottery`) — a discrimination the bare names provably cannot support,
    /// since both sides of the failure sit at token-Jaccard 0.500.
    description: String,
    description_len: usize,
}

// ─── Public entry-point ───────────────────────────────────────────────────────

/// Run L5 surface-form canonicalization over all entities in `group_id`.
///
/// # Algorithm
///
/// 1. Fetch all entity rows with non-NULL embeddings from `group_id`.
/// 2. For every pair (i, j) where i < j, compute cosine similarity via
///    libsql's `vector_distance_cos`.
/// 3. Pairs with `similarity > threshold` are scheduled for merge:
///    - **Keeper** = the entity with the longer `description` in `properties`.
///    - **Loser**  = the other entity.
/// 4. Union-find collapses transitive chains so a loser that was already merged
///    into a keeper is not processed twice.
/// 5. Each merge is committed inside a single `BEGIN IMMEDIATE` transaction:
///    - Remap `facts.subject_id` / `facts.object_id` from loser → keeper.
///    - Remap `episodic_edges.entity_id` from loser → keeper.
///    - Accumulate loser's `access_count` into keeper.
///    - Delete loser from `entities_fts`, `entities`.
///
/// # No-op conditions
///
/// Returns a zero-merge report when:
/// - `group_id` has fewer than 2 entities with embeddings.
/// - No pair scores above `threshold`.
///
/// # Idempotency
///
/// After a successful run, no pair in `group_id` will have similarity above
/// `threshold` (assuming embeddings are stable), so a second call returns
/// `merges_applied = 0`.
pub async fn canonicalize_surface_forms(
    graph: &TemporalGraph,
    group_id: &str,
    threshold: f32,
) -> Result<CanonicalizationReport> {
    canonicalize_surface_forms_with_embedder(
        graph,
        CanonicalizeSurfaceFormsParams {
            group_id,
            threshold,
            embedder: None,
            adjudicator: None,
        },
    )
    .await
}

/// Bundled parameters for [`canonicalize_surface_forms_with_embedder`] —
/// args-as-object per TD-042 (`clippy.toml` `too-many-arguments-threshold = 3`).
/// `graph` stays a lead positional param (receiver-like dep, project
/// convention — mirrors [`ApplyMergeWithAuditParams`]).
pub struct CanonicalizeSurfaceFormsParams<'a> {
    pub group_id: &'a str,
    pub threshold: f32,
    /// When `Some`, every candidate merge is ADJUDICATED before it is applied:
    /// `names_lexically_compatible` is demoted from decider to NOMINATOR and the
    /// shared ADR-063 `write_gate` makes the call (see [`adjudicate`]).
    ///
    /// `None` preserves the pre-adjudication behavior EXACTLY — the deterministic
    /// cosine + lexical path, which is `write_gate` row 1's "clear case" by
    /// another name. That is deliberate rather than a stub: a consumer running
    /// `dream()` without a chat provider must not lose L5 entirely, and it makes
    /// the ~8 existing abbreviation tests across 5 files an untouched regression
    /// guard on this change.
    ///
    /// The live facade path always supplies `Some` — `dream()` already resolves a
    /// chat provider and a model id.
    pub adjudicator: Option<L5Adjudicator<'a>>,
    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): when `Some`,
    /// every merge this pass applies recomputes + persists the keeper's name
    /// embedding post-commit (see [`EntityMergeParams::embedder`]). `None`
    /// preserves [`canonicalize_surface_forms`]'s pre-TD-112 behavior (stored
    /// embedding left stale after a merge).
    pub embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// [`canonicalize_surface_forms`] extended with an embedder handle (TD-112,
/// `.ai-docs/tech-debt/tech-debt-register.md:2547`) so L5's own pairwise-cosine
/// merges re-embed the keeper instead of leaving its stored embedding stale.
/// Additive — the bare 3-arg [`canonicalize_surface_forms`] is this entry
/// point's `embedder: None` degradation (same "extended with an optional
/// param" shape as [`apply_merge_with_audit`]); all pre-existing call sites
/// and tests keep working unchanged.
pub async fn canonicalize_surface_forms_with_embedder(
    graph: &TemporalGraph,
    params: CanonicalizeSurfaceFormsParams<'_>,
) -> Result<CanonicalizationReport> {
    let CanonicalizeSurfaceFormsParams {
        group_id,
        threshold,
        embedder,
        adjudicator,
    } = params;
    // ── Step 1: load entity slots with embeddings ─────────────────────────────
    let slots = load_entity_slots(graph, group_id).await?;

    if slots.len() < 2 {
        tracing::debug!(
            target: "kremory.l5",
            group_id,
            entity_count = slots.len(),
            "kremory.l5.skip_too_few_entities"
        );
        return Ok(CanonicalizationReport {
            group_id: group_id.to_string(),
            pairs_examined: 0,
            merges_applied: 0,
            threshold_used: threshold,
        });
    }

    // ── Step 2: pairwise cosine similarity via SQL ────────────────────────────
    let raw_pairs = find_merge_pairs(graph, &slots, threshold).await?;
    let pairs_examined = (slots.len() * (slots.len().saturating_sub(1))) / 2;

    // ADR-057: the L5 batch merge is the MORE dangerous destructive-merge site —
    // its 0.80 threshold sits below the ~0.90 anisotropic-cosine floor of bare-name
    // embeddings, so a weak consumer embedder would merge nearly every entity in a
    // group (`tests/spike_td080_embedder_cosine.rs`). Apply the SAME deterministic
    // name-compatibility gate as L4: a cosine-high pair whose names are lexically
    // incompatible is anisotropy, not identity — drop it. Entity ids ARE normalized
    // names, so they are the correct lexical comparand.
    //
    // ⚠️ This gate is a NOMINATOR, not a decider, whenever `adjudicator` is `Some`.
    // Passing it earns a pair an LLM adjudication, NOT a merge. It stays at
    // `L4_LEXICAL_JACCARD_MIN` = 0.5 on purpose: raising it to 0.6 was measured
    // (it did recover the -9.9 nDCG) and REVERTED, because `alice j`/`alice
    // johnson` is Jaccard 0.500 exactly like `pottery class`/`pottery`. No
    // threshold separates a hypernym collapse from an abbreviated person name;
    // the distinction is semantic. See `adjudicate`'s module header.
    let pairs_to_merge: Vec<(String, String, f32)> = raw_pairs
        .into_iter()
        .filter(|(loser_id, keeper_id, _cosine)| {
            let compatible =
                crate::core::disambiguation::names_lexically_compatible(loser_id, keeper_id);
            if !compatible {
                counter!("kremory.l5.merge_blocked_lexical_total").increment(1);
                tracing::info!(
                    target: "kremory.l5",
                    loser_id = %loser_id,
                    keeper_id = %keeper_id,
                    "kremory.l5.merge_blocked_lexical"
                );
            }
            compatible
        })
        .collect();

    if pairs_to_merge.is_empty() {
        tracing::debug!(
            target: "kremory.l5",
            group_id,
            pairs_examined,
            "kremory.l5.no_merges_needed"
        );
        return Ok(CanonicalizationReport {
            group_id: group_id.to_string(),
            pairs_examined,
            merges_applied: 0,
            threshold_used: threshold,
        });
    }

    // ── Step 3: Resolve unique (loser → keeper) pairs ────────────────────────
    //
    // The `pairs_to_merge` upper-triangle list may contain the same loser in
    // multiple pairs (e.g., for a triplet A↔B, A↔C, B↔C all above threshold).
    // Strategy:
    //   a) Build a `loser → best_keeper` map: for each loser keep the keeper with
    //      the longest description (preserves LightRAG heuristic transitively).
    //   b) Iterate unique losers, resolving the best_keeper transitively (if the
    //      chosen keeper is itself a loser in the map, follow the chain until we
    //      reach a non-loser root).
    //   c) Skip if resolved keeper == loser (degenerate self-merge).

    // Build description_len lookup for fast access.
    let desc_len_map: std::collections::HashMap<String, usize> = slots
        .iter()
        .map(|s| (s.id.clone(), s.description_len))
        .collect();

    // loser_id → keeper_id (the keeper with the longest description seen so far
    // for this loser across all pairs it appears in).
    let mut loser_to_keeper: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Cosine of each nominated pair, keyed by (loser, keeper), for audit fidelity
    // on the adjudicated path. Chain resolution below can produce a
    // `(loser, effective_keeper)` pair absent from this map — that is a genuine
    // `None`, recorded as such rather than back-filled with a neighbour's value.
    let pair_cosine: std::collections::HashMap<(String, String), f32> = pairs_to_merge
        .iter()
        .map(|(loser_id, keeper_id, cosine)| ((loser_id.clone(), keeper_id.clone()), *cosine))
        .collect();

    for (loser_id, keeper_id, _cosine) in &pairs_to_merge {
        loser_to_keeper
            .entry(loser_id.clone())
            .and_modify(|existing_keeper| {
                // Keep the longer-description keeper.
                let new_len = desc_len_map.get(keeper_id).copied().unwrap_or(0);
                let existing_len = desc_len_map
                    .get(existing_keeper.as_str())
                    .copied()
                    .unwrap_or(0);
                if new_len > existing_len {
                    *existing_keeper = keeper_id.clone();
                }
            })
            .or_insert_with(|| keeper_id.clone());
    }

    // Resolve transitive chains: if keeper_X is itself a loser, follow the map.
    // Bounded iteration (max chain length = number of unique losers).
    fn resolve_keeper(
        _loser: &str,
        keeper: &str,
        map: &std::collections::HashMap<String, String>,
    ) -> String {
        let mut current = keeper.to_string();
        // Safety bound: stop after map.len() hops to prevent infinite loops.
        for _ in 0..=map.len() {
            if let Some(next) = map.get(&current) {
                if next == &current {
                    break;
                }
                current = next.clone();
            } else {
                break;
            }
        }
        current
    }

    // Anti-re-merge nogood (Site #2 of THREE, spec §6.2 V3 fix): load the
    // group's split-pair bans ONCE before the merge loop; a pair `unmerge`
    // recorded must NOT be re-merged here on the next `dream()`.
    let nogoods =
        crate::core::dream::provenance::reversal::load_merge_nogoods(graph, group_id).await?;

    // Resolve every loser to its effective keeper and apply the two cheap,
    // deterministic filters (self-merge, nogood) BEFORE any LLM call — a banned or
    // degenerate pair must never cost an adjudication.
    let mut resolved: Vec<(String, String)> = Vec::new();
    for (loser_id, raw_keeper_id) in &loser_to_keeper {
        let effective_keeper = resolve_keeper(loser_id, raw_keeper_id, &loser_to_keeper);

        // Skip self-merges (shouldn't happen, but guard defensively).
        if loser_id == &effective_keeper {
            continue;
        }

        // Nogood guard (spec §6.2, Site #2): drop a split pair before the merge.
        if nogoods.contains(&crate::core::dream::provenance::reversal::sorted_pair(
            loser_id,
            &effective_keeper,
        )) {
            counter!(
                "kremory.graph.merge_nogood_skip_total",
                "site" => "canonicalize",
            )
            .increment(1);
            continue;
        }

        resolved.push((loser_id.clone(), effective_keeper));
    }

    let merges_applied = match adjudicator {
        // ── Adjudicated path (ADR-063) ────────────────────────────────────────
        Some(adjudicator) => {
            apply_adjudicated_merges(
                graph,
                AdjudicatedMergesParams {
                    adjudicator: &adjudicator,
                    resolved: &resolved,
                    slots: &slots,
                    pair_cosine: &pair_cosine,
                    group_id,
                    embedder,
                },
            )
            .await?
        }
        // ── Deterministic path — pre-adjudication behavior, byte-for-byte ─────
        //
        // This is `write_gate` row 1's "clear case" (cosine above threshold AND a
        // deterministic signal fired, no LLM consulted) expressed directly rather
        // than routed through the gate. It is NOT routed through it on purpose:
        // chain resolution can produce a `(loser, effective_keeper)` pair with no
        // cosine of its own, and feeding the gate a fabricated `0.0` there would
        // flip these merges to `Reject` — a behavior change for consumers with no
        // chat provider, and a lie in any audit row it wrote.
        None => {
            let mut applied = 0usize;
            for (loser_id, effective_keeper) in &resolved {
                tracing::info!(
                    target: "kremory.l5",
                    group_id,
                    loser_id = %loser_id,
                    keeper_id = %effective_keeper,
                    "kremory.l5.merge"
                );
                apply_merge(
                    graph,
                    ApplyMergeParams {
                        loser_id,
                        keeper_id: effective_keeper,
                        group_id,
                        embedder,
                    },
                )
                .await?;
                applied += 1;
            }
            applied
        }
    };

    counter!("kremory.l5.merges_applied_total").increment(merges_applied as u64);
    counter!("kremory.l5.canonicalization_runs_total").increment(1);

    tracing::info!(
        target: "kremory.l5",
        group_id,
        pairs_examined,
        merges_applied,
        threshold,
        "kremory.l5.done"
    );

    Ok(CanonicalizationReport {
        group_id: group_id.to_string(),
        pairs_examined,
        merges_applied,
        threshold_used: threshold,
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Bundled params for [`apply_adjudicated_merges`] — args-as-object per TD-042
/// (`clippy.toml` `too-many-arguments-threshold = 3`). `graph` stays a lead
/// positional param (receiver-like dep, project convention).
struct AdjudicatedMergesParams<'a> {
    adjudicator: &'a L5Adjudicator<'a>,
    /// `(loser_id, effective_keeper_id)` pairs surviving the self-merge and
    /// nogood filters.
    resolved: &'a [(String, String)],
    slots: &'a [EntitySlot],
    pair_cosine: &'a std::collections::HashMap<(String, String), f32>,
    group_id: &'a str,
    embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// Adjudicate every resolved candidate and apply only those the shared
/// [`crate::core::identity_verdict::write_gate`] authorizes. Returns the number
/// of merges actually applied.
///
/// Decision handling (ADR-063 spec §2.2, §5.1):
/// - `Merge` — destructive remap, with its `identity_verdict_audit` row written
///   INSIDE the same `BEGIN IMMEDIATE` (RISK-003: the audit row and the remap are
///   one logical event, so a rollback must undo both).
/// - `PotentialAlias` — audit row ONLY. Deliberately no `potential_alias` fact:
///   Site #5's alias facts are inert at L7 because its pairs share zero tokens by
///   construction, but L5's pairs are token-Jaccard ≥ 0.5 BY CONSTRUCTION, so they
///   would pass `resolve_pending_aliases`' own lexical gate and re-merge on a later
///   cycle — a back door to the very collapse this adjudicator exists to stop.
/// - `Reject` — audit row when a verdict exists (see the module header of
///   [`adjudicate`]: a rejected hypernym collapse IS the fix working, and its
///   `reasoning` is the diagnostic surface), nothing otherwise.
async fn apply_adjudicated_merges(
    graph: &TemporalGraph,
    params: AdjudicatedMergesParams<'_>,
) -> Result<usize> {
    let AdjudicatedMergesParams {
        adjudicator,
        resolved,
        slots,
        pair_cosine,
        group_id,
        embedder,
    } = params;

    if resolved.is_empty() {
        return Ok(0);
    }

    let descriptions: std::collections::HashMap<&str, &str> = slots
        .iter()
        .map(|s| (s.id.as_str(), s.description.as_str()))
        .collect();

    let candidates: Vec<adjudicate::Candidate<'_>> = resolved
        .iter()
        .map(|(loser_id, keeper_id)| adjudicate::Candidate {
            loser_id,
            keeper_id,
            cosine: pair_cosine
                .get(&(loser_id.clone(), keeper_id.clone()))
                .copied(),
            loser_description: descriptions.get(loser_id.as_str()).copied().unwrap_or(""),
            keeper_description: descriptions.get(keeper_id.as_str()).copied().unwrap_or(""),
        })
        .collect();

    let decisions = adjudicate::adjudicate(
        adjudicator,
        adjudicate::AdjudicateParams {
            candidates: &candidates,
            group_id,
        },
    )
    .await?;

    let run_id = uuid::Uuid::new_v4().to_string();
    let mut merges_applied = 0usize;

    for (candidate, adjudicated) in candidates.iter().zip(decisions.iter()) {
        let cosine = candidate.cosine;
        // The write_gate's own DeterministicSignal input, RE-DERIVED rather than
        // carried over from the nominator — same discipline as `adjudicate::decide`,
        // and it is what the audit column is defined to record. NOT the same field
        // as `ApplyMergeWithAuditParams::structural_signal` below, which records
        // STRUCTURAL corroboration and is correctly `false` for L5's pairwise
        // cosine (`graph_mutation_log.inputs.structural_signal`).
        let deterministic_signal = crate::core::disambiguation::names_lexically_compatible(
            candidate.loser_id,
            candidate.keeper_id,
        );

        let decision_label = match adjudicated.decision {
            WriteDecision::Merge => "merge",
            WriteDecision::PotentialAlias => "potential_alias",
            WriteDecision::Reject => "reject",
        };

        // `candidate_a` = keeper, `candidate_b` = loser (the direction the merge
        // would take), stated here because the column names do not say it.
        let audit = IdentityVerdictAuditRow {
            site: adjudicate::SITE_LABEL,
            group_id,
            candidate_a: candidate.keeper_id,
            candidate_b: candidate.loser_id,
            cosine,
            structural_signal: deterministic_signal,
            verdict: adjudicated.verdict.as_ref(),
            decision: decision_label,
            run_id: &run_id,
        };

        match adjudicated.decision {
            WriteDecision::Merge => {
                tracing::info!(
                    target: "kremory.l5",
                    group_id,
                    loser_id = %candidate.loser_id,
                    keeper_id = %candidate.keeper_id,
                    adjudicated = true,
                    "kremory.l5.merge"
                );
                apply_merge_with_audit(
                    graph,
                    ApplyMergeWithAuditParams {
                        loser_id: candidate.loser_id,
                        keeper_id: candidate.keeper_id,
                        group_id,
                        audit: Some(audit),
                        site: MergeSite::Canonicalize,
                        // L5 has no STRUCTURAL corroboration signal — its
                        // deterministic signal is lexical. Unchanged from the
                        // pre-adjudication path on purpose (Quinn L3 honesty).
                        structural_signal: false,
                        embedder,
                    },
                )
                .await?;
                merges_applied += 1;
            }
            WriteDecision::PotentialAlias | WriteDecision::Reject => {
                // Nothing is written to the graph. Record WHY, but only when a
                // verdict exists — a row with every `llm_*` column NULL says
                // nothing the `write_gate_decision_total` counter has not said.
                if adjudicated.verdict.is_some() {
                    adjudicate::write_non_merge_audit_row(&graph.conn, audit).await?;
                }
                tracing::info!(
                    target: "kremory.l5",
                    group_id,
                    loser_id = %candidate.loser_id,
                    keeper_id = %candidate.keeper_id,
                    decision = decision_label,
                    "kremory.l5.merge_not_applied"
                );
            }
        }
    }

    Ok(merges_applied)
}

/// Fetch all entity IDs and their description lengths from `group_id`.
/// Only entities with a non-NULL embedding are returned (no embedding = can't
/// compute cosine similarity).
async fn load_entity_slots(graph: &TemporalGraph, group_id: &str) -> Result<Vec<EntitySlot>> {
    let mut rows = graph
        .conn
        .query(
            "SELECT id, properties \
             FROM entities \
             WHERE group_id = ?1 AND embedding IS NOT NULL",
            libsql::params![group_id],
        )
        .await?;

    let mut slots = Vec::new();
    while let Some(row) = rows.next().await? {
        let id: String = row.get(0)?;
        let props_text: Option<String> = row.get(1)?;
        let description = props_text
            .as_deref()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| {
                v.get("description")
                    .and_then(|d| d.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        // Derived, never separately parsed — the keeper heuristic and the
        // adjudicator's context must not be able to disagree about the same field.
        let description_len = description.len();
        slots.push(EntitySlot {
            id,
            description,
            description_len,
        });
    }
    Ok(slots)
}

/// For every (i, j) pair with i < j, compute cosine similarity via SQL.
/// Returns `(loser_id, keeper_id, similarity)` triples where similarity >
/// threshold.
///
/// The similarity is carried out (rather than discarded once the comparison is
/// made) so the adjudicated path can record the REAL cosine in each
/// `identity_verdict_audit` row. It is audit fidelity, not a gate input — see
/// `adjudicate`'s divergence 1.
///
/// Keeper selection: longer description wins (LightRAG heuristic).
async fn find_merge_pairs(
    graph: &TemporalGraph,
    slots: &[EntitySlot],
    threshold: f32,
) -> Result<Vec<(String, String, f32)>> {
    let mut merge_pairs: Vec<(String, String, f32)> = Vec::new();

    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            let id_a = &slots[i].id;
            let id_b = &slots[j].id;

            // Use SQL vector_distance_cos: distance = 1 - similarity (cosine distance).
            // similarity = 1.0 - distance
            // NULL is returned when either vector has zero magnitude — skip those.
            let mut rows = graph
                .conn
                .query(
                    "SELECT vector_distance_cos(a.embedding, b.embedding) \
                     FROM entities a, entities b \
                     WHERE a.id = ?1 AND b.id = ?2",
                    libsql::params![id_a.clone(), id_b.clone()],
                )
                .await?;

            let Some(row) = rows.next().await? else {
                continue;
            };

            let distance_opt: Option<f64> = row.get(0)?;
            let Some(distance) = distance_opt else {
                // Zero-magnitude vector(s) — skip.
                continue;
            };

            let similarity = (1.0_f32 - distance as f32).clamp(0.0, 1.0);

            if similarity > threshold {
                // Keeper = longer description (LightRAG heuristic).
                let (keeper_idx, loser_idx) =
                    if slots[i].description_len >= slots[j].description_len {
                        (i, j)
                    } else {
                        (j, i)
                    };
                merge_pairs.push((
                    slots[loser_idx].id.clone(),
                    slots[keeper_idx].id.clone(),
                    similarity,
                ));
            }
        }
    }

    Ok(merge_pairs)
}

/// Apply a single merge: remap all edges from `loser_id` → `keeper_id`,
/// accumulate `access_count`, then delete the loser row.
///
/// Wrapped in a single `BEGIN IMMEDIATE` transaction — partial failure rolls back.
///
/// Thin wrapper over [`apply_merge_with_audit`] with `audit = None` — L5's own
/// pairwise cosine merges are not LLM-adjudicated, so no `identity_verdict_audit`
/// row is written for them (ADR-063 spec §5.2: "audit only LLM-touched decisions").
async fn apply_merge(graph: &TemporalGraph, params: ApplyMergeParams<'_>) -> Result<()> {
    let ApplyMergeParams {
        loser_id,
        keeper_id,
        group_id,
        embedder,
    } = params;
    apply_entity_merge(
        graph,
        EntityMergeParams {
            loser_id,
            keeper_id,
            group_id,
            site: MergeSite::Canonicalize,
            // L5 surface-form merges are pairwise-cosine + lexical-variant gated,
            // NOT structural-corroboration driven (spec §2.3, Quinn L3).
            structural_signal: false,
            embedder,
        },
    )
    .await
}

/// Bundled parameters for [`apply_merge`] — args-as-object per TD-042
/// (`clippy.toml` `too-many-arguments-threshold = 3`). `graph` stays a lead
/// positional param (receiver-like dep, project convention).
struct ApplyMergeParams<'a> {
    loser_id: &'a str,
    keeper_id: &'a str,
    /// ADR-029d: the namespace this merge is scoped to (L5's own group_id,
    /// already in scope at the call site). See [`EntityMergeParams::group_id`].
    group_id: &'a str,
    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): when `Some`,
    /// the keeper's name embedding is recomputed + persisted post-commit so it
    /// reflects the merged identity instead of going stale. `None` preserves
    /// pre-TD-112 behavior (no re-embed).
    embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// The shared structural entity-merge executor (ADR-066 spec DoD-P0.3).
///
/// Remaps all edges `loser_id → keeper_id` (`facts.subject_id`/`object_id`,
/// `episodic_edges.entity_id`), accumulates the loser's **`Entity.access_count`**
/// into the keeper (DENT-001 — the LIVE search-tracking field `schema.rs:106`, NOT
/// the deprecated `facts.access_count`; retained verbatim, test T9 asserts it),
/// combines `ner_confidence` via noisy-OR, then deletes the loser from
/// `entities_fts` + `entities`. All inside one `BEGIN IMMEDIATE`.
///
/// Extracted so BOTH `canonicalize_surface_forms` (L5 reconciliation) AND the
/// dream CONSOLIDATION `cross_episode_merges` op (ADR-066 §2.2 / spec P3) call ONE
/// merge code path — no second, drift-prone executor (R-02). This is the
/// `audit = None` (no `identity_verdict_audit` row) surface — the pairwise-cosine /
/// structural-corroboration merges these two callers perform are not LLM-adjudicated.
pub(crate) async fn apply_entity_merge(
    graph: &TemporalGraph,
    params: EntityMergeParams<'_>,
) -> Result<()> {
    let EntityMergeParams {
        loser_id,
        keeper_id,
        group_id,
        site,
        structural_signal,
        embedder,
    } = params;
    apply_merge_with_audit(
        graph,
        ApplyMergeWithAuditParams {
            loser_id,
            keeper_id,
            group_id,
            audit: None,
            site,
            structural_signal,
            embedder,
        },
    )
    .await
}

/// Bundled parameters for [`apply_entity_merge`] — args-as-object per TD-042
/// (`clippy.toml` `too-many-arguments-threshold = 3`; `#[allow]` banned in src).
/// Threads the merge `site` (reversible-graph-mutations spec §2.3) through the
/// shared structural-merge entry without exceeding the arg-count bar.
pub(crate) struct EntityMergeParams<'a> {
    pub(crate) loser_id: &'a str,
    pub(crate) keeper_id: &'a str,
    /// ADR-029d: entity identity is per-namespace-open — the same `id` can now
    /// legitimately exist in TWO namespaces. Every read/write this executor
    /// performs MUST be scoped by this `group_id` (the namespace the caller is
    /// operating in), never a bare `id`-only predicate, or a merge in namespace
    /// A can remap/delete rows that belong to namespace B (cross-namespace FK
    /// corruption — the bug this field closes).
    pub(crate) group_id: &'a str,
    pub(crate) site: MergeSite,
    /// Whether a deterministic STRUCTURAL corroboration signal drove this merge
    /// (reversible-graph-mutations spec §2.3 — recorded into
    /// `graph_mutation_log.inputs.structural_signal`, the SEE-surface honesty
    /// the inspect view reports, Quinn L3). `true` for cross-episode structural
    /// merges (the corroboration gate); `false` for L5's pairwise-cosine surface
    /// merges (no structural signal). Threaded from the call-site rather than
    /// derived from `audit` (which is `None` on both non-LLM sites).
    pub(crate) structural_signal: bool,
    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): when `Some`,
    /// the keeper's name embedding is recomputed from its canonical id and
    /// persisted post-commit, so the surviving entity's stored embedding
    /// reflects its post-merge identity instead of going stale. `None`
    /// preserves pre-TD-112 behavior (embedding left untouched) — the shape
    /// callers without embedder access (e.g. Site #5's embedder-independent
    /// design) use today.
    pub(crate) embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// One `identity_verdict_audit` row to write INSIDE the same `BEGIN IMMEDIATE`
/// transaction as a destructive merge (ADR-063 spec §5.1 RISK-003 — "the audit
/// row and the destructive remap are the same logical event; a rollback undoes
/// the audit row along with the remap, which is correct"). Shared by Site #5
/// (`core::dream::acronym_nickname_recall`) and any future LLM-adjudicated
/// caller of [`apply_merge_with_audit`].
pub(crate) struct IdentityVerdictAuditRow<'a> {
    /// `'site5_acronym_nickname'` | `'site3_type_registry'` (spec §5.1).
    pub(crate) site: &'a str,
    pub(crate) group_id: &'a str,
    pub(crate) candidate_a: &'a str,
    pub(crate) candidate_b: &'a str,
    /// `None` for Site #5 (no meaningful cosine signal, spec §3.3/§5.1).
    pub(crate) cosine: Option<f32>,
    /// The write_gate's `DeterministicSignal` input value for this pair.
    pub(crate) structural_signal: bool,
    /// The LLM verdict that authorized this merge. `Merge` via `write_gate`
    /// row 5 always carries a verdict for Site #5 (row 1 never fires there —
    /// `cosine` is always `0.0`, spec §3.3).
    pub(crate) verdict: Option<&'a crate::core::identity_verdict::IdentityVerdictItem>,
    /// `'merge'` — this row always documents a `Merge` decision (the
    /// `PotentialAlias`/`Reject` arms never call `apply_merge_with_audit`).
    pub(crate) decision: &'a str,
    pub(crate) run_id: &'a str,
}

/// Bundled parameters for [`apply_merge_with_audit`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments, threshold 3). `graph` stays
/// a lead positional param (receiver-like dep, project convention).
pub(crate) struct ApplyMergeWithAuditParams<'a> {
    pub(crate) loser_id: &'a str,
    pub(crate) keeper_id: &'a str,
    /// ADR-029d: the namespace this merge is scoped to. See
    /// [`EntityMergeParams::group_id`] doc for the full rationale — every
    /// read/write below MUST filter on this value, never a bare `id`.
    pub(crate) group_id: &'a str,
    pub(crate) audit: Option<IdentityVerdictAuditRow<'a>>,
    /// Which merge-producing site fired this merge — recorded into the
    /// `graph_mutation_log` row's `inputs.site` (reversible-graph-mutations
    /// spec §2.3), the provenance the nogood + undo consume in later sub-phases.
    pub(crate) site: MergeSite,
    /// Whether a deterministic STRUCTURAL corroboration signal drove this merge
    /// — recorded into `inputs.structural_signal` (spec §2.3, Quinn L3). `true`
    /// for the cross-episode structural gate and Site #5's structural pre-filter;
    /// `false` for L5 pairwise cosine. Threaded from the call-site (not derived
    /// from `audit`, which is `None` on the non-LLM sites).
    pub(crate) structural_signal: bool,
    /// TD-112: when `Some`, the keeper's name embedding is recomputed +
    /// persisted (best-effort, post-commit) so it reflects the merged
    /// identity. See [`EntityMergeParams::embedder`] doc for the full rationale.
    pub(crate) embedder: Option<&'a dyn DynEmbeddingProvider>,
}

/// Does merging `pair.0` (loser) into `pair.1` (keeper) EXTEND an existing live
/// merge chain in `group_id`?
///
/// Returns `Some((prior_counterpart, kind))` when it does, where `kind` is one of:
///
/// - `"loser_is_prior_survivor"` — the loser previously ABSORBED something. Merging
///   it away now moves whatever it absorbed a SECOND hop, to a destination that was
///   never adjudicated against the original. This is the cascade signature measured
///   on the corrupted conv0 database (`melanie` → `caroline`, then `caroline` →
///   `loved ones`: melanie's identity travelled two hops on approvals about melanie).
/// - `"keeper_is_prior_victim"` — the keeper was itself absorbed earlier, so this
///   merge targets a row that has already ceased to be a distinct identity.
///
/// Reads the DURABLE `graph_mutation_log` (reversible-graph-mutations spec §2.1),
/// which is what makes it see chains a caller's per-invocation resolution map
/// cannot: across dream cycles, and across sites.
///
/// **Never fails the merge.** A detector that can abort a write it does not
/// understand is worse than no detector; every error path returns `None` after
/// warning, so a schema drift or a malformed row degrades to "not detected"
/// LOUDLY rather than to a spurious rollback. Scoped by `group_id` per ADR-029d —
/// under per-namespace-open the same `id` legitimately exists in two namespaces
/// and an unscoped read would manufacture cross-namespace chains that do not exist.
///
/// Takes the pair as a tuple rather than two params: `clippy.toml` sets
/// `too-many-arguments-threshold = 3` and `#[allow]` is banned in `src`
/// (TD-042 / rust-conventions).
async fn detect_merge_chain_extension(
    graph: &TemporalGraph,
    group_id: &str,
    pair: (&str, &str),
) -> Option<(String, &'static str)> {
    let (loser_id, keeper_id) = pair;

    // RED-proven in both directions 2026-08-19, levers removed after:
    //   - forced `return None`      -> both `chain_detector_fires_*` FAIL (it can see)
    //   - `group_id` filter dropped -> `chain_detector_is_namespace_scoped` FAILS
    //     (the scoping is load-bearing, not decorative)
    let mut rows = match graph
        .conn
        .query(
            "SELECT id, inputs FROM graph_mutation_log \
             WHERE kind = 'entity_merge' AND group_id = ?1 AND undone_at IS NULL \
             ORDER BY id",
            libsql::params![group_id],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "kremory.l5",
                group_id,
                error = %e,
                "merge-chain detector: could not read graph_mutation_log — chain \
                 detection is DEGRADED for this merge, not clean"
            );
            return None;
        }
    };

    loop {
        let row = match rows.next().await {
            Ok(Some(row)) => row,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(
                    target: "kremory.l5",
                    group_id,
                    error = %e,
                    "merge-chain detector: graph_mutation_log row read failed — \
                     chain detection is DEGRADED for this merge, not clean"
                );
                return None;
            }
        };

        let Ok(inputs_json) = row.get::<String>(1) else {
            continue;
        };
        let Ok(prior) = serde_json::from_str::<MergeInputs>(&inputs_json) else {
            // Producer-shape drift. Warn rather than skip silently: a detector
            // that cannot parse its own producer's rows reports "no chains" for
            // exactly the same reason a broken one does.
            tracing::warn!(
                target: "kremory.l5",
                group_id,
                mutation_id = row.get::<i64>(0).unwrap_or(-1),
                "merge-chain detector: entity_merge row carries unparseable inputs \
                 — this prior merge is INVISIBLE to chain detection"
            );
            continue;
        };

        // The loser previously absorbed something → this merge moves that
        // absorbed identity a second, unadjudicated hop.
        if prior.keeper == loser_id {
            return Some((prior.loser, "loser_is_prior_survivor"));
        }
        // The keeper was previously absorbed → we are merging into a row that
        // already lost its distinct identity.
        if prior.loser == keeper_id {
            return Some((prior.keeper, "keeper_is_prior_victim"));
        }
    }

    None
}

/// [`apply_merge`] extended with an OPTIONAL `identity_verdict_audit` INSERT
/// (ADR-063 spec §3.3/§5.1) inside the SAME `BEGIN IMMEDIATE` transaction as
/// the destructive remap. `audit = None` preserves `apply_merge`'s exact
/// prior behavior (existing L5 callers); `audit = Some(..)` is the new Site #5
/// path. This is the "extended `apply_merge` with an optional audit param"
/// approach named in the implementation brief — additive, no change to the
/// existing `apply_merge` call site's signature or semantics.
pub(crate) async fn apply_merge_with_audit(
    graph: &TemporalGraph,
    params: ApplyMergeWithAuditParams<'_>,
) -> Result<()> {
    let ApplyMergeWithAuditParams {
        loser_id,
        keeper_id,
        group_id,
        audit,
        site,
        structural_signal,
        embedder,
    } = params;
    let guard = graph.begin_immediate_if_needed().await?;
    // Whether THIS guard opened the txn — decides post-commit o11y placement
    // (spec §8.1: the snapshot counter fires only after the durable commit).
    let committed_here = guard.opened();

    // Collect loser's access_count + ner_confidence before deletion. Site #6
    // (ADR-063): the loser's confidence is combined into the keeper via noisy-OR,
    // not discarded (SYNTHESIS §2 — merging the same entity must never lower it).
    // ADR-029d: scoped by `group_id` — under per-namespace-open the same `id`
    // can exist in a DIFFERENT namespace; an unscoped read here could pick up
    // the wrong namespace's row.
    let mut rows = graph
        .conn
        .query(
            "SELECT access_count, ner_confidence FROM entities WHERE id = ?1 AND group_id = ?2",
            libsql::params![loser_id, group_id],
        )
        .await;

    let (loser_access_count, loser_ner_confidence): (i64, Option<f32>) = match rows {
        Ok(ref mut r) => match r.next().await {
            Ok(Some(row)) => (
                row.get::<i64>(0).unwrap_or(0),
                row.get::<Option<f64>>(1).ok().flatten().map(|v| v as f32),
            ),
            _ => (0, None),
        },
        Err(_) => (0, None),
    };

    // ─── TD-223 write-time merge-chain DETECTOR (2026-08-19) ────────────────
    //
    // Runs BEFORE `snapshot_merge_pre_state` so the log holds only PRIOR merges
    // — this merge's own row does not exist yet and cannot self-trigger.
    //
    // ⚠️ THIS OBSERVES. IT DOES NOT BLOCK, AND THAT IS DELIBERATE.
    //
    // The obvious design — "make invariant 6 a write-time gate so no pass can
    // cascade" — has no correct trigger AT THIS LAYER, and the repo already said
    // so before this was written. Invariant 6's own header
    // (`kremory-eval/src/layer_b/graph_integrity.rs:16`) records that on the only
    // evidence that exists (n = 2 databases) roughly 4 of 6 flags were
    // catastrophic and roughly 2 were BENIGN, and states: "acceptable for a
    // REPORTED metric and needs more evidence before it BLOCKS anything". A
    // legitimate three-variant canonicalisation forms the same chain shape
    // (`pottery class` -> `pottery` -> `pottery project`: two defensible merges,
    // one flagged entity).
    //
    // The deeper reason is structural, not just evidential. By the time control
    // reaches this executor EVERY caller has already justified the pair:
    //   - Site #5   — an LLM verdict on this exact pair (`audit: Some`)
    //   - L5        — its own per-pair cosine + lexical-variant rule
    //   - cross-episode — the structural corroboration gate
    // The fact that made the cascade wrong — "this pair is a RETARGET of a
    // different adjudicated pair" — lives only in the CALLER's resolution loop
    // and is unrecoverable here. Refusing chains at this layer would therefore
    // block correct work while not being the check that catches the real defect.
    // That refusal belongs where the retarget happens, and it is already there
    // (`dream/acronym_nickname_recall.rs:575` merge arm, and the alias arm as of
    // 2026-08-19).
    //
    // What this DOES close is the blind spot neither of those can see: Site #5's
    // `merged_into` map is per-INVOCATION, so a chain formed across two dream
    // cycles, or across two different sites, is invisible to it. This reads the
    // DURABLE log, so it sees both — at zero false-positive cost, because it
    // changes no behaviour. It is also what generates the per-site evidence base
    // invariant 6 says is missing, so a future decision to BLOCK can rest on
    // measured chain rates rather than on n = 2.
    if let Some((prior_counterpart, kind)) =
        detect_merge_chain_extension(graph, group_id, (loser_id, keeper_id)).await
    {
        counter!(
            "kremory.identity.merge_chain_extension_total",
            "site" => site.as_str(),
            "kind" => kind,
            "adjudicated" => if audit.is_some() { "true" } else { "false" },
        )
        .increment(1);
        // ALWAYS-ON warn. A cascade that stayed silent for an entire corrupted
        // run is the reason this exists; a debug-gated signal would reproduce
        // exactly that failure.
        tracing::warn!(
            target: "kremory.l5",
            group_id,
            keeper_id,
            loser_id,
            prior_counterpart = %prior_counterpart,
            chain_kind = kind,
            site = site.as_str(),
            adjudicated = audit.is_some(),
            "merge EXTENDS a live merge chain — an identity is moving a second hop. \
             Not blocked (see the detector's comment for why blocking here is wrong); \
             this is the signal to inspect `graph_mutation_log`."
        );
    }

    // ─── Reversible-graph-mutations snapshot (sub-phase 1b, spec §4) ─────────
    // Capture the COMPLETE pre-state and INSERT the `graph_mutation_log` row
    // BEFORE the destructive block below, on `graph.conn` — the SAME connection
    // the destructive statements use, inside the `guard` transaction opened
    // above. The snapshot INSERT therefore commits atomically with the merge
    // (or rolls back with it via the r1..r7 chain / the early-rollback here), so
    // provenance can never diverge from what the merge actually destroyed
    // (spec §1 / §8.1). Capture-BEFORE-destroy is load-bearing: the loser row,
    // its facts' prior `corroboration_inert`, the keeper's pre-overwrite values,
    // and the episodic-edge collision flags only exist until the UPDATEs below
    // run (spec §2.3 / §2.4 DERIVATION deps).
    // TD-203 D1 — `doomed_self_loops` is the set of facts this merge will
    // collapse to `keeper -> keeper`; expired after the endpoint re-points land.
    let (snapshot_group_id, doomed_self_loops) = match snapshot_merge_pre_state(
        graph,
        MergeSnapshotParams {
            loser_id,
            keeper_id,
            group_id,
            site,
            audit: audit.as_ref(),
            structural_signal,
        },
    )
    .await
    {
        Ok(pair) => pair,
        Err(e) => {
            let _ = guard.rollback().await;
            return Err(e);
        }
    };

    // Remap facts.subject_id — ADR-067 §C0 (V1): also stamp `corroboration_inert = 1`
    // on every rewritten row. The endpoint remap makes the keeper INHERIT the
    // loser's fact-neighbours; without this stamp `cross_episode`'s corroboration
    // reads (`neighbours_of`/`assertions_of`, which filter `corroboration_inert = 0`)
    // would treat the inherited structure as directly-asserted on the NEXT pass,
    // re-opening a deferred bridge partner's eligibility (the V1 convergence bug).
    // The flag is monotone (set here, never cleared) — see impl-spec §6 fixpoint proof.
    // ADR-029d: scoped by `subject_group_id` — the composite FK is
    // `(subject_id, subject_group_id) → entities(id, group_id)`; an unscoped
    // `WHERE subject_id = loser_id` remaps EVERY namespace's facts pointing at
    // that id, including a same-id row this merge has no business touching
    // (the cross-namespace FK-corruption bug this scoping closes).
    let r1 = graph
        .conn
        .execute(
            "UPDATE facts SET subject_id = ?1, corroboration_inert = 1 \
             WHERE subject_id = ?2 AND subject_group_id = ?3",
            libsql::params![keeper_id, loser_id, group_id],
        )
        .await;

    // Remap facts.object_id — same C0 stamp, object-position analogue. Both UNION
    // arms of `neighbours_of` (and `assertions_of`) filter `corroboration_inert = 0`,
    // so a neighbour reached via the loser's OBJECT-position fact must be equally
    // inerted (impl-spec §C0 DoD: "BOTH arms of neighbours_of's UNION").
    // ADR-029d: scoped by `object_group_id` — same rationale as the subject-side
    // remap above (`(object_id, object_group_id) → entities(id, group_id)` FK).
    let r2 = if r1.is_ok() {
        graph
            .conn
            .execute(
                "UPDATE facts SET object_id = ?1, corroboration_inert = 1 \
                 WHERE object_id = ?2 AND object_group_id = ?3",
                libsql::params![keeper_id, loser_id, group_id],
            )
            .await
    } else {
        r1
    };

    // Remap episodic_edges.entity_id onto the keeper.
    // Migration 017: episodic_edges carries UNIQUE(episode_id, entity_id,
    // entity_group_id). When keeper and loser both link the same episode the
    // remap collides; `UPDATE OR IGNORE` skips those rows (the keeper already
    // owns that presence edge — the loser's is the same presence fact about the
    // now-merged entity), then `DELETE` removes the now-orphaned loser rows so no
    // dangling entity_id survives the merge.
    // ADR-029d: both statements scoped by `entity_group_id` — an unscoped
    // `WHERE entity_id = loser_id` would remap/delete another namespace's
    // presence edges for the same id (the same cross-namespace corruption
    // class as the facts remap above).
    let r3 =
        if r2.is_ok() {
            match graph
                .conn
                .execute(
                    "UPDATE OR IGNORE episodic_edges SET entity_id = ?1 \
                 WHERE entity_id = ?2 AND entity_group_id = ?3",
                    libsql::params![keeper_id, loser_id, group_id],
                )
                .await
            {
                Ok(_) => graph
                    .conn
                    .execute(
                        "DELETE FROM episodic_edges WHERE entity_id = ?1 AND entity_group_id = ?2",
                        libsql::params![loser_id, group_id],
                    )
                    .await,
                Err(e) => Err(e),
            }
        } else {
            r2
        };

    // TD-203 D1 — expire the facts the re-points above just collapsed onto the
    // keeper on BOTH endpoints. `A pred B` is meaningful only while A and B are
    // distinct entities; once merged it reads `A pred A` and asserts nothing.
    //
    // Left unguarded this manufactured 41 live self-loops on the shipped LoCoMo
    // corpus — 18 of them on the reserved `potential_alias` predicate, a shape
    // L4 disambiguation CANNOT emit (it compares a NEW entity against a
    // DIFFERENT existing one), which is how the corruption was first spotted.
    // Attributed to `canonicalize` merges via `graph_mutation_log` rows 2-3.
    //
    // SOFT expiry, not DELETE: `expired_at` is the project's reversible
    // tombstone, and `pre_state.self_loops_expired` carries these ids so
    // `unmerge` (reversal.rs step (d2)) clears the stamp after un-pointing the
    // endpoints — at which point the fact is meaningful again. A hard DELETE
    // would make the merge irreversible in violation of ADR-073.
    //
    // Placed AFTER r2 (the endpoint re-points) because the ids were predicted
    // pre-merge; running it earlier would expire facts that are still live and
    // still meaningful.
    let r3b = if r3.is_ok() && !doomed_self_loops.is_empty() {
        // RFC3339, matching `TemporalGraph::invalidate_fact` (`graph/facts.rs:142`)
        // so every `expired_at` in the table is written in one format.
        let now = chrono::Utc::now().to_rfc3339();
        let mut last = Ok(0u64);
        for fact_id in &doomed_self_loops {
            last = graph
                .conn
                .execute(
                    "UPDATE facts SET expired_at = ?1 WHERE id = ?2 AND expired_at IS NULL",
                    libsql::params![now.clone(), *fact_id],
                )
                .await;
            if last.is_err() {
                break;
            }
        }
        if last.is_ok() {
            counter!(
                "kremory.merge.self_loop_facts_expired_total",
                "site" => site.as_str()
            )
            .increment(doomed_self_loops.len() as u64);
            tracing::info!(
                target: "kremory.merge",
                loser_id,
                keeper_id,
                group_id,
                expired = doomed_self_loops.len(),
                "kremory.merge.self_loop_facts_expired"
            );
        }
        last
    } else {
        r3
    };

    // Accumulate access_count into keeper. ADR-029d: scoped by `group_id` — the
    // keeper lives in THIS namespace (it's the survivor of a merge this caller
    // scoped to `group_id`); an unscoped write would hit whichever namespace's
    // row libSQL matches first if the same id also exists elsewhere.
    let r4 = if r3b.is_ok() && loser_access_count > 0 {
        graph
            .conn
            .execute(
                "UPDATE entities SET access_count = access_count + ?1 WHERE id = ?2 AND group_id = ?3",
                libsql::params![loser_access_count, keeper_id, group_id],
            )
            .await
    } else {
        r3b
    };

    // Site #6 (ADR-063 §"six sites" #6 / SYNTHESIS §2): combine the loser's
    // ner_confidence into the keeper via noisy-OR (a + b − a·b), null-safe — merging
    // the same real-world entity must never LOWER its confidence. This is the
    // deterministic merged-confidence FORMULA half; the reject-FLOOR gate is deferred
    // (S4-blocked — the floor value + null-prevalence are both unmeasured, R4 open
    // item; building it now would hardcode a guessed floor).
    // ADR-029d: both the keeper read and the keeper write are scoped by
    // `group_id` — same rationale as the access_count accumulate above.
    let r4b =
        if r4.is_ok() {
            let keeper_ner_confidence: Option<f32> = match graph
                .conn
                .query(
                    "SELECT ner_confidence FROM entities WHERE id = ?1 AND group_id = ?2",
                    libsql::params![keeper_id, group_id],
                )
                .await
            {
                Ok(mut kr) => match kr.next().await {
                    Ok(Some(row)) => row.get::<Option<f64>>(0).ok().flatten().map(|v| v as f32),
                    _ => None,
                },
                Err(_) => None,
            };
            match crate::core::confidence::merged_confidence(
                keeper_ner_confidence,
                loser_ner_confidence,
            ) {
                Some(merged) => graph
                    .conn
                    .execute(
                        "UPDATE entities SET ner_confidence = ?1 WHERE id = ?2 AND group_id = ?3",
                        libsql::params![f64::from(merged), keeper_id, group_id],
                    )
                    .await,
                None => Ok(0), // both null — nothing to combine
            }
        } else {
            r4
        };

    // Delete loser FTS entry.
    // TODO(ADR-029d): entities_fts is NOT group-aware (FTS5 virtual table has no
    // group_id column) — under per-namespace-open, two same-id entities in
    // different namespaces share one ambiguous FTS row. This delete is
    // unavoidably unscoped; it is a coarseness in cross-namespace full-text
    // search, NOT the FK-corruption bug (facts/episodic_edges are the FK'd
    // surfaces and ARE scoped above). See TD-130.
    let r5 = if r4b.is_ok() {
        graph
            .conn
            .execute(
                "DELETE FROM entities_fts WHERE entity_id = ?1",
                libsql::params![loser_id],
            )
            .await
    } else {
        r4b
    };

    // Delete loser entity row. ADR-029d: scoped by `group_id` — without this,
    // deleting `WHERE id = loser_id` alone would delete the loser row in EVERY
    // namespace it exists in, not just this merge's namespace (composite PK is
    // `(id, group_id)`).
    let r6 = if r5.is_ok() {
        graph
            .conn
            .execute(
                "DELETE FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![loser_id, group_id],
            )
            .await
    } else {
        r5
    };

    // Optional identity_verdict_audit INSERT — spec §5.1 RISK-003: runs INSIDE
    // this same BEGIN IMMEDIATE transaction, not as a separate best-effort
    // write. `r7` folds into the same all-must-succeed chain as r1-r6 so a
    // failed audit insert rolls back the merge too (same commit/rollback
    // boundary as the remap itself).
    let r7: std::result::Result<u64, libsql::Error> = if r6.is_ok() {
        if let Some(a) = audit.as_ref() {
            graph
                .conn
                .execute(
                    "INSERT INTO identity_verdict_audit \
                     (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
                      llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    libsql::params![
                        a.site,
                        a.group_id,
                        a.candidate_a,
                        a.candidate_b,
                        a.cosine.map(f64::from),
                        a.structural_signal,
                        a.verdict.map(|v| v.is_same_entity),
                        a.verdict.map(|v| f64::from(v.confidence)),
                        a.verdict.map(|v| v.reasoning.clone()),
                        a.decision,
                        a.run_id,
                    ],
                )
                .await
        } else {
            Ok(0)
        }
    } else {
        r6.map(|_| 0)
    };

    match r7 {
        Ok(_) => {
            guard.commit().await?;
            // Spec §8.1: the `graph_mutation_log` row is a durable in-txn write
            // (correct on rollback); its SUMMARY counter must fire only AFTER the
            // real commit, never mid-txn. When `committed_here` is false the
            // durable commit belongs to an outer txn (no current caller nests —
            // all three merge sites open their own txn) so the counter would move
            // to that outer committer; today it is always `true`.
            if committed_here {
                // Spec §8.2 canonical mutation-log counter (renamed from the
                // sub-phase-1b `kremory.dream.provenance.snapshot_total`). Label set
                // is `{kind, source}` ONLY — `group_id` is UNBOUNDED cardinality and
                // MUST NOT be a metric label (ADR-071 Item 5 discipline; spec §8.2
                // reconciled). It is carried as a `tracing::` event FIELD instead, so
                // per-namespace attribution stays observable without exploding the
                // metric's label dimension.
                counter!(
                    "kremory.graph.mutation_logged_total",
                    "kind" => "entity_merge",
                    "source" => "entity_merge_executor",
                )
                .increment(1);
                tracing::debug!(
                    target: "kremory.graph.provenance",
                    kind = "entity_merge",
                    group_id = %snapshot_group_id,
                    source = "entity_merge_executor",
                    "kremory.graph.mutation_logged: reversible-mutation provenance row committed"
                );
                // TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): recompute
                // + persist the keeper's name embedding AFTER the durable commit —
                // the merge itself already succeeded, so a re-embed failure here is
                // best-effort (never rolls back a durable merge) and must not hold
                // the write-lock open across an external embedder call. Without this,
                // the keeper's stored embedding stays at its PRE-merge value forever
                // (register-verified: no production dream/reconciliation site
                // re-persisted a recomputed embedding — `discover_types.rs:735` only
                // recomputes for a similarity comparison, never persists).
                if let Some(emb) = embedder {
                    // TD-143 KNOWN GAP (documented, not silent): this re-embed writes
                    // into `entities.embedding` — the same index ingest-time writes
                    // document-prefix when `embed_task_prefix_enabled` is on — but this
                    // call site does NOT thread that knob (would require adding it to
                    // `EntityMergeParams`/`ApplyMergeWithAuditParams` and both callers,
                    // `canonicalize_surface_forms` L5 and
                    // `core::dream::consolidation::cross_episode`). Left unprefixed:
                    // out of scope for TD-143 (best-effort, dream-phase-only, tracked
                    // separately as TD-112) — but means a keeper re-embedded via THIS
                    // path after the knob is flipped on stays in the unprefixed task
                    // space until the next full re-ingest/backfill. Revisit if/when the
                    // knob defaults on.
                    //
                    // Attribute the re-embed outcome per merge site (Quinn L3) so
                    // once multiple sites thread the embedder, their re-embed
                    // success/failure rates stay distinguishable. `site` is a
                    // BOUNDED enum tag (3 variants) — safe as a metric label.
                    let site_label = site.as_str();
                    match emb.embed_dyn(keeper_id).await {
                        Ok(fresh_embedding) => {
                            match graph
                                .update_entity_embedding(keeper_id, &fresh_embedding)
                                .await
                            {
                                Ok(()) => {
                                    counter!(
                                        "kremory.graph.merge_reembed_total",
                                        "outcome" => "success",
                                        "site" => site_label,
                                    )
                                    .increment(1);
                                }
                                Err(e) => {
                                    counter!(
                                        "kremory.graph.merge_reembed_total",
                                        "outcome" => "persist_failed",
                                        "site" => site_label,
                                    )
                                    .increment(1);
                                    tracing::warn!(
                                        target: "kremory.graph.merge_reembed",
                                        keeper_id = %keeper_id,
                                        site = %site_label,
                                        error = %e,
                                        "kremory.graph.merge_reembed_persist_failed: keeper \
                                         embedding left at its stale pre-merge value"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            counter!(
                                "kremory.graph.merge_reembed_total",
                                "outcome" => "embed_failed",
                                "site" => site_label,
                            )
                            .increment(1);
                            tracing::warn!(
                                target: "kremory.graph.merge_reembed",
                                keeper_id = %keeper_id,
                                site = %site_label,
                                error = %e,
                                "kremory.graph.merge_reembed_embed_failed: keeper embedding \
                                 left at its stale pre-merge value"
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            let _ = guard.rollback().await;
            Err(e.into())
        }
    }
}

/// Capture the COMPLETE `entity_merge` pre-state (spec §2.3) and INSERT the
/// `graph_mutation_log` row, on `graph.conn` — the SAME connection/txn as the
/// destructive merge statements (the caller holds an open `BeginGuard`), so the
/// snapshot commits atomically with the merge (spec §4 / §8.1). MUST be called
/// BEFORE the destructive block: every captured value (loser row, per-fact
/// prior `corroboration_inert`, keeper's pre-overwrite `access_count` /
/// `ner_confidence`, per-edge collision flag) only exists until the merge's
/// UPDATEs run (spec §2.4 DERIVATION deps).
///
/// A missing loser or keeper entity is a hard `Error` (parse-loudly, spec §2.1):
/// a merge whose endpoints we cannot snapshot could not be reversed, so we fail
/// loudly rather than persist an un-undoable log row.
/// Bundled parameters for [`snapshot_merge_pre_state`] — args-as-object per
/// TD-042 (`clippy.toml` `too-many-arguments-threshold = 3`).
struct MergeSnapshotParams<'a> {
    loser_id: &'a str,
    keeper_id: &'a str,
    /// ADR-029d: the namespace this merge is scoped to. See
    /// [`EntityMergeParams::group_id`] doc — every read below MUST filter on
    /// this value so the captured pre-state matches EXACTLY what the caller's
    /// scoped merge statements go on to touch (an unscoped snapshot read would
    /// desync from a scoped merge write, corrupting the undo/audit record even
    /// after the FK-crash itself is fixed).
    group_id: &'a str,
    site: MergeSite,
    audit: Option<&'a IdentityVerdictAuditRow<'a>>,
    /// The authoritative structural-corroboration bool for `inputs.structural_signal`
    /// (spec §2.3, Quinn L3) — threaded from the call-site, NOT derived from
    /// `audit` (which is `None` on the non-LLM structural sites).
    structural_signal: bool,
}

/// TD-203 D1 — the facts a merge of `loser` into `keeper` would collapse into a
/// self-loop (`X pred X`), which asserts nothing.
///
/// Three shapes qualify, all scoped to the merge's namespace (ADR-029d) and all
/// still live: `loser -> keeper`, `keeper -> loser`, and an already
/// self-referential `loser -> loser`. After the endpoint re-points at
/// `apply_merge_with_audit` every one of them reads `keeper -> keeper`.
///
/// Deliberately does NOT match a PRE-EXISTING `keeper -> keeper` row: this merge
/// did not create it, so this merge must not expire it (and `unmerge` would then
/// revive a fact it never killed).
///
/// Declared once and used by BOTH the snapshot and the expiry so the two cannot
/// drift apart — the pair is a single logical predicate evaluated twice inside
/// one transaction.
const SELF_LOOP_FACT_IDS_SQL: &str = "SELECT id FROM facts \
     WHERE expired_at IS NULL \
       AND subject_group_id = ?3 AND object_group_id = ?3 \
       AND ((subject_id = ?1 AND object_id = ?2) \
         OR (subject_id = ?2 AND object_id = ?1) \
         OR (subject_id = ?1 AND object_id = ?1))";

/// Bundled parameters for [`self_loop_fact_ids`] — args-as-object per TD-042
/// (workspace clippy `too_many_arguments` threshold is 3, and
/// `#[allow(clippy::*)]` is banned in `src/`). `graph` stays a lead positional
/// param (receiver-like dep, project convention).
#[derive(Clone, Copy)]
struct SelfLoopScanParams<'a> {
    /// The entity being merged AWAY.
    loser_id: &'a str,
    /// The surviving entity both endpoints will point at.
    keeper_id: &'a str,
    /// ADR-029d — the namespace this merge is scoped to.
    group_id: &'a str,
}

/// Collect the fact ids [`SELF_LOOP_FACT_IDS_SQL`] identifies.
async fn self_loop_fact_ids(
    graph: &TemporalGraph,
    params: SelfLoopScanParams<'_>,
) -> Result<Vec<i64>> {
    let SelfLoopScanParams {
        loser_id,
        keeper_id,
        group_id,
    } = params;
    let mut rows = graph
        .conn
        .query(
            SELF_LOOP_FACT_IDS_SQL,
            libsql::params![loser_id, keeper_id, group_id],
        )
        .await?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next().await? {
        ids.push(row.get::<i64>(0)?);
    }
    Ok(ids)
}

/// Returns the snapshot's `group_id` and — TD-203 D1 — the fact ids this merge
/// must expire because re-pointing collapses both their endpoints onto the
/// keeper. The ids are returned rather than re-derived by the caller so the
/// value written into `pre_state.self_loops_expired` and the value the caller
/// expires are the SAME list by construction, not two queries that agree today.
async fn snapshot_merge_pre_state(
    graph: &TemporalGraph,
    params: MergeSnapshotParams<'_>,
) -> Result<(String, Vec<i64>)> {
    let MergeSnapshotParams {
        loser_id,
        keeper_id,
        group_id,
        site,
        audit,
        structural_signal,
    } = params;
    // (1) loser entity row — all 11 live columns (spec §2.3(1); no `label`,
    //     dropped Mig 009). ADR-029d: scoped by `group_id` so the snapshot
    //     matches the caller's scoped loser DELETE (`WHERE id = loser AND
    //     group_id = ?`) exactly — an unscoped SELECT could capture a
    //     DIFFERENT namespace's same-id row under per-namespace-open.
    let loser_entity_row = {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, group_id, properties, embedding, recorded_at, updated_at, \
                        access_count, entity_type_id, entity_type_source, \
                        entity_type_assigned_at, ner_confidence \
                 FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![loser_id, group_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "reversible-mutations snapshot: loser entity `{loser_id}` not found in \
                 namespace `{group_id}` — cannot capture a reversible pre-state for this merge"
            )));
        };
        // `embedding` is an F32_BLOB; capture the raw bytes faithfully and
        // base64-encode for JSON transport (spec §2.3(1)). A missed byte here is
        // silent embedding loss on undo — the exact failure this sub-phase guards.
        let embedding: Option<Vec<u8>> = row.get::<Option<Vec<u8>>>(3)?;
        LoserEntityRow {
            id: row.get::<String>(0)?,
            group_id: row.get::<String>(1)?,
            properties: row.get::<Option<String>>(2)?,
            embedding_b64: embedding.as_deref().map(|b| BASE64_STANDARD.encode(b)),
            recorded_at: row.get::<String>(4)?,
            updated_at: row.get::<Option<String>>(5)?,
            access_count: row.get::<i64>(6)?,
            entity_type_id: row.get::<i64>(7)?,
            entity_type_source: row.get::<Option<String>>(8)?,
            entity_type_assigned_at: row.get::<Option<String>>(9)?,
            ner_confidence: row.get::<Option<f64>>(10)?,
        }
    };
    // Recorded provenance group_id — equals the caller's `group_id` (the row
    // above is now scoped to it); captured from the row itself so the
    // persisted snapshot reflects exactly what was read, not an assumption.
    let captured_group_id = loser_entity_row.group_id.clone();

    // (2) keeper's PRE-merge access_count + ner_confidence (spec §2.3(2)) —
    //     captured BEFORE the merge's accumulate + noisy-OR overwrite (both
    //     non-invertible), so undo restores these exact values, not a subtraction.
    //     ADR-029d: scoped by `group_id` — the keeper lives in this namespace.
    let keeper_pre = {
        let mut rows = graph
            .conn
            .query(
                "SELECT access_count, ner_confidence FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![keeper_id, group_id],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Err(Error::Other(anyhow::anyhow!(
                "reversible-mutations snapshot: keeper entity `{keeper_id}` not found in \
                 namespace `{group_id}` — cannot capture a reversible pre-state for this merge"
            )));
        };
        KeeperPre {
            id: keeper_id.to_string(),
            access_count: row.get::<i64>(0)?,
            ner_confidence: row.get::<Option<f64>>(1)?,
        }
    };

    // (3) facts re-pointed loser→keeper (spec §2.3(3)). The merge stamps each
    //     `corroboration_inert = 1`; capture the PRIOR flag per (fact_id,
    //     endpoint) so undo restores it — a fact already inert from an EARLIER
    //     merge must NOT be cleared (the monotone-undo trap, §12 CH-3). ADR-029d:
    //     both SELECT predicates now filter on `subject_group_id`/`object_group_id`
    //     to match the merge's scoped endpoint UPDATEs exactly (subject then
    //     object). A self-referential fact (loser on BOTH endpoints) is captured
    //     twice, once per endpoint — correct, undo reverts both.
    let mut repointed_facts: Vec<RepointedFact> = Vec::new();
    {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, corroboration_inert FROM facts \
                 WHERE subject_id = ?1 AND subject_group_id = ?2",
                libsql::params![loser_id, group_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            repointed_facts.push(RepointedFact {
                fact_id: row.get::<i64>(0)?,
                endpoint: FactEndpoint::Subject,
                prior_corroboration_inert: row.get::<i64>(1)?,
            });
        }
    }
    {
        let mut rows = graph
            .conn
            .query(
                "SELECT id, corroboration_inert FROM facts \
                 WHERE object_id = ?1 AND object_group_id = ?2",
                libsql::params![loser_id, group_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            repointed_facts.push(RepointedFact {
                fact_id: row.get::<i64>(0)?,
                endpoint: FactEndpoint::Object,
                prior_corroboration_inert: row.get::<i64>(1)?,
            });
        }
    }

    // (4) loser episodic_edges remapped/orphan-deleted by the merge (spec
    //     §2.3(4)). Capture every loser edge FIRST (drain the cursor), THEN
    //     compute its `collided` flag: does the keeper ALREADY own an edge for
    //     the same (episode_id, entity_group_id)? Under
    //     UNIQUE(episode_id, entity_id, entity_group_id) (Mig 017) the merge's
    //     `UPDATE OR IGNORE` drops a colliding loser edge (undo re-INSERTs it
    //     under the loser) and re-points a non-colliding one (undo re-points it
    //     back) — §4.2 step 4. Cursors never overlap: the loser edges are drained
    //     into a Vec before the per-edge collision sub-queries run. ADR-029d:
    //     scoped by `entity_group_id` to match the merge's scoped
    //     `UPDATE OR IGNORE episodic_edges ... AND entity_group_id = ?` exactly.
    let loser_edges: Vec<(i64, String, String, String, String)> = {
        let mut rows = graph
            .conn
            .query(
                "SELECT episode_id, entity_group_id, entity_id, role, recorded_at \
                 FROM episodic_edges WHERE entity_id = ?1 AND entity_group_id = ?2",
                libsql::params![loser_id, group_id],
            )
            .await?;
        let mut v = Vec::new();
        while let Some(row) = rows.next().await? {
            v.push((
                row.get::<i64>(0)?,
                row.get::<String>(1)?,
                row.get::<String>(2)?,
                row.get::<String>(3)?,
                row.get::<String>(4)?,
            ));
        }
        v
    };
    let mut episodic_edges: Vec<EpisodicEdgeSnapshot> = Vec::with_capacity(loser_edges.len());
    for (episode_id, entity_group_id, entity_id, role, recorded_at) in loser_edges {
        let collided = {
            let mut rows = graph
                .conn
                .query(
                    "SELECT 1 FROM episodic_edges \
                     WHERE entity_id = ?1 AND episode_id = ?2 AND entity_group_id = ?3 \
                     LIMIT 1",
                    libsql::params![keeper_id, episode_id, entity_group_id.clone()],
                )
                .await?;
            rows.next().await?.is_some()
        };
        episodic_edges.push(EpisodicEdgeSnapshot {
            episode_id,
            entity_group_id,
            collided,
            cols: EpisodicEdgeCols {
                entity_id,
                role,
                recorded_at,
            },
        });
    }

    // (5) TD-203 D1 — facts this merge will collapse to `keeper -> keeper`.
    //     Captured here, inside the same txn and BEFORE the destructive
    //     re-points, for the same reason as (1)-(4): the mutation-log row is
    //     written before the writes, so the pre-state must PREDICT the affected
    //     set rather than observe it afterwards.
    let self_loops_expired = self_loop_fact_ids(
        graph,
        SelfLoopScanParams {
            loser_id,
            keeper_id,
            group_id: &captured_group_id,
        },
    )
    .await?;

    let pre_state = EntityMergePreState {
        loser_entity_row,
        keeper_pre,
        repointed_facts,
        episodic_edges,
        self_loops_expired: self_loops_expired.clone(),
    };

    // `inputs`: the SORTED unordered pair is the nogood key (spec §6.2 / §2.3) —
    // a keeper/loser role-flip between passes cannot evade the anti-re-merge ban.
    // `cosine` comes from the `audit` row when present (Site #5 passes it; the
    // structural/cosine sites pass `audit = None`). `structural_signal` is the
    // AUTHORITATIVE bool threaded from the call-site (Quinn L3) — it is accurate
    // even on the `audit = None` cross-episode path, so the inspect SEE-surface
    // reports the merge's real provenance.
    let (pair_lo, pair_hi) = if loser_id <= keeper_id {
        (loser_id.to_string(), keeper_id.to_string())
    } else {
        (keeper_id.to_string(), loser_id.to_string())
    };
    let inputs = MergeInputs {
        pair_lo,
        pair_hi,
        keeper: keeper_id.to_string(),
        loser: loser_id.to_string(),
        site,
        cosine: audit.and_then(|a| a.cosine).map(f64::from),
        structural_signal,
    };

    // Our OWN structured emit (never LLM-authored, spec §2.1) — serialization is
    // deterministic; a failure is a hard error, never a silent skip.
    let pre_state_json = serde_json::to_string(&pre_state)
        .map_err(|e| Error::Other(anyhow::anyhow!("serialize merge pre_state: {e}")))?;
    let inputs_json = serde_json::to_string(&inputs)
        .map_err(|e| Error::Other(anyhow::anyhow!("serialize merge inputs: {e}")))?;

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.graph.provenance",
            kind = "entity_merge",
            group_id = %captured_group_id,
            pre_state = %pre_state_json,
            inputs = %inputs_json,
            "reversible-mutations merge snapshot captured (pre-destroy)"
        );
    }

    // INSERT on `graph.conn` = the SAME txn as the destructive block (spec §2.1
    // / §8.1). `undone_at` (NULL = live) + `enabled_at_time` (DEFAULT 1) take
    // their DDL defaults; `kind` is the fixed `entity_merge` string.
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                "entity_merge",
                captured_group_id.clone(),
                now,
                pre_state_json,
                inputs_json
            ],
        )
        .await?;

    // Return the namespace this merge scoped, for the post-commit
    // `mutation_logged_total{group_id}` label (spec §8.2).
    Ok((captured_group_id, self_loops_expired))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::InsertEntityWithGroupParams;
    use crate::core::schema::TemporalGraph;

    // ── Helper ────────────────────────────────────────────────────────────────

    /// Insert an entity with a given embedding and description into `group_id`.
    // Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a
    // documented too_many_arguments allow); TD-042 args-as-object targets `src/`
    // production fns, not `#[cfg(test)]` seeders.
    #[allow(clippy::too_many_arguments)]
    async fn insert_entity_with_embedding(
        graph: &TemporalGraph,
        id: &str,
        group_id: &str,
        description: &str,
        embedding: &[f32],
    ) {
        let props = serde_json::json!({ "name": id, "description": description });
        // entity_type_id = 0 is the "Entity" catch-all sentinel.
        graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0u32,
                properties: props,
                group_id: Some(group_id),
            })
            .await
            .expect("insert entity");
        graph
            .set_entity_embedding(id, embedding)
            .await
            .expect("set embedding");
    }

    /// Build a unit-normalised f32 embedding of given dimension where all components
    /// are equal. Useful as "same direction" vectors for high cosine similarity.
    fn unit_vec(dim: usize) -> Vec<f32> {
        let v = 1.0_f32 / (dim as f32).sqrt();
        vec![v; dim]
    }

    /// Build a unit-normalised vector orthogonal to `unit_vec` in 384 dims.
    /// Sets component 0 to 1 and all others to 0.
    fn orthogonal_vec() -> Vec<f32> {
        let mut v = vec![0.0_f32; 384];
        v[0] = 1.0;
        v
    }

    /// TD-112 test helper: cosine distance between an entity's PERSISTED
    /// embedding and an ad-hoc probe vector, via the same `vector_distance_cos`
    /// SQL function `find_merge_pairs` uses in production. Reading back through
    /// SQL (rather than raw bytes) sidesteps endianness/precision concerns —
    /// only the semantic (cosine) equality matters for this assertion.
    /// Returns `None` if the entity has no embedding (zero-magnitude vector).
    async fn entity_embedding_distance(
        graph: &TemporalGraph,
        id: &str,
        probe: &[f32],
    ) -> Option<f32> {
        let vec_str = format!(
            "vector32('[{}]')",
            probe
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut rows = graph
            .conn
            .query(
                &format!(
                    "SELECT vector_distance_cos(embedding, {vec_str}) FROM entities WHERE id = ?1"
                ),
                libsql::params![id],
            )
            .await
            .expect("query entity embedding distance");
        let row = rows
            .next()
            .await
            .expect("row")
            .expect("entity must exist for embedding-distance probe");
        row.get::<Option<f64>>(0)
            .expect("distance column")
            .map(|d| d as f32)
    }

    // ── TD-223 write-time merge-chain detector ────────────────────────────────
    //
    // Proven in BOTH directions, per the discipline that killed the two previous
    // TD-222 remedies: a guard tested only on the cases it should catch is
    // untested. `chain_detector_silent_on_independent_merges` and
    // `chain_detector_is_namespace_scoped` are the over-block half — revert the
    // detector to an unscoped or unconditional form and they fail.

    /// Apply a real merge so `graph_mutation_log` gets a genuine producer-written
    /// row. Deliberately NOT a hand-inserted log row: hand-shaped fixtures test a
    /// model of the producer, and this detector's whole job is to read what the
    /// producer actually writes.
    /// Pair is a tuple `(loser, keeper)` to stay at 3 params — `clippy.toml` sets
    /// `too-many-arguments-threshold = 3` and this crate bans `#[allow]` in `src`,
    /// which includes `#[cfg(test)]` modules living in `src` files.
    async fn merge(graph: &TemporalGraph, group_id: &str, pair: (&str, &str)) {
        apply_entity_merge(
            graph,
            EntityMergeParams {
                loser_id: pair.0,
                keeper_id: pair.1,
                group_id,
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("apply merge");
    }

    #[tokio::test]
    async fn chain_detector_fires_when_loser_previously_absorbed() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        for id in ["melanie", "caroline", "loved ones"] {
            insert_entity_with_embedding(&graph, id, "g_chain", id, &unit_vec(384)).await;
        }

        // The measured cascade's first hop: melanie -> caroline.
        merge(&graph, "g_chain", ("melanie", "caroline")).await;

        // The second hop is what the detector must see: caroline is now a
        // SURVIVOR, so absorbing it moves melanie's identity a second time to a
        // destination nobody adjudicated against melanie.
        let hit = detect_merge_chain_extension(&graph, "g_chain", ("caroline", "loved ones")).await;

        let (prior, kind) = hit.expect("chain extension must be detected");
        assert_eq!(kind, "loser_is_prior_survivor");
        assert_eq!(prior, "melanie", "must name the identity being moved twice");
    }

    #[tokio::test]
    async fn chain_detector_fires_when_keeper_was_previously_absorbed() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        for id in ["a", "b", "c"] {
            insert_entity_with_embedding(&graph, id, "g_victim", id, &unit_vec(384)).await;
        }
        merge(&graph, "g_victim", ("a", "b")).await;

        // `a` has ceased to be a distinct identity; merging INTO it is the other
        // half of the chain shape.
        let hit = detect_merge_chain_extension(&graph, "g_victim", ("c", "a")).await;

        let (prior, kind) = hit.expect("keeper-is-prior-victim must be detected");
        assert_eq!(kind, "keeper_is_prior_victim");
        assert_eq!(prior, "b");
    }

    #[tokio::test]
    async fn chain_detector_silent_on_independent_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        for id in ["a", "b", "c", "d"] {
            insert_entity_with_embedding(&graph, id, "g_indep", id, &unit_vec(384)).await;
        }
        merge(&graph, "g_indep", ("a", "b")).await;

        // Two disjoint merges are not a chain. If this fires, the detector is
        // flagging ordinary canonicalisation and would be pure noise on any
        // corpus with more than one merge.
        assert!(
            detect_merge_chain_extension(&graph, "g_indep", ("c", "d"))
                .await
                .is_none(),
            "independent merge must NOT be reported as a chain extension"
        );
    }

    #[tokio::test]
    async fn chain_detector_is_namespace_scoped() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "a", "ns_one", "a", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "b", "ns_one", "b", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "b", "ns_two", "b", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "c", "ns_two", "c", &unit_vec(384)).await;

        merge(&graph, "ns_one", ("a", "b")).await;

        // ADR-029d: the same `id` legitimately exists in two namespaces. An
        // unscoped read would see ns_one's `b -> keeper` row and manufacture a
        // chain in ns_two that does not exist — a false positive that grows with
        // every namespace, which is exactly how an unscoped alias query
        // previously matched 100 rows (`disambiguation/mod.rs:687`).
        assert!(
            detect_merge_chain_extension(&graph, "ns_two", ("b", "c"))
                .await
                .is_none(),
            "a merge in ANOTHER namespace must not count as a chain here"
        );
    }

    // ── T1: Empty group ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn t1_empty_group_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let report = canonicalize_surface_forms(&graph, "g_empty", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 0);
        assert_eq!(report.pairs_examined, 0);
        assert_eq!(report.group_id, "g_empty");
        assert_eq!(report.threshold_used, L5_CANONICALIZATION_THRESHOLD);
    }

    /// Migration 017 interaction: when keeper and loser BOTH have a presence edge
    /// to the same episode, remapping loser→keeper collides on
    /// UNIQUE(episode_id, entity_id, entity_group_id). `apply_merge`'s
    /// `UPDATE OR IGNORE` + cleanup `DELETE` must leave exactly one surviving edge
    /// on the keeper and zero dangling loser edges — no FK violation, no error.
    #[tokio::test]
    async fn merge_collapses_duplicate_episodic_edges() {
        use crate::core::graph::{InsertEpisodeParams, InsertEpisodicEdgeParams};
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "keeper", "g1", "keeper", &unit_vec(384)).await;
        insert_entity_with_embedding(&graph, "loser", "g1", "loser", &unit_vec(384)).await;
        let ep = graph
            .insert_episode(InsertEpisodeParams {
                content: "Keeper and loser appear together.",
                timestamp: chrono::Utc::now(),
                source_type: Some("transcript"),
                metadata: None,
            })
            .await
            .expect("episode");
        // Both entities link the SAME episode → post-merge they would collide.
        for ent in ["keeper", "loser"] {
            graph
                .insert_episodic_edge(InsertEpisodicEdgeParams {
                    episode_id: ep,
                    entity_id: ent,
                    entity_group_id: Some("g1"),
                    role: "mention",
                })
                .await
                .expect("edge");
        }

        apply_merge(
            &graph,
            ApplyMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                group_id: "g1",
                embedder: None,
            },
        )
        .await
        .expect("merge must not error on episodic-edge collision");

        let keeper_edges = graph.episodic_edges_for_entity("keeper").await.unwrap();
        assert_eq!(
            keeper_edges.len(),
            1,
            "keeper keeps exactly one presence edge after merge; got {keeper_edges:?}"
        );
        let loser_edges = graph.episodic_edges_for_entity("loser").await.unwrap();
        assert!(
            loser_edges.is_empty(),
            "no dangling loser edges after merge; got {loser_edges:?}"
        );
    }

    // ── T2: Single entity ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn t2_single_entity_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "e1", "g_single", "A person.", &unit_vec(384)).await;
        let report = canonicalize_surface_forms(&graph, "g_single", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 0);
        assert_eq!(report.pairs_examined, 0);
    }

    // ── T3: No similar pairs (orthogonal embeddings) ──────────────────────────

    #[tokio::test]
    async fn t3_orthogonal_embeddings_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Five entities with mutually orthogonal embeddings.
        for i in 0..5usize {
            let mut emb = vec![0.0_f32; 384];
            emb[i] = 1.0; // each entity sits on a different axis
            insert_entity_with_embedding(
                &graph,
                &format!("orth_{i}"),
                "g_orth",
                &format!("Entity {i}"),
                &emb,
            )
            .await;
        }
        let report = canonicalize_surface_forms(&graph, "g_orth", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 0);
        assert_eq!(report.pairs_examined, 10); // C(5,2) = 10
    }

    // ── T3b: noisy-OR ner_confidence combination on merge (ADR-063 Site #6) ────

    #[tokio::test]
    async fn merge_combines_ner_confidence_via_noisy_or() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // Same surface-variant pair as T4 (merges); keeper = longer description.
        insert_entity_with_embedding(
            &graph,
            "alice johnson",
            "g_conf",
            "A detailed description of Alice Johnson, software engineer.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(&graph, "alice j", "g_conf", "Alice.", &unit_vec(384)).await;

        // Seed ner_confidence: keeper 0.6, loser 0.8 → noisy_or(0.6,0.8) = 0.92.
        graph
            .conn
            .execute(
                "UPDATE entities SET ner_confidence = 0.6 WHERE id = 'alice johnson'",
                (),
            )
            .await
            .expect("set keeper conf");
        graph
            .conn
            .execute(
                "UPDATE entities SET ner_confidence = 0.8 WHERE id = 'alice j'",
                (),
            )
            .await
            .expect("set loser conf");

        let report = canonicalize_surface_forms(&graph, "g_conf", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(report.merges_applied, 1, "the surface-variant pair merges");

        // Keeper ("alice johnson", longer desc) now carries noisy_or(0.6, 0.8) = 0.92
        // — the loser's confidence was COMBINED, not discarded (Site #6 formula half).
        let mut rows = graph
            .conn
            .query(
                "SELECT ner_confidence FROM entities WHERE id = 'alice johnson'",
                (),
            )
            .await
            .expect("query keeper conf");
        let conf: f64 = rows
            .next()
            .await
            .expect("row")
            .expect("keeper survives")
            .get::<Option<f64>>(0)
            .expect("conf col")
            .expect("keeper ner_confidence is set after merge");
        assert!(
            (conf - 0.92).abs() < 1e-4,
            "keeper ner_confidence must be noisy_or(0.6, 0.8) = 0.92, got {conf}"
        );
    }

    // ── T4: One pair above threshold → 1 merge, keeper = longer description ───

    #[tokio::test]
    async fn t4_one_pair_above_threshold_keeper_is_longer_description() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // Two very similar embeddings: both unit vectors → cosine ≈ 1.0 (above 0.8).
        // ADR-057: entity ids are SURFACE VARIANTS of the same name (Jaccard ≥ 0.5),
        // so the lexical merge gate permits the merge (this is what L5 is FOR — merging
        // variants of one entity, never unrelated names). "alice j" → "alice johnson".
        // e_short has the shorter description → it should become the loser.
        insert_entity_with_embedding(
            &graph,
            "alice johnson",
            "g_pair",
            "A detailed description of Alice Johnson, software engineer at Acme Corp.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(&graph, "alice j", "g_pair", "Alice.", &unit_vec(384)).await;

        let report = canonicalize_surface_forms(&graph, "g_pair", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");

        assert_eq!(report.merges_applied, 1, "expected exactly 1 merge");
        assert_eq!(report.pairs_examined, 1);

        // Verify: "alice johnson" (longer desc) survives, "alice j" is gone.
        let entities = graph.list_entities_in_group("g_pair").await.expect("list");
        let ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"alice johnson"), "keeper must survive");
        assert!(!ids.contains(&"alice j"), "loser must be deleted");
    }

    // ── T5: Multi-pair chain (A↔B, B↔C, A↔C) → all coalesce to 1 keeper ─────

    #[tokio::test]
    async fn t5_triplet_all_similar_coalesces_to_one_keeper() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // All three entities have the same unit vector → pairwise cosine = 1.0 (>0.8).
        // ADR-057: surface variants of one name (shared {alice, johnson} significant
        // tokens → Jaccard ≥ 0.5 pairwise) so the lexical gate permits all merges.
        // Longest description → "alice johnson engineer" is the keeper.
        insert_entity_with_embedding(&graph, "alice johnson", "g_tri", "Alice.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "alice johnson m",
            "g_tri",
            "Alice Johnson.",
            &unit_vec(384),
        )
        .await;
        insert_entity_with_embedding(
            &graph,
            "alice johnson engineer",
            "g_tri",
            "Alice Johnson, software engineer.",
            &unit_vec(384),
        )
        .await;

        let report = canonicalize_surface_forms(&graph, "g_tri", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");

        // 2 merges: a_short → keeper, b_mid → keeper
        assert_eq!(report.merges_applied, 2, "2 losers should be merged");
        assert_eq!(report.pairs_examined, 3); // C(3,2) = 3

        // Only 1 entity survives.
        let entities = graph.list_entities_in_group("g_tri").await.expect("list");
        assert_eq!(entities.len(), 1, "only 1 keeper must remain");
    }

    // ── T6: Idempotency — second run applies 0 merges ─────────────────────────

    #[tokio::test]
    async fn t6_idempotent_second_run_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // ADR-057: surface variants (Jaccard 2/3 ≥ 0.5) so the lexical gate permits merge.
        insert_entity_with_embedding(&graph, "sam carter", "g_idem", "Short.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "sam carter phd",
            "g_idem",
            "Longer description here.",
            &unit_vec(384),
        )
        .await;

        let r1 = canonicalize_surface_forms(&graph, "g_idem", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("first run");
        assert_eq!(r1.merges_applied, 1, "first run must merge 1 pair");

        let r2 = canonicalize_surface_forms(&graph, "g_idem", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("second run");
        assert_eq!(r2.merges_applied, 0, "second run must be a no-op");
    }

    // ── T7: Threshold boundary — strict `>` (0.8 does NOT merge, >0.8 does) ───

    #[tokio::test]
    async fn t7_threshold_boundary_strict_greater_than() {
        // Use a threshold of 0.99 and unit vectors (similarity ≈ 1.0) → should merge.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        // ADR-057: surface variants (Jaccard 2/3 ≥ 0.5) so the lexical gate permits merge.
        insert_entity_with_embedding(&graph, "omega corp", "g_bound", "Alpha.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "omega corp ltd",
            "g_bound",
            "Alpha extended.",
            &unit_vec(384),
        )
        .await;

        // Threshold 0.99 — unit vectors have cosine = 1.0 > 0.99 → merge.
        let r = canonicalize_surface_forms(&graph, "g_bound", 0.99)
            .await
            .expect("canonicalize");
        assert_eq!(
            r.merges_applied, 1,
            "cosine=1.0 > threshold=0.99 must merge"
        );
    }

    #[tokio::test]
    async fn t7b_orthogonal_below_threshold_no_merge() {
        // Orthogonal vectors → cosine = 0.0; way below any sane threshold.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        insert_entity_with_embedding(&graph, "o1", "g_bound2", "Alpha.", &orthogonal_vec()).await;
        let mut perp = vec![0.0_f32; 384];
        perp[1] = 1.0; // orthogonal to o1
        insert_entity_with_embedding(&graph, "o2", "g_bound2", "Beta.", &perp).await;

        let r = canonicalize_surface_forms(&graph, "g_bound2", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(r.merges_applied, 0, "orthogonal vectors must not merge");
    }

    // ── T8: Cross-group isolation ─────────────────────────────────────────────

    #[tokio::test]
    async fn t8_cross_group_isolation() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // group_a: two surface variants (Jaccard 2/3 ≥ 0.5) → merge within a (ADR-057).
        insert_entity_with_embedding(&graph, "nova labs", "group_a", "Short.", &unit_vec(384))
            .await;
        insert_entity_with_embedding(
            &graph,
            "nova labs inc",
            "group_a",
            "Longer description.",
            &unit_vec(384),
        )
        .await;

        // group_b: one entity only (no pair → no merge).
        insert_entity_with_embedding(
            &graph,
            "gb1",
            "group_b",
            "Unrelated entity.",
            &unit_vec(384),
        )
        .await;

        // Canonicalize group_a only.
        let ra = canonicalize_surface_forms(&graph, "group_a", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("group_a");
        assert_eq!(ra.merges_applied, 1, "group_a: 1 merge");

        // group_b must be untouched.
        let rb = canonicalize_surface_forms(&graph, "group_b", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("group_b");
        assert_eq!(rb.merges_applied, 0, "group_b: no merges");

        let gb_entities = graph.list_entities_in_group("group_b").await.expect("list");
        assert_eq!(gb_entities.len(), 1, "gb1 must still exist");
    }

    // ── T9: access_count is accumulated on keeper ─────────────────────────────

    #[tokio::test]
    async fn t9_access_count_accumulated_on_keeper() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");

        // ADR-057: surface variants (Jaccard 2/3 ≥ 0.5) so the lexical gate permits merge.
        insert_entity_with_embedding(&graph, "zeta group", "g_ac", "Short.", &unit_vec(384)).await;
        insert_entity_with_embedding(
            &graph,
            "zeta group holdings",
            "g_ac",
            "Longer desc.",
            &unit_vec(384),
        )
        .await;

        // Artificially bump access_count on loser via SQL.
        graph
            .conn
            .execute(
                "UPDATE entities SET access_count = 7 WHERE id = 'zeta group'",
                (),
            )
            .await
            .expect("bump loser access_count");

        let r = canonicalize_surface_forms(&graph, "g_ac", L5_CANONICALIZATION_THRESHOLD)
            .await
            .expect("canonicalize");
        assert_eq!(r.merges_applied, 1);

        // Keeper should have accumulated loser's access_count (0 + 7 = 7).
        let entities = graph.list_entities_in_group("g_ac").await.expect("list");
        let keeper = entities
            .iter()
            .find(|e| e.id == "zeta group holdings")
            .expect("keeper");
        assert_eq!(
            keeper.access_count, 7,
            "keeper must accumulate loser access_count"
        );
    }

    // ── ADR-067 §C0: apply_entity_merge stamps corroboration_inert on remapped facts ──

    /// Insert a bare entity (no embedding needed — these tests exercise the merge
    /// executor directly via `apply_entity_merge`, not the cosine-gated
    /// `canonicalize_surface_forms` path).
    async fn insert_bare_entity(graph: &TemporalGraph, id: &str, group_id: &str) {
        graph
            .insert_entity_with_group(crate::core::graph::InsertEntityWithGroupParams {
                id,
                entity_type_id: 0,
                properties: serde_json::json!({ "name": id }),
                group_id: Some(group_id),
            })
            .await
            .expect("insert bare entity");
    }

    /// Plant a relational fact `subject --predicate--> object` directly via SQL
    /// (mirrors `cross_episode.rs`'s test `fact_rel` helper).
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn plant_fact_rel(
        graph: &TemporalGraph,
        gid: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO facts \
                 (subject_id, predicate, object_id, valid_from, recorded_at, group_id, \
                  subject_group_id, object_group_id, confidence) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1.0)",
                libsql::params![subject, predicate, object, now.clone(), now, gid, gid, gid],
            )
            .await
            .expect("plant relational fact");
    }

    async fn corroboration_inert_of(graph: &TemporalGraph, fact_id: i64) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT corroboration_inert FROM facts WHERE id = ?1",
                libsql::params![fact_id],
            )
            .await
            .expect("query corroboration_inert");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("corroboration_inert col")
    }

    async fn fact_id_of(graph: &TemporalGraph, subject: &str, predicate: &str) -> i64 {
        let mut rows = graph
            .conn
            .query(
                "SELECT id FROM facts WHERE subject_id = ?1 AND predicate = ?2",
                libsql::params![subject, predicate],
            )
            .await
            .expect("query fact id");
        rows.next()
            .await
            .expect("row")
            .expect("present")
            .get::<i64>(0)
            .expect("id col")
    }

    #[tokio::test]
    async fn apply_entity_merge_stamps_corroboration_inert_on_subject_endpoint() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_c0_subj";
        insert_bare_entity(&graph, "keeper", gid).await;
        insert_bare_entity(&graph, "loser", gid).await;
        insert_bare_entity(&graph, "untouched_subject", gid).await;
        insert_bare_entity(&graph, "neighbour", gid).await;

        // loser's fact (loser is SUBJECT) — endpoint gets remapped to keeper.
        plant_fact_rel(&graph, gid, "loser", "knows", "neighbour").await;
        // An UNRELATED fact (neither endpoint touches the merge) must stay live (=0).
        plant_fact_rel(&graph, gid, "untouched_subject", "knows", "neighbour").await;

        let remapped_id = fact_id_of(&graph, "loser", "knows").await;
        let untouched_id = fact_id_of(&graph, "untouched_subject", "knows").await;

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                group_id: gid,
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("merge");

        // The remapped fact's endpoint is now `keeper` (subject_id rewritten) AND it
        // must be stamped corroboration_inert = 1.
        assert_eq!(
            corroboration_inert_of(&graph, remapped_id).await,
            1,
            "fact whose subject_id endpoint was rewritten by the merge must be \
             corroboration_inert = 1"
        );
        // The untouched fact (no endpoint rewritten) must remain corroboration_inert = 0.
        assert_eq!(
            corroboration_inert_of(&graph, untouched_id).await,
            0,
            "fact NOT touched by the merge remap must remain corroboration_inert = 0"
        );
    }

    #[tokio::test]
    async fn apply_entity_merge_stamps_corroboration_inert_on_object_endpoint() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_c0_obj";
        insert_bare_entity(&graph, "keeper", gid).await;
        insert_bare_entity(&graph, "loser", gid).await;
        insert_bare_entity(&graph, "asserter", gid).await;

        // `asserter --knows--> loser`: loser is the OBJECT here. Verifies the C0 stamp
        // fires equally on the object_id UPDATE branch (impl-spec §C0 DoD: "a neighbour
        // inherited via the loser's OBJECT-position fact is equally inert").
        plant_fact_rel(&graph, gid, "asserter", "knows", "loser").await;
        let remapped_id = fact_id_of(&graph, "asserter", "knows").await;

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                group_id: gid,
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("merge");

        assert_eq!(
            corroboration_inert_of(&graph, remapped_id).await,
            1,
            "fact whose object_id endpoint was rewritten by the merge must be \
             corroboration_inert = 1"
        );
        // object_id itself must have been remapped to keeper.
        let mut rows = graph
            .conn
            .query(
                "SELECT object_id FROM facts WHERE id = ?1",
                libsql::params![remapped_id],
            )
            .await
            .expect("query object_id");
        let object_id: Option<String> = rows
            .next()
            .await
            .expect("row")
            .expect("present")
            .get(0)
            .expect("object_id col");
        assert_eq!(
            object_id.as_deref(),
            Some("keeper"),
            "object_id must be remapped"
        );
    }

    // ── TD-112: keeper re-embed on merge ──────────────────────────────────────

    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): a merge that
    /// carries an embedder must recompute + persist the KEEPER's embedding from
    /// its canonical id, replacing whatever stale value it held before the
    /// merge. Before the fix, `apply_merge_with_audit` never touched
    /// `entities.embedding` — the keeper's stored vector stayed at its
    /// pre-merge value forever.
    #[tokio::test]
    async fn apply_entity_merge_reembeds_keeper_when_embedder_present() {
        use crate::core::provider::{DeterministicEmbeddingProvider, EmbeddingProvider};

        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_td112_reembed";
        // Keeper starts with a STALE embedding unrelated to its post-merge
        // identity — simulates an entity embedded before this merge ever ran.
        insert_entity_with_embedding(&graph, "keeper", gid, "keeper", &unit_vec(384)).await;
        insert_bare_entity(&graph, "loser", gid).await;

        let embedder = DeterministicEmbeddingProvider::new(384);
        let fresh_keeper_embedding = embedder.embed("keeper").await.expect("embed keeper id");

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                group_id: gid,
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: Some(&embedder),
            },
        )
        .await
        .expect("merge");

        // The keeper's stored embedding must now match a fresh embed of its own
        // id — NOT the stale pre-merge value.
        let distance_to_fresh =
            entity_embedding_distance(&graph, "keeper", &fresh_keeper_embedding).await;
        assert!(
            distance_to_fresh.map(|d| d < 1e-4).unwrap_or(false),
            "keeper's stored embedding must equal a fresh embed of its id post-merge \
             (TD-112); distance was {distance_to_fresh:?}"
        );

        // Regression guard: it must have actually CHANGED from the stale value
        // (rules out a no-op that happens to also satisfy the first assertion).
        let distance_to_stale = entity_embedding_distance(&graph, "keeper", &unit_vec(384)).await;
        assert!(
            distance_to_stale.map(|d| d > 1e-4).unwrap_or(false),
            "keeper's stored embedding must have CHANGED from its stale pre-merge \
             value (TD-112 regression guard); distance was {distance_to_stale:?}"
        );
    }

    /// TD-112 counterpart: `embedder: None` (the pre-fix / degraded-mode shape)
    /// must leave the keeper's embedding untouched — the fix is additive, never
    /// a mandatory re-embed.
    #[tokio::test]
    async fn apply_entity_merge_leaves_keeper_embedding_untouched_when_no_embedder() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let gid = "g_td112_no_embedder";
        insert_entity_with_embedding(&graph, "keeper", gid, "keeper", &unit_vec(384)).await;
        insert_bare_entity(&graph, "loser", gid).await;

        apply_entity_merge(
            &graph,
            EntityMergeParams {
                loser_id: "loser",
                keeper_id: "keeper",
                group_id: gid,
                site: MergeSite::Canonicalize,
                structural_signal: false,
                embedder: None,
            },
        )
        .await
        .expect("merge");

        let distance_to_stale = entity_embedding_distance(&graph, "keeper", &unit_vec(384)).await;
        assert!(
            distance_to_stale.map(|d| d < 1e-4).unwrap_or(false),
            "with embedder: None, keeper's embedding must be UNCHANGED from its \
             pre-merge value; distance was {distance_to_stale:?}"
        );
    }
}
