//! Anti-redundancy gate for Pass 0 type discovery (ADR-037 §3.2 / §9.4).
//!
//! Each surviving shape-validated proposal is checked against existing registered
//! entity types.  Two signals, either triggers rejection:
//!
//! | Comparison | Signal | Rationale |
//! |---|---|---|
//! | proposal name ↔ existing type name | normalized **exact** match (`normalize_name`) | free deterministic pre-filter for true duplicates ("Person"/"person") |
//! | proposal description ↔ existing type description | 0.85 **cosine** | both are definitional prose; tight gate catches near-duplicate vocab ("LegalPrecedent"/"LegalRuling") |
//!
//! ## Site #2 — type-novelty DESCRIPTION-gate LLM-verify band (ADR-063 spec §4.3 sibling)
//!
//! `check_proposal` classifies each proposal against every existing type's best
//! description-cosine match into one of three [`GateOutcome`]s:
//!
//! - Exact-name match → `Redundant` (unchanged — the unambiguous duplicate case).
//! - Best desc-cosine ≥ 0.85 AND the two names share a lemma/exact match → `Redundant`
//!   (unambiguous auto-reject — no ambiguity to adjudicate).
//! - Best desc-cosine ≥ 0.85 with ZERO lemma overlap → `NeedsLlmVerify` (SYNTHESIS §2
//!   row 1 / EDC's over-generalization finding: a distinct-but-similar type, e.g.
//!   `LegalPrecedent` vs `LegalRuling`, must not be silently rejected by cosine alone).
//! - Best desc-cosine in `[0.70, 0.85)` → `NeedsLlmVerify` (ambiguous zone).
//! - Otherwise → `Pass`.
//!
//! `check_proposal` stays a PURE classification function — no LLM call, no I/O. The
//! caller (`discover_types`) decides what to DO with `NeedsLlmVerify`: when
//! `DreamOpts::include_type_novelty_llm_verify` is `false` (default), the caller
//! falls back to the pre-Site-#2 behaviour (≥0.85 → reject, else accept) so the
//! DEFAULT build's outcome is byte-for-byte unchanged from before Site #2 landed.
//! When `true`, the caller adjudicates via the shared `write_gate` (spec §2.2),
//! reusing the same `IdentityVerdictBatch` machinery Site #3 (`type_registry_collapse.rs`)
//! already ships.
//!
//! ## TD-097 — why the NAME signal is exact-match, not cosine (Site #1 of ADR-063)
//!
//! Bare-label cosine is a DEGENERATE identity signal: `nomic-embed-text` returns
//! cosine ≈ 1.0 for unrelated short strings (`cos(Person, Date) = 1.0000`, TD-097's
//! own measurement; R3 confirms no surveyed embedding technique discriminates bare
//! proper nouns).  The old name-COSINE gate (≥0.70) therefore falsely rejected
//! DISTINCT type proposals as "redundant".  Per ADR-063's surface-dependent signal
//! table, a type's stable identity signal is its DESCRIPTION (definitional prose),
//! never its bare name — so the name signal is reduced to a normalized exact-match
//! pre-filter (catches the trivial dup for free) and the description-cosine gate is
//! primary + sufficient for near-duplicates.  (A singular/plural lemma pre-filter is
//! deferred to Site #3's spike-gated pass, SYNTHESIS §4 S3 — not Phase 1.)
//!
//! ## Degraded mode (D7)
//!
//! When no embedder is configured, the gate is **skipped entirely**.  The caller
//! receives a warning string `"no embedder configured — anti-redundancy gate skipped"`
//! in `DiscoveryResult.warnings`.  A counter
//! `kremory.dream.anti_redundancy_gate_skipped_total{reason="no_embedder"}` is emitted.
//!
//! ## Cosine helper
//!
//! Uses a plain dot-product / magnitude implementation.  Both vectors are
//! already normalised by the embedder (all kremory-standard embedders return
//! L2-normalised outputs), so dot-product == cosine similarity.

use metrics::counter;

use crate::core::entity_types::EntityTypeSpec;

/// Cosine similarity between two equal-length f32 vectors.
///
/// Assumes both inputs are L2-normalised (dot == cosine).  Returns 0.0 for
/// zero-length or empty vectors rather than NaN.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.len() != a.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Outcome of the anti-redundancy gate for a single proposal.
///
/// Degraded-mode skips (no embedder) are handled by the caller in
/// `discover_types` via the else-branch — `check_proposal` is never
/// called in degraded mode, so `GateOutcome` has no `Skipped` variant.
/// The degraded-mode counter / warning are emitted by `emit_gate_skipped()`.
// `Eq` dropped: `NeedsLlmVerify.desc_cosine` is `f32`, which is `PartialEq` only.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GateOutcome {
    /// Proposal is distinct from all existing types — accepted by this gate.
    Pass,
    /// Proposal is too similar to an existing type (name or description exceeded threshold).
    Redundant { existing_name: String },
    /// Proposal's best desc-cosine match against an existing type falls in the
    /// AMBIGUOUS band (ADR-063 spec §4.3 sibling table, Site #2): either
    /// `[0.70, 0.85)`, or `≥0.85` with zero name-lemma overlap (SYNTHESIS §2 row 1
    /// / EDC's over-generalization finding — a hard 0.85 cutoff alone would
    /// silently reject a distinct-but-similar type like `LegalPrecedent` vs
    /// `LegalRuling`). The caller (`discover_types`) decides whether to adjudicate
    /// via the shared `write_gate`, gated by `DreamOpts::include_type_novelty_llm_verify`.
    NeedsLlmVerify {
        existing_name: String,
        desc_cosine: f32,
    },
}

/// Description-cosine threshold per ADR-037 §9.4.
///
/// TD-097 (Site #1 of ADR-063): the former `NAME_COSINE_THRESHOLD` (0.70) was
/// REMOVED — bare-label cosine is a degenerate identity signal (`cos(Person,Date)
/// =1.0000`). The name signal is now a deterministic normalized exact-match
/// pre-filter (see `check_proposal`), and the description-cosine gate below is the
/// primary + sufficient near-duplicate detector.
pub(crate) const DESC_COSINE_THRESHOLD: f32 = 0.85;

/// Lower band edge for the Site #2 LLM-verify ambiguous zone (ADR-063 spec §4.3
/// sibling table). PROVISIONAL — spike-gated S3 (spec §8), carried by analogy from
/// Site #3's `TYPE_COLLAPSE_LOWER_BAND_COSINE`; NOT independently derived for the
/// at-proposal gate here. Gated behind `DreamOpts::include_type_novelty_llm_verify`,
/// default `false` — when the flag is off, this constant is inert (the default
/// build's outcome for the `[0.70, 0.85)` band is unchanged: accept, per the
/// pre-Site-#2 behaviour).
pub(crate) const TYPE_NOVELTY_LOWER_BAND: f32 = 0.70;

/// Bundled parameters for [`check_proposal`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
pub(crate) struct CheckProposalParams<'a> {
    /// The proposal's raw type name. Compared via `normalize_name` against each
    /// existing type name as a free, deterministic, embedder-independent pre-filter
    /// (TD-097: bare-NAME cosine is degenerate, so the name signal is exact-match).
    pub(crate) proposal_name: &'a str,
    pub(crate) proposal_desc_emb: &'a [f32],
    /// `(spec, description_embedding)` per existing type. The name embedding was
    /// removed with the name-cosine gate (TD-097); `spec.name` supplies the name for
    /// the exact-match pre-filter.
    pub(crate) existing_type_embeddings: &'a [(EntityTypeSpec, Vec<f32>)],
    pub(crate) namespace: &'a str,
    pub(crate) model: &'a str,
}

/// Case-insensitive-normalized exact match OR a naive singular/plural lemma match
/// (strip a trailing `s`) between two type names (ADR-063 spec §4.2).
///
/// Duplicated from `type_registry_collapse.rs::names_share_lemma_or_exact` with
/// attribution, per spec §4.5's explicit "duplication with attribution is
/// acceptable" ruling — that helper is a private `fn` scoped to its own module and
/// Site #2's at-proposal gate is a distinct call site (proposal name vs. type-pair
/// name), so a shared `pub(crate)` extraction was not attempted for one small
/// dictionary-free string helper. `pub(crate)` (not private) because
/// `discover_types.rs` reuses it as the Site #2 `write_gate` deterministic signal
/// (the same lemma test used to classify `Redundant` vs `NeedsLlmVerify` here is
/// also the correct deterministic corroboration signal at adjudication time).
pub(crate) fn names_share_lemma_or_exact(a: &str, b: &str) -> bool {
    let norm_a = crate::core::resolver::normalize_name(a);
    let norm_b = crate::core::resolver::normalize_name(b);
    if norm_a == norm_b {
        return true;
    }
    strip_trailing_s(&norm_a) == strip_trailing_s(&norm_b)
}

/// Strip a single trailing `s` (naive singular/plural lemma heuristic, spec §4.2).
fn strip_trailing_s(s: &str) -> &str {
    s.strip_suffix('s').unwrap_or(s)
}

/// Check a single proposal against all existing types.
///
/// Classification order (ADR-063 spec §4.3 sibling table, Site #2):
///
/// 1. Proposal name normalizes-EQUAL (exact) to an existing type name →
///    `Redundant` (unambiguous — free deterministic pre-filter, unchanged from
///    pre-Site-#2 behaviour).
/// 2. Best desc-cosine match across all existing types ≥ 0.85 AND the two names
///    share a lemma/exact match → `Redundant` (unambiguous auto-reject — high
///    similarity AND name overlap, no ambiguity to adjudicate).
/// 3. Best desc-cosine match ≥ 0.85 with ZERO lemma overlap → `NeedsLlmVerify`
///    (SYNTHESIS §2 row 1 / EDC's over-generalization finding: a hard 0.85 cutoff
///    alone would silently reject a distinct-but-similar type, e.g.
///    `LegalPrecedent` vs `LegalRuling` — the LLM-verify band exists for exactly
///    this case).
/// 4. Best desc-cosine match in `[0.70, 0.85)` → `NeedsLlmVerify` (ambiguous zone).
/// 5. Otherwise → `Pass`.
///
/// TD-097 (Site #1 of ADR-063): the former bare-NAME cosine gate (≥0.70) was
/// removed — `cos(Person, Date) = 1.0000` under `nomic-embed-text`, so it falsely
/// rejected distinct proposals. Near-duplicate (non-exact) names are caught by the
/// description-cosine gate, which embeds definitional prose (a stable signal), not a
/// bare label. See SYNTHESIS §2 row 1 + the surface-dependent signal table (ADR-063).
///
/// This function is PURE — no LLM call, no I/O — mirroring `write_gate`'s
/// counter-free discipline for the classification step. `discover_types` (the
/// caller) decides what to DO with `NeedsLlmVerify`; only `Redundant` emits the
/// `types_rejected_total` counter here (an LLM-adjudicated rejection is counted by
/// the caller once the write_gate decision is known, avoiding double-counting).
///
/// `namespace` is the group_id string used as the `namespace` metrics label.
pub(crate) fn check_proposal(params: CheckProposalParams<'_>) -> GateOutcome {
    let CheckProposalParams {
        proposal_name,
        proposal_desc_emb,
        existing_type_embeddings,
        namespace,
        model,
    } = params;
    let proposal_name_norm = crate::core::resolver::normalize_name(proposal_name);

    // Step 1: free deterministic pre-filter (TD-097) — a normalized EXACT name
    // match is a real duplicate ("Person"/"person") — reject with no embedding.
    for (spec, _desc_emb) in existing_type_embeddings {
        if crate::core::resolver::normalize_name(&spec.name) == proposal_name_norm {
            counter!(
                "kremory.dream.types_rejected_total",
                "reason" => "redundant_with_existing",
                "model" => model.to_string(),
                "namespace" => namespace.to_string()
            )
            .increment(1);
            tracing::debug!(
                target: "kremory::dream::anti_redundancy",
                existing_name = %spec.name,
                proposal_name = %proposal_name,
                "anti-redundancy: proposal name normalizes-equal to an existing type — rejecting"
            );
            return GateOutcome::Redundant {
                existing_name: spec.name.clone(),
            };
        }
    }

    // Step 2: find the BEST (highest) desc-cosine match across all existing types,
    // so the ambiguous-band decision (steps 3/4) is made against the single
    // strongest candidate, not the first one iterated.
    let mut best: Option<(&EntityTypeSpec, f32)> = None;
    for (spec, desc_emb) in existing_type_embeddings {
        let desc_sim = cosine(proposal_desc_emb, desc_emb);
        if best.map(|(_, b)| desc_sim > b).unwrap_or(true) {
            best = Some((spec, desc_sim));
        }
    }

    let Some((best_spec, best_cosine)) = best else {
        return GateOutcome::Pass; // no existing types to compare against
    };

    if best_cosine >= DESC_COSINE_THRESHOLD {
        if names_share_lemma_or_exact(proposal_name, &best_spec.name) {
            // Unambiguous auto-reject: high similarity AND name overlap.
            counter!(
                "kremory.dream.types_rejected_total",
                "reason" => "redundant_with_existing",
                "model" => model.to_string(),
                "namespace" => namespace.to_string()
            )
            .increment(1);
            tracing::debug!(
                target: "kremory::dream::anti_redundancy",
                existing_name = %best_spec.name,
                desc_cosine = %best_cosine,
                threshold = DESC_COSINE_THRESHOLD,
                "anti-redundancy: description-pair cosine exceeded threshold with lemma overlap — rejecting proposal"
            );
            return GateOutcome::Redundant {
                existing_name: best_spec.name.clone(),
            };
        }
        // ≥0.85 but ZERO lemma overlap — ambiguous per EDC's over-generalization
        // finding (SYNTHESIS §2 row 1): do not silently reject.
        tracing::debug!(
            target: "kremory::dream::anti_redundancy",
            existing_name = %best_spec.name,
            desc_cosine = %best_cosine,
            "anti-redundancy: description-pair cosine exceeded threshold with ZERO lemma overlap — needs LLM verify"
        );
        return GateOutcome::NeedsLlmVerify {
            existing_name: best_spec.name.clone(),
            desc_cosine: best_cosine,
        };
    }

    if best_cosine >= TYPE_NOVELTY_LOWER_BAND {
        tracing::debug!(
            target: "kremory::dream::anti_redundancy",
            existing_name = %best_spec.name,
            desc_cosine = %best_cosine,
            "anti-redundancy: description-pair cosine in ambiguous band — needs LLM verify"
        );
        return GateOutcome::NeedsLlmVerify {
            existing_name: best_spec.name.clone(),
            desc_cosine: best_cosine,
        };
    }

    GateOutcome::Pass
}

/// Emit the degraded-mode counter when no embedder is available.
///
/// Called by `discover_types` when it skips the gate.  Separated so the
/// caller can also record the warning string without duplicating the counter emit.
pub(crate) fn emit_gate_skipped() {
    counter!(
        "kremory.dream.anti_redundancy_gate_skipped_total",
        "reason" => "no_embedder"
    )
    .increment(1);
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entity_types::EntityTypeSpec;

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot] = 1.0;
        v
    }

    fn spec(id: u32, name: &str) -> EntityTypeSpec {
        EntityTypeSpec {
            id,
            name: name.to_string(),
            description: format!("{name} description"),
        }
    }

    #[test]
    fn cosine_identical_vectors_returns_one() {
        let v = vec![0.6, 0.8];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors_returns_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_mismatched_dims_returns_zero() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0);
    }

    #[test]
    fn gate_passes_distinct_proposal() {
        // Distinct name (no normalized match) + orthogonal desc embedding → Pass.
        let existing = vec![(spec(1, "Person"), unit_vec(4, 0))]; // desc emb at dim 0
        let desc = unit_vec(4, 2);
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Vehicle",
            proposal_desc_emb: &desc,
            existing_type_embeddings: &existing,
            namespace: "test_ns",
            model: "test_model",
        });
        assert_eq!(outcome, GateOutcome::Pass);
    }

    #[test]
    fn gate_rejects_on_description_similarity_with_lemma_overlap() {
        // Distinct-but-lemma-related NAME ("Organization"/"Organizations" share a
        // trailing-s lemma) + identical DESCRIPTION embedding (cosine = 1.0 ≥ 0.85)
        // → Redundant via the desc gate (unambiguous auto-reject, Site #2 step 2).
        let desc_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let existing = vec![(spec(1, "Organization"), desc_vec.clone())];
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Organizations",
            proposal_desc_emb: &desc_vec,
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        match outcome {
            GateOutcome::Redundant { existing_name } => assert_eq!(existing_name, "Organization"),
            other => panic!("expected Redundant, got {other:?}"),
        }
    }

    #[test]
    fn gate_needs_llm_verify_on_high_cosine_zero_lemma_overlap() {
        // Site #2 (EDC over-generalization finding, SYNTHESIS §2 row 1): distinct
        // NAME with ZERO lemma overlap ("Human" vs "Person" — the exact-match
        // pre-filter does NOT fire, and no lemma relationship exists) but identical
        // DESCRIPTION embedding (cosine = 1.0 ≥ 0.85) → NeedsLlmVerify, NOT a
        // silent Redundant reject. A hard 0.85 cutoff alone would have falsely
        // rejected a distinct-but-similar type (e.g. LegalPrecedent vs LegalRuling).
        let desc_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let existing = vec![(spec(1, "Person"), desc_vec.clone())];
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Human",
            proposal_desc_emb: &desc_vec,
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        match outcome {
            GateOutcome::NeedsLlmVerify {
                existing_name,
                desc_cosine,
            } => {
                assert_eq!(existing_name, "Person");
                assert!((desc_cosine - 1.0).abs() < 1e-5);
            }
            other => panic!("expected NeedsLlmVerify, got {other:?}"),
        }
    }

    #[test]
    fn gate_needs_llm_verify_on_ambiguous_mid_band_cosine() {
        // Site #2: best desc-cosine in [0.70, 0.85) → NeedsLlmVerify (ambiguous
        // zone, step 3), regardless of lemma overlap.
        // A=[1,0,0,0], B=[0.8,0.6,0,0] (unit-norm) → dot = 0.80.
        let existing = vec![(spec(1, "Vehicle"), vec![1.0f32, 0.0, 0.0, 0.0])];
        let proposal_desc = vec![0.8f32, 0.6, 0.0, 0.0];
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Conveyance",
            proposal_desc_emb: &proposal_desc,
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        match outcome {
            GateOutcome::NeedsLlmVerify {
                existing_name,
                desc_cosine,
            } => {
                assert_eq!(existing_name, "Vehicle");
                assert!((desc_cosine - 0.80).abs() < 0.02);
            }
            other => panic!("expected NeedsLlmVerify, got {other:?}"),
        }
    }

    #[test]
    fn gate_passes_below_lower_band() {
        // Best desc-cosine below 0.70 → Pass (step 5).
        let existing = vec![(spec(1, "Vehicle"), unit_vec(4, 0))];
        let proposal_desc = unit_vec(4, 1);
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Statute",
            proposal_desc_emb: &proposal_desc,
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        assert_eq!(outcome, GateOutcome::Pass);
    }

    #[test]
    fn gate_rejects_on_name_exact_match() {
        // TD-097: the name signal is a normalized EXACT match, not cosine. A
        // case/whitespace variant of an existing type name → Redundant, with NO name
        // embedding. Orthogonal description embedding proves the rejection came from
        // the name pre-filter, not the description gate.
        let existing = vec![(spec(1, "Organisation"), unit_vec(4, 3))];
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "  organisation ", // normalizes-equal to "Organisation"
            proposal_desc_emb: &unit_vec(4, 2),
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        match outcome {
            GateOutcome::Redundant { existing_name } => assert_eq!(existing_name, "Organisation"),
            other => panic!("expected Redundant (exact name match), got {other:?}"),
        }
    }

    #[test]
    fn gate_passes_unrelated_short_names_no_degenerate_reject() {
        // TD-097 regression (Site #1 DoD): two UNRELATED short type names ("Person"
        // vs "Date") whose bare-name embeddings would DEGENERATELY cosine ≈ 1.0 must
        // NOT be rejected — distinct names (no exact match) + orthogonal descriptions
        // → Pass. Pre-fix the name-cosine gate (≥0.70) falsely rejected this pair.
        let existing = vec![(spec(1, "Person"), unit_vec(4, 0))];
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Date",
            proposal_desc_emb: &unit_vec(4, 2), // orthogonal desc → cosine 0
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        assert_eq!(
            outcome,
            GateOutcome::Pass,
            "unrelated short names must not be falsely rejected (TD-097)"
        );
    }

    #[test]
    fn gate_passes_with_empty_existing_types() {
        let desc = unit_vec(4, 0);
        let outcome = check_proposal(CheckProposalParams {
            proposal_name: "Person",
            proposal_desc_emb: &desc,
            existing_type_embeddings: &[],
            namespace: "ns",
            model: "m",
        });
        assert_eq!(outcome, GateOutcome::Pass);
    }
}
