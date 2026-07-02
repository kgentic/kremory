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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GateOutcome {
    /// Proposal is distinct from all existing types — accepted by this gate.
    Pass,
    /// Proposal is too similar to an existing type (name or description exceeded threshold).
    Redundant { existing_name: String },
}

/// Description-cosine threshold per ADR-037 §9.4.
///
/// TD-097 (Site #1 of ADR-063): the former `NAME_COSINE_THRESHOLD` (0.70) was
/// REMOVED — bare-label cosine is a degenerate identity signal (`cos(Person,Date)
/// =1.0000`). The name signal is now a deterministic normalized exact-match
/// pre-filter (see `check_proposal`), and the description-cosine gate below is the
/// primary + sufficient near-duplicate detector.
pub(crate) const DESC_COSINE_THRESHOLD: f32 = 0.85;

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

/// Check a single proposal against all existing types.
///
/// Returns [`GateOutcome::Redundant`] if the proposal name normalizes-equal to ANY
/// existing type name (free deterministic pre-filter), OR if the proposal
/// description embedding is within 0.85 cosine of ANY existing type's description
/// embedding.  Either check triggering rejects.
///
/// TD-097 (Site #1 of ADR-063): the former bare-NAME cosine gate (≥0.70) was
/// removed — `cos(Person, Date) = 1.0000` under `nomic-embed-text`, so it falsely
/// rejected distinct proposals. Near-duplicate (non-exact) names are caught by the
/// description-cosine gate, which embeds definitional prose (a stable signal), not a
/// bare label. See SYNTHESIS §2 row 1 + the surface-dependent signal table (ADR-063).
///
/// Emits `kremory.dream.types_rejected_total{reason="redundant_with_existing"}` on
/// rejection — built-in observability per [[observability-first-class]].
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
    for (spec, desc_emb) in existing_type_embeddings {
        // Free deterministic pre-filter (TD-097): a normalized EXACT name match is a
        // real duplicate ("Person"/"person") — reject with no embedding. This REPLACES
        // the old bare-NAME cosine gate, which was degenerate (unrelated short names
        // cosine ≈ 1.0 → false rejects of distinct proposals).
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

        // Primary gate: description-pair cosine ≥ 0.85 (definitional prose — the
        // stable, information-bearing identity signal for a TYPE per ADR-063).
        let desc_sim = cosine(proposal_desc_emb, desc_emb);
        if desc_sim >= DESC_COSINE_THRESHOLD {
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
                desc_cosine = %desc_sim,
                threshold = DESC_COSINE_THRESHOLD,
                "anti-redundancy: description-pair cosine exceeded threshold — rejecting proposal"
            );
            return GateOutcome::Redundant {
                existing_name: spec.name.clone(),
            };
        }
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
    fn gate_rejects_on_description_similarity() {
        // Distinct NAME (so the exact-match pre-filter does NOT fire) but identical
        // DESCRIPTION embedding (cosine = 1.0 ≥ 0.85) → Redundant via the desc gate.
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
            GateOutcome::Redundant { existing_name } => assert_eq!(existing_name, "Person"),
            other => panic!("expected Redundant, got {other:?}"),
        }
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
