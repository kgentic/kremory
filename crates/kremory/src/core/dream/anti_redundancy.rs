//! Anti-redundancy gate for Pass 0 type discovery (ADR-037 §3.2 / §9.4).
//!
//! Each surviving shape-validated proposal is checked against the embeddings
//! of existing registered entity types.  Two thresholds, either triggers rejection:
//!
//! | Comparison | Threshold | Rationale |
//! |---|---|---|
//! | proposal description ↔ existing type description | 0.85 cosine | both are definitional prose; tight gate prevents near-duplicate vocab |
//! | proposal name ↔ existing type name | 0.70 cosine | surface-form near-duplicate catch (e.g. "LegalPrecedent" vs "LegalRuling") |
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

/// Thresholds per ADR-037 §9.4.
pub(crate) const DESC_COSINE_THRESHOLD: f32 = 0.85;
pub(crate) const NAME_COSINE_THRESHOLD: f32 = 0.70;

/// Bundled parameters for [`check_proposal`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
pub(crate) struct CheckProposalParams<'a> {
    pub(crate) proposal_desc_emb: &'a [f32],
    pub(crate) proposal_name_emb: &'a [f32],
    pub(crate) existing_type_embeddings: &'a [(EntityTypeSpec, Vec<f32>, Vec<f32>)],
    pub(crate) namespace: &'a str,
    pub(crate) model: &'a str,
}

/// Check a single proposal against all existing types via cosine similarity.
///
/// Returns [`GateOutcome::Redundant`] if the proposal description embedding is
/// within 0.85 cosine of ANY existing type's description embedding, OR if the
/// proposal name embedding is within 0.70 cosine of ANY existing type's name
/// embedding.  Either check triggering rejects.
///
/// Emits `kremory.dream.types_rejected_total{reason="redundant_with_existing"}` on
/// rejection — built-in observability per [[observability-first-class]].
///
/// `namespace` is the group_id string used as the `namespace` metrics label.
pub(crate) fn check_proposal(params: CheckProposalParams<'_>) -> GateOutcome {
    let CheckProposalParams {
        proposal_desc_emb,
        proposal_name_emb,
        existing_type_embeddings,
        namespace,
        model,
    } = params;
    for (spec, desc_emb, name_emb) in existing_type_embeddings {
        // Primary gate: description-pair cosine ≥ 0.85
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

        // Secondary gate: name-pair cosine ≥ 0.70
        let name_sim = cosine(proposal_name_emb, name_emb);
        if name_sim >= NAME_COSINE_THRESHOLD {
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
                name_cosine = %name_sim,
                threshold = NAME_COSINE_THRESHOLD,
                "anti-redundancy: name-pair cosine exceeded threshold — rejecting proposal"
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
        let existing = vec![(
            spec(1, "Person"),
            unit_vec(4, 0), // desc embedding at dim 0
            unit_vec(4, 1), // name embedding at dim 1
        )];
        // Proposal embeddings at completely different dims
        let desc = unit_vec(4, 2);
        let name = unit_vec(4, 3);
        let outcome = check_proposal(CheckProposalParams {
            proposal_desc_emb: &desc,
            proposal_name_emb: &name,
            existing_type_embeddings: &existing,
            namespace: "test_ns",
            model: "test_model",
        });
        assert_eq!(outcome, GateOutcome::Pass);
    }

    #[test]
    fn gate_rejects_on_description_similarity() {
        let desc_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let existing = vec![(
            spec(1, "Person"),
            desc_vec.clone(), // same description embedding — cosine = 1.0
            unit_vec(4, 1),
        )];
        let outcome = check_proposal(CheckProposalParams {
            proposal_desc_emb: &desc_vec,
            proposal_name_emb: &unit_vec(4, 3),
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
    fn gate_rejects_on_name_similarity() {
        let name_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let existing = vec![(
            spec(1, "Organisation"),
            unit_vec(4, 3), // distinct desc
            name_vec.clone(),
        )];
        // Proposal name is identical to existing name embedding → cosine = 1.0 ≥ 0.70
        let outcome = check_proposal(CheckProposalParams {
            proposal_desc_emb: &unit_vec(4, 2),
            proposal_name_emb: &name_vec,
            existing_type_embeddings: &existing,
            namespace: "ns",
            model: "m",
        });
        match outcome {
            GateOutcome::Redundant { existing_name } => assert_eq!(existing_name, "Organisation"),
            other => panic!("expected Redundant, got {other:?}"),
        }
    }

    #[test]
    fn gate_passes_with_empty_existing_types() {
        let desc = unit_vec(4, 0);
        let name = unit_vec(4, 1);
        let outcome = check_proposal(CheckProposalParams {
            proposal_desc_emb: &desc,
            proposal_name_emb: &name,
            existing_type_embeddings: &[],
            namespace: "ns",
            model: "m",
        });
        assert_eq!(outcome, GateOutcome::Pass);
    }
}
