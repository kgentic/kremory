//! Confidence-aware merge helpers (ADR-063 Site #6 — the deterministic half).
//!
//! When two entities merge (L5 canonicalization, or Site #5 acronym/nickname
//! recall), the surviving keeper should COMBINE the two entities' extraction
//! confidence rather than silently keep only the keeper's and discard the loser's.
//! Per R4 / SYNTHESIS §2, the correct combination is **noisy-OR** (`a + b - a·b`),
//! NOT a weighted average — noisy-OR is MONOTONE (merging never lowers confidence)
//! and models "at least one extraction was confident," which is the right semantic
//! for "the same real-world entity was extracted twice."
//!
//! ## What ships here (deterministic, enabled) vs what is S4-blocked
//!
//! SYNTHESIS §2's Site #6 row splits the site by readiness:
//! - **Merged-confidence FORMULA (noisy-OR)** — mechanically simple, deterministic,
//!   no threshold to calibrate → ships here + is wired into `apply_merge_with_audit`.
//! - **`CONFIDENCE_REJECT_FLOOR` GATE** (`min(conf_a, conf_b) ≥ floor` as a third
//!   required gate alongside cosine + lexical) — "direct-implement once the floor is
//!   set" (SYNTHESIS §2). The floor VALUE and the null-`ner_confidence` prevalence
//!   are BOTH unmeasured (R4 open item #2, spike **S4**). Building the gate now would
//!   mean hardcoding a guessed floor + guessed null-policy into the hot `classify_pair`
//!   path — exactly the "build on an unvalidated spike assumption" the project's
//!   spike-gating discipline forbids. It is therefore DEFERRED to a precise S4-blocked
//!   TD, not built here. When S4 lands, the gate composes via the write-gate's
//!   already-present `min_confidence_floor` input (`identity_verdict::WriteGateInputs`).

/// Noisy-OR combination of two confidence values in `[0, 1]`:  `a + b − a·b`.
///
/// Monotone in each argument (the result is ≥ `max(a, b)` for inputs in `[0, 1]`),
/// commutative, and associative — so a left-fold over an N-way merge is
/// order-independent (R4 §7: the associative/commutative generalization is a proven
/// mathematical property). Inputs are clamped to `[0, 1]` defensively.
pub(crate) fn noisy_or(a: f32, b: f32) -> f32 {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    a + b - a * b
}

/// Combine two optional entity confidences for a merge, with an explicit null
/// policy (the `ner_confidence` column is nullable and frequently absent):
///
/// - both `Some` → `Some(noisy_or(a, b))`
/// - exactly one `Some` → that one (a present signal is not diluted by a missing one)
/// - both `None` → `None` (nothing to record)
///
/// The null policy here is a DEFINED default (present-signal-wins), independent of
/// the S4 floor calibration — it only governs the FORMULA, never a reject decision.
pub(crate) fn merged_confidence(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(noisy_or(x, y)),
        (Some(x), None) => Some(x.clamp(0.0, 1.0)),
        (None, Some(y)) => Some(y.clamp(0.0, 1.0)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_or_is_monotone_and_bounded() {
        // Result ≥ each input, ≤ 1.0, for inputs in [0,1].
        for &(a, b) in &[(0.0, 0.0), (0.5, 0.5), (0.9, 0.2), (1.0, 0.0), (0.3, 0.8)] {
            let r = noisy_or(a, b);
            assert!(
                r >= a - 1e-6 && r >= b - 1e-6,
                "noisy_or({a},{b})={r} must be ≥ both"
            );
            assert!(
                (0.0..=1.0).contains(&r),
                "noisy_or({a},{b})={r} must be in [0,1]"
            );
        }
    }

    #[test]
    fn noisy_or_known_values() {
        assert!((noisy_or(0.5, 0.5) - 0.75).abs() < 1e-6); // 0.5+0.5-0.25
        assert!((noisy_or(1.0, 0.3) - 1.0).abs() < 1e-6); // absorbing at 1.0
        assert!((noisy_or(0.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn noisy_or_commutative() {
        assert!((noisy_or(0.2, 0.9) - noisy_or(0.9, 0.2)).abs() < 1e-6);
    }

    #[test]
    fn noisy_or_clamps_out_of_range_inputs() {
        // Defensive: inputs outside [0,1] are clamped, never producing NaN/>1.
        let r = noisy_or(1.5, -0.2);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "clamped to (1.0, 0.0) → 1.0, got {r}"
        );
    }

    #[test]
    fn merged_confidence_null_policy() {
        assert_eq!(merged_confidence(None, None), None);
        assert_eq!(merged_confidence(Some(0.4), None), Some(0.4));
        assert_eq!(merged_confidence(None, Some(0.6)), Some(0.6));
        assert_eq!(merged_confidence(Some(0.5), Some(0.5)), Some(0.75));
    }
}
