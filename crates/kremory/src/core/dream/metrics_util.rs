//! Shared statistical helpers for dream-pass precision/recall spikes and the
//! metrics harness (spec `dream-adversarial-corpora-and-metrics-2026-07-02.md`
//! §3 step 0, H1).
//!
//! Extracted from the inline Wilson-interval block in
//! `acronym_nickname_recall::tests::initialism_pre_filter_precision_recall_s1`
//! (ADR-063 Site #5 S1 spike) so the new metrics harness can reuse it instead
//! of re-deriving the formula per call site.

/// Wilson 95% score interval (z = 1.96) on the proportion `successes / n`.
///
/// Unlike the naive normal-approximation interval, the Wilson interval stays
/// well-behaved near 0.0/1.0 and for small `n` — the exact regime this helper
/// targets (small-N dream-pass precision/recall spikes where the point
/// estimate alone is misleading; see the S1 test's inline commentary this was
/// ported from).
///
/// Returns `(lower, upper)`, both clamped to `[0.0, 1.0]`. When `n == 0` the
/// proportion is undefined; this mirrors the original S1 guard and returns
/// `(1.0, 1.0)` (vacuously-perfect / "nothing to reject" convention used by
/// the dream-pass precision/recall spikes for the zero-denominator case).
///
/// Gated `#[cfg(any(test, feature = "test-utils"))]`: `tests/dream_metrics_
/// harness.rs` (spec `dream-adversarial-corpora-and-metrics-2026-07-02.md`
/// §3) is the first `tests/*.rs` integration-test consumer of this helper via
/// the `metrics_util` re-export in `dream/mod.rs`, so the `feature =
/// "test-utils"` arm is no longer dead code in kremory's `[dev-dependencies]`
/// self-reference compilation unit (`kremory = { path = ".", features =
/// ["test-utils"] }`, built with `feature = "test-utils"` active but
/// `cfg(test)` NOT active — see git history for the prior `#[cfg(test)]`-only
/// gate and its rationale while no such consumer existed).
///
/// `pub` (not `pub(crate)`) + `#[doc(hidden)]`: `tests/dream_metrics_
/// harness.rs` lives outside the crate boundary (an external integration-test
/// binary), so a `pub(crate)` item cannot be re-exported `pub` from
/// `dream/mod.rs` (E0364) for it to reach — same visibility requirement as
/// the sibling MNT-002 re-exports (`acronym_nickname_recall`,
/// `type_registry_collapse`), which are `pub` + `#[doc(hidden)]` for the
/// identical reason. `#[doc(hidden)]` hides it from rustdoc but not from
/// `feature = "test-utils"` consumers, which must treat it as unstable.
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub fn wilson_lower_upper(successes: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (1.0, 1.0);
    }

    let n = n as f64;
    let p = successes as f64 / n;

    let z = 1.96_f64;
    let z2 = z * z;
    let centre = p + z2 / (2.0 * n);
    let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    let denom = 1.0 + z2 / n;

    (
        ((centre - margin) / denom).max(0.0),
        ((centre + margin) / denom).min(1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1 fixture regression: TP=18, FP=2 → n=20, successes=18. Must match the
    /// committed S1 test output exactly: CI=[0.6990, 0.9721].
    #[test]
    fn wilson_18_of_20_matches_s1_committed_output() {
        let (lo, hi) = wilson_lower_upper(18, 20);
        assert!(
            (lo - 0.6990).abs() < 1e-4,
            "lower bound {lo:.4} does not match S1 committed CI lower 0.6990"
        );
        assert!(
            (hi - 0.9721).abs() < 1e-4,
            "upper bound {hi:.4} does not match S1 committed CI upper 0.9721"
        );
    }

    #[test]
    fn wilson_zero_denominator_returns_vacuous_one_one() {
        assert_eq!(wilson_lower_upper(0, 0), (1.0, 1.0));
    }

    #[test]
    fn wilson_zero_of_ten_is_low_but_bounded_above_zero() {
        let (lo, hi) = wilson_lower_upper(0, 10);
        assert_eq!(lo, 0.0, "lower bound for 0 successes must clamp to 0.0");
        assert!(
            hi > 0.0 && hi < 0.5,
            "upper bound {hi:.4} out of expected range"
        );
    }

    #[test]
    fn wilson_ten_of_ten_is_high_but_bounded_below_one() {
        let (lo, hi) = wilson_lower_upper(10, 10);
        assert_eq!(hi, 1.0, "upper bound for all-successes must clamp to 1.0");
        assert!(
            lo > 0.5 && lo < 1.0,
            "lower bound {lo:.4} out of expected range"
        );
    }
}
