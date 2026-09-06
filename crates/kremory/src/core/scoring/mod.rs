//! Post-RRF recall scoring axes (recall-v2-architecture-2026-07-03, Phase 2b).
//!
//! Read-side-pure staged boost pipeline consumed by
//! [`crate::core::context::Engine::contextualize`]. This module and its
//! submodules MUST NOT call `execute(` / issue graph queries (spec NFR /
//! RISK-003): every axis operates only on data the expansion already fetched.
//!
//! # Composition model (Fork-1 hybrid)
//! All axes ship **additive + bounded + single-`.min(1.0)`-clamp**:
//! graph-degree ([`crate::core::search::graph_degree_bonus`]), temporal
//! ([`temporal::temporal_boost`]), and axis-C proximity
//! ([`crate::core::proximity::proximity_bonus`], ADR-062, Phase 3) are all
//! additive → no composition split, each measurable in isolation. ADR-062
//! itself specified proximity as a "post-RRF multiplicative boost" — ADR-067
//! **Amendment 1** (2026-07-20) supersedes that literal text: proximity's own
//! landing IS the amendment's named migration trigger ("axis-C proximity
//! lands and would coexist with the additive axes"), and the amendment
//! already resolved that trigger to "stay additive" (scored 132/135,
//! confidence HIGH) rather than migrate the whole chain to a normalized
//! multiplicative shape. No dead multiplicative plumbing ships.
//!
//! # Intent is consulted but neutral until Phase 7
//! [`weight_overrides_for`] applies per-intent multipliers over the config
//! base. Every multiplier is `1.0` for now — intent is *read* (and the reorder
//! counters below make that visible) but does not yet change scoring, resolving
//! spec risk R3 (intent-as-dead-weight) with a VISIBLE half-state rather than a
//! silent one. Phase 7 calibrates the `*_MULT` consts against LongMemEval.

use crate::core::config::SearchConfig;
use crate::core::intent::Intent;

pub(crate) mod temporal;

/// Per-axis boost weights resolved for a single recall query.
///
/// `Copy` (three `f32`s) so it threads through the per-seed loop cheaply.
/// truth-boost is deliberately ABSENT (Fork-2 verdict: axis deferred until a
/// real per-fact confidence signal exists — see the new per-fact-confidence
/// TD; all writers currently hardcode `confidence = 1.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScoringWeights {
    /// Additive graph-degree bonus weight (default `0.05`, the one live axis).
    pub(crate) graph_degree_weight: f32,
    /// Additive temporal-recency boost weight (default `0.0` = off).
    pub(crate) temporal_weight: f32,
    /// Additive graph-proximity boost weight (ADR-062, Phase 3; default
    /// `0.0` = off — see [`crate::core::proximity`]).
    pub(crate) proximity_weight: f32,
}

impl ScoringWeights {
    /// The config base, before any per-intent multiplier.
    pub(crate) fn from_config(cfg: &SearchConfig) -> Self {
        Self {
            graph_degree_weight: cfg.graph_degree_weight,
            temporal_weight: cfg.temporal_weight,
            proximity_weight: cfg.proximity_weight,
        }
    }
}

// ── Phase-7 calibration knobs (recall-v2 Phase 7) ────────────────────────────
//
// Per-intent multipliers over the config base. ALL 1.0 for now → every intent
// resolves to the config base (neutrality proof: `Broad == base`). Kept as
// DISTINCT named consts per intent so (a) Phase 7 tunes each independently and
// (b) `clippy::match_same_arms` does not fire on the equal-valued arms below —
// each arm is a distinct const path, not a duplicated literal body.
const FACTUAL_DEGREE_MULT: f32 = 1.0;
const FACTUAL_TEMPORAL_MULT: f32 = 1.0;
const FACTUAL_PROXIMITY_MULT: f32 = 1.0;
const RELATIONAL_DEGREE_MULT: f32 = 1.0;
const RELATIONAL_TEMPORAL_MULT: f32 = 1.0;
const RELATIONAL_PROXIMITY_MULT: f32 = 1.0;
const BROAD_DEGREE_MULT: f32 = 1.0;
const BROAD_TEMPORAL_MULT: f32 = 1.0;
const BROAD_PROXIMITY_MULT: f32 = 1.0;

/// Resolve the per-intent axis weights over the config `base`.
///
/// Phase 2b: every multiplier is `1.0`, so this returns `base` unchanged for
/// all intents — intent is consulted (the caller emits `intent_total`) but
/// behaviourally neutral. Phase 7 sets the `*_MULT` consts from eval feedback.
pub(crate) fn weight_overrides_for(intent: Intent, base: ScoringWeights) -> ScoringWeights {
    let (degree_mult, temporal_mult, proximity_mult) = match intent {
        Intent::Factual => (
            FACTUAL_DEGREE_MULT,
            FACTUAL_TEMPORAL_MULT,
            FACTUAL_PROXIMITY_MULT,
        ),
        Intent::Relational => (
            RELATIONAL_DEGREE_MULT,
            RELATIONAL_TEMPORAL_MULT,
            RELATIONAL_PROXIMITY_MULT,
        ),
        Intent::Broad => (BROAD_DEGREE_MULT, BROAD_TEMPORAL_MULT, BROAD_PROXIMITY_MULT),
    };
    ScoringWeights {
        graph_degree_weight: base.graph_degree_weight * degree_mult,
        temporal_weight: base.temporal_weight * temporal_mult,
        proximity_weight: base.proximity_weight * proximity_mult,
    }
}

/// One seed's per-axis boost contributions, captured during the expansion loop
/// so [`axis_reorders`] can attribute output-order changes to each axis after
/// the fact (recall-v2 Phase 2b measurement discipline).
#[derive(Debug, Clone)]
pub(crate) struct SeedAxisContribution {
    /// Entity id (used as the deterministic tie-break, matching the recall sort).
    pub(crate) id: String,
    /// Base (pre-boost) RRF-normalised score.
    pub(crate) base: f32,
    /// Additive graph-degree bonus applied to this seed this recall.
    pub(crate) degree_delta: f32,
    /// Additive temporal boost applied to this seed this recall.
    pub(crate) temporal_delta: f32,
    /// Additive graph-proximity boost applied to this seed this recall
    /// (ADR-062, Phase 3).
    pub(crate) proximity_delta: f32,
}

/// Did each axis change the score-descending OUTPUT order of the seed set?
///
/// Returns `(graph_degree_reordered, temporal_reordered, proximity_reordered)`.
/// This is the HONEST signal behind `kremory.search.<axis>_reorder_total{changed}`
/// — the cheap gate read before spending an llm-judge run ("if 0, the axis
/// reordered nothing → the judge run measures nothing"). It compares real
/// output orders under the SAME `[0, 1]` clamp the live pipeline applies, not
/// raw sums, so a non-zero boost that does NOT actually move the order
/// correctly reads as "no reorder" (a counter that would lie otherwise —
/// observability Rule 19 #9):
///
/// - graph-degree: order by `base` vs order by `min(base + degree, 1)`
/// - temporal (applied AFTER degree): order by `min(base + degree, 1)` vs
///   order by `min(base + degree + temporal, 1)`
/// - proximity (applied AFTER temporal, ADR-062 Phase 3): order by
///   `min(base + degree + temporal, 1)` vs
///   order by `min(base + degree + temporal + proximity, 1)`
///
/// Sort key matches the recall comparator: score DESC, then id ASC (stable,
/// deterministic tie-break).
///
/// **Scope = the seed set only.** 1-hop neighbours (whose decayed scores derive
/// from their connecting seed's score) are NOT counted here — a seed-score
/// change could in principle reshuffle cross-seed neighbour ordering without
/// changing seed-set order. This is a deliberate CONSERVATIVE under-count for
/// the cheap pre-judge gate (Quinn MNT-001): a `changed=true` is always a real
/// reorder; a `changed=false` means the SEEDS did not reorder (neighbours may
/// have shifted marginally). The gate reads it as "is this axis doing anything
/// worth an llm-judge run", for which seed-set reorder is the load-bearing signal.
pub(crate) fn axis_reorders(seeds: &[SeedAxisContribution]) -> (bool, bool, bool) {
    let order_by = |score: &dyn Fn(&SeedAxisContribution) -> f32| -> Vec<&str> {
        let mut idx: Vec<&SeedAxisContribution> = seeds.iter().collect();
        idx.sort_by(|a, b| {
            score(b)
                .partial_cmp(&score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        idx.iter().map(|s| s.id.as_str()).collect()
    };

    let base = order_by(&|s| s.base);
    let with_degree = order_by(&|s| (s.base + s.degree_delta).min(1.0));
    let with_temporal = order_by(&|s| (s.base + s.degree_delta + s.temporal_delta).min(1.0));
    let with_proximity =
        order_by(&|s| (s.base + s.degree_delta + s.temporal_delta + s.proximity_delta).min(1.0));

    (
        base != with_degree,
        with_degree != with_temporal,
        with_temporal != with_proximity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ScoringWeights {
        ScoringWeights {
            graph_degree_weight: 0.05,
            temporal_weight: 0.3,
            proximity_weight: 0.2,
        }
    }

    #[test]
    fn broad_intent_returns_base_unchanged() {
        let b = base();
        assert_eq!(
            weight_overrides_for(Intent::Broad, b),
            b,
            "Broad must resolve to the config base (all multipliers 1.0)"
        );
    }

    #[test]
    fn all_intents_neutral_until_phase_7() {
        // Neutrality proof: every intent resolves to base while the *_MULT
        // consts are all 1.0 — scores are byte-identical to pre-change.
        let b = base();
        for intent in [Intent::Factual, Intent::Relational, Intent::Broad] {
            assert_eq!(
                weight_overrides_for(intent, b),
                b,
                "intent {intent:?} must be behaviourally neutral in Phase 2b"
            );
        }
    }

    /// `deltas` = `(degree_delta, temporal_delta, proximity_delta)` — bundled
    /// to keep the helper under the 3-arg clippy threshold (args-as-object,
    /// test-scoped).
    fn seed(id: &str, base: f32, deltas: (f32, f32, f32)) -> SeedAxisContribution {
        SeedAxisContribution {
            id: id.to_owned(),
            base,
            degree_delta: deltas.0,
            temporal_delta: deltas.1,
            proximity_delta: deltas.2,
        }
    }

    #[test]
    fn no_boost_means_no_reorder() {
        // All deltas zero → no axis reorders (the default-config state:
        // graph_degree at 0.05 may still boost, but here we pin the zero case).
        let seeds = vec![
            seed("a", 0.9, (0.0, 0.0, 0.0)),
            seed("b", 0.5, (0.0, 0.0, 0.0)),
        ];
        assert_eq!(axis_reorders(&seeds), (false, false, false));
    }

    #[test]
    fn degree_boost_that_flips_order_is_detected() {
        // b starts below a, but a huge degree bonus on b flips the order.
        let seeds = vec![
            seed("a", 0.50, (0.0, 0.0, 0.0)),
            seed("b", 0.48, (0.30, 0.0, 0.0)),
        ];
        let (degree_reordered, temporal_reordered, proximity_reordered) = axis_reorders(&seeds);
        assert!(
            degree_reordered,
            "degree boost flipping b above a must count"
        );
        assert!(
            !temporal_reordered,
            "temporal delta is 0 → no temporal reorder"
        );
        assert!(
            !proximity_reordered,
            "proximity delta is 0 → no proximity reorder"
        );
    }

    #[test]
    fn temporal_boost_that_flips_order_is_detected() {
        // Equal base+degree, but a temporal boost on the id-later seed lifts it
        // above the id-earlier one (which would otherwise win the tie-break).
        let seeds = vec![
            seed("a", 0.50, (0.0, 0.0, 0.0)),
            seed("z", 0.50, (0.0, 0.20, 0.0)),
        ];
        let (degree_reordered, temporal_reordered, proximity_reordered) = axis_reorders(&seeds);
        assert!(!degree_reordered, "no degree delta → no degree reorder");
        assert!(
            temporal_reordered,
            "temporal boost lifting z above a must count"
        );
        assert!(
            !proximity_reordered,
            "proximity delta is 0 → no proximity reorder"
        );
    }

    /// ADR-062 Phase 3: proximity is applied AFTER degree and temporal — a
    /// proximity boost that flips order must be attributed to `proximity`
    /// alone, not smeared across the earlier axes.
    #[test]
    fn proximity_boost_that_flips_order_is_detected() {
        // Equal base+degree+temporal, but a proximity boost on the id-later
        // seed lifts it above the id-earlier one (which would otherwise win
        // the tie-break).
        let seeds = vec![
            seed("a", 0.50, (0.0, 0.0, 0.0)),
            seed("z", 0.50, (0.0, 0.0, 0.15)),
        ];
        let (degree_reordered, temporal_reordered, proximity_reordered) = axis_reorders(&seeds);
        assert!(!degree_reordered, "no degree delta → no degree reorder");
        assert!(
            !temporal_reordered,
            "no temporal delta → no temporal reorder"
        );
        assert!(
            proximity_reordered,
            "proximity boost lifting z above a must count"
        );
    }

    #[test]
    fn boost_that_does_not_change_order_reads_as_no_reorder() {
        // a is far ahead; a small degree boost on b cannot flip it → honest
        // "no reorder" even though a non-zero boost was applied (the counter
        // must not lie).
        let seeds = vec![
            seed("a", 0.95, (0.0, 0.0, 0.0)),
            seed("b", 0.10, (0.05, 0.0, 0.0)),
        ];
        assert_eq!(axis_reorders(&seeds), (false, false, false));
    }
}
