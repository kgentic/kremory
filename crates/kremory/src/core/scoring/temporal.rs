//! Temporal-recency boost axis (recall-v2-architecture-2026-07-03, Phase 2b).
//!
//! Additive + bounded `[0, weight]`, matching [`crate::core::search::
//! graph_degree_bonus`] so a single `.min(1.0)` clamp covers BOTH axes at the
//! insertion point — no second normalization pass (the Fork-1 hybrid:
//! additive-with-single-clamp for now; the spec's normalized-multiplicative
//! shape is deferred + eval-gated until axis-C proximity coexists).
//!
//! Signal: `valid_from` (world-time recency), `recorded_at` audit-only fallback.
//! No new query — operates only on facts already fetched by the 1-hop
//! expansion, so the axis stays read-side-pure (spec NFR / RISK-003: zero
//! `execute(` in `core/scoring/*`).

use chrono::{DateTime, Utc};

use crate::core::schema::Fact;

/// Seconds in a day, as `f64`, for the age→days conversion.
const SECS_PER_DAY: f64 = 86_400.0;

/// Bundled parameters for [`temporal_boost`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments; the codebase's `GetNeighboursAtParams`
/// / `ContextualizeParams` convention). `#[allow(clippy::too_many_arguments)]`
/// is banned in src.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TemporalBoostParams<'a> {
    /// The seed's connecting facts (already fetched by the 1-hop expansion).
    pub(crate) facts: &'a [Fact],
    /// The temporal axis weight (`SearchConfig::temporal_weight`; `<= 0` no-op).
    pub(crate) weight: f32,
    /// Decay rate for `exp(-lambda * age_days)` (`SearchConfig::temporal_decay_lambda`).
    pub(crate) lambda: f64,
    /// Reference "now" for the age computation.
    pub(crate) now: DateTime<Utc>,
}

/// Additive temporal-recency boost for a seed, in `[0, weight]`.
///
/// Computed as the **MAX** over the seed's connecting `facts` of
/// `weight * exp(-lambda * age_days)`, where `age_days = (now - valid_from)`
/// in days, clamped to `>= 0`. Because `exp(-lambda * age)` is monotonically
/// decreasing in `age`, the MAX picks the **freshest** connecting fact — a
/// seed connected to any recent fact scores near `weight`; one connected only
/// to old facts scores near `0`.
///
/// Boundary behaviour (all unit-pinned below):
/// - `weight <= 0.0` → `0.0` (true no-op; the default `temporal_weight = 0.0`
///   makes the whole axis inert without touching this code path's hot loop).
/// - `facts` empty → `0.0`.
/// - future-dated fact (`valid_from > now`) → `age` clamps to `0` → full
///   `weight` (a fact valid "as of" the future is maximally recent now).
/// - monotonic: a fresher fact never yields a smaller boost than an older one.
pub(crate) fn temporal_boost(params: TemporalBoostParams) -> f32 {
    let TemporalBoostParams {
        facts,
        weight,
        lambda,
        now,
    } = params;
    if weight <= 0.0 || facts.is_empty() {
        return 0.0;
    }

    let mut best = 0.0_f32;
    for fact in facts {
        // `valid_from` is the world-time recency signal (recall-v2 Phase 2b).
        let age_days = ((now - fact.valid_from).num_seconds() as f64 / SECS_PER_DAY).max(0.0);
        // exp(-lambda * age) ∈ (0, 1]; age=0 (fresh/future) → 1.0.
        let decay = (-lambda * age_days).exp() as f32;
        let boost = weight * decay;
        if boost > best {
            best = boost;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::Fact;
    use chrono::Duration;

    /// Minimal [`Fact`] carrying only the `valid_from` the temporal axis reads.
    fn fact_valid_from(valid_from: DateTime<Utc>) -> Fact {
        Fact {
            id: 0,
            subject_id: "s".to_owned(),
            predicate: "p".to_owned(),
            object_id: Some("o".to_owned()),
            object_value: None,
            properties: None,
            valid_from,
            valid_to: None,
            recorded_at: valid_from,
            expired_at: None,
            invalid_at: None,
            group_id: None,
            confidence: 1.0,
            source_episode_id: None,
            memory_type: None,
            content_hash: None,
            access_count: 0,
            subject_group_id: None,
            object_group_id: None,
        }
    }

    const LAMBDA: f64 = 0.01;
    const W: f32 = 0.2;

    /// Test wrapper: [`temporal_boost`] with the fixed test `LAMBDA`.
    fn tb(facts: &[Fact], weight: f32, now: DateTime<Utc>) -> f32 {
        temporal_boost(TemporalBoostParams {
            facts,
            weight,
            lambda: LAMBDA,
            now,
        })
    }

    #[test]
    fn zero_weight_is_no_op() {
        let now = Utc::now();
        let facts = vec![fact_valid_from(now)];
        assert_eq!(tb(&facts, 0.0, now), 0.0);
        // negative weight also clamps to no-op
        assert_eq!(tb(&facts, -1.0, now), 0.0);
    }

    #[test]
    fn empty_facts_is_zero() {
        let now = Utc::now();
        assert_eq!(tb(&[], W, now), 0.0);
    }

    #[test]
    fn fresh_fact_scores_near_full_weight() {
        let now = Utc::now();
        let facts = vec![fact_valid_from(now)];
        let boost = tb(&facts, W, now);
        // age 0 → decay 1.0 → boost == weight.
        assert!(
            (boost - W).abs() < 1e-6,
            "fresh fact should score full weight ({W}), got {boost}"
        );
    }

    #[test]
    fn older_fact_scores_below_fresher() {
        let now = Utc::now();
        let fresh = tb(&[fact_valid_from(now - Duration::days(1))], W, now);
        let old = tb(&[fact_valid_from(now - Duration::days(100))], W, now);
        assert!(
            old < fresh,
            "older fact ({old}) must score below fresher ({fresh})"
        );
        assert!(old >= 0.0 && fresh <= W, "boost must stay in [0, weight]");
    }

    #[test]
    fn max_picks_freshest_connecting_fact() {
        let now = Utc::now();
        // A seed connected to one ancient + one fresh fact scores as if it were
        // just the fresh one (MAX over connecting facts).
        let facts = vec![
            fact_valid_from(now - Duration::days(365)),
            fact_valid_from(now - Duration::days(1)),
        ];
        let mixed = tb(&facts, W, now);
        let fresh_only = tb(&[fact_valid_from(now - Duration::days(1))], W, now);
        assert!(
            (mixed - fresh_only).abs() < 1e-6,
            "MAX must pick the freshest fact: mixed={mixed} fresh_only={fresh_only}"
        );
    }

    #[test]
    fn future_dated_fact_clamps_to_full_weight() {
        let now = Utc::now();
        let facts = vec![fact_valid_from(now + Duration::days(30))];
        let boost = tb(&facts, W, now);
        assert!(
            (boost - W).abs() < 1e-6,
            "future-dated fact should clamp to age 0 → full weight ({W}), got {boost}"
        );
    }

    #[test]
    fn boost_never_exceeds_weight() {
        let now = Utc::now();
        for age in [0_i64, 1, 10, 100, 1_000, 10_000] {
            let boost = tb(&[fact_valid_from(now - Duration::days(age))], W, now);
            assert!(
                (0.0..=W + 1e-6).contains(&boost),
                "age={age} produced boost={boost} outside [0, {W}]"
            );
        }
    }
}
