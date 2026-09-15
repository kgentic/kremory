//! TD-177 — an interlock on bulk fact retirement.
//!
//! No sweep that invalidates facts (contradiction detection, dream
//! supersession) previously had any check on HOW MANY it was about to
//! retire relative to a namespace's size. Worked example (TD-167): 81 of
//! 1,021 facts invalidated on an 8-session run, ≥31% of them provably
//! multi-valued (contradiction detection wrongly treated a multi-valued
//! predicate as functional). Nothing in the system objected — it was caught
//! by a human challenging a metric, not by any guard.
//!
//! This does NOT fix contradiction-detection quality (that's a separate,
//! much larger problem). It gives every bulk-retiring sweep a cheap,
//! borrowed-from-Graphify circuit breaker: if a SINGLE pass would retire an
//! unusually large fraction of a namespace's live facts, skip applying that
//! batch and make the near-miss loudly visible (counter + warn) instead of
//! either corrupting data or silently proceeding. It fails toward KEEPING
//! facts live, never toward blocking the surrounding operation — an ingest
//! call or dream pass that trips this guard still completes; it just
//! doesn't apply that one batch of retirements.

use metrics::counter;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

/// A single pass retiring MORE than this fraction of a namespace's live
/// facts is refused. Deliberately generous — TD-177 explicitly warns against
/// setting this tight enough to trip on ordinary knowledge-update work,
/// since an over-blocking guard gets disabled and then protects nothing.
const BULK_INVALIDATION_THRESHOLD_FRACTION: f64 = 0.25;

/// Namespaces with fewer live facts than this are EXEMPT from the fraction
/// check. Below this floor, fractions are noisy and pathological — fixing 2
/// of 5 facts (40%) in a small, legitimately-evolving namespace is ordinary
/// work, not an incident.
const BULK_INVALIDATION_FLOOR: i64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BulkInvalidationDecision {
    /// Proceed — apply the batch.
    Allow,
    /// Refuse — do not apply this batch; the caller must skip it.
    Blocked,
}

/// Pure decision logic, no I/O — unit-testable without a database.
pub(crate) fn decide(candidate_count: usize, live_count: i64) -> BulkInvalidationDecision {
    if candidate_count == 0 || live_count < BULK_INVALIDATION_FLOOR {
        return BulkInvalidationDecision::Allow;
    }
    let fraction = candidate_count as f64 / live_count as f64;
    if fraction > BULK_INVALIDATION_THRESHOLD_FRACTION {
        BulkInvalidationDecision::Blocked
    } else {
        BulkInvalidationDecision::Allow
    }
}

/// Count live facts (`expired_at IS NULL`) in `group_id`. Scoped `COUNT(*)`
/// (WHERE-filtered, not bare) — verified safe against this project's own
/// vector-index COUNT(*) trap by precedent: `archive.rs::live_fact_count_for_entity`
/// already runs an identically-shaped scoped COUNT(*) against this same table.
pub(crate) async fn count_live_facts_in_group(
    graph: &TemporalGraph,
    group_id: &str,
) -> Result<i64> {
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM facts WHERE group_id = ?1 AND expired_at IS NULL",
            libsql::params![group_id],
        )
        .await?;
    let row = rows
        .next()
        .await?
        .ok_or_else(|| crate::core::error::Error::Parse("COUNT(*) returned no row".to_string()))?;
    Ok(row.get::<i64>(0)?)
}

/// Decide + emit the counter/warn side effects for a batch of
/// `candidate_count` facts about to be invalidated in `group_id` by
/// `mechanism`, given an ALREADY-KNOWN `live_count`. Emits
/// `kremory.bulk_invalidation_interlock_total{mechanism,outcome}` on BOTH
/// branches (allowed AND blocked) so near-misses are visible before they
/// become incidents — the explicit TD-177 requirement, not an afterthought.
///
/// Split from [`check_bulk_invalidation_interlock`] so a caller that needs to
/// check MULTIPLE batches against a single FIXED baseline (see
/// `ingest_with.rs`'s per-episode cumulative check, TD-177 finding #1 — a
/// per-fact check against a freshly-requeried, already-shrunk live count lets
/// several individually-small batches cumulatively wipe a large fraction of a
/// namespace without ever tripping) can fetch `live_count` ONCE and pass it
/// to this fn repeatedly, rather than re-querying the DB (and re-baselining
/// against an already-diminished count) per sub-batch.
pub(crate) struct BulkInvalidationRecord<'a> {
    pub group_id: &'a str,
    pub mechanism: &'static str,
    pub candidate_count: usize,
    pub live_count: i64,
}

pub(crate) fn record_bulk_invalidation_decision(
    params: BulkInvalidationRecord<'_>,
) -> BulkInvalidationDecision {
    let BulkInvalidationRecord {
        group_id,
        mechanism,
        candidate_count,
        live_count,
    } = params;
    let decision = decide(candidate_count, live_count);
    let outcome = match decision {
        BulkInvalidationDecision::Allow => "allowed",
        BulkInvalidationDecision::Blocked => "blocked",
    };
    counter!(
        "kremory.bulk_invalidation_interlock_total",
        "mechanism" => mechanism,
        "outcome" => outcome,
    )
    .increment(1);
    if decision == BulkInvalidationDecision::Blocked {
        let pct = candidate_count as f64 / live_count.max(1) as f64 * 100.0;
        tracing::warn!(
            target: "kremory.bulk_invalidation_interlock",
            group_id,
            mechanism,
            candidate_count,
            live_count,
            pct,
            "kremory.bulk_invalidation_interlock.blocked — {mechanism} would retire \
             {candidate_count} of {live_count} live facts ({pct:.1}%) in one pass, above \
             the {:.0}% threshold; skipping this batch (TD-177)",
            BULK_INVALIDATION_THRESHOLD_FRACTION * 100.0,
        );
    }
    decision
}

/// Bundled params for [`check_bulk_invalidation_interlock`] — args-as-object
/// (`too_many_arguments` threshold 3).
pub(crate) struct BulkInvalidationCheck<'a> {
    pub graph: &'a TemporalGraph,
    pub group_id: &'a str,
    pub mechanism: &'static str,
    pub candidate_count: usize,
}

/// Single-shot interlock check: fetches `live_count` fresh and decides once.
/// Correct for a caller applying ONE batch in isolation (e.g.
/// `window_closeout`'s single upfront sweep). NOT correct for a caller
/// checking several sub-batches against the SAME namespace within one
/// operation — use [`count_live_facts_in_group`] once +
/// [`record_bulk_invalidation_decision`] per sub-batch for that shape
/// (see its doc comment, TD-177 finding #1).
pub(crate) async fn check_bulk_invalidation_interlock(
    params: BulkInvalidationCheck<'_>,
) -> Result<BulkInvalidationDecision> {
    let BulkInvalidationCheck {
        graph,
        group_id,
        mechanism,
        candidate_count,
    } = params;
    if candidate_count == 0 {
        // Nothing to guard; skip the COUNT(*) round-trip.
        return Ok(BulkInvalidationDecision::Allow);
    }
    let live_count = count_live_facts_in_group(graph, group_id).await?;
    Ok(record_bulk_invalidation_decision(BulkInvalidationRecord {
        group_id,
        mechanism,
        candidate_count,
        live_count,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_correction_in_a_small_namespace_is_allowed() {
        // The false-positive case FIRST (TD-177 explicit requirement): fixing
        // 2 of 5 facts (40%) in a namespace under the floor must NOT trip the
        // guard — this is ordinary knowledge-update work.
        assert_eq!(decide(2, 5), BulkInvalidationDecision::Allow);
    }

    #[test]
    fn ordinary_correction_in_a_large_namespace_is_allowed() {
        // 20 of 200 (10%) is well under threshold — ordinary work at scale.
        assert_eq!(decide(20, 200), BulkInvalidationDecision::Allow);
    }

    #[test]
    fn a_genuine_bulk_wipe_is_blocked() {
        // 81 of 1021 is only 7.9% (would NOT trip at this size) — but the
        // SAME fraction concentrated in a smaller namespace must trip:
        // 30 of 100 = 30%, above the 25% threshold.
        assert_eq!(decide(30, 100), BulkInvalidationDecision::Blocked);
    }

    #[test]
    fn exactly_at_threshold_is_allowed_not_blocked() {
        // 25 of 100 = exactly 25% — the guard is "> threshold", not ">=", so
        // the boundary itself is NOT blocked (avoids off-by-one over-blocking).
        assert_eq!(decide(25, 100), BulkInvalidationDecision::Allow);
    }

    #[test]
    fn one_fact_over_threshold_in_a_large_namespace_is_blocked() {
        assert_eq!(decide(26, 100), BulkInvalidationDecision::Blocked);
    }

    #[test]
    fn zero_candidates_is_always_allowed() {
        assert_eq!(decide(0, 0), BulkInvalidationDecision::Allow);
        assert_eq!(decide(0, 1000), BulkInvalidationDecision::Allow);
    }

    #[test]
    fn a_namespace_exactly_at_the_floor_uses_the_fraction_check() {
        // At the floor boundary the fraction check applies: 6 of 20 = 30%,
        // above threshold.
        assert_eq!(decide(6, 20), BulkInvalidationDecision::Blocked);
    }

    #[test]
    fn a_namespace_one_below_the_floor_is_exempt() {
        // 19 live facts is BELOW the floor — exempt regardless of fraction,
        // even a 100% wipe.
        assert_eq!(decide(19, 19), BulkInvalidationDecision::Allow);
    }
}
