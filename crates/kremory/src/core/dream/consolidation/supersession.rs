//! CONSOLIDATION op — supersession sweep (P1).
//!
//! Two lanes:
//! - **deterministic world-time window close-out** (default, this op's payload):
//!   retire facts whose world-time window has demonstrably ended but were never
//!   marked expired — `valid_to IS NOT NULL AND valid_to < now AND expired_at IS
//!   NULL AND invalid_at IS NULL AND is_dream_generated = 0`. Set
//!   `expired_at = valid_to` via `invalidate_fact`. Pure date-compare, ZERO LLM.
//!   Orthogonal to ingest's same-object dedup (`contradiction.rs:285-294`) — that
//!   keys on same-object duplicates + overlap, NOT on a closed `valid_to`.
//! - **optional LLM-nominated value-change** (default-off via
//!   `include_llm_nominate`): different-object value-changes where the LLM only
//!   NOMINATES the pairing and a deterministic emit-invariant REJECTS any
//!   nomination with `newer.valid_from <= older.valid_from` (an LLM can never
//!   invert the arrow of time). **STUB-OFF this phase** — the lane is plumbed but
//!   returns 0 with a documented no-op (P1 ships the deterministic lane fully).
//!
use chrono::{DateTime, Utc};
use metrics::counter;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::{
    emit_decision, ConsolidationBudget, ConsolidationOpKind, DecisionMode, DecisionRecord, OpReport,
};

/// Bundled params for [`supersession`] — args-as-object
/// (`too_many_arguments` threshold 3). `graph` is the receiver-like lead dep.
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)`
/// so the deterministic corpus harness (`tests/consolidation_supersession_test.rs`)
/// can construct it under `feature = "test-utils"` (external test binaries cannot
/// import `pub(crate)` items — E0365). NOT part of the stable public API.
#[doc(hidden)]
pub struct SupersessionParams<'a> {
    pub graph: &'a TemporalGraph,
    pub group_id: &'a str,
    /// Shared soft budget (the optional LLM lane advances it via `record`).
    pub budget: &'a mut ConsolidationBudget,
    /// Select the optional LLM-nominated value-change lane (P1.3). Default-off.
    pub include_llm_nominate: bool,
    /// Resolved dream model id, threaded for the LLM lane.
    pub model_id: &'a str,
}

/// One window-closeout candidate: a fact whose `valid_to` window has already
/// closed. `expired_at` is set to `valid_to` (the demonstrated close-out time).
#[derive(Debug, Clone, PartialEq, Eq)]
struct WindowCloseoutCandidate {
    fact_id: i64,
    /// The RFC3339 `valid_to` string — becomes the fact's `expired_at`.
    valid_to: String,
}

/// Emit-invariant for the LLM value-change lane (P1.3): a nomination is VALID
/// only when the newer fact's `valid_from` is strictly AFTER the older fact's.
/// An LLM can never invert the arrow of time; a nomination that would supersede a
/// newer fact with an older one is REJECTED structurally (returns `false`).
///
/// This is a pure function (unit-tested) so the emit guard is enforced at the
/// emission boundary, not via a prompt instruction
/// (`load-bearing-invariants-at-emit-not-prompt`).
#[cfg_attr(not(test), allow(dead_code))] // wired into the LLM lane (stub-off P1); unit-tested now.
fn llm_nomination_is_time_ordered(
    older_valid_from: DateTime<Utc>,
    newer_valid_from: DateTime<Utc>,
) -> bool {
    newer_valid_from > older_valid_from
}

/// Run the supersession sweep over `group_id`.
///
/// P1 ships the **deterministic window-closeout lane** fully. The optional
/// LLM-nominated value-change lane is plumbed but STUB-OFF (returns 0 with a
/// documented no-op) — it is default-off (no functional-predicate registry) and
/// its emit-invariant is unit-tested via [`llm_nomination_is_time_ordered`].
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)`
/// so the deterministic corpus harness can call it under `feature = "test-utils"`.
/// NOT part of the stable public API.
#[doc(hidden)]
pub async fn supersession(params: SupersessionParams<'_>) -> Result<OpReport> {
    let SupersessionParams {
        graph,
        group_id,
        budget: _budget,
        include_llm_nominate,
        model_id: _model_id,
    } = params;

    let mut report = OpReport::default();

    // ── Lane 1: deterministic world-time window close-out (P1.1) ─────────────────
    let window_closed = window_closeout(graph, group_id).await?;
    report.count += window_closed;

    // ── Lane 2: optional LLM-nominated value-change (P1.3) — STUB-OFF this phase ──
    if include_llm_nominate {
        // The lane is plumbed (flag threaded, emit-invariant `llm_nomination_is_
        // time_ordered` implemented + unit-tested) but its nominator is NOT built
        // this phase. It is default-off (no functional-predicate registry makes
        // auto-superseding different-object facts unsafe without LLM nomination).
        // Documented no-op rather than a silent skip so the surface is observable.
        counter!("kremory.dream.consolidation.supersession_llm_nominate_stub_off_total",)
            .increment(1);
        // Uniform decision record alongside the existing counter (Fork 2/3).
        // No entity refs (the lane is a whole-op stub-off signal, not a
        // per-entity decision); supersession has no dry_run concept → `Applied`.
        emit_decision(DecisionRecord {
            op: ConsolidationOpKind::Supersession,
            mode: DecisionMode::Applied,
            outcome: "llm_nominate_stub_off",
            group_id,
            entity_refs: &[],
            debug_context: None,
        });
        report.warnings.push(
            "supersession LLM-nominate lane requested but is stub-off this phase (P1 \
             ships the deterministic window-closeout lane only)"
                .to_string(),
        );
    }

    Ok(report)
}

/// Deterministic window close-out (P1.1): retire facts whose world-time window
/// (`valid_to`) has already ended but which were never marked expired.
///
/// SELECT predicate (P1.1, P1.2 — the double-handle guard is the `expired_at IS
/// NULL AND invalid_at IS NULL` exclusion; the anti-loop guard is
/// `is_dream_generated = 0`):
///
/// ```sql
/// valid_to IS NOT NULL AND valid_to < now
///   AND expired_at IS NULL AND invalid_at IS NULL
///   AND is_dream_generated = 0
///   AND group_id = ?
/// ```
///
/// For each match, set `expired_at = valid_to` via `invalidate_fact`. Returns the
/// count of retirements THIS sweep applied (never ingest-handled facts — those
/// are excluded by construction). Idempotent: a retired fact now has `expired_at`
/// set, so the next run's SELECT excludes it (second run → 0).
///
/// The whole sweep runs inside ONE `BEGIN IMMEDIATE` so the SELECT + the batch of
/// `expired_at` writes are atomic (no partial retirement on a mid-sweep failure).
///
/// **Visibility (D4, consumer-API hardening):** `pub(crate)` so the
/// `SupersedeRequest::close_now()` builder (`facade/supersede.rs`) can run the
/// retirement inline for a one-call close of already-past-dated bounds. NOT part of
/// the stable public API — the consumer entry point is the builder.
pub(crate) async fn window_closeout(graph: &TemporalGraph, group_id: &str) -> Result<usize> {
    // RFC3339 UTC strings sort lexicographically in chronological order, so the
    // `valid_to < now` string compare is a correct temporal compare (all temporal
    // columns are stored via `to_rfc3339()`, e.g. `graph/facts.rs`).
    let now = Utc::now().to_rfc3339();

    let guard = graph.begin_immediate_if_needed().await?;
    let result: Result<Vec<i64>> = async {
        // Candidate SELECT — the double-handle guard (`expired_at IS NULL AND
        // invalid_at IS NULL`, P1.2) excludes any fact the ingest resolver already
        // superseded/invalidated; `is_dream_generated = 0` (F-4 anti-loop) excludes
        // dream output so a dream-retired fact is never re-retired.
        let mut rows = graph
            .conn
            .query(
                "SELECT id, valid_to FROM facts \
                 WHERE valid_to IS NOT NULL \
                   AND valid_to < ?1 \
                   AND expired_at IS NULL \
                   AND invalid_at IS NULL \
                   AND is_dream_generated = 0 \
                   AND group_id = ?2",
                libsql::params![now.clone(), group_id],
            )
            .await?;

        let mut candidates: Vec<WindowCloseoutCandidate> = Vec::new();
        while let Some(row) = rows.next().await? {
            let fact_id: i64 = row.get(0)?;
            // `valid_to` is NOT NULL by the SELECT predicate, so this is always Some.
            let valid_to: String = row.get(1)?;
            candidates.push(WindowCloseoutCandidate { fact_id, valid_to });
        }
        drop(rows);

        // TD-177: an interlock on bulk fact retirement. Even though window
        // close-out only retires facts whose world-time window has objectively
        // already ended (deterministic, not a judgment call), a single sweep
        // wanting to retire an unusually large fraction of a namespace in one
        // pass is still a symptom worth surfacing — e.g. a batch of facts
        // inserted with wrong `valid_to` dates upstream. Checked BEFORE the
        // retirement loop so this is all-or-nothing per sweep: if blocked, none
        // of this sweep's candidates are retired (they remain live and will be
        // re-considered by the NEXT sweep) rather than the sweep itself failing.
        let interlock_decision = crate::core::graph::check_bulk_invalidation_interlock(
            crate::core::graph::BulkInvalidationCheck {
                graph,
                group_id,
                mechanism: "window_closeout",
                candidate_count: candidates.len(),
            },
        )
        .await?;

        // Collect the retired fact ids; the per-fact decision records are emitted ONLY
        // after this txn durably commits (see the Ok arm) so a mid-sweep rollback never
        // leaves an un-revertable decision_total increment.
        let mut retired_ids: Vec<i64> = Vec::new();
        let candidates_to_retire: &[WindowCloseoutCandidate] =
            if interlock_decision == crate::core::graph::BulkInvalidationDecision::Allow {
                &candidates
            } else {
                &[]
            };
        for cand in candidates_to_retire {
            // Set `expired_at = valid_to` — the demonstrated window-close time, NOT
            // `now` (the fact expired when its world-time window ended, not when the
            // sweep noticed). `invalidate_fact` runs `UPDATE facts SET expired_at=?1
            // WHERE id=?2` (`graph/facts.rs:140-146`, verified).
            // Parse loudly: a malformed `valid_to` is a real data-corruption signal
            // (every temporal column is written via `to_rfc3339()`), so surface it as
            // `Error::Parse` — the dispatcher's warn-and-continue folds it into the
            // summary rather than silently skipping the fact (`llm-output-parse-loudly`).
            let valid_to_dt: DateTime<Utc> = DateTime::parse_from_rfc3339(&cand.valid_to)
                .map_err(|e| {
                    crate::core::error::Error::Parse(format!(
                        "window-closeout: fact {} has unparseable valid_to {:?}: {e}",
                        cand.fact_id, cand.valid_to
                    ))
                })?
                .with_timezone(&Utc);
            graph.invalidate_fact(cand.fact_id, valid_to_dt).await?;
            retired_ids.push(cand.fact_id);
        }
        Ok(retired_ids)
    }
    .await;

    match result {
        Ok(retired_ids) => {
            guard.commit().await?;
            let retired = retired_ids.len();
            // Post-commit emission of the per-fact decision records (Fork 2/3):
            // fired ONLY after the txn durably commits so
            // decision_total{op=supersession,outcome=window_closeout} never overcounts
            // a mid-sweep rollback — mirroring the post-commit
            // counter below, which is why the two stay equal. fact id is
            // HIGH-cardinality → trace-only via debug_context (Fork 3); no entity refs
            // are loaded on this fact-level lane; no dry_run → `Applied`.
            for fact_id in &retired_ids {
                emit_decision(DecisionRecord {
                    op: ConsolidationOpKind::Supersession,
                    mode: DecisionMode::Applied,
                    outcome: "window_closeout",
                    group_id,
                    entity_refs: &[],
                    debug_context: Some(format!("fact_id={fact_id}")),
                });
            }
            // P1.4: source-attributed counter split by lane. This lane == the count
            // this op reports for the deterministic path; the o11y cross-check asserts
            // counter == OpReport.count.
            counter!(
                "kremory.dream.consolidation.supersessions_recorded_total",
                "lane" => "window_closeout",
            )
            .increment(retired as u64);
            tracing::info!(
                target: "kremory.dream.consolidation.supersession",
                group_id,
                retired,
                "window-closeout sweep complete"
            );
            Ok(retired)
        }
        Err(e) => {
            let _ = guard.rollback().await;
            Err(e)
        }
    }
}

#[cfg(test)]
#[path = "supersession_tests.rs"]
mod tests;
