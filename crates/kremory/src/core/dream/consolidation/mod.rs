//! Dream CONSOLIDATION sub-phase (ADR-066 axes D+E) — graph-global cleanup that
//! runs AFTER the per-entity reconciliation pass chain.
//!
//! Four independently-gated, independently-budgeted, independently-testable ops
//! over ONE shared substrate (`substrate.rs`):
//!
//! 1. `supersession` — retire facts whose world-time window has closed (P1).
//! 2. `archive` — move long-expired, unreferenced facts to `facts_archive` (P2).
//! 3. `cross_episode` — merge the same referent recurring across episodes (P3).
//! 4. `communities` — deterministic label-propagation community detection (P4).
//!
//! The four ops are STUBS this phase (P0) — they return `OpReport::default()`;
//! P1-P4 fill them. `run_consolidation` (DoD-P0.4) dispatches them in the strict
//! dependency order supersession→archive→cross_episode→communities (§2.6), each
//! gated by its `DreamOpts.include_*` flag + a shared soft `ConsolidationBudget`
//! pre-check. Wired into `facade/dream.rs` after the reconciliation chain; inert by
//! default (all `include_*` flags default `false`).
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`.

pub(crate) mod archive;
pub(crate) mod communities;
pub(crate) mod cross_episode;
pub(crate) mod substrate;
pub(crate) mod supersession;

use std::sync::Arc;

use metrics::counter;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;
use crate::memory::events::{EnrichmentEventSink, MergeProposed};
use crate::memory::types::DreamOpts;

pub(crate) use substrate::ConsolidationSummary;
use substrate::{ConsolidationBudget, OpReport};

/// Conservative per-op token projection used for the SOFT budget pre-check. Only
/// supersession's optional LLM lane + community detection ever spend tokens; the
/// deterministic ops project 0. This is a coarse gate — the real spend is recorded
/// via `ConsolidationBudget::record` once an op runs (P1/P4). A precise projection
/// is an op-internal concern folded in as each op lands.
const OP_TOKEN_PROJECTION: u64 = 0;

/// Bundled params for [`run_consolidation`] — args-as-object per TD-042
/// (`too_many_arguments` threshold 3). `graph` is the receiver-like lead dep.
pub(crate) struct RunConsolidationParams<'a> {
    pub(crate) graph: &'a TemporalGraph,
    pub(crate) group_id: &'a str,
    pub(crate) opts: &'a DreamOpts,
    /// Resolved dream model id (TD-094 style), threaded to supersession's LLM lane.
    pub(crate) model_id: &'a str,
    /// Optional consumer event sink. The orchestrator fires `on_merge_proposed` from
    /// here for each cross_episode merge decision (ADR-070 Fork 5, Risk #17
    /// orchestrator-fires) — the sink lives at this layer already (`facade/dream.rs`
    /// `resolve_sink`), so the op itself does not need it threaded in.
    pub(crate) sink: Option<&'a Arc<dyn EnrichmentEventSink>>,
}

/// Dispatch the four consolidation ops over `group_id` in dependency order
/// (ADR-066 §2.6), each gated by its `include_*` flag + a soft budget pre-check.
///
/// Order: **supersession → archive → cross_episode → communities.** supersession's
/// `expired_at` writes are archive's input; cross_episode delegates to the shared
/// merge executor; communities benefits from a settled entity population.
///
/// Non-fatal per op: an op error is captured as a warning and the sweep continues
/// (kremory warn-and-continue posture) — the whole dispatcher only returns `Err`
/// on an unrecoverable substrate failure, and even that is folded via
/// `unwrap_or_default()` at the facade call site.
///
/// A budget skip emits `kremory.dream.consolidation.budget_skip_total{op}` and a
/// warning; later ops still run (soft partial-abort, F-1).
///
/// Args-as-object per TD-042 (`too_many_arguments` threshold 3).
pub(crate) async fn run_consolidation(
    params: RunConsolidationParams<'_>,
) -> Result<ConsolidationSummary> {
    let RunConsolidationParams {
        graph,
        group_id,
        opts,
        model_id,
        sink,
    } = params;
    let mut summary = ConsolidationSummary::default();
    let mut budget = ConsolidationBudget::new(opts.consolidation_budget_tokens);

    // 1. supersession (P1) — build FIRST; feeds archive's `expired_at` input.
    if opts.include_supersession_sweep {
        if budget.check(OP_TOKEN_PROJECTION) {
            let report = supersession::supersession(supersession::SupersessionParams {
                graph,
                group_id,
                budget: &mut budget,
                include_llm_nominate: opts.include_supersession_llm_nominate,
                model_id,
            })
            .await;
            fold(
                &mut summary.supersessions_recorded,
                report,
                &mut summary.warnings,
            );
        } else {
            skip("supersession", &mut summary.warnings);
        }
    }

    // 2. archive (P2) — consumes supersession's `expired_at` output.
    if opts.include_fact_archival {
        if budget.check(OP_TOKEN_PROJECTION) {
            let grace_days = opts.archive_grace_days.unwrap_or(0);
            let report = archive::archive(graph, group_id, grace_days).await;
            fold(&mut summary.facts_archived, report, &mut summary.warnings);
        } else {
            skip("archive", &mut summary.warnings);
        }
    }

    // 3. cross_episode (P3) — runs after canonicalize (reconciliation chain), so it
    //    only handles the exact/fuzzy cases canonicalize's cosine band missed.
    if opts.include_cross_episode_merges {
        if budget.check(OP_TOKEN_PROJECTION) {
            let report =
                cross_episode::cross_episode(graph, group_id, opts.cross_episode_dry_run).await;
            // Orchestrator-fires the consumer event for each merge decision (ADR-070
            // Fork 5, Risk #17 contingency): the sink lives at THIS layer, so
            // cross_episode returns its merge pairs via `OpReport.merges` and we fan
            // them out here. Fires for shadow AND applied decisions (dry_run carried).
            if let (Some(sink), Ok(op)) = (sink, &report) {
                for m in &op.merges {
                    sink.on_merge_proposed(MergeProposed {
                        group_id,
                        loser: &m.loser,
                        keeper: &m.keeper,
                        dry_run: opts.cross_episode_dry_run,
                    });
                }
            }
            fold(
                &mut summary.cross_episode_merges,
                report,
                &mut summary.warnings,
            );
        } else {
            skip("cross_episode", &mut summary.warnings);
        }
    }

    // 4. communities (P4) — LAST; benefits from a settled entity population.
    if opts.include_community_detection {
        if budget.check(OP_TOKEN_PROJECTION) {
            let report = communities::communities(graph, group_id).await;
            fold(
                &mut summary.communities_updated,
                report,
                &mut summary.warnings,
            );
        } else {
            skip("communities", &mut summary.warnings);
        }
    }

    Ok(summary)
}

/// Fold an op's `Result<OpReport>` into the running summary: on `Ok`, add its
/// count + carry any warnings; on `Err`, record a non-fatal warning and leave the
/// count untouched (warn-and-continue).
fn fold(count: &mut usize, report: Result<OpReport>, warnings: &mut Vec<String>) {
    match report {
        Ok(r) => {
            *count += r.count;
            warnings.extend(r.warnings);
        }
        Err(e) => {
            warnings.push(format!("consolidation op failed (non-fatal): {e}"));
        }
    }
}

/// Record a budget skip: emit the source-attributed counter + a summary warning.
fn skip(op: &str, warnings: &mut Vec<String>) {
    counter!("kremory.dream.consolidation.budget_skip_total", "op" => op.to_string()).increment(1);
    warnings.push(format!(
        "consolidation op '{op}' skipped: budget ceiling reached"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::TemporalGraph;
    use crate::memory::types::DreamOpts;

    #[tokio::test]
    async fn run_consolidation_inert_by_default() {
        // Default DreamOpts has all consolidation flags OFF → dispatcher runs no
        // op, returns an all-zero summary (matches the facade's inert-by-default
        // contract). `any_consolidation_enabled()` is false so the facade would not
        // even call this, but the dispatcher must itself be inert if called.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let opts = DreamOpts::default();
        assert!(!opts.any_consolidation_enabled());
        let summary = run_consolidation(RunConsolidationParams {
            graph: &graph,
            group_id: "g_default",
            opts: &opts,
            model_id: "gemma4:e4b",
            sink: None,
        })
        .await
        .expect("run_consolidation");
        assert_eq!(summary, ConsolidationSummary::default());
    }

    #[tokio::test]
    async fn run_consolidation_stub_ops_return_zero_when_enabled() {
        // Enable all four ops — the P0 stubs each return count 0, so the summary is
        // still all-zero, but this exercises every dispatch arm + budget pre-check.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let opts = DreamOpts {
            include_supersession_sweep: true,
            include_fact_archival: true,
            include_cross_episode_merges: true,
            include_community_detection: true,
            ..DreamOpts::default()
        };
        assert!(opts.any_consolidation_enabled());
        let summary = run_consolidation(RunConsolidationParams {
            graph: &graph,
            group_id: "g_all",
            opts: &opts,
            model_id: "gemma4:e4b",
            sink: None,
        })
        .await
        .expect("run_consolidation");
        assert_eq!(summary.supersessions_recorded, 0);
        assert_eq!(summary.facts_archived, 0);
        assert_eq!(summary.cross_episode_merges, 0);
        assert_eq!(summary.communities_updated, 0);
        assert!(summary.warnings.is_empty(), "stub ops emit no warnings");
    }

    #[tokio::test]
    async fn run_consolidation_zero_ceiling_allows_zero_projection_ops() {
        // A zero ceiling denies every op's pre-check (used(0)+proj(0)=0 <= 0 is
        // TRUE, so with OP_TOKEN_PROJECTION=0 the zero-token ops still run). Use a
        // ceiling below a forced non-zero projection is not reachable at P0
        // (projection is 0), so assert the P0 reality: zero ceiling still ALLOWS
        // zero-projection ops. This locks the soft-check arithmetic for P1 when
        // projections become non-zero.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let opts = DreamOpts {
            include_supersession_sweep: true,
            consolidation_budget_tokens: Some(0),
            ..DreamOpts::default()
        };
        let summary = run_consolidation(RunConsolidationParams {
            graph: &graph,
            group_id: "g_budget",
            opts: &opts,
            model_id: "gemma4:e4b",
            sink: None,
        })
        .await
        .expect("run_consolidation");
        // OP_TOKEN_PROJECTION == 0 and ceiling == 0 → check(0) is true → op runs
        // (stub, count 0), no skip warning. This is the correct soft-check behaviour.
        assert_eq!(summary.supersessions_recorded, 0);
        assert!(
            summary.warnings.is_empty(),
            "zero-projection op is not skipped by a zero ceiling"
        );
    }
}
