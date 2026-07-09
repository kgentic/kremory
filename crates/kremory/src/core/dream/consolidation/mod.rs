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
//! The four ops are fully implemented (P1-P4) and each adopts the uniform ADR-070
//! `emit_decision` telemetry contract. `run_consolidation` (DoD-P0.4) dispatches them
//! in the strict dependency order supersession→archive→cross_episode→communities
//! (§2.6), each gated by its `DreamOpts.include_*` flag + a shared soft
//! `ConsolidationBudget` pre-check, and fires the `on_merge_proposed` consumer event
//! (ADR-070 Fork 5) for each cross_episode merge decision. Wired into
//! `facade/dream.rs` after the reconciliation chain; inert by default (all
//! `include_*` flags default `false`).
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

/// Sibling of [`OP_TOKEN_PROJECTION`] for the USD-micro ceiling (TD-060). Same
/// coarse-gate rationale: 0 until an op's real per-call USD projection is wired.
const OP_USD_PROJECTION: u64 = 0;

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
    let mut budget = ConsolidationBudget::new(
        opts.consolidation_budget_tokens,
        opts.consolidation_budget_usd_micro,
    );

    // 1. supersession (P1) — build FIRST; feeds archive's `expired_at` input.
    if opts.include_supersession_sweep {
        if budget.check(OP_TOKEN_PROJECTION) && budget.check_usd(OP_USD_PROJECTION) {
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
            skip("supersession", &budget, &mut summary);
        }
    }

    // 2. archive (P2) — consumes supersession's `expired_at` output.
    if opts.include_fact_archival {
        if budget.check(OP_TOKEN_PROJECTION) && budget.check_usd(OP_USD_PROJECTION) {
            let grace_days = opts.archive_grace_days.unwrap_or(0);
            let report = archive::archive(graph, group_id, grace_days).await;
            fold(&mut summary.facts_archived, report, &mut summary.warnings);
        } else {
            skip("archive", &budget, &mut summary);
        }
    }

    // 3. cross_episode (P3) — runs after canonicalize (reconciliation chain), so it
    //    only handles the exact/fuzzy cases canonicalize's cosine band missed.
    if opts.include_cross_episode_merges {
        if budget.check(OP_TOKEN_PROJECTION) && budget.check_usd(OP_USD_PROJECTION) {
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
            skip("cross_episode", &budget, &mut summary);
        }
    }

    // 4. communities (P4) — LAST; benefits from a settled entity population.
    if opts.include_community_detection {
        if budget.check(OP_TOKEN_PROJECTION) && budget.check_usd(OP_USD_PROJECTION) {
            let report = communities::communities(graph, group_id).await;
            fold(
                &mut summary.communities_updated,
                report,
                &mut summary.warnings,
            );
        } else {
            skip("communities", &budget, &mut summary);
        }
    }

    check_net_mutation_warn(group_id, opts.net_mutation_warn_floor, &summary);

    Ok(summary)
}

/// TD-106 (ADR-071 §Item 4b) — net-mutation warn guard. Pure post-aggregation
/// check, no change to any op's own decision logic. `communities_updated` is
/// EXCLUDED: a community "update" is a recomputed partition write, not a
/// destructive mutation of fact/entity identity (ADR-071 Item 2's own
/// reversibility argument), so including it would conflate a safe, fully
/// reversible op with the three ops that DO destroy/move source rows.
///
/// NOTE (Vera MED-2, carried from impl-spec §4b): supersession→archive are
/// coupled (the dispatcher runs supersession before archive; archive consumes
/// its `expired_at` output). A fact superseded THIS pass could — rarely, given
/// `archive_grace_days` default 90 — also be archived same-pass, double-counting
/// one logical retirement toward the floor. Acceptable for a WARN-ONLY guard: no
/// correctness risk, at worst a slightly-early warn. Documented, not corrected.
///
/// Extracted as its own fn (rather than inlined in `run_consolidation`) so unit
/// tests can drive the arithmetic + counter directly against a constructed
/// `ConsolidationSummary`, without needing a fixture graph that produces real
/// non-zero op counts.
fn check_net_mutation_warn(group_id: &str, floor: Option<usize>, summary: &ConsolidationSummary) {
    let Some(floor) = floor else { return };
    let net_mutations =
        summary.cross_episode_merges + summary.supersessions_recorded + summary.facts_archived;
    if net_mutations > floor {
        // Always-on trace (not KREMORY_DEBUG-gated) per ADR-071 §Item 5 table:
        // "a warn is itself the operator signal."
        tracing::warn!(
            target: "kremory.dream.consolidation",
            group_id,
            net_mutations,
            floor,
            "consolidation pass exceeded net-mutation warn floor"
        );
        counter!("kremory.dream.consolidation.net_mutation_warn_total").increment(1);
    }
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

/// Record a budget skip: emit the source-attributed counter (fires identically
/// whether the token or the USD-micro ceiling tripped — TD-060), set the summary's
/// typed `budget_exhausted` flag, and append a warning. `KREMORY_DEBUG`-gated trace
/// carries the USD ceiling/used values as FIELDS (never counter labels), mirroring
/// the pattern at `cross_episode.rs`'s corroboration-score debug trace.
fn skip(op: &str, budget: &ConsolidationBudget, summary: &mut ConsolidationSummary) {
    counter!("kremory.dream.consolidation.budget_skip_total", "op" => op.to_string()).increment(1);
    summary.budget_exhausted = true;
    summary.warnings.push(format!(
        "consolidation op '{op}' skipped: budget ceiling reached"
    ));
    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory.dream.consolidation",
            op,
            used_tokens = budget.used_tokens,
            ceiling_tokens = budget.ceiling_tokens,
            used_usd_micro = budget.used_usd_micro,
            ceiling_usd_micro = ?budget.ceiling_usd_micro,
            "consolidation op skipped: budget ceiling reached"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::TemporalGraph;
    use crate::memory::types::DreamOpts;

    #[tokio::test]
    async fn run_consolidation_inert_when_all_ops_off() {
        // With every consolidation flag explicitly OFF, `any_consolidation_enabled()`
        // is false (facade skips the dispatcher) and the dispatcher is itself inert
        // (all-zero summary) if called. NOTE: this is NO LONGER the DEFAULT — ADR-071
        // Item 1 enabled `include_cross_episode_merges` (SHADOW) by default, so the
        // flags are pinned OFF explicitly here. The enabled path is covered by
        // `run_consolidation_stub_ops_return_zero_when_enabled` + the cross_episode
        // shadow tests + the P3 corpus gate.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let opts = DreamOpts {
            include_cross_episode_merges: false,
            include_community_detection: false,
            include_supersession_sweep: false,
            include_fact_archival: false,
            ..DreamOpts::default()
        };
        assert!(!opts.any_consolidation_enabled());
        let summary = run_consolidation(RunConsolidationParams {
            graph: &graph,
            group_id: "g_all_off",
            opts: &opts,
            model_id: "gemma4:e4b",
            sink: None,
        })
        .await
        .expect("run_consolidation");
        assert_eq!(summary, ConsolidationSummary::default());
    }

    #[tokio::test]
    async fn run_consolidation_ops_return_zero_on_empty_graph_when_enabled() {
        // Enable all four ops. They are REAL impls (not stubs); on an EMPTY graph each
        // finds no candidates and returns count 0, so the summary is still all-zero —
        // this exercises every dispatch arm + budget pre-check (per-op logic coverage
        // lives in each op's own tests, not here).
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
        assert!(
            !summary.budget_exhausted,
            "zero-projection op does not trip the token budget_exhausted flag"
        );
    }

    // ── USD budget (TD-060) ─────────────────────────────────────────────────────

    #[test]
    fn budget_exhausted_flag_set_on_usd_skip() {
        // Synthetic (impl-spec §4a DoD): `skip()` is the single call site that sets
        // `ConsolidationSummary.budget_exhausted`, fired identically whether the
        // token or the USD ceiling tripped. OP_TOKEN_PROJECTION/OP_USD_PROJECTION
        // are both 0 at this stage (no op yet spends token/USD budget), so a real
        // `run_consolidation` skip can't be triggered non-synthetically — assert the
        // flag-setting mechanism directly instead.
        let mut summary = ConsolidationSummary::default();
        assert!(!summary.budget_exhausted, "starts false");
        let budget = ConsolidationBudget::new(Some(0), Some(0));
        skip("supersession", &budget, &mut summary);
        assert!(
            summary.budget_exhausted,
            "skip() must set budget_exhausted regardless of which ceiling tripped"
        );
    }

    #[tokio::test]
    async fn budget_exhausted_not_set_when_usd_projection_zero() {
        // OP_USD_PROJECTION is currently 0 for every op (no op yet spends USD) — a
        // zero USD ceiling with a zero projection still passes check_usd (0<=0), so
        // no op is skipped and budget_exhausted stays false. Mirrors
        // run_consolidation_zero_ceiling_allows_zero_projection_ops for the USD path.
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let opts = DreamOpts {
            include_supersession_sweep: true,
            consolidation_budget_usd_micro: Some(0),
            ..DreamOpts::default()
        };
        let summary = run_consolidation(RunConsolidationParams {
            graph: &graph,
            group_id: "g_usd_budget",
            opts: &opts,
            model_id: "gemma4:e4b",
            sink: None,
        })
        .await
        .expect("run_consolidation");
        assert_eq!(summary.supersessions_recorded, 0);
        assert!(
            !summary.budget_exhausted,
            "zero-projection op does not trip the usd budget_exhausted flag"
        );
    }

    // ── Net-mutation warn guard (TD-106) ────────────────────────────────────────

    fn summary_with(
        cross_episode_merges: usize,
        supersessions_recorded: usize,
        facts_archived: usize,
        communities_updated: usize,
    ) -> ConsolidationSummary {
        ConsolidationSummary {
            communities_updated,
            cross_episode_merges,
            supersessions_recorded,
            facts_archived,
            warnings: Vec::new(),
            budget_exhausted: false,
        }
    }

    /// Does `net_mutation_warn_total` (zero labels) carry the given count?
    fn net_mutation_warn_counter(snapshotter: &metrics_util::debugging::Snapshotter) -> Option<u64> {
        use metrics_util::debugging::DebugValue;
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .find_map(|(composite_key, _, _, value)| {
                let key = composite_key.key();
                if key.name() != "kremory.dream.consolidation.net_mutation_warn_total" {
                    return None;
                }
                match value {
                    DebugValue::Counter(n) => Some(n),
                    _ => None,
                }
            })
    }

    #[test]
    fn net_mutation_warn_fires_above_floor() {
        use metrics_util::debugging::DebuggingRecorder;
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        // floor=2, net=3 (1 cross_episode + 1 supersession + 1 archive = 3 > 2).
        let summary = summary_with(1, 1, 1, 0);
        check_net_mutation_warn("g1", Some(2), &summary);

        assert_eq!(
            net_mutation_warn_counter(&snapshotter),
            Some(1),
            "net_mutation_warn_total must fire once when net_mutations(3) > floor(2)"
        );
    }

    #[test]
    fn net_mutation_warn_silent_below_floor() {
        use metrics_util::debugging::DebuggingRecorder;
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        // floor=10, net=3 — well below floor, no fire.
        let summary = summary_with(1, 1, 1, 0);
        check_net_mutation_warn("g1", Some(10), &summary);

        assert_eq!(
            net_mutation_warn_counter(&snapshotter),
            None,
            "net_mutation_warn_total must NOT fire when net_mutations(3) <= floor(10)"
        );
    }

    #[test]
    fn net_mutation_warn_floor_none_disables_check() {
        use metrics_util::debugging::DebuggingRecorder;
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        // Arbitrarily large net count, floor disabled (None) — never fires.
        let summary = summary_with(1_000_000, 1_000_000, 1_000_000, 0);
        check_net_mutation_warn("g1", None, &summary);

        assert_eq!(
            net_mutation_warn_counter(&snapshotter),
            None,
            "net_mutation_warn_total must never fire when floor is None"
        );
    }

    #[test]
    fn net_mutation_warn_excludes_communities_updated() {
        use metrics_util::debugging::DebuggingRecorder;
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        // Large communities_updated ALONE (all three destructive counts at 0) must
        // NOT trip the floor — communities is excluded from the formula.
        let summary = summary_with(0, 0, 0, 1_000_000);
        check_net_mutation_warn("g1", Some(2), &summary);

        assert_eq!(
            net_mutation_warn_counter(&snapshotter),
            None,
            "communities_updated must be excluded from the net-mutation formula"
        );
    }
}
