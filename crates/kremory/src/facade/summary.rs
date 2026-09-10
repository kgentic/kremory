//! Dream/recall RESULT + TEMPLATE types for the facade — the shapes a consumer
//! reads back, split out of `facade/mod.rs` (TD-043).
//!
//! `ConsolidationOpsRan` / `DreamSummary` are what `dream()` returns;
//! `RecallTemplate` is the rendering choice `recall()` takes. They share a file
//! because they are the facade's plain data types: no `Memory` methods live here,
//! only the structs, their conversions from core-layer results, and the tests over
//! those conversions.

use super::*;

// ── ConsolidationOpsRan (D1b) ─────────────────────────────────────────────────

/// Per-op ACTUALLY-RAN signal on [`DreamSummary`] (consumer-API hardening D1b).
///
/// With the D1 all-ops-ON defaults a consumer can no longer read an all-zero
/// consolidation count as "the op was off". This makes the distinction in-band: a
/// field is `true` iff the op's `DreamOpts.include_*` flag was on AND the op
/// actually executed (it is `false` when the op was toggled off OR budget-skipped —
/// a budget skip is separately observable via [`DreamSummary::budget_exhausted`]).
///
/// So `communities_updated == 0 && consolidation_ops_ran.community == true` reads
/// as "community detection ran and found nothing to change", NOT "it was disabled".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationOpsRan {
    /// The community-detection op (`include_community_detection`) executed.
    pub community: bool,
    /// The cross-episode merge op (`include_cross_episode_merges`) executed.
    pub cross_episode: bool,
    /// The fact-archival op (`include_fact_archival`) executed.
    pub archival: bool,
    /// The supersession sweep (`include_supersession_sweep`) executed.
    pub supersession_sweep: bool,
}

// ── DreamSummary ─────────────────────────────────────────────────────────────

/// Summary returned when a dream phase completes via the facade.
///
/// **Real fields (populated by `mem.dream()`):**
/// - `types_discovered` — entity types proposed and accepted by Pass 0 type-discovery.
/// - `entities_reclassified` — entities reclassified by the reclassify pass.
/// - `aliases_resolved` — pending potential-alias facts resolved (merged/revoked).
/// - `canonicalization_merges` — near-duplicate entities merged by canonicalize.
/// - `consistency_check_corrected` — entity types corrected by consistency_check.
/// - `duration_ms` — wall-clock time of the dream call.
/// - `warnings` — non-fatal notices from any pass.
///
/// **Consolidation fields (populated by the CONSOLIDATION sub-phase):**
/// - `communities_updated`, `cross_episode_would_merge`, `cross_episode_merged`,
///   `supersessions_recorded`, `facts_archived` — filled by `run_consolidation`
///   when the corresponding `DreamOpts.include_*` op is enabled (all default `true`
///   since consumer-API hardening D1 — made safe by Tier-1 reversibility).
///   Zero when the op is off OR ran and found nothing to do — use
///   `consolidation_ops_ran` (D1b) to tell the two apart.
///
/// # Observability — metrics stack version lock
///
/// The counters kremory emits (`kremory.*` via the [`metrics`] facade) register
/// against the process-global recorder. To READ those counters from a consumer's
/// own recorder/exporter you MUST pin the SAME minor versions kremory links, or
/// the registries silently do not share and you capture ZERO:
///
/// ```toml
/// metrics = "0.24"       # must match kremory's metrics minor
/// metrics-util = "0.18"  # must match kremory's metrics-util minor
/// ```
///
/// A minor-version mismatch (e.g. `metrics = "0.23"`) links a second, incompatible
/// global recorder — kremory's `counter!`/`histogram!` calls hit a registry your
/// exporter never sees. Every count reads `0`. The all-in-band fields on this
/// struct (the D5 would-merge/merged split, [`ConsolidationOpsRan`],
/// [`Self::budget_exhausted`]) are readable WITHOUT any metrics recorder — prefer
/// them when you only need the dream accounting.
#[derive(Debug, Clone)]
pub struct DreamSummary {
    pub communities_updated: usize,
    /// Cross-episode merge DECISIONS this pass — "would-merge" (D5, consumer-API
    /// hardening). Counts every merge decision the op reached on BOTH the shadow and
    /// apply branches, so this does NOT imply entities were actually fused. See
    /// [`Self::cross_episode_merged`] for the count that ACTUALLY committed a fusion
    /// — the in-band would-merge/merged split (no metrics recorder required).
    pub cross_episode_would_merge: usize,
    /// Cross-episode merges that ACTUALLY committed this pass (D5). Equals
    /// [`Self::cross_episode_would_merge`] in apply mode
    /// ([`CrossEpisodeMode::Apply`]) and `0` in shadow mode
    /// ([`CrossEpisodeMode::Shadow`], the default). `would_merge > 0 && merged == 0`
    /// reads as "the op WOULD have merged N pairs but is in shadow — fused nothing".
    pub cross_episode_merged: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    /// Per-op ACTUALLY-RAN signal (D1b) — disambiguates "op disabled" from "op ran,
    /// found nothing" for the all-zero consolidation counts above. See
    /// [`ConsolidationOpsRan`].
    pub consolidation_ops_ran: ConsolidationOpsRan,
    pub duration_ms: u64,
    /// Entity types proposed and accepted by Dream Pass 0 type discovery.
    /// Empty when Pass 0 was not run or produced no accepted proposals.
    pub types_discovered: Vec<crate::core::dream::TypeProposal>,
    /// Total entities reclassified by Dream Pass 2 (catch_all_cascade + low_confidence arms).
    /// Zero when Pass 2 was not run or found no candidates.
    pub entities_reclassified: usize,
    /// Pending `potential_alias` facts resolved (merged or revoked) by the aliases
    /// pass (§D3). Zero when the pass was not run or found no pending alias facts.
    pub aliases_resolved: usize,
    /// Near-duplicate entities merged by the canonicalize pass (§D3). Zero when the
    /// pass was not run or found no merges above `L5_CANONICALIZATION_THRESHOLD`.
    pub canonicalization_merges: usize,
    /// Entity-instance pairs merged by the Site #5 acronym/nickname recall pass.
    /// Zero when opted out (`include_acronym_nickname_recall = false`)
    /// or no acronym/nickname pairs were adjudicated as the same entity.
    pub acronym_nickname_merges: usize,
    /// Near-duplicate `entity_types` rows merged by the Site #3 type-registry
    /// collapse pass. Zero when opted out
    /// (`include_type_registry_collapse = false`) or no type pairs were collapsed.
    pub type_registry_merges: usize,
    /// Entity types corrected by the consistency_check pass. Zero
    /// when opted out (`include_consistency_check = false`) or nothing to correct.
    pub consistency_check_corrected: usize,
    /// Warnings emitted during the dream phase.
    /// Includes degraded-mode notices (e.g. anti-redundancy gate skipped).
    pub warnings: Vec<String>,
    /// `true` when the consolidation
    /// sub-phase's `ConsolidationBudget` (token or USD ceiling) was exhausted this
    /// run, skipping at least one op. Set from the fold site in `facade/dream.rs`
    /// (`consolidation.budget_exhausted`) — inert (`false`) unless a consolidation
    /// op was enabled and its budget ceiling actually tripped.
    pub budget_exhausted: bool,
}

impl From<DreamPhaseResult> for DreamSummary {
    fn from(r: DreamPhaseResult) -> Self {
        Self {
            communities_updated: r.communities_recomputed,
            cross_episode_would_merge: r.cross_meeting_merges,
            // No consolidation dispatcher ran on this `DreamPhaseResult` path (it is
            // freshly `default()`-constructed before consolidation folds in at the
            // facade), so the actual-fusion count + ran-signal are all zero/false;
            // the facade fold site overwrites them with the real values.
            cross_episode_merged: 0,
            consolidation_ops_ran: ConsolidationOpsRan::default(),
            supersessions_recorded: r.supersessions_recorded,
            facts_archived: r.facts_archived,
            duration_ms: r.duration_ms,
            types_discovered: r.types_discovered,
            entities_reclassified: 0,
            aliases_resolved: 0,
            canonicalization_merges: 0,
            acronym_nickname_merges: 0,
            type_registry_merges: 0,
            consistency_check_corrected: 0,
            warnings: r.dream_warnings,
            // Carried through structurally (not hardcoded false) — DreamPhaseResult
            // gained this field for exactly this propagation. In the live
            // `facade/dream.rs` call path `r` is always freshly
            // `DreamPhaseResult::default()`-constructed BEFORE consolidation runs, so
            // this is `false` here and is overwritten by the fold site immediately
            // after with the real `consolidation.budget_exhausted` value.
            budget_exhausted: r.budget_exhausted,
        }
    }
}

impl From<crate::core::ingest::DreamPassSummary> for DreamSummary {
    fn from(s: crate::core::ingest::DreamPassSummary) -> Self {
        Self {
            // DreamPassSummary fields map to DreamSummary where applicable.
            // Fields without a direct mapping are zeroed.
            communities_updated: 0,
            cross_episode_would_merge: s.ghost_episodes_retried,
            cross_episode_merged: 0,
            consolidation_ops_ran: ConsolidationOpsRan::default(),
            supersessions_recorded: 0,
            facts_archived: 0,
            duration_ms: s.duration_ms,
            types_discovered: Vec::new(),
            entities_reclassified: s.entities_reclassified,
            aliases_resolved: 0,
            canonicalization_merges: 0,
            acronym_nickname_merges: 0,
            type_registry_merges: 0,
            consistency_check_corrected: 0,
            warnings: Vec::new(),
            // DreamPassSummary has no budget concept — no consolidation sub-phase
            // ran on this path.
            budget_exhausted: false,
        }
    }
}

#[cfg(test)]
mod dream_summary_budget_exhausted_tests {
    //! Consumer-observability
    //! coverage for `budget_exhausted` propagation. `run_consolidation`'s real op
    //! projections are all 0 at this stage (§Item 4a DoD note), so a true
    //! budget-capped skip cannot be triggered end-to-end through `mem.dream()`
    //! yet — these tests instead prove the STRUCTURAL propagation chain a live
    //! skip will ride once a non-zero projection ships: `ConsolidationSummary`
    //! (mod.rs `skip()`) → `DreamSummary` (facade/dream.rs fold site) and
    //! `DreamPhaseResult` → `DreamSummary` (the `From` impl above).

    use super::*;
    use crate::core::dream::consolidation::ConsolidationSummary;

    #[test]
    fn dream_phase_result_budget_exhausted_field_accessible() {
        // Structural/compile-shape test (mirrors the established
        // `d6_dream_summary_types_discovered_field_accessible` pattern) — the field
        // must exist on DreamPhaseResult and accept a bool value.
        let r = DreamPhaseResult {
            budget_exhausted: true,
            ..Default::default()
        };
        assert!(
            r.budget_exhausted,
            "DreamPhaseResult.budget_exhausted must be settable"
        );
    }

    #[test]
    fn dream_summary_from_dream_phase_result_carries_budget_exhausted() {
        // Proves the `From<DreamPhaseResult>` impl above threads the flag through
        // rather than hardcoding `false`.
        let r = DreamPhaseResult {
            budget_exhausted: true,
            ..Default::default()
        };
        let summary: DreamSummary = r.into();
        assert!(
            summary.budget_exhausted,
            "DreamSummary::from(DreamPhaseResult) must carry budget_exhausted through, not hardcode false"
        );
    }

    #[test]
    fn dream_summary_fold_site_pattern_carries_consolidation_budget_exhausted() {
        // Proves the exact assignment shape used at the `facade/dream.rs` fold site
        // (`summary.budget_exhausted = consolidation.budget_exhausted;`) correctly
        // reads a `true` ConsolidationSummary.budget_exhausted through to the
        // consumer-facing DreamSummary (a live
        // budget-capped run cannot yet drive this end-to-end per the module doc
        // above, so this test exercises the fold assignment directly).
        let consolidation = ConsolidationSummary {
            budget_exhausted: true,
            ..Default::default()
        };
        let mut summary = DreamSummary::from(DreamPhaseResult::default());
        assert!(!summary.budget_exhausted, "starts false (no skip yet)");
        summary.budget_exhausted = consolidation.budget_exhausted;
        assert!(
            summary.budget_exhausted,
            "fold-site assignment must carry a tripped ConsolidationSummary.budget_exhausted through to DreamSummary"
        );
    }
}

// ── RecallTemplate ────────────────────────────────────────────────────────────

/// Render strategy for a `RecallRequest`. Facade-level enum mapping to
/// `ContextTemplate` variants with a stable, serializable `as_str` surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallTemplate {
    /// Render entities with name + summary. Matches `ContextTemplate::Entities`.
    Entities,
    /// Render one-line-per-edge compact summary. Matches `ContextTemplate::EdgeSummary`.
    EdgeSummary,
    /// Render temporal facts with `valid_at` annotations (default).
    TemporalFacts,
}

impl RecallTemplate {
    /// Parse from a string slug. Returns `None` for unknown values.
    pub fn parse_str(s: &str) -> Option<Self> {
        match s {
            "entities" => Some(Self::Entities),
            "edge_summary" => Some(Self::EdgeSummary),
            "temporal_facts" => Some(Self::TemporalFacts),
            _ => None,
        }
    }

    /// Return the stable string slug for this template.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entities => "entities",
            Self::EdgeSummary => "edge_summary",
            Self::TemporalFacts => "temporal_facts",
        }
    }
}

impl From<RecallTemplate> for ContextTemplate {
    fn from(t: RecallTemplate) -> Self {
        match t {
            RecallTemplate::Entities => ContextTemplate::Entities,
            RecallTemplate::EdgeSummary => ContextTemplate::EdgeSummary,
            RecallTemplate::TemporalFacts => ContextTemplate::TemporalFacts,
        }
    }
}

