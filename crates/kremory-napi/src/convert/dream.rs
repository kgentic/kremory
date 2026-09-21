//! Dream napi conversions — `Memory.dream`/pass options, summary, and status types.
//! Split out of `convert.rs` (TD-243); see `convert/mod.rs` for the domain map.

use napi_derive::napi;

use kremory::DreamSummary;

/// A new entity type proposed + accepted by Dream Pass 0 type-discovery.
/// Mirrors substrate `kremory::core::dream::TypeProposal`.
#[napi(object, js_name = "TypeProposal")]
pub struct JsTypeProposal {
    pub name: String,
    pub description: String,
    pub justification: String,
}

/// Per-op ACTUALLY-RAN signal on `DreamSummary`.
///
/// Mirrors substrate `kremory::ConsolidationOpsRan`. With the all-ops-ON dream
/// defaults, a consumer can no longer read an all-zero consolidation count as
/// "the op was off" — a field here is `true` iff the op's `include_*` flag was on
/// AND the op executed. So `communities_updated == 0 && consolidationOpsRan.community
/// == true` reads as "community detection ran and found nothing", NOT "disabled".
#[napi(object, js_name = "ConsolidationOpsRan")]
pub struct JsConsolidationOpsRan {
    /// The community-detection op (`includeCommunityDetection`) executed.
    pub community: bool,
    /// The cross-episode merge op (`crossEpisodeMode != "off"`) executed.
    pub cross_episode: bool,
    /// The fact-archival op (`includeFactArchival`) executed.
    pub archival: bool,
    /// The supersession sweep (`includeSupersessionSweep`) executed.
    pub supersession_sweep: bool,
}

/// Summary returned by `Memory.dream`.
///
/// Field names mirror the substrate `DreamSummary` struct exactly. All numeric
/// fields are safe as JS `number` (f64) — usize/u64 values far below 2^53 at
/// practical memory scale.
///
/// The `crossEpisodeWouldMerge` / `crossEpisodeMerged` split is the in-band
/// would-merge/did-merge distinction — `wouldMerge` counts every merge DECISION
/// (both shadow + apply branches), `merged` counts only ACTUAL fusions (equals
/// `wouldMerge` in apply mode, `0` in shadow). Use `consolidationOpsRan` to
/// tell "op disabled" from "op ran, found nothing" on any all-zero count.
#[napi(object, js_name = "DreamSummary")]
pub struct JsDreamSummary {
    pub communities_updated: f64,
    /// Cross-episode merge DECISIONS this pass ("would-merge"). Does NOT imply
    /// entities were fused — see `crossEpisodeMerged`.
    pub cross_episode_would_merge: f64,
    /// Cross-episode merges that ACTUALLY committed this pass. Equals
    /// `crossEpisodeWouldMerge` in apply mode, `0` in shadow (the default).
    pub cross_episode_merged: f64,
    pub supersessions_recorded: f64,
    pub facts_archived: f64,
    /// Per-op ran-signal — disambiguates "op disabled" from "op ran, found
    /// nothing" for the all-zero consolidation counts above.
    pub consolidation_ops_ran: JsConsolidationOpsRan,
    /// `true` when a consolidation op was skipped because a per-pass budget
    /// ceiling (token or USD) tripped — distinguishes "nothing to spend" from
    /// "spend was capped".
    pub budget_exhausted: bool,
    pub duration_ms: f64,
    pub types_discovered: Vec<JsTypeProposal>,
    pub entities_reclassified: f64,
    pub aliases_resolved: f64,
    pub canonicalization_merges: f64,
    pub acronym_nickname_merges: f64,
    pub type_registry_merges: f64,
    pub consistency_check_corrected: f64,
    pub warnings: Vec<String>,
}

/// Options for `JsMemory.dream`. All fields are optional — omit to use the
/// substrate `DreamOpts::default()` — all four consolidation ops default ON,
/// made safe by Tier-1 reversibility (each op is undoable/restorable).
///
/// # DreamOpts field enumeration
///
/// The substrate `DreamOpts` has 18 pub fields. Each is either EXPOSED here or
/// deliberately OMITTED with a reason:
///
/// **Exposed (consolidation control — the binding-parity surface):**
/// `includeCommunityDetection`, `includeFactArchival`, `includeSupersessionSweep`,
/// `includeSupersessionLlmNominate`, `crossEpisodeMode` (the honest tri-state that
/// maps onto `include_cross_episode_merges` + `cross_episode_dry_run` —
/// exposed as ONE string, never the two raw bools, to prevent the illegal
/// `apply`+`dry_run` combo), `archiveGraceDays`, `netMutationWarnFloor`,
/// `consolidationBudgetTokens`, `consolidationBudgetUsdMicro`.
///
/// **Omitted (with reason):**
/// - `since`, `maxEpisodesPerRun` — run-scoping filters; the napi `dream()` operates
///   over all un-dreamed episodes. Per-run scoping is reachable via
///   `runDreamPassSync`'s `DreamPassOptions.maxEpisodesPerRun`; a timestamp `since`
///   filter is deferred to a future release.
/// - `includeTypeDiscovery`, `includeConsistencyCheck`, `includeTypeRegistryCollapse`,
///   `includeAcronymNicknameRecall`, `includeTypeNoveltyLlmVerify` — RECONCILIATION
///   pass toggles (a different concern from the consolidation sub-phase this
///   surface targets). All default-ON; type-discovery is separately tunable via
///   `runDreamPassSync`'s `DreamPassOptions.includeTypeDiscovery`. Exposing the full
///   reconciliation-pass matrix on `dream()` is deferred to a future release.
/// - `includeEvidenceRetypeBySimilarity` — reason: unvalidated and quarantined
///   pending further evaluation, default-false. NOT exposed on `JsDreamOpts`;
///   the default-false pass-through (substrate `DreamOpts::default()`) is the
///   correct binding behaviour until the quarantine lifts.
#[napi(object, js_name = "DreamOptions")]
pub struct JsDreamOpts {
    /// Namespace to dream within. `null`/omit for Memory handle's default.
    pub namespace: Option<String>,
    /// Run the community-detection consolidation op (zero-LLM label propagation).
    /// Default `true`.
    pub include_community_detection: Option<bool>,
    /// Run the fact-archival consolidation op (MOVE long-expired unreferenced facts
    /// into `facts_archive`; reversible via `restoreArchivedFact`). Default `true`.
    pub include_fact_archival: Option<bool>,
    /// Run the supersession sweep (deterministic world-time window close-out;
    /// reversible via `unsupersede`). Default `true`.
    pub include_supersession_sweep: Option<bool>,
    /// Also run supersession's OPT-IN LLM-nominated value-change lane (only
    /// meaningful when `includeSupersessionSweep` is on). Default `false`.
    pub include_supersession_llm_nominate: Option<bool>,
    /// Cross-episode entity-merge control. One of `"off"` | `"shadow"` |
    /// `"apply"` — the honest tri-state that maps onto the substrate's coupled
    /// `include_cross_episode_merges` + `cross_episode_dry_run` bools. Default
    /// (omit) = the substrate default (`"shadow"` — computes decisions, fuses
    /// nothing). An unknown value rejects the `dream()` Promise.
    pub cross_episode_mode: Option<String>,
    /// Grace window (days) before an expired fact is archival-eligible. Default `90`.
    pub archive_grace_days: Option<f64>,
    /// WARN-only floor on the aggregate destructive-mutation count. Default `500`.
    pub net_mutation_warn_floor: Option<f64>,
    /// Per-run token budget ceiling for the consolidation sub-phase (soft
    /// partial-abort). Default `50000`.
    pub consolidation_budget_tokens: Option<f64>,
    /// Per-run USD-micro budget ceiling. INERT today (every op's per-call USD
    /// projection is 0 — the effective budget is token-based); reserved for a
    /// future real cost source. Default: no USD cap.
    pub consolidation_budget_usd_micro: Option<f64>,
}

/// Convert JS `DreamOptions` to substrate `DreamOpts` via FIELD MUTATION seeded
/// from `DreamOpts::default()` (compatible with the substrate's `#[non_exhaustive]`
/// posture — never a struct literal). Template = `js_dream_pass_opts_to_rust`.
///
/// `namespace` is NOT consumed here — it is resolved separately by `dream()`.
/// An unknown `cross_episode_mode` string fails LOUDLY rather than silently
/// defaulting.
pub fn js_dream_opts_to_rust(js: Option<JsDreamOpts>) -> napi::Result<kremory::DreamOpts> {
    let mut base = kremory::DreamOpts::default();
    let Some(o) = js else {
        return Ok(base);
    };
    if let Some(v) = o.include_community_detection {
        base.include_community_detection = v;
    }
    if let Some(v) = o.include_fact_archival {
        base.include_fact_archival = v;
    }
    if let Some(v) = o.include_supersession_sweep {
        base.include_supersession_sweep = v;
    }
    if let Some(v) = o.include_supersession_llm_nominate {
        base.include_supersession_llm_nominate = v;
    }
    if let Some(v) = o.archive_grace_days {
        base.archive_grace_days = Some(v as u32);
    }
    if let Some(v) = o.net_mutation_warn_floor {
        base.net_mutation_warn_floor = Some(v as usize);
    }
    if let Some(v) = o.consolidation_budget_tokens {
        base.consolidation_budget_tokens = Some(v as u64);
    }
    if let Some(v) = o.consolidation_budget_usd_micro {
        base.consolidation_budget_usd_micro = Some(v as u64);
    }
    // cross_episode as a single mode string, mapped onto the two coupled bools
    // (never exposed raw — prevents the illegal apply+dry_run combination). Parse
    // loudly: an unknown value is a caller error, surfaced not silently defaulted.
    if let Some(mode) = o.cross_episode_mode.as_deref() {
        match mode {
            "off" => base.include_cross_episode_merges = false,
            "shadow" => {
                base.include_cross_episode_merges = true;
                base.cross_episode_dry_run = true;
            }
            "apply" => {
                base.include_cross_episode_merges = true;
                base.cross_episode_dry_run = false;
            }
            other => {
                return Err(napi::Error::from_reason(format!(
                    "JsDreamOpts.cross_episode_mode: unknown value {other:?} \
                     (expected \"off\" | \"shadow\" | \"apply\")"
                )));
            }
        }
    }
    Ok(base)
}

/// Options for `JsMemory.runDreamPassSync`.
///
/// All fields optional — omit to use defaults from `DreamPassOpts::default()`.
#[napi(object, js_name = "DreamPassOptions")]
pub struct JsDreamPassOpts {
    /// When `true`, Pass 0 type-discovery runs to find novel entity types.
    /// Requires LLM. Default: `false`.
    pub include_type_discovery: Option<bool>,
    /// Minimum confidence threshold for entity type assignments to be re-examined
    /// during the reclassify pass. Range `[0.0, 1.0]`. Default: `0.5`.
    pub confidence_threshold: Option<f64>,
    /// Cap on the number of ghost episodes processed per run.
    /// `null`/omit = process all available ghost episodes.
    pub max_episodes_per_run: Option<f64>,
    /// Confidence threshold above which `ConsumerPinned` entities are protected
    /// from reclassification. Range `[0.0, 1.0]`. Default: `0.7`.
    pub reclassify_high_conf_threshold: Option<f64>,
}

/// Convert JS `DreamPassOptions` to substrate `DreamPassOpts`.
pub fn js_dream_pass_opts_to_rust(js: Option<JsDreamPassOpts>) -> kremory::DreamPassOpts {
    let base = kremory::DreamPassOpts::default();
    match js {
        None => base,
        Some(o) => kremory::DreamPassOpts {
            include_type_discovery: o
                .include_type_discovery
                .unwrap_or(base.include_type_discovery),
            confidence_threshold: o
                .confidence_threshold
                .map(|v| v as f32)
                .unwrap_or(base.confidence_threshold),
            max_episodes_per_run: o
                .max_episodes_per_run
                .map(|v| Some(v as usize))
                .unwrap_or(base.max_episodes_per_run),
            reclassify_high_conf_threshold: o
                .reclassify_high_conf_threshold
                .map(|v| v as f32)
                .unwrap_or(base.reclassify_high_conf_threshold),
        },
    }
}

/// Result of `JsMemory.awaitDream`.
///
/// Maps substrate `DreamStatus`. `status` is one of:
/// `"pending"` | `"processing"` | `"complete"` | `"failed"`.
#[napi(object, js_name = "DreamStatusResult")]
pub struct JsDreamStatusResult {
    /// Discriminator string. See struct rustdoc.
    pub status: String,
    /// Human-readable error detail. Present only when `status === "failed"`.
    pub error_message: Option<String>,
}

/// Convert a substrate `DreamSummary` to `JsDreamSummary`.
///
/// `usize`/`u64` → `f64` casts: safe up to 2^53 (~9 quadrillion).
/// At practical kremory scale these counters will never approach that limit.
pub fn dream_summary_to_js(s: DreamSummary) -> JsDreamSummary {
    JsDreamSummary {
        communities_updated: s.communities_updated as f64,
        cross_episode_would_merge: s.cross_episode_would_merge as f64,
        cross_episode_merged: s.cross_episode_merged as f64,
        supersessions_recorded: s.supersessions_recorded as f64,
        facts_archived: s.facts_archived as f64,
        consolidation_ops_ran: JsConsolidationOpsRan {
            community: s.consolidation_ops_ran.community,
            cross_episode: s.consolidation_ops_ran.cross_episode,
            archival: s.consolidation_ops_ran.archival,
            supersession_sweep: s.consolidation_ops_ran.supersession_sweep,
        },
        budget_exhausted: s.budget_exhausted,
        duration_ms: s.duration_ms as f64,
        types_discovered: s
            .types_discovered
            .into_iter()
            .map(|t| JsTypeProposal {
                name: t.name,
                description: t.description,
                justification: t.justification,
            })
            .collect(),
        entities_reclassified: s.entities_reclassified as f64,
        aliases_resolved: s.aliases_resolved as f64,
        canonicalization_merges: s.canonicalization_merges as f64,
        acronym_nickname_merges: s.acronym_nickname_merges as f64,
        type_registry_merges: s.type_registry_merges as f64,
        consistency_check_corrected: s.consistency_check_corrected as f64,
        warnings: s.warnings,
    }
}

/// Convert a substrate `kremory::DreamStatus` to `JsDreamStatusResult`.
pub fn dream_status_to_js(s: kremory::DreamStatus) -> JsDreamStatusResult {
    match s {
        kremory::DreamStatus::Pending => JsDreamStatusResult {
            status: "pending".to_string(),
            error_message: None,
        },
        kremory::DreamStatus::Processing => JsDreamStatusResult {
            status: "processing".to_string(),
            error_message: None,
        },
        kremory::DreamStatus::Complete => JsDreamStatusResult {
            status: "complete".to_string(),
            error_message: None,
        },
        kremory::DreamStatus::Failed(msg) => JsDreamStatusResult {
            status: "failed".to_string(),
            error_message: Some(msg),
        },
        // Non-exhaustive guard: forward-compat for variants added after v0.1.8.
        _ => JsDreamStatusResult {
            status: "pending".to_string(),
            error_message: None,
        },
    }
}

