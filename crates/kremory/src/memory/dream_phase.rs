//! Dream phase — CONSOLIDATION (sub-phase 2) pass-table scaffolding. DORMANT.
//!
//! ## What this is — and what it is NOT
//!
//! This module describes the **consolidation** sub-phase — graph-global passes
//! (community detection, distillation, supersession, archival). **Consolidation
//! is NOT yet implemented** (`graph_run_consolidation` is `NotImplemented` — F-01;
//! the `DreamSummary` consolidation fields are honest zeros). `DreamMode` +
//! `PASSES` here are the *future* mode-gating for those unbuilt passes — nothing
//! in production reads them today (only a shape test, `b1_static_pass_table.rs`).
//!
//! This does **NOT** gate the live **reconciliation** sub-phase (the 5 passes:
//! type_discovery / aliases / reclassify / consistency_check / canonicalize).
//! Reconciliation cost-control is the per-pass `DreamOpts` knobs
//! (`include_type_discovery`, `include_consistency_check`) — NOT `DreamMode`.
//! When consolidation is scoped, its gating is (re)designed with the real
//! passes; do not repurpose this table for reconciliation.
//!
//! ## Static pass table (consolidation, per architecture spec §2.4.B)
//!
//! `PASSES` is a compile-time static array of `PassDef` descriptors — the
//! intended ordering + mode-gating for the consolidation passes.
//!
//! | Mode | Consolidation passes that run |
//! |------|----------------|
//! | `Full` | community + distillation + supersession + archive |
//! | `Light` | archive only (skip community/distillation/supersession) |

/// Which CONSOLIDATION passes to run in a dream phase (sub-phase 2, DORMANT).
///
/// Per runbook §5.8 and architecture spec §2.4.B. This is the intended
/// mode-gating for the UNBUILT consolidation passes; it does not gate the live
/// reconciliation passes (those use the per-pass `DreamOpts` knobs). There is no
/// `DreamOpts::mode` field today — consolidation is not yet wired (F-01).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DreamMode {
    /// Run all passes: community, distillation, supersession, archive.
    #[default]
    Full,
    /// Fast consolidation snapshot: archive only.
    Light,
}

/// Descriptor for one pass in the dream phase static table.
///
/// Callers iterate `PASSES`, call `runs_in(mode)` to filter, then dispatch
/// the pass logic. The `name` field is used for metrics labels (ADR D7) and
/// tracing span names (ADR D12: `kremory.dream_phase.<name>`).
pub struct PassDef {
    /// Canonical name — used as metrics label and span name segment.
    /// Bounded `&'static str` (cardinality safe per ADR D7).
    pub name: &'static str,
    /// Which modes include this pass.
    modes: &'static [DreamMode],
    /// If true, a pass failure aborts the remaining passes in this run.
    /// If false, the pass failure is recorded and the orchestrator continues.
    pub fail_fast: bool,
}

impl PassDef {
    /// Returns `true` if this pass runs in the given mode.
    pub fn runs_in(&self, mode: DreamMode) -> bool {
        self.modes.contains(&mode)
    }
}

/// Static pass table for CONSOLIDATION (sub-phase 2) — canonical ordering +
/// mode-gating per architecture spec §2.4.B, to be wired when consolidation is
/// built. DORMANT scaffolding: not iterated by any production code today (only
/// the shape test `b1_static_pass_table.rs`).
///
/// Order matters: community must complete before distillation (graph topology
/// must stabilise before cross-episode synthesis). Supersession precedes archive
/// (facts to-be-superseded must be identified before archival).
pub static PASSES: &[PassDef] = &[
    PassDef {
        name: "community",
        modes: &[DreamMode::Full],
        fail_fast: false,
    },
    PassDef {
        name: "distillation",
        modes: &[DreamMode::Full],
        fail_fast: false,
    },
    PassDef {
        name: "supersession",
        modes: &[DreamMode::Full],
        fail_fast: false,
    },
    PassDef {
        name: "archive",
        modes: &[DreamMode::Full, DreamMode::Light],
        fail_fast: false,
    },
];
