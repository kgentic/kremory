//! Dream phase (batch consolidation) pass orchestration.
//!
//! ## Static pass table (ADR §2.4.B)
//!
//! `PASSES` is a compile-time static array of `PassDef` descriptors. Each pass
//! declares its name, which `DreamMode`s it runs in, and a `fail_fast` flag.
//!
//! The orchestrator iterates `PASSES`, filters by the requested `DreamMode`,
//! and dispatches each pass. No dynamic registration — the table is the
//! single source of truth for pass ordering and mode-gating.
//!
//! ## Mode semantics
//!
//! | Mode | Passes that run |
//! |------|----------------|
//! | `Full` | community + distillation + supersession + archive |
//! | `Light` | archive only (skip community/distillation/supersession) |
//!
//! `Light` is used when the caller wants a quick consolidation snapshot without
//! the CPU-intensive community-recompute or distillation passes.

/// Which passes to run in a dream phase.
///
/// Per runbook §5.8 and architecture spec §2.4.B.
/// `DreamOpts::mode` (default = `Full`) selects the pass subset.
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

/// Static pass table — canonical ordering and mode-gating per architecture spec §2.4.B.
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
