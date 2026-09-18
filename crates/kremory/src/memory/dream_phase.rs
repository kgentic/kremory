//! Dream phase — CONSOLIDATION (sub-phase 2) pass-table scaffolding. DORMANT.
//!
//! ## What this is — and what it is NOT
//!
//! This module describes the **consolidation** sub-phase — graph-global passes
//! (community detection, distillation, supersession, archival).
//!
//! ⚠️ **CORRECTED 2026-09-18 (TD-086). The previous claim here — "Consolidation is
//! NOT yet implemented ... the `DreamSummary` consolidation fields are honest
//! zeros" — was FALSE, and had been for some time.** The truth is split by path,
//! which is exactly what a blanket statement hid:
//!
//! - **Facade path: IMPLEMENTED and shipping.** `facade/dream.rs:753` calls
//!   `consolidation::run_consolidation` for real, gated by
//!   `opts.any_consolidation_enabled()`. `communities.rs`, `supersession.rs` and
//!   `archive.rs` all exist under `core/dream/consolidation/`. A
//!   `ConsolidationSummary::default()` appears there only as the fallback when
//!   consolidation is disabled or there is no temporal graph — NOT as a stub.
//! - **Engine-handle path: still `NotImplemented`.** `graph_run_consolidation`
//!   returns `Err(MemoryError::NotImplemented)` on that surface
//!   (`memory/engine_handle.rs:14`). This is the only sense in which the old
//!   claim was ever true, and it was never the whole picture.
//! - **`distillation` is the one genuinely unbuilt pass** (TD-121) — no
//!   `distillation.rs` exists.
//!
//! `DreamMode` + `PASSES` here remain **dormant scaffolding regardless**: nothing
//! in production reads them (only the shape test `b1_static_pass_table.rs`), and
//! the three passes that DID ship route around this table entirely. Re-verified
//! 2026-09-18.
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
/// mode-gating per architecture spec §2.4.B.
///
/// ⚠️ **ASPIRATIONAL AND UNWIRED. This table is NOT the dispatch source of truth,
/// and the passes it names are not dispatched from here** (TD-086, corrected
/// 2026-09-18).
///
/// The original note here said consolidation was "to be wired when built". That
/// became false without anyone updating it: **3 of these 4 names are now built and
/// shipping** — `communities.rs`, `supersession.rs` and `archive.rs` all exist
/// under `core/dream/consolidation/` and run for real via
/// `consolidation::run_consolidation`, dispatched from `facade/dream.rs:753`.
/// They have simply never travelled through this table. Only `distillation`
/// remains genuinely unbuilt (see TD-121) — there is no `distillation.rs`.
///
/// So the drift is the reverse of what it looks like: this is not scaffolding
/// waiting on an unbuilt feature, it is a stale table sitting beside a feature
/// that shipped past it.
///
/// **Footgun if you wire it.** `PassDef::runs_in` reads as a ready-made
/// orchestrator. Iterating `PASSES` today would dispatch a pass vocabulary that
/// does not match the shipped one and would attempt `distillation`, which does not
/// exist. Before making this the dispatch SoT, reconcile the names against
/// `core/dream/mod.rs` — that is TD-086 resolution path 1, a deliberate design
/// change, not a cleanup.
///
/// Verified dormant 2026-09-18: nothing in `crates/kremory/src` outside this file
/// references `PASSES` (only the shape test `b1_static_pass_table.rs`).
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
