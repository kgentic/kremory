#![allow(clippy::unwrap_used, clippy::expect_used)]
//! B.1 — Static pass table RED gate.
//!
//! Verifies that `DreamMode` enum exists in `kremory::memory::types` and
//! that the static pass table in `kremory::memory::dream_phase` respects
//! mode-gating — specifically that `DreamMode::Light` skips community,
//! distillation, and supersession passes and only runs archive.
//!
//! Shape test for the CONSOLIDATION (sub-phase 2) pass-table scaffolding —
//! `DreamMode` + `PassDef` + `PASSES` in `kremory::memory::dream_phase`. These
//! describe the UNBUILT consolidation passes (community/distillation/
//! supersession/archive) and are DORMANT: no production code reads them, and
//! there is intentionally NO `DreamOpts::mode` field — reconciliation uses the
//! per-pass `DreamOpts` knobs (`include_type_discovery`, `include_consistency_check`),
//! not `DreamMode`. See `.ai-docs/architecture/dream-phase-two-sub-phases-2026-07-01.md`.

use kremory::memory::{
    dream_phase::{PassDef, PASSES},
    types::DreamMode,
};

/// Light mode skips community, distillation, and supersession — only archive runs.
/// Per architecture spec §2.4.B and runbook §5.8.
#[test]
fn light_mode_skips_non_light_passes() {
    let light_passes: Vec<&PassDef> = PASSES
        .iter()
        .filter(|p| p.runs_in(DreamMode::Light))
        .collect();
    let pass_names: Vec<&str> = light_passes.iter().map(|p| p.name).collect();

    // Archive MUST run in Light mode
    assert!(
        pass_names.contains(&"archive"),
        "archive pass must run in Light mode; got: {:?}",
        pass_names
    );

    // Community, distillation, supersession MUST NOT run in Light mode
    for skipped in &["community", "distillation", "supersession"] {
        assert!(
            !pass_names.contains(skipped),
            "{} must be skipped in Light mode; got: {:?}",
            skipped,
            pass_names
        );
    }
}

/// Full mode runs ALL passes.
#[test]
fn full_mode_runs_all_passes() {
    let full_passes: Vec<&PassDef> = PASSES
        .iter()
        .filter(|p| p.runs_in(DreamMode::Full))
        .collect();
    let pass_names: Vec<&str> = full_passes.iter().map(|p| p.name).collect();

    for required in &["community", "distillation", "supersession", "archive"] {
        assert!(
            pass_names.contains(required),
            "{} must run in Full mode; got: {:?}",
            required,
            pass_names
        );
    }
}

/// PASSES is non-empty — static table must have at least 4 entries.
#[test]
fn static_pass_table_has_minimum_four_passes() {
    assert!(
        PASSES.len() >= 4,
        "PASSES must have at least 4 entries (community, distillation, supersession, archive); got {}",
        PASSES.len()
    );
}
