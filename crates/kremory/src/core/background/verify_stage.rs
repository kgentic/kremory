//! Stage 2 verify hook — no-op stub until ADR-049 Stage 2 wiring.
//!
//! Sprint plan T2.1 / ADR-049 §Decision 6 §5.1 §5.4.
//!
//! This module exists as a structural prerequisite for Phase B wiring.
//! The public stub function signature is load-bearing: downstream callers
//! in `deferred_pipeline` will call `run_verify_stage` once Stage 2 is
//! wired in a subsequent sprint phase.
//!
//! ## ADR-049 §5.4 spec
//!
//! > `verify_stage.rs` — Stage 2 hook (no-op until Stage 2 wires, but the
//! > module + public stub function MUST exist now per ADR-049 spec §5.1).
//!
//! ## What lives here in Phase B
//!
//! - Call into `consistency_check::verify_batch_for_candidates` (promoted
//!   to `pub(crate)` as ADR-049 §Decision 6 prerequisite T2.2).
//! - Demote-on-failure contract: missing decisions → `entity_type_id=0`,
//!   `entity_type_source='Phase1Ner'` per ADR-049 §Decision 4.
//! - `Memory::with_background_verify(bool)` builder knob wiring.

use crate::core::error::Result;

/// Run the Stage 2 verify gate for a batch of Phase 1 NER candidates.
///
/// **Current status**: no-op stub — returns `Ok(())` immediately.
/// Full implementation wired in ADR-049 Phase B Stage 2.
///
/// # Future signature note
///
/// When Phase B wires this function, it will accept:
/// - candidate entity rowids produced by Phase 1
/// - source-episode content for the verify prompt
/// - a frontier `ChatProvider` reference
/// - a `libsql::Connection` for the demote-on-failure write path
///
/// The signature will expand; callers should pass through all fields from
/// the completed `DeferredRequest` + `IngestResult`.
// Quinn MED-02 fix: `pub(super)` not `pub`. The no-op stub returning Ok(())
// would be misleadingly reachable in the v0.2.0 public API surface as
// `kremory::core::background::run_verify_stage`. Phase B will call it from
// the sibling `deferred_pipeline.rs` via `super::verify_stage::run_verify_stage`;
// no other caller exists or should exist until Phase B wires Stage 2.
//
// `#[allow(dead_code)]` is justified by the ADR-049 §Decision 6 spec mandate
// that this stub MUST EXIST as Phase B's wiring target. Per CLAUDE.md Rule 8
// (treat cause not symptom): the spec is the cause; the dead-code lint is the
// symptom. Removing the function would violate the spec. Marking it dead-code
// is acknowledging the spec-mandated structural prerequisite, not a band-aid.
#[allow(dead_code)]
pub(super) async fn run_verify_stage() -> Result<()> {
    // No-op: Stage 2 wiring is Phase B of ADR-049.
    // This stub satisfies the structural prerequisite from §Decision 6
    // so the module directory compiles cleanly before Phase B ships.
    Ok(())
}
