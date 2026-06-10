//! Dream phase — batch background passes (ADR-037 / ADR-046 / ADR-047).
//!
//! Pass 0: type discovery via LLM proposal + anti-redundancy gate + persistence.
//! Pass 2: reclassify — two-arm SELECT (catch_all_cascade + low_confidence) with
//!         confidence-aware source-tier stamping (ADR-046 Option E).
//! Pass 4: consistency_check — hybrid embed-prefilter + LLM-verify for
//!         high-confidence wrong-type detection (ADR-047).

pub(crate) mod anti_redundancy;
pub mod consistency_check;
pub(crate) mod discover_types;
pub(crate) mod proposed_type;
pub mod reclassify;

pub use discover_types::{DiscoveryResult, TypeProposal};
pub use reclassify::ReclassifyResult;

// Test-utils re-exports for GAP-002 DoD compile checks.
// Items are `pub` + `#[doc(hidden)]` in consistency_check.rs; re-exported here
// so `tests/spike_c6_uses_pub_crate_primitives.rs` can import them under
// `kremory::core::dream::consistency_check::*` with `feature = "test-utils"`.
//
// ## pub + #[doc(hidden)] semver contract (MNT-002)
//
// These items are intentionally `pub` rather than `pub(crate)` due to an E0365
// constraint: `pub(crate)` items cannot be re-exported as `pub` in this re-export
// block (integration test crates live outside the kremory crate boundary).
// `#[doc(hidden)]` hides them from rustdoc but NOT from autocomplete or downstream
// crates that import with `feature = "test-utils"`.
//
// Consumers of `feature = "test-utils"` MUST treat these as explicitly unstable:
// they will change without semver notice. The `test-utils` feature is not
// part of the public API contract.
#[cfg(any(test, feature = "test-utils"))]
pub use consistency_check::{
    build_verify_messages,
    verify_batch,
    verify_batch_schema,
    CandidateRow,
    // ARCH-001 fix types — also exposed for Phase B verify_stage.rs integration tests.
    VerifyAction,
    VerifyBatchDecision,
    VerifyBatchOutcome,
    VerifyBatchParams,
};
