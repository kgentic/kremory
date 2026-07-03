//! Dream phase — batch background passes (ADR-037 / ADR-046 / ADR-047).
//!
//! Pass 0: type discovery via LLM proposal + anti-redundancy gate + persistence.
//! Pass 2: reclassify — two-arm SELECT (catch_all_cascade + low_confidence) with
//!         confidence-aware source-tier stamping (ADR-046 Option E).
//! Pass 4: consistency_check — hybrid embed-prefilter + LLM-verify for
//!         high-confidence wrong-type detection (ADR-047).

/// Site #5 (ADR-063 spec §3) — instance acronym/nickname recall. Gated behind
/// `DreamOpts::include_acronym_nickname_recall`, default `true` (VALIDATED
/// 2026-07-03: site5_metrics.json precision 1.00, Wilson-lower 0.955, 0 false merges).
pub(crate) mod acronym_nickname_recall;
pub(crate) mod anti_redundancy;
pub mod consistency_check;
/// Dream CONSOLIDATION sub-phase (ADR-066 axes D+E) — graph-global cleanup ops
/// (supersession / archive / cross_episode / communities) over a shared budget +
/// idempotency substrate. Runs after the reconciliation chain in `facade/dream.rs`.
/// Ops are STUBS at P0; `run_consolidation` dispatches them opt-in (default off).
pub(crate) mod consolidation;
pub(crate) mod discover_types;
/// Dream-pass idempotency key — canonical entity view + content hash (v0.2.4
/// Phase 3 / ADR-050 Guard #1). Compile-spike landed ahead of the build sprint
/// per the impl-spec §11 readiness-gate contingency.
pub(crate) mod idempotency;
/// Shared statistical helpers (Wilson interval, etc.) for dream-pass
/// precision/recall spikes and the metrics harness (spec
/// `dream-adversarial-corpora-and-metrics-2026-07-02.md` §3 step 0, H1).
pub(crate) mod metrics_util;
pub(crate) mod proposed_type;
pub mod reclassify;
/// Site #3 (ADR-063 spec §4) — type-registry post-hoc collapse. Gated behind
/// `DreamOpts::include_type_registry_collapse`, default `true` (VALIDATED
/// 2026-07-03: site3_metrics.json precision 0.949, Wilson-lower 0.861, 0 false merges).
pub(crate) mod type_registry_collapse;

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

// Same MNT-002 pattern, for the ADR-063 Site #3 S3 spike integration test
// (`tests/type_registry_collapse_s3_spike.rs`). `type_registry_collapse` and
// `TypeRegistryCollapseParams` are `pub` + `#[doc(hidden)]` inside
// `type_registry_collapse.rs` (not part of the stable public API contract)
// for the identical E0365 reason documented above.
#[cfg(any(test, feature = "test-utils"))]
pub use type_registry_collapse::{
    type_registry_collapse, TypeRegistryCollapseParams, TypeRegistryCollapseReport,
};

// Same MNT-002 pattern, for the ADR-063 Site #5 S2 spike integration test
// (`tests/acronym_nickname_recall_s2_spike.rs`). `acronym_nickname_recall` and
// `AcronymNicknameRecallParams` are `pub` + `#[doc(hidden)]` inside
// `acronym_nickname_recall.rs` (not part of the stable public API contract)
// for the identical E0365 reason documented above.
#[cfg(any(test, feature = "test-utils"))]
pub use acronym_nickname_recall::{acronym_nickname_recall, AcronymNicknameRecallParams};

// Wilson-interval helper (spec `dream-adversarial-corpora-and-metrics-
// 2026-07-02.md` §3 step 0, H1) — extracted from the S1 spike's inline block
// so the metrics harness can reuse it. Gated `#[cfg(any(test, feature =
// "test-utils"))]` in lockstep with `metrics_util::wilson_lower_upper`'s own
// gate. `pub use` (NOT `pub(crate) use`): `tests/dream_metrics_harness.rs`
// lives outside the crate boundary (an external integration-test binary), so
// a `pub(crate)` re-export would be unreachable from it — same E0365-adjacent
// visibility requirement as the MNT-002 re-exports above, which are `pub`
// for the identical reason.
#[cfg(any(test, feature = "test-utils"))]
pub use metrics_util::wilson_lower_upper;

// Same MNT-002 pattern, for the Site #2 (ADR-063 spec §4.3 sibling / §2.2)
// proposal-time type-novelty gate metrics harness
// (`tests/dream_metrics_harness_site2.rs`). `check_proposal`,
// `CheckProposalParams`, `GateOutcome`, `names_share_lemma_or_exact`,
// `DESC_COSINE_THRESHOLD`, and `TYPE_NOVELTY_LOWER_BAND` are `pub` +
// `#[doc(hidden)]` inside `anti_redundancy.rs`; `adjudicate_type_novelty` and
// `AdjudicateTypeNoveltyParams` are `pub` + `#[doc(hidden)]` inside
// `discover_types.rs` — same E0365 visibility requirement as every other
// re-export in this block (an external integration-test binary cannot import
// a `pub(crate)` item, so these are promoted to `pub` + hidden from rustdoc,
// not part of the stable public API contract).
#[cfg(any(test, feature = "test-utils"))]
pub use anti_redundancy::{
    check_proposal, names_share_lemma_or_exact, CheckProposalParams, GateOutcome,
    DESC_COSINE_THRESHOLD, TYPE_NOVELTY_LOWER_BAND,
};
#[cfg(any(test, feature = "test-utils"))]
pub use discover_types::{
    adjudicate_type_novelty, type_novelty_is_redundant, AdjudicateTypeNoveltyParams,
};
