//! Pass 0 type discovery primitive.
//!
//! ## What this does
//!
//! 1. **Signal source** — loads all `entity_type_id = 0` entities for `group_id`.
//! 2. **Top-K clusters** — groups catch-alls by name, ranks by frequency, takes
//!    top `max_proposals` clusters as proposal candidates (one cluster → one LLM prompt entry).
//!    (Embedding-based clustering is deferred to v0.2.0; name-frequency grouping is the
//!    v0.1.1 minimal-first implementation.)
//! 3. **LLM proposal call** — single `StructuredCallBuilder` call asking the LLM to
//!    propose entity types for the supplied clusters.  `max_proposals` is enforced
//!    PROMPT-SIDE (cap in the system prompt), not post-emission filter.
//! 4. **Shape validator** — each proposal runs through `validate_proposed_name`
//!    (9 rejection categories).
//! 5. **Anti-redundancy gate** — if an embedder is available, each proposal's
//!    description and name are embedded and compared against existing types at
//!    0.85 / 0.70 cosine thresholds.  Without an embedder the gate is skipped
//!    and a warning is recorded (D7 degraded mode).
//! 6. **Persistence** — accepted proposals are inserted into `entity_types` via
//!    `INSERT OR IGNORE` with 4 provenance columns (Migration 014).
//! 7. **In-place evidence retype** — evidence entities whose name is semantically
//!    close to the new type's description (cosine ≥ 0.75) are updated to
//!    `entity_type_id = new_id, entity_type_source = 'DreamPass0'`.
//!    Without an embedder: retype ALL evidence entities for the accepted type (D4).
//!
//!    **Evidence-retype guard (default OFF):** this cosine-only comparison is a BARE
//!    ENTITY NAME (`entities.id`) embedded against a TYPE DESCRIPTION — the same
//!    degenerate-embedding failure class documented elsewhere (short bare
//!    labels collapse to near-identical vectors under `nomic-embed-text`), just
//!    cross-domain instead of name-vs-name. Unlike the other embedding-based
//!    identity checks in this codebase, this
//!    comparison was never spike-validated ("every kremory-specific
//!    numeric threshold MUST PASS a spike... BEFORE it is wired into the
//!    production path") and has no deterministic corroboration signal — a
//!    lexical gate on entity-name-vs-type-name would reject legitimate matches
//!    (an instance name like "Nobu Malibu" shares no lemma with its type
//!    "Restaurant"), so the lexical-gate pattern used elsewhere does not
//!    transplant here unmodified. Gated behind `DreamOpts::include_evidence_retype_by_
//!    similarity` (default `false`) pending a proper spike (mirrors the
//!    quarantine-until-spiked posture used for every other new
//!    mechanism). When OFF, evidence entities are left as catch-all
//!    (`entity_type_id = 0`) — Pass 2 `reclassify` (LLM + confidence-gated, not
//!    cosine-alone) runs immediately after Pass 0 in the same `mem.dream()` call
//!    and safely picks up promotion instead, so disabling this
//!    path does not lose retype coverage, only the risky cosine-alone shortcut.
//!
//! ## Observability
//!
//! All 6 required metrics are emitted:
//! - `kremory.dream.types_proposed_total{model, namespace}`
//! - `kremory.dream.types_accepted_total{model, namespace}`
//! - `kremory.dream.types_rejected_total{reason, model, namespace}`
//! - `kremory.dream.entities_retyped_total{source, namespace}`
//! - `kremory.dream.proposal_call_duration_ms{model}` (histogram)
//! - `kremory.dream.anti_redundancy_gate_skipped_total{reason="no_embedder"}`

mod adjudicate;
mod cluster;
mod discover;
mod helpers;
mod types;

#[cfg(test)]
use chrono::Utc;

#[cfg(test)]
use crate::core::{
    error::Result,
    identity_verdict::{IdentityVerdictItem, LLM_VERIFY_CONFIDENCE_FLOOR},
    provider::{ChatProvider, DynEmbeddingProvider},
};

// Gated to MATCH the parent re-export in `core/dream/mod.rs`, which carries
// the same `#[cfg(any(test, feature = "test-utils"))]`. Without this gate the
// re-export has no consumer in a non-test-utils build -- the parent is compiled
// out -- and this crate treats unused imports as a hard error, so `cargo check
// -p kremory-eval --bin spike_c6_async_gate` fails to compile the lib. A
// re-export must be conditional on exactly the condition its consumer is.
#[cfg(any(test, feature = "test-utils"))]
pub use adjudicate::{adjudicate_type_novelty, type_novelty_is_redundant, AdjudicateTypeNoveltyParams};
pub(crate) use discover::{discover_types, DiscoverTypesParams};
#[cfg(test)]
use helpers::{accept_proposal, AcceptProposalParams};
pub use types::{DiscoveryResult, TypeProposal};
pub(crate) use types::MAX_PROPOSALS;

// ─── type_novelty_is_redundant unit tests ────────────────────────────────────
#[cfg(test)]
#[path = "adr065_type_novelty_redundant_tests.rs"]
mod adr065_type_novelty_redundant_tests;

// ─── Regression tests: composite-PK INSERT id omission ───────────────────────
#[cfg(test)]
#[path = "td051_tests.rs"]
mod td051_tests;

#[cfg(test)]
#[path = "td050_full_workflow_tests.rs"]
mod td050_full_workflow_tests;

#[cfg(test)]
#[path = "site2_type_novelty_tests.rs"]
mod site2_type_novelty_tests;

#[cfg(test)]
#[path = "td123_evidence_retype_guard_tests.rs"]
mod td123_evidence_retype_guard_tests;

#[cfg(all(test, feature = "llm-integration"))]
#[path = "td050_real_llm_tests.rs"]
mod td050_real_llm_tests;
