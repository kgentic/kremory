//! Dream phase — batch background passes (ADR-037 / ADR-046).
//!
//! Pass 0: type discovery via LLM proposal + anti-redundancy gate + persistence.
//! Pass 2: reclassify — two-arm SELECT (catch_all_cascade + low_confidence) with
//!         confidence-aware source-tier stamping (ADR-046 Option E).

pub(crate) mod anti_redundancy;
pub(crate) mod discover_types;
pub(crate) mod proposed_type;
pub mod reclassify;

pub use discover_types::{DiscoveryResult, TypeProposal};
pub use reclassify::ReclassifyResult;
