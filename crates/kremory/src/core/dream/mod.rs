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
