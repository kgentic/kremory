//! Dream phase — batch background passes (ADR-037).
//!
//! Pass 0: type discovery via LLM proposal + anti-redundancy gate + persistence.

pub(crate) mod anti_redundancy;
pub(crate) mod discover_types;
pub(crate) mod proposed_type;

pub use discover_types::{DiscoveryResult, TypeProposal};
