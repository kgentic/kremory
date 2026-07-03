//! CONSOLIDATION op — community detection (ADR-066 §2.1, spec P4). **STUB (P0).**
//!
//! Filled in P4: build a `petgraph::UnGraph` of the namespace's entity
//! co-occurrence (shared episodes), run deterministic synchronous label
//! propagation (compile-spike PASS, `spike/community_detect.rs`), persist
//! `entity_communities` + `community_summaries` (migration 019), and count
//! communities whose sorted-member hash CHANGED (`communities_updated`). Zero-LLM.
//! P4's go/no-go rides the DoD-P4.6 modularity spike on the hairball fixture.
//!
//! P0 scope: signature + a no-op returning `OpReport::default()`.

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::OpReport;

/// Run community detection over `group_id`. **STUB — returns count 0 (P0).**
#[allow(clippy::unused_async)] // async signature is the P4 contract; body is a stub.
pub(crate) async fn communities(_graph: &TemporalGraph, _group_id: &str) -> Result<OpReport> {
    // P4 will implement: UnGraph build + deterministic label propagation +
    // entity_communities/community_summaries persistence + member-hash-change count.
    Ok(OpReport::default())
}
