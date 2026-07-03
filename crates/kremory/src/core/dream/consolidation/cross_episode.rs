//! CONSOLIDATION op — cross-episode entity merge (ADR-066 §2.2, spec P3).
//! **STUB (P0).**
//!
//! Filled in P3: admit exact/fuzzy-label candidate pairs across DISTINCT episodes,
//! apply the MANDATORY structural-corroboration gate (shared neighbour OR identical
//! `(predicate, object)` — homonymy guard, RISK-001), then delegate the structural
//! merge to the shared `canonicalization::apply_entity_merge` executor (P0.3) — NO
//! second merge code path. Zero-LLM, zero-embedding.
//!
//! P0 scope: signature + a no-op returning `OpReport::default()`.

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::OpReport;

/// Run the cross-episode merge op over `group_id`. **STUB — returns count 0 (P0).**
#[allow(clippy::unused_async)] // async signature is the P3 contract; body is a stub.
pub(crate) async fn cross_episode(_graph: &TemporalGraph, _group_id: &str) -> Result<OpReport> {
    // P3 will implement: exact + fuzzy (Jaccard ≥0.9) candidate admission,
    // structural-corroboration gate (RISK-001), delegate to apply_entity_merge.
    Ok(OpReport::default())
}
