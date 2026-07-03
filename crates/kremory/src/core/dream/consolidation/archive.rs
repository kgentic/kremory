//! CONSOLIDATION op — fact archival (ADR-066 §2.4, spec P2). **STUB (P0).**
//!
//! Filled in P2: MOVE a long-expired (`expired_at < now - grace`), unreferenced
//! fact from live `facts` into the append-only `facts_archive` table (migration
//! 019) — INSERT + `DELETE FROM facts_fts` (shadow-row cleanup, RISK-003) + DELETE
//! in one `BEGIN IMMEDIATE`, gated by a ref-count "orphans nothing" guard. Zero-LLM.
//!
//! P0 scope: signature + a no-op returning `OpReport::default()`.

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::OpReport;

/// Run the fact-archival op over `group_id`. **STUB — returns count 0 (P0).**
///
/// `grace_days` is the archival grace window (`DreamOpts.archive_grace_days`, P2.1).
#[allow(clippy::unused_async)] // async signature is the P2 contract; body is a stub.
pub(crate) async fn archive(
    _graph: &TemporalGraph,
    _group_id: &str,
    _grace_days: u32,
) -> Result<OpReport> {
    // P2 will implement: candidate SELECT (expired past grace) + ref-count guard +
    // atomic INSERT-into-facts_archive / delete-fts-shadow / delete-live move.
    Ok(OpReport::default())
}
