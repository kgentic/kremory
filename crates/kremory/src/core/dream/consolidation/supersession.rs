//! CONSOLIDATION op — supersession sweep (ADR-066 §2.3, spec P1). **STUB (P0).**
//!
//! Two lanes (filled in P1):
//! - deterministic world-time window close-out (`valid_to < now AND expired_at IS
//!   NULL`) — zero-LLM, pure date-compare, orthogonal to ingest's same-object dedup;
//! - optional LLM-nominated value-change (default-off, emit-guarded against time
//!   inversion).
//!
//! P0 scope: signature + a no-op returning `OpReport::default()` so the dispatcher
//! + wiring compile and run inert by default. The real sweep lands in P1.

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

use super::substrate::{ConsolidationBudget, OpReport};

/// Bundled params for [`supersession`] — args-as-object per TD-042
/// (`too_many_arguments` threshold 3). `graph` is the receiver-like lead dep.
pub(crate) struct SupersessionParams<'a> {
    pub(crate) graph: &'a TemporalGraph,
    pub(crate) group_id: &'a str,
    /// Shared soft budget (the optional LLM lane advances it via `record`).
    pub(crate) budget: &'a mut ConsolidationBudget,
    /// Select the optional LLM-nominated value-change lane (P1.3). Default-off.
    pub(crate) include_llm_nominate: bool,
    /// Resolved dream model id, threaded for the LLM lane (TD-094 style).
    pub(crate) model_id: &'a str,
}

/// Run the supersession sweep over `group_id`. **STUB — returns count 0 (P0).**
#[allow(clippy::unused_async)] // async signature is the P1 contract; body is a stub.
pub(crate) async fn supersession(params: SupersessionParams<'_>) -> Result<OpReport> {
    // Bind the params so the P1 impl has the fields in scope; the budget-mut +
    // llm-nominate + model_id are the P1 LLM-lane inputs. Discarded at P0 (stub).
    let SupersessionParams {
        graph: _graph,
        group_id: _group_id,
        budget: _budget,
        include_llm_nominate: _include_llm_nominate,
        model_id: _model_id,
    } = params;
    // P1 will implement: window-closeout SELECT + `invalidate_fact`, optional
    // LLM-nominated value-change lane with the time-inversion emit guard.
    Ok(OpReport::default())
}
