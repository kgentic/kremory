//! `IngestEventSink` — Phase 2 per-episode event sink trait for kremory::core.
//!
//! Per ADR D.6.5 + ADR rqlm-async-event-handle-api-design-2026-05-19 §4.8:
//! rqlc owns this trait because rqlc owns the Phase 2 work. rqlc consumers
//! who skip rqlm can subscribe to ingest events directly.
//!
//! rqlm extends this with `EnrichmentEventSink: IngestEventSink` at
//! `kremory::memory::events`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::error::{ContradictionResolution, IngestStatus, IngestionErrorKind};

// ─── SQL → IngestStatus bridge ───────────────────────────────────────────────

/// Convert an `episode_processing_status` SQL TEXT value to [`IngestStatus`].
///
/// ## One-way bridge declaration
///
/// This function is the **ONE-WAY bridge** from the 4-state SQL
/// `episode_processing_status` column to the 7-state `IngestStatus` enum.
/// There is **NO inverse function**. Do not add one.
///
/// The reverse direction (`IngestStatus` → SQL string) is intentionally absent
/// because `Complete`, `Deduplicating`, and `Invalidating` have no SQL column
/// representation — the status column is intentionally simpler than the push
/// enum (polling consumers need coarser state; push consumers need finer).
///
/// ## Mapping table
///
/// | SQL value       | `IngestStatus` variant                                   |
/// |-----------------|----------------------------------------------------------|
/// | `'Pending'`     | `IngestStatus::Pending`                                  |
/// | `'Extracting'`  | `IngestStatus::Extracting`                               |
/// | `'Verified'`    | `IngestStatus::EntitiesReady`  ← **NOT `Complete`**      |
/// | `'Failed'`      | `IngestStatus::Failed("from_sql_status: SQL-level failure; reason not captured")` |
/// | any other value | `IngestStatus::Failed(format!("unknown_sql_status:{s}"))` |
///
/// ### Why `'Verified'` → `EntitiesReady` (not `Complete`)
///
/// The SQL column writes `'Verified'` at Phase 2a completion (entity write done,
/// inside `run_verify_stage`). `IngestStatus::Complete` fires later, after
/// `ingest_deferred` completes fact extraction — there is no second SQL write at
/// that point. `EntitiesReady` is the moment `wait_for_processing` resolves;
/// `Complete` fires later with no SQL equivalent. (ADR-052 Gap 5 Decision D5.)
///
/// ### Why unknown values fall back to `Failed` (not `unreachable!()`)
///
/// Migration 015a explicitly omits a `CHECK` constraint on
/// `episode_processing_status` (see migration doc-comment: "SQLite CHECK is
/// per-row but not enforced retroactively"). We cannot assume the SQL layer
/// enforces the 4-value set, so an unknown SQL value is treated as a best-effort
/// failure sentinel rather than an unrecoverable panic.
///
/// Placement: `crates/kremory/src/core/sink.rs` (MED-03 — sibling to
/// `IngestStatus` import and the event-sink trait definitions).
///
/// ## Visibility — Phase 1 impl-time deviation from arch spec §4.2
///
/// `pub` (not `pub(crate)` as arch spec §4.2 drafted). Reasoning:
///
/// 1. **Phase 6 integration test access** — impl spec §9.2 schedules
///    `sink_ingest_status_from_sql_status` in `tests/sink_wiring.rs`, an
///    integration test crate. Integration tests cannot reach `pub(crate)`
///    items. `pub` is required to keep the planned test placement coherent.
///
/// 2. **Consumer intent** — the function is a bridge for any caller that
///    polls the `episode_processing_status` SQL column (a schema-level
///    surface external consumers may observe directly). External consumers
///    benefit from the typed mapping; `pub(crate)` would force them to
///    re-implement the bridge.
///
/// 3. **API surface cost** — the function is 6 LoC, pure, has no internal
///    dependencies, and is fully documented. The semver cost of `pub` is
///    minimal.
///
/// Placement (MED-03) is preserved. Visibility raised. Documented in the
/// Phase 1 commit message.
pub fn from_sql_status(s: &str) -> IngestStatus {
    match s {
        "Pending" => IngestStatus::Pending,
        "Extracting" => IngestStatus::Extracting,
        "Verified" => IngestStatus::EntitiesReady,
        "Failed" => IngestStatus::Failed(
            "from_sql_status: SQL-level failure; reason not captured".to_string(),
        ),
        // DUR-3 (V1-CANONICAL §4.2): terminal state for a `skip_extraction`
        // ingest. This arm is load-bearing, not cosmetic — without it 'Skipped'
        // falls to the catch-all below and a SUCCESSFUL ingest is reported to
        // every sink consumer as `Failed("unknown_sql_status:Skipped")`.
        "Skipped" => IngestStatus::ExtractionSkipped,
        other => IngestStatus::Failed(format!("unknown_sql_status:{other}")),
    }
}

/// Newtype for entity identity in event payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityId(pub String);

/// Newtype for edge identity in event payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeId(pub String);

/// Reference to an entity or edge in error/event context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EntityOrEdgeRef {
    Entity(EntityId),
    Edge(EdgeId),
}

/// Minimal snapshot of a fact for contradiction reporting.
///
/// Per ADR §4.8 — carries just enough context for sink subscribers to
/// understand what changed, without requiring a full database row read.
///
/// Named `SinkFact` to disambiguate from `core::schema::Fact` (the
/// bi-temporal graph-storage struct). Wildcard imports of both modules
/// previously collided.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_at: Option<DateTime<Utc>>,
}

/// A contradiction detected and resolved during Phase 2 enrichment.
///
/// `Supersession` is NOT a separate event — it is `Contradiction` with
/// `resolution = ContradictionResolution::Superseded` (G2.2 prior-art
/// finding: no surveyed system distinguishes these as separate event types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionDetected {
    pub entity_id: EntityId,
    pub prior_fact: SinkFact,
    pub new_fact: SinkFact,
    pub resolution: ContradictionResolution,
    pub detected_at: DateTime<Utc>,
}

/// An error event emitted by Phase 2 enrichment for one entity or edge.
///
/// Per ADR §4.8 / G2.1 prior-art finding: 4 of 7 surveyed systems make
/// errors first-class (LangChain on_*_error per domain, Letta
/// PostToolUseFailure, OpenAI per-line error, LlamaIndex EXCEPTION).
/// rqlm had zero error events — anomalous. Fixed here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionError {
    /// `None` when the failure occurred before entity/edge identity was
    /// established (e.g. pre-extraction parsing failure).
    pub entity_or_edge_ref: Option<EntityOrEdgeRef>,
    pub error_kind: IngestionErrorKind,
    pub is_retryable: bool,
}

/// Bundled parameters for [`IngestEventSink::on_edge_added`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
#[derive(Debug, Clone, Copy)]
pub struct OnEdgeAddedParams<'a> {
    /// Source entity of the new edge.
    pub from_entity_id: &'a str,
    /// Target entity of the new edge.
    pub to_entity_id: &'a str,
    /// Edge predicate (e.g. `"mention"`, `"subject"`, `"object"`).
    pub predicate: &'a str,
}

/// Event sink for per-episode Phase 2 events.
///
/// Per ADR §4.8: rqlc owns this trait because rqlc owns the Phase 2 work.
/// rqlc consumers who skip rqlm can subscribe to ingest events directly.
///
/// All methods have no-op defaults so implementors only override what they
/// care about. `Send + Sync` because sinks are shared across tokio tasks.
///
/// TD-133 D3: the "no-op defaults" claim above was previously a doc-lie — every
/// method was declared without a body (`;`), so all 10 implementors were forced
/// to implement all 6 (the object_store-wrapper brittleness pattern). The
/// `{}` default bodies below make the doc true AND remove the brittleness: a new
/// consumer can subscribe to just the one event it cares about. Params are
/// `_`-prefixed because the default body intentionally ignores them (no
/// `#[allow(unused_variables)]` — the underscore is the structural way to say
/// "the default deliberately does nothing with this").
pub trait IngestEventSink: Send + Sync {
    /// A new entity was extracted or resolved during the add_episode cycle.
    fn on_entity_extracted(&self, _entity_id: &str, _name: &str) {}
    /// A new edge was added between two entities.
    fn on_edge_added(&self, _params: OnEdgeAddedParams<'_>) {}
    /// A contradiction between prior and new facts was detected and resolved.
    fn on_contradiction(&self, _event: ContradictionDetected) {}
    /// Two entity records were merged (dedup collapse).
    fn on_dedup_merge(&self, _surviving_id: &str, _absorbed_id: &str) {}
    /// The Phase 2 pipeline stage changed.
    fn on_stage_change(&self, _stage: IngestStatus) {}
    /// A non-fatal error occurred for a specific entity or edge.
    fn on_ingestion_error(&self, _event: IngestionError) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-133 D3: a sink that overrides ONLY `on_entity_extracted`. The fact
    /// that this `impl` compiles at all is the proof that the other five
    /// methods have no-op defaults — before the fix all six were required and
    /// this would fail to compile. Calling an un-overridden `&str`-only method
    /// (`on_dedup_merge`) confirms the default is a callable no-op, not a panic.
    struct SingleMethodSink;

    impl IngestEventSink for SingleMethodSink {
        fn on_entity_extracted(&self, _entity_id: &str, _name: &str) {}
    }

    #[test]
    fn ingest_event_sink_defaults_allow_single_method_impl() {
        let sink = SingleMethodSink;
        sink.on_entity_extracted("e1", "Alice"); // overridden
        sink.on_dedup_merge("keep", "drop"); // default no-op — must not panic
        // on_edge_added / on_contradiction / on_stage_change /
        // on_ingestion_error take richer params; their defaults existing is
        // already proven by `SingleMethodSink` compiling without them.
    }
}
