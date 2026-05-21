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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
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
    pub prior_fact: Fact,
    pub new_fact: Fact,
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

/// Event sink for per-episode Phase 2 events.
///
/// Per ADR §4.8: rqlc owns this trait because rqlc owns the Phase 2 work.
/// rqlc consumers who skip rqlm can subscribe to ingest events directly.
///
/// All methods have no-op defaults so implementors only override what they
/// care about. `Send + Sync` because sinks are shared across tokio tasks.
pub trait IngestEventSink: Send + Sync {
    /// A new entity was extracted or resolved during the add_episode cycle.
    fn on_entity_extracted(&self, entity_id: &str, name: &str);
    /// A new edge was added between two entities.
    fn on_edge_added(&self, from_entity_id: &str, to_entity_id: &str, predicate: &str);
    /// A contradiction between prior and new facts was detected and resolved.
    fn on_contradiction(&self, event: ContradictionDetected);
    /// Two entity records were merged (dedup collapse).
    fn on_dedup_merge(&self, surviving_id: &str, absorbed_id: &str);
    /// The Phase 2 pipeline stage changed.
    fn on_stage_change(&self, stage: IngestStatus);
    /// A non-fatal error occurred for a specific entity or edge.
    fn on_ingestion_error(&self, event: IngestionError);
}
