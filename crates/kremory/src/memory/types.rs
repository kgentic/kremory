//! Public types for rqlm — scoping primitives, source references, ingest /
//! retrieval result shapes, and the context-block template enum.
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface": only the types listed here
//! are part of the crate's public surface. Internal modules (added in D.2)
//! stay `pub(crate)` so external consumers consume the 4 entry points only.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Multi-tenant scope for a single rqlm operation.
///
/// `workspace_id` isolates one customer's graph from another; `thread_id`
/// further isolates a conversational thread / meeting session within that
/// workspace. Both are caller-supplied strings — rqlm does not mint IDs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkspaceScope {
    pub workspace_id: String,
    pub thread_id: Option<String>,
}

impl WorkspaceScope {
    pub fn new(workspace_id: impl Into<String>) -> Self {
        Self {
            workspace_id: workspace_id.into(),
            thread_id: None,
        }
    }

    pub fn with_thread(workspace_id: impl Into<String>, thread_id: impl Into<String>) -> Self {
        Self {
            workspace_id: workspace_id.into(),
            thread_id: Some(thread_id.into()),
        }
    }
}

/// Kind of source an episode came from. Domain-agnostic — the host application writes
/// `Meeting`, a doc-ingestion consumer writes `Document`, a chatbot writes
/// `Chat`. No the host application-specific names leak into the public surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Meeting,
    Document,
    Chat,
}

/// Reference to the originating event for an ingested episode.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRef {
    pub kind: SourceKind,
    pub id: String,
    pub occurred_at: DateTime<Utc>,
}

/// Caller-supplied structured fact attached to an episode at ingest time.
///
/// Optional — rqlc will extract facts from raw `content` regardless. Callers
/// pass this when they already have high-confidence pre-extracted data they
/// want pinned into the graph alongside the LLM-extracted facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
}

/// Outcome of a single `ingest_episode` call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestResult {
    pub entities_added: usize,
    pub edges_added: usize,
    pub facts_invalidated: usize,
    pub duration_ms: u64,
}

/// Outcome of a single `run_dream_phase` batch consolidation cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamPhaseResult {
    pub communities_recomputed: usize,
    pub cross_meeting_merges: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    pub duration_ms: u64,
}

/// Options for `search`. All fields optional — defaults are the
/// "opinionated retrieval defaults" rqlm provides on top of rqlc's
/// hybrid retrieval primitives.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchOpts {
    /// Top-K results to return after rerank. Default: 10.
    pub limit: Option<usize>,
    /// `Some(t)` answers "what was true at time t"; `None` returns
    /// "true now" results.
    pub as_of: Option<DateTime<Utc>>,
    /// Restrict to a specific source kind (e.g. only `Document` results).
    pub source_kind: Option<SourceKind>,
}

/// A single retrieved result composed of an entity, the edges anchoring
/// it, and the temporal facts that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedContext {
    pub entity_id: String,
    pub entity_name: String,
    pub summary: String,
    pub score: f32,
    pub source_refs: Vec<SourceRef>,
}

/// Template strategy for `context_block` — which dimension of the retrieved
/// results to render into the final string handed to the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextTemplate {
    /// Render entities (name + summary). Closest to Zep's `%{entities}`.
    Entities,
    /// Render edges as a one-line-per-edge summary. Closest to Zep's `%{edges}`.
    EdgeSummary,
    /// Render the underlying temporal facts directly with `valid_at` /
    /// `invalid_at` annotations.
    TemporalFacts,
}

/// Error surface for rqlm operations.
#[derive(Debug, Error)]
pub enum RqlmError {
    #[error("rqlc layer error: {0}")]
    Core(#[from] crate::core::error::RqlError),
    #[error("invalid scope: {0}")]
    InvalidScope(String),
    #[error("unimplemented — landed in D.2: {0}")]
    Unimplemented(&'static str),
    #[error("other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, RqlmError>;
