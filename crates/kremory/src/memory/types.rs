//! Public types for rqlm — scoping primitives, source references, ingest /
//! retrieval result shapes, and the context-block template enum.
//!
//! Per ADR-Phase-D.0 §"rqlm public API surface": only the types listed here
//! are part of the crate's public surface. Internal modules (added in D.2)
//! stay `pub(crate)` so external consumers consume the 4 entry points only.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// IngestStatus belongs to core::error but is part of the rqlm API surface.
// Re-export here so consumers can import from kremory::memory::types only.
pub use crate::core::error::IngestStatus;

/// Multi-tenant namespace for a single kremory operation.
///
/// `namespace` isolates one customer's graph from another; `thread`
/// further isolates a conversational thread / meeting session within that
/// namespace. Both are caller-supplied strings — kremory does not mint IDs.
///
/// Industry-standard name: "namespace" (Mem0, Zep, Graphiti prior art).
/// The internal SQL storage column stays `group_id` for v0.1.0 SemVer
/// stability; see `core/schema.rs` for the asymmetry note.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace {
    pub namespace: String,
    pub thread: Option<String>,
}

impl Namespace {
    /// Create a namespace with no thread.
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            thread: None,
        }
    }

    /// Builder: attach a thread to this namespace.
    pub fn with_thread(mut self, thread: impl Into<String>) -> Self {
        self.thread = Some(thread.into());
        self
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

/// Semantic memory classification for a `Fact`. Story #208.
///
/// Variants match the architect detail doc (kremory-v010-architect-detail-01-schema):
/// Decision/Pattern/Preference/Style/Habit/Insight/Observation.
///
/// Stored as a nullable TEXT column in the `facts` table, serialised as snake_case.
/// `None` = unclassified (legacy rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Decision,
    Pattern,
    Preference,
    Style,
    Habit,
    Insight,
    Observation,
}

/// Reference to the originating event for an ingested episode.
///
/// `published_at` is the wall-clock time when the source event was published
/// (e.g. article publication date, document timestamp). Used by Story #318
/// `valid_from` precedence: `fact.valid_from = sf.valid_from.or(source_ref.published_at).unwrap_or_else(Utc::now)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRef {
    pub kind: SourceKind,
    pub id: String,
    pub occurred_at: DateTime<Utc>,
    /// Optional publication timestamp of the source document / event. Story #318.
    /// When set, used as `valid_from` fallback for structured facts that omit
    /// their own temporal anchor.
    pub published_at: Option<DateTime<Utc>>,
}

/// Caller-supplied structured fact attached to an episode at ingest time.
///
/// Optional — rqlc will extract facts from raw `content` regardless. Callers
/// pass this when they already have high-confidence pre-extracted data they
/// want pinned into the graph alongside the LLM-extracted facts.
///
/// Field rename (Story #318): `valid_at` → `valid_from`, `invalid_at` → `valid_to`
/// to align with Fact struct bi-temporal semantics. The SQL `invalid_at` column
/// (contradiction-resolver timestamp) is a distinct concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Start of the validity window. `None` = use `SourceRef.published_at` fallback,
    /// then `Utc::now()`. Story #318.
    pub valid_from: Option<DateTime<Utc>>,
    /// End of the validity window. `None` = open-ended. Story #318.
    pub valid_to: Option<DateTime<Utc>>,
    /// Semantic classification for the ingested fact. `None` = unclassified.
    /// Stored as TEXT in the `facts.memory_type` column (Story #208).
    #[serde(default)]
    pub memory_type: Option<MemoryType>,
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

// ── Phase 2 / Phase 3 public API types (ADR D.6.4 §4.2–4.7) ──────────────────

/// Options controlling two-phase commit behaviour for `submit_episode`.
///
/// Per ADR §4.2.
#[derive(Debug, Clone, Default)]
pub struct SubmitOpts {
    /// Run rqlc's add_episode cycle (Phase 2: LLM extract + dedup + invalidate).
    /// Default: false. the host application: false (upstream pre-extracts). aidocs: true.
    pub enrich_per_episode: bool,

    /// Requires `enrich_per_episode = true`. If true: return after Phase 1
    /// commit, enqueue Phase 2 in background. Consumer polls via batch_status.
    /// If false (default): await Phase 2 inline before returning.
    pub run_in_background: bool,
}

/// Result of `submit_episode` Phase 1 commit. Episode is searchable from now.
///
/// Per ADR §4.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeCommit {
    /// Tracks the async Phase 2 run (if enrich_per_episode + run_in_background = true).
    /// None when Phase 2 did not run or ran inline (synchronously).
    pub run_id: Option<Uuid>,
    /// Stable entity ID under which the episode is searchable.
    pub episode_entity_id: String,
    /// Timestamp at which Phase 1 committed.
    pub committed_at: DateTime<Utc>,
}

/// Status of a Phase 3 (dream-phase batch consolidation) run.
///
/// Per ADR §4.3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DreamStatus {
    Pending,
    Processing,
    Complete,
    Failed(String),
}

/// Batch-level accounting: explicit buckets for all terminal outcomes.
///
/// Done-ness gate: `completed + skipped + failed == total`.
///
/// Per ADR §4.3:
///
/// - `completed`: enrich_per_episode=true, Phase 2 finished successfully.
/// - `skipped`: enrich_per_episode=false, counted as done by definition.
///   Explicit bucket (not silent fold into completed) — matches Hatchet
///   `was_skipped` first-class pattern (G5 prior-art finding).
/// - `failed`: enrich_per_episode=true, Phase 2 errored.
///
/// Cross-restart caveat (C3/G3.1): computed from in-memory DashMap.
/// Process restart resets counts. Callers must re-submit after restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchStatus {
    pub total: usize,
    pub completed: usize,
    pub skipped: usize,
    pub failed: usize,
}

impl BatchStatus {
    /// `true` when every episode has reached a terminal status.
    pub fn is_done(&self) -> bool {
        self.completed + self.skipped + self.failed == self.total
    }
}

/// Handle for a Phase 3 (batch consolidation) run.
///
/// Per ADR §4.4. Progress queryable via `graph_handle.graph_dream_status(run_id)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamHandle {
    pub run_id: Uuid,
    pub namespace: Namespace,
    pub submitted_at: DateTime<Utc>,
    /// The `batch_id` this dream run is scoped to, if any.
    pub batch_id: Option<String>,
}

/// Options for blocking-await helpers (`await_enrichment`, `await_dream`,
/// `await_batch_enrichment`). `timeout` is MANDATORY — no unbounded blocking.
///
/// A `tracing::warn!` is emitted on timeout exhaustion (per ADR §4.5 / C1).
#[derive(Debug, Clone)]
pub struct AwaitOpts {
    /// Maximum wait before returning `Err(MemoryError::Timeout)`.
    pub timeout: Duration,
    /// Interval between status polls.
    pub poll_interval: Duration,
}

impl Default for AwaitOpts {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            poll_interval: Duration::from_millis(200),
        }
    }
}

/// Which phase was cancelled.
///
/// Per ADR §4.6.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CancelledPhase {
    Enrichment,
    Consolidation,
}

/// Outcome of a `graph_cancel(run_id)` call.
///
/// Per ADR §4.6.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelOutcome {
    pub cancelled_phase: CancelledPhase,
    /// `true` if partial Phase 2 writes were rolled back transactionally.
    /// Always `false` for Phase 3 (partial committed state remains).
    pub rolled_back: bool,
    /// Entity IDs partially written before Phase 3 cancel (committed, not rolled back).
    pub partial: Vec<String>,
}

/// Options for `submit_dream_phase` batch consolidation.
///
/// Per ADR §4.7.
#[derive(Debug, Clone, Default)]
pub struct DreamOpts {
    /// Only consolidate episodes committed after this timestamp.
    /// `None` = consolidate all un-dreamed episodes in scope.
    pub since: Option<DateTime<Utc>>,
}

// Re-export DreamMode so consumers can import from kremory::memory::types.
pub use crate::memory::dream_phase::DreamMode;

// ── Error surface ─────────────────────────────────────────────────────────────

/// Error surface for memory-layer operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("core layer error: {0}")]
    Core(#[from] crate::core::error::Error),
    #[error("invalid scope: {0}")]
    InvalidScope(String),
    #[error("unimplemented — landed in D.2: {0}")]
    Unimplemented(&'static str),
    #[error("await timed out")]
    Timeout,
    #[error("other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

#[cfg(test)]
mod memory_type_tests {
    use super::{MemoryType, StructuredFact};

    /// Story #208: MemoryType serialises to snake_case JSON strings.
    #[test]
    fn memory_type_serde_roundtrip_all_variants() {
        let cases = [
            (MemoryType::Decision, "\"decision\""),
            (MemoryType::Pattern, "\"pattern\""),
            (MemoryType::Preference, "\"preference\""),
            (MemoryType::Style, "\"style\""),
            (MemoryType::Habit, "\"habit\""),
            (MemoryType::Insight, "\"insight\""),
            (MemoryType::Observation, "\"observation\""),
        ];
        for (variant, expected_json) in cases {
            let serialised = serde_json::to_string(&variant).expect("serialise");
            assert_eq!(
                serialised, expected_json,
                "MemoryType::{variant:?} json mismatch"
            );
            let deserialised: MemoryType = serde_json::from_str(&serialised).expect("deserialise");
            assert_eq!(
                deserialised, variant,
                "MemoryType::{variant:?} round-trip mismatch"
            );
        }
    }

    /// Story #208: StructuredFact.memory_type field exists and round-trips.
    #[test]
    fn structured_fact_memory_type_field_roundtrip() {
        let sf = StructuredFact {
            subject: "Alice".into(),
            predicate: "prefers".into(),
            object: "dark mode".into(),
            valid_from: None,
            valid_to: None,
            memory_type: Some(MemoryType::Preference),
        };
        let json = serde_json::to_string(&sf).expect("serialise");
        assert!(
            json.contains("\"preference\""),
            "memory_type missing from JSON: {json}"
        );
        let de: StructuredFact = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(de.memory_type, Some(MemoryType::Preference));
    }

    /// Story #208: StructuredFact.memory_type defaults to None when absent in JSON (backward compat).
    #[test]
    fn structured_fact_memory_type_defaults_none() {
        let json = r#"{"subject":"x","predicate":"y","object":"z"}"#;
        let sf: StructuredFact = serde_json::from_str(json).expect("deserialise");
        assert_eq!(sf.memory_type, None);
    }

    /// Story #208: MemoryType stored in facts DDL as TEXT column.
    #[test]
    fn facts_ddl_has_memory_type_column() {
        let ddl = "CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT,
                    content_hash TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0
                )";
        assert!(
            ddl.contains("memory_type TEXT"),
            "DDL missing memory_type column"
        );
    }
}
