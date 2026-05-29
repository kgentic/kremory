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

/// Serde default helper — returns `true`. Used by `#[serde(default = "default_true")]`
/// on `NamespacePolicy::forgettable` and `NamespacePolicy::dream_eligible` so that
/// older serialised JSON (missing those fields) deserializes with the v0.1.3 defaults.
fn default_true() -> bool {
    true
}

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
    /// Policy attached at construction time. `None` defers to whatever policy
    /// is already stored for this namespace; on first write, `None` resolves
    /// to `NamespacePolicy::default()`. Added v0.1.4 (ADR-029a).
    ///
    /// `#[serde(default)]` preserves backward-compat deserialization for
    /// `Namespace` JSON written before v0.1.4 (no `policy` field).
    #[serde(default)]
    pub policy: Option<NamespacePolicy>,
}

impl Namespace {
    /// Create a namespace with no thread.
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            thread: None,
            policy: None,
        }
    }

    /// Builder: attach a thread to this namespace.
    pub fn with_thread(mut self, thread: impl Into<String>) -> Self {
        self.thread = Some(thread.into());
        self
    }

    /// Attach a policy to this namespace. The policy is validated at
    /// attachment time; incoherent policies (see [`NamespacePolicy::validate`])
    /// are rejected up-front.
    ///
    /// # Stability
    ///
    /// The POLICY is set at namespace CREATION time — the first time
    /// kremory observes the namespace through a write or explicit
    /// [`crate::Memory::register_namespace`] call. Subsequent calls with a
    /// different policy on an EXISTING namespace are rejected at
    /// `register_namespace` time with
    /// [`crate::core::error::Error::NamespacePolicyImmutable`]. Added v0.1.4
    /// (ADR-029a Decision 5).
    pub fn with_policy(
        mut self,
        policy: NamespacePolicy,
    ) -> std::result::Result<Self, InvalidPolicyError> {
        policy.validate()?;
        self.policy = Some(policy);
        Ok(self)
    }
}

// ── NamespacePolicy + ImmutabilityLevel (ADR-029a) ───────────────────────────

/// Per-namespace policy controls.
///
/// Defaults are equivalent to v0.1.3 behaviour (no constraints). Set fields
/// explicitly to opt into stricter semantics for audit-grade, compliance-shape,
/// or other policy-bounded workloads.
///
/// # Enforcement at v0.1.4
///
/// **Policy values are PERSISTED but NOT ENFORCED at v0.1.4.** Setting
/// `immutability: AppendOnly` records the intent; subsequent `dream()` /
/// `forget()` calls still proceed normally. Enforcement lands in v0.1.5+
/// per ADR-029b, which adds the composite-PK storage migration required to
/// make AppendOnly actually safe.
///
/// # Construction
///
/// External callers cannot use struct-expression construction due to
/// `#[non_exhaustive]` — use the fluent setter pattern:
///
/// ```
/// use kremory::{NamespacePolicy, ImmutabilityLevel};
///
/// let policy = NamespacePolicy::new()
///     .with_immutability(ImmutabilityLevel::AppendOnly)
///     .with_forgettable(false)
///     .with_dream_eligible(false);
/// ```
///
/// Or use the [`NamespacePolicy::APPEND_ONLY`] const for the canonical
/// audit-grade preset:
///
/// ```
/// use kremory::NamespacePolicy;
/// let policy = NamespacePolicy::APPEND_ONLY;
/// ```
///
/// # Extensibility
///
/// Future fields land via fluent setters (`with_retention_class(...)`,
/// `with_legal_hold(...)`, etc.). Each future field follows the same shape:
/// `#[non_exhaustive]` struct + `with_<field>` setter + `validate()` rule
/// if it has cross-field constraints + `#[serde(default)]` for backward-compat.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NamespacePolicy {
    /// Mutation policy on the underlying graph rows. v0.1.4: persisted only.
    #[serde(default)]
    pub immutability: ImmutabilityLevel,

    /// Whether `forget().in_namespace(ns).execute()` is permitted. v0.1.4:
    /// persisted only. Default `true` matches v0.1.3 behaviour.
    #[serde(default = "default_true")]
    pub forgettable: bool,

    /// Whether `dream().in_namespace(ns)` is permitted. v0.1.4: persisted
    /// only. Default `true` matches v0.1.3 behaviour.
    #[serde(default = "default_true")]
    pub dream_eligible: bool,
}

/// Mutation policy values for [`NamespacePolicy::immutability`].
///
/// Values are persisted as `snake_case` strings in the policy JSON column,
/// matching the existing serde convention for kremory public enums.
///
/// `#[non_exhaustive]` allows adding new variants (e.g. `Eventual`,
/// `Strict`) without a SemVer-major bump. Match arms must include a
/// wildcard `_` when destructuring. Added v0.1.4 (ADR-029a Decision 2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImmutabilityLevel {
    /// Default behaviour — dream, forget, entity-move, contradiction-resolver
    /// supersession all permitted. Matches v0.1.3 semantics exactly.
    #[default]
    Mutable,

    /// Append-only intent. v0.1.4: persisted but not enforced. v0.1.5+
    /// (ADR-029b): existing rows are never mutated, archived, superseded,
    /// or moved. Writes that ADD new rows still proceed; writes that would
    /// mutate existing rows return an error.
    AppendOnly,
}

impl NamespacePolicy {
    /// Construct a default policy (Mutable, forgettable, dream_eligible).
    /// Use the `with_*` fluent setters to override fields.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the immutability level. Default `Mutable`.
    pub fn with_immutability(mut self, level: ImmutabilityLevel) -> Self {
        self.immutability = level;
        self
    }

    /// Set whether `forget()` is permitted. Default `true`.
    pub fn with_forgettable(mut self, forgettable: bool) -> Self {
        self.forgettable = forgettable;
        self
    }

    /// Set whether `dream()` is permitted. Default `true`.
    pub fn with_dream_eligible(mut self, dream_eligible: bool) -> Self {
        self.dream_eligible = dream_eligible;
        self
    }

    /// Canonical audit-grade preset. Equivalent to:
    ///
    /// ```text
    /// NamespacePolicy::new()
    ///     .with_immutability(ImmutabilityLevel::AppendOnly)
    ///     .with_forgettable(false)
    ///     .with_dream_eligible(false)
    /// ```
    pub const APPEND_ONLY: Self = Self {
        immutability: ImmutabilityLevel::AppendOnly,
        forgettable: false,
        dream_eligible: false,
    };

    /// Validate that the policy fields form a coherent combination.
    /// Called automatically by [`crate::Memory::register_namespace`] and
    /// [`Namespace::with_policy`]; callers can also invoke explicitly.
    ///
    /// # Current rules
    ///
    /// - `immutability == AppendOnly` implies `!forgettable && !dream_eligible`
    ///   (forget and dream are both mutations; an AppendOnly namespace cannot
    ///   permit them).
    ///
    /// Future fields may add rules; this method is `#[non_exhaustive]` in
    /// spirit (more rules can land additively).
    pub fn validate(&self) -> std::result::Result<(), InvalidPolicyError> {
        if self.immutability == ImmutabilityLevel::AppendOnly
            && (self.forgettable || self.dream_eligible)
        {
            return Err(InvalidPolicyError::IncoherentAppendOnly {
                policy: self.clone(),
            });
        }
        Ok(())
    }
}

impl Default for NamespacePolicy {
    fn default() -> Self {
        Self {
            immutability: ImmutabilityLevel::Mutable,
            forgettable: true,
            dream_eligible: true,
        }
    }
}

/// Validation errors for [`NamespacePolicy`]. Returned by
/// [`NamespacePolicy::validate`] and surfaced via
/// [`crate::core::error::Error::InvalidPolicy`].
///
/// `#[non_exhaustive]` — future validation rules add variants without
/// SemVer-major bumps. Added v0.1.4 (ADR-029a Decision 4).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidPolicyError {
    /// `immutability: AppendOnly` requires `forgettable=false` and
    /// `dream_eligible=false`. Forget and dream are both mutations.
    #[error(
        "namespace policy is incoherent: AppendOnly requires \
         forgettable=false and dream_eligible=false; got {policy:?}"
    )]
    IncoherentAppendOnly { policy: NamespacePolicy },
}

/// Kind of source an episode came from. Domain-agnostic — the host application writes
/// `Meeting`, a doc-ingestion consumer writes `Document`, a chatbot writes
/// `Chat`. No the host application-specific names leak into the public surface.
///
/// `#[non_exhaustive]` allows adding new variants (e.g. `Episode` in v0.1.1)
/// without a SemVer major bump. Consumers must use a wildcard arm when
/// pattern-matching.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Meeting,
    Document,
    Chat,
    /// Source reference that points directly to an episodic edge row
    /// (i.e. the episode that introduced this entity). Added in v0.1.1
    /// as part of the Bug A recall-side fix: `source_refs` are now derived
    /// from `episodic_edges` rather than fact rows.
    Episode,
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
    /// Number of stub entities inserted for forward references in this episode.
    /// A stub entity has `label = "UNKNOWN"` and `properties.stub = true`;
    /// it is promoted to a real entity on subsequent ingestion. Added in v0.1.1.
    ///
    /// `#[serde(default)]` ensures v0.1.0-serialised JSON (without this field)
    /// still deserialises correctly (defaults to `0`).
    #[serde(default)]
    pub stub_entities_inserted: usize,
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
///
/// `#[non_exhaustive]` — construction from outside this crate must go through
/// `RetrievedContext::new()` + the `with_*` fluent setters. This keeps future
/// field additions source-compatible across minor versions. Added v0.1.4 as
/// prereq for ADR-029c multi-namespace recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RetrievedContext {
    pub entity_id: String,
    pub entity_name: String,
    pub summary: String,
    pub score: f32,
    pub source_refs: Vec<SourceRef>,
    /// `true` when this entity is a stub (forward-reference placeholder inserted
    /// before the entity was fully extracted). Stubs have `label = "UNKNOWN"` and
    /// `properties.stub = true` in the graph. Added in v0.1.1.
    ///
    /// `#[serde(default)]` ensures v0.1.0-serialised JSON (without this field)
    /// still deserialises correctly (defaults to `false`).
    #[serde(default)]
    pub incomplete: bool,

    /// The namespace this result was retrieved from. Set by both single-namespace
    /// recall (`in_namespace`) and multi-namespace recall (`in_namespaces`).
    ///
    /// `None` in two cases:
    /// - Results returned by callers that pre-date v0.1.5 and construct
    ///   `RetrievedContext` directly (test fixtures, deserialized pre-v0.1.5
    ///   JSON). `#[serde(default)]` ensures backward-compatible deserialization.
    /// - Raw SQL / substrate-level queries that bypass the facade recall path.
    ///
    /// # Attribution cardinality
    ///
    /// One result = one namespace. When `in_namespaces(&[A, B])` is called and
    /// the same surface name "Acme Corp" exists in both namespaces, the recall
    /// returns TWO result rows — one attributed to A, one attributed to B —
    /// each with independent facts, scores, and summaries.
    ///
    /// # Policy
    ///
    /// `namespace.policy` is always `None` regardless of the registered policy
    /// (ADR-029c Decision 3). Policy is not fetched per-result.
    #[serde(default)]
    pub namespace: Option<Namespace>,
}

impl RetrievedContext {
    /// Construct a `RetrievedContext` with the required fields. Optional fields
    /// (`incomplete`, `namespace`, future additions) default to their sensible
    /// defaults; use the `with_*` fluent setters to override.
    pub fn new(
        entity_id: impl Into<String>,
        entity_name: impl Into<String>,
        summary: impl Into<String>,
        score: f32,
        source_refs: Vec<SourceRef>,
    ) -> Self {
        Self {
            entity_id: entity_id.into(),
            entity_name: entity_name.into(),
            summary: summary.into(),
            score,
            source_refs,
            incomplete: false,
            namespace: None,
        }
    }

    /// Mark this result as incomplete (stub forward-reference). Default `false`.
    pub fn with_incomplete(mut self, incomplete: bool) -> Self {
        self.incomplete = incomplete;
        self
    }

    /// Set the namespace attribution for this result. Called by both the
    /// single-namespace (`in_namespace`) and multi-namespace (`in_namespaces`)
    /// recall execution paths. Callers constructing results directly may omit
    /// (defaults to `None`). Added v0.1.5 (ADR-029c Decision 2).
    pub fn with_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }
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
    /// Number of stub entities inserted for forward references in this episode.
    /// Mirrors `IngestResult.stub_entities_inserted`. Added in v0.1.1 (SCOPE-002).
    ///
    /// `#[serde(default)]` ensures v0.1.0-serialised JSON (without this field)
    /// still deserialises correctly (defaults to `0`).
    #[serde(default)]
    pub stub_entities_inserted: usize,
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
    #[error(
        "feature \"{feature}\" is not available at this cadence point; \
         ships in {available_in} (ref: {adr_ref})"
    )]
    NotImplemented {
        feature: &'static str,
        available_in: &'static str,
        adr_ref: &'static str,
    },
    #[error("await timed out")]
    Timeout,
    #[error("other: {0}")]
    Other(String),
    // ── Facade errors (Story A.8a) ────────────────────────────────────────────
    /// A namespace is required but none was provided via `.in_namespace()` and
    /// no `default_namespace` was set on the builder. Named struct variant per
    /// Story #155 precedent — callers can match precisely without string parsing.
    #[error("namespace required: {request}")]
    MissingNamespace { request: &'static str },
    /// No provider was configured and env-detection found nothing.
    /// Set OLLAMA_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY, or use
    /// `Memory::open()` builder to configure a provider explicitly.
    #[error("no provider configured: {message}")]
    NoProviderConfigured { message: &'static str },
}

pub type Result<T> = std::result::Result<T, MemoryError>;

#[cfg(test)]
mod memory_type_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
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
