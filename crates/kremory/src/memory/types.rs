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

/// Kind of source an episode came from. Domain-agnostic source classifier;
/// consumers tag according to their upstream type (e.g. `Meeting`, `Document`,
/// `Chat`). No consumer-specific names leak into the public surface.
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
    /// Types proposed and accepted by Dream Pass 0 type discovery (ADR-037 §3).
    /// Empty when Pass 0 was not run or produced no accepted proposals.
    #[serde(default)]
    pub types_discovered: Vec<crate::core::dream::TypeProposal>,
    /// Warnings emitted during the dream phase (e.g. degraded-mode notices).
    #[serde(default)]
    pub dream_warnings: Vec<String>,
    /// TD-060 (ADR-071 §Item 4a step 6) — `true` when the consolidation
    /// sub-phase's `ConsolidationBudget` (token or USD ceiling) was exhausted this
    /// run, skipping at least one op. `#[serde(default)]` so older persisted/
    /// deserialized payloads without this field still parse (this is an internal
    /// operator signal, not LLM-emitted output, but the same loud-vs-silent-default
    /// posture applies to any struct crossing a (de)serialization boundary).
    #[serde(default)]
    pub budget_exhausted: bool,
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

/// Default value for `RetrievedContext::entity_type_name` when absent from
/// serialised JSON (pre-TD-013 data) or constructed via `::new()` without
/// a type name. Mirrors the SQL COALESCE sentinel: `COALESCE(et.name, 'Entity')`.
fn default_entity_type_name() -> String {
    "Entity".to_string()
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

    /// Integer entity-type id for this result's entity within its namespace.
    ///
    /// Mirrors `entities.entity_type_id` (introduced by Migration 008, TD-013).
    /// `0` = "Entity" catch-all sentinel. Populated by the engine recall path
    /// from the SQL LEFT JOIN with `entity_types`. Defaults to `0` for results
    /// constructed via `RetrievedContext::new()` (e.g. test fixtures,
    /// pre-TD-013 serialised JSON).
    ///
    /// `#[serde(default)]` ensures backward-compatible deserialisation.
    #[serde(default)]
    pub entity_type_id: u32,

    /// Resolved entity type name for this result's entity.
    ///
    /// Populated from `COALESCE(entity_types.name, 'Entity')` at query time
    /// via the SQL LEFT JOIN already present in all entity SELECT paths.
    /// Mirrors `Entity::label` (which IS the type label, not the entity name).
    /// Defaults to `"Entity"` for results constructed via `RetrievedContext::new()`.
    ///
    /// `#[serde(default)]` ensures backward-compatible deserialisation.
    #[serde(default = "default_entity_type_name")]
    pub entity_type_name: String,

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

/// Bundled parameters for [`RetrievedContext::new`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
pub struct RetrievedContextNewParams {
    pub entity_id: String,
    pub entity_name: String,
    pub summary: String,
    pub score: f32,
    pub source_refs: Vec<SourceRef>,
}

impl RetrievedContext {
    /// Construct a `RetrievedContext` with the required fields. Optional fields
    /// (`incomplete`, `namespace`, future additions) default to their sensible
    /// defaults; use the `with_*` fluent setters to override.
    pub fn new(params: RetrievedContextNewParams) -> Self {
        let RetrievedContextNewParams {
            entity_id,
            entity_name,
            summary,
            score,
            source_refs,
        } = params;
        Self {
            entity_id,
            entity_name,
            summary,
            score,
            source_refs,
            incomplete: false,
            entity_type_id: 0,
            entity_type_name: default_entity_type_name(),
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
    /// Default: false. Set to false when caller has pre-extracted facts via with_facts(); set to true to have substrate's LLM extractor process raw content.
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
/// Per ADR §4.7 / ADR-037 §3 (D6).
///
/// `#[non_exhaustive]` (consumer-API hardening O5, dream-consumer-api-hardening
/// spec §5) — this struct gains fields frequently (17 and counting). External
/// callers construct it from `DreamOpts::default()` + the `DreamRequest::opts` /
/// `DreamRequest::cross_episode` builders (or field-mutation), never a struct
/// literal, so new fields never break a consumer. The napi mirror
/// (`js_dream_opts_to_rust`) is already field-mutation-based and thus compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DreamOpts {
    /// Only consolidate episodes committed after this timestamp.
    /// `None` = consolidate all un-dreamed episodes in scope.
    pub since: Option<DateTime<Utc>>,
    /// Run Dream Pass 0 type discovery (ADR-037 §3).
    /// When `true` (default), catch-all entities are clustered and presented
    /// to the LLM for new entity-type proposals.
    /// Set to `false` to skip type discovery on this dream cycle.
    pub include_type_discovery: bool,
    /// Run Dream Pass consistency_check (ADR-047) — the LLM type-verification
    /// pass. When `true` (default), typed entities are re-verified against the
    /// registry and corrected. Set to `false` to skip this LLM-cost pass on this
    /// dream cycle (the deterministic aliases + canonicalize passes still run).
    ///
    /// This is the per-pass cost lever, mirroring `include_type_discovery`. The
    /// coarser Full/Light `DreamMode` gating is a separate concern deferred to
    /// SCOPE-002 (Phase 5) — see `dream-phase-reconciliation-v2-2026-06-30` §D4.
    pub include_consistency_check: bool,
    /// Maximum number of episodes processed per dream run.
    ///
    /// Caps both Dream Pass 0 (type discovery) and Dream Pass 2 (reclassify)
    /// episode batches.  `None` (default) = no cap; all qualifying episodes
    /// are processed in one run.
    ///
    /// When the cap is hit the counter
    /// `kremory.dream.batch_cap_hit_total{phase=pass0|pass2}` is incremented
    /// per CLAUDE.md Rule 19 (observability-first-class).  Use this knob for
    /// rate-limiting dream-phase LLM spend on large corpora.
    pub max_episodes_per_run: Option<usize>,
    /// Run Site #3 type-registry post-hoc collapse (ADR-063 spec §4) — merges
    /// near-duplicate `entity_types` rows (e.g. Pass-0-discovered "Company" +
    /// "Business Organisation") via description-cosine + lexical pre-filter +
    /// LLM-verify band, remapping `entities.entity_type_id` onto the keeper.
    ///
    /// DEFAULT `true` — VALIDATED (2026-07-03). The S3 spike + the fair
    /// adversarial metrics harness (`crates/kremory/tests/corpora/site3_metrics.json`,
    /// n=119) cleared the spec §4.2 enablement gate: precision 0.949, Wilson-95%
    /// lower bound 0.861 ≥ 0.85 (strict lower-CI bar met, no N-limited fallback
    /// needed), ZERO false merges on the distinct-lemma-collision + distinct
    /// categories (asserted EXACTLY, never `>=`). The singular/plural lemma
    /// pre-filter (F3) is validated: distinct-concept cosines 0.56–0.64 ≪ 0.85
    /// (findings-log F3 RESOLVED). The 0.70 lower cosine band edge (F4) is
    /// exercised REJECT-side only — the `band_edge_moderate` corpus category
    /// (n=26) records zero false merges — but merge-side recall in [0.70, 0.85)
    /// remains unspiked (findings-log F4: **open**); enablement rests on the
    /// strict lower-CI + zero-false-merge gate and does NOT depend on it.
    /// NOTE: one ground-truth-ambiguous same-domain pair (Suspenders/Suspender,
    /// which gemma4:e4b did merge) was reclassified borderline and excluded from
    /// the gate rather than counted as a false merge (site3_corpus_v2_changelog.md
    /// v2.1) — the pass CAN merge genuinely ambiguous same-domain lemma
    /// collisions. Set to `false` to skip this LLM-cost collapse pass on a given
    /// dream cycle.
    pub include_type_registry_collapse: bool,
    /// Run Site #5 instance acronym/nickname recall (ADR-063 spec §3) — a new
    /// dream pass (NOT an extension of L7 `resolve_pending_aliases`) that
    /// nominates entity-instance pairs via a deterministic structural
    /// pre-filter (initialism test OR graph co-occurrence, spec §3.1) and
    /// adjudicates nominated pairs via batched LLM verdicts (spec §3.2),
    /// closing the acronym/nickname gap `names_lexically_compatible`
    /// documents as inherent (`disambiguation/lexical.rs`).
    ///
    /// DEFAULT `true` — VALIDATED (2026-07-03). S1/S2/S6 spikes + the fair
    /// adversarial metrics harness (`crates/kremory/tests/corpora/site5_metrics.json`,
    /// n=140) cleared the spec §4.2 enablement gate: precision 1.00, Wilson-95%
    /// lower bound 0.955 ≥ 0.85 (strict lower-CI bar met), recall 0.988, ZERO
    /// false merges — zero false positives on the coincidental-collision and
    /// distinct-people-same-nickname safety categories, with context-aware merge
    /// proven (merges typo-variants of the same person, keeps distinct siblings
    /// separate). Set to `false` to skip this LLM-cost recall pass on a given
    /// dream cycle.
    pub include_acronym_nickname_recall: bool,
    /// Run Site #2 type-novelty DESCRIPTION-gate LLM-verify band (ADR-063
    /// "The six sites" #2; impl spec §4.3 sibling table) — extends Pass 0's
    /// existing anti-redundancy gate (`discover_types` / `anti_redundancy.rs`) so a
    /// proposal whose best desc-cosine match against an existing type falls in the
    /// AMBIGUOUS zone (`[0.70, 0.85)`, or `≥0.85` with zero name-lemma overlap) is
    /// adjudicated by the shared `write_gate` (spec §2.2) instead of a hard cosine
    /// cutoff alone.
    ///
    /// DEFAULT `true` since 2026-07-03 (ADR-065). A `NeedsLlmVerify`
    /// classification is adjudicated by `adjudicate_type_novelty` and decided by
    /// the Site-#2-LOCAL `type_novelty_is_redundant` (trust the confident LLM
    /// verdict as terminal arbiter) — NOT the shared `write_gate`, whose Row 6
    /// deterministic-corroboration requirement (an ADR-057 entity-homonymy guard)
    /// over-generalized to type synonyms and made this gate inert (12/12
    /// false-accept). Re-validated on the 26-row site2 corpus (cross-family
    /// qwen2.5:7b audited, Phase 0): 11/12 redundant rejected + 14/14 novel +
    /// 8/8 band-edge accepted (precision 0.933). See `site2_metrics.json` /
    /// `site2_corpus_audit.md`. Set to `false` to preserve the pre-Site-#2 hard
    /// cutoff (≥0.85 rejects, `[0.70, 0.85)` accepts) if you do not want the
    /// LLM-verify-band adjudication.
    pub include_type_novelty_llm_verify: bool,
    /// Run the CONSOLIDATION community-detection op (ADR-066 §2.1, spec P4) —
    /// deterministic in-Rust label propagation over the entity co-occurrence graph,
    /// persisting `entity_communities` + `community_summaries`. Zero-LLM.
    ///
    /// DEFAULT `true` (consumer-API hardening D1) — all four consolidation ops
    /// default ON, made safe by ADR-073 Tier-1 reversibility. Community detection
    /// is a full-recompute (wipe-then-rebuild) with no accumulated corruption, so
    /// it is intrinsically reversible. Toggle off with `false` to skip the
    /// graph-global co-occurrence partition on a given dream cycle.
    pub include_community_detection: bool,
    /// Run the CONSOLIDATION cross-episode entity-merge op (ADR-066 §2.2, spec P3)
    /// — merge the SAME referent re-extracted verbatim (or trivially fuzzy) across
    /// DISTINCT episodes, gated by a mandatory structural-corroboration signal
    /// (shared neighbour / identical `(predicate, object)`) so a shared label alone
    /// never merges (homonymy guard, R-01b). Zero-LLM; delegates the structural
    /// merge to the shared `apply_entity_merge` executor.
    ///
    /// DEFAULT `true` (consumer-API hardening D1), landing in SHADOW by
    /// construction (`cross_episode_dry_run: true` below) — the op computes every
    /// merge decision + emits telemetry but fuses nothing until a consumer opts
    /// into apply. A fused merge is reversible via ADR-073 Tier-1
    /// (`mem.unmerge(...)` / nogood), which is what makes default-ON safe.
    ///
    /// **Prefer the `CrossEpisodeMode` builder** (`DreamRequest::cross_episode`) —
    /// it maps the tri-state Off/Shadow/Apply onto this flag + `cross_episode_dry_run`
    /// as one honest control. Setting these two raw bools directly is the L2 escape
    /// hatch (ADR-038) for callers who need to compose `DreamOpts` manually.
    pub include_cross_episode_merges: bool,
    /// Shadow-mode gate for the CONSOLIDATION cross-episode merge op (ADR-070 Stage 2
    /// of the enablement ratchet). When `true` (and `include_cross_episode_merges` is
    /// also `true`), the op computes every clique-cover + F2 corroboration decision
    /// EXACTLY as it would live, but SKIPS the `apply_entity_merge` call — no entity is
    /// actually fused. Every decision (merge-would-fire or defer/skip) still emits its
    /// `DecisionRecord` (ADR-070 Fork 2/3) with `mode = Shadow`, so an operator can
    /// observe the op's real decision distribution on THEIR data before trusting it
    /// with destructive writes.
    ///
    /// Mirrors the existing `ConsistencyCheckOpts.dry_run` precedent — same "decision
    /// computed, write skipped, skip-signal emitted" contract, generalized to this op.
    ///
    /// **Compound default `true`** (ADR-070 §2.2): a consumer flipping
    /// `include_cross_episode_merges: true` for the FIRST time, WITHOUT also explicitly
    /// setting `cross_episode_dry_run: false`, gets Stage 2 (shadow) automatically —
    /// the op runs, computes decisions, emits telemetry, but does not commit merges.
    /// Reaching Stage 5 (apply) requires setting BOTH `include_cross_episode_merges:
    /// true` AND `cross_episode_dry_run: false`. Only matters when
    /// `include_cross_episode_merges` is true (the op is off otherwise).
    pub cross_episode_dry_run: bool,
    /// Run the CONSOLIDATION supersession sweep (ADR-066 §2.3, spec P1) — the
    /// deterministic world-time window close-out lane (retire facts whose
    /// `valid_to` has passed but were never marked expired). Zero-LLM,
    /// pure date-compare, orthogonal to ingest's same-object dedup.
    ///
    /// DEFAULT `true` (consumer-API hardening D1). Zero-LLM, pure date-compare;
    /// only retires facts whose `valid_to` has demonstrably passed, and the bound
    /// can be re-opened (`mem.unsupersede`). Toggle off with `false` to skip the
    /// window close-out on a given dream cycle.
    pub include_supersession_sweep: bool,
    /// Also run supersession's OPT-IN LLM-nominated value-change lane (ADR-066
    /// §2.3, spec P1.3) — only meaningful when `include_supersession_sweep` is
    /// also `true`. The LLM only NOMINATES the value-change pairing; the write
    /// decision stays the deterministic date-compare (a time-inverted nomination
    /// is rejected structurally). Default-off because auto-superseding a
    /// DIFFERENT-object fact is unsafe without a functional-predicate registry
    /// (kremory has none).
    ///
    /// DEFAULT `false`.
    pub include_supersession_llm_nominate: bool,
    /// Run the CONSOLIDATION fact-archival op (ADR-066 §2.4, spec P2) — MOVE a
    /// long-expired, unreferenced fact from the live `facts` table into the
    /// append-only `facts_archive` audit table (INSERT + FTS-shadow delete +
    /// DELETE in one transaction), gated by a ref-count "orphans nothing" guard.
    /// Zero-LLM.
    ///
    /// DEFAULT `true` (consumer-API hardening D1). Append-only MOVE into
    /// `facts_archive` (recoverable), guarded by the ref-count "orphans nothing"
    /// check — reversible by design. Toggle off with `false` to keep long-expired
    /// facts in the live table.
    pub include_fact_archival: bool,
    /// Per-run token budget ceiling for the consolidation sub-phase (ADR-066 §2.5
    /// F-1, spec P0.1). Soft partial-abort: an op whose projected spend would push
    /// cumulative usage past this ceiling is SKIPPED (later ops still run), matching
    /// kremory's warn-and-continue posture. `None` = unbounded.
    ///
    /// DEFAULT `Some(50_000)` — a conservative cap. Only supersession's optional
    /// LLM-nominate lane consumes tokens; the other three consolidation ops
    /// (cross_episode, archive, communities) are zero-LLM / zero-token.
    pub consolidation_budget_tokens: Option<u64>,
    /// Per-run USD-micro budget ceiling for the consolidation sub-phase (TD-060,
    /// ADR-071 §Item 4a). Mirrors `consolidation_budget_tokens` exactly — soft
    /// partial-abort, AND-gated with the token ceiling at each op's pre-check
    /// (`budget.check(...) && budget.check_usd(...)`), both must pass for the op to
    /// run. `None` = unbounded (no USD cap configured; most local-Ollama runs are
    /// $0-cost and never need one).
    ///
    /// DEFAULT `None` — the ADR leaves a conservative non-`None` default as a
    /// product/UX call it does not make; this spec does not invent one.
    ///
    /// **INERT TODAY (D6, consumer-API hardening):** USD-budget enforcement does
    /// nothing at present. Every op's per-call USD projection is `0`
    /// (`OP_USD_PROJECTION`, `consolidation/mod.rs`), so `check_usd` never denies —
    /// the effective budget is **token-based** (`consolidation_budget_tokens`). This
    /// field is reserved for the day a real per-call cost source is wired
    /// (`TokenTrackingChatProvider` → AutoAgents `usage` → a rate table); until
    /// then it is recorded in the ledger but never enforces a ceiling. Most
    /// local-Ollama runs are $0-cost and never need one regardless.
    pub consolidation_budget_usd_micro: Option<u64>,
    /// Grace window (days) before an expired fact becomes archival-eligible
    /// (ADR-066 §2.4, spec P2.1). A fact is a candidate only when
    /// `expired_at < now - archive_grace_days`. `None` = no grace (archive
    /// immediately on expiry — not recommended).
    ///
    /// DEFAULT `Some(90)`.
    pub archive_grace_days: Option<u32>,
    /// TD-106 (ADR-071 §Item 4b) — post-aggregation WARN-only guard on the
    /// consolidation sub-phase's aggregate destructive-mutation count
    /// (`cross_episode_merges + supersessions_recorded + facts_archived`;
    /// `communities_updated` is EXCLUDED — a community "update" is a recomputed
    /// partition write, not a destructive mutation of fact/entity identity). When
    /// the aggregate exceeds this floor, `run_consolidation` fires an always-on
    /// `tracing::warn!` + `kremory.dream.consolidation.net_mutation_warn_total`
    /// counter — a WARN signal only, no behavior change to any op's own decision
    /// logic. `None` disables the check.
    ///
    /// DEFAULT `Some(500)` — calibration detail (mirrors gbrain's
    /// `NET_DELETION_WARN_FLOOR = 50` scaled ~10x for kremory's typical namespace
    /// size), not an architectural decision.
    pub net_mutation_warn_floor: Option<usize>,
}

/// L1 opinionated control (ADR-038) for the cross-episode entity-merge op — the
/// honest tri-state that the two raw `DreamOpts` bools
/// (`include_cross_episode_merges` + `cross_episode_dry_run`) encode implicitly.
///
/// Set via [`DreamRequest::cross_episode`](crate::DreamRequest::cross_episode).
/// Prefer this over toggling the raw bools directly (the raw fields remain the L2
/// escape hatch for callers composing a full `DreamOpts`).
///
/// | Mode | `include_cross_episode_merges` | `cross_episode_dry_run` | Effect |
/// |---|---|---|---|
/// | `Off` | `false` | — | op does not run |
/// | `Shadow` | `true` | `true` | compute + emit decisions, fuse nothing |
/// | `Apply` | `true` | `false` | compute + fuse (reversible via ADR-073 unmerge) |
///
/// Only cross_episode carries this shadow/apply distinction — the other three
/// consolidation ops have no `dry_run` axis, so a mode enum on them would be dead
/// structure (a plain `include_*` bool is the honest surface there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrossEpisodeMode {
    /// The cross-episode merge op does not run.
    Off,
    /// The op computes every merge decision + emits telemetry, but fuses no
    /// entities (shadow gate, ADR-070 Stage 2).
    Shadow,
    /// The op computes AND commits merges. Each fusion is reversible via ADR-073
    /// Tier-1 (`mem.unmerge` / nogood).
    Apply,
}

impl DreamOpts {
    /// True when ANY of the four CONSOLIDATION ops is enabled (ADR-066 spec §5).
    ///
    /// The facade uses this to skip the whole `run_consolidation` dispatcher when
    /// no op is on. With the consumer-API-hardening D1 defaults (all four ops ON)
    /// this returns `true` by default — the dispatcher runs. Explicitly toggling
    /// every op off restores the inert reconciliation-only path.
    ///
    /// `include_supersession_llm_nominate` is NOT itself an enabling flag — it only
    /// modifies the supersession sweep's behaviour, so it is excluded here (an
    /// LLM-nominate flag with the sweep off is a no-op).
    pub fn any_consolidation_enabled(&self) -> bool {
        self.include_community_detection
            || self.include_cross_episode_merges
            || self.include_supersession_sweep
            || self.include_fact_archival
    }
}

impl Default for DreamOpts {
    fn default() -> Self {
        Self {
            since: None,
            include_type_discovery: true,
            include_consistency_check: true,
            max_episodes_per_run: None,
            // ENABLED 2026-07-03 — both fair-validated against 100+-pair
            // adversarial corpora (gemma4:e4b + real nomic), spec §4.2 gate PASS,
            // ZERO false merges. See site3_metrics.json / site5_metrics.json.
            include_type_registry_collapse: true,
            include_acronym_nickname_recall: true,
            // ENABLED 2026-07-03 (ADR-065) — Site #2 now trusts the LLM as
            // terminal arbiter (`type_novelty_is_redundant`) instead of the
            // over-generalized shared `write_gate` Row 6. Re-validated on the
            // 26-row site2 corpus (cross-family qwen2.5:7b audited, Phase 0):
            // 11/12 redundant rejected + 14/14 novel + 8/8 band-edge accepted
            // (precision 0.933, recall 1.0). See site2_metrics.json /
            // site2_corpus_audit.md. The single miss (s2-025 Employer/Company)
            // is an orthogonal LLM-judgment miss tracked by a follow-up TD.
            include_type_novelty_llm_verify: true,
            // CONSOLIDATION sub-phase (ADR-066 §5) — ALL FOUR ops default ON. Made
            // safe by ADR-073 Tier-1 (reversible graph mutations: provenance +
            // unmerge + nogood + inspect) — every destructive consolidation write is
            // now reversible, so an embedded library can settle-and-improve the graph
            // by default without stranding a consumer's data. Toggle any op off via
            // its `include_*` knob (this is the "keep-on + honest surfaces" posture,
            // NOT ADR-071's original defaults-off — see the consumer-API hardening
            // spec `.ai-docs/specs/dream-consumer-api-hardening-arch-spec-2026-07-10.md`).
            //
            // ADR-071 Item 2: P4 communities has NO Wilson-LB gate (unlike Item 1's
            // P3 corpus gate) — the mechanical proof is the reversibility argument:
            // full-recompute (wipe-then-rebuild, `communities.rs`) with no accumulated
            // corruption, verified in code (`.ai-docs/specs/adr-071-dream-phase-
            // hardening-impl-spec-2026-07-06.md` §Item 2). ENABLED.
            include_community_detection: true,
            // ADR-071 Item 1: P3 corpus gate PASSED (Wilson-LB 0.971297 >= 0.95, zero
            // hub/two-hub false-merges — see
            // .ai-docs/research/adr-071-p3-corpus-calibration-findings-2026-07-09.md) ->
            // ENABLED. Lands in SHADOW by construction (`cross_episode_dry_run: true`
            // below, ADR-070 §2.2): decisions are computed + emit DecisionRecord{mode=
            // Shadow}, but `apply_entity_merge` is SKIPPED — no entity is fused. Stage-5
            // (real apply) is a deferred later ratchet (set `cross_episode_dry_run:
            // false`), NOT this build (locked-decision-3).
            include_cross_episode_merges: true,
            // Compound default TRUE (ADR-070 §2.2): the FIRST enablement of
            // `include_cross_episode_merges` lands in shadow mode (decisions computed +
            // observed, no entity fused) until an operator explicitly sets this `false`
            // to reach Stage 5 (apply). Irrelevant while the op is off (default).
            cross_episode_dry_run: true,
            // Consumer-API hardening (D1): the deterministic world-time
            // window-closeout sweep defaults ON alongside the other three ops.
            // Reversible via `mem.unsupersede(fact_id)` (ADR-071 §Item 3 companion) —
            // the sweep only sets `expired_at = valid_to` on already-past-dated
            // bounds, and a bounded window can be re-opened. The optional
            // LLM-nominate value-change lane stays OFF (unsafe without a
            // functional-predicate registry — see field doc).
            include_supersession_sweep: true,
            include_supersession_llm_nominate: false,
            // ADR-071 Item 2: P2 archive has NO Wilson-LB gate (unlike Item 1's P3
            // corpus gate) — the mechanical proof is the reversibility argument:
            // archive is an append-only MOVE into `facts_archive` (recoverable),
            // guarded by the P2.2 ref-count "orphans nothing" check, verified in
            // code (`.ai-docs/specs/adr-071-dream-phase-hardening-impl-spec-
            // 2026-07-06.md` §Item 2). ENABLED.
            include_fact_archival: true,
            // Conservative token cap (only supersession's LLM-nominate lane spends).
            consolidation_budget_tokens: Some(50_000),
            // TD-060: no USD cap by default — most consolidation ops run against
            // local Ollama at $0 cost; a non-None default is a product/UX call the
            // ADR does not make.
            consolidation_budget_usd_micro: None,
            archive_grace_days: Some(90),
            // TD-106: calibration default (see field doc — mirrors gbrain's
            // NET_DELETION_WARN_FLOOR scaled ~10x).
            net_mutation_warn_floor: Some(500),
        }
    }
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
