//! MCP wire types — floor-3 tool surface over `kremory::Memory` (ADR
//! kremory-mcp-rewrite-0.4.0).
//!
//! These duplicate a subset of `kremory`'s facade types with the
//! `schemars::JsonSchema` derive rmcp's macros need to generate tool input
//! schemas (kremory itself stays free of `schemars` — substrate-purity
//! boundary, see `feedback_substrate_purity_boundary`). The bidirectional
//! conversions to/from kremory facade types live in `conversions.rs`.
//!
//! Three tools, matching `kremory::Memory`'s three primary entry points:
//! - `kremory_remember` — ingest (`Memory::remember`)
//! - `kremory_recall` — the ONLY search tool; hybrid keyword + semantic +
//!   graph retrieval (`Memory::recall`)
//! - `kremory_dream` — batch consolidation (`Memory::dream`)

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────
// Shared enums
// ─────────────────────────────────────────────────────────────────────────

/// Wire form of the source a `kremory_remember` episode originates from.
///
/// `Note` and `Document` both map to `kremory::SourceKind::Document` at the
/// facade — `RememberRequest::from_note` is a caller-facing synonym for
/// `from_document`, not a distinct substrate kind (see `conversions.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceKindWire {
    Document,
    Chat,
    Note,
}

/// Wire form of `kremory::RecallTemplate` — the prompt-ready rendering
/// strategy used when `RecallParams::format` is [`RecallFormat::Text`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RecallTemplateWire {
    Entities,
    EdgeSummary,
    #[default]
    TemporalFacts,
}

/// Output shape for `kremory_recall`: a prompt-ready rendered string, or the
/// raw entity-shaped structured results.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RecallFormat {
    #[default]
    Text,
    Structured,
}

/// Caller-supplied structured fact to pin alongside an episode (mirrors
/// `kremory::StructuredFact`, with `valid_at`/`invalid_at` as wire RFC 3339
/// strings instead of `chrono::DateTime<Utc>`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StructuredFactWire {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// RFC 3339 UTC start of the validity window. Falls back to
    /// `published_at`, then `now()`, when omitted.
    pub valid_at: Option<String>,
    /// RFC 3339 UTC end of the validity window. `None` = open-ended.
    pub invalid_at: Option<String>,
}

/// Wire form of `kremory::SourceRef` for `kremory_recall`'s structured output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceRefWire {
    pub kind: String,
    pub id: String,
    pub occurred_at: String,
    pub published_at: Option<String>,
}

/// Wire form of `kremory::RetrievedFact` (ADR-074 / TD-116) for `kremory_recall`'s
/// structured output — the LLM-consumable knowledge. Timestamps are RFC-3339
/// strings.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RetrievedFactWire {
    /// Natural-language rendering, e.g. `"Grace Hopper invented the compiler"`.
    pub fact: String,
    pub subject: String,
    pub predicate: String,
    /// Object — a literal value, or an object-entity's display name.
    pub object: String,
    /// `true` when `object` is an entity (edge), `false` when a literal value.
    pub object_is_entity: bool,
    /// World clock: when the fact became true (RFC-3339).
    pub valid_at: String,
    /// World clock: when the fact stopped being true, if ever (RFC-3339).
    pub invalid_at: Option<String>,
    /// System clock: when the fact was recorded (RFC-3339).
    pub recorded_at: String,
    /// System clock: when the fact row was superseded/expired, if ever (RFC-3339).
    pub expired_at: Option<String>,
    /// Extraction/caller confidence in `[0, 1]`.
    pub confidence: f64,
    /// Source episode id(s) this fact was asserted from.
    pub source_episode_ids: Vec<i64>,
    /// Relevance score inherited from the anchoring entity.
    pub score: f32,
}

/// Wire form of `kremory::RetrievedContext` for `kremory_recall`'s structured
/// output. Fields are read OUT of the real (`#[non_exhaustive]`) facade type
/// — never constructed as a `RetrievedContext` struct-literal (see
/// `conversions.rs::From<RetrievedContext>`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RetrievedContextWire {
    pub entity_id: String,
    pub entity_name: String,
    pub summary: String,
    pub score: f32,
    /// `true` when this entity is a stub forward-reference (not yet fully
    /// extracted).
    pub incomplete: bool,
    /// `0` = "Entity" catch-all sentinel.
    pub entity_type_id: u32,
    pub entity_type_name: String,
    /// Namespace group-id this result was retrieved from.
    pub namespace: Option<String>,
    pub source_refs: Vec<SourceRefWire>,
    /// Connected facts anchored on this entity (ADR-074 / TD-116) — the
    /// LLM-consumable knowledge. Empty for entities with no connected facts.
    pub facts: Vec<RetrievedFactWire>,
}

// ─────────────────────────────────────────────────────────────────────────
// kremory_remember
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RememberParams {
    /// Multi-tenant scoping partition. Required.
    pub namespace: String,
    /// Optional thread / session id within the namespace.
    pub thread: Option<String>,
    /// Raw content to remember (chat turn, document chunk, note).
    pub content: String,
    /// Source kind this episode originates from. Omit for the default
    /// (auto-tagged as chat, random id, current timestamp).
    pub source_kind: Option<SourceKindWire>,
    /// Caller-supplied source id. A random id is generated when
    /// `source_kind` is set but this is omitted.
    pub source_id: Option<String>,
    /// RFC 3339 UTC publication timestamp — bi-temporal anchor used as the
    /// `valid_at` fallback for structured facts that omit their own.
    pub published_at: Option<String>,
    /// Pre-extracted structured facts, pinned into the graph before Phase-2
    /// LLM extraction runs. LLM-duplicates of the same triple are silently
    /// swallowed (caller wins).
    #[serde(default)]
    pub structured_facts: Vec<StructuredFactWire>,
    /// Skip Phase-2 LLM extraction entirely — the episode + embedding + any
    /// `structured_facts` are still persisted. Use for bulk-import workloads
    /// where the caller is the sole source of truth for facts.
    #[serde(default)]
    pub skip_extraction: bool,
}

/// Real fields on `kremory::EpisodeCommit` (verified against
/// `crates/kremory/src/memory/types.rs` — the spec §8 provisional guess of
/// `entities_added`/`edges_added`/`facts_invalidated`/`duration_ms` belonged
/// to the deprecated `IngestResult` type, not the facade's remember outcome).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RememberOutput {
    /// Set only when Phase-2 enrichment ran in the background
    /// (`no_wait()`); `None` when it ran inline (the default).
    pub run_id: Option<String>,
    /// Stable entity id under which the episode is now searchable.
    pub episode_entity_id: String,
    /// RFC 3339 UTC timestamp at which Phase-1 (store + embed) committed.
    pub committed_at: String,
    /// Number of stub entities inserted for forward references in this
    /// episode.
    pub stub_entities_inserted: usize,
}

// ─────────────────────────────────────────────────────────────────────────
// kremory_recall
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RecallParams {
    pub namespace: String,
    pub thread: Option<String>,
    /// Search query — hybrid keyword + semantic + graph retrieval; find /
    /// recall what you know about X.
    pub query: String,
    /// Top-k results to return.
    pub k: Option<usize>,
    /// RFC 3339 UTC point-in-time filter.
    ///
    /// NOTE: not yet implemented in the kremory substrate — setting this
    /// currently makes `kremory_recall` fail loud with an internal error
    /// (`kremory::MemoryError::Core(Error::Unsupported)`) rather than
    /// silently ignoring it. Tracked upstream; this tool surfaces the real
    /// facade behaviour rather than papering over it.
    pub as_of: Option<String>,
    /// `text` (default) returns a prompt-ready rendered string; `structured`
    /// returns the raw entity-shaped results with a count.
    #[serde(default)]
    pub format: RecallFormat,
    /// Rendering strategy used when `format` is `text`. Ignored when
    /// `format` is `structured`.
    #[serde(default)]
    pub template: RecallTemplateWire,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RecallTextOutput {
    pub block: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RecallStructuredOutput {
    pub results: Vec<RetrievedContextWire>,
    pub count: usize,
}

// ─────────────────────────────────────────────────────────────────────────
// kremory_dream
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DreamParams {
    pub namespace: String,
    pub thread: Option<String>,
    /// Idempotent batch key. Repeated calls with the same
    /// `(namespace, batch_id)` return the existing run rather than starting
    /// a new one.
    pub batch_id: Option<String>,
}

/// Per-op ACTUALLY-RAN signal (mirrors `kremory::ConsolidationOpsRan`).
/// `true` iff the op's `DreamOpts.include_*` flag was on AND it actually
/// executed — disambiguates "op disabled" from "op ran, found nothing".
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConsolidationOpsRanWire {
    pub community: bool,
    pub cross_episode: bool,
    pub archival: bool,
    pub supersession_sweep: bool,
}

/// Real fields on `kremory::DreamSummary` (verified against
/// `crates/kremory/src/facade/mod.rs` — the spec §8 provisional guess of
/// `communities_recomputed`/`cross_episode_merges` used the OLD
/// `DreamPhaseResult` field names; the current facade type has a much wider,
/// differently-named surface — see `deviations` in the task report).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DreamOutput {
    pub communities_updated: usize,
    /// Cross-episode merge DECISIONS this pass (shadow + apply).
    pub cross_episode_would_merge: usize,
    /// Cross-episode merges that ACTUALLY committed (0 in shadow mode).
    pub cross_episode_merged: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    pub entities_reclassified: usize,
    pub aliases_resolved: usize,
    pub canonicalization_merges: usize,
    pub acronym_nickname_merges: usize,
    pub type_registry_merges: usize,
    pub consistency_check_corrected: usize,
    /// Count of `kremory::core::dream::TypeProposal` entries accepted by
    /// Pass 0 type discovery (the full proposals are substrate-internal —
    /// not re-mirrored here to keep the wire surface at floor 3).
    pub types_discovered_count: usize,
    pub consolidation_ops_ran: ConsolidationOpsRanWire,
    pub duration_ms: u64,
    /// `true` when the consolidation budget (token/USD ceiling) was
    /// exhausted this run, skipping at least one op.
    pub budget_exhausted: bool,
    pub warnings: Vec<String>,
}
