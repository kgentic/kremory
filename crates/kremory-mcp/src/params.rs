//! MCP wire types — floor-5 tool surface over `kremory::Memory` (ADR
//! kremory-mcp-rewrite-0.4.0 + G3 reversible-mutations gap closure).
//!
//! These duplicate a subset of `kremory`'s facade types with the
//! `schemars::JsonSchema` derive rmcp's macros need to generate tool input
//! schemas (kremory itself stays free of `schemars` — substrate-purity
//! boundary, see `feedback_substrate_purity_boundary`). The bidirectional
//! conversions to/from kremory facade types live in `conversions.rs`.
//!
//! Five tools, matching `kremory::Memory`'s primary entry points:
//! - `kremory_remember` — ingest (`Memory::remember`)
//! - `kremory_recall` — the ONLY search tool; hybrid keyword + semantic +
//!   graph retrieval (`Memory::recall`)
//! - `kremory_dream` — batch consolidation (`Memory::dream`)
//! - `kremory_list_mutations` — the SEE half of the reversible-mutations
//!   story (`Memory::list_mutations` / `Memory::mutation_history`)
//! - `kremory_undo` — the unified FIX dispatcher (`Memory::undo`)

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

/// Wire form of `kremory::RetrievedFact` for `kremory_recall`'s
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
    /// Connected facts anchored on this entity — the
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
    /// RFC 3339 UTC point-in-time (valid-time) filter. Filters
    /// which facts the recall's 1-hop expansion surfaces to what was TRUE in
    /// the world at this timestamp — entity search itself is unaffected.
    /// `None` (the default) returns present-day results.
    pub as_of: Option<String>,
    /// `text` (default) returns a prompt-ready rendered string; `structured`
    /// returns the raw entity-shaped results with a count.
    #[serde(default)]
    pub format: RecallFormat,
    /// Rendering strategy used when `format` is `text`. Ignored when
    /// `format` is `structured`.
    #[serde(default)]
    pub template: RecallTemplateWire,
    /// Rerank the top-`n` post-fusion
    /// candidates with a local cross-encoder before returning, for
    /// precision beyond BM25/vector/RRF-rank proxies. `None` (default) = no
    /// rerank. A no-op unless the server was built with kremory's `rerank`
    /// Cargo feature. Exposed
    /// here so the MCP tool surface can reach the same knob the Rust
    /// `RecallRequest::rerank_k` builder method exposes.
    #[serde(default)]
    pub rerank_k: Option<usize>,
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
    /// Batch key, carried through to `DreamHandle::batch_id` for correlation.
    ///
    /// ⚠️ **NOT idempotent today.** This doc previously claimed *"repeated calls
    /// with the same `(namespace, batch_id)` return the existing run rather than
    /// starting a new one"* — FALSE, and never implemented. The awaited path
    /// (`kremory::facade::dream::DreamRequest::execute_blocking`, the one
    /// `do_dream` selects) never reads `batch_id` at all, so every call re-runs
    /// the full pass chain.
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

/// Wire form of `kremory::TypeProposal` — an entity type proposed AND accepted
/// by Dream Pass 0 type discovery. Carries the full proposal detail (not just a
/// count) so an MCP consumer can see WHAT dream learned, mirroring the napi
/// `JsTypeProposal` surface and the recall→facts detail-carry pattern used
/// elsewhere in this crate.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TypeProposalWire {
    /// Proposed entity-type name (e.g. `"Firm"`).
    pub name: String,
    /// One-line description of what the type captures.
    pub description: String,
    /// Why Pass 0 proposed this type (the LLM's justification).
    pub justification: String,
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
    /// Entity types proposed AND accepted by Dream Pass 0 type discovery, with
    /// full proposal detail (name/description/justification). Empty when Pass 0
    /// did not run or accepted nothing. The count is `types_discovered.len()`.
    /// (Previously a bare `types_discovered_count: usize`, which dropped
    /// the `TypeProposal` detail on the wire; same parity-drop class as the
    /// recall→facts bug fixed elsewhere in this crate.)
    pub types_discovered: Vec<TypeProposalWire>,
    pub consolidation_ops_ran: ConsolidationOpsRanWire,
    pub duration_ms: u64,
    /// `true` when the consolidation budget (token/USD ceiling) was
    /// exhausted this run, skipping at least one op.
    pub budget_exhausted: bool,
    pub warnings: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────
// kremory_list_mutations (G3 — the SEE half of the reversible-mutations story)
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListMutationsParams {
    pub namespace: String,
    pub thread: Option<String>,
    /// When set, lists the mutations that touched this ONE entity (as keeper
    /// OR loser) — routes to `Memory::mutation_history` and always includes
    /// already-undone mutations, regardless of `include_undone`. `kind` /
    /// `since` are ignored in this mode (`mutation_history` has no such
    /// filters). Omit to list namespace-wide via `Memory::list_mutations`.
    pub entity_id: Option<String>,
    /// Restrict to one mutation kind, e.g. `"entity_merge"` / `"entity_edit"`
    /// / `"entity_delete"` / `"fact_delete"` / `"fact_archive"` — the FIVE
    /// LOGGED kinds. The remaining three `MutationKind` variants are reserved
    /// and return empty. An unrecognised string is rejected as invalid_params.
    /// Ignored when `entity_id` is set.
    ///
    /// ⚠️ This said FOUR logged kinds and that `fact_archive` "will always
    /// return empty" until 2026-09-14. That became false when fact archival
    /// started writing to the mutation log, and a stale parameter doc on an MCP
    /// tool is not cosmetic — it is the prompt. It actively steered the model
    /// away from a live capability, so the agent could not find the dream
    /// mutation it was looking for and concluded none existed.
    pub kind: Option<String>,
    /// RFC 3339 UTC lower bound on `created_at`. Ignored when `entity_id` is
    /// set.
    pub since: Option<String>,
    /// Include already-undone mutations. Default `false` (live / still-
    /// reversible only). Ignored (always effectively `true`) when
    /// `entity_id` is set.
    pub include_undone: Option<bool>,
}

/// Wire form of `kremory::MutationRecord` — one logged graph mutation, the
/// **SEE** half of the see+fix story. `kind` is the snake_case tag (see
/// `kremory::MutationKind`); `mutation_id` is what `kremory_undo` takes.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MutationRecordWire {
    pub mutation_id: i64,
    pub kind: String,
    /// RFC 3339 timestamp when the mutation was applied.
    pub created_at: String,
    /// `true` once the mutation has been reversed via `kremory_undo`.
    pub undone: bool,
    /// The namespace (group) this mutation scoped.
    pub group_id: String,
    /// Entity ids this mutation touched (for `entity_merge`, `[keeper, loser]`).
    pub affected_entities: Vec<String>,
    /// A short human/agent-readable summary of what the mutation did.
    pub summary: String,
}

// ─────────────────────────────────────────────────────────────────────────
// kremory_undo (G3 — the unified FIX dispatcher over Memory::undo)
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UndoParams {
    /// Guards the undo to the mutation's ORIGINAL namespace — a mismatch is
    /// rejected loudly rather than silently reversing a mutation the caller
    /// did not intend to touch.
    pub namespace: String,
    pub thread: Option<String>,
    /// The `mutation_id` from a `kremory_list_mutations` record (or from a
    /// prior `kremory_dream` / `kremory_remember` response).
    pub mutation_id: i64,
}

/// Wire mirror of `kremory::UnmergeOutcome` (§3.1) — every field is the
/// ACTUAL count reversed, never a bare "applied".
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UnmergeOutcomeWire {
    /// The restored loser entity id (the entity that had been hard-DELETEd).
    pub restored_entity: String,
    /// The keeper whose overwritten access_count / ner_confidence were restored.
    pub keeper: String,
    pub facts_repointed: usize,
    pub edges_restored: usize,
    pub entities_reopened: usize,
    /// A NOGOOD was recorded for the split pair — the next `dream()` will
    /// NOT re-merge it.
    pub nogood_recorded: bool,
    /// `true` if the mutation was already undone — an idempotent no-op.
    pub already_undone: bool,
}

/// Wire mirror of `kremory::EditEntityOutcome` (§4.3).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EditEntityOutcomeWire {
    /// The entity id AFTER the reversal (the RESTORED prior id for an undo).
    pub entity_id: String,
    /// `true` if the reversed edit was a rename/rekey.
    pub rekeyed: bool,
    /// `true` if the reversed edit was a retype.
    pub retyped: bool,
    pub facts_repointed: usize,
    pub archived_repointed: usize,
    pub edges_repointed: usize,
    pub communities_repointed: usize,
    pub entities_reopened: usize,
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — an idempotent no-op.
    pub already_undone: bool,
}

/// Wire mirror of `kremory::DeleteEntityOutcome` (§4.4).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeleteEntityOutcomeWire {
    /// The restored entity id.
    pub entity_id: String,
    /// Facts restored from archive.
    pub facts_retracted: usize,
    /// Episodic edges re-inserted.
    pub edges_removed: usize,
    /// The entity's OWN community memberships restored — 0 or 1.
    pub communities_removed: usize,
    /// NEIGHBOURS whose community membership was restored by the cascade.
    pub neighbors_retracted: usize,
    pub entities_reopened: usize,
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — an idempotent no-op.
    pub already_undone: bool,
}

/// Wire mirror of `kremory::DeleteFactOutcome` (§4.5).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeleteFactOutcomeWire {
    /// The restored fact id.
    pub fact_id: i64,
    /// `true` only when the undo actually moved the fact back out of the
    /// archive (`false` if it was found already live — an honest no-op).
    pub fact_restored: bool,
    /// NEIGHBOURS whose community membership was restored by the cascade.
    pub neighbors_retracted: usize,
    pub entities_reopened: usize,
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — an idempotent no-op.
    pub already_undone: bool,
}

/// Wire mirror of `kremory::RestoreArchivedOutcome` — what `kremory_undo`
/// reversed when the mutation was a `fact_archive` written by a dream.
///
/// This mirror did not exist until 2026-09-14, and its absence was not a missing
/// feature — it was a WRONG ANSWER. `UndoOutcomeWire::try_from` fell through to
/// its catch-all `Err` arm for this variant, and that conversion runs AFTER
/// `.execute()` has already committed. The agent was told a reversal it had
/// successfully performed had failed, so the honest response was to retry an
/// operation that had already happened.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RestoreArchivedOutcomeWire {
    /// The fact id moved back out of the archive.
    pub restored_fact_id: i64,
    /// `true` if the fact was already live — an honest no-op, nothing restored.
    pub already_live: bool,
}

/// Wire mirror of `kremory::UndoOutcome` (§3.1) — what `kremory_undo`
/// ACTUALLY reversed. Internally tagged on `reversed_kind` so every per-kind
/// count survives to the wire (no flattening to a string — the exact class
/// of parity-drop bug fixed elsewhere in this crate).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "reversed_kind", rename_all = "snake_case")]
pub enum UndoOutcomeWire {
    /// The mutation was an `entity_merge`.
    Unmerge(UnmergeOutcomeWire),
    /// The mutation was an `entity_edit`.
    EditEntity(EditEntityOutcomeWire),
    /// The mutation was an `entity_delete`.
    DeleteEntity(DeleteEntityOutcomeWire),
    /// The mutation was a `fact_delete`.
    DeleteFact(DeleteFactOutcomeWire),
    /// The mutation was a `fact_archive` — a fact a dream retired, now restored.
    RestoreArchived(RestoreArchivedOutcomeWire),
}
