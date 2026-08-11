//! Reversible-graph-mutations provenance types (arch-spec
//! `reversible-graph-mutations-arch-spec-2026-07-10.md` §2.3 + §3.1).
//!
//! FOUNDATION ONLY — this module defines the schema-layer Rust types for the
//! Stage-1 reversible-mutation substrate (Migration 021, `graph_mutation_log`
//! + `merge_nogood`). It contains **no behaviour**: snapshot capture (the
//! in-txn `pre_state` INSERT), undo replay (`unmerge` / `restore_archived_fact`
//! / `unsupersede`), and nogood consultation are wired in later sub-phases.
//!
//! ## Two families of type
//!
//! 1. **`pre_state` / `inputs` snapshot structs** (§2.3) — the per-kind JSON
//!    payloads persisted into `graph_mutation_log.pre_state` / `.inputs`. These
//!    are the substrate's OWN structured emit (never LLM-authored,
//!    [[load-bearing-invariants-at-emit-not-prompt]]), so they round-trip
//!    through serde. Per [[llm-output-parse-loudly]] there is **no
//!    `#[serde(default)]` on any required field** — a snapshot that fails to
//!    deserialize is a hard error at the undo boundary, never a silent skip.
//!    Genuinely-nullable DB columns are modelled as `Option<T>` (missing key
//!    still errors; explicit JSON `null` decodes to `None`).
//!
//! 2. **Honest outcome types** (§3.1) — what an undo call ACTUALLY did, never a
//!    bare `Applied`. Returned from the `Memory` undo methods (later sub-phase)
//!    and napi-mirrored. `#[non_exhaustive]` keeps field/variant additions
//!    non-breaking.
//!
//! Sub-phase 1b (snapshot capture) wires the `entity_merge` families
//! (`LoserEntityRow`, `KeeperPre`, `FactEndpoint`, `RepointedFact`,
//! `EpisodicEdgeCols`, `EpisodicEdgeSnapshot`, `EntityMergePreState`,
//! `MergeSite`, `MergeInputs`) into `apply_merge_with_audit`, so those are no
//! longer dead. The remaining items — the `MutationKind` discriminator, the
//! non-merge `pre_state` shapes, and the honest outcome types — are consumed by
//! the later undo/restore/cascade sub-phases; each carries a surgical per-item
//! `#[allow(dead_code)]` with a note, mirroring the established
//! foundation-code precedent in `core/migrations/mod.rs` (`backup_workspace` /
//! `prune_old_backups` — "planned consumer" surgical exemptions).

use serde::{Deserialize, Serialize};

/// Reversal core (sub-phase 1c) — `unmerge` / `restore_archived_fact` /
/// `unsupersede` + `load_merge_nogoods` (arch-spec §4.2 / §4.4 / §4.5 / §6).
pub(crate) mod reversal;

/// Consumer INSPECT surface (Tier-1) — `list_mutations` / `mutation_history` +
/// the `MutationRecord` consumer view (arch-spec §3 "Inspect surface"). The SEE
/// half of the see+fix story: locate a mutation + its `mutation_id` to undo.
pub(crate) mod inspect;

/// Deterministic EDIT-ENTITY cascade (Tier-2a) — `edit_entity` (retype /
/// rename-rekey with full FK propagation + provenance snapshot + freeze re-open)
/// and its inverse `undo_entity_edit` (arch-spec §4.3 / §2.3 `entity_edit`). The
/// forward op and its undo live together because they share the FK-rekey helper
/// (undo is the inverse rekey), so keeping them in one module keeps the rekey
/// column-set a single source of truth.
pub(crate) mod edit;

/// Deterministic DELETE cascade (Tier-2b) — `delete_entity` / `delete_fact`
/// (reachability-based retract-on-zero + provenance snapshot + reconciler-freeze
/// re-open) and their inverses `undo_delete_entity` / `undo_delete_fact` (arch-spec
/// §4.4 / §4.5 / §2.3 `entity_delete` / `fact_delete`). A delete is REVERSIBLE:
/// facts are ARCHIVED (not hard-deleted) so they restore by id, and the deleted
/// entity row + its edges + community memberships are snapshotted for exact undo.
pub(crate) mod delete;

// The public inspect DATA types are re-exported at the module path (mirroring the
// honest outcome types below) so the `Memory` facade can `pub use` them 1:1.
pub use inspect::{MutationFilter, MutationRecord};

// ─── Mutation kind (the `graph_mutation_log.kind` tag) ──────────────────────

/// The kind tag stored in `graph_mutation_log.kind` (§2.1). One generic
/// kind-tagged log row per destructive graph mutation; the undo dispatcher
/// (later sub-phase) matches on this to select the `pre_state` shape to
/// deserialize. Serialises to the exact snake_case strings the spec enumerates
/// (`'entity_merge'`, `'fact_supersede'`, …).
///
/// `#[non_exhaustive]`: Stage 2 reserves an additive `derived_write` kind
/// (§2.1) and future consolidation ops become new kinds with zero schema
/// change (§5 D1) — new variants must not be a breaking change.
///
/// # Tracked-kind boundary (4 of 8 are logged today)
///
/// Only FOUR variants are currently PRODUCED into `graph_mutation_log`, and hence
/// LOGGED, listable (via `Memory::list_mutations` / `mutation_history`), and
/// reversible through the unified `Memory::undo` dispatcher:
/// [`EntityMerge`](Self::EntityMerge), [`EntityEdit`](Self::EntityEdit),
/// [`EntityDelete`](Self::EntityDelete), [`FactDelete`](Self::FactDelete).
///
/// The other four — [`FactSupersede`](Self::FactSupersede),
/// [`FactArchive`](Self::FactArchive), [`CommunityAssign`](Self::CommunityAssign),
/// [`CanonicalForm`](Self::CanonicalForm) — are RESERVED: they are not written to
/// the log yet, so `list_mutations(kind = <reserved>)` returns empty BY
/// CONSTRUCTION, and `Memory::undo` on a would-be row of that kind is a loud
/// `Error::UndoUnsupportedKind`. (`FactSupersede` / `FactArchive` are still
/// reversible — via `Memory::unsupersede` / `Memory::restore_archived_fact`, which
/// take a domain `fact_id` / `archived_fact_id`, not a `mutation_id`.) The runtime
/// enforcement of this boundary is the `undo` dispatcher's loud error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MutationKind {
    /// Two entities fused into one (`apply_merge_with_audit`).
    EntityMerge,
    /// A fact bounded (`valid_to` / `expired_at`) by supersession.
    FactSupersede,
    /// A fact moved to `facts_archive` (P2 archival).
    FactArchive,
    /// An entity property/id edited (retype or rename/rekey).
    EntityEdit,
    /// An entity deleted (its facts archived, artifacts retracted).
    EntityDelete,
    /// A single fact deleted (archived + cascade retract-on-zero).
    FactDelete,
    /// A community membership assignment (P4 communities).
    CommunityAssign,
    /// A canonical-form assignment (canonicalize pass).
    CanonicalForm,
}

impl MutationKind {
    /// The exact snake_case tag stored in `graph_mutation_log.kind` (§2.1) — the
    /// same string serde emits, but as a `const` `&'static str` for SQL binding
    /// and summaries without a `serde_json` round-trip.
    pub(crate) fn as_tag(self) -> &'static str {
        match self {
            Self::EntityMerge => "entity_merge",
            Self::FactSupersede => "fact_supersede",
            Self::FactArchive => "fact_archive",
            Self::EntityEdit => "entity_edit",
            Self::EntityDelete => "entity_delete",
            Self::FactDelete => "fact_delete",
            Self::CommunityAssign => "community_assign",
            Self::CanonicalForm => "canonical_form",
        }
    }

    /// Parse the `graph_mutation_log.kind` tag (§2.1). Parse-loudly
    /// ([[llm-output-parse-loudly]] extended to substrate emit): an unrecognised
    /// tag is a hard `Error`, never a silent default — a log row we cannot
    /// classify must surface, not be dropped from the inspect view.
    pub(crate) fn from_tag(tag: &str) -> crate::core::error::Result<Self> {
        match tag {
            "entity_merge" => Ok(Self::EntityMerge),
            "fact_supersede" => Ok(Self::FactSupersede),
            "fact_archive" => Ok(Self::FactArchive),
            "entity_edit" => Ok(Self::EntityEdit),
            "entity_delete" => Ok(Self::EntityDelete),
            "fact_delete" => Ok(Self::FactDelete),
            "community_assign" => Ok(Self::CommunityAssign),
            "canonical_form" => Ok(Self::CanonicalForm),
            other => Err(crate::core::error::Error::Other(anyhow::anyhow!(
                "graph_mutation_log: unknown mutation kind tag `{other}`"
            ))),
        }
    }
}

// ─── entity_merge snapshot (§2.3) ───────────────────────────────────────────

/// The full loser `entities` row hard-DELETEd by a merge, snapshotted so undo
/// re-INSERTs it exactly (§2.3(1)).
///
/// The live `entities` column set is exactly these 11 fields (`defs_g2.rs`
/// recreate DDL) — there is **NO `label` column** (dropped by Migration 009).
/// `entity_type_assigned_at` MUST be captured: reclassify stamps it, so
/// restoring the loser without it would lose the pre-merge type-assignment
/// timestamp. Nullable columns (`properties`, `embedding`, `updated_at`,
/// `entity_type_source`, `entity_type_assigned_at`, `ner_confidence`) are
/// `Option<T>` so undo restores a NULL as a NULL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LoserEntityRow {
    pub(crate) id: String,
    pub(crate) group_id: String,
    pub(crate) properties: Option<String>,
    /// The `embedding` BLOB, base64-encoded for JSON transport.
    pub(crate) embedding_b64: Option<String>,
    pub(crate) recorded_at: String,
    pub(crate) updated_at: Option<String>,
    pub(crate) access_count: i64,
    pub(crate) entity_type_id: i64,
    pub(crate) entity_type_source: Option<String>,
    pub(crate) entity_type_assigned_at: Option<String>,
    pub(crate) ner_confidence: Option<f64>,
}

/// The keeper's PRE-merge values that the merge OVERWRITES (§2.3(2)):
/// `access_count` is accumulated and `ner_confidence` is noisy-OR'd, both
/// non-invertible — so undo restores these snapshotted values exactly rather
/// than trying to subtract the loser back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KeeperPre {
    pub(crate) id: String,
    pub(crate) access_count: i64,
    pub(crate) ner_confidence: Option<f64>,
}

/// Which fact endpoint a merge re-pointed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FactEndpoint {
    Subject,
    Object,
}

/// A fact endpoint re-pointed loser→keeper by a merge (§2.3(3)). The endpoint
/// was stamped `corroboration_inert = 1`; undo must restore the **prior** flag,
/// NOT blanket-clear it, so a fact already inert from an EARLIER merge is not
/// wrongly re-corroborated (the monotone-undo trap, §12 CH-3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RepointedFact {
    pub(crate) fact_id: i64,
    pub(crate) endpoint: FactEndpoint,
    /// The `facts.corroboration_inert` value (0 / 1) BEFORE the merge stamped it.
    pub(crate) prior_corroboration_inert: i64,
}

/// The non-key `episodic_edges` columns of a snapshotted loser edge (§2.3(4)).
/// The presence key (`episode_id`, `entity_group_id`) lives on the enclosing
/// [`EpisodicEdgeSnapshot`]; these are the remaining columns undo needs to
/// re-INSERT the row (the `id` is `AUTOINCREMENT`, not preserved).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EpisodicEdgeCols {
    pub(crate) entity_id: String,
    pub(crate) role: String,
    pub(crate) recorded_at: String,
}

/// A loser `episodic_edges` row, remapped or orphan-deleted by a merge
/// (§2.3(4)). `collided` records whether the keeper ALREADY owned an edge for
/// `(episode_id, entity_group_id)` pre-merge — it decides the undo op:
/// collided → re-INSERT the loser row (keeper's untouched); non-collided →
/// re-point the remapped row back to the loser (§4.2 step 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EpisodicEdgeSnapshot {
    pub(crate) episode_id: i64,
    pub(crate) entity_group_id: String,
    pub(crate) collided: bool,
    pub(crate) cols: EpisodicEdgeCols,
}

/// `pre_state` for `MutationKind::EntityMerge` — the FOUR things a merge
/// destroys (§2.3), all captured inside the merge's own txn before the
/// destructive writes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EntityMergePreState {
    pub(crate) loser_entity_row: LoserEntityRow,
    pub(crate) keeper_pre: KeeperPre,
    pub(crate) repointed_facts: Vec<RepointedFact>,
    pub(crate) episodic_edges: Vec<EpisodicEdgeSnapshot>,
    /// TD-203 D1 — fact ids the merge EXPIRED because re-pointing would have
    /// collapsed both endpoints onto the keeper (`X pred X`).
    ///
    /// A fact linking loser and keeper is meaningful only while they are
    /// distinct; once merged it asserts nothing. Before this field existed the
    /// re-point silently manufactured self-loops — 41 live ones on the shipped
    /// LoCoMo corpus, 18 of them on the reserved `potential_alias` predicate,
    /// which L4 cannot even emit (it compares a NEW entity to a DIFFERENT
    /// existing one). Traced to `canonicalize` merges via `graph_mutation_log`
    /// rows 2-3 (`12 july 2023` and `20 july 2023` → `3 july 2023`).
    ///
    /// Captured so `unmerge` can REVIVE them: undo un-points the endpoints back
    /// to the loser, at which point the fact is meaningful again. Without this
    /// the expiry would be one-way and `unmerge` would silently return a
    /// strictly smaller graph than it was handed — breaking the ADR-073
    /// reversibility guarantee.
    ///
    /// `#[serde(default)]` is REQUIRED for backward compatibility and is NOT a
    /// silent-default anti-pattern (Rule 21 governs LLM-emitted output; this is
    /// our own persisted state): `graph_mutation_log` rows written before this
    /// field existed have no key, and an empty list is the CORRECT reading of
    /// them — those merges expired nothing, because the code could not.
    #[serde(default)]
    pub(crate) self_loops_expired: Vec<i64>,
}

/// Which of the three merge-producing sites fired a merge — the `inputs.site`
/// enum (§2.3). Load-bearing for the nogood: all THREE sites must consult it
/// (V3 fix, §6.2), and Site #5 was the previously-unguarded bypass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum MergeSite {
    #[serde(rename = "cross_episode")]
    CrossEpisode,
    #[serde(rename = "canonicalize")]
    Canonicalize,
    #[serde(rename = "site5_acronym_nickname")]
    Site5AcronymNickname,
}

impl MergeSite {
    /// The stable snake_case tag (matches the serde rename) — used to render the
    /// consumer-facing merge summary (`site=…`) in [`inspect::MutationRecord`].
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CrossEpisode => "cross_episode",
            Self::Canonicalize => "canonicalize",
            Self::Site5AcronymNickname => "site5_acronym_nickname",
        }
    }
}

/// `inputs` for `MutationKind::EntityMerge` (§2.3). `pair_lo` / `pair_hi` are
/// the SORTED unordered pair — the nogood key (§6.2) — so a keeper/loser
/// role-flip between passes cannot evade the anti-re-merge ban.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MergeInputs {
    pub(crate) pair_lo: String,
    pub(crate) pair_hi: String,
    pub(crate) keeper: String,
    pub(crate) loser: String,
    pub(crate) site: MergeSite,
    /// Embedding cosine that drove the merge (NULL for a clear/structural-only
    /// decision, mirroring `identity_verdict_audit.cosine`).
    pub(crate) cosine: Option<f64>,
    /// Whether a deterministic structural signal contributed to the decision.
    pub(crate) structural_signal: bool,
}

// ─── other Stage-1 kinds (§2.3) ─────────────────────────────────────────────

/// `pre_state` for `MutationKind::FactSupersede` (§2.3). The fact body is
/// intact — supersession only bounds it — so undo clears whichever bound was
/// set. Both are `Option` because either (or both) may have been NULL before.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // planned consumer: `unsupersede` undo sub-phase (Tier 2, §3.1).
pub(crate) struct FactSupersedePreState {
    pub(crate) fact_id: i64,
    pub(crate) prior_valid_to: Option<String>,
    pub(crate) prior_expired_at: Option<String>,
}

/// `pre_state` for `MutationKind::FactArchive` (§2.3) — the `facts_archive.id`
/// of the archived row. Undo is the `restore_archived_fact` helper (§3.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // planned consumer: `restore_archived_fact` undo sub-phase (Tier 2, §3.2).
pub(crate) struct FactArchivePreState {
    pub(crate) archived_fact_id: i64,
}

/// A presence key for an `episodic_edges` row re-keyed by an entity rename
/// (§2.3 `entity_edit`). The `entity_id` is `old_id`/`new_id`, known from the
/// enclosing edit, so only the per-episode key is snapshotted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EpisodeEdgeKey {
    pub(crate) episode_id: i64,
    pub(crate) entity_group_id: String,
}

/// `pre_state` for `MutationKind::EntityEdit` (§2.3 + §4.3) — a `retype`
/// (type-only, id unchanged) or `rename`/REKEY (id changed → cascade re-points
/// every dependent). On a rekey, `affected_archived_fact_ids` snapshots the
/// `facts_archive` rows re-pointed (V5 fix) so undo reverts the archived-fact
/// ids too — a rekey that skips `facts_archive` would leave dangling ids that
/// `restore_archived_fact` would resurrect pointing at the dead id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EntityEditPreState {
    pub(crate) old_id: String,
    pub(crate) new_id: String,
    pub(crate) old_label: Option<String>,
    pub(crate) new_label: Option<String>,
    pub(crate) old_type_id: i64,
    pub(crate) new_type_id: i64,
    /// `true` for a rename/rekey (id changed); `false` for a retype.
    pub(crate) rekey: bool,
    pub(crate) affected_fact_ids: Vec<i64>,
    pub(crate) affected_archived_fact_ids: Vec<i64>,
    pub(crate) affected_episode_edge_keys: Vec<EpisodeEdgeKey>,
    pub(crate) prior_type_source: Option<String>,
    pub(crate) prior_type_assigned_at: Option<String>,
    pub(crate) prior_community_id: Option<i64>,
}

/// `inputs` for `MutationKind::EntityEdit` (§2.3) — the derivation source for the
/// consumer INSPECT summary (§3), NOT the undo payload (that is
/// [`EntityEditPreState`]). Our OWN structured emit (never LLM-authored, §2.1) so
/// it round-trips through serde; parse-loudly at the inspect boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EditInputs {
    pub(crate) old_id: String,
    pub(crate) new_id: String,
    /// `true` for a rename/rekey (id changed); `false` for a retype.
    pub(crate) rekey: bool,
    pub(crate) old_type_id: i64,
    pub(crate) new_type_id: i64,
}

// ─── entity_delete / fact_delete snapshots (Tier-2b cascade, §2.3 / §4.4 / §4.5) ─

/// A deleted `episodic_edges` row captured for a `delete_entity` undo (§4.4). The
/// `entity_id` is the deleted entity (known from the enclosing snapshot's
/// `entity_row.id`), so only the per-edge presence key + the non-key columns undo
/// needs to re-INSERT the row are captured (the `id` is `AUTOINCREMENT`, not
/// preserved).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DeletedEpisodicEdge {
    pub(crate) episode_id: i64,
    pub(crate) entity_group_id: String,
    pub(crate) role: String,
    pub(crate) recorded_at: String,
}

/// A NEIGHBOUR whose DERIVED artifact (community membership) a retract-on-zero
/// cascade retracted (§4.4 / §4.5 — B5/DRed) because the delete dropped its
/// live-fact support to zero. `prior_community_id` is snapshotted so undo restores
/// the membership exactly. The base entity is NEVER auto-deleted — only its derived
/// artifacts are retracted (HippoRAG #17 reference-count discipline: teardown the
/// derived, keep the base).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RetractedNeighbor {
    pub(crate) entity_id: String,
    pub(crate) prior_community_id: i64,
}

/// `pre_state` for `MutationKind::EntityDelete` (§4.4). Everything an undo needs to
/// fully restore the deleted entity: its full row (reuses [`LoserEntityRow`] — the
/// same 11 live `entities` columns a merge snapshots), the facts ARCHIVED (restorable
/// by id, never hard-deleted), its episodic edges, its OWN community membership, and
/// the neighbours whose derived artifacts the retract-on-zero cascade retracted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EntityDeletePreState {
    pub(crate) entity_row: LoserEntityRow,
    pub(crate) archived_fact_ids: Vec<i64>,
    pub(crate) episodic_edges: Vec<DeletedEpisodicEdge>,
    pub(crate) prior_community_id: Option<i64>,
    pub(crate) retracted_neighbors: Vec<RetractedNeighbor>,
}

/// `inputs` for `MutationKind::EntityDelete` — the derivation source for the
/// consumer INSPECT summary (§3), NOT the undo payload (that is
/// [`EntityDeletePreState`]). Our OWN structured emit (never LLM-authored, §2.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DeleteEntityInputs {
    pub(crate) entity_id: String,
    pub(crate) facts_retracted: usize,
}

/// `pre_state` for `MutationKind::FactDelete` (§4.5). The fact is ARCHIVED
/// (restorable by id via `restore_archived_fact`); undo restores it + un-retracts
/// any neighbour whose support the deletion dropped to zero. `object_id` is `None`
/// for a literal-object fact (`facts.object_id IS NULL`, `object_value` carries the
/// literal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FactDeletePreState {
    pub(crate) archived_fact_id: i64,
    pub(crate) subject_id: String,
    pub(crate) object_id: Option<String>,
    pub(crate) retracted_neighbors: Vec<RetractedNeighbor>,
}

/// `inputs` for `MutationKind::FactDelete` — the derivation source for the consumer
/// INSPECT summary (§3), NOT the undo payload. Our OWN structured emit (§2.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DeleteFactInputs {
    pub(crate) fact_id: i64,
    pub(crate) subject_id: String,
    pub(crate) object_id: Option<String>,
}

// ─── honest outcome types (§3.1) ────────────────────────────────────────────

/// Returned by `unmerge(...)` — every field is the ACTUAL count reversed
/// (success-signal honesty, §3.1), never a bare `Applied`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UnmergeOutcome {
    /// The restored loser entity id (the entity that had been hard-DELETEd).
    pub restored_entity: String,
    /// The keeper whose overwritten access_count / ner_confidence were restored.
    pub keeper: String,
    /// Facts whose endpoint + corroboration_inert flag were reverted.
    pub facts_repointed: usize,
    /// Episodic edges re-inserted (collided) or re-pointed back (non-collided).
    pub edges_restored: usize,
    /// Entities whose reconciler-freeze stamp was re-opened (§6.1).
    pub entities_reopened: usize,
    /// A NOGOOD was recorded for the split pair.
    pub nogood_recorded: bool,
    /// `true` if the mutation was already undone — a no-op idempotent call
    /// (all counts zero, `nogood_recorded = false`), never a re-application.
    pub already_undone: bool,
}

/// Returned by `restore_archived_fact(...)` (§3.1).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RestoreArchivedOutcome {
    pub restored_fact_id: i64,
    /// `true` if the fact was already live — honest no-op, nothing restored.
    pub already_live: bool,
}

/// Returned by `edit_entity(...)` and its inverse `undo_entity_edit(...)` (§4.3).
/// Every count is the ACTUAL number of rows re-pointed / affected (success-signal
/// honesty, §3.1), never a bare `Applied`. A `retype` leaves the FK counts zero
/// (id unchanged); a `rename`/rekey populates them.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EditEntityOutcome {
    /// The entity id AFTER the edit — the new id for a rename, the unchanged id
    /// for a retype (or the RESTORED prior id for an undo).
    pub entity_id: String,
    /// `true` if this was a rename/rekey (id changed, FKs re-pointed).
    pub rekeyed: bool,
    /// `true` if the entity's `entity_type_id` was changed (retype).
    pub retyped: bool,
    /// `facts` rows re-pointed (subject + object endpoints). Zero for a retype.
    pub facts_repointed: usize,
    /// `facts_archive` rows re-pointed (V5 — archived facts carry TEXT endpoint
    /// ids too, so a rekey MUST update them). Zero for a retype.
    pub archived_repointed: usize,
    /// `episodic_edges` rows re-pointed. Zero for a retype.
    pub edges_repointed: usize,
    /// `entity_communities` rows re-pointed (rename) or invalidated (retype
    /// drops membership so community detection re-places the re-typed entity).
    pub communities_repointed: usize,
    /// Entities whose reconciler-freeze stamp was re-opened (§6.1) so the next
    /// `dream()` re-processes the edited entity.
    pub entities_reopened: usize,
    /// The `graph_mutation_log.id` of the `entity_edit` row — for a forward edit,
    /// the newly recorded row (pass it to `undo_entity_edit` to reverse); for an
    /// undo, the id of the row that was reversed (its `undone_at` is now set).
    pub mutation_id: i64,
    /// `true` if `undo_entity_edit` found the mutation already reversed — a
    /// zero-count idempotent no-op (never a double-reversal). Always `false` for a
    /// forward `edit_entity`. Mirrors the `unmerge` / delete-undo `already_undone`
    /// precedent so the undo counter is not inflated by a repeat call.
    pub already_undone: bool,
}

/// Returned by `delete_entity(...)` and its inverse `undo_delete_entity(...)`
/// (§4.4). Every count is the ACTUAL number of rows affected (success-signal
/// honesty, §3.1), never a bare `Applied`. On a FORWARD delete the counts are what
/// was retracted/removed; on an UNDO they are the inverse — the rows RESTORED (facts
/// un-archived, edges re-inserted, memberships restored) — mirroring the
/// `EditEntityOutcome` forward/undo reuse precedent.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DeleteEntityOutcome {
    /// The deleted (or, on undo, restored) entity id.
    pub entity_id: String,
    /// Facts ARCHIVED (forward) or restored from archive (undo) — never hard-deleted,
    /// so a delete is always recoverable.
    pub facts_retracted: usize,
    /// Episodic edges removed (forward) or re-inserted (undo).
    pub edges_removed: usize,
    /// The entity's OWN community memberships removed (forward) or restored (undo)
    /// — 0 or 1.
    pub communities_removed: usize,
    /// NEIGHBOURS whose community membership the retract-on-zero cascade retracted
    /// (forward) or restored (undo) because the delete dropped their live-fact
    /// support to zero.
    pub neighbors_retracted: usize,
    /// Entities whose reconciler-freeze stamp was re-opened (§6.1) so the next
    /// `dream()` re-processes them.
    pub entities_reopened: usize,
    /// The `graph_mutation_log.id` of the `entity_delete` row — for a forward delete,
    /// the newly recorded row (pass it to `undo_delete_entity`); for an undo, the id
    /// of the row reversed (its `undone_at` is now set).
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — a zero-count idempotent no-op.
    pub already_undone: bool,
}

/// Returned by `delete_fact(...)` and its inverse `undo_delete_fact(...)` (§4.5).
/// Every count is the ACTUAL number of rows affected (success-signal honesty).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DeleteFactOutcome {
    /// The deleted (or, on undo, restored) fact id.
    pub fact_id: i64,
    /// `true` only when an UNDO actually moved the fact back out of the archive.
    /// `false` on a forward `delete_fact` (which archives, restoring nothing) and on
    /// an undo that found the fact ALREADY live — an honest no-op restore (the archived
    /// fact had been restored out-of-band), so the undo is not misreported as having
    /// restored it. Distinct from `already_undone`, which flags the WHOLE mutation as
    /// previously reversed (§3.1 success-signal honesty).
    pub fact_restored: bool,
    /// NEIGHBOURS (endpoint entities) whose community membership the retract-on-zero
    /// cascade retracted (forward) or restored (undo).
    pub neighbors_retracted: usize,
    /// Entities whose reconciler-freeze stamp was re-opened (§6.1).
    pub entities_reopened: usize,
    /// The `graph_mutation_log.id` of the `fact_delete` row — pass to
    /// `undo_delete_fact` to reverse (forward), or the reversed row's id (undo).
    pub mutation_id: i64,
    /// `true` if the mutation was already undone — a zero-count idempotent no-op.
    pub already_undone: bool,
}

/// Returned by `unsupersede(...)` (§3.1).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum UnsupersedeOutcome {
    /// The bound was cleared (fact re-opened as currently-true).
    Cleared {
        fact_id: i64,
        cleared_valid_to: bool,
        cleared_expired_at: bool,
    },
    /// The fact had no bound set — nothing to clear (honest no-op, not a lie).
    NotSuperseded { fact_id: i64 },
}
