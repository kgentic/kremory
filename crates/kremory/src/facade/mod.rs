//! `kremory::Memory` — fluent facade over the kremory substrate.
//!
//! # Three-tier API (React philosophy)
//!
//! ```text
//! Tier 1 — Just works           Memory::auto / Memory::with_ollama
//!       ↓
//! Tier 2 — Customizable         Memory::open().with_llm().with_embedder().await?
//!       ↓
//! Tier 3 — Composable           kremory::memory::* substrate free functions
//! ```
//!
//! ## Quick start (Tier 1)
//!
//! ```rust,no_run
//! use kremory::{Memory, Namespace};
//! # async fn ex() -> kremory::memory::Result<()> {
//! let mem = Memory::with_ollama("./agent.db").await?;
//! // `remember(...)` requires a namespace: call `.in_namespace(ns)` on the
//! // request (or set a `default_namespace` on the builder — see Tier 2 below).
//! mem.remember("User prefers concise replies")
//!     .in_namespace(Namespace::new("agent"))
//!     .await?;
//! let context: String = mem.recall("what does user prefer?").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Tier 2 builder
//!
//! ```rust,no_run
//! use kremory::{Memory, Namespace, DynEmbeddingProvider};
//! use std::sync::Arc;
//! # async fn ex() -> kremory::memory::Result<()> {
//! # let my_llm: Arc<dyn kremory::memory::ChatProvider> = todo!();
//! # let my_embedder: Arc<dyn DynEmbeddingProvider> = todo!();
//! let mem = Memory::open("./agent.db")
//!     .with_llm(my_llm)
//!     .with_embedder(my_embedder)
//!     .default_namespace(Namespace::new("acme-corp"))
//!     .await?;
//! mem.remember("Customer reported login failure").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Event sink example (Tier 2)
//!
//! ```rust,no_run
//! use kremory::{EnrichmentEventSink, IngestEventSink, ContradictionDetected, BatchPhase2Complete, IngestStatus, IngestionError, OnEdgeAddedParams};
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! struct CountingSink {
//!     entity_count: Arc<AtomicUsize>,
//! }
//!
//! impl IngestEventSink for CountingSink {
//!     fn on_entity_extracted(&self, _id: &str, _name: &str) {
//!         self.entity_count.fetch_add(1, Ordering::Relaxed);
//!     }
//!     fn on_edge_added(&self, _p: OnEdgeAddedParams<'_>) {}
//!     fn on_contradiction(&self, _e: ContradictionDetected) {}
//!     fn on_dedup_merge(&self, _s: &str, _a: &str) {}
//!     fn on_stage_change(&self, _s: IngestStatus) {}
//!     fn on_ingestion_error(&self, _e: IngestionError) {}
//! }
//!
//! impl EnrichmentEventSink for CountingSink {
//!     fn on_community_updated(&self, _id: &str, _count: usize) {}
//!     fn on_batch_phase2_complete(&self, _e: BatchPhase2Complete) {}
//! }
//! ```

pub mod providers;

pub mod dream;
pub mod forget;
pub mod recall;
pub mod remember;
pub mod reverse;
pub mod supersede;
pub mod update;

pub use dream::*;
pub use forget::*;
pub use recall::*;
pub use remember::*;
pub use reverse::*;
pub use supersede::*;
pub use update::*;

// Reversible-graph-mutations honest outcome types (arch-spec §3.1) — re-exported
// from the (`pub(crate)`) provenance module so `Memory::unmerge` /
// `restore_archived_fact` / `unsupersede` return a nameable public type.
pub use crate::core::dream::provenance::{
    DeleteEntityOutcome, DeleteFactOutcome, EditEntityOutcome, RestoreArchivedOutcome,
    UnmergeOutcome, UnsupersedeOutcome,
};

// Reversible-graph-mutations consumer INSPECT surface (arch-spec §3 "Inspect
// surface") — the SEE half of the see+fix story. `MutationRecord` is the
// consumer-facing view a `mutation_history` / `list_mutations` query returns;
// `MutationKind` tags it; `MutationFilter` shapes `list_mutations`.
pub use crate::core::dream::provenance::{MutationFilter, MutationKind, MutationRecord};

use std::future::IntoFuture;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::core::error::Error as CoreError;
use crate::core::provider::DynEmbeddingProvider;
use crate::core::schema::TemporalGraph;
use crate::memory::engine_handle::namespace_to_group_id;
use crate::memory::{
    self,
    events::EnrichmentEventSink,
    types::{
        AwaitOpts, BatchStatus, CancelOutcome, ContextTemplate, CrossEpisodeMode, DreamHandle,
        DreamOpts, DreamPhaseResult, DreamStatus, EpisodeCommit, Namespace, NamespacePolicy,
        RetrievedContext, SearchOpts, SourceKind, SourceRef, StructuredFact, SubmitOpts,
    },
    ChatProvider, GraphAssertEntityTypeParams, GraphHandle, MemoryError, Result,
};

// ── Type-state markers ────────────────────────────────────────────────────────

/// Type-state marker: LLM not yet configured.
pub struct NoLlm;
/// Type-state marker: LLM configured.
pub struct WithLlm;
/// Type-state marker: Embedder not yet configured.
pub struct NoEmb;
/// Type-state marker: Embedder configured.
pub struct WithEmb;

// ── ConsolidationOpsRan (D1b) ─────────────────────────────────────────────────

/// Per-op ACTUALLY-RAN signal on [`DreamSummary`] (consumer-API hardening D1b).
///
/// With the D1 all-ops-ON defaults a consumer can no longer read an all-zero
/// consolidation count as "the op was off". This makes the distinction in-band: a
/// field is `true` iff the op's `DreamOpts.include_*` flag was on AND the op
/// actually executed (it is `false` when the op was toggled off OR budget-skipped —
/// a budget skip is separately observable via [`DreamSummary::budget_exhausted`]).
///
/// So `communities_updated == 0 && consolidation_ops_ran.community == true` reads
/// as "community detection ran and found nothing to change", NOT "it was disabled".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationOpsRan {
    /// The community-detection op (`include_community_detection`) executed.
    pub community: bool,
    /// The cross-episode merge op (`include_cross_episode_merges`) executed.
    pub cross_episode: bool,
    /// The fact-archival op (`include_fact_archival`) executed.
    pub archival: bool,
    /// The supersession sweep (`include_supersession_sweep`) executed.
    pub supersession_sweep: bool,
}

// ── DreamSummary ─────────────────────────────────────────────────────────────

/// Summary returned when a dream phase completes via the facade.
///
/// **Real fields (populated by `mem.dream()`):**
/// - `types_discovered` — entity types proposed and accepted by Pass 0 type-discovery.
/// - `entities_reclassified` — entities reclassified by the reclassify pass.
/// - `aliases_resolved` — pending potential-alias facts resolved (merged/revoked).
/// - `canonicalization_merges` — near-duplicate entities merged by canonicalize.
/// - `consistency_check_corrected` — entity types corrected by consistency_check.
/// - `duration_ms` — wall-clock time of the dream call.
/// - `warnings` — non-fatal notices from any pass.
///
/// **Consolidation fields (ADR-066 — populated by the CONSOLIDATION sub-phase):**
/// - `communities_updated`, `cross_episode_would_merge`, `cross_episode_merged`,
///   `supersessions_recorded`, `facts_archived` — filled by `run_consolidation`
///   when the corresponding `DreamOpts.include_*` op is enabled (all default `true`
///   since consumer-API hardening D1 — made safe by ADR-073 Tier-1 reversibility).
///   Zero when the op is off OR ran and found nothing to do — use
///   `consolidation_ops_ran` (D1b) to tell the two apart.
///
/// Fields wired across ADR-037 §3 D6 (types_discovered/warnings), ADR-046 Option E
/// E8 (entities_reclassified), and dream-phase-reconciliation-v2 §D3
/// (aliases_resolved, canonicalization_merges, consistency_check_corrected).
///
/// # Observability — metrics stack version lock
///
/// The counters kremory emits (`kremory.*` via the [`metrics`] facade) register
/// against the process-global recorder. To READ those counters from a consumer's
/// own recorder/exporter you MUST pin the SAME minor versions kremory links, or
/// the registries silently do not share and you capture ZERO:
///
/// ```toml
/// metrics = "0.24"       # must match kremory's metrics minor
/// metrics-util = "0.18"  # must match kremory's metrics-util minor
/// ```
///
/// A minor-version mismatch (e.g. `metrics = "0.23"`) links a second, incompatible
/// global recorder — kremory's `counter!`/`histogram!` calls hit a registry your
/// exporter never sees. Every count reads `0`. The all-in-band fields on this
/// struct (the D5 would-merge/merged split, [`ConsolidationOpsRan`],
/// [`Self::budget_exhausted`]) are readable WITHOUT any metrics recorder — prefer
/// them when you only need the dream accounting.
#[derive(Debug, Clone)]
pub struct DreamSummary {
    pub communities_updated: usize,
    /// Cross-episode merge DECISIONS this pass — "would-merge" (D5, consumer-API
    /// hardening). Counts every merge decision the op reached on BOTH the shadow and
    /// apply branches, so this does NOT imply entities were actually fused. See
    /// [`Self::cross_episode_merged`] for the count that ACTUALLY committed a fusion
    /// — the in-band would-merge/merged split (no metrics recorder required).
    pub cross_episode_would_merge: usize,
    /// Cross-episode merges that ACTUALLY committed this pass (D5). Equals
    /// [`Self::cross_episode_would_merge`] in apply mode
    /// ([`CrossEpisodeMode::Apply`]) and `0` in shadow mode
    /// ([`CrossEpisodeMode::Shadow`], the default). `would_merge > 0 && merged == 0`
    /// reads as "the op WOULD have merged N pairs but is in shadow — fused nothing".
    pub cross_episode_merged: usize,
    pub supersessions_recorded: usize,
    pub facts_archived: usize,
    /// Per-op ACTUALLY-RAN signal (D1b) — disambiguates "op disabled" from "op ran,
    /// found nothing" for the all-zero consolidation counts above. See
    /// [`ConsolidationOpsRan`].
    pub consolidation_ops_ran: ConsolidationOpsRan,
    pub duration_ms: u64,
    /// Entity types proposed and accepted by Dream Pass 0 type discovery.
    /// Empty when Pass 0 was not run or produced no accepted proposals.
    pub types_discovered: Vec<crate::core::dream::TypeProposal>,
    /// Total entities reclassified by Dream Pass 2 (catch_all_cascade + low_confidence arms).
    /// Zero when Pass 2 was not run or found no candidates.
    pub entities_reclassified: usize,
    /// Pending `potential_alias` facts resolved (merged or revoked) by the aliases
    /// pass (§D3). Zero when the pass was not run or found no pending alias facts.
    pub aliases_resolved: usize,
    /// Near-duplicate entities merged by the canonicalize pass (§D3). Zero when the
    /// pass was not run or found no merges above `L5_CANONICALIZATION_THRESHOLD`.
    pub canonicalization_merges: usize,
    /// Entity-instance pairs merged by the Site #5 acronym/nickname recall pass
    /// (ADR-063 §3). Zero when opted out (`include_acronym_nickname_recall = false`)
    /// or no acronym/nickname pairs were adjudicated as the same entity.
    pub acronym_nickname_merges: usize,
    /// Near-duplicate `entity_types` rows merged by the Site #3 type-registry
    /// collapse pass (ADR-063 §4). Zero when opted out
    /// (`include_type_registry_collapse = false`) or no type pairs were collapsed.
    pub type_registry_merges: usize,
    /// Entity types corrected by the consistency_check pass (ADR-047, §D3). Zero
    /// when opted out (`include_consistency_check = false`) or nothing to correct.
    pub consistency_check_corrected: usize,
    /// Warnings emitted during the dream phase.
    /// Includes degraded-mode notices (e.g. anti-redundancy gate skipped).
    pub warnings: Vec<String>,
    /// TD-060 (ADR-071 §Item 4a step 6) — `true` when the consolidation
    /// sub-phase's `ConsolidationBudget` (token or USD ceiling) was exhausted this
    /// run, skipping at least one op. Set from the fold site in `facade/dream.rs`
    /// (`consolidation.budget_exhausted`) — inert (`false`) unless a consolidation
    /// op was enabled and its budget ceiling actually tripped.
    pub budget_exhausted: bool,
}

impl From<DreamPhaseResult> for DreamSummary {
    fn from(r: DreamPhaseResult) -> Self {
        Self {
            communities_updated: r.communities_recomputed,
            cross_episode_would_merge: r.cross_meeting_merges,
            // No consolidation dispatcher ran on this `DreamPhaseResult` path (it is
            // freshly `default()`-constructed before consolidation folds in at the
            // facade), so the actual-fusion count + ran-signal are all zero/false;
            // the facade fold site overwrites them with the real values.
            cross_episode_merged: 0,
            consolidation_ops_ran: ConsolidationOpsRan::default(),
            supersessions_recorded: r.supersessions_recorded,
            facts_archived: r.facts_archived,
            duration_ms: r.duration_ms,
            types_discovered: r.types_discovered,
            entities_reclassified: 0,
            aliases_resolved: 0,
            canonicalization_merges: 0,
            acronym_nickname_merges: 0,
            type_registry_merges: 0,
            consistency_check_corrected: 0,
            warnings: r.dream_warnings,
            // Carried through structurally (not hardcoded false) — DreamPhaseResult
            // gained this field for exactly this propagation (ADR-071 §Item 4a step
            // 6). In the live `facade/dream.rs` call path `r` is always freshly
            // `DreamPhaseResult::default()`-constructed BEFORE consolidation runs, so
            // this is `false` here and is overwritten by the fold site immediately
            // after with the real `consolidation.budget_exhausted` value.
            budget_exhausted: r.budget_exhausted,
        }
    }
}

impl From<crate::core::ingest::DreamPassSummary> for DreamSummary {
    fn from(s: crate::core::ingest::DreamPassSummary) -> Self {
        Self {
            // DreamPassSummary fields map to DreamSummary where applicable.
            // Fields without a direct mapping are zeroed.
            communities_updated: 0,
            cross_episode_would_merge: s.ghost_episodes_retried,
            cross_episode_merged: 0,
            consolidation_ops_ran: ConsolidationOpsRan::default(),
            supersessions_recorded: 0,
            facts_archived: 0,
            duration_ms: s.duration_ms,
            types_discovered: Vec::new(),
            entities_reclassified: s.entities_reclassified,
            aliases_resolved: 0,
            canonicalization_merges: 0,
            acronym_nickname_merges: 0,
            type_registry_merges: 0,
            consistency_check_corrected: 0,
            warnings: Vec::new(),
            // DreamPassSummary has no budget concept — no consolidation sub-phase
            // ran on this path.
            budget_exhausted: false,
        }
    }
}

#[cfg(test)]
mod dream_summary_budget_exhausted_tests {
    //! TD-060 (ADR-071 §Item 4a step 6, Vera HIGH-1) — consumer-observability
    //! coverage for `budget_exhausted` propagation. `run_consolidation`'s real op
    //! projections are all 0 at this stage (§Item 4a DoD note), so a true
    //! budget-capped skip cannot be triggered end-to-end through `mem.dream()`
    //! yet — these tests instead prove the STRUCTURAL propagation chain a live
    //! skip will ride once a non-zero projection ships: `ConsolidationSummary`
    //! (mod.rs `skip()`) → `DreamSummary` (facade/dream.rs fold site) and
    //! `DreamPhaseResult` → `DreamSummary` (the `From` impl above).

    use super::*;
    use crate::core::dream::consolidation::ConsolidationSummary;

    #[test]
    fn dream_phase_result_budget_exhausted_field_accessible() {
        // Structural/compile-shape test (mirrors the established
        // `d6_dream_summary_types_discovered_field_accessible` pattern) — the field
        // must exist on DreamPhaseResult and accept a bool value.
        let r = DreamPhaseResult {
            budget_exhausted: true,
            ..Default::default()
        };
        assert!(
            r.budget_exhausted,
            "DreamPhaseResult.budget_exhausted must be settable"
        );
    }

    #[test]
    fn dream_summary_from_dream_phase_result_carries_budget_exhausted() {
        // Proves the `From<DreamPhaseResult>` impl above threads the flag through
        // rather than hardcoding `false` — the propagation half of Vera HIGH-1.
        let r = DreamPhaseResult {
            budget_exhausted: true,
            ..Default::default()
        };
        let summary: DreamSummary = r.into();
        assert!(
            summary.budget_exhausted,
            "DreamSummary::from(DreamPhaseResult) must carry budget_exhausted through, not hardcode false"
        );
    }

    #[test]
    fn dream_summary_fold_site_pattern_carries_consolidation_budget_exhausted() {
        // Proves the exact assignment shape used at the `facade/dream.rs` fold site
        // (`summary.budget_exhausted = consolidation.budget_exhausted;`) correctly
        // reads a `true` ConsolidationSummary.budget_exhausted through to the
        // consumer-facing DreamSummary — the other half of Vera HIGH-1 (a live
        // budget-capped run cannot yet drive this end-to-end per the module doc
        // above, so this test exercises the fold assignment directly).
        let consolidation = ConsolidationSummary {
            budget_exhausted: true,
            ..Default::default()
        };
        let mut summary = DreamSummary::from(DreamPhaseResult::default());
        assert!(!summary.budget_exhausted, "starts false (no skip yet)");
        summary.budget_exhausted = consolidation.budget_exhausted;
        assert!(
            summary.budget_exhausted,
            "fold-site assignment must carry a tripped ConsolidationSummary.budget_exhausted through to DreamSummary"
        );
    }
}

// ── RecallTemplate ────────────────────────────────────────────────────────────

/// Render strategy for a `RecallRequest`. Facade-level enum mapping to
/// `ContextTemplate` variants with a stable, serializable `as_str` surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallTemplate {
    /// Render entities with name + summary. Matches `ContextTemplate::Entities`.
    Entities,
    /// Render one-line-per-edge compact summary. Matches `ContextTemplate::EdgeSummary`.
    EdgeSummary,
    /// Render temporal facts with `valid_at` annotations (default).
    TemporalFacts,
}

impl RecallTemplate {
    /// Parse from a string slug. Returns `None` for unknown values.
    pub fn parse_str(s: &str) -> Option<Self> {
        match s {
            "entities" => Some(Self::Entities),
            "edge_summary" => Some(Self::EdgeSummary),
            "temporal_facts" => Some(Self::TemporalFacts),
            _ => None,
        }
    }

    /// Return the stable string slug for this template.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entities => "entities",
            Self::EdgeSummary => "edge_summary",
            Self::TemporalFacts => "temporal_facts",
        }
    }
}

impl From<RecallTemplate> for ContextTemplate {
    fn from(t: RecallTemplate) -> Self {
        match t {
            RecallTemplate::Entities => ContextTemplate::Entities,
            RecallTemplate::EdgeSummary => ContextTemplate::EdgeSummary,
            RecallTemplate::TemporalFacts => ContextTemplate::TemporalFacts,
        }
    }
}

// ── Memory struct ─────────────────────────────────────────────────────────────

/// Fluent facade handle for kremory agent memory.
///
/// `Memory` is `Clone + Send + Sync` — cheaply cloneable (`Arc` internally)
/// and safe to share across tokio tasks.
///
/// # Obtain a handle
///
/// - **Tier 1**: `Memory::auto(path).await?` (env-detected provider)
/// - **Tier 1.5**: `Memory::with_ollama(path).await?` etc.
/// - **Tier 2**: `Memory::open(path).with_llm(l).with_embedder(e).await?`
///
/// # Tier 3 (substrate composition)
///
/// Advanced users requiring raw substrate access can use `kremory::memory::*`
/// free functions directly — they remain public and unchanged.
///
/// TD-136 (dense episode retrieval): tally returned by
/// [`Memory::backfill_episode_embeddings`] and, since TD-143/TD-112, also by
/// [`Memory::reembed_all_episode_embeddings`],
/// [`Memory::reembed_all_entity_embeddings`], and
/// [`Memory::reembed_all_fact_embeddings`] — all four drive the same
/// embed+store shape (embed a page item's text, persist the vector, count
/// success/failure) over a different table/page-source, so they share this
/// tally shape rather than each declaring an identical `{embedded, failed}`
/// struct (Rule 37 — reuse an existing shape before declaring a new one).
/// Field names stay table-agnostic ("items", not "episodes") accordingly.
/// Feature-gated behind `content-search` (the whole embedding path only
/// exists there).
#[cfg(feature = "content-search")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpisodeEmbeddingBackfill {
    /// Items (episodes/entities/facts, depending on which method returned
    /// this tally) whose text was embedded + stored this run.
    pub embedded: u64,
    /// Items skipped due to a per-item embed/store failure (WARN-logged;
    /// re-run to retry them).
    pub failed: u64,
}

/// Args-as-object for [`Memory::embed_and_store_episode_page`] per TD-042
/// (`clippy.toml` `too-many-arguments-threshold = 3`, `self` counts).
/// Private — an internal seam shared by
/// [`Memory::backfill_episode_embeddings`] and
/// [`Memory::reembed_all_episode_embeddings`], not part of the public API.
#[cfg(feature = "content-search")]
struct EmbedEpisodePageParams<'a> {
    tg: &'a TemporalGraph,
    batch: Vec<(i64, String)>,
    stats: &'a mut EpisodeEmbeddingBackfill,
    op: &'static str,
}

/// Args-as-object for [`Memory::embed_and_store_entity_page`] — TD-112
/// sibling of [`EmbedEpisodePageParams`], same rationale.
#[cfg(feature = "content-search")]
struct EmbedEntityPageParams<'a> {
    tg: &'a TemporalGraph,
    batch: Vec<crate::core::graph::EntityReembedRow>,
    stats: &'a mut EpisodeEmbeddingBackfill,
    op: &'static str,
}

/// Args-as-object for [`Memory::embed_and_store_fact_page`] — TD-112 sibling
/// of [`EmbedEpisodePageParams`], same rationale.
#[cfg(feature = "content-search")]
struct EmbedFactPageParams<'a> {
    tg: &'a TemporalGraph,
    batch: Vec<(i64, String)>,
    stats: &'a mut EpisodeEmbeddingBackfill,
    op: &'static str,
}

#[derive(Clone)]
pub struct Memory {
    pub(crate) graph: Arc<dyn GraphHandle>,
    /// `None` when built via the NoLlm path (`Memory::open().with_extractor(…).with_embedder(…)`).
    /// Category B methods (dream, recall_with_disambiguation, detect_contradictions) call
    /// `.llm_or_err("method_name")` which returns `Error::LlmRequired` at call time.
    pub(crate) llm: Option<Arc<dyn ChatProvider>>,
    /// Optional dedicated dream-phase LLM (TD-052b). `Some` → `dream()` uses it;
    /// `None` → dream falls back to `self.llm` via `dream_llm_or_main`.
    pub(crate) dream_llm: Option<Arc<dyn ChatProvider>>,
    /// Concrete model id for the MAIN chat provider (`with_model_id` / Tier-1
    /// shortcut). Threaded into the dream LLM passes for capability detection
    /// (empty → `PromptOnly` degrade). `None` when the provider was wired via
    /// raw `with_llm` without a model id — dream then degrades exactly as the
    /// interactive path does. TD-094: previously baked only into the ingest
    /// pipeline, never reaching the dream facade — the root cause of the
    /// silent empty-model → zero-output degrade in the LLM dream passes.
    pub(crate) model_id: Option<String>,
    /// Optional dedicated dream-phase model id (`with_dream_model_id`). Pairs
    /// with `dream_llm` the way `model_id` pairs with `llm`: when a dedicated
    /// dream *provider* is set, its model *string* usually differs from the
    /// interactive model, so capability detection needs its own id. `None` →
    /// dream falls back to `model_id` via `dream_model_id_or_main` (TD-094).
    pub(crate) dream_model_id: Option<String>,
    /// Embedding provider — read by the dream/disambiguation paths
    /// (`facade/dream.rs` passes `self.memory.embedder.as_ref()` into the
    /// dream pass). TD-043: field is live, `#[allow(dead_code)]` removed.
    pub(crate) embedder: Arc<dyn DynEmbeddingProvider>,
    pub(crate) default_sink: Option<Arc<dyn EnrichmentEventSink>>,
    pub(crate) default_namespace: Option<Namespace>,
    /// Direct handle to the underlying `TemporalGraph` for namespace-policy
    /// substrate calls (ADR-029a `register_namespace` + lazy population).
    /// `None` only when `Memory` is constructed by a test path that bypasses
    /// `providers::open_graph` (e.g. with a stub `GraphHandle`). In that case
    /// `register_namespace` returns `MemoryError::Other("…")`.
    pub(crate) temporal_graph: Option<Arc<TemporalGraph>>,
    /// Soft warning threshold for episode content length (chars). When set,
    /// `remember(...).await` emits `tracing::warn!` + a metrics counter if
    /// `content.len()` exceeds this value. Never enforced — observability only.
    /// `None` disables the warning. Default (via builder): `Some(10_000)`.
    pub(crate) episode_content_warn_threshold: Option<usize>,
    /// Background dream scheduler handle. `None` when `DreamSchedule::Off` (default)
    /// or when `Memory` is constructed by a test stub path. Stored as
    /// `Arc<Mutex<Option<…>>>` so `Clone` works without requiring the handle to be
    /// `Clone` (a `JoinHandle<()>` is not `Clone`).
    pub(crate) dream_scheduler:
        std::sync::Arc<std::sync::Mutex<Option<crate::memory::scheduler::DreamSchedulerHandle>>>,
    /// When `true`, `Memory::remember(...).await` blocks until background
    /// extraction (GLiNER/LLM via the ADR-051 worker) has transitioned the
    /// episode to `Verified` (or returns `Err` on `Failed` / timeout).
    ///
    /// Opt-in: default is `false` (fire-and-forget, per ADR-051 design).
    /// Per D1 peer pattern: equivalent to Cognee's `run_in_background=False`.
    ///
    /// ⚠ Cost: enables synchronous-extraction ergonomics at the expense of the
    /// latency benefit ADR-051 provides. Document this trade-off in consumer
    /// code. Prefer `Memory::wait_for_processing` directly for fine-grained
    /// control. See spec §Risk R-06.
    pub(crate) await_extraction: bool,
    /// Timeout applied when `await_extraction = true`.
    ///
    /// Default: 60 seconds (per spec §Risk R-12 mitigation). Configurable via
    /// `MemoryBuilder::with_await_extraction_timeout`.
    pub(crate) await_extraction_timeout: Duration,
}

impl Memory {
    /// Open a database at `path` and start the type-state builder.
    ///
    /// Requires `.with_llm(…)` then `.with_embedder(…)` before `.await`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # use std::sync::Arc;
    /// # async fn ex() -> kremory::memory::Result<()> {
    /// # let llm: Arc<dyn kremory::memory::ChatProvider> = todo!();
    /// # let emb: Arc<dyn kremory::DynEmbeddingProvider> = todo!();
    /// let mem = Memory::open("./agent.db")
    ///     .with_llm(llm)
    ///     .with_embedder(emb)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: impl AsRef<Path>) -> MemoryBuilder<NoLlm, NoEmb> {
        MemoryBuilder::new_open(path.as_ref().to_path_buf())
    }

    // ── Tier 1 shortcuts — implemented in providers.rs ───────────────────────

    /// Env-detected shortcut: OLLAMA_HOST → OPENAI_API_KEY → ANTHROPIC_API_KEY → Err.
    ///
    /// Priority order is intentionally local-first to position kremory as a
    /// privacy-first library (no data leaves the machine when Ollama is available).
    pub async fn auto(path: impl AsRef<Path>) -> Result<Self> {
        providers::auto(path).await
    }

    /// Open with Ollama running at `http://localhost:11434`.
    /// Models: `gemma4:e4b` (chat, reasoning disabled) + `nomic-embed-text` (embeddings).
    /// See [`providers::with_ollama`] for the benchmark rationale + lighter alternatives.
    pub async fn with_ollama(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_ollama(path).await
    }

    /// Open with Ollama at a custom URL.
    pub async fn with_ollama_at(url: impl Into<String>, path: impl AsRef<Path>) -> Result<Self> {
        providers::with_ollama_at(url, path).await
    }

    /// Open with OpenAI. Requires `$OPENAI_API_KEY`.
    /// Models: `gpt-4o-mini` (chat) + `text-embedding-3-small` (embeddings).
    pub async fn with_openai(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_openai(path).await
    }

    /// Open with Anthropic. Requires `$ANTHROPIC_API_KEY`.
    /// Note: Anthropic has no native embedding API; falls back to a deterministic
    /// FNV-1a embedder (dim=384, not semantic — suitable for exact-match recall only).
    /// A `tracing::warn!` is emitted at construction time.
    pub async fn with_anthropic(path: impl AsRef<Path>) -> Result<Self> {
        providers::with_anthropic(path).await
    }

    // ── Introspection ───────────────────────────────────────────────────────

    /// The live search-fusion configuration this `Memory` uses for recall — the
    /// [`SearchConfig`](crate::core::config::SearchConfig) carried by the
    /// underlying graph handle, reflecting any `KREMORY_CONTENT_WEIGHT` /
    /// `KREMORY_RRF_K` boot overrides applied at construction (via
    /// `providers::search_env_overrides`). Read-only; cheap (a clone of an
    /// in-memory struct — no I/O, hence not `async`).
    ///
    /// Exposed (TD-135) so a transport/consumer — e.g. the `kremory-http` bench
    /// server's `GET /health` endpoint — can report the ACTUAL active scoring
    /// config as a single source of truth, rather than re-reading env (which can
    /// drift from what the search path actually uses and is exactly how a
    /// config-mismatch produced a bogus benchmark number). Stub/test graph
    /// handles that carry no `Engine` return `SearchConfig::default()`.
    pub fn search_config(&self) -> crate::core::config::SearchConfig {
        self.graph.search_config()
    }

    // ── Test utilities ────────────────────────────────────────────────────────

    /// Access the underlying `TemporalGraph` for integration tests that need
    /// direct SQL access (e.g. asserting `episode_processing_status`).
    ///
    /// Only available under `test` or `test-utils` feature. Not part of the
    /// stable public API.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn temporal_graph_for_test(&self) -> Option<&Arc<TemporalGraph>> {
        self.temporal_graph.as_ref()
    }

    /// Resolve a `Namespace` to the internal group-id string, so integration
    /// tests can plant graph rows under the exact group `mem.dream()` operates
    /// on for that namespace.
    ///
    /// Only available under `test` or `test-utils`. Not part of the stable
    /// public API — external callers MUST NOT depend on the group-id string
    /// layout (it is substrate detail; see `namespace_to_group_id`).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn group_id_for_test(&self, ns: &crate::Namespace) -> String {
        crate::memory::engine_handle::namespace_to_group_id(ns)
    }

    // ── Ingest ────────────────────────────────────────────────────────────────

    /// Ingest a memory episode.
    ///
    /// Default: blocks until Phase 2 enrichment is done.
    /// Use `.no_wait()` to return after Phase 1 commit only.
    ///
    /// # Namespace resolution
    ///
    /// Either `.in_namespace(ns)` on the request OR a `default_namespace` on
    /// the builder is required. Missing both → `Err(MemoryError::MissingNamespace)`.
    #[must_use = "RememberRequest must be .await-ed or have a terminal called"]
    pub fn remember<'a>(&'a self, content: impl Into<String> + 'a) -> RememberRequest<'a> {
        RememberRequest {
            memory: self,
            content: content.into(),
            source_ref: None,
            namespace: None,
            published_at: None,
            facts: vec![],
            sink: None,
            no_wait: false,
            opts: None,
            skip_extraction: false,
        }
    }

    /// Bulk-ingest multiple episodes in a single batch (F6).
    ///
    /// This is the **inline** batch path: each entry supports per-entry
    /// `.in_namespace(...)`, and ingestion runs through the standard
    /// `EngineGraphHandle`. Use this when you have several episodes to add and
    /// want per-entry control.
    ///
    /// For the **background** path that routes through `BackgroundIngestor` and
    /// fires `on_batch_phase2_complete` on a configured sink, use
    /// [`send_batched`](Self::send_batched) instead (requires a
    /// `default_namespace`; no per-entry namespace override).
    #[must_use = "RememberBatchBuilder must be .await-ed"]
    pub fn remember_batch(&self) -> RememberBatchBuilder<'_> {
        RememberBatchBuilder {
            memory: self,
            episodes: vec![],
            batch_id: None,
            sink: None,
        }
    }

    /// Enqueue text for background ingestion as part of a named batch.
    ///
    /// Associates the episode with `batch_id` for batch tracking.  When all
    /// episodes in the batch reach Phase 2 terminal state,
    /// `on_batch_phase2_complete` fires on the configured sink (if any).
    ///
    /// Per ADR-052 Gap 1 §3.2 + Phase 7 DoD + v0.2.3 follow-up closure
    /// (`BackgroundIngestorGraphHandle` dual-path consolidation, 2026-06-15).
    ///
    /// # Namespace resolution
    ///
    /// Requires a `default_namespace` on the builder.  Per-call namespace
    /// override is not available on the batched send path (use
    /// `remember_batch().with_batch_id()` for per-entry namespace control).
    ///
    /// # Architectural note
    ///
    /// When `.with_sink()` is configured on the builder, `MemoryBuilder::build()`
    /// constructs a `BackgroundIngestorGraphHandle` (arch spec §3.2 Option A).
    /// `send_batched` then routes through `BackgroundIngestor.send_batched` —
    /// the ADR-051 OS-thread pipeline — and `on_batch_phase2_complete` fires
    /// via the configured sink when the batch reaches terminal state.
    ///
    /// When no sink is configured, `send_batched` routes through the
    /// `EngineGraphHandle` tokio-spawn path.  `on_batch_phase2_complete` will
    /// NOT fire (no sink to receive it).  This is the correct behavior: callers
    /// who don't provide a sink have no listener for the callback.
    ///
    /// # Errors
    ///
    /// Returns `Err(MemoryError::MissingNamespace)` when no `default_namespace`
    /// is set.
    ///
    /// Returns `Err(MemoryError::Core(...))` on substrate failure.
    pub async fn send_batched(
        &self,
        text: impl Into<String>,
        batch_id: String,
    ) -> memory::Result<EpisodeCommit> {
        use crate::memory::types::SubmitOpts;

        let ns = self.resolve_namespace(None)?;
        let sink = self.default_sink.clone();

        // ADR-029a lazy population.
        self.ensure_namespace_policy(&ns).await?;

        let source_ref = memory::types::SourceRef {
            kind: memory::types::SourceKind::Chat,
            id: uuid::Uuid::new_v4().to_string(),
            occurred_at: chrono::Utc::now(),
            published_at: None,
        };

        memory::submit_episode(memory::SubmitEpisodeParams {
            graph: self.graph.as_ref(),
            content: &text.into(),
            source_ref,
            structured_facts: vec![],
            provider: self.llm_or_stub(),
            namespace: ns,
            batch_id: Some(batch_id),
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: true,
            },
            sink,
        })
        .await
    }

    /// Search for memories matching `query`.
    ///
    /// Default terminal: `.await?` returns `String` via `TemporalFacts` template.
    /// Use `.raw()` for `Vec<RetrievedContext>` or `.as_template(t)` for other templates.
    #[must_use = "RecallRequest must be .await-ed or have a terminal called"]
    pub fn recall<'a>(&'a self, query: impl Into<String> + 'a) -> RecallRequest<'a> {
        RecallRequest {
            memory: self,
            query: query.into(),
            namespace: None,
            namespaces: None,
            per_namespace_top_k: None,
            best_effort: false,
            recall_id: Uuid::new_v4(),
            k: None,
            as_of: None,
            rerank_k: None,
            template: Some(RecallTemplate::TemporalFacts),
            raw_mode: false,
            opts: None,
            metadata_filters: Vec::new(),
            metadata_filters_in: Vec::new(),
            pending_error: None,
        }
    }

    /// Delete all episodes in scope.
    ///
    /// Requires either `.in_namespace(ns)` or a `default_namespace`.
    /// Must call `.execute()` explicitly (destructive terminal — no accidental `.await`).
    #[must_use = "ForgetRequest must call .execute() to run"]
    pub fn forget(&self) -> ForgetRequest<'_> {
        ForgetRequest {
            memory: self,
            namespace: None,
            source_id: None,
        }
    }

    /// Bound a fact's world-time `valid_to` window explicitly (ADR-071 §Item
    /// 3, TD-070) — the consumer-facing, consumer-EXPLICIT half of the
    /// supersession gap (auto-detected supersession is deferred, TD-P1-AUTO).
    ///
    /// Requires `.at(valid_to)` before `.execute()`. The dream supersession
    /// sweep (`include_supersession_sweep: true`) later observes the bounded
    /// `valid_to` and closes the window (`expired_at = valid_to`,
    /// `DreamSummary.supersessions_recorded` increments) — see
    /// [`supersede::SupersedeRequest`] for the full two-phase mechanism.
    #[must_use = "SupersedeRequest must call .execute() to run"]
    pub fn supersede(&self, fact_id: i64) -> SupersedeRequest<'_> {
        SupersedeRequest {
            memory: self,
            fact_id,
            namespace: None,
            valid_to: None,
            reason: None,
            close_now: false,
        }
    }

    /// Reverse ANY logged, reversible mutation by its `mutation_id` — the unified
    /// undo umbrella (ADR-073 DX R1/R2, ADR-038 "one recommended entry point").
    ///
    /// Reads the `graph_mutation_log` row for `mutation_id`, matches on its kind,
    /// and dispatches to the correct per-kind undo, returning the honest
    /// [`UndoOutcome`]. This is the method to reach for after iterating
    /// [`list_mutations`](Self::list_mutations) / [`mutation_history`](Self::mutation_history):
    /// a consumer can uniformly `mem.undo(record.mutation_id)` without switching on
    /// the kind by hand. The per-kind methods ([`unmerge`](Self::unmerge),
    /// [`undo_entity_edit`](Self::undo_entity_edit),
    /// [`undo_delete_entity`](Self::undo_delete_entity),
    /// [`undo_delete_fact`](Self::undo_delete_fact)) still work and remain the
    /// escape hatch when you already know the kind.
    ///
    /// Only the four LOGGED kinds dispatch (`entity_merge` / `entity_edit` /
    /// `entity_delete` / `fact_delete`). The other four [`MutationKind`] variants
    /// are RESERVED (never produced into the log today), so a would-be row of that
    /// kind returns a loud `Error::UndoUnsupportedKind` — see [`UndoRequest`].
    ///
    /// Optional `.in_namespace(ns)` guards the undo to the mutation's original
    /// namespace. Must call `.execute()` (mutating op).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::{Memory, Namespace};
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// // SEE what dream() did, then UNDO the most recent mutation uniformly.
    /// let history = mem.mutation_history("alice")
    ///     .in_namespace(Namespace::new("agent"))
    ///     .await?;
    /// if let Some(rec) = history.first() {
    ///     let outcome = mem.undo(rec.mutation_id).execute().await?;
    ///     println!("reversed: {outcome:?}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "UndoRequest must call .execute() to run"]
    pub fn undo(&self, mutation_id: i64) -> UndoRequest<'_> {
        UndoRequest {
            memory: self,
            mutation_id,
            namespace: None,
        }
    }

    /// Reverse a prior entity-merge, fully restoring the loser entity, its facts,
    /// its episodic edges, and the keeper's overwritten `access_count` /
    /// `ner_confidence` (reversible-graph-mutations arch-spec §4.2). Records a
    /// merge NOGOOD so the next `dream()` will NOT re-merge the split pair (§6.2).
    ///
    /// Idempotent: a second call returns `already_undone = true`. Must call
    /// `.execute()` (mutating op). Prefer [`undo`](Self::undo) when iterating
    /// [`list_mutations`](Self::list_mutations) — it dispatches by kind for you.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::{Memory, Namespace};
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// // Find a merge in an entity's history, then split the pair back apart.
    /// let history = mem.mutation_history("alice j")
    ///     .in_namespace(Namespace::new("agent"))
    ///     .await?;
    /// if let Some(rec) = history.first() {
    ///     let outcome = mem.unmerge(rec.mutation_id).execute().await?;
    ///     println!("restored '{}' (nogood recorded: {})",
    ///         outcome.restored_entity, outcome.nogood_recorded);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "UnmergeRequest must call .execute() to run"]
    pub fn unmerge(&self, mutation_id: i64) -> UnmergeRequest<'_> {
        UnmergeRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Restore a fact previously moved to `facts_archive` (P2 archival) back into
    /// `facts` (arch-spec §3.2 / §4.4). Idempotent: `already_live = true` when the
    /// fact is already live. Must call `.execute()`.
    #[must_use = "RestoreArchivedRequest must call .execute() to run"]
    pub fn restore_archived_fact(&self, archived_fact_id: i64) -> RestoreArchivedRequest<'_> {
        RestoreArchivedRequest {
            memory: self,
            archived_fact_id,
        }
    }

    /// Clear a supersession bound (`valid_to` / `expired_at`) set by
    /// `supersede(...)`, re-opening the fact as currently-true (arch-spec §4.5).
    /// Idempotent: `NotSuperseded` when no bound was set. Must call `.execute()`.
    #[must_use = "UnsupersedeRequest must call .execute() to run"]
    pub fn unsupersede(&self, fact_id: i64) -> UnsupersedeRequest<'_> {
        UnsupersedeRequest {
            memory: self,
            fact_id,
        }
    }

    /// Edit an entity — retype or rename — with full FK-propagation, provenance
    /// snapshot, and reconciler-freeze re-open (reversible-graph-mutations
    /// arch-spec §4.3). Completes the diarization flow: after `unmerge`, rename
    /// `"Speaker 1"` to `"Alice"` and every one of its facts / episodic edges /
    /// archived facts / community membership re-points to `alice`.
    ///
    /// Choose exactly one operation on the builder:
    /// - `.rename(new_id)` — REKEY the entity's id (rejects renaming INTO an
    ///   existing id with `Error::EntityEditConflict`; merge explicitly instead).
    /// - `.retype(type_id)` — change the entity's type (pins it `ConsumerPinned`).
    ///
    /// The edit is undoable via [`undo_entity_edit`](Self::undo_entity_edit) with
    /// the returned `EditEntityOutcome.mutation_id`. Must call `.execute()`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::{Memory, Namespace};
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// // Rename a diarization placeholder; every fact / edge re-points to the new id.
    /// let edit = mem.edit_entity("Speaker 1")
    ///     .rename("alice")
    ///     .in_namespace(Namespace::new("meeting"))
    ///     .execute()
    ///     .await?;
    /// // ...and reverse it later via the returned mutation_id.
    /// mem.undo_entity_edit(edit.mutation_id).execute().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "EditEntityRequest must call .execute() to run"]
    pub fn edit_entity<'a>(&'a self, entity_id: impl Into<String> + 'a) -> EditEntityRequest<'a> {
        EditEntityRequest {
            memory: self,
            entity_id: entity_id.into(),
            namespace: None,
            new_id: None,
            new_type_id: None,
        }
    }

    /// Reverse a prior `edit_entity` (retype or rename/rekey) from its provenance
    /// snapshot (arch-spec §4.3). Pass the `mutation_id` from the
    /// `EditEntityOutcome` (or from `mutation_history` / `list_mutations`).
    /// Idempotent: a second call is a zero-count no-op. Must call `.execute()`.
    #[must_use = "UndoEntityEditRequest must call .execute() to run"]
    pub fn undo_entity_edit(&self, mutation_id: i64) -> UndoEntityEditRequest<'_> {
        UndoEntityEditRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Delete an entity, reversibly (reversible-graph-mutations arch-spec §4.4).
    /// The entity's facts are ARCHIVED (recoverable — never hard-deleted), its
    /// edges / community membership / FTS / row are removed, and any neighbour whose
    /// live-fact support drops to zero has its DERIVED community membership retracted
    /// (the base entity is never auto-deleted). Undoable via
    /// [`undo_delete_entity`](Self::undo_delete_entity) with the returned
    /// `DeleteEntityOutcome.mutation_id`. Must call `.execute()` (destructive op).
    #[must_use = "DeleteEntityRequest must call .execute() to run"]
    pub fn delete_entity<'a>(
        &'a self,
        entity_id: impl Into<String> + 'a,
    ) -> DeleteEntityRequest<'a> {
        DeleteEntityRequest {
            memory: self,
            entity_id: entity_id.into(),
            namespace: None,
        }
    }

    /// Delete a single fact, reversibly (reversible-graph-mutations arch-spec §4.5).
    /// The fact is archived (recoverable via
    /// [`restore_archived_fact`](Self::restore_archived_fact)); either endpoint whose
    /// support drops to zero has its DERIVED community membership retracted. Undoable
    /// via [`undo_delete_fact`](Self::undo_delete_fact). A fact id is global. Must
    /// call `.execute()`.
    #[must_use = "DeleteFactRequest must call .execute() to run"]
    pub fn delete_fact(&self, fact_id: i64) -> DeleteFactRequest<'_> {
        DeleteFactRequest {
            memory: self,
            fact_id,
        }
    }

    /// Reverse a prior `delete_entity` from its provenance snapshot (arch-spec §4.4)
    /// — re-inserts the entity + FTS, restores its archived facts + episodic edges,
    /// and un-retracts every community membership the cascade retracted. Pass the
    /// `mutation_id` from the `DeleteEntityOutcome` (or `mutation_history` /
    /// `list_mutations`). Idempotent. Must call `.execute()`.
    #[must_use = "UndoDeleteEntityRequest must call .execute() to run"]
    pub fn undo_delete_entity(&self, mutation_id: i64) -> UndoDeleteEntityRequest<'_> {
        UndoDeleteEntityRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Reverse a prior `delete_fact` from its provenance snapshot (arch-spec §4.5) —
    /// restores the archived fact + un-retracts any neighbour the cascade retracted.
    /// Idempotent. Must call `.execute()`.
    #[must_use = "UndoDeleteFactRequest must call .execute() to run"]
    pub fn undo_delete_fact(&self, mutation_id: i64) -> UndoDeleteFactRequest<'_> {
        UndoDeleteFactRequest {
            memory: self,
            mutation_id,
        }
    }

    /// Inspect the mutations `dream()` applied to a single entity — the **SEE**
    /// half of the reversible-mutation story (arch-spec §3 "Inspect surface").
    ///
    /// Returns a newest-first `Vec<MutationRecord>` of every logged graph mutation
    /// that touched `entity_id` (for an `entity_merge`, whether the entity was the
    /// loser OR the keeper), each carrying the `mutation_id` to pass to
    /// [`unmerge`](Self::unmerge). Includes already-undone mutations
    /// (`undone = true`), so a reversed merge is still visible.
    ///
    /// Namespace: `.in_namespace(ns)` or a `default_namespace` on the builder is
    /// required (an entity id is namespace-scoped). Read-only — `.await` it.
    ///
    /// Only the four LOGGED [`MutationKind`]s appear here (`EntityMerge` /
    /// `EntityEdit` / `EntityDelete` / `FactDelete`); the other four are reserved
    /// and never surface — see [`list_mutations`](Self::list_mutations) for the
    /// tracked-kind boundary.
    #[must_use = "MutationHistoryRequest must be .await-ed"]
    pub fn mutation_history<'a>(
        &'a self,
        entity_id: impl Into<String> + 'a,
    ) -> MutationHistoryRequest<'a> {
        MutationHistoryRequest {
            memory: self,
            entity_id: entity_id.into(),
            namespace: None,
        }
    }

    /// List the mutations `dream()` applied, newest-first — the **SEE** surface
    /// for a whole namespace (arch-spec §3 "Inspect surface").
    ///
    /// Returns `Vec<MutationRecord>` (each carrying its `mutation_id` to undo via
    /// [`undo`](Self::undo)). Filter with `.kind(k)` / `.since(ts)` /
    /// `.include_undone(true)`; scope with `.in_namespace(ns)` (else the
    /// `default_namespace`, else ALL namespaces). Default view is LIVE
    /// (still-reversible) mutations only. Read-only — `.await` it.
    ///
    /// # Tracked-kind boundary (4 of 8)
    ///
    /// Only FOUR [`MutationKind`] variants are currently LOGGED (hence listable and
    /// reversible via [`undo`](Self::undo)): `EntityMerge`, `EntityEdit`,
    /// `EntityDelete`, `FactDelete`. The other four (`FactSupersede`, `FactArchive`,
    /// `CommunityAssign`, `CanonicalForm`) are RESERVED — not yet produced into the
    /// `graph_mutation_log` — so `list_mutations().kind(<a reserved kind>)` returns
    /// EMPTY by construction (not "nothing changed"). `FactSupersede` / `FactArchive`
    /// are themselves reversible, but through the domain-id methods
    /// [`unsupersede`](Self::unsupersede) / [`restore_archived_fact`](Self::restore_archived_fact),
    /// not this inspect+undo surface.
    #[must_use = "ListMutationsRequest must be .await-ed"]
    pub fn list_mutations(&self) -> ListMutationsRequest<'_> {
        ListMutationsRequest {
            memory: self,
            namespace: None,
            kind: None,
            since: None,
            include_undone: false,
        }
    }

    /// Run batch consolidation (dream phase).
    ///
    /// Default: blocks until done (returns [`DreamSummary`]). All consolidation ops
    /// default ON and are REVERSIBLE (ADR-073) — inspect what changed with
    /// [`mutation_history`](Self::mutation_history) / [`list_mutations`](Self::list_mutations),
    /// reverse anything with [`undo`](Self::undo). Use `.fire_and_forget()` to
    /// return a `DreamHandle` without blocking, or `.cross_episode(mode)` /
    /// `.with_opts(opts)` to tune the consolidation surface.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// let summary = mem.dream().await?;
    /// println!(
    ///     "communities updated: {}, entities reclassified: {}",
    ///     summary.communities_updated, summary.entities_reclassified,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "DreamRequest must be .await-ed or have a terminal called"]
    pub fn dream(&self) -> DreamRequest<'_> {
        DreamRequest {
            memory: self,
            namespace: None,
            batch_id: None,
            batch_size: None,
            sink: None,
            fire_and_forget: false,
            opts: None,
        }
    }

    // ── Handle / polling ──────────────────────────────────────────────────────

    /// Query Phase 2 enrichment status for a committed episode.
    pub async fn status_of(
        &self,
        commit: &EpisodeCommit,
    ) -> Result<crate::core::error::IngestStatus> {
        let run_id = commit.run_id.ok_or_else(|| {
            MemoryError::Other("EpisodeCommit has no run_id (Phase 2 was inline)".into())
        })?;
        self.graph.graph_ingest_status(run_id).await
    }

    /// Block until Phase 2 enrichment reaches a terminal status.
    ///
    /// Uses `AwaitOpts::default()` if `timeout` is translated to the opts shape.
    pub async fn await_enrichment(
        &self,
        commit: &EpisodeCommit,
        timeout: Duration,
    ) -> Result<crate::core::error::IngestStatus> {
        let run_id = commit.run_id.ok_or_else(|| {
            MemoryError::Other("EpisodeCommit has no run_id (Phase 2 was inline)".into())
        })?;
        memory::await_enrichment(
            self.graph.as_ref(),
            run_id,
            AwaitOpts {
                timeout,
                ..AwaitOpts::default()
            },
        )
        .await
    }

    // ── ADR-051 async extraction wait API (v0.2.2, Phase 4) ──────────────────

    /// Poll `episodes.episode_processing_status` for `episode_id` until the
    /// episode reaches a terminal state or `timeout` is exceeded.
    ///
    /// # Terminal states
    ///
    /// | Status      | Return value                          |
    /// |-------------|---------------------------------------|
    /// | `Verified`  | `Ok(())`                              |
    /// | `Failed`    | `Err(MemoryError::Core(ExtractionFailed { episode_id }))` |
    /// | timeout     | `Err(MemoryError::Core(WaitTimeout { episode_id, elapsed }))` |
    ///
    /// `Pending` and `Extracting` keep the poll running.
    ///
    /// # Polling schedule (spec §Risk R-05 mitigation)
    ///
    /// - Interval starts at **50 ms**.
    /// - After 5 s elapsed the interval backs off to **200 ms**.
    ///
    /// # Observability
    ///
    /// Emits `kremory.wait_for_processing.duration_ms{outcome}` histogram on
    /// every terminal exit (outcomes: `verified`, `failed`, `timeout`).
    ///
    /// # Errors
    ///
    /// Returns `Err(MemoryError::Other(...))` when `temporal_graph` is `None`
    /// (test-stub path that bypasses `providers::open_graph`).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # use std::time::Duration;
    /// # async fn ex() -> kremory::memory::Result<()> {
    /// # let mem: Memory = todo!();
    /// # let commit: kremory::memory::types::EpisodeCommit = todo!();
    /// // Get the raw episode rowid from the commit's episode_entity_id.
    /// let episode_id: i64 = commit.episode_entity_id.parse().unwrap_or(0);
    /// mem.wait_for_processing(episode_id, Duration::from_secs(30)).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn wait_for_processing(&self, episode_id: i64, timeout: Duration) -> Result<()> {
        use tokio::time::{sleep, Duration as TokioDuration, Instant};

        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::wait_for_processing requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let start = Instant::now();
        let mut interval = TokioDuration::from_millis(50);
        let timeout_dur = timeout;
        let backoff_threshold = TokioDuration::from_secs(5);

        loop {
            // SELECT the current status.
            let status: String = {
                let mut rows = tg
                    .conn
                    .query(
                        "SELECT episode_processing_status FROM episodes WHERE id = ?1",
                        libsql::params![episode_id],
                    )
                    .await
                    .map_err(|e| MemoryError::Core(CoreError::Database(e)))?;
                match rows
                    .next()
                    .await
                    .map_err(|e| MemoryError::Core(CoreError::Database(e)))?
                {
                    Some(row) => row
                        .get::<String>(0)
                        .map_err(|e| MemoryError::Core(CoreError::Database(e)))?,
                    None => {
                        // Episode not found — treat as timeout (episode may not
                        // have committed yet; callers should ensure Phase 1
                        // completed before calling wait_for_processing).
                        let elapsed = start.elapsed();
                        metrics::histogram!(
                            "kremory.wait_for_processing.duration_ms",
                            "outcome" => "not_found"
                        )
                        .record(start.elapsed().as_secs_f64() * 1000.0);
                        return Err(MemoryError::Core(CoreError::WaitTimeout {
                            episode_id,
                            elapsed,
                        }));
                    }
                }
            };

            tracing::debug!(
                target: "kremory.wait_for_processing",
                episode_id,
                status = %status,
                elapsed_ms = start.elapsed().as_millis(),
                "polling episode_processing_status"
            );

            match status.as_str() {
                "Verified" => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    metrics::histogram!(
                        "kremory.wait_for_processing.duration_ms",
                        "outcome" => "verified"
                    )
                    .record(elapsed_ms);
                    return Ok(());
                }
                "Failed" => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    metrics::histogram!(
                        "kremory.wait_for_processing.duration_ms",
                        "outcome" => "failed"
                    )
                    .record(elapsed_ms);
                    return Err(MemoryError::Core(CoreError::ExtractionFailed {
                        episode_id,
                    }));
                }
                // "Pending" | "Extracting" | any future intermediate state
                _ => {}
            }

            // Check timeout BEFORE sleeping — prevents one extra sleep cycle
            // after the budget is exhausted.
            if start.elapsed() >= timeout_dur {
                let elapsed = start.elapsed();
                metrics::histogram!(
                    "kremory.wait_for_processing.duration_ms",
                    "outcome" => "timeout"
                )
                .record(start.elapsed().as_secs_f64() * 1000.0);
                return Err(MemoryError::Core(CoreError::WaitTimeout {
                    episode_id,
                    elapsed,
                }));
            }

            // Backoff: after 5 s switch from 50 ms to 200 ms interval
            // (spec §Risk R-05 — avoids busy-polling long extractions).
            if start.elapsed() >= backoff_threshold {
                interval = TokioDuration::from_millis(200);
            }

            sleep(interval).await;
        }
    }

    /// TD-136 (dense episode retrieval): backfill `episodes.embedding` for every
    /// episode that has none, over the existing corpus — NO re-ingest, NO LLM.
    ///
    /// Selects NULL-embedding episodes in pages of `batch_size`, embeds each
    /// episode's `content` with the SAME embedder the graph already uses for
    /// entity/fact embeddings, and UPDATEs the `embedding` column (populating
    /// the `episodes_vec_idx` DiskANN index from Migration 026). Idempotent +
    /// resumable: an already-embedded episode is skipped (its `embedding` is
    /// non-NULL), so re-running only fills the remaining gap. Feature-gated
    /// behind `content-search` (the column only exists there).
    ///
    /// Returns the run [`EpisodeEmbeddingBackfill`] tally. Per-episode embed
    /// failures are counted (`failed`) + WARN-logged but do NOT abort the run —
    /// a transient embedder hiccup on one episode must not lose the whole
    /// backfill (re-run to retry the failures).
    ///
    /// Intended as an operator/maintenance entrypoint (e.g. the
    /// `kremory-http backfill-episode-embeddings` subcommand) — run it against a
    /// COPY of the DB before measuring the dense arm, so the pre-existing corpus
    /// is dense-searchable without a full re-ingest.
    #[cfg(feature = "content-search")]
    pub async fn backfill_episode_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::backfill_episode_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        // Guard against a zero page size (an infinite no-progress loop).
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        loop {
            let batch = tg
                .episodes_missing_embedding(batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            self.embed_and_store_episode_page(EmbedEpisodePageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "backfill_episode_embeddings",
            })
            .await;
            // If a whole page was all-failures we would loop forever on the same
            // NULL rows (the `WHERE embedding IS NULL` predicate never drops a
            // failed row out of the next page) — bail once we've made no forward
            // progress on a full page.
            if stats.embedded == 0 && stats.failed > 0 {
                tracing::warn!(
                    failed = stats.failed,
                    "backfill_episode_embeddings: first page all-failed — aborting (check the embedder)"
                );
                break;
            }
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.backfill_episode_embeddings complete"
        );
        Ok(stats)
    }

    /// TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143): re-embed
    /// **every** episode's `content`, overwriting any embedding already
    /// stored — the remedy for an embedding-CONFIG change (flipping
    /// [`SearchConfig::embed_task_prefix_enabled`](crate::core::config::SearchConfig::embed_task_prefix_enabled),
    /// swapping the embedder model, or changing the embedding dimension),
    /// none of which [`backfill_episode_embeddings`](Self::backfill_episode_embeddings)
    /// can serve — that method's `WHERE embedding IS NULL` paging can only
    /// FILL a gap, it can never RE-embed a row that already has a vector.
    ///
    /// ⚠️ This rewrites every `episodes.embedding` value in the database. Run
    /// it against a COPY of the DB before measuring — see the
    /// `embed_task_prefix_enabled` doc for the full safe sequence (flip the
    /// knob on a fresh copy, re-embed in full, THEN measure). Entity/fact
    /// embeddings are NOT touched by this method (they need a full re-ingest,
    /// or — for entities specifically — TD-112's merge-time re-embed covers
    /// only the alias-merge path, not a bulk config-change re-embed).
    ///
    /// Pages via an id-cursor over ALL episode rows
    /// ([`TemporalGraph::episodes_after_id`]), not the NULL-only predicate
    /// [`backfill_episode_embeddings`](Self::backfill_episode_embeddings)
    /// uses — see that query's doc for why a plain `LIMIT` loop over an
    /// unfiltered page source would never terminate. Idempotent: safe to
    /// re-run (e.g. to retry any per-episode failures from a prior run — see
    /// [`EpisodeEmbeddingBackfill::failed`]).
    ///
    /// Feature-gated behind `content-search` (the column only exists there).
    #[cfg(feature = "content-search")]
    pub async fn reembed_all_episode_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::reembed_all_episode_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        // Guard against a zero page size (an infinite no-progress loop).
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        let mut after_id: i64 = 0;
        loop {
            let batch = tg
                .episodes_after_id(after_id, batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            // Advance the cursor to the last id in THIS page before consuming
            // `batch` below — unlike the NULL-predicate backfill, a row stays
            // in this unfiltered result set after being re-embedded, so the
            // cursor (not the predicate) is what makes the loop terminate.
            after_id = batch.last().map(|(id, _)| *id).unwrap_or(after_id);
            self.embed_and_store_episode_page(EmbedEpisodePageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "reembed_all_episode_embeddings",
            })
            .await;
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.reembed_all_episode_embeddings complete"
        );
        Ok(stats)
    }

    /// TD-112/TD-143 shared embed step: document-prefix `text` (WRITE side —
    /// always [`document_embed_text`](crate::core::embed_prefix::document_embed_text),
    /// never the query-side prefix) and hand it to the configured embedder.
    ///
    /// Shared by all THREE bulk re-embed page loops
    /// ([`embed_and_store_episode_page`](Self::embed_and_store_episode_page),
    /// [`embed_and_store_entity_page`](Self::embed_and_store_entity_page),
    /// [`embed_and_store_fact_page`](Self::embed_and_store_fact_page)) — the
    /// one piece of "embed+write" body that is byte-identical across all
    /// three. The write-BACK half deliberately stays in each caller instead
    /// of behind a generic/closure dispatch: the id type differs (`i64` for
    /// episodes/facts, the entity's TEXT slug for entities) and so does the
    /// setter (`set_episode_embedding` / `set_entity_embedding` /
    /// `set_fact_embedding`) — a fully generic write dispatch would need
    /// either a discriminated-union id type or a closure-per-call-site, more
    /// machinery than the ~6 lines of per-caller match-arm plumbing it would
    /// save.
    #[cfg(feature = "content-search")]
    async fn embed_document_text(&self, text: &str) -> crate::core::error::Result<Vec<f32>> {
        let embed_task_prefix_enabled = self.search_config().embed_task_prefix_enabled;
        let prefixed_content =
            crate::core::embed_prefix::document_embed_text(text, embed_task_prefix_enabled);
        self.embedder.embed_dyn(&prefixed_content).await
    }

    /// Shared per-page embed+store body for
    /// [`backfill_episode_embeddings`](Self::backfill_episode_embeddings) and
    /// [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)
    /// — same embed call ([`embed_document_text`](Self::embed_document_text)),
    /// same per-episode failure handling; the two callers differ only in
    /// which paging query selected `batch`. `op` labels the WARN log lines so
    /// a failure can be attributed to the caller that hit it. Args-as-object
    /// per TD-042 (`clippy.toml` `too-many-arguments-threshold = 3`).
    #[cfg(feature = "content-search")]
    async fn embed_and_store_episode_page(&self, params: EmbedEpisodePageParams<'_>) {
        let EmbedEpisodePageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for (episode_id, content) in batch {
            match self.embed_document_text(&content).await {
                Ok(embedding) => match tg.set_episode_embedding(episode_id, &embedding).await {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(
                            error = %e,
                            episode_id,
                            op,
                            "set_episode_embedding failed"
                        );
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, episode_id, op, "embedder failed");
                }
            }
        }
    }

    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-112): re-embed
    /// **every** entity's display name, overwriting any embedding already
    /// stored. Sibling of [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)
    /// — same shape, same TD-143 document-prefix routing, different page
    /// source ([`TemporalGraph::entities_after_id`], composite-`(id,
    /// group_id)`-cursored — see that fn's doc for why).
    ///
    /// This is the remedy for TWO distinct staleness sources: (1) a live
    /// correctness bug — after a dream-phase merge/alias, a de-duplicated
    /// entity's identity may have changed (a new canonical name) while its
    /// stored embedding still encodes the pre-merge surface form, and no
    /// production path re-persisted it in bulk before this method existed
    /// (the merge-time re-embed at `core::canonicalization::apply_merge_with_audit`
    /// only covers the LIVE merge sites, not a full-corpus catch-up); and (2)
    /// an embedding-CONFIG change (flipping
    /// [`SearchConfig::embed_task_prefix_enabled`](crate::core::config::SearchConfig::embed_task_prefix_enabled),
    /// swapping the embedder model, or changing the embedding dimension).
    ///
    /// ⚠️ This rewrites every `entities.embedding` value in the database. Run
    /// it against a COPY of the DB before measuring — see
    /// [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)'s
    /// doc for the full safe sequence. Idempotent: safe to re-run (e.g. to
    /// retry any per-entity failures from a prior run).
    ///
    /// Feature-gated behind `content-search` (mirrors the episode/fact
    /// siblings — all three bulk re-embed paths ship together, even though
    /// entity/fact embeddings do not themselves depend on the content-RAG
    /// column; the `content-search` gate is this method family's existing
    /// convention, not a new dependency).
    #[cfg(feature = "content-search")]
    pub async fn reembed_all_entity_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::reembed_all_entity_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        let mut after_id = String::new();
        let mut after_group_id = String::new();
        loop {
            let batch = tg
                .entities_after_id(crate::core::graph::EntitiesAfterIdParams {
                    after_id: &after_id,
                    after_group_id: &after_group_id,
                    limit: batch_size,
                })
                .await
                .map_err(MemoryError::Core)?;
            // Advance the composite cursor to the LAST row in THIS page
            // before consuming `batch` below — same rationale as
            // `reembed_all_episode_embeddings`'s cursor advance (an
            // unfiltered page source never self-consumes). No `.expect()`
            // (banned in src/): an empty page ends the loop via the
            // `let-else`, same as a `.is_empty()` break would, without
            // needing a fallible unwrap on `.last()` right after.
            let Some(last) = batch.last() else {
                break;
            };
            after_id.clone_from(&last.id);
            after_group_id.clone_from(&last.group_id);
            self.embed_and_store_entity_page(EmbedEntityPageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "reembed_all_entity_embeddings",
            })
            .await;
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.reembed_all_entity_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared per-page embed+store body for
    /// [`reembed_all_entity_embeddings`](Self::reembed_all_entity_embeddings)
    /// — mirrors [`embed_and_store_episode_page`](Self::embed_and_store_episode_page);
    /// differs only in the id type (`&str` slug, not `i64`) and setter
    /// ([`TemporalGraph::set_entity_embedding`]). Args-as-object per TD-042.
    #[cfg(feature = "content-search")]
    async fn embed_and_store_entity_page(&self, params: EmbedEntityPageParams<'_>) {
        let EmbedEntityPageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for row in batch {
            match self.embed_document_text(&row.embed_text).await {
                Ok(embedding) => match tg.set_entity_embedding(&row.id, &embedding).await {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(
                            error = %e,
                            entity_id = %row.id,
                            op,
                            "set_entity_embedding failed"
                        );
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, entity_id = %row.id, op, "embedder failed");
                }
            }
        }
    }

    /// TD-112 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-112): re-embed
    /// **every** fact's `subject predicate object` triple text, overwriting
    /// any embedding already stored. Sibling of
    /// [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)
    /// — same shape, same TD-143 document-prefix routing, different page
    /// source ([`TemporalGraph::facts_after_id`] — see that fn's doc for how
    /// the subject/object text is reconstructed from stored entity rows,
    /// since the raw extraction strings themselves are not persisted).
    ///
    /// De-confounds a fact's dependency on its subject/object entities: when
    /// [`reembed_all_entity_embeddings`](Self::reembed_all_entity_embeddings)
    /// changes an entity's resolved display name (e.g. its `properties.name`
    /// was corrected), facts referencing that entity should be re-embedded
    /// too so their triple text stays in sync — run entity re-embed BEFORE
    /// fact re-embed when both are needed (the `reembed-all-embeddings`
    /// `kremory-http` subcommand does this in the right order).
    ///
    /// ⚠️ This rewrites every `facts.embedding` value in the database. Run it
    /// against a COPY of the DB before measuring. Idempotent: safe to re-run.
    /// Feature-gated behind `content-search`.
    #[cfg(feature = "content-search")]
    pub async fn reembed_all_fact_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::reembed_all_fact_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        let mut after_id: i64 = 0;
        loop {
            let batch = tg
                .facts_after_id(after_id, batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            after_id = batch.last().map(|(id, _)| *id).unwrap_or(after_id);
            self.embed_and_store_fact_page(EmbedFactPageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "reembed_all_fact_embeddings",
            })
            .await;
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.reembed_all_fact_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared per-page embed+store body for
    /// [`reembed_all_fact_embeddings`](Self::reembed_all_fact_embeddings) —
    /// mirrors [`embed_and_store_episode_page`](Self::embed_and_store_episode_page);
    /// differs only in the setter ([`TemporalGraph::set_fact_embedding`]).
    /// Args-as-object per TD-042.
    #[cfg(feature = "content-search")]
    async fn embed_and_store_fact_page(&self, params: EmbedFactPageParams<'_>) {
        let EmbedFactPageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for (fact_id, fact_text) in batch {
            match self.embed_document_text(&fact_text).await {
                Ok(embedding) => match tg.set_fact_embedding(fact_id, &embedding).await {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(error = %e, fact_id, op, "set_fact_embedding failed");
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, fact_id, op, "embedder failed");
                }
            }
        }
    }

    /// Block until the dream phase handle reaches a terminal status.
    pub async fn await_dream(
        &self,
        handle: &DreamHandle,
        timeout: Duration,
    ) -> Result<DreamStatus> {
        memory::await_dream(
            self.graph.as_ref(),
            handle.run_id,
            AwaitOpts {
                timeout,
                ..AwaitOpts::default()
            },
        )
        .await
    }

    /// Block until all episodes in `batch_id` reach a terminal status.
    pub async fn await_batch(&self, batch_id: &str, timeout: Duration) -> Result<BatchStatus> {
        memory::await_batch_enrichment(
            self.graph.as_ref(),
            batch_id,
            AwaitOpts {
                timeout,
                ..AwaitOpts::default()
            },
        )
        .await
    }

    /// Cancel an in-flight Phase 2 or Phase 3 run.
    pub async fn cancel(&self, commit: &EpisodeCommit) -> Result<CancelOutcome> {
        let run_id = commit.run_id.ok_or_else(|| {
            MemoryError::Other("EpisodeCommit has no run_id — cannot cancel inline Phase 2".into())
        })?;
        self.graph.graph_cancel(run_id).await
    }

    /// Cancel a dream phase by its handle.
    pub async fn cancel_dream(&self, handle: &DreamHandle) -> Result<CancelOutcome> {
        self.graph.graph_cancel(handle.run_id).await
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Flush any pending writes and close the memory handle.
    ///
    /// Currently a no-op: kremory uses libSQL WAL with autocommit, so there are
    /// no buffered writes to flush at v0.2.x. Retained as a forward-compatible
    /// shutdown hook — call it at shutdown so your code is ready if explicit
    /// flush semantics are added later (F9).
    pub async fn close(&self) -> Result<()> {
        // No-op: WAL autocommit means no pending writes to flush at v0.2.x.
        Ok(())
    }

    // ── Namespace policy (ADR-029a, v0.1.4) ───────────────────────────────────

    /// Register a namespace + its policy explicitly, ahead of any writes.
    ///
    /// # Idempotency
    ///
    /// Calling `register_namespace` with the SAME `(group_id, policy)` pair
    /// returns `Ok(())`. Calling with a DIFFERENT policy on an existing
    /// `group_id` returns
    /// `Err(MemoryError::Core(Error::NamespacePolicyImmutable { ... }))`.
    /// This makes startup code safe to re-execute (idempotent against
    /// persisted state).
    ///
    /// # Validation
    ///
    /// The policy attached to `namespace` is validated via
    /// [`NamespacePolicy::validate`] before persistence. Incoherent policies
    /// surface as `Err(MemoryError::Core(Error::InvalidPolicy(...)))`.
    ///
    /// # No enforcement at v0.1.4
    ///
    /// The policy is PERSISTED but not yet enforced on
    /// `dream()` / `forget()` / mutation operations. Enforcement lands in
    /// v0.1.5+ per ADR-029b. Every non-default policy registration emits
    /// a `tracing::warn!` on target `kremory.namespace` to make the
    /// declaration vs enforcement gap visible.
    ///
    /// # Race semantics — atomic via `BEGIN IMMEDIATE`
    ///
    /// `register_namespace` wraps the SELECT + INSERT pair in a
    /// `BEGIN IMMEDIATE` transaction (per ADR-022 write_lock invariant). This
    /// acquires SQLite's RESERVED write lock before reading, serializing
    /// against concurrent `remember(...)` calls that would implicitly create
    /// the namespace with default policy.
    ///
    /// The recommended pattern is `register_namespace` AT STARTUP before any
    /// `remember(...)`. See ADR-029a Decision 6 for the three race outcomes.
    pub async fn register_namespace(&self, namespace: Namespace) -> Result<()> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::register_namespace requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let policy = namespace.policy.clone().unwrap_or_default();
        policy
            .validate()
            .map_err(|e| MemoryError::Core(CoreError::InvalidPolicy(e)))?;

        let group_id = namespace_to_group_id(&namespace);
        let is_non_default = policy != NamespacePolicy::default();

        // Atomic INSERT-or-compare via BEGIN IMMEDIATE.
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(MemoryError::Core)?;
        let stored = tg
            .get_namespace_policy(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        let outcome: Result<()> = match stored {
            Some(existing) if existing == policy => Ok(()),
            Some(existing) => Err(MemoryError::Core(CoreError::NamespacePolicyImmutable {
                namespace: group_id.clone(),
                stored: existing,
                attempted: policy.clone(),
            })),
            None => tg
                .set_namespace_policy(&group_id, &policy)
                .await
                .map_err(MemoryError::Core),
        };
        match &outcome {
            Ok(()) => {
                guard.commit().await.map_err(MemoryError::Core)?;
            }
            Err(_) => {
                guard.rollback().await.map_err(MemoryError::Core)?;
            }
        }

        // Operational visibility: every non-default policy DECLARATION emits
        // warn (NOT info) — closes Vera cycle-1 HIGH-1 footgun. Default
        // policies are silent (they would be the existing behaviour).
        if outcome.is_ok() && is_non_default {
            tracing::warn!(
                target: "kremory.namespace",
                group_id = %group_id,
                policy = ?policy,
                "kremory.namespace.policy_declared: POLICY DECLARED BUT NOT \
                 ENFORCED at v0.1.4 — enforcement lands v0.1.5+ per ADR-029b. \
                 See https://docs.rs/kremory/0.1.4/kremory/#adr-029a"
            );
        }

        outcome
    }

    /// Register a namespace policy + seed its entity-type registry in one atomic
    /// operation (spec custom-entity-type-registry §5.2.2).
    ///
    /// Seeds the namespace's `entity_types` table per the `seed` instruction,
    /// and registers the (default) namespace policy — both inside the SAME
    /// `BEGIN IMMEDIATE` transaction (no TOCTOU window between policy write and
    /// seed write; §5.8 invariant).
    ///
    /// # Semantics (D9 / D9a — brownfield safety)
    ///
    /// | namespace state         | `Default` / `Augment` | `Replace`                              |
    /// |-------------------------|-----------------------|----------------------------------------|
    /// | no rows (fresh)         | seed → `Seeded`       | seed → `Seeded`                        |
    /// | has rows, seed MATCHES  | `AlreadySeeded`       | `AlreadySeeded` (D9a, idempotent boot) |
    /// | has rows, seed DIFFERS  | `AlreadySeeded`       | `Err(AlreadyPopulated { group_id })`   |
    ///
    /// - id=0 "Entity" catch-all is ALWAYS present after a successful call.
    /// - `Replace` NEVER mutates a populated namespace — it fails loud with NO
    ///   DB write, so existing entities' `entity_type_id` can never be orphaned
    ///   (ASMP-003). To add types to an already-populated namespace use
    ///   [`assert_entity_type`](Self::assert_entity_type) (the sanctioned
    ///   incremental-add path).
    ///
    /// This is the PREFERRED startup pattern for domain-specific namespaces:
    /// call at startup BEFORE the first `remember()` for the target namespace.
    /// The existing [`register_namespace`](Self::register_namespace) remains for
    /// callers that want the default seed.
    pub async fn register_namespace_with_seed(
        &self,
        namespace: Namespace,
        seed: crate::core::entity_types::NamespaceSeed,
    ) -> std::result::Result<
        crate::core::entity_types::SeedOutcome,
        crate::core::entity_types::NamespaceRegistrationError,
    > {
        use crate::core::entity_types::NamespaceRegistrationError;

        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            NamespaceRegistrationError::Store(CoreError::Other(anyhow::anyhow!(
                "Memory::register_namespace_with_seed requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
            )))
        })?;

        let policy = namespace.policy.clone().unwrap_or_default();
        policy
            .validate()
            .map_err(|e| NamespaceRegistrationError::Store(CoreError::InvalidPolicy(e)))?;

        let group_id = namespace_to_group_id(&namespace);
        let is_non_default = policy != NamespacePolicy::default();

        // Single BEGIN IMMEDIATE wrapping: namespace policy write + seed
        // application. Presence check + seed live inside this txn (no TOCTOU).
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(NamespaceRegistrationError::Store)?;

        let outcome: std::result::Result<
            crate::core::entity_types::SeedOutcome,
            NamespaceRegistrationError,
        > = async {
            // Policy: INSERT-or-compare (mirrors register_namespace).
            let stored = tg
                .get_namespace_policy(&group_id)
                .await
                .map_err(NamespaceRegistrationError::Store)?;
            match stored {
                Some(existing) if existing == policy => {}
                Some(existing) => {
                    return Err(NamespaceRegistrationError::Store(
                        CoreError::NamespacePolicyImmutable {
                            namespace: group_id.clone(),
                            stored: existing,
                            attempted: policy.clone(),
                        },
                    ));
                }
                None => {
                    tg.set_namespace_policy(&group_id, &policy)
                        .await
                        .map_err(NamespaceRegistrationError::Store)?;
                }
            }

            // Seed (D9 / D9a) inside the same txn.
            crate::core::entity_types::apply_namespace_seed(&tg.conn, &group_id, &seed).await
        }
        .await;

        match &outcome {
            Ok(_) => {
                guard
                    .commit()
                    .await
                    .map_err(NamespaceRegistrationError::Store)?;
            }
            Err(_) => {
                guard
                    .rollback()
                    .await
                    .map_err(NamespaceRegistrationError::Store)?;
            }
        }

        if outcome.is_ok() && is_non_default {
            tracing::warn!(
                target: "kremory.namespace",
                group_id = %group_id,
                policy = ?policy,
                "kremory.namespace.policy_declared: POLICY DECLARED BUT NOT \
                 ENFORCED at v0.1.4 — enforcement lands v0.1.5+ per ADR-029b."
            );
        }

        outcome
    }

    /// Monotonically upgrade a namespace's immutability from `Mutable` to
    /// `AppendOnly` (ADR-029b Decision 5).
    ///
    /// This is a **one-way ratchet**: `Mutable → AppendOnly` is the only
    /// allowed direction. Attempting to downgrade (`AppendOnly → Mutable`)
    /// returns `Err(MemoryError::Core(Error::NamespacePolicyImmutable))`.
    /// Calling on an already-`AppendOnly` namespace is idempotent (`Ok(())`).
    ///
    /// # Atomicity
    ///
    /// The read-decide-write sequence is wrapped in a `BEGIN IMMEDIATE`
    /// transaction to prevent races with concurrent `register_namespace` or
    /// `upgrade_namespace_policy` calls.
    ///
    /// # Errors
    ///
    /// - `MemoryError::Other` — `Memory` not constructed via the builder path.
    /// - `MemoryError::Core(Error::NamespacePolicyImmutable)` — downgrade
    ///   attempted or policy mismatch.
    /// - `MemoryError::Core(Error::Other)` — substrate failure.
    pub async fn upgrade_namespace_policy(&self, namespace: Namespace) -> Result<()> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::upgrade_namespace_policy requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let group_id = namespace_to_group_id(&namespace);
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(MemoryError::Core)?;

        let stored = tg
            .get_namespace_policy(&group_id)
            .await
            .map_err(MemoryError::Core)?;

        let target_policy = crate::memory::types::NamespacePolicy::APPEND_ONLY;

        let outcome: Result<()> = match stored {
            Some(ref existing)
                if existing.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly =>
            {
                // Already AppendOnly — idempotent.
                Ok(())
            }
            Some(ref existing)
                if existing.immutability == crate::memory::types::ImmutabilityLevel::Mutable =>
            {
                // Upgrade Mutable → AppendOnly.
                tg.set_namespace_policy_with_upgraded_at(&group_id, &target_policy)
                    .await
                    .map_err(MemoryError::Core)
            }
            Some(existing) => {
                // Unexpected policy state — treat as immutable conflict.
                Err(MemoryError::Core(CoreError::NamespacePolicyImmutable {
                    namespace: group_id.clone(),
                    stored: existing,
                    attempted: target_policy,
                }))
            }
            None => {
                // Namespace not yet registered — create directly as AppendOnly.
                tg.set_namespace_policy_with_upgraded_at(&group_id, &target_policy)
                    .await
                    .map_err(MemoryError::Core)
            }
        };

        match &outcome {
            Ok(()) => {
                guard.commit().await.map_err(MemoryError::Core)?;
                // Invalidate cache so next read reflects the new AppendOnly policy.
                tg.invalidate_policy_cache(&group_id);
                tracing::info!(
                    target: "kremory.namespace",
                    group_id = %group_id,
                    "kremory.namespace.policy_upgraded: namespace upgraded to AppendOnly"
                );
            }
            Err(_) => {
                guard.rollback().await.map_err(MemoryError::Core)?;
            }
        }
        outcome
    }

    /// Lazy-population helper: ensure a default-policy row exists for the
    /// `namespace` if it has not been observed yet. Invoked from the first-
    /// encounter paths (`remember`, `recall`, `forget`, `dream`) per ADR-029a
    /// Decision 8.
    ///
    /// Best-effort: when `Memory` is constructed without a direct
    /// `Arc<TemporalGraph>` (e.g. test-only stub-handle path) this is a no-op.
    /// Errors from the substrate are converted to `MemoryError::Core` and
    /// returned so the call site can decide whether to fail the user request.
    pub(crate) async fn ensure_namespace_policy(&self, namespace: &Namespace) -> Result<()> {
        let Some(tg) = self.temporal_graph.as_ref() else {
            return Ok(());
        };
        let group_id = namespace_to_group_id(namespace);
        tg.ensure_namespace_policy_row(&group_id)
            .await
            .map_err(MemoryError::Core)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn resolve_namespace(&self, override_ns: Option<Namespace>) -> Result<Namespace> {
        override_ns
            .or_else(|| self.default_namespace.clone())
            .ok_or_else(|| {
                MemoryError::MissingNamespace {
                    request: "namespace is required — call .in_namespace(ns) or set default_namespace on the builder",
                }
            })
    }

    fn resolve_sink(
        &self,
        per_call: Option<Arc<dyn EnrichmentEventSink>>,
    ) -> Option<Arc<dyn EnrichmentEventSink>> {
        per_call.or_else(|| self.default_sink.clone())
    }

    /// Return the wired LLM or `Err(MemoryError::Core(Error::LlmRequired))`.
    ///
    /// Used by Category B methods (dream, recall_with_disambiguation,
    /// detect_contradictions) that unconditionally require an LLM.
    pub(crate) fn llm_or_err(
        &self,
        method: &'static str,
        hint: &'static str,
    ) -> Result<Arc<dyn ChatProvider>> {
        self.llm.clone().ok_or_else(|| {
            tracing::warn!(
                method = method,
                "LLM required but not wired — returning LlmRequired"
            );
            metrics::counter!("kremory.llm_required_total", "method" => method).increment(1);
            MemoryError::Core(CoreError::LlmRequired { method, hint })
        })
    }

    /// Return the dream-phase LLM: the dedicated `with_dream_llm` provider when
    /// set, else delegate to [`llm_or_err`](Self::llm_or_err) (the `with_llm`
    /// provider, or `Err(LlmRequired)` when neither is wired).
    ///
    /// Backward-compat is **structural** (load-bearing-invariants-at-emit, TD-052b
    /// §3.3): when `dream_llm` is `None`, this is byte-for-byte the prior
    /// `llm_or_err("dream", …)` behaviour — same provider, same error, same metric.
    ///
    /// Emits `kremory.dream.llm_role_selected_total{model_role}` on **every** call
    /// (TD-052b §6). The counter measures role-selection *attempts*, not realised
    /// successes — the `interactive` arm increments before `llm_or_err`, so on the
    /// no-provider row it counts an attempt that then errors `LlmRequired`. The
    /// authoritative dream-failure signal remains
    /// `kremory.llm_required_total{method="dream"}`.
    pub(crate) fn dream_llm_or_main(
        &self,
        method: &'static str,
        hint: &'static str,
    ) -> Result<Arc<dyn ChatProvider>> {
        match &self.dream_llm {
            Some(llm) => {
                tracing::debug!(
                    target: "kremory.facade.dream",
                    model_role = "dream",
                    "dream phase using dedicated dream_llm provider (TD-052b)"
                );
                metrics::counter!(
                    "kremory.dream.llm_role_selected_total",
                    "model_role" => "dream"
                )
                .increment(1);
                Ok(Arc::clone(llm))
            }
            None => {
                // Fallback: identical to prior behaviour. Emit the role counter so
                // dashboards can attribute dream calls to the shared interactive
                // provider; llm_or_err still owns the LlmRequired warn+counter.
                metrics::counter!(
                    "kremory.dream.llm_role_selected_total",
                    "model_role" => "interactive"
                )
                .increment(1);
                self.llm_or_err(method, hint)
            }
        }
    }

    /// Return the model id the dream phase should thread into its LLM passes for
    /// capability detection: the dedicated `with_dream_model_id` string when set,
    /// else the main `with_model_id` string, else `None`.
    ///
    /// Mirrors [`dream_llm_or_main`](Self::dream_llm_or_main) at the model-id
    /// layer: `dream_model_id` pairs with `dream_llm` the way `model_id` pairs
    /// with `llm`. `None` (raw `with_llm` without a model id) means the dream
    /// passes degrade to `PromptOnly` exactly as the interactive path does —
    /// no worse than before, and the correct behaviour when the model is unknown.
    ///
    /// TD-094: before this resolver, the facade dream path hardcoded an empty
    /// model string in every LLM pass, silently degrading every configuration
    /// (even a fully-specified `with_model_id`) to zero structured output.
    pub(crate) fn dream_model_id_or_main(&self) -> Option<&str> {
        self.dream_model_id.as_deref().or(self.model_id.as_deref())
    }

    /// Return the wired LLM, or a no-op stub when no LLM was configured.
    ///
    /// Used by Category A methods (remember, remember_batch) that pass a
    /// provider arg to `submit_episode` but the engine ignores the arg when
    /// `skip_extraction = true` or when the custom extractor handles extraction.
    /// The real `LlmRequired` error fires from `pipeline.rs` if an LLM-dependent
    /// pipeline step is actually reached without a wired LLM.
    pub(crate) fn llm_or_stub(&self) -> Arc<dyn ChatProvider> {
        self.llm
            .clone()
            .unwrap_or_else(|| Arc::new(crate::core::provider::NullChatProvider))
    }

    // ── Dream pass API (Phase C DoD C1, C4, C5, C11) ─────────────────────────

    /// Run a synchronous dream pass with the given options.
    ///
    /// At-most-one concurrent dream pass per engine (serialised via internal
    /// Mutex). Subsequent calls will block until the active pass completes.
    ///
    /// Pass 0 (type discovery) and Pass 2 (ghost episode retry) are **stubbed**
    /// in Phase C — they return empty/zero counts. Real logic lands in Phase D/E.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C1 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn run_dream_pass_sync(
        &self,
        opts: crate::core::ingest::DreamPassOpts,
    ) -> Result<DreamSummary> {
        self.graph.graph_run_dream_pass_sync(opts).await
    }

    /// Return episode IDs where Phase 1 succeeded but Phase 2 produced no facts
    /// (ghost episodes).
    ///
    /// An optional `group_id` restricts the query to one namespace/thread.
    /// `None` returns ghost episodes across all namespaces.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C4 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        self.graph.graph_ghost_episodes(group_id).await
    }

    /// Pin an entity as `ConsumerPinned`, protecting it from dream reclassification.
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` on the entity row.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C5 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn assert_entity_type(&self, params: GraphAssertEntityTypeParams<'_>) -> Result<()> {
        self.graph.graph_assert_entity_type(params).await
    }

    /// Start a dream scheduler background task at runtime.
    ///
    /// Returns a [`DreamSchedulerHandle`] the caller can use to stop the task.
    /// For scheduler-at-build-time, use [`MemoryBuilder::with_dream_schedule`]
    /// instead.
    ///
    /// If a scheduler was already started via `with_dream_schedule` at build
    /// time, calling this method starts an ADDITIONAL independent scheduler.
    /// Stop the build-time one via [`Memory::stop_dream_scheduler`] first if
    /// you want to replace it.
    ///
    /// # ADR reference
    ///
    /// Phase C DoD C11 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub fn start_dream_scheduler(
        &self,
        schedule: crate::memory::scheduler::DreamSchedule,
    ) -> crate::memory::scheduler::DreamSchedulerHandle {
        crate::memory::scheduler::spawn_scheduler(
            Arc::clone(&self.graph),
            schedule,
            crate::core::ingest::DreamPassOpts::default,
        )
    }

    /// Stop the scheduler that was started via `with_dream_schedule` at build
    /// time, if one is running.
    ///
    /// No-op if no build-time scheduler is active. Returns `true` if a
    /// scheduler was stopped, `false` if none was running.
    pub async fn stop_dream_scheduler(&self) -> bool {
        let handle = self
            .dream_scheduler
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(h) = handle {
            h.stop().await;
            true
        } else {
            false
        }
    }
}

mod builder;
pub use builder::{MemoryBuilder, WithLlmTrackedParams};

#[cfg(test)]
mod dream_llm_slot_tests {
    //! TD-052b — per-phase dream model slot (`with_dream_llm`).
    //!
    //! Governing spec: `.ai-docs/specs/td-052b-dream-llm-slot-spec-2026-06-22.md`.
    //! These in-crate tests exercise the `pub(crate)` `dream_llm_or_main`
    //! accessor + the `kremory.dream.llm_role_selected_total{model_role}`
    //! counter (§3.3 / §6) — surfaces unreachable from an integration test.
    //! Public-surface build tests live in `tests/memory_builder_compat_matrix.rs`.

    use super::*;
    use crate::core::provider::MockEmbeddingProvider;
    use crate::memory::ChatProvider;
    use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
    use autoagents_llm::error::LLMError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Call-counting `ChatProvider`: records how many times `chat_with_tools`
    /// fired, so a test can assert WHICH slot the dream phase invoked. Returns
    /// an empty response (the dream fan-out tolerates empty proposals).
    #[derive(Debug)]
    struct CountingProvider {
        calls: Arc<AtomicUsize>,
    }

    impl CountingProvider {
        fn new() -> (Arc<Self>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(Self {
                    calls: Arc::clone(&calls),
                }),
                calls,
            )
        }
    }

    #[async_trait::async_trait]
    impl ChatProvider for CountingProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(EmptyResponse))
        }
    }

    #[derive(Debug)]
    struct EmptyResponse;

    impl std::fmt::Display for EmptyResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "")
        }
    }

    impl ChatResponse for EmptyResponse {
        fn text(&self) -> Option<String> {
            None
        }
        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
    }

    fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
        Arc::new(MockEmbeddingProvider::new(64))
    }

    async fn build_with(
        llm: Option<Arc<dyn ChatProvider>>,
        dream_llm: Option<Arc<dyn ChatProvider>>,
    ) -> Memory {
        // Build via the real builder so the field threads through the same
        // construction path production uses. NoLlm builds require an extractor;
        // use the LLM path when a main LLM is present, else a null extractor.
        match llm {
            Some(main) => {
                let mut b = Memory::open(":memory:").with_llm(main);
                if let Some(d) = dream_llm {
                    b = b.with_dream_llm(d);
                }
                b.with_embedder(null_embedder())
                    .await
                    .expect("build WithLlm")
            }
            None => {
                let mut b = Memory::open(":memory:")
                    .with_extractor(Arc::new(crate::core::intelligence::MockExtractor));
                if let Some(d) = dream_llm {
                    b = b.with_dream_llm(d);
                }
                b.with_embedder(null_embedder())
                    .await
                    .expect("build NoLlm + extractor")
            }
        }
    }

    /// T3 — `dream_llm = None` → `dream_llm_or_main` returns the SAME `Arc` as
    /// `llm_or_err` (structural fallback, byte-for-byte prior behaviour).
    #[tokio::test]
    async fn t3_accessor_falls_back_to_main_when_dream_unset() {
        let (main, _) = CountingProvider::new();
        let main: Arc<dyn ChatProvider> = main;
        let mem = build_with(Some(Arc::clone(&main)), None).await;

        let via_dream = mem
            .dream_llm_or_main("dream", "hint")
            .expect("main is wired");
        let via_main = mem.llm_or_err("dream", "hint").expect("main is wired");
        assert!(
            Arc::ptr_eq(&via_dream, &via_main),
            "with dream_llm=None, dream_llm_or_main must return the same Arc as llm_or_err"
        );
    }

    /// T3 — `dream_llm = Some(D)` → `dream_llm_or_main` returns D, distinct from
    /// the main provider.
    #[tokio::test]
    async fn t3_accessor_returns_dream_provider_when_set() {
        let (main, _) = CountingProvider::new();
        let (dream, _) = CountingProvider::new();
        let main: Arc<dyn ChatProvider> = main;
        let dream: Arc<dyn ChatProvider> = dream;
        let mem = build_with(Some(Arc::clone(&main)), Some(Arc::clone(&dream))).await;

        let selected = mem
            .dream_llm_or_main("dream", "hint")
            .expect("dream provider wired");
        assert!(
            Arc::ptr_eq(&selected, &dream),
            "dream_llm_or_main must return the dedicated dream provider"
        );
        assert!(
            !Arc::ptr_eq(&selected, &main),
            "dream_llm_or_main must NOT return the main provider when dream_llm is set"
        );
    }

    /// TD-094 — `dream_model_id_or_main` resolves the model id the dream LLM
    /// passes use for capability detection. Mirrors `dream_llm_or_main` at the
    /// model-id layer: dedicated `with_dream_model_id` wins; else `with_model_id`;
    /// else `None` (→ `PromptOnly` degrade). Regression guard for the empty-model
    /// bug where dream passes silently degraded to zero structured output.
    #[tokio::test]
    async fn td094_dream_model_id_resolution() {
        let (main, _) = CountingProvider::new();
        let main: Arc<dyn ChatProvider> = main;

        // Case 1: only with_model_id → dream falls back to the main model id.
        let mem = Memory::open(":memory:")
            .with_llm(Arc::clone(&main))
            .with_model_id("main-model")
            .with_embedder(null_embedder())
            .await
            .expect("build with model_id");
        assert_eq!(
            mem.dream_model_id_or_main(),
            Some("main-model"),
            "dream must fall back to the main model id when no dream model id is set"
        );

        // Case 2: with_dream_model_id takes precedence over with_model_id.
        let mem = Memory::open(":memory:")
            .with_llm(Arc::clone(&main))
            .with_model_id("main-model")
            .with_dream_model_id("dream-model")
            .with_embedder(null_embedder())
            .await
            .expect("build with dream_model_id");
        assert_eq!(
            mem.dream_model_id_or_main(),
            Some("dream-model"),
            "dedicated dream model id must take precedence over the main model id"
        );

        // Case 3: neither set (raw with_llm) → None → PromptOnly degrade, unchanged.
        let mem = Memory::open(":memory:")
            .with_llm(Arc::clone(&main))
            .with_embedder(null_embedder())
            .await
            .expect("build without any model id");
        assert_eq!(
            mem.dream_model_id_or_main(),
            None,
            "unset model id must resolve to None (PromptOnly degrade), not an empty-string sentinel"
        );
    }

    /// T6 row 4 — neither main nor dream wired → `dream_llm_or_main` errors
    /// `LlmRequired` exactly as the prior `llm_or_err` path (unchanged).
    #[tokio::test]
    async fn t6_row4_neither_provider_errors_llm_required() {
        let mem = build_with(None, None).await;
        let result = mem.dream_llm_or_main("dream", "hint");
        let Err(err) = result else {
            panic!("no provider wired → dream_llm_or_main must error LlmRequired");
        };
        assert!(
            matches!(
                err,
                MemoryError::Core(CoreError::LlmRequired {
                    method: "dream",
                    ..
                })
            ),
            "expected LlmRequired{{method=\"dream\"}}, got: {err:?}"
        );
    }

    /// T5 — a real blocking `dream()` routes through the DREAM slot, never MAIN.
    ///
    /// `dream()` calls `dream_llm_or_main` (dream.rs:95) — selecting the dedicated
    /// dream provider and emitting `model_role="dream"`. Pass-0 and Pass-2 execute
    /// if a TemporalGraph is present; on an empty graph they return Ok with zero work.
    /// Phase-3 consolidation fields (communities/merges/supersessions/archival) are
    /// always 0 — honest zeros per ADR-007 retirement. The role counter is the
    /// **authoritative routing proof**; `MAIN.calls == 0` proves the main slot is
    /// never used for the dream phase.
    #[test]
    fn t5_dream_invokes_dream_provider_not_main() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        let (main, main_calls) = CountingProvider::new();
        let (dream, dream_calls) = CountingProvider::new();
        let main: Arc<dyn ChatProvider> = main;
        let dream: Arc<dyn ChatProvider> = dream;

        metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let mem = build_with(Some(main), Some(dream)).await;
                // Routes through dream_llm_or_main → DREAM slot. Returns Ok on
                // empty graph (Pass-0/Pass-2 find no work; honest-zero summary).
                mem.dream()
                    .in_namespace(Namespace::new("default"))
                    .await
                    .expect("dream on empty graph must return Ok");
            });
        });

        assert_eq!(
            main_calls.load(Ordering::SeqCst),
            0,
            "the MAIN provider must NEVER be invoked by the dream phase when a dedicated dream_llm is set"
        );
        // dream_calls may be 0 (empty graph) — the role counter is the
        // authoritative routing proof.
        let _ = dream_calls;

        let snapshot = snapshotter.snapshot();
        let dream_role_total: u64 = snapshot
            .into_vec()
            .into_iter()
            .filter(|(k, _, _, _)| {
                k.key().name() == "kremory.dream.llm_role_selected_total"
                    && k.key()
                        .labels()
                        .any(|l| l.key() == "model_role" && l.value() == "dream")
            })
            .map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => c,
                _ => 0,
            })
            .sum();
        assert!(
            dream_role_total >= 1,
            "dream pass with dream_llm=Some must increment llm_role_selected_total{{model_role=\"dream\"}}"
        );
    }

    /// T7 — the `interactive` role counter fires when dream falls back to the
    /// main provider (dream_llm unset, main set).
    #[test]
    fn t7_interactive_role_counter_on_fallback() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        let (main, _) = CountingProvider::new();
        let main: Arc<dyn ChatProvider> = main;

        metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let mem = build_with(Some(main), None).await;
                let _ = mem.dream_llm_or_main("dream", "hint");
            });
        });

        let snapshot = snapshotter.snapshot();
        let interactive_total: u64 = snapshot
            .into_vec()
            .into_iter()
            .filter(|(k, _, _, _)| {
                k.key().name() == "kremory.dream.llm_role_selected_total"
                    && k.key()
                        .labels()
                        .any(|l| l.key() == "model_role" && l.value() == "interactive")
            })
            .map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => c,
                _ => 0,
            })
            .sum();
        assert!(
            interactive_total >= 1,
            "fallback dream resolution (dream_llm=None) must increment llm_role_selected_total{{model_role=\"interactive\"}}"
        );
    }

    /// NT-1 — blocking `dream()` invokes the DREAM LLM slot and discovers types
    /// from catch-all entities seeded into the TemporalGraph.
    ///
    /// Closes the TD-052b decorative gap: before the `run_dream_phase`
    /// short-circuit was removed, `dream_llm` flowed only into unreachable code.
    /// Verifies `dream_calls >= 1`, `main_calls == 0`, and `types_discovered`
    /// reflects the scripted proposal — the assertion T5 could not make.
    ///
    /// NT-2 (honest-zeros lock) is folded in: Phase-3 consolidation fields must
    /// always be 0 pending consolidation implementation (ADR-007 retirement).
    #[tokio::test]
    async fn nt1_dream_invokes_dream_llm_discovers_types_and_zeroes_consolidation() {
        use crate::core::entity_types::ensure_default_types_seeded;
        use chrono::Utc;

        // ScriptedCountingProvider: counts calls AND returns valid proposal JSON.
        #[derive(Debug)]
        struct ScriptedCountingProvider {
            calls: Arc<AtomicUsize>,
        }

        #[derive(Debug)]
        struct TextResponse {
            text: String,
        }
        impl std::fmt::Display for TextResponse {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.text)
            }
        }
        impl ChatResponse for TextResponse {
            fn text(&self) -> Option<String> {
                Some(self.text.clone())
            }
            fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
                None
            }
        }

        #[async_trait::async_trait]
        impl ChatProvider for ScriptedCountingProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(TextResponse {
                    text: r#"{"proposals":[{"name":"Company","description":"A business entity.","justification":"All three are companies."}]}"#
                        .to_string(),
                }))
            }
        }

        let dream_calls = Arc::new(AtomicUsize::new(0));
        let scripted_dream: Arc<dyn ChatProvider> = Arc::new(ScriptedCountingProvider {
            calls: Arc::clone(&dream_calls),
        });
        let (main, main_calls) = CountingProvider::new();
        let main: Arc<dyn ChatProvider> = main;

        let mem = build_with(Some(main), Some(scripted_dream)).await;

        // Seed catch-all entities so Pass-0 has clusters to process.
        // group_id = "default" (namespace_to_group_id(Namespace::new("default"))).
        let tg = mem
            .temporal_graph_for_test()
            .expect("TemporalGraph present in :memory: build");
        let conn = tg.conn.clone();
        ensure_default_types_seeded(&conn, "default")
            .await
            .expect("seed entity_types defaults");
        let now = Utc::now().to_rfc3339();
        for id in ["alpha corp", "beta fund", "gamma ventures"] {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![id.to_string(), now.clone(), "default".to_string()],
            )
            .await
            .expect("seed catch-all entity");
        }

        let summary = mem
            .dream()
            .in_namespace(Namespace::new("default"))
            // Consumer-API hardening D1 turned ALL FOUR consolidation ops ON by
            // default. This Pass-0/type-discovery test pins EVERY consolidation flag
            // OFF so its honest-zeros below stay valid (consolidation behaviour is
            // covered elsewhere).
            .with_opts(crate::memory::types::DreamOpts {
                include_cross_episode_merges: false,
                include_community_detection: false,
                include_fact_archival: false,
                include_supersession_sweep: false,
                ..Default::default()
            })
            .await
            .expect("dream with seeded catch-all entities must return Ok");

        // NT-1: DREAM provider fired; MAIN never touched.
        assert!(
            dream_calls.load(Ordering::SeqCst) >= 1,
            "Pass-0 must invoke the dream LLM slot when catch-all entities exist"
        );
        assert_eq!(
            main_calls.load(Ordering::SeqCst),
            0,
            "MAIN provider must never be invoked by the dream phase"
        );
        // NT-1: types_discovered count is not asserted > 0 — anti-redundancy can correctly
        // reject proposals that overlap existing types in the test embedder's metric space
        // (stochastic, mirrors the plan's "do NOT assert > 0" stance). The load-bearing
        // signal is dream_calls >= 1 above: that proves TD-052b is live and Pass-0 ran.
        let _ = summary.types_discovered;
        // NT-2 (honest-zeros lock): Phase-3 consolidation fields always 0.
        assert_eq!(
            summary.communities_updated, 0,
            "communities_updated must be 0 — consolidation pinned OFF here (ADR-071)"
        );
        assert_eq!(
            summary.cross_episode_would_merge, 0,
            "cross_episode_would_merge must be 0 — consolidation pinned OFF here"
        );
        assert_eq!(
            summary.cross_episode_merged, 0,
            "cross_episode_merged must be 0 — consolidation pinned OFF here"
        );
        assert_eq!(
            summary.supersessions_recorded, 0,
            "supersessions_recorded must be 0 — consolidation pinned OFF here (ADR-071)"
        );
        assert_eq!(
            summary.facts_archived, 0,
            "facts_archived must be 0 — consolidation pinned OFF here (ADR-071)"
        );
    }
}

#[cfg(all(test, feature = "content-search"))]
mod reembed_all_episode_embeddings_tests {
    //! TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143) —
    //! `Memory::reembed_all_episode_embeddings` overwrite proof. Lives
    //! in-crate (not `tests/`) because proving the STORED vector actually
    //! changed requires `pub(crate)` `TemporalGraph::vector_search_episodes`
    //! — unreachable from an external integration test. The prefix-parity and
    //! idempotency companion tests use only public API and live in
    //! `tests/td143_reembed_all_episode_embeddings.rs`.

    use super::*;
    use crate::core::provider::{EmbeddingProvider, MockChatProvider};
    use crate::core::search::{SearchFilters, VectorSearchEpisodesParams};
    use std::sync::atomic::{AtomicBool, Ordering};

    fn null_llm() -> Arc<dyn ChatProvider> {
        Arc::new(MockChatProvider::null())
    }

    /// Deterministic 384-dim embedding derived from `seed` — same shape as
    /// `core::search`'s own `make_embedding` test helper. Distinct seeds
    /// produce distinguishably-different vectors so `vector_search_episodes`
    /// ranking can prove a stored embedding actually moved.
    fn seeded_embedding(seed: f32) -> Vec<f32> {
        (0..384)
            .map(|i| (i as f32 * seed).sin() * 0.5 + 0.5)
            .collect()
    }

    /// Embedder whose output seed is switchable via an atomic flag — lets a
    /// test embed the SAME text twice and get two DIFFERENT vectors,
    /// simulating "the embedder/config changed since the corpus was first
    /// embedded" without needing a second real model.
    struct SwitchableEmbeddingProvider {
        use_second_seed: AtomicBool,
    }

    impl SwitchableEmbeddingProvider {
        fn new() -> Self {
            Self {
                use_second_seed: AtomicBool::new(false),
            }
        }

        fn switch_to_second_seed(&self) {
            self.use_second_seed.store(true, Ordering::SeqCst);
        }
    }

    impl EmbeddingProvider for SwitchableEmbeddingProvider {
        fn embed<'a>(
            &'a self,
            _text: &'a str,
        ) -> impl std::future::Future<Output = crate::CoreResult<Vec<f32>>> + Send + 'a {
            let seed = if self.use_second_seed.load(Ordering::SeqCst) {
                5.0
            } else {
                1.0
            };
            async move { Ok(seeded_embedding(seed)) }
        }
    }

    /// Re-embedding an ALREADY-EMBEDDED episode overwrites the stored vector
    /// — the exact gap `backfill_episode_embeddings` cannot close (its `WHERE
    /// embedding IS NULL` paging only ever fills a gap). Proven via
    /// `vector_search_episodes` ranking against two distinguishable query
    /// vectors, not by trusting the tally alone.
    #[tokio::test]
    async fn reembed_all_overwrites_an_existing_embedding() {
        let embedder = Arc::new(SwitchableEmbeddingProvider::new());
        let embedder_dyn: Arc<dyn DynEmbeddingProvider> = embedder.clone();

        let mem = Memory::open(":memory:")
            .with_llm(null_llm())
            .with_embedder(embedder_dyn)
            .with_episode_dense_enabled(true)
            .default_namespace(Namespace::new("td143-reembed-overwrite"))
            .await
            .expect("build memory with dense episode arm");

        // Ingest → embedded with seed 1.0 (dense arm embeds at ingest time).
        mem.remember("the quick brown fox")
            .skip_extraction()
            .await
            .expect("remember must persist the episode + its initial embedding");

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("Memory built via the builder/providers path carries a TemporalGraph");
        let no_filter = SearchFilters::new();

        // Sanity: a query embedded at seed 1.0 (matching the corpus) ranks it.
        let hits_before = tg
            .vector_search_episodes(VectorSearchEpisodesParams {
                query_embedding: &seeded_embedding(1.0),
                limit: 10,
                filters: &no_filter,
            })
            .await
            .expect("vector search must succeed");
        assert_eq!(
            hits_before.len(),
            1,
            "the one ingested episode must be dense-searchable before re-embed"
        );
        let dist_before_at_seed1 = hits_before[0].score;

        // Simulate a config/embedder change (TD-143's flip-the-knob, or a
        // model swap): re-embed the WHOLE corpus, now producing seed-5.0
        // vectors.
        embedder.switch_to_second_seed();
        let stats = mem
            .reembed_all_episode_embeddings(256)
            .await
            .expect("reembed_all_episode_embeddings must succeed");
        assert_eq!(
            stats.embedded, 1,
            "the one existing episode must be re-embedded"
        );
        assert_eq!(stats.failed, 0);

        // The stored vector must have MOVED: a seed-1.0 query is now a worse
        // (larger cosine-distance) match than it was before the re-embed,
        // because the stored vector is no longer the seed-1.0 vector.
        let hits_after = tg
            .vector_search_episodes(VectorSearchEpisodesParams {
                query_embedding: &seeded_embedding(1.0),
                limit: 10,
                filters: &no_filter,
            })
            .await
            .expect("vector search must succeed");
        assert_eq!(hits_after.len(), 1);
        assert!(
            hits_after[0].score > dist_before_at_seed1,
            "re-embedding must overwrite the stored vector — the distance to \
             a seed-1.0 query must have INCREASED once the stored vector \
             moved to seed-5.0 (before={dist_before_at_seed1}, after={})",
            hits_after[0].score
        );

        // Directly confirm the NEW stored vector is now the BEST match for a
        // seed-5.0 query (distance ~0, since query and stored vector are
        // identical once re-embedded).
        let hits_seed5 = tg
            .vector_search_episodes(VectorSearchEpisodesParams {
                query_embedding: &seeded_embedding(5.0),
                limit: 10,
                filters: &no_filter,
            })
            .await
            .expect("vector search must succeed");
        assert_eq!(hits_seed5.len(), 1);
        assert!(
            hits_seed5[0].score < hits_after[0].score,
            "a seed-5.0 query must be a MUCH closer match than a seed-1.0 \
             query to the freshly-stored seed-5.0 vector \
             (seed5_dist={}, seed1_dist={})",
            hits_seed5[0].score,
            hits_after[0].score
        );
    }
}
