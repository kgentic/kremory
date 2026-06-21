//! Ingest module — Engine struct, constructors, and the public ingest API.
//!
//! Split from `ingest.rs` as part of TD-001 (E0-C).
//!
//! Sub-modules:
//! - `helpers` — context snippet extraction + SimpleGraph test alias
//! - `pipeline` — `ingest_with` + `ingest_deferred` impl blocks

use std::sync::Arc;
use tokio::sync::Mutex;

use chrono::{DateTime, Utc};
use metrics;

use crate::core::config::{ContentType, PipelineConfig};
use crate::core::error::Result;
use crate::core::graph::{InsertEntityParams, UpdateEntitySourceTierParams};

use crate::core::provider::{
    capability_of, ChatProvider, EmbeddingProvider, ProviderCaps, TokenCountingChatProvider,
    TokenUsage,
};
use crate::core::schema::TemporalGraph;
use crate::core::text_utils;

mod helpers;
mod pipeline;

#[cfg(test)]
mod tests;

// ─── Test-only re-exports ─────────────────────────────────────────────────────
// `tests.rs` uses `use super::*;` — these items must be in scope at the mod level
// for the test block to compile, mirroring the original ingest.rs top-level imports.

#[cfg(test)]
pub(crate) use crate::core::intelligence::{EntityExtractor, ExtractedEntity, ExtractedFact};

// Re-export helpers items needed by integration tests and other modules.
#[cfg(any(test, feature = "test-utils"))]
pub use helpers::SimpleGraph;

// Phase A foundational types (C6 spec §5.5 GAP-001) — public so integration
// tests and verify_stage can reference them.
pub use pipeline::{EntityCandidate, IngestPhase1Result, ResolvedDecision, UpsertedEntities};

// Args-as-object params structs for the pipeline `impl Engine` methods (TD-042).
// Re-exported here so consumers reach them at `kremory::core::ingest::*`
// alongside `Engine` and `SourceParams`.
pub use pipeline::{IngestDeferredParams, IngestWithParams, WriteVerifiedEntitiesParams};

/// Optional source provenance fields forwarded to the `episodes` table
/// (Migration 007 columns: source_id, source_uri, recorded_at).
///
/// All fields are `None` / empty by default — callers that don't have source
/// provenance pass `SourceParams::default()` and the columns stay NULL.
///
/// `Debug` is hand-written (not derived) because the `sink` field holds a
/// trait object (`dyn IngestEventSink`) that is not `Debug`. The manual impl
/// reports whether a sink is wired without requiring `Debug` on the trait.
#[derive(Default, Clone)]
pub struct SourceParams {
    /// Stable identifier for the originating source document or event.
    /// Caller-defined; substrate treats it as an opaque key for round-trip lookup.
    pub source_id: Option<String>,
    /// URI of the originating source, if known.
    pub source_uri: Option<String>,
    /// Wall-clock time the episode was recorded (defaults to ingest time when None).
    pub recorded_at: Option<DateTime<Utc>>,
    /// Per-call entity type vocabulary override (TD-013 §1 hybrid registry).
    ///
    /// When `Some(specs)`:
    /// - Build registry from override: `EntityTypeRegistry::from_specs(specs)`.
    /// - If the DB registry for the active `group_id` is empty, persist the specs
    ///   (first-call init via `upsert_entity_types`). Subsequent calls without
    ///   an override will see the persisted vocabulary.
    /// - If the DB registry already has rows, use the override ephemerally for
    ///   THIS call only (L2 prompt + L3 validation). DB is NOT modified.
    ///   Fires metric `rql.ingest.registry_override_applied{persisted=false}`.
    ///
    /// When `None`: fall back to `EntityTypeRegistry::load_for_group` (current
    /// behaviour — reads from DB).
    ///
    /// The override list MUST include `id=0 "Entity"` catch-all (caller
    /// responsibility).
    pub entity_types_override: Option<Vec<crate::core::entity_types::EntityTypeSpec>>,
    /// ADR-035 §5 Option A: caller-pre-extracted facts to pin into the graph
    /// BEFORE Phase 2 LLM extraction runs. Engine.ingest writes these via
    /// `graph.try_insert_fact_with_group` after episode insertion; subsequent
    /// Phase 2 LLM duplicates of the same triple are silently swallowed by
    /// `try_insert_fact` (caller wins via pre-write ordering).
    ///
    /// Empty (default) = no caller pins, behavior identical to pre-v0.1.8.
    ///
    /// engine_handle's `graph_ingest_episode` impl translates the higher-level
    /// `memory::types::StructuredFact` into this core-level type to avoid a
    /// circular `core → memory` dependency.
    pub pre_pinned_facts: Vec<PrePinnedFact>,
    /// ADR-035 §2/§6 — when `true`, engine.ingest bails after episode insert
    /// and caller-pre-pinned-facts write, without invoking Phase 2 LLM extraction.
    /// Backs the facade `RememberRequest::skip_extraction()` builder method;
    /// default `false` preserves pre-v0.1.8 full-pipeline behavior.
    pub skip_extraction: bool,
    /// ADR-052 Gap 1 — per-episode event sink fired synchronously at the real
    /// extraction/edge/stage-transition call sites inside [`Engine::ingest_with`].
    ///
    /// This carries the **core-layer** [`IngestEventSink`](crate::core::sink::IngestEventSink)
    /// trait (NOT the memory-layer `EnrichmentEventSink` supertrait) so the core
    /// layer stays free of a `core → memory` dependency. The memory layer
    /// (`engine_handle::graph_ingest_episode`) coerces its
    /// `Arc<dyn EnrichmentEventSink>` into this `Arc<dyn IngestEventSink>` via
    /// the supertrait `as`-cast before calling `ingest`.
    ///
    /// `None` (default) = no sink wired; all fire-site call sites are guarded by
    /// `if let Some(s) = &source_params.sink`. Re-established by the ADR-052
    /// Gap 1 sink-callsite cause-fix after the fb85ba8 consolidation regressed
    /// the original 737e152/34fdc60 fire-sites.
    pub sink: Option<Arc<dyn crate::core::sink::IngestEventSink>>,
}

impl std::fmt::Debug for SourceParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceParams")
            .field("source_id", &self.source_id)
            .field("source_uri", &self.source_uri)
            .field("recorded_at", &self.recorded_at)
            .field("entity_types_override", &self.entity_types_override)
            .field("pre_pinned_facts", &self.pre_pinned_facts)
            .field("skip_extraction", &self.skip_extraction)
            .field("sink", &self.sink.as_ref().map(|_| "<IngestEventSink>"))
            .finish()
    }
}

/// Core-level pre-pinned fact passed via [`SourceParams`] (ADR-035 §5).
///
/// This is the engine's internal representation of `memory::types::StructuredFact`,
/// kept in `core` to preserve substrate's layer boundaries.
#[derive(Debug, Clone)]
pub struct PrePinnedFact {
    /// Subject entity identifier or literal.
    pub subject: String,
    /// Predicate / relationship name.
    pub predicate: String,
    /// Object entity identifier (when caller resolved it to an existing entity).
    pub object_id: Option<String>,
    /// Object literal value (when not an entity reference).
    pub object_value: Option<String>,
    /// World-time start of the fact's validity window.
    pub valid_from: DateTime<Utc>,
    /// Caller's confidence score [0.0, 1.0]. Defaults to 1.0 at the translation layer.
    pub confidence: f64,
}

/// Result of a single ingest() call.
#[derive(Debug)]
pub struct IngestionResult {
    /// The episode stored for this ingestion.
    pub episode_id: i64,
    /// Entity IDs that were created or merged.
    pub upserted_entities: Vec<String>,
    /// Fact IDs that were inserted.
    pub inserted_fact_ids: Vec<i64>,
    /// Fact IDs that were invalidated (contradicted/updated).
    pub invalidated_fact_ids: Vec<i64>,
    /// Entity pairs merged (canonical_id, alias_id).
    pub merged_entities: Vec<(String, String)>,
    /// Token usage across all LLM calls.
    pub token_usage: TokenUsage,
    /// Forward references in the fact list that had no corresponding extracted entity
    /// and were inserted as UNKNOWN stub entities so facts can resolve correctly.
    pub stub_entities_inserted: usize,
}

// ── Dream pass types (Phase C DoD C1/C2) ──────────────────────────────────────

/// Options for a synchronous dream pass.
///
/// Passed to [`Engine::run_dream_pass_sync`]. All fields have sensible defaults
/// via [`Default`] — callers can construct with `DreamPassOpts::default()` and
/// selectively override.
///
/// # ADR reference
///
/// ADR-045 §3 (dream pass architecture); Phase C DoD C2
/// (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
#[derive(Debug, Clone)]
pub struct DreamPassOpts {
    /// When `true`, Pass 0 type-discovery runs to find novel entity types not
    /// in the current registry. Requires LLM. Default: `false`.
    ///
    /// Phase D wires the actual Pass 0 logic; Phase C stubs this path.
    pub include_type_discovery: bool,
    /// Minimum confidence threshold for entity type assignments to be re-examined
    /// during the reclassify pass. Range [0.0, 1.0]. Default: `0.5`.
    pub confidence_threshold: f32,
    /// Cap on the number of ghost episodes processed per run.
    /// `None` = process all available ghost episodes. Default: `None`.
    pub max_episodes_per_run: Option<usize>,
    /// Confidence threshold above which `ConsumerPinned` entities are protected
    /// from reclassification during the dream pass. Range [0.0, 1.0].
    /// Default: `0.7` (ADR-045 §3).
    pub reclassify_high_conf_threshold: f32,
}

impl Default for DreamPassOpts {
    fn default() -> Self {
        Self {
            include_type_discovery: false,
            confidence_threshold: 0.5,
            max_episodes_per_run: None,
            // ADR-045 §3: 0.7 is the ConsumerPinned protection threshold.
            reclassify_high_conf_threshold: 0.7,
        }
    }
}

/// Summary returned by [`Engine::run_dream_pass_sync`].
///
/// Phase C returns zeroed counts (Pass 0 + Pass 2 are stubbed). Phase D/E
/// will populate with real counts from the LLM consolidation pass.
#[derive(Debug, Clone, Default)]
pub struct DreamPassSummary {
    /// Number of ghost episodes reprocessed (Phase 2 retry).
    pub ghost_episodes_retried: usize,
    /// Number of entity types discovered during Pass 0 (type discovery).
    pub types_discovered: usize,
    /// Number of entities reclassified during the dream reclassify pass.
    pub entities_reclassified: usize,
    /// Wall-clock duration of the dream pass in milliseconds.
    pub duration_ms: u64,
}

/// High-level kremory graph engine with intelligence pipeline.
/// Wraps TemporalGraph and adds extraction, resolution, and contradiction detection.
pub struct Engine<L: ChatProvider, Emb: EmbeddingProvider> {
    pub(crate) graph: Arc<TemporalGraph>,
    /// `None` when constructed via the NoLlm BYOE path (`Engine::with_custom_extractor`).
    /// LLM-dependent pipeline steps (CascadeResolver, TwoPoolDetector) return
    /// `Error::LlmRequired` when this is `None`.
    pub(crate) llm: Option<Arc<L>>,
    pub(crate) embedder: Arc<Emb>,
    pub(crate) config: PipelineConfig,
    /// Optional OOV auditor for language-agnostic entity safety net.
    /// When set, runs after each chunk extraction to catch domain terms the LLM missed.
    pub(crate) oov_auditor: Option<text_utils::OovAuditor>,
    /// Model identifier read from `llm.model()` at construction.
    /// None when llm.model() returns empty string.
    pub(crate) model: Option<String>,
    /// Entity extractor — constructed once per Engine, shared via Arc across
    /// all ingest hot-path callsites.  Built-in variants are `Llm` and
    /// `GlinerLlm` (behind `ner` feature); consumer extensions via `Custom`.
    /// See ADR-039 and `factory.rs`.
    pub(crate) extractor: Arc<crate::core::extraction::factory::ExtractorKind<L>>,
    /// Serialization lock for dream passes (Phase C DoD C3).
    ///
    /// At-most-one dream pass runs concurrently per engine instance. Concurrent
    /// callers block on this mutex — dream passes are expected to be rare
    /// (nightly / manual trigger) so mutex contention is not a concern.
    ///
    /// Arc-wrapped so the lock survives `Engine` moves into `Arc<Engine>`.
    pub(crate) dream_lock: Arc<Mutex<()>>,
}

/// Bundled parameters for [`Engine::new`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments). Generic over the provider params
/// because every field is provider-typed (`Arc<L>` / `Arc<Emb>`).
pub struct EngineNewParams<L: ChatProvider, Emb: EmbeddingProvider> {
    pub graph: Arc<TemporalGraph>,
    pub llm: Arc<L>,
    pub embedder: Arc<Emb>,
    pub config: PipelineConfig,
}

/// Bundled parameters for [`Engine::with_extractor`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
pub(crate) struct EngineWithExtractorParams<L: ChatProvider, Emb: EmbeddingProvider> {
    pub graph: Arc<TemporalGraph>,
    pub llm: Arc<L>,
    pub embedder: Arc<Emb>,
    pub config: PipelineConfig,
    pub extractor: Arc<crate::core::extraction::factory::ExtractorKind<L>>,
}

/// Bundled parameters for [`Engine::with_custom_extractor_no_llm`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments).
pub(crate) struct EngineWithCustomExtractorNoLlmParams<L: ChatProvider, Emb: EmbeddingProvider> {
    pub graph: Arc<TemporalGraph>,
    pub embedder: Arc<Emb>,
    pub config: PipelineConfig,
    pub extractor: Arc<crate::core::extraction::factory::ExtractorKind<L>>,
}

/// Bundled parameters for [`Engine::ingest_document`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
pub struct IngestDocumentParams<'a> {
    pub source: &'a str,
    pub title: &'a str,
    pub text: &'a str,
}

/// Bundled parameters for [`Engine::assert_entity_type`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
pub struct AssertEntityTypeParams<'a> {
    pub entity_id: &'a str,
    /// Currently unused by the storage layer (Phase C DoD C5 future-compat);
    /// see [`Engine::assert_entity_type`] doc.
    pub entity_type_id: u32,
    pub group_id: Option<&'a str>,
}

/// Bundled parameters for [`Engine::ingest`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
pub struct IngestParams<'a> {
    pub text: &'a str,
    pub reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<&'a str>,
    pub content_type: Option<ContentType>,
    pub source_params: SourceParams,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Create a new `Engine` wrapping a `TemporalGraph`.
    ///
    /// Accepts `Arc<TemporalGraph>` so callers can share the same graph
    /// instance with the process-global singleton returned by `engine()`.
    ///
    /// The entity extractor is supplied via the `ExtractorKind` argument.
    /// Use [`MemoryBuilder::with_llm`], [`MemoryBuilder::with_gliner`], or
    /// [`MemoryBuilder::with_extractor`] to configure the extractor at the
    /// facade layer; `Engine::new` receives the resolved [`ExtractorKind`].
    ///
    /// Returns `Err` if `ExtractorKind::GlinerLlm` is requested and the
    /// GLiNER weights fail to load (e.g. no network on first run + no
    /// hf-hub cache).
    pub fn new(params: EngineNewParams<L, Emb>) -> Self {
        let EngineNewParams {
            graph,
            llm,
            embedder,
            config,
        } = params;
        let m = llm.model().trim().to_string();
        let model = if m.is_empty() { None } else { Some(m) };
        let extractor = Arc::new(crate::core::extraction::factory::ExtractorKind::Llm(
            crate::core::extraction::graphiti::LlmExtractor::new(Arc::clone(&llm)),
        ));
        Self {
            graph,
            llm: Some(llm),
            embedder,
            config,
            oov_auditor: None,
            model,
            extractor,
            dream_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Construct an Engine with an explicit `ExtractorKind` and a wired LLM.
    ///
    /// Used by the `MemoryBuilder` when `.with_gliner()` or `.with_extractor()`
    /// is combined with `.with_llm()` — the caller selects the extractor variant
    /// explicitly while the LLM is still available for Category B pipeline steps
    /// (CascadeResolver, TwoPoolDetector).
    pub(crate) fn with_extractor(params: EngineWithExtractorParams<L, Emb>) -> Self {
        let EngineWithExtractorParams {
            graph,
            llm,
            embedder,
            config,
            extractor,
        } = params;
        let m = llm.model().trim().to_string();
        let model = if m.is_empty() { None } else { Some(m) };
        Self {
            graph,
            llm: Some(llm),
            embedder,
            config,
            oov_auditor: None,
            model,
            extractor,
            dream_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Construct an Engine with a `Custom` extractor and NO LLM provider.
    ///
    /// Used by the `MemoryBuilder` when `.with_extractor(…)` is set but no
    /// `.with_llm()` was called (the NoLlm BYOE path). All Category A pipeline
    /// operations work normally. Category B steps (CascadeResolver, TwoPoolDetector)
    /// return `Error::LlmRequired` at call time — the caller is responsible for
    /// ensuring those paths are not reached, or for handling the error.
    pub(crate) fn with_custom_extractor_no_llm(
        params: EngineWithCustomExtractorNoLlmParams<L, Emb>,
    ) -> Self {
        let EngineWithCustomExtractorNoLlmParams {
            graph,
            embedder,
            config,
            extractor,
        } = params;
        Self {
            graph,
            llm: None,
            embedder,
            config,
            oov_auditor: None,
            model: None,
            extractor,
            dream_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Attach an OOV auditor for language-agnostic entity extraction safety net.
    pub fn with_oov_auditor(mut self, auditor: text_utils::OovAuditor) -> Self {
        self.oov_auditor = Some(auditor);
        self
    }

    /// Access the underlying `Arc<TemporalGraph>`.
    ///
    /// Returns a clone of the `Arc` so callers can hold a shared reference
    /// without borrowing the `Engine`.
    pub fn graph(&self) -> Arc<TemporalGraph> {
        Arc::clone(&self.graph)
    }

    /// Model identifier captured from `llm.model()` at construction.
    ///
    /// Returns `None` when the LLM provider reported an empty or whitespace-only
    /// model string (semantically "no model configured"). Pub-scoped pending a
    /// facade `Memory::model()` caller.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Unified document ingestion: store document as a searchable entity with
    /// full-text embedding, then run the intelligence pipeline to extract
    /// sub-entities and facts.
    ///
    /// This is the single entry point for adding documents to the knowledge base.
    /// Consumers (seed scripts, UI "Add Documents", tests) call this instead of
    /// manually coordinating `insert_entity` + `set_entity_embedding` + `ingest`.
    ///
    /// Steps:
    ///   1. Store document as an entity (full text in `properties.text` for FTS5)
    ///   2. Embed the full document text (for vector search)
    ///   3. Run the intelligence pipeline (entity extraction, resolution, contradiction)
    pub async fn ingest_document(
        &self,
        params: IngestDocumentParams<'_>,
    ) -> Result<IngestionResult> {
        let IngestDocumentParams {
            source,
            title,
            text,
        } = params;
        // 1. Store the document as a searchable entity.
        // Documents use entity_type_id=0 (catch-all); the title is stored in properties.
        let properties = serde_json::json!({ "text": text, "title": title });
        self.graph
            .insert_entity(InsertEntityParams {
                id: source,
                entity_type_id: 0,
                properties,
            })
            .await?;
        metrics::counter!("rql.ingest.entity_persisted_total", "source" => "source_anchor")
            .increment(1);

        // 2. Embed the full document text for vector search.
        let embedding = self.embedder.embed(text).await?;
        self.graph.set_entity_embedding(source, &embedding).await?;

        // 3. Run the intelligence pipeline on the document content.
        let doc_text = format!("# {title}\n\n{text}");
        self.ingest(IngestParams {
            text: &doc_text,
            reference_time: None,
            group_id: None,
            content_type: Some(ContentType::Document),
            source_params: SourceParams::default(),
        })
        .await
    }

    // ── Dream pass API (Phase C DoD C1–C5, C8) ───────────────────────────────

    /// Run a synchronous dream pass on the graph.
    ///
    /// **Phase C stub**: Pass 0 (type discovery) and Pass 2 (LLM re-extraction of
    /// ghost episodes) are not yet implemented. This method acquires the
    /// serialization lock (C3), records timing + metrics (C8), and returns a
    /// zeroed [`DreamPassSummary`]. Phase D wires the actual pass logic.
    ///
    /// Concurrent callers block until the running pass completes — at-most-one
    /// dream pass runs per engine instance (ADR-045 §3 / Phase C DoD C3).
    ///
    /// # ADR reference
    ///
    /// ADR-045 §3; Phase C DoD C1 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn run_dream_pass_sync(&self, opts: DreamPassOpts) -> Result<DreamPassSummary> {
        let start = std::time::Instant::now();

        // C3: acquire dream serialization lock (blocks concurrent passes).
        // Uses tokio::sync::Mutex so the guard is Send across .await points.
        // The lock is released when `_guard` is dropped at end of scope.
        let _guard = self.dream_lock.lock().await;

        // C8: emit dream pass start counter.
        metrics::counter!("rql.dream.pass_started_total").increment(1);

        // Phase C stub: Pass 0 (type discovery) — wired in Phase D.
        if opts.include_type_discovery {
            metrics::counter!(
                "rql.dream.pass0_type_discovery_total",
                "status" => "stub"
            )
            .increment(1);
            tracing::debug!(
                confidence_threshold = opts.confidence_threshold,
                "kremory.dream.pass0 type discovery: stubbed in Phase C — Phase D wires impl"
            );
        }

        // Phase C stub: ghost episode retry pass — wired in Phase D.
        let ghost_episodes_retried = 0usize;

        // Phase E: entity reclassify pass (ADR-046 Option E, 2-arm SELECT).
        // Runs only when LLM is wired; returns 0 silently on NoLlm path.
        //
        // ADR-050 Phase 4 — budget tracking:
        // The provider is wrapped in TokenCountingChatProvider for the ENTIRE
        // duration of reclassify_all_groups, so ALL LLM calls made during the
        // pass (reclassify + sub-calls) are counted through the decorator.
        //
        // SEAM VERIFIED: reclassify_all_groups<L: ChatProvider>(graph, llm: &L, opts)
        // is the only LLM-call path reachable from run_dream_pass_sync. verify_stage
        // LLM calls run in the background pipeline (process_deferred path), not here.
        // Wrapping this provider captures ALL dream-pass LLM budget — no under-count.
        //
        // TokenCountingChatProvider: ChatProvider (via #[async_trait] impl), so
        // &TokenCountingChatProvider satisfies the &L: ChatProvider generic bound.
        let entities_reclassified = if let Some(llm_arc) = self.llm.as_ref() {
            // ── ADR-050 Phase 4: budget-tracking setup ────────────────────────
            //
            // Generate pass_run_id before the pass: same ID used for both the
            // cooldown checkpoint (Phase 3) and the budget_usage INSERT (Phase 4).
            // microsecond timestamp → unique per pass, human-readable in SQL.
            let pass_run_id = format!("dream_pass_{}", Utc::now().timestamp_micros());

            // Coerce Arc<L> → Arc<dyn ChatProvider> then wrap in the decorator.
            // Arc::clone alone cannot coerce to dyn; we must clone as the concrete
            // Arc<L> first, then cast to Arc<dyn ChatProvider> via `as`.
            // L: ChatProvider + 'static ensures the unsized cast is valid.
            let cloned: Arc<L> = Arc::clone(llm_arc);
            let dyn_arc: Arc<dyn ChatProvider> = cloned as Arc<dyn ChatProvider>;
            let counting = TokenCountingChatProvider::new(dyn_arc);
            // Clone the accumulator handle BEFORE moving counting into the reclassify call.
            // The budget-INSERT site reads totals from this handle after pass completes.
            let acc_handle = counting.accumulator();
            let model_str = counting.model().to_string();

            match crate::core::dream::reclassify::reclassify_all_groups(
                &self.graph,
                &counting,
                crate::core::dream::reclassify::ReclassifyOpts {
                    confidence_threshold: opts.confidence_threshold,
                    high_conf_threshold: opts.reclassify_high_conf_threshold,
                    max_batch_size: crate::core::dream::reclassify::MAX_RECLASSIFY_BATCH,
                },
            )
            .await
            {
                Ok(r) => {
                    // ── ADR-050 Phase 3 — Guard #3: cooldown-on-success ───────
                    //
                    // Records pass completion ONLY on success so transient
                    // failures do NOT update the cooldown timestamp. A consumer
                    // checking "was the last dream pass successful?" reads the
                    // most-recent row for op_name='dream_pass_cooldown'; if no
                    // row exists, no successful pass has run yet.
                    //
                    // op_run_id = microsecond timestamp → each successful run
                    // gets its own row (INSERT OR REPLACE upserts on PK clash,
                    // so same-microsecond duplicate is safe — idempotent).
                    // cursor = epoch-seconds of completion (human-readable signal
                    // of "last success at T"). updated_at same value.
                    //
                    // Soft-fail: cooldown write failure is logged + metered but
                    // NOT propagated — it degrades schedule enforcement (scheduler
                    // may fire again sooner than intended) but does NOT corrupt
                    // data. Rule 8: the cause is a DB write failure, NOT a
                    // dream-pass logic error.
                    //
                    // D7: no high-cardinality metric labels.
                    let now_epoch = Utc::now().timestamp();
                    let cursor_val = now_epoch.to_string();
                    if let Err(e) = self
                        .graph
                        .conn
                        .execute(
                            "INSERT OR REPLACE INTO op_checkpoints \
                             (op_name, op_run_id, cursor, updated_at) \
                             VALUES ('dream_pass_cooldown', ?1, ?2, ?3)",
                            libsql::params![pass_run_id.clone(), cursor_val, now_epoch],
                        )
                        .await
                    {
                        tracing::warn!(
                            error = %e,
                            "kremory.dream.cooldown_write_fail — schedule enforcement degraded"
                        );
                        metrics::counter!("kremory.dream.cooldown_write_fail_total").increment(1);
                    } else {
                        // Phase 5 seam: on_dream_pass_complete (or equivalent) fires here
                        // when Phase 5 sink-wiring lands for the inline dream-pass path.
                        metrics::counter!("kremory.dream.cooldown_recorded_total").increment(1);
                        tracing::debug!(
                            completed_at = now_epoch,
                            "kremory.dream.cooldown_on_success recorded"
                        );
                    }

                    // ── ADR-050 Phase 4 — budget INSERT ───────────────────────
                    //
                    // Read accumulator totals after reclassify completes (all LLM
                    // calls for this pass have already settled). Soft-fail: a budget
                    // write failure is observable via counter + warn log but MUST NOT
                    // propagate — budget accounting is advisory, not crash-safety.
                    //
                    // Schema (migrations.rs ground truth, Migration 016):
                    //   dream_pass_budget_usage (
                    //     pass_run_id TEXT NOT NULL,
                    //     pass_name   TEXT NOT NULL,
                    //     provider    TEXT NOT NULL,
                    //     model       TEXT NOT NULL,
                    //     tokens_input  INTEGER NOT NULL,
                    //     tokens_output INTEGER NOT NULL,
                    //     cost_usd_micro INTEGER,   -- nullable: NULL when rate unknown
                    //     recorded_at INTEGER NOT NULL,
                    //     PRIMARY KEY (pass_run_id, pass_name)
                    //   )
                    //
                    // D7: provider / model are NOT metric labels (high cardinality).
                    // Budget metric is a simple increment of cost in micro-USD.
                    //
                    // calls_with_usage < calls_total → backend did not report usage
                    // for some calls (typical for local Ollama builds). The row is
                    // still written with the partial totals so the gap is observable
                    // via `calls_with_usage` vs `calls_total` in the accumulator log.
                    //
                    // Scoped block ensures MutexGuard<TokenAccumulator> is dropped
                    // before any .await point — required for the async fn to be Send.
                    let (tokens_input, tokens_output, calls_total, calls_with_usage) = {
                        let acc = acc_handle
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        (
                            acc.prompt_tokens,
                            acc.completion_tokens,
                            acc.calls_total,
                            acc.calls_with_usage,
                        )
                        // MutexGuard drops here at end of block
                    };

                    let provider_name = detect_provider_name(&model_str);
                    // cost_usd_micro: None = rate unknown (NULL in DB).
                    // Anthropic Haiku rate (Claude Haiku 3.5): $0.80/M input + $4/M output
                    //   = 800 micro_usd/M input = 0.8 micro_usd/K input tokens.
                    // Source: https://www.anthropic.com/pricing (2026-06-16 spot check).
                    // claude-haiku-4-* uses the same tier.
                    // Ollama: always 0 (local inference, no API cost).
                    // Other providers (OpenAI, unknown): NULL (rate not implemented).
                    let cost_usd_micro = compute_dream_cost_micro(ComputeDreamCostMicroParams {
                        provider: &provider_name,
                        model: &model_str,
                        tokens_input,
                        tokens_output,
                    });

                    tracing::debug!(
                        pass_run_id = %pass_run_id,
                        provider = %provider_name,
                        model = %model_str,
                        tokens_input,
                        tokens_output,
                        calls_total,
                        calls_with_usage,
                        cost_usd_micro = ?cost_usd_micro,
                        "kremory.dream.budget_tracking: pass complete"
                    );

                    if let Err(e) = self
                        .graph
                        .conn
                        .execute(
                            "INSERT OR REPLACE INTO dream_pass_budget_usage \
                             (pass_run_id, pass_name, provider, model, \
                              tokens_input, tokens_output, cost_usd_micro, recorded_at) \
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                            libsql::params![
                                pass_run_id.clone(),
                                "reclassify",
                                provider_name,
                                model_str,
                                tokens_input as i64,
                                tokens_output as i64,
                                cost_usd_micro.map(|c| c as i64),
                                now_epoch
                            ],
                        )
                        .await
                    {
                        tracing::warn!(
                            error = %e,
                            "kremory.dream.budget_write_fail — budget accounting degraded (non-fatal)"
                        );
                        metrics::counter!("kremory.dream.budget_write_fail_total").increment(1);
                    } else {
                        // Emit budget_used counter in micro-USD (0 for Ollama,
                        // computed value for known Anthropic models, 0 for NULL).
                        // D7: no high-cardinality labels — provider/model stay out of metric labels.
                        let cost_for_metric = cost_usd_micro.unwrap_or(0);
                        metrics::counter!("rql.dream.budget_used_micro_usd")
                            .increment(cost_for_metric);
                        tracing::debug!(
                            pass_run_id = %pass_run_id,
                            tokens_input,
                            tokens_output,
                            cost_usd_micro = cost_for_metric,
                            "kremory.dream.budget_usage recorded"
                        );
                    }

                    r.entities_reclassified
                }
                Err(e) => {
                    // Non-fatal per E7 — surface as debug log, continue.
                    // ADR-050 Guard #3: transient failure → cooldown NOT recorded.
                    // ADR-050 Phase 4: reclassify failure → budget row NOT written
                    // (partial token counts not meaningful for a failed pass).
                    tracing::warn!(
                        error = %e,
                        "kremory.dream.reclassify_all_groups failed — skipping; dream pass unaffected; cooldown + budget NOT recorded (transient failure)"
                    );
                    0
                }
            }
        } else {
            0
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        // C8: emit dream pass completion histogram + summary counters.
        metrics::histogram!("rql.dream.pass_duration_ms").record(duration_ms as f64);
        metrics::counter!("rql.dream.ghost_episodes_retried_total")
            .increment(ghost_episodes_retried as u64);
        metrics::counter!("rql.dream.entities_reclassified_total")
            .increment(entities_reclassified as u64);
        metrics::counter!("rql.dream.pass_completed_total").increment(1);

        tracing::info!(
            duration_ms,
            ghost_episodes_retried,
            entities_reclassified,
            include_type_discovery = opts.include_type_discovery,
            max_episodes_per_run = opts.max_episodes_per_run,
            "kremory.dream.pass_sync completed"
        );

        Ok(DreamPassSummary {
            ghost_episodes_retried,
            types_discovered: 0,
            entities_reclassified,
            duration_ms,
        })
    }

    /// Return episode IDs where Phase 1 (NER + episode commit) succeeded but
    /// Phase 2 (LLM fact extraction) produced zero facts.
    ///
    /// These are "ghost episodes" — they exist in the `episodes` table and are
    /// searchable, but have no associated facts. The dream pass retries Phase 2
    /// for ghost episodes. Callers can also use this list to decide whether to
    /// trigger a dream pass.
    ///
    /// Ghost episodes are identified as: episodes that have no rows in `facts`
    /// with a matching `source_episode_id`.
    ///
    /// An optional `group_id` restricts the query to a single namespace/thread
    /// (use `namespace_to_group_id()` to derive it from a `Namespace`). `None`
    /// returns ghost episodes across all namespaces.
    ///
    /// # ADR reference
    ///
    /// ADR-045 §3; Phase C DoD C4 (`v0-1-1-dream-impl-sprint-plan-2026-06-09.md`).
    pub async fn ghost_episodes(&self, group_id: Option<&str>) -> Result<Vec<i64>> {
        let start = std::time::Instant::now();

        let ids = if let Some(gid) = group_id {
            let mut rows = self
                .graph
                .conn
                .query(
                    "SELECT e.id FROM episodes e \
                     WHERE e.group_id = ?1 \
                       AND NOT EXISTS ( \
                           SELECT 1 FROM facts f \
                           WHERE f.source_episode_id = e.id \
                       ) \
                     ORDER BY e.id",
                    libsql::params![gid],
                )
                .await
                .map_err(crate::core::error::Error::from)?;
            let mut ids = Vec::new();
            while let Some(row) = rows.next().await.map_err(crate::core::error::Error::from)? {
                let id: i64 = row.get(0).map_err(crate::core::error::Error::from)?;
                ids.push(id);
            }
            ids
        } else {
            let mut rows = self
                .graph
                .conn
                .query(
                    "SELECT e.id FROM episodes e \
                     WHERE NOT EXISTS ( \
                         SELECT 1 FROM facts f \
                         WHERE f.source_episode_id = e.id \
                     ) \
                     ORDER BY e.id",
                    (),
                )
                .await
                .map_err(crate::core::error::Error::from)?;
            let mut ids = Vec::new();
            while let Some(row) = rows.next().await.map_err(crate::core::error::Error::from)? {
                let id: i64 = row.get(0).map_err(crate::core::error::Error::from)?;
                ids.push(id);
            }
            ids
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // C8: ghost episodes gauge + query duration.
        metrics::gauge!("rql.dream.ghost_episodes_count").set(ids.len() as f64);
        metrics::histogram!("rql.dream.ghost_episodes_query_ms").record(elapsed_ms as f64);

        tracing::debug!(
            count = ids.len(),
            elapsed_ms,
            group_id = group_id.unwrap_or("<all>"),
            "kremory.dream.ghost_episodes queried"
        );

        Ok(ids)
    }

    /// Pin an entity's type as `ConsumerPinned`, protecting it from dream-pass
    /// reclassification.
    ///
    /// Writes `entity_type_source = 'ConsumerPinned'` and
    /// `entity_type_assigned_at = <now>` on the entity row identified by
    /// `entity_id` within `group_id` (the SQL `entities.group_id` column).
    ///
    /// When `group_id` is `None`, defaults to `"default"` — matching the
    /// `update_entity_source_tier` behaviour for legacy rows that predate the
    /// namespace migration.
    ///
    /// The `entity_type_id` parameter is currently unused by the storage layer
    /// (the tier-update path only stamps `source` + `assigned_at`). It is
    /// present in the public API per Phase C DoD C5 for future-compat — Phase E
    /// may wire it to update the `entity_type_id` column when an explicit type
    /// is supplied alongside the pin.
    ///
    /// # ADR reference
    ///
    /// ADR-045 §3 (ConsumerPinned protection); Phase C DoD C5.
    pub async fn assert_entity_type(&self, params: AssertEntityTypeParams<'_>) -> Result<()> {
        let AssertEntityTypeParams {
            entity_id,
            entity_type_id: _entity_type_id,
            group_id,
        } = params;
        self.graph
            .update_entity_source_tier(UpdateEntitySourceTierParams {
                id: entity_id,
                group_id,
                source_tier: "ConsumerPinned",
            })
            .await?;

        tracing::debug!(
            entity_id,
            group_id = group_id.unwrap_or("default"),
            "kremory.dream.assert_entity_type ConsumerPinned stamped"
        );

        Ok(())
    }

    /// Full pipeline: text → chunk → extract → resolve → contradict → store.
    /// Uses the engine's configured `EntityExtractor`. For a caller-supplied
    /// extractor, use `ingest_with()`.
    pub async fn ingest(&self, params: IngestParams<'_>) -> Result<IngestionResult> {
        let IngestParams {
            text,
            reference_time,
            group_id,
            content_type,
            source_params,
        } = params;
        #[cfg(feature = "ner")]
        {
            // MNT-001: use the process-wide singleton from ner::ner_singleton() so
            // both ingest_with and ingest_phase1_ner share a single loaded model
            // (~650 MB INT8). Previously this path had its own OnceLock, risking
            // double model load (~1.3 GB) when both paths were used in the same process.
            let extractor = crate::core::ner::ner_singleton()?;
            return self
                .ingest_with(
                    extractor,
                    crate::core::ingest::IngestWithParams {
                        text,
                        reference_time,
                        group_id,
                        content_type,
                        source_params,
                    },
                )
                .await;
        }
        #[cfg(not(feature = "ner"))]
        {
            let extractor = Arc::clone(&self.extractor);
            self.ingest_with(
                &extractor,
                crate::core::ingest::IngestWithParams {
                    text,
                    reference_time,
                    group_id,
                    content_type,
                    source_params,
                },
            )
            .await
        }
    }
}

// ── ADR-050 Phase 4: budget-tracking helpers ──────────────────────────────────
//
// These are free functions (not methods) so they carry no generic parameter
// and can be tested without an Engine instance.

/// Detect the billing-provider name from a model string.
///
/// Returns a lowercase short name suitable for the `provider` column in
/// `dream_pass_budget_usage`. Rules mirror `capability_of()` patterns but
/// collapse to billing-entity names instead of capability tiers.
///
/// | Model prefix         | Returns        |
/// |----------------------|----------------|
/// | `claude-*`           | `"anthropic"`  |
/// | Ollama colon pattern | `"ollama"`     |
/// | `gpt-*` / `o1-*` …  | `"openai"`     |
/// | Everything else      | `"unknown"`    |
pub(crate) fn detect_provider_name(model: &str) -> String {
    // Bedrock ARN prefixes → unknown (not directly billed via Anthropic API).
    if model.starts_with("anthropic.claude-")
        || model.starts_with("amazon.")
        || model.starts_with("meta.")
        || model.starts_with("mistral.")
        || model.starts_with("us.")
        || model.starts_with("eu.")
        || model.starts_with("ap.")
    {
        return "unknown".to_string();
    }
    // Anthropic direct-API Claude models.
    if model.starts_with("claude-") {
        return "anthropic".to_string();
    }
    // OpenAI models (gpt-*, o1-*, o3-*, o4-*).
    if model.starts_with("gpt-")
        || model.starts_with("o1-")
        || model.starts_with("o3-")
        || model.starts_with("o4-")
    {
        return "openai".to_string();
    }
    // Ollama: colon-separated name:tag, or well-known bare family names.
    // capability_of() uses the same heuristic for FormatSchema detection.
    if capability_of(model) == ProviderCaps::FormatSchema {
        return "ollama".to_string();
    }
    "unknown".to_string()
}

/// Compute cost in micro-USD for a dream pass given token counts.
///
/// Returns `Some(micro_usd)` when the rate is known, `None` when the model
/// is an unknown tier (caller writes NULL to the DB column).
///
/// # Rate table (verified 2026-06-16)
///
/// | Family              | Input ($/M) | Output ($/M) |
/// |---------------------|-------------|--------------|
/// | claude-haiku-*      | 0.80        | 4.00         |
/// | claude-sonnet-*     | 3.00        | 15.00        |
/// | claude-opus-*       | 15.00       | 75.00        |
/// | ollama (local)      | 0           | 0            |
/// | other / unknown     | `None`      | `None`       |
///
/// micro_usd = (tokens / 1_000_000) × ($/M) × 1_000_000
///           = tokens × ($/M)   (dollars cancel; result is micro-USD per token)
/// Simplified: micro_usd = tokens_input × input_rate_per_token
///                        + tokens_output × output_rate_per_token
/// where rate_per_token = $/M (numerically equal to micro-USD per token).
/// Bundled parameters for [`compute_dream_cost_micro`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
pub(crate) struct ComputeDreamCostMicroParams<'a> {
    pub provider: &'a str,
    pub model: &'a str,
    pub tokens_input: u64,
    pub tokens_output: u64,
}

pub(crate) fn compute_dream_cost_micro(params: ComputeDreamCostMicroParams<'_>) -> Option<u64> {
    let ComputeDreamCostMicroParams {
        provider,
        model,
        tokens_input,
        tokens_output,
    } = params;
    match provider {
        "ollama" => Some(0),
        "anthropic" => {
            // Per-token cost in micro-USD = $/M (same number: $1/M = $0.000001/token
            // = 1 micro-USD/token).
            let (input_rate, output_rate) = anthropic_rate_per_token(model)?;
            let cost = tokens_input
                .saturating_mul(input_rate)
                .saturating_add(tokens_output.saturating_mul(output_rate));
            Some(cost)
        }
        _ => None,
    }
}

/// Return (input_rate, output_rate) in micro-USD per token for known Anthropic
/// model families. Returns `None` for unknown / future model strings so cost is
/// stored as NULL rather than silently wrong.
///
/// Rates: https://www.anthropic.com/pricing (spot-checked 2026-06-16).
/// micro-USD per token = $/M numerically (1 $/M = 1 μ$/token).
fn anthropic_rate_per_token(model: &str) -> Option<(u64, u64)> {
    if model.starts_with("claude-haiku-") {
        // Haiku 3.5 / 4.x: $0.80/M in, $4.00/M out
        Some((1, 4))
    } else if model.starts_with("claude-sonnet-") {
        // Sonnet 3.5 / 4.x: $3.00/M in, $15.00/M out
        Some((3, 15))
    } else if model.starts_with("claude-opus-") {
        // Opus 4.x: $15.00/M in, $75.00/M out
        Some((15, 75))
    } else {
        // Unknown claude-* (e.g. claude-mythos-preview) → NULL
        None
    }
}

#[cfg(test)]
mod budget_helpers_tests {
    use super::*;

    #[test]
    fn detect_provider_ollama_colon() {
        assert_eq!(detect_provider_name("qwen2.5:14b"), "ollama");
        assert_eq!(detect_provider_name("gemma4-e2b:latest"), "ollama");
        assert_eq!(detect_provider_name("llama3.2:3b-instruct"), "ollama");
    }

    #[test]
    fn detect_provider_anthropic() {
        assert_eq!(
            detect_provider_name("claude-haiku-4-5-20251001"),
            "anthropic"
        );
        assert_eq!(detect_provider_name("claude-sonnet-4-6"), "anthropic");
        assert_eq!(detect_provider_name("claude-opus-4-7"), "anthropic");
        assert_eq!(detect_provider_name("claude-mythos-preview"), "anthropic");
    }

    #[test]
    fn detect_provider_openai() {
        assert_eq!(detect_provider_name("gpt-4.1"), "openai");
        assert_eq!(detect_provider_name("o1-preview"), "openai");
        assert_eq!(detect_provider_name("o3-mini"), "openai");
    }

    #[test]
    fn detect_provider_bedrock_unknown() {
        assert_eq!(
            detect_provider_name("anthropic.claude-3-opus-bedrock"),
            "unknown"
        );
        assert_eq!(
            detect_provider_name("us.anthropic.claude-opus-4-7-v1:0"),
            "unknown"
        );
    }

    #[test]
    fn detect_provider_unknown_fallback() {
        assert_eq!(detect_provider_name("random-model"), "unknown");
        assert_eq!(detect_provider_name(""), "unknown");
    }

    #[test]
    fn compute_cost_ollama_zero() {
        assert_eq!(
            compute_dream_cost_micro(ComputeDreamCostMicroParams {
                provider: "ollama",
                model: "qwen2.5:14b",
                tokens_input: 1000,
                tokens_output: 500,
            }),
            Some(0)
        );
    }

    #[test]
    fn compute_cost_anthropic_haiku() {
        // haiku: input rate = 1 micro_usd/token, output rate = 4 micro_usd/token
        // 1000 input → 1000; 500 output → 2000; total 3000
        assert_eq!(
            compute_dream_cost_micro(ComputeDreamCostMicroParams {
                provider: "anthropic",
                model: "claude-haiku-4-5-20251001",
                tokens_input: 1000,
                tokens_output: 500,
            }),
            Some(1000 + 500 * 4)
        );
    }

    #[test]
    fn compute_cost_anthropic_sonnet() {
        // sonnet: 100 input × 3 + 50 output × 15 = 300 + 750 = 1050
        assert_eq!(
            compute_dream_cost_micro(ComputeDreamCostMicroParams {
                provider: "anthropic",
                model: "claude-sonnet-4-6",
                tokens_input: 100,
                tokens_output: 50,
            }),
            Some(100 * 3 + 50 * 15)
        );
    }

    #[test]
    fn compute_cost_anthropic_unknown_model_none() {
        // Unknown claude model → None (stored as NULL)
        assert_eq!(
            compute_dream_cost_micro(ComputeDreamCostMicroParams {
                provider: "anthropic",
                model: "claude-mythos-preview",
                tokens_input: 100,
                tokens_output: 50,
            }),
            None
        );
    }

    #[test]
    fn compute_cost_unknown_provider_none() {
        assert_eq!(
            compute_dream_cost_micro(ComputeDreamCostMicroParams {
                provider: "unknown",
                model: "mystery-model",
                tokens_input: 100,
                tokens_output: 50,
            }),
            None
        );
        assert_eq!(
            compute_dream_cost_micro(ComputeDreamCostMicroParams {
                provider: "openai",
                model: "gpt-4.1",
                tokens_input: 100,
                tokens_output: 50,
            }),
            None
        );
    }
}
