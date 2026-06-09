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

use crate::core::provider::{ChatProvider, EmbeddingProvider, TokenUsage};
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

/// Optional source provenance fields forwarded to the `episodes` table
/// (Migration 007 columns: source_id, source_uri, recorded_at).
///
/// All fields are `None` / empty by default — callers that don't have source
/// provenance pass `SourceParams::default()` and the columns stay NULL.
#[derive(Debug, Default, Clone)]
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
    pub fn new(
        graph: Arc<TemporalGraph>,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
    ) -> Self {
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
    pub(crate) fn with_extractor(
        graph: Arc<TemporalGraph>,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
        extractor: Arc<crate::core::extraction::factory::ExtractorKind<L>>,
    ) -> Self {
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
        graph: Arc<TemporalGraph>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
        extractor: Arc<crate::core::extraction::factory::ExtractorKind<L>>,
    ) -> Self {
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
        source: &str,
        title: &str,
        text: &str,
    ) -> Result<IngestionResult> {
        // 1. Store the document as a searchable entity.
        // Documents use entity_type_id=0 (catch-all); the title is stored in properties.
        let properties = serde_json::json!({ "text": text, "title": title });
        self.graph.insert_entity(source, 0, properties).await?;
        metrics::counter!("rql.ingest.entity_persisted_total", "source" => "source_anchor")
            .increment(1);

        // 2. Embed the full document text for vector search.
        let embedding = self.embedder.embed(text).await?;
        self.graph.set_entity_embedding(source, &embedding).await?;

        // 3. Run the intelligence pipeline on the document content.
        let doc_text = format!("# {title}\n\n{text}");
        self.ingest(
            &doc_text,
            None,
            None,
            Some(ContentType::Document),
            SourceParams::default(),
        )
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
        let entities_reclassified = if let Some(llm_arc) = self.llm.as_ref() {
            let llm_ref: &L = llm_arc.as_ref();
            match crate::core::dream::reclassify::reclassify_all_groups(
                &self.graph,
                llm_ref,
                crate::core::dream::reclassify::ReclassifyOpts {
                    confidence_threshold: opts.confidence_threshold,
                    high_conf_threshold: opts.reclassify_high_conf_threshold,
                    max_batch_size: crate::core::dream::reclassify::MAX_RECLASSIFY_BATCH,
                },
            )
            .await
            {
                Ok(r) => r.entities_reclassified,
                Err(e) => {
                    // Non-fatal per E7 — surface as debug log, continue.
                    tracing::warn!(
                        error = %e,
                        "kremory.dream.reclassify_all_groups failed — skipping; dream pass unaffected"
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
    pub async fn assert_entity_type(
        &self,
        entity_id: &str,
        _entity_type_id: u32,
        group_id: Option<&str>,
    ) -> Result<()> {
        self.graph
            .update_entity_source_tier(entity_id, group_id, "ConsumerPinned")
            .await?;

        tracing::debug!(
            entity_id,
            group_id = group_id.unwrap_or("default"),
            "kremory.dream.assert_entity_type ConsumerPinned stamped"
        );

        Ok(())
    }

    /// Full pipeline: text → chunk → extract → resolve → contradict → store.
    /// Uses `NuExtractExtractor` (unified extraction template). For alternative extractors,
    /// use `ingest_with()`.
    // 7 args (threshold 5): text + reference_time + group_id + content_type + source_params
    // + sink + episode_content_warn_threshold are all orthogonal call-context parameters
    // that cannot be merged into a single typed struct without an opaque builder layer.
    #[allow(clippy::too_many_arguments)]
    pub async fn ingest(
        &self,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<&str>,
        content_type: Option<ContentType>,
        source_params: SourceParams,
    ) -> Result<IngestionResult> {
        #[cfg(feature = "ner")]
        {
            use std::sync::OnceLock;
            static GLINER: OnceLock<crate::core::ner::GlinerExtractor> = OnceLock::new();
            if GLINER.get().is_none() {
                let g = crate::core::ner::GlinerExtractor::new()
                    .map_err(crate::core::error::Error::from)?;
                let _ = GLINER.set(g);
            }
            let extractor = GLINER.get().unwrap_or_else(|| {
                panic!("invariant: GLINER OnceLock empty immediately after set")
            });
            return self
                .ingest_with(
                    extractor,
                    text,
                    reference_time,
                    group_id,
                    content_type,
                    source_params,
                )
                .await;
        }
        #[cfg(not(feature = "ner"))]
        {
            let extractor = Arc::clone(&self.extractor);
            self.ingest_with(
                &extractor,
                text,
                reference_time,
                group_id,
                content_type,
                source_params,
            )
            .await
        }
    }
}
