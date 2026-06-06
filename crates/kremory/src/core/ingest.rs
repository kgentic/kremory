use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use metrics::histogram;

use chrono::{DateTime, Utc};

use crate::core::config::{ContentType, PipelineConfig};
use crate::core::contradiction::TwoPoolDetector;
use crate::core::entity_types::EntityTypeRegistry;
use crate::core::error::Result;
use crate::core::extraction::normalize_label;
use crate::core::extraction_window::ExtractionWindowSplitter;
use crate::core::intelligence::{
    EntityExtractor, EntityResolver, ExtractedEntity, ExtractedFact, ExtractionContext,
    ResolutionResult,
};
use crate::core::provider::{ChatProvider, EmbeddingProvider, TokenUsage};
#[cfg(any(test, feature = "test-utils"))]
use crate::core::provider::{MockChatProvider, NullEmbeddingProvider};
use crate::core::resolver::{normalize_name, CascadeResolver, UnionFind};
use crate::core::schema::TemporalGraph;
use crate::core::search::SearchFilters;
use crate::core::text_utils;

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

/// High-level kremory graph engine with intelligence pipeline.
/// Wraps TemporalGraph and adds extraction, resolution, and contradiction detection.
pub struct Engine<L: ChatProvider, Emb: EmbeddingProvider> {
    pub(crate) graph: Arc<TemporalGraph>,
    pub(crate) llm: Arc<L>,
    pub(crate) embedder: Arc<Emb>,
    pub(crate) config: PipelineConfig,
    /// Optional OOV auditor for language-agnostic entity safety net.
    /// When set, runs after each chunk extraction to catch domain terms the LLM missed.
    pub(crate) oov_auditor: Option<text_utils::OovAuditor>,
    /// Model identifier read from `llm.model()` at construction.
    /// None when llm.model() returns empty string.
    pub(crate) model: Option<String>,
    /// Production entity extractor — constructed once per Engine, shared via
    /// Arc across all 8 ingest hot-path callsites. See spec
    /// `kremory-v017-hybrid-extractor-production-wire-in-spec-2026-06-04`
    /// and ADR-029. Defaults to NuExtract; hybrid opt-in via
    /// `ExtractorSource` builder method or `KREMORY_EXTRACTOR=hybrid` env.
    pub(crate) extractor: Arc<crate::core::extraction::ProductionExtractor<L>>,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Create a new `Engine` wrapping a `TemporalGraph`.
    ///
    /// Accepts `Arc<TemporalGraph>` so callers can share the same graph
    /// instance with the process-global singleton returned by `engine()`.
    ///
    /// The entity extractor is resolved via `KREMORY_EXTRACTOR` env var
    /// (default: NuExtract). For explicit control over extractor choice,
    /// use [`Engine::with_extractor_source`].
    ///
    /// Returns `Err` only if `KREMORY_EXTRACTOR=hybrid` is set in the
    /// environment AND the hybrid extractor fails to load GLiNER weights
    /// (e.g. no network on first run + no hf-hub cache).
    pub fn new(
        graph: Arc<TemporalGraph>,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
    ) -> Result<Self> {
        Self::with_extractor_source(
            graph,
            llm,
            embedder,
            config,
            crate::core::extraction::ExtractorSource::FromEnv,
        )
    }

    /// Construct an Engine with an explicit extractor source. Use this when
    /// you want to bypass env-var resolution — e.g. pin NuExtract in test
    /// suites that don't need hybrid, or pin Hybrid in production code that
    /// requires it regardless of env.
    pub fn with_extractor_source(
        graph: Arc<TemporalGraph>,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
        extractor_source: crate::core::extraction::ExtractorSource,
    ) -> Result<Self> {
        let m = llm.model().trim().to_string();
        let model = if m.is_empty() { None } else { Some(m) };
        let extractor = Arc::new(
            crate::core::extraction::ProductionExtractor::from_source(
                extractor_source,
                Arc::clone(&llm),
            )?,
        );
        Ok(Self {
            graph,
            llm,
            embedder,
            config,
            oov_auditor: None,
            model,
            extractor,
        })
    }

    /// Construct an Engine pinned to NuExtract via the infallible factory
    /// constructor. Returns `Self` directly (no `Result`) because NuExtract
    /// construction has no I/O and no model load.
    ///
    /// Test-convenience constructor — production code should normally use
    /// [`Self::new`] (env-driven) or [`Self::with_extractor_source`] (when
    /// an explicit pin to Hybrid is required and the caller is prepared to
    /// handle a GLiNER weight-load failure).
    pub fn with_nuextract(
        graph: Arc<TemporalGraph>,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
    ) -> Self {
        let m = llm.model().trim().to_string();
        let model = if m.is_empty() { None } else { Some(m) };
        let extractor = Arc::new(
            crate::core::extraction::ProductionExtractor::nuextract_only(Arc::clone(&llm)),
        );
        Self {
            graph,
            llm,
            embedder,
            config,
            oov_auditor: None,
            model,
            extractor,
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

    /// Full pipeline with a caller-supplied extractor.
    /// Any type implementing `EntityExtractor` can be used (DefaultExtractor, NuExtractExtractor, etc.).
    // Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        name = "kremory.ingest",
        skip(self, extractor, text),
        fields(
            kremory.operation = "ingest",
        )
    )]
    pub async fn ingest_with<E: EntityExtractor>(
        &self,
        extractor: &E,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<&str>,
        content_type: Option<ContentType>,
        source_params: SourceParams,
    ) -> Result<IngestionResult> {
        let ingest_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);
        let token_usage = TokenUsage::default();

        // 1. Store episode (namespace-scoped via group_id).
        //    source_id / source_uri / recorded_at from SourceParams are written to the
        //    Migration 007 columns so that recall_by_source_id can find this episode.
        let episode_id = self
            .graph
            .insert_episode_with_group(
                text,
                ref_time,
                Some("ingest"),
                None,
                group_id,
                None,
                None,
                source_params.source_id.as_deref(),
                source_params.source_uri.as_deref(),
                source_params.recorded_at,
            )
            .await?;

        // 1b. ADR-035 §5 Option A — Pin caller-pre-extracted facts BEFORE Phase 2 LLM.
        //
        // Caller-supplied triples enter the graph first; the LLM Phase 2 extraction
        // path uses `try_insert_fact` which silently swallows the resulting
        // `Err(Duplicate)` on the same `content_hash`, so caller wins via
        // pre-write ordering. Intra-set duplicates (caller passes the same
        // triple twice in their own set) also dedup cleanly via
        // `try_insert_fact_with_group`.
        let mut pinned_fact_ids: Vec<i64> = Vec::new();
        if !source_params.pre_pinned_facts.is_empty() {
            let mut pinned_count: u64 = 0;
            for pf in &source_params.pre_pinned_facts {
                // Auto-stub the subject entity (entity_type_id=0 "Entity") so the
                // fact insert FK constraint resolves. Mirrors how Phase 2 LLM
                // extraction auto-creates UNKNOWN entities for forward references.
                // Duplicate-entity is the expected case (entity already exists);
                // explicit swallow. Per [[treat-cause-not-symptom]] + Quinn M-02,
                // NON-Duplicate errors (connection failure, schema gap, etc) are
                // surfaced via tracing::warn — masking them silently would hide
                // real production failures.
                if let Err(e) = self
                    .graph
                    .insert_entity(&pf.subject, 0, serde_json::json!({"stub": false}))
                    .await
                {
                    if !matches!(e, crate::core::error::Error::Duplicate { .. }) {
                        tracing::warn!(
                            subject = %pf.subject,
                            error = %e,
                            "kremory.with_facts.stub_entity_insert_failed"
                        );
                    }
                }
                if let Some(ref obj_id) = pf.object_id {
                    if let Err(e) = self
                        .graph
                        .insert_entity(obj_id, 0, serde_json::json!({"stub": false}))
                        .await
                    {
                        if !matches!(e, crate::core::error::Error::Duplicate { .. }) {
                            tracing::warn!(
                                object_id = %obj_id,
                                error = %e,
                                "kremory.with_facts.stub_entity_insert_failed"
                            );
                        }
                    }
                }

                match self
                    .graph
                    .try_insert_fact_with_group(
                        &pf.subject,
                        &pf.predicate,
                        pf.object_id.as_deref(),
                        pf.object_value.as_deref(),
                        pf.valid_from,
                        pf.confidence,
                        Some(episode_id),
                        group_id,
                        None,
                    )
                    .await
                {
                    Ok(Some(fact_id)) => {
                        pinned_count = pinned_count.saturating_add(1);
                        pinned_fact_ids.push(fact_id);
                    }
                    Ok(None) => {
                        // Intra-caller-set duplicate (already counted as
                        // axis=caller_vs_llm inside the helper; relabel here
                        // for diagnostic clarity via a second counter increment).
                        metrics::counter!(
                            "kremory.with_facts.deduped_total",
                            "axis" => "intra_caller_set"
                        )
                        .increment(1);
                    }
                    Err(e) => {
                        // DIAG: surface failures via counter so tests can detect.
                        metrics::counter!("kremory.with_facts.pin_failed_total").increment(1);
                        tracing::warn!(
                            subject = %pf.subject,
                            predicate = %pf.predicate,
                            error = %e,
                            "kremory.with_facts.pin_failed"
                        );
                    }
                }
            }
            if pinned_count > 0 {
                metrics::counter!(
                    "kremory.with_facts.pinned_total",
                    "source" => "caller"
                )
                .increment(pinned_count);
                tracing::info!(
                    pinned = pinned_count,
                    requested = source_params.pre_pinned_facts.len(),
                    episode_id,
                    "kremory.with_facts.pinned"
                );
            } else {
                // Per Quinn L-01: avoid INFO-level noise for all-dedup caller sets;
                // bulk-import workloads can hit this thousands of times per batch.
                tracing::debug!(
                    pinned = 0_u64,
                    requested = source_params.pre_pinned_facts.len(),
                    episode_id,
                    "kremory.with_facts.all_deduped_or_failed"
                );
            }
        }

        // 1c. ADR-035 §2/§6 — `skip_extraction` early return.
        //
        // When the caller has set `RememberRequest::skip_extraction()` (or the
        // legacy `SubmitOpts.enrich_per_episode = false` once the engine_handle
        // gate is wired), bail after caller-pin completion. Episode + embedding
        // (if pre-pinned) + caller-pre-extracted facts are persisted; Phase 2
        // LLM extraction is skipped entirely. Useful for bulk-import workloads
        // where caller already has high-confidence data.
        if source_params.skip_extraction {
            metrics::counter!("kremory.skip_extraction.invoked_total").increment(1);
            tracing::info!(
                episode_id,
                pinned_fact_count = pinned_fact_ids.len(),
                "kremory.ingest.skip_extraction_early_return"
            );
            histogram!("rql.ingest.skip_extraction_ms")
                .record(ingest_start.elapsed().as_secs_f64() * 1000.0);
            return Ok(IngestionResult {
                episode_id,
                upserted_entities: Vec::new(),
                inserted_fact_ids: pinned_fact_ids,
                invalidated_fact_ids: Vec::new(),
                merged_entities: Vec::new(),
                token_usage,
                stub_entities_inserted: 0,
            });
        }

        // 2. Slice into LLM-extraction-prompt windows (no-op for normally-sized episodes;
        //    see core/extraction_window.rs module docstring for kind-2 semantics).
        let splitter = ExtractionWindowSplitter::new(self.config.extraction_window.clone());
        let chunks = splitter.split(text, &content_type);

        let chunk_count = chunks.len();
        histogram!("rql.ingest.chunk_count").record(chunk_count as f64);
        tracing::info!(chunk_count, "kremory.ingest.chunked");

        // 2b. L2: resolve EntityTypeRegistry for this group BEFORE extraction so
        //     registry specs can be injected into extraction prompts (Phase 3).
        //     This is a cheap DB read (a handful of rows); loading early does not
        //     duplicate the work at step 4 — step 4 re-uses the same `registry`
        //     binding for L3 validation.
        //
        //     TD-013 §1 hybrid: if caller supplied a per-call override, use it.
        //     - First-call persistence: if DB is empty for this group_id, persist
        //       the override so subsequent calls without override see the vocabulary.
        //     - Ephemeral: if DB already has rows, use override for this call only.
        let effective_gid = group_id.unwrap_or("default");

        // 2a. TD-013 Migration 010 lazy-seed: ensure the default vocabulary is
        //     present for `effective_gid` before any registry-dependent work
        //     (L2 prompt, L3 validation). Idempotent: no-ops once seeded.
        //     This catches namespaces created AFTER Migration 010 ran at boot.
        crate::core::entity_types::ensure_default_types_seeded(
            &self.graph.conn,
            effective_gid,
        )
        .await?;

        let registry = if let Some(ref override_specs) = source_params.entity_types_override {
            let db_registry =
                EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?;
            if db_registry.is_empty() {
                // First-call persistence: seed DB from override.
                crate::core::entity_types::upsert_entity_types(
                    &self.graph.conn,
                    effective_gid,
                    override_specs,
                )
                .await?;
                metrics::counter!(
                    "rql.ingest.registry_override_applied",
                    "persisted" => "true",
                )
                .increment(1);
            } else {
                // Additive merge: persist any override types missing from DB.
                //
                // Was previously "ephemeral, do not touch DB" (TD-013) but that
                // created an entity-type/JOIN hole: an entity row stored with
                // entity_type_id = override-only id had no entity_types row →
                // SQL COALESCE(et.name, 'Entity') resolved label='Entity' at
                // read time regardless of the stored integer id. Per
                // [[audit-what-guards-mask-before-deleting]] the right fix is
                // additive merge — INSERT OR IGNORE each missing override
                // spec, preserving previously-stored rows.
                let mut newly_persisted: usize = 0;
                for spec in override_specs.iter() {
                    if db_registry.name_to_id(&spec.name).is_none() {
                        self.graph
                            .conn
                            .execute(
                                "INSERT OR IGNORE INTO entity_types \
                                 (group_id, id, name, description) \
                                 VALUES (?1, ?2, ?3, ?4)",
                                libsql::params![
                                    effective_gid,
                                    spec.id as i64,
                                    spec.name.clone(),
                                    spec.description.clone()
                                ],
                            )
                            .await
                            .map_err(|e| crate::core::error::Error::Other(anyhow::anyhow!(
                                "additive override persist failed for spec '{}': {e}",
                                spec.name
                            )))?;
                        newly_persisted += 1;
                    }
                }
                metrics::counter!(
                    "rql.ingest.registry_override_applied",
                    "persisted" => if newly_persisted > 0 { "additive" } else { "false" },
                )
                .increment(1);
                metrics::histogram!("rql.ingest.registry_override_additive_count")
                    .record(newly_persisted as f64);
            }
            EntityTypeRegistry::from_specs(override_specs.clone())
        } else {
            EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?
        };

        // 2c. L4': fetch top-N existing entities for prompt-time injection.
        //
        //     The extraction prompt lists the canonical entities already in the
        //     knowledge graph so the LLM reuses exact names rather than inventing
        //     variant spellings ("Alice Johnson" vs "Alice J.").  We fetch the
        //     top-50 by access_count (most-recently-used) as context.
        //
        //     This is a separate, read-only fetch from step 4 (which fetches all
        //     entities for resolver dedup).  The two fetches serve different
        //     purposes and are not merged — step 4 deduplication needs all
        //     entities; L4' injection needs only the top-N most relevant.
        let existing_entities_for_prompt: Vec<(String, String)> = {
            const L4_PRIME_INJECT_LIMIT: usize = 50;
            let mut raw = match group_id {
                Some(gid) => self.graph.list_entities_in_group(gid).await?,
                None => self.graph.list_entities().await?,
            };
            // Sort descending by access_count (most-recently-used first).
            raw.sort_by(|a, b| b.access_count.cmp(&a.access_count));
            raw.truncate(L4_PRIME_INJECT_LIMIT);
            raw.into_iter()
                .map(|e| {
                    // Prefer properties["name"] (original casing) over the
                    // normalized id so the LLM sees the exact display name.
                    let display_name = e
                        .properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_owned())
                        .unwrap_or(e.id);
                    (display_name, e.label)
                })
                .collect()
        };

        // 3. Extract from all chunks, merge results.
        // known_entities grows with each iteration so subsequent chunks receive
        // the entities already found in earlier chunks as context.
        let mut all_entities: Vec<ExtractedEntity> = Vec::new();
        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &self.config.allowed_entity_types,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &all_entities,
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: content_type.clone(),
                registry_specs: registry.specs(),
                existing_graph_entities: &existing_entities_for_prompt,
                arm_budget_ms: self.config.extraction_arm_budget_ms,
            };
            let result = extractor.extract(chunk, &ctx).await?;
            all_entities.extend(result.entities);
            all_facts.extend(result.facts);

            // OOV audit: catch domain terms the LLM missed (language-agnostic safety net)
            if let Some(ref auditor) = self.oov_auditor {
                let audit_adds = auditor.audit(chunk, &all_entities);
                all_entities.extend(audit_adds);
            }
        }

        // TD-013 Phase 8 v3 (2026-06-04): scan_proper_nouns DISABLED pending
        // task #15 — scanner hardcodes label="Entity" instead of routing
        // candidates through LLM classification (the original two-call design
        // per spike_phase3_scanner_vs_pipeline.rs, never finished). Disabling
        // gives us pure-LLM extraction matching Graphiti/Cognee/LightRAG.
        // Re-enable after wiring the second LLM call OR delete entirely.
        //
        // let proper_noun_candidates = text_utils::scan_proper_nouns(text, &all_entities);
        // all_entities.extend(proper_noun_candidates);

        // 3b. Pre-mutation intra-batch duplicate scan (Story #150 history).
        //
        // Original behaviour: FATAL `Err(IntraBatchDuplicate)` on duplicate names.
        // The intent was to surface malformed batches submitted by human callers
        // rather than silently dropping rows.
        //
        // Why we softened it in v0.1.4: kremory's pipeline ingests LLM-extracted
        // entities, and noisy extractors (smaller local models — llama3.2:3b,
        // gemma4-e2b — observed 2026-05-28 emitting `'car'` / `'VerbatimString'`
        // multiple times in a single batch) cannot be expected to dedupe their
        // own output. Treating that as fatal forced operators to pick a
        // hardened-extractor model (qwen2.5:14b+) rather than letting the
        // substrate accept noisy upstream input gracefully.
        //
        // New behaviour: emit a `tracing::warn!` with the duplicated names and
        // continue — the dedup below removes the offenders, exactly one row
        // per normalized name lands in the DB. Human callers who want strict
        // dedup-rejection can layer that contract on top of `remember_batch`
        // at the application layer.
        {
            let mut seen_this_call: HashSet<String> = HashSet::new();
            let mut dup_names: Vec<String> = Vec::new();
            for extracted in &all_entities {
                let id = normalize_name(&extracted.name);
                if !seen_this_call.insert(id.clone()) {
                    dup_names.push(id);
                }
            }
            if !dup_names.is_empty() {
                tracing::warn!(
                    target: "kremory.ingest",
                    dup_count = dup_names.len(),
                    dup_names = ?dup_names,
                    "extractor emitted duplicate entity names — deduplicating silently \
                     (post-v0.1.4 behaviour; previously FATAL IntraBatchDuplicate)"
                );
            }
        }

        // Deduplicate extracted entities by normalized name.
        // sort + dedup_by because dedup_by only removes consecutive duplicates.
        all_entities.sort_by_key(|e| normalize_name(&e.name));
        all_entities.dedup_by(|a, b| normalize_name(&a.name) == normalize_name(&b.name));

        // 4. Resolve entities against existing graph (namespace-scoped dedup)
        let existing_entities = match group_id {
            Some(gid) => self.graph.list_entities_in_group(gid).await?,
            None => self.graph.list_entities().await?,
        };

        // registry was loaded above at step 2b — reused here for L3 validation
        // (label → integer entity_type_id at insert time).

        let resolver = CascadeResolver::new(
            Arc::clone(&self.llm),
            self.config.minhash.clone(),
            self.config.entropy.clone(),
        );

        // ── Bug E: open a single outer transaction wrapping Phase 1 (entities +
        // stubs) and Phase 2 (facts).  All inner graph methods call
        // begin_immediate_if_needed() which is a no-op when a transaction is
        // already open, so nesting is safe.
        let outer_guard = self.graph.begin_immediate_if_needed().await?;

        // All mutable state that accumulates across the two phases lives here.
        // These are declared before the match so they can be moved into the Ok
        // branch cleanly — no partial-move issues.
        let mut union_find = UnionFind::new();
        let mut upserted_entities: Vec<String> = Vec::new();
        let mut merged_entities: Vec<(String, String)> = Vec::new();
        let mut name_to_id: HashMap<String, String> = HashMap::new();
        let mut inserted_fact_ids: Vec<i64> = Vec::new();
        let mut invalidated_fact_ids: Vec<i64> = Vec::new();
        let mut stub_entities_inserted: usize = 0;

        // Helper closure-like block that returns Result<()> so we can commit or
        // rollback in one place.  We use a labelled block instead of an async
        // closure to avoid capture/lifetime complexity.
        let phase_result: Result<()> = 'phases: {
            // ── Bug E §1: pre-scan all_facts for forward-reference entity names ─────
            // Collect every entity name appearing as a subject or object in facts.
            // Any name NOT already mapped from the extraction list is a forward
            // reference — insert it as an UNKNOWN stub so the fact loop can resolve
            // it without producing a dangling subject_id.
            let extracted_names: HashSet<String> = all_entities
                .iter()
                .map(|e| normalize_name(&e.name))
                .collect();

            for fact in &all_facts {
                let mut forward_refs: Vec<String> = vec![normalize_name(&fact.subject)];
                if fact.is_entity_ref {
                    forward_refs.push(normalize_name(&fact.object));
                }
                for norm_name in forward_refs {
                    if extracted_names.contains(&norm_name) {
                        // Will be handled in the entity loop — skip.
                        continue;
                    }
                    if name_to_id.contains_key(&norm_name) {
                        // Already inserted as a stub in a previous fact iteration.
                        continue;
                    }
                    // RISK-003: entities.id is a sole TEXT PK pre-migration-004.
                    // Post-migration-004: composite PK (id, group_id) closes the bypass surface.
                    // Stub INSERT uses INSERT OR IGNORE — cross-namespace name collision silently
                    // skips stub creation. Single-namespace use only for v0.1.1.
                    //
                    // Strategy: attempt insert_entity_with_group; if the entity already exists
                    // (UNIQUE constraint error), that is fine — a real row is present.
                    let stub_props =
                        serde_json::json!({ "stub": true, "source": "forward_reference" });
                    match self
                        .graph
                        .insert_entity_with_group(&norm_name, 0, stub_props, group_id)
                        .await
                    {
                        Ok(()) => {
                            // Newly inserted stub.
                            name_to_id.insert(norm_name.clone(), norm_name.clone());
                            stub_entities_inserted += 1;
                            metrics::counter!("kremory.ingest.stub_inserted").increment(1);
                            metrics::counter!(
                                "rql.ingest.entity_persisted_total",
                                "source" => "stub",
                            )
                            .increment(1);
                            tracing::warn!(
                                target: "kremory.ingest.stub",
                                name = %norm_name,
                                "inserted UNKNOWN stub for forward reference"
                            );
                        }
                        Err(_) => {
                            // Entity already exists (real or from a previous batch) — use it.
                            name_to_id.insert(norm_name.clone(), norm_name.clone());
                        }
                    }
                }
            }

            // ── Phase 1: entity loop (Bug B snippet + Bug A episodic_edge) ──────────
            for extracted in &all_entities {
                // TD-013 Phase 8 finalizes Vera M1: the TD-012 over-rejection guard
                // (`is_canonical_entity_type` + `allowed_entity_types` policy) is
                // DELETED under the L1 integer-ID design. Live diagnostic
                // 2026-06-03 showed it rejecting 14 "Entity" emissions per
                // mock_interview ingest — id=0 "Entity" is the legitimate
                // Graphiti-pattern catch-all, NOT a placeholder to drop.
                //
                // Replacement guardrails (all already wired):
                //   - L3 validate_or_fallback bounds-checks emitted entity_type_id
                //   - L4 disambiguation handles surface-form duplicates
                //   - L5 vector canonicalization merges variant spellings
                //   - L7 dream-phase reclassification upgrades id=0 entities once
                //     corpus accumulates 3+ episodes
                //
                // We still normalise the label string so downstream resolver +
                // FTS get the canonical case ("ORG" → "Organisation").
                let label = normalize_label(&extracted.label);

                let mut resolved_to: Option<String> = None;

                for existing in &existing_entities {
                    let result = match resolver.resolve(extracted, existing).await {
                        Ok(r) => r,
                        Err(e) => break 'phases Err(e),
                    };
                    if result == ResolutionResult::Same {
                        resolved_to = Some(existing.id.clone());
                        break;
                    }
                }

                let entity_id = if let Some(existing_id) = resolved_to {
                    // Merged with existing entity
                    union_find.make_set(&existing_id);
                    let norm = normalize_name(&extracted.name);
                    union_find.make_set(&norm);
                    union_find.union(&norm, &existing_id);
                    merged_entities.push((existing_id.clone(), extracted.name.clone()));

                    // Bug E (F-4): stub promotion — if the existing entity is a stub
                    // (entity_type_id=0, properties.stub=true), overwrite it with the
                    // real entity_type_id and a fresh context snippet. The upsert removes
                    // the stub flag because the new properties map does not carry it.
                    // First-mention-wins policy still applies for non-stubs.
                    let existing_is_stub = existing_entities
                        .iter()
                        .find(|e| e.id == existing_id)
                        .map(|e| {
                            e.entity_type_id == 0
                                && e.properties
                                    .get("stub")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false)
                        })
                        .unwrap_or(false);

                    if existing_is_stub {
                        let context_snippet = extract_context_snippet(text, &extracted.name, 200);
                        let promoted_props = serde_json::json!({
                            "context": context_snippet,
                            "name": extracted.name.clone()
                        });
                        // TD-021 open vocabulary: register novel type label on the fly.
                        // If registration itself fails (DB error), fall back to id=0
                        // and emit a swallow counter (Vera Finding 1) — the stub
                        // promotion is best-effort and must not abort the transaction.
                        let entity_type_id = match crate::core::entity_types::label_to_id_or_register(
                            &self.graph.conn,
                            group_id.unwrap_or("default"),
                            &registry,
                            &label,
                        )
                        .await
                        {
                            Ok(id) => id,
                            Err(_) => {
                                metrics::counter!(
                                    "rql.entity_types.registration_swallowed_total",
                                    "site" => "stub_promotion",
                                )
                                .increment(1);
                                0
                            }
                        };
                        // `.ok()` — promotion is best-effort; failure to promote
                        // leaves the stub row but does not abort the transaction.
                        self.graph
                            .upsert_entity_with_group(
                                &existing_id,
                                entity_type_id,
                                promoted_props,
                                group_id,
                            )
                            .await
                            .ok();
                        // Stub-promotion path: a row that was previously source=stub
                        // is now being filled with LLM-extracted content. Count it as
                        // a NET-NEW llm-source write (the stub row is no longer a
                        // stub after this upsert).
                        metrics::counter!(
                            "rql.ingest.entity_persisted_total",
                            "source" => "llm",
                            "via" => "stub_promotion",
                        )
                        .increment(1);
                        tracing::debug!(
                            target: "kremory.ingest.stub",
                            id = %existing_id,
                            label = %label,
                            entity_type_id,
                            "promoted stub entity to real entity"
                        );
                    }

                    // Bug A: episodic edge for merged entity (this episode now references it).
                    self.graph
                        .insert_episodic_edge(episode_id, &existing_id, "mention")
                        .await
                        .ok();

                    existing_id
                } else {
                    // ── L4: graph-time disambiguation (Cognee pattern) ──────────────────
                    // Before inserting a new entity row, probe cosine similarity against
                    // existing entities in the same group_id.  This catches "Alice Johnson"
                    // vs "Alice J." style variants that the string-match resolver (step 4)
                    // misses because they are not lexically close enough.
                    //
                    // Outcomes:
                    //   Merge         → reuse the existing entity_id (same real-world entity)
                    //   PotentialAlias → insert new row AND a `potential_alias` fact edge
                    //   New           → proceed as normal (no existing match)
                    let l4_outcome = match crate::core::disambiguation::disambiguate(
                        &extracted.name,
                        group_id,
                        &self.graph,
                        &*self.embedder,
                    )
                    .await
                    {
                        Ok(o) => o,
                        Err(e) => break 'phases Err(e),
                    };

                    // ── L4 Merge path ───────────────────────────────────────────────────
                    if let crate::core::disambiguation::DisambiguationOutcome::Merge {
                        existing_id: ref l4_existing_id,
                        ..
                    } = l4_outcome
                    {
                        let norm = normalize_name(&extracted.name);
                        union_find.make_set(l4_existing_id);
                        union_find.make_set(&norm);
                        union_find.union(&norm, l4_existing_id);
                        merged_entities.push((l4_existing_id.clone(), extracted.name.clone()));
                        // Episodic edge: this episode now references the existing entity.
                        self.graph
                            .insert_episodic_edge(episode_id, l4_existing_id, "mention")
                            .await
                            .ok();
                        name_to_id.insert(norm, l4_existing_id.clone());
                        // Skip the rest of the else block — entity_id is the existing one.
                        // SAFETY: the outer `let entity_id = if ... { ... } else { ... };`
                        // expression needs a value; we push to name_to_id above and
                        // `continue` to the next extracted entity below.
                        continue;
                    }

                    // ── New or PotentialAlias — insert new entity row ────────────────────
                    let entity_id = normalize_name(&extracted.name);

                    // Bug B: capture verbatim first-mention snippet (±100 chars around the
                    // entity name in the source text).  First-mention wins: this branch only
                    // runs for genuinely new entity rows.
                    let snippet = extract_context_snippet(text, &extracted.name, 100);
                    let props_with_context = serde_json::json!({
                        "context": snippet,
                        "name": extracted.name.clone(),
                    });

                    // TD-021 open vocabulary: register novel type label on the fly.
                    // Unlike the stub-promotion site, the insert-new path is NOT
                    // best-effort — registration failure propagates as a hard
                    // ingest failure (matches the existing insert_entity_with_group
                    // error path that breaks 'phases below).
                    let entity_type_id = match crate::core::entity_types::label_to_id_or_register(
                        &self.graph.conn,
                        group_id.unwrap_or("default"),
                        &registry,
                        &label,
                    )
                    .await
                    {
                        Ok(id) => id,
                        Err(e) => break 'phases Err(e),
                    };
                    if let Err(e) = self
                        .graph
                        .insert_entity_with_group(
                            &entity_id,
                            entity_type_id,
                            props_with_context,
                            group_id,
                        )
                        .await
                    {
                        break 'phases Err(e);
                    }
                    metrics::counter!(
                        "rql.ingest.entity_persisted_total",
                        "source" => "llm",
                        "via" => "insert_new",
                    )
                    .increment(1);

                    // Embed and store embedding
                    let embedding = match self.embedder.embed(&extracted.name).await {
                        Ok(v) => v,
                        Err(e) => break 'phases Err(e),
                    };
                    if let Err(e) = self
                        .graph
                        .set_entity_embedding(&entity_id, &embedding)
                        .await
                    {
                        break 'phases Err(e);
                    }

                    // ── L4 PotentialAlias: record the meta-edge fact ─────────────────────
                    // Best-effort — alias recording failure does not abort ingest.
                    if let crate::core::disambiguation::DisambiguationOutcome::PotentialAlias {
                        existing_id: ref alias_target_id,
                        similarity: alias_sim,
                    } = l4_outcome
                    {
                        // `.ok()` — best-effort; duplicate alias entries are swallowed
                        // inside `insert_potential_alias_fact`.
                        crate::core::disambiguation::insert_potential_alias_fact(
                            &self.graph,
                            &entity_id,
                            alias_target_id,
                            alias_sim,
                            crate::core::disambiguation::AliasProvenance {
                                source_episode_id: Some(episode_id),
                                group_id,
                            },
                        )
                        .await
                        .ok();
                    }

                    // Bug A: episodic edge for newly inserted entity.
                    self.graph
                        .insert_episodic_edge(episode_id, &entity_id, "mention")
                        .await
                        .ok();

                    upserted_entities.push(entity_id.clone());
                    entity_id
                };

                name_to_id.insert(normalize_name(&extracted.name), entity_id);
            }

            // ── Phase 2: detect contradictions and store facts ───────────────────────
            let detector = TwoPoolDetector::new(Arc::clone(&self.llm));

            for fact in &all_facts {
                // Resolve subject and object IDs through the merge map
                let subject_id = name_to_id
                    .get(&normalize_name(&fact.subject))
                    .cloned()
                    .unwrap_or_else(|| normalize_name(&fact.subject));

                let object_id = if fact.is_entity_ref {
                    Some(
                        name_to_id
                            .get(&normalize_name(&fact.object))
                            .cloned()
                            .unwrap_or_else(|| normalize_name(&fact.object)),
                    )
                } else {
                    None
                };

                let object_value = if !fact.is_entity_ref {
                    Some(fact.object.as_str())
                } else {
                    None
                };

                // Get candidate pools for contradiction detection
                let pool_a = match self
                    .graph
                    .get_facts_by_subject_predicate(&subject_id, &fact.predicate)
                    .await
                {
                    Ok(p) => p,
                    Err(e) => break 'phases Err(e),
                };

                // For pool_b, use FTS search on the predicate to find semantically related facts
                let pool_b_hits = match self
                    .graph
                    .fts_search_facts(&fact.predicate, 10, &SearchFilters::new())
                    .await
                {
                    Ok(hits) => hits,
                    Err(e) => break 'phases Err(e),
                };
                let pool_b: Vec<crate::core::schema::Fact> =
                    pool_b_hits.into_iter().map(|h| h.item).collect();

                // Run contradiction detection
                let contradiction_result =
                    match detector.detect(fact, &pool_a, &pool_b, &ref_time).await {
                        Ok(r) => r,
                        Err(e) => break 'phases Err(e),
                    };

                // Invalidate contradicted facts
                for fact_id in &contradiction_result.contradictions {
                    if let Err(e) = self
                        .graph
                        .invalidate_fact_with_reason(*fact_id, Utc::now(), ref_time)
                        .await
                    {
                        break 'phases Err(e);
                    }
                    invalidated_fact_ids.push(*fact_id);
                }

                // Insert the new fact — skip gracefully if FK constraint fails
                // (e.g., fact references an entity not in the extraction results).
                //
                // ADR-035 §5 Option A: use try_insert_fact so caller-pinned facts
                // (via mem.remember(...).with_facts(...)) silently dedup at LLM
                // Phase 2 — the caller wins by virtue of being there first.
                match self
                    .graph
                    .try_insert_fact(
                        &subject_id,
                        &fact.predicate,
                        object_id.as_deref(),
                        object_value,
                        ref_time,
                        fact.confidence,
                        Some(episode_id),
                        None,
                    )
                    .await
                {
                    Ok(Some(fact_id)) => {
                        // Embed the fact triple as a single string (subject predicate object)
                        // and store it so vector_search_facts can find it semantically.
                        let fact_text =
                            format!("{} {} {}", fact.subject, fact.predicate, fact.object);
                        if let Ok(embedding) = self.embedder.embed(&fact_text).await {
                            self.graph
                                .set_fact_embedding(fact_id, &embedding)
                                .await
                                .ok();
                        }
                        inserted_fact_ids.push(fact_id);
                    }
                    Ok(None) => {
                        // ADR-035 Path X: caller pre-pinned this triple via with_facts;
                        // LLM duplicate silently deduped. Counter already incremented in
                        // try_insert_fact. This is the expected silent-dedup path.
                        tracing::debug!(
                            subject = %fact.subject,
                            predicate = %fact.predicate,
                            object = %fact.object,
                            "kremory.ingest.phase2_fact_dedup_against_prior_pin"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            subject = %fact.subject,
                            predicate = %fact.predicate,
                            object = %fact.object,
                            error = %e,
                            "kremory.ingest.phase2_fact_insert_failed"
                        );
                    }
                }

                // Bug A: subject-side episodic edge is now written in the entity loop
                // (role="mention"), guaranteeing coverage for all extracted entities
                // regardless of whether they appear in facts.
                // The object-side episodic edge is retained here to handle fact objects
                // that are stubs or entities not present in the current extraction batch.
                if let Some(ref obj_id) = object_id {
                    if name_to_id.contains_key(&normalize_name(&fact.object)) {
                        self.graph
                            .insert_episodic_edge(episode_id, obj_id, "object")
                            .await
                            .ok();
                    }
                }
            }

            Ok(())
        }; // end 'phases block

        // Commit or rollback the outer transaction based on phase result.
        match phase_result {
            Ok(()) => outer_guard.commit().await?,
            Err(e) => {
                let _ = outer_guard.rollback().await;
                return Err(e);
            }
        }

        let total_ms = ingest_start.elapsed().as_secs_f64() * 1000.0;
        let entity_count = upserted_entities.len();
        let fact_count = inserted_fact_ids.len();
        let merge_count = merged_entities.len();
        let contradiction_count = invalidated_fact_ids.len();
        histogram!("rql.ingest.total_ms").record(total_ms);
        // TD-019 Gap 2: phase1_ms exposes the sync ingest-with-extraction phase
        // separately from total_ms so consumers can distinguish user-facing
        // latency (this) from deferred relationship work (phase2_ms).
        histogram!("rql.ingest.phase1_ms").record(total_ms);
        histogram!("rql.ingest.entity_count").record(entity_count as f64);
        histogram!("rql.ingest.fact_count").record(fact_count as f64);
        histogram!("rql.ingest.merge_count").record(merge_count as f64);
        histogram!("rql.ingest.contradiction_count").record(contradiction_count as f64);
        tracing::info!(
            total_ms,
            entity_count,
            fact_count,
            merge_count,
            contradiction_count,
            stub_entities_inserted,
            "kremory.ingest.completed"
        );

        Ok(IngestionResult {
            episode_id,
            upserted_entities,
            inserted_fact_ids,
            invalidated_fact_ids,
            merged_entities,
            token_usage,
            stub_entities_inserted,
        })
    }

    /// Phase 2 deferred LLM fact extraction.
    ///
    /// Called by the background worker when the NER channel is idle.  Takes the
    /// same text that was processed in Phase 1 along with the entity names that
    /// NER already inserted, runs the LLM extractor to discover relationship
    /// triplets, and stores them linked to the existing episode.
    ///
    /// Returns the number of facts successfully inserted.
    ///
    /// Entity insertion is skipped — Phase 1 (NER) already owns that path.
    /// Only facts (relationship triplets) are added in this phase.
    // Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
    #[allow(clippy::too_many_arguments)]
    pub async fn ingest_deferred(
        &self,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<&str>,
        content_type: Option<ContentType>,
        episode_id: i64,
        ner_entity_names: &[String],
    ) -> Result<usize> {
        // TD-019 Gap 2: phase2_ms — deferred relationship extraction latency,
        // distinct from phase1_ms (user-facing sync work).
        let phase2_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);

        // Build known_entities hint list from the NER entity names.
        let known_entities: Vec<ExtractedEntity> = ner_entity_names
            .iter()
            .map(|name| ExtractedEntity {
                label: String::new(),
                name: name.clone(),
                properties: serde_json::Value::Null,
            })
            .collect();

        // Run LLM extractor to obtain relationship triplets.
        let extractor = Arc::clone(&self.extractor);
        let splitter = ExtractionWindowSplitter::new(self.config.extraction_window.clone());
        let chunks = splitter.split(text, &content_type);

        // L2: load registry so deferred-fact extraction can inject registry specs
        // into prompts. Uses the same effective_gid as the primary ingest path.
        let deferred_effective_gid = group_id.unwrap_or("default");
        let deferred_registry =
            EntityTypeRegistry::load_for_group(&self.graph.conn, deferred_effective_gid).await?;

        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &self.config.allowed_entity_types,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &known_entities,
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: content_type.clone(),
                registry_specs: deferred_registry.specs(),
                // Deferred-fact extraction is relationship-only (entities are
                // pre-supplied as `ner_entity_names`); existing-entity injection
                // is not needed here — the NER names already serve as the anchor.
                existing_graph_entities: &[],
                arm_budget_ms: self.config.extraction_arm_budget_ms,
            };
            let result = extractor.extract(chunk, &ctx).await?;
            all_facts.extend(result.facts);
        }

        if all_facts.is_empty() {
            histogram!("rql.ingest.deferred_fact_count").record(0.0);
            tracing::info!(
                deferred_fact_count = 0,
                "kremory.ingest.deferred_facts empty"
            );
            return Ok(0);
        }

        // Build a name → entity_id map by resolving against the existing graph (namespace-scoped).
        let existing_entities = match group_id {
            Some(gid) => self.graph.list_entities_in_group(gid).await?,
            None => self.graph.list_entities().await?,
        };
        let mut name_to_id: HashMap<String, String> = HashMap::new();
        for entity in &existing_entities {
            // entity.label is the type name ("Person", "Entity") — not the entity name.
            // Use entity.id (normalized name) + properties["name"] (original case) as
            // lookup keys so fact subject/object resolution finds existing entities.
            if let Some(name_val) = entity.properties.get("name").and_then(|v| v.as_str()) {
                name_to_id.insert(normalize_name(name_val), entity.id.clone());
            }
            name_to_id.insert(normalize_name(&entity.id), entity.id.clone());
        }
        // Also seed the map with the NER entity names directly so facts that
        // reference them by their original surface form are resolved correctly.
        for name in ner_entity_names {
            let norm = normalize_name(name);
            name_to_id
                .entry(norm)
                .or_insert_with(|| normalize_name(name));
        }

        // Store facts (contradiction detection + insert), same logic as ingest_with.
        let detector = TwoPoolDetector::new(Arc::clone(&self.llm));
        let mut inserted_count: usize = 0;

        for fact in &all_facts {
            let subject_id = name_to_id
                .get(&normalize_name(&fact.subject))
                .cloned()
                .unwrap_or_else(|| normalize_name(&fact.subject));

            let object_id = if fact.is_entity_ref {
                Some(
                    name_to_id
                        .get(&normalize_name(&fact.object))
                        .cloned()
                        .unwrap_or_else(|| normalize_name(&fact.object)),
                )
            } else {
                None
            };

            let object_value = if !fact.is_entity_ref {
                Some(fact.object.as_str())
            } else {
                None
            };

            let pool_a = self
                .graph
                .get_facts_by_subject_predicate(&subject_id, &fact.predicate)
                .await?;

            let pool_b_hits = self
                .graph
                .fts_search_facts(&fact.predicate, 10, &SearchFilters::new())
                .await?;
            let pool_b: Vec<crate::core::schema::Fact> =
                pool_b_hits.into_iter().map(|h| h.item).collect();

            let contradiction_result = detector.detect(fact, &pool_a, &pool_b, &ref_time).await?;

            for fact_id in &contradiction_result.contradictions {
                self.graph
                    .invalidate_fact_with_reason(*fact_id, Utc::now(), ref_time)
                    .await?;
            }

            // ADR-035 §5 Option A: use try_insert_fact for caller-pin dedup parity
            // with the inline path. Deferred Phase 2 writes also silently skip
            // triples already pre-pinned by the caller via with_facts.
            match self
                .graph
                .try_insert_fact(
                    &subject_id,
                    &fact.predicate,
                    object_id.as_deref(),
                    object_value,
                    ref_time,
                    fact.confidence,
                    Some(episode_id),
                    None,
                )
                .await
            {
                Ok(Some(fact_id)) => {
                    let fact_text = format!("{} {} {}", fact.subject, fact.predicate, fact.object);
                    if let Ok(embedding) = self.embedder.embed(&fact_text).await {
                        self.graph
                            .set_fact_embedding(fact_id, &embedding)
                            .await
                            .ok();
                    }
                    if name_to_id.contains_key(&normalize_name(&fact.subject)) {
                        self.graph
                            .insert_episodic_edge(episode_id, &subject_id, "subject")
                            .await
                            .ok();
                    }
                    if let Some(ref obj_id) = object_id {
                        if name_to_id.contains_key(&normalize_name(&fact.object)) {
                            self.graph
                                .insert_episodic_edge(episode_id, obj_id, "object")
                                .await
                                .ok();
                        }
                    }
                    inserted_count += 1;
                }
                Ok(None) => {
                    tracing::debug!(
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        object = %fact.object,
                        "kremory.ingest.deferred_phase2_fact_dedup_against_prior_pin"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        object = %fact.object,
                        error = %e,
                        "kremory.ingest.deferred_phase2_fact_insert_failed"
                    );
                }
            }
        }

        histogram!("rql.ingest.deferred_fact_count").record(inserted_count as f64);
        histogram!("rql.ingest.phase2_ms")
            .record(phase2_start.elapsed().as_secs_f64() * 1000.0);
        tracing::info!(inserted_count, "kremory.ingest.deferred_facts inserted");
        Ok(inserted_count)
    }
}

// ─── Ingest helpers ──────────────────────────────────────────────────────────

/// Extract a verbatim context snippet for an entity's first mention.
///
/// Searches for `name` (case-insensitive) in `text` and returns a ±`half_window`
/// character window around the first match. If the name is not found in `text`
/// (e.g. LLM hallucinated the entity), returns the first `half_window * 2` chars
/// of `text` as a fallback so the property is never empty.
fn extract_context_snippet(text: &str, name: &str, half_window: usize) -> String {
    /// Walk a byte offset forward to the nearest valid UTF-8 char boundary (ceiling).
    fn ceil_char_boundary(s: &str, byte_pos: usize) -> usize {
        let mut pos = byte_pos.min(s.len());
        while pos < s.len() && !s.is_char_boundary(pos) {
            pos += 1;
        }
        pos
    }
    /// Walk a byte offset backward to the nearest valid UTF-8 char boundary (floor).
    fn floor_char_boundary(s: &str, byte_pos: usize) -> usize {
        let mut pos = byte_pos.min(s.len());
        while pos > 0 && !s.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }

    let lower_text = text.to_lowercase();
    let lower_name = name.to_lowercase();
    if let Some(pos) = lower_text.find(lower_name.as_str()) {
        let raw_start = pos.saturating_sub(half_window);
        let raw_end = (pos + name.len() + half_window).min(text.len());
        let start = floor_char_boundary(text, raw_start);
        let end = ceil_char_boundary(text, raw_end);
        text[start..end].to_owned()
    } else {
        // Name not found in text — use the opening portion as a fallback.
        let raw_end = half_window.saturating_mul(2).min(text.len());
        let end = ceil_char_boundary(text, raw_end);
        text[..end].to_owned()
    }
}

/// Convenience type alias for tests and simple usage (no LLM/embedding).
///
/// Gated: test-infra only — not part of the production public API.
#[cfg(any(test, feature = "test-utils"))]
pub type SimpleGraph = Engine<MockChatProvider, NullEmbeddingProvider>;

#[cfg(any(test, feature = "test-utils"))]
impl SimpleGraph {
    /// Open an in-memory graph with null providers and default config.
    pub async fn open_in_memory_simple() -> Result<Self> {
        let graph = Arc::new(TemporalGraph::open_in_memory().await?);
        let config = PipelineConfig::builder().build()?;
        Self::new(
            graph,
            Arc::new(MockChatProvider::null()),
            Arc::new(NullEmbeddingProvider {
                dim: config.embedding_dim.0,
            }),
            config,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::intelligence::{ExtractionContext, ExtractionResult};
    use crate::core::provider::{MockChatProvider, MockEmbeddingProvider};
    use std::collections::HashMap;

    /// A FixedExtractor returns a predetermined entity list regardless of input text.
    /// Used to simulate an LLM that missed certain proper nouns.
    struct FixedExtractor {
        entities: Vec<ExtractedEntity>,
    }

    impl EntityExtractor for FixedExtractor {
        async fn extract<'a>(
            &'a self,
            _text: &'a str,
            _ctx: &'a ExtractionContext<'a>,
        ) -> crate::core::error::Result<ExtractionResult> {
            Ok(ExtractionResult {
                entities: self.entities.clone(),
                facts: vec![],
            })
        }
    }

    /// Build a MockChatProvider with staged responses matching the prompt substrings
    /// used by NuExtractExtractor, DefaultExtractor, CascadeResolver, and TwoPoolDetector.
    fn build_mock_llm(
        entities_json: &str,
        relations_json: &str,
        triplets_json: &str,
        resolution_response: &str,
        contradiction_response: &str,
    ) -> MockChatProvider {
        let mut map = HashMap::new();
        // NuExtractExtractor prompt contains "# Template:"
        // NuExtract expects a JSON object with "entities" and "relationships" arrays
        let entities: Vec<serde_json::Value> =
            serde_json::from_str(entities_json).unwrap_or_default();
        let relationships: Vec<serde_json::Value> =
            serde_json::from_str(triplets_json).unwrap_or_default();
        let nuextract_response = serde_json::json!({
            "entities": entities,
            "relationships": relationships
        });
        map.insert("# Template:".to_string(), nuextract_response.to_string());
        // DefaultExtractor stage 1: entity extraction prompt ends with this substring
        map.insert(
            "Output a JSON array of objects with \"name\" and \"label\" fields.".to_string(),
            entities_json.to_string(),
        );
        // DefaultExtractor stage 2: relation names prompt ends with this substring
        map.insert(
            "Output a JSON array of relationship name strings.".to_string(),
            relations_json.to_string(),
        );
        // DefaultExtractor stage 3: triplet extraction prompt ends with this substring
        map.insert(
            "Output a concise JSON array of objects with".to_string(),
            triplets_json.to_string(),
        );
        // CascadeResolver LLM escalation (prompt contains "Are these two entities")
        map.insert(
            "Are these two entities".to_string(),
            format!("\"{}\"", resolution_response),
        );
        // TwoPoolDetector (build_dual_list_prompt ends with "Output a JSON array of index numbers")
        map.insert(
            "Output a JSON array of index numbers".to_string(),
            contradiction_response.to_string(),
        );
        MockChatProvider::new(map)
    }

    async fn make_engine_with_mock() -> Engine<MockChatProvider, MockEmbeddingProvider> {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        let config = PipelineConfig::builder()
            .allowed_entity_types(vec![
                "Person".to_string(),
                "Organisation".to_string(),
                "project".to_string(),
                "technology".to_string(),
                "metric".to_string(),
            ])
            .build()
            .unwrap();
        let llm = Arc::new(build_mock_llm(
            r#"[{"name":"Alice","label":"Person"},{"name":"Acme","label":"Organisation"}]"#,
            r#"["works_at"]"#,
            r#"[{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.95}]"#,
            "different",
            "[]",
        ));
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        Engine::new(graph, llm, embedder, config).expect("Engine::new should succeed in tests")
    }

    #[tokio::test]
    async fn test_simple_graph_open() {
        let rql = SimpleGraph::open_in_memory_simple().await;
        assert!(
            rql.is_ok(),
            "SimpleGraph::open_in_memory_simple should succeed"
        );
    }

    #[tokio::test]
    async fn test_ingest_creates_episode() {
        let rql = make_engine_with_mock().await;
        let result = rql
            .ingest(
                "Alice works at Acme",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .unwrap();
        assert!(
            result.episode_id > 0,
            "episode_id should be a positive integer"
        );
    }

    #[tokio::test]
    async fn test_ingest_creates_entities_and_facts() {
        let rql = make_engine_with_mock().await;
        let extractor = Arc::clone(&rql.extractor);
        let result = rql
            .ingest_with(
                &extractor,
                "Alice works at Acme",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .unwrap();

        // Two new entities (alice, acme) should have been upserted
        assert_eq!(
            result.upserted_entities.len(),
            2,
            "should upsert 2 entities: Alice and Acme"
        );

        // One fact (works_at) should have been inserted
        assert_eq!(
            result.inserted_fact_ids.len(),
            1,
            "should insert 1 fact: works_at"
        );

        // No invalidations on a fresh graph
        assert!(
            result.invalidated_fact_ids.is_empty(),
            "no facts should be invalidated on a fresh graph"
        );
    }

    #[tokio::test]
    async fn test_ingest_creates_episodic_edges() {
        let rql = make_engine_with_mock().await;
        let extractor = Arc::clone(&rql.extractor);
        let result = rql
            .ingest_with(
                &extractor,
                "Alice works at Acme",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .unwrap();

        // The fact has subject=alice and object=acme, so we expect 2 episodic edges.
        // We verify indirectly by checking that the entities are stored
        // and can be retrieved via episodic_edges_for_entity.
        let alice_id = result
            .upserted_entities
            .iter()
            .find(|id| id.as_str() == "alice")
            .cloned();
        assert!(
            alice_id.is_some(),
            "alice entity should be in upserted_entities"
        );

        let edges = rql.graph.episodic_edges_for_entity("alice").await.unwrap();
        assert!(
            !edges.is_empty(),
            "alice should have at least one episodic edge"
        );
    }

    #[tokio::test]
    async fn test_ingest_returns_result_with_correct_counts() {
        let rql = make_engine_with_mock().await;
        let extractor = Arc::clone(&rql.extractor);
        let result = rql
            .ingest_with(
                &extractor,
                "Alice works at Acme",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .unwrap();

        // Structural checks on IngestionResult
        assert!(result.episode_id > 0);
        assert_eq!(result.upserted_entities.len(), 2);
        assert_eq!(result.inserted_fact_ids.len(), 1);
        assert!(result.invalidated_fact_ids.is_empty());
        assert!(result.merged_entities.is_empty());
    }

    #[tokio::test]
    async fn test_ingest_entities_stored_in_graph() {
        let rql = make_engine_with_mock().await;
        rql.ingest(
            "Alice works at Acme",
            None,
            None,
            None,
            SourceParams::default(),
        )
        .await
        .unwrap();

        // Alice and Acme should now be in the graph
        let alice = rql.graph.get_entity("alice").await.unwrap();
        let acme = rql.graph.get_entity("acme").await.unwrap();

        assert!(alice.is_some(), "alice should exist in graph after ingest");
        assert!(acme.is_some(), "acme should exist in graph after ingest");
    }

    /// Verifies that scan_proper_nouns() runs automatically after LLM extraction
    /// and catches proper nouns the extractor deliberately omitted.
    ///
    /// The FixedExtractor returns only "Alice"; "Zenith Dynamics" appears in the
    /// The proper noun scanner catches Title-Case tokens the LLM extractor missed.
    /// It assigns label="Entity" (a placeholder) because it cannot classify the type.
    ///
    /// Post-TD-012 fix: the L2 guard at the persistence boundary rejects label="Entity".
    /// This means proper-noun-scanned entities with placeholder labels are correctly
    /// filtered out — we accept fewer entities with correct labels over many with garbage
    /// labels.  The canonical-label entities ("Alice" / "Person") still persist.
    ///
    /// If the caller needs untyped proper nouns persisted, the scanner should be updated
    /// to assign a domain-appropriate canonical label via an LLM call.
    #[tokio::test]
    async fn test_ingest_catches_proper_nouns_missed_by_extractor() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        let config = PipelineConfig::builder().build().unwrap();
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let rql: Engine<MockChatProvider, MockEmbeddingProvider> =
            Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");

        // Extractor provides "Alice" (canonical label). Proper noun scanner will
        // detect "Zenith Dynamics" but assign label="Entity" (placeholder).
        let extractor = FixedExtractor {
            entities: vec![ExtractedEntity {
                name: "Alice".into(),
                label: "Person".into(),
                properties: serde_json::json!({}),
            }],
        };

        let result = rql
            .ingest_with(
                &extractor,
                "Alice discussed the proposal with Zenith Dynamics executives.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .unwrap();

        // "Alice" (Person — canonical) must persist.
        assert!(
            result.upserted_entities.iter().any(|e| e.contains("alice")),
            "canonical-label entity 'Alice' must persist; got: {:?}",
            result.upserted_entities
        );

        // TD-013 Phase 8 v3: scan_proper_nouns DISABLED — pure-LLM extraction
        // matches Graphiti/Cognee/LightRAG peer pattern. "Zenith Dynamics"
        // was previously caught by the scanner; with scanner off, it relies on
        // the LLM extractor. When task #15 re-enables scanner with LLM
        // classification round-trip, restore the original assertion.
        assert!(
            !result
                .upserted_entities
                .iter()
                .any(|e| e.contains("zenith")),
            "Phase 8 v3: scanner-only entities no longer persist; \
             got: {:?}",
            result.upserted_entities
        );

        // TD-013 Phase 8 v3: scanner disabled — only LLM-extracted entities persist.
        let entities = graph.list_entities().await.expect("list");
        assert_eq!(
            entities.len(),
            1,
            "Phase 8 v3 (pure LLM): only LLM-extracted entity persists; got: {entities:?}"
        );
        assert!(
            entities.iter().any(|e| e.id == "alice"),
            "alice must be in graph; got: {entities:?}"
        );
    }

    /// v0.1.4 — soft-warn behaviour: extractor-emitted duplicate entity names
    /// no longer FATAL the ingest. The substrate dedupes silently and emits a
    /// `tracing::warn!` so noisy small-model extractors (llama3.2, gemma4-e2b)
    /// can be used safely.
    ///
    /// Story #150 history: previously this returned
    /// `Err(IntraBatchDuplicate)`. The variant is retained on the error enum
    /// for forward-compat with a possible explicit-strict-batch API in v0.2.0+
    /// but is not currently raised from `ingest_with`.
    #[tokio::test]
    async fn ingest_intra_batch_duplicate_dedupes_silently() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(graph, llm, embedder, config).expect("Engine::new should succeed in tests");

        // Two entities with identical names — normalize to the same id.
        let extractor = FixedExtractor {
            entities: vec![
                ExtractedEntity {
                    name: "Alice Corp".to_string(),
                    label: "Organisation".to_string(),
                    properties: serde_json::Value::Null,
                },
                ExtractedEntity {
                    name: "Alice Corp".to_string(),
                    label: "Organisation".to_string(),
                    properties: serde_json::Value::Null,
                },
            ],
        };

        let result = engine
            .ingest_with(
                &extractor,
                "Alice Corp is a company.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("dup-name batch must succeed (soft-dedup post-v0.1.4)");

        // After dedup exactly one entity must land in the graph.
        let alice_count = result
            .upserted_entities
            .iter()
            .filter(|e| e.to_lowercase().contains("alice"))
            .count();
        assert_eq!(
            alice_count, 1,
            "exactly one Alice entity should be persisted after silent dedup; got: {:?}",
            result.upserted_entities
        );
    }

    /// A FixedExtractor that also returns a predetermined set of facts.
    /// Used by stub pre-scan tests to inject `is_entity_ref=true` facts that
    /// reference entities NOT in the entity list — bypassing the extraction
    /// layer's `is_entity_ref` fixup (which resets the flag based on the entity
    /// list and would silently suppress the forward reference).
    struct FixedExtractorWithFacts {
        entities: Vec<ExtractedEntity>,
        facts: Vec<ExtractedFact>,
    }

    impl EntityExtractor for FixedExtractorWithFacts {
        async fn extract<'a>(
            &'a self,
            _text: &'a str,
            _ctx: &'a ExtractionContext<'a>,
        ) -> crate::core::error::Result<ExtractionResult> {
            Ok(ExtractionResult {
                entities: self.entities.clone(),
                facts: self.facts.clone(),
            })
        }
    }

    /// Bug E (F-3, part 1): stub pre-scan inserts UNKNOWN stub when a fact
    /// references an entity (is_entity_ref=true) that is NOT in the entity list.
    ///
    /// Uses FixedExtractorWithFacts to bypass extraction.rs's is_entity_ref fixup
    /// (which resets the flag based on the entity list and would suppress the
    /// forward reference). Direct construction guarantees is_entity_ref=true for
    /// Bob even though Bob is absent from the entity list.
    #[tokio::test]
    async fn stub_entity_inserted_on_forward_reference_lib() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");

        // Alice in entity list; Bob only referenced in facts (forward reference).
        let extractor = FixedExtractorWithFacts {
            entities: vec![ExtractedEntity {
                name: "Alice".into(),
                label: "Person".into(),
                properties: serde_json::json!({}),
            }],
            facts: vec![ExtractedFact {
                subject: "Alice".into(),
                predicate: "works_with".into(),
                object: "Bob".into(),
                is_entity_ref: true, // Bob is a forward reference
                confidence: 0.9,
            }],
        };

        let result = engine
            .ingest_with(
                &extractor,
                "Alice works with Bob on research.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("ingest_with OK");

        // Pre-scan must have inserted exactly 1 stub (Bob).
        assert_eq!(
            result.stub_entities_inserted, 1,
            "pre-scan must insert 1 stub entity for Bob (forward ref); \
             stub_entities_inserted={} upserted={:?}",
            result.stub_entities_inserted, result.upserted_entities
        );

        // Verify Bob exists in the graph with stub=true and entity_type_id=0 (catch-all).
        // Phase 2 (Migration 009): the label column is gone; stubs use entity_type_id=0
        // which resolves to "Entity" via COALESCE in the SELECT.  The "UNKNOWN" label
        // was a pre-Phase-2 sentinel stored in the now-dropped label column.
        let bob = graph
            .get_entity("bob")
            .await
            .expect("get_entity OK")
            .expect("Bob must exist as stub");
        assert_eq!(
            bob.entity_type_id, 0,
            "stub entity must have entity_type_id=0 (catch-all)"
        );
        assert_eq!(
            bob.label, "Entity",
            "stub entity_type_id=0 resolves to 'Entity' via COALESCE"
        );
        assert_eq!(
            bob.properties.get("stub").and_then(|v| v.as_bool()),
            Some(true),
            "stub entity must have properties.stub=true"
        );
    }

    /// Bug E (F-3 + F-4): stub entity is promoted when a subsequent ingest
    /// extracts the same entity as a real one.
    ///
    /// Ingest 1: Alice + forward-ref Bob (is_entity_ref=true) → Bob inserted as stub.
    /// Ingest 2: Bob extracted with label=Person → resolver matches existing Bob stub
    ///           → merge branch fires → F-4 upsert promotes Bob (stub flag removed).
    #[tokio::test]
    async fn stub_entity_promoted_on_reingestion_lib() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");

        // Ingest 1: Alice + forward-ref Bob → Bob becomes stub.
        let extractor_1 = FixedExtractorWithFacts {
            entities: vec![ExtractedEntity {
                name: "Alice".into(),
                label: "Person".into(),
                properties: serde_json::json!({}),
            }],
            facts: vec![ExtractedFact {
                subject: "Alice".into(),
                predicate: "works_with".into(),
                object: "Bob".into(),
                is_entity_ref: true,
                confidence: 0.9,
            }],
        };
        let r1 = engine
            .ingest_with(
                &extractor_1,
                "Alice works with Bob.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("ingest 1 OK");
        assert_eq!(r1.stub_entities_inserted, 1, "ingest 1 must create 1 stub");

        // Confirm Bob is a stub before promotion.
        // Phase 2: stubs use entity_type_id=0; the "UNKNOWN" label sentinel was in
        // the now-dropped label column.  entity_type_id=0 resolves to "Entity" via
        // COALESCE in the SELECT — assert on entity_type_id, not the label string.
        let bob_pre = graph
            .get_entity("bob")
            .await
            .expect("get OK")
            .expect("Bob exists");
        assert_eq!(
            bob_pre.entity_type_id, 0,
            "Bob must have entity_type_id=0 (catch-all) before promotion"
        );
        assert_eq!(
            bob_pre.properties.get("stub").and_then(|v| v.as_bool()),
            Some(true),
            "Bob must have stub=true before promotion"
        );

        // Ingest 2: Bob now extracted as a full entity.
        let extractor_2 = FixedExtractorWithFacts {
            entities: vec![ExtractedEntity {
                name: "Bob".into(),
                label: "Person".into(),
                properties: serde_json::json!({}),
            }],
            facts: vec![],
        };
        engine
            .ingest_with(
                &extractor_2,
                "Bob is a researcher at Stanford.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("ingest 2 OK");

        // After promotion, Bob must have stub flag removed.
        // Phase 2: entity_type_id is used instead of the dropped label column.
        // On a fresh in-memory DB the entity_types registry only has id=0 ("Entity")
        // so "Person" (unregistered) maps to entity_type_id=0 via label_to_id().
        // The primary invariant here is that the stub flag is cleared on promotion;
        // type-label accuracy is tested by label_precision_benchmark (requires Ollama).
        let bob_post = graph
            .get_entity("bob")
            .await
            .expect("get OK")
            .expect("Bob still exists");
        let stub_flag = bob_post
            .properties
            .get("stub")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(
            !stub_flag,
            "promoted Bob must not have stub=true in properties"
        );
    }

    /// v0.1.4 — soft-dedup ingest writes exactly one row per duplicated name.
    /// Previously (Story #150): after FATAL `IntraBatchDuplicate`, zero rows
    /// were written. Post-v0.1.4: the substrate dedupes silently so one row
    /// per normalized name lands, regardless of the extractor's noise.
    #[tokio::test]
    async fn ingest_intra_batch_duplicate_writes_one_per_name() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));

        let before_count = graph.list_entities().await.expect("list").len();

        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");

        // Use a canonical label ("Person") so the TD-012 guard does not filter
        // these out before the dedup logic runs.  The test invariant is dedup,
        // not label validation — use a label that passes the L2 allowlist check.
        let extractor = FixedExtractor {
            entities: vec![
                ExtractedEntity {
                    name: "Dup Entity".to_string(),
                    label: "Person".to_string(),
                    properties: serde_json::Value::Null,
                },
                ExtractedEntity {
                    name: "Dup Entity".to_string(),
                    label: "Person".to_string(),
                    properties: serde_json::Value::Null,
                },
            ],
        };

        let result = engine
            .ingest_with(
                &extractor,
                "two duplicates submitted in single batch.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("dup-name batch must succeed (soft-dedup post-v0.1.4)");

        // Exactly one row per normalized name lands.
        let after_count = engine.graph.list_entities().await.expect("list").len();
        assert_eq!(
            after_count,
            before_count + 1,
            "exactly one entity row must be written after silent dedup; got delta={}",
            after_count - before_count
        );
        assert_eq!(
            result.upserted_entities.len(),
            1,
            "upserted_entities should report exactly one entity after dedup; got: {:?}",
            result.upserted_entities
        );
    }

    // ── P7b Red: Engine model field threading tests ──────────────────────────
    //
    // These tests target AC6-AC8: Engine must store the model string supplied
    // by the LLM provider at construction time via llm.model().
    //
    // All three tests FAIL until Green phase adds `model: Option<String>` to
    // the Engine struct and derives it in Engine::new().expect("Engine::new should succeed in tests").

    /// AC6+AC8: Engine stores the model string from the LLM provider at construction.
    ///
    /// When llm.model() returns a non-empty string the Engine must store
    /// `Some(model_str)` in its `model` field.
    ///
    /// FAILS (Red) until Engine gains `pub(crate) model: Option<String>` field.
    #[tokio::test]
    async fn engine_stores_model_string_from_llm() {
        struct ModelledMock {
            model_str: &'static str,
        }

        #[async_trait::async_trait]
        impl ChatProvider for ModelledMock {
            async fn chat_with_tools(
                &self,
                _messages: &[crate::core::provider::ChatMessage],
                _tools: Option<&[autoagents_llm::chat::Tool]>,
                _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
            ) -> std::result::Result<
                Box<dyn autoagents_llm::chat::ChatResponse>,
                autoagents_llm::error::LLMError,
            > {
                Err(autoagents_llm::error::LLMError::Generic(
                    "not needed in this test".to_string(),
                ))
            }

            fn model(&self) -> &str {
                self.model_str
            }
        }

        let graph = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        let config = PipelineConfig::builder().build().unwrap();
        let llm = Arc::new(ModelledMock {
            model_str: "qwen2.5:14b",
        });
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(graph, llm, embedder, config).expect("Engine::new should succeed in tests");

        // AC8: model field must hold Some("qwen2.5:14b") — the value returned by llm.model().
        assert_eq!(
            engine.model.as_deref(),
            Some("qwen2.5:14b"),
            "Engine::new must capture llm.model() into engine.model; \
             got: {:?}",
            engine.model
        );
    }

    /// AC8 (empty-string branch): when llm.model() returns "" the Engine must
    /// store `None` — the empty string is semantically "unknown model" and should
    /// not propagate as a model identifier.
    ///
    /// FAILS (Red) until Engine gains `pub(crate) model: Option<String>` field.
    #[tokio::test]
    async fn engine_model_is_none_when_llm_model_empty() {
        // MockChatProvider::null() uses the default ChatProvider::model() impl
        // which returns "" — exactly the "no model configured" case.
        let graph = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        let config = PipelineConfig::builder().build().unwrap();
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(graph, llm, embedder, config).expect("Engine::new should succeed in tests");

        // AC8: empty model string must map to None, not Some("").
        assert_eq!(
            engine.model, None,
            "Engine::new must store None when llm.model() returns empty string; \
             got: {:?}",
            engine.model
        );
    }

    /// AC7: Engine::new signature must remain (graph, llm, embedder, config) —
    /// no new parameters. The model is derived from llm.model() internally.
    ///
    /// This test verifies AC7 by constructing Engine::new with the existing
    /// 4-argument signature and asserting the model is populated from the
    /// provider's model() method — not from a separate argument.
    ///
    /// FAILS (Red) until Engine gains `pub(crate) model: Option<String>` field.
    #[tokio::test]
    async fn engine_new_signature_unchanged_model_derived_from_llm() {
        struct NamedMock;

        #[async_trait::async_trait]
        impl ChatProvider for NamedMock {
            async fn chat_with_tools(
                &self,
                _messages: &[crate::core::provider::ChatMessage],
                _tools: Option<&[autoagents_llm::chat::Tool]>,
                _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
            ) -> std::result::Result<
                Box<dyn autoagents_llm::chat::ChatResponse>,
                autoagents_llm::error::LLMError,
            > {
                Err(autoagents_llm::error::LLMError::Generic(
                    "not needed".to_string(),
                ))
            }

            fn model(&self) -> &str {
                "llama3.2:3b-instruct"
            }
        }

        let graph = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        let config = PipelineConfig::builder().build().unwrap();
        // Exactly 4 arguments to Engine::new — signature unchanged per AC7.
        let engine = Engine::new(
            graph,
            Arc::new(NamedMock),
            Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0)),
            config,
        ).expect("Engine::new should succeed in tests");

        assert_eq!(
            engine.model.as_deref(),
            Some("llama3.2:3b-instruct"),
            "model must be derived from llm.model() without adding new Engine::new params"
        );
    }

    // ── End P7b Red tests ─────────────────────────────────────────────────────

    /// TD-013 Phase 8 finalises Vera M1: the TD-012 over-rejection guard is
    /// DELETED. Under the L1 integer-ID design, "Entity" id=0 is the legitimate
    /// catch-all (per Graphiti peer pattern); "UNKNOWN" / unrecognised labels
    /// also resolve to id=0 via registry.label_to_id fallback. These entities
    /// MUST be persisted, not rejected.
    ///
    /// Replacement guardrails (all wired by Phase 1-7):
    /// - L3 validate_or_fallback bounds-checks emitted entity_type_id
    /// - L4 disambiguation handles surface-form duplicates
    /// - L5 vector canonicalisation merges variant spellings
    /// - L7 dream-phase reclassification upgrades id=0 entities once corpus
    ///   accumulates 3+ episodes per entity
    ///
    /// Invariant: a FixedExtractor returning "Entity"-labelled entities MUST
    /// persist them with entity_type_id=0 (catch-all). Earlier behaviour was
    /// to drop them entirely — that was the TD-012 mistake reverted here.
    #[tokio::test]
    async fn ingest_persists_entity_catchall_under_l1_design() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));

        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");

        let extractor = FixedExtractor {
            entities: vec![
                ExtractedEntity {
                    name: "Some Person".to_string(),
                    label: "Entity".to_string(), // id=0 catch-all — must persist
                    properties: serde_json::Value::Null,
                },
                ExtractedEntity {
                    name: "Unknown Thing".to_string(),
                    label: "UNKNOWN".to_string(), // unrecognised → registry fallback to id=0
                    properties: serde_json::Value::Null,
                },
            ],
        };

        let result = engine
            .ingest_with(
                &extractor,
                "test text for L1 catch-all persistence.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("ingest must succeed");

        // Both entities must be persisted with entity_type_id=0 (catch-all).
        assert_eq!(
            result.upserted_entities.len(),
            2,
            "L1 design: both Entity + UNKNOWN labels persist as id=0 catch-all; got {:?}",
            result.upserted_entities
        );

        // Verify via direct SQL — both entities have entity_type_id=0.
        let mut rows = engine
            .graph
            .conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE entity_type_id = 0",
                libsql::params![],
            )
            .await
            .expect("count");
        let count: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("some")
            .get(0)
            .expect("col");
        assert!(
            count >= 2,
            "expected >= 2 entities with entity_type_id=0; got {count}"
        );
    }

    // ── TD-013 per-call entity_types override tests ───────────────────────────
    //
    // Tests #4-#6 (spec §1, Graphiti pattern).
    //
    // Tests #4 and #5 call ingest_with with `entity_types_override: Some(...)` on
    // SourceParams — a field that does not yet exist.
    // Expected failure: E0560 (struct `SourceParams` has no field named `entity_types_override`).
    //
    // Test #6 uses `entity_types_override: None` (the absent-field variant) — it will
    // also fail to compile until the field exists, documenting the CURRENT broken
    // behaviour as a regression baseline.

    /// #4 — entity_types_override persists on first call then loads from DB on second call.
    ///
    /// First ingest_with: override Some([Entity(0), Person(1), Organisation(2)]).
    ///   → entity_types must gain 3 rows for the namespace.
    ///   → Alice stored with entity_type_id=1 (Person).
    ///   → Acme stored with entity_type_id=2 (Organisation).
    ///
    /// Second ingest_with: override None (DB already has rows).
    ///   → Registry loaded from DB.
    ///   → Subsequent entity resolution still uses the persisted type ids.
    ///
    /// COMPILE FAIL (Red): SourceParams has no `entity_types_override` field yet.
    #[tokio::test]
    async fn test_ingest_with_entity_types_override_persists_on_first_call_then_reuses() {
        use crate::core::entity_types::EntityTypeSpec;

        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder()
            .allowed_entity_types(vec![
                "Person".to_string(),
                "Organisation".to_string(),
            ])
            .build()
            .expect("config");

        // Mock that returns Alice(Person) + Acme(Organisation) as a NuExtract response.
        let llm = Arc::new(build_mock_llm(
            r#"[{"name":"Alice","label":"Person"},{"name":"Acme","label":"Organisation"}]"#,
            r#"[]"#,
            r#"[]"#,
            "different",
            "[]",
        ));
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");
        let extractor =
            Arc::clone(&engine.extractor);

        let override_specs = vec![
            EntityTypeSpec { id: 0, name: "Entity".to_string(), description: "Catch-all.".to_string() },
            EntityTypeSpec { id: 1, name: "Person".to_string(), description: "A human individual.".to_string() },
            EntityTypeSpec { id: 2, name: "Organisation".to_string(), description: "A company or institution.".to_string() },
        ];

        // ── First call: override present, DB empty for "fresh" group ─────────────
        let r1 = engine
            .ingest_with(
                &extractor,
                "Alice works at Acme",
                None,
                Some("fresh"),
                None,
                SourceParams {
                    entity_types_override: Some(override_specs.clone()),
                    ..SourceParams::default()
                },
            )
            .await
            .expect("first ingest_with must succeed");

        // entity_types for "fresh" must now have 3 rows (persisted on first call).
        let mut rows = engine
            .graph
            .conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id = 'fresh'",
                libsql::params![],
            )
            .await
            .expect("count query after first call");
        let count_row = rows.next().await.expect("row iter").expect("row");
        let type_count: i64 = count_row.get(0).expect("count col");
        // Migration 010 lazy-seeds the default 10-row vocabulary for any new
        // group_id at the start of ingest_with. The override's 3 specs are
        // subset of the defaults (Entity, Person, Organisation), so upsert
        // no-ops. The substrate invariant is that the registry IS populated
        // (≥10 rows means defaults are in place; override augmented if its
        // ids extend beyond the defaults).
        assert!(
            type_count >= 10,
            "#4: entity_types must have at least 10 rows for 'fresh' (defaults + override); got {type_count}"
        );

        // NOTE: entity_type_id flow-through assertions are validated by
        // label_precision_benchmark (live qwen2.5:14b) — the mock extraction
        // path here cannot reproduce the real registry → L3 validation flow
        // because the JSON mock format does not match NuExtractExtractor's
        // parser. The substrate invariant covered by THIS test is the
        // registry-population side (override → upsert → DB has rows).

        // ── Second-call (no override → DB load) path is covered by test #2 in
        // entity_types.rs (test_upsert_entity_types_noop_when_rows_exist) and
        // by the no_override_fresh_db_extracts_zero_typed_entities test below.
        // Combining them with the mock-extractor setup here adds no extra signal.

        let _ = r1;
    }

    /// #5 — entity_types_override is ADDITIVELY MERGED when the DB has rows (TD-023).
    ///
    /// Was previously "ephemeral, do not touch DB" (TD-013 design). That created
    /// a JOIN hole: an entity row stored with entity_type_id = an override-only
    /// id had no entity_types row, so SQL COALESCE(et.name, 'Entity') resolved
    /// label='Entity' at read time regardless of the stored integer id. The
    /// TD-023 hybrid extractor exposed this — see
    /// [[td022-gliner-empirical-speed-wins-precision-ties-or-loses]].
    ///
    /// New invariant: pre-existing rows survive unchanged AND missing override
    /// types are added (INSERT OR IGNORE — additive only, no clobber).
    ///
    /// Setup: entity_types pre-populated for "g1" with [Entity(0), Person(1)].
    /// Action: ingest_with with override Some([Entity(0), CustomType(5)]) for "g1".
    /// Assert:
    ///   - entity_types for "g1" has 3 rows (2 original + CustomType added).
    ///   - Entity(0) + Person(1) rows still present (additive, not clobber).
    ///   - CustomType(5) row now present (so future JOINs resolve correctly).
    ///
    /// COMPILE FAIL (Red): SourceParams has no `entity_types_override` field yet.
    #[tokio::test]
    async fn test_ingest_with_entity_types_override_ephemeral_when_db_has_rows() {
        use crate::core::entity_types::EntityTypeSpec;

        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));

        // Pre-seed entity_types for "g1" with 2 rows.
        graph
            .conn
            .execute(
                "INSERT OR IGNORE INTO entity_types (group_id, id, name, description) \
                 VALUES ('g1', 0, 'Entity', 'catch-all'), ('g1', 1, 'Person', 'A person.')",
                libsql::params![],
            )
            .await
            .expect("pre-seed entity_types for g1");

        let config = PipelineConfig::builder()
            .allowed_entity_types(vec!["CustomType".to_string()])
            .build()
            .expect("config");

        // Mock returns one entity with label "CustomType" (id=5 in the override).
        let mut mock_map = HashMap::new();
        let nuextract_response = serde_json::json!({
            "entities": [{"name": "WidgetCo", "label": "CustomType"}],
            "relationships": []
        });
        mock_map.insert("# Template:".to_string(), nuextract_response.to_string());
        mock_map.insert(
            "Output a JSON array of index numbers".to_string(),
            "[]".to_string(),
        );
        let llm = Arc::new(MockChatProvider::new(mock_map));
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");
        let extractor =
            Arc::clone(&engine.extractor);

        // Override with a type list that includes CustomType(5) — not in the DB.
        let override_specs = vec![
            EntityTypeSpec { id: 0, name: "Entity".to_string(), description: "catch-all".to_string() },
            EntityTypeSpec { id: 5, name: "CustomType".to_string(), description: "A custom type.".to_string() },
        ];

        let _ = engine
            .ingest_with(
                &extractor,
                "WidgetCo is a company.",
                None,
                Some("g1"),
                None,
                SourceParams {
                    entity_types_override: Some(override_specs),
                    ..SourceParams::default()
                },
            )
            .await
            .expect("ingest_with with ephemeral override must succeed");

        // TD-023: DB has 3 rows for "g1" — original 2 (Entity, Person) PLUS
        // CustomType added by the additive merge.
        let mut rows = engine
            .graph
            .conn
            .query(
                "SELECT id, name FROM entity_types WHERE group_id = 'g1' ORDER BY id",
                libsql::params![],
            )
            .await
            .expect("count query");
        let mut found: Vec<(i64, String)> = Vec::new();
        while let Some(r) = rows.next().await.expect("row iter") {
            let id: i64 = r.get(0).expect("id col");
            let name: String = r.get(1).expect("name col");
            found.push((id, name));
        }
        assert_eq!(
            found.len(),
            3,
            "#5 TD-023: entity_types for 'g1' must have 3 rows after additive merge \
             (2 original + 1 added override type); got {:?}",
            found
        );
        assert!(
            found.iter().any(|(id, n)| *id == 0 && n == "Entity"),
            "#5 TD-023: original Entity(0) row preserved; got {:?}",
            found
        );
        assert!(
            found.iter().any(|(id, n)| *id == 1 && n == "Person"),
            "#5 TD-023: original Person(1) row preserved; got {:?}",
            found
        );
        assert!(
            found.iter().any(|(id, n)| *id == 5 && n == "CustomType"),
            "#5 TD-023: CustomType(5) row added by additive merge; got {:?}",
            found
        );
    }

    /// #6 — baseline: no override + empty DB produces zero typed entities (the bug we're fixing).
    ///
    /// This test documents the CURRENT broken behaviour before the per-call override fix.
    /// It is a regression baseline — not a spec of desired behaviour.
    ///
    /// When entity_types is empty for the group and no override is provided, the registry
    /// is empty. The L3 validation maps all entity type ids to 0 (catch-all). Entities
    /// ARE still persisted (the extractor runs), but they all land with entity_type_id=0
    /// instead of meaningful ids.
    ///
    /// This test COMPILES once the `entity_types_override` field exists on SourceParams
    /// (even with `None`). It may PASS or FAIL depending on current code — that is
    /// acceptable and documented. The test's role is to pin the current behaviour so that
    /// the Green phase fix (which provides a caller-supplied vocabulary) does not silently
    /// regress this path.
    ///
    /// COMPILE FAIL (Red): SourceParams has no `entity_types_override` field yet.
    #[tokio::test]
    async fn test_ingest_with_no_override_fresh_db_extracts_zero_typed_entities() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder()
            .allowed_entity_types(vec!["Person".to_string(), "Organisation".to_string()])
            .build()
            .expect("config");

        // Mock returns Person + Organisation — but entity_types is empty for "g_fresh".
        let llm = Arc::new(build_mock_llm(
            r#"[{"name":"Alice","label":"Person"},{"name":"Acme","label":"Organisation"}]"#,
            r#"[]"#,
            r#"[]"#,
            "different",
            "[]",
        ));
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");
        let extractor =
            Arc::clone(&engine.extractor);

        let result = engine
            .ingest_with(
                &extractor,
                "Alice works at Acme",
                None,
                Some("g_fresh"),
                None,
                SourceParams {
                    entity_types_override: None,
                    ..SourceParams::default()
                },
            )
            .await
            .expect("ingest_with without override must not error");

        // Post-Migration-010 / lazy-seed: even WITHOUT a per-call override,
        // the default vocabulary (Entity, Person, Organisation, Location, ...)
        // is seeded for "g_fresh" at the start of ingest_with. The registry
        // resolves the mock LLM's "Person" / "Organisation" labels to ids
        // 1 / 2 respectively. This is the substrate guarantee that consumers
        // can call ingest on a fresh namespace and get typed entities out of
        // the box without pre-knowing the vocabulary.
        //
        // Assertion: at least one persisted entity should have a non-zero
        // entity_type_id (proving the registry → label → id flow works).
        let mut typed_count = 0usize;
        for entity_id in &result.upserted_entities {
            if let Ok(Some(entity)) = engine.graph.get_entity(entity_id).await {
                if entity.entity_type_id != 0 {
                    typed_count += 1;
                }
            }
        }
        assert!(
            typed_count >= 1,
            "#6: at least one persisted entity should have a non-zero entity_type_id \
             (Migration 010 seeds defaults so 'Person'/'Organisation' labels resolve); got typed_count={typed_count}"
        );
    }

    /// M5 fix (PR #1 review finding): runtime-configured `allowed_entity_types`
    /// must be honored by the L2 guard at persistence boundary.
    ///
    /// Pre-fix: caller configures `allowed_entity_types: ["project"]`, LLM
    /// emits an entity labelled `"project"` → silently dropped because
    /// `ENTITY_TYPE_ALLOWLIST` did not include lowercase `"project"`.
    ///
    /// Invariant: entities labelled with a runtime-allowed type MUST be
    /// persisted even when the label is absent from the compile-time allowlist.
    #[tokio::test]
    async fn ingest_persists_runtime_allowed_entity_label() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        // Runtime configures a domain-specific lowercase label not in the
        // compile-time ENTITY_TYPE_ALLOWLIST.
        let config = PipelineConfig::builder()
            .allowed_entity_types(vec!["project".to_string()])
            .build()
            .expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));

        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config).expect("Engine::new should succeed in tests");

        // Inject an entity whose label matches the runtime config but NOT the
        // compile-time allowlist. Without the M5 fix this is silently rejected.
        let extractor = FixedExtractor {
            entities: vec![ExtractedEntity {
                name: "Apollo Mission".to_string(),
                label: "project".to_string(),
                properties: serde_json::Value::Null,
            }],
        };

        let result = engine
            .ingest_with(
                &extractor,
                "some text mentioning a runtime-allowed entity type.",
                None,
                None,
                None,
                SourceParams::default(),
            )
            .await
            .expect("ingest must succeed");

        let after_entities = engine.graph.list_entities().await.expect("list");
        assert_eq!(
            after_entities.len(),
            1,
            "M5: runtime-allowed entity label must be persisted; \
             got {} entities",
            after_entities.len()
        );
        assert_eq!(
            result.upserted_entities.len(),
            1,
            "M5: upserted_entities must contain the runtime-allowed entity; \
             got: {:?}",
            result.upserted_entities
        );
    }
}
