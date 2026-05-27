use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use metrics::histogram;

use chrono::{DateTime, Utc};

use crate::core::config::{ContentType, PipelineConfig};
use crate::core::contradiction::TwoPoolDetector;
use crate::core::error::Result;
use crate::core::extraction::NuExtractExtractor;
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
}

impl<L: ChatProvider, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Create a new `Engine` wrapping a `TemporalGraph`.
    ///
    /// Accepts `Arc<TemporalGraph>` so callers can share the same graph
    /// instance with the process-global singleton returned by `engine()`.
    pub fn new(
        graph: Arc<TemporalGraph>,
        llm: Arc<L>,
        embedder: Arc<Emb>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            graph,
            llm,
            embedder,
            config,
            oov_auditor: None,
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
        let properties = serde_json::json!({ "text": text });
        self.graph.insert_entity(source, title, properties).await?;

        // 2. Embed the full document text for vector search.
        let embedding = self.embedder.embed(text).await?;
        self.graph.set_entity_embedding(source, &embedding).await?;

        // 3. Run the intelligence pipeline on the document content.
        let doc_text = format!("# {title}\n\n{text}");
        self.ingest(&doc_text, None, None, Some(ContentType::Document))
            .await
    }

    /// Full pipeline: text → chunk → extract → resolve → contradict → store.
    /// Uses `NuExtractExtractor` (unified extraction template). For alternative extractors,
    /// use `ingest_with()`.
    pub async fn ingest(
        &self,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<&str>,
        content_type: Option<ContentType>,
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
                .ingest_with(extractor, text, reference_time, group_id, content_type)
                .await;
        }
        #[cfg(not(feature = "ner"))]
        {
            let extractor = NuExtractExtractor::new(Arc::clone(&self.llm));
            self.ingest_with(&extractor, text, reference_time, group_id, content_type)
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
    ) -> Result<IngestionResult> {
        let ingest_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);
        let token_usage = TokenUsage::default();

        // 1. Store episode (namespace-scoped via group_id)
        let episode_id = self
            .graph
            .insert_episode_with_group(text, ref_time, Some("ingest"), None, group_id, None, None)
            .await?;

        // 2. Slice into LLM-extraction-prompt windows (no-op for normally-sized episodes;
        //    see core/extraction_window.rs module docstring for kind-2 semantics).
        let splitter = ExtractionWindowSplitter::new(self.config.extraction_window.clone());
        let chunks = splitter.split(text, &content_type);

        let chunk_count = chunks.len();
        histogram!("rql.ingest.chunk_count").record(chunk_count as f64);
        tracing::info!(chunk_count, "kremory.ingest.chunked");

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

        // Post-extraction: scan for proper nouns the LLM missed (domain-agnostic, <2ms)
        let proper_noun_candidates = text_utils::scan_proper_nouns(text, &all_entities);
        all_entities.extend(proper_noun_candidates);

        // 3b. Pre-mutation intra-batch duplicate scan (Tier-3 per-call HashSet, Story #150).
        // Run BEFORE dedup so that a caller who submits two entities with the same
        // normalized name gets a structured error rather than a silent drop. This
        // guards the contract: no DB write occurs for a malformed batch.
        {
            let mut seen_this_call: HashSet<String> = HashSet::new();
            for extracted in &all_entities {
                let id = normalize_name(&extracted.name);
                if !seen_this_call.insert(id.clone()) {
                    return Err(crate::core::error::Error::IntraBatchDuplicate { id });
                }
            }
        } // seen_this_call dropped here — Tier-3 lifetime ends.

        // Deduplicate extracted entities by normalized name.
        // sort + dedup_by because dedup_by only removes consecutive duplicates.
        all_entities.sort_by_key(|e| normalize_name(&e.name));
        all_entities.dedup_by(|a, b| normalize_name(&a.name) == normalize_name(&b.name));

        // 4. Resolve entities against existing graph (namespace-scoped dedup)
        let existing_entities = match group_id {
            Some(gid) => self.graph.list_entities_in_group(gid).await?,
            None => self.graph.list_entities().await?,
        };
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
                    // RISK-003: rql_entities.id is a sole TEXT PK (no composite (id, group_id) PK).
                    // Stub INSERT uses INSERT OR IGNORE — cross-namespace name collision silently
                    // skips stub creation. Single-namespace use only for v0.1.1.
                    // Composite PK migration tracked for v0.2.0 schema audit.
                    //
                    // Strategy: attempt insert_entity_with_group; if the entity already exists
                    // (UNIQUE constraint error), that is fine — a real row is present.
                    let stub_props =
                        serde_json::json!({ "stub": true, "source": "forward_reference" });
                    match self
                        .graph
                        .insert_entity_with_group(&norm_name, "UNKNOWN", stub_props, group_id)
                        .await
                    {
                        Ok(()) => {
                            // Newly inserted stub.
                            name_to_id.insert(norm_name.clone(), norm_name.clone());
                            stub_entities_inserted += 1;
                            metrics::counter!("kremory.ingest.stub_inserted").increment(1);
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
                    // (label="UNKNOWN", properties.stub=true), overwrite it with the
                    // real label and a fresh context snippet. The upsert removes the
                    // stub flag because the new properties map does not carry it.
                    // First-mention-wins policy still applies for non-stubs.
                    let existing_is_stub = existing_entities
                        .iter()
                        .find(|e| e.id == existing_id)
                        .map(|e| {
                            e.label == "UNKNOWN"
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
                        // `.ok()` — promotion is best-effort; failure to promote
                        // leaves the stub row but does not abort the transaction.
                        self.graph
                            .upsert_entity_with_group(
                                &existing_id,
                                &extracted.label,
                                promoted_props,
                                group_id,
                            )
                            .await
                            .ok();
                        tracing::debug!(
                            target: "kremory.ingest.stub",
                            id = %existing_id,
                            label = %extracted.label,
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
                    // New entity — insert (namespace-scoped via group_id).
                    let entity_id = normalize_name(&extracted.name);

                    // Bug B: capture verbatim first-mention snippet (±100 chars around the
                    // entity name in the source text).  First-mention wins: this branch only
                    // runs for genuinely new entity rows.
                    let snippet = extract_context_snippet(text, &extracted.name, 100);
                    let props_with_context = serde_json::json!({
                        "context": snippet,
                        "name": extracted.name.clone(),
                    });

                    if let Err(e) = self
                        .graph
                        .insert_entity_with_group(
                            &entity_id,
                            &extracted.label,
                            props_with_context,
                            group_id,
                        )
                        .await
                    {
                        break 'phases Err(e);
                    }

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
                match self
                    .graph
                    .insert_fact(
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
                    Ok(fact_id) => {
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
                    Err(e) => {
                        eprintln!(
                            "warn: skipping fact ({} -> {} -> {}): {e}",
                            fact.subject, fact.predicate, fact.object
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
        let extractor = NuExtractExtractor::new(Arc::clone(&self.llm));
        let splitter = ExtractionWindowSplitter::new(self.config.extraction_window.clone());
        let chunks = splitter.split(text, &content_type);

        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &self.config.allowed_entity_types,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &known_entities,
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: content_type.clone(),
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
            name_to_id.insert(normalize_name(&entity.label), entity.id.clone());
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

            match self
                .graph
                .insert_fact(
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
                Ok(fact_id) => {
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
                Err(e) => {
                    eprintln!(
                        "warn: deferred: skipping fact ({} -> {} -> {}): {e}",
                        fact.subject, fact.predicate, fact.object
                    );
                }
            }
        }

        histogram!("rql.ingest.deferred_fact_count").record(inserted_count as f64);
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
        Ok(Self::new(
            graph,
            Arc::new(MockChatProvider::null()),
            Arc::new(NullEmbeddingProvider {
                dim: config.embedding_dim.0,
            }),
            config,
        ))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
                "person".to_string(),
                "organization".to_string(),
                "project".to_string(),
                "technology".to_string(),
                "metric".to_string(),
            ])
            .build()
            .unwrap();
        let llm = Arc::new(build_mock_llm(
            r#"[{"name":"Alice","label":"Person"},{"name":"Acme","label":"Organization"}]"#,
            r#"["works_at"]"#,
            r#"[{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.95}]"#,
            "different",
            "[]",
        ));
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        Engine::new(graph, llm, embedder, config)
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
            .ingest("Alice works at Acme", None, None, None)
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
        let extractor = crate::core::extraction::NuExtractExtractor::new(Arc::clone(&rql.llm));
        let result = rql
            .ingest_with(&extractor, "Alice works at Acme", None, None, None)
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
        let extractor = crate::core::extraction::NuExtractExtractor::new(Arc::clone(&rql.llm));
        let result = rql
            .ingest_with(&extractor, "Alice works at Acme", None, None, None)
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
        let extractor = crate::core::extraction::NuExtractExtractor::new(Arc::clone(&rql.llm));
        let result = rql
            .ingest_with(&extractor, "Alice works at Acme", None, None, None)
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
        rql.ingest("Alice works at Acme", None, None, None)
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
    /// text mid-sentence (not sentence-initial, multi-word) and must be caught by
    /// the proper noun scan that runs inside ingest_with().
    #[tokio::test]
    async fn test_ingest_catches_proper_nouns_missed_by_extractor() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.unwrap());
        let config = PipelineConfig::builder().build().unwrap();
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let rql: Engine<MockChatProvider, MockEmbeddingProvider> =
            Engine::new(graph, llm, embedder, config);

        // Extractor deliberately omits "Zenith Dynamics"
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
            )
            .await
            .unwrap();

        // "Zenith Dynamics" should appear in upserted_entities (normalized)
        assert!(
            result
                .upserted_entities
                .iter()
                .any(|e| e.contains("zenith")),
            "proper noun scan should have caught 'Zenith Dynamics'; got: {:?}",
            result.upserted_entities
        );
    }

    /// Story #150: batch with two entities that normalise to the same ID returns
    /// Err(IntraBatchDuplicate) before any DB write occurs.
    #[tokio::test]
    async fn ingest_intra_batch_duplicate_returns_error() {
        use crate::core::error::Error;

        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let engine = Engine::new(graph, llm, embedder, config);

        // FixedExtractor returns two entities that normalise to the same ID.
        // normalize_name("Alice Corp") == normalize_name("alice corp") == "alice_corp"
        // (or similar — what matters is two entries with same normalized name).
        let extractor = FixedExtractor {
            entities: vec![
                ExtractedEntity {
                    name: "Alice Corp".to_string(),
                    label: "Organization".to_string(),
                    properties: serde_json::Value::Null,
                },
                ExtractedEntity {
                    name: "Alice Corp".to_string(), // exact duplicate → same normalized id
                    label: "Organization".to_string(),
                    properties: serde_json::Value::Null,
                },
            ],
        };

        let result = engine
            .ingest_with(&extractor, "Alice Corp is a company.", None, None, None)
            .await;

        match result {
            Err(Error::IntraBatchDuplicate { id }) => {
                assert!(
                    !id.is_empty(),
                    "IntraBatchDuplicate must carry the offending id"
                );
            }
            other => panic!(
                "expected IntraBatchDuplicate, got: {:?}",
                other.map(|_| "<ok>")
            ),
        }
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
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config);

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

        // Verify Bob exists in the graph with stub=true and label=UNKNOWN.
        let bob = graph
            .get_entity("bob")
            .await
            .expect("get_entity OK")
            .expect("Bob must exist as stub");
        assert_eq!(bob.label, "UNKNOWN", "stub entity must have label=UNKNOWN");
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
        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config);

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
            .ingest_with(&extractor_1, "Alice works with Bob.", None, None, None)
            .await
            .expect("ingest 1 OK");
        assert_eq!(r1.stub_entities_inserted, 1, "ingest 1 must create 1 stub");

        // Confirm Bob is a stub before promotion.
        let bob_pre = graph
            .get_entity("bob")
            .await
            .expect("get OK")
            .expect("Bob exists");
        assert_eq!(
            bob_pre.label, "UNKNOWN",
            "Bob must be UNKNOWN before promotion"
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
            )
            .await
            .expect("ingest 2 OK");

        // After promotion, Bob must have real label and no stub flag.
        let bob_post = graph
            .get_entity("bob")
            .await
            .expect("get OK")
            .expect("Bob still exists");
        assert_eq!(
            bob_post.label, "Person",
            "promoted Bob must have label=Person; got={}",
            bob_post.label
        );
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

    /// Story #150: after a batch rejection due to IntraBatchDuplicate, no entities
    /// are written to the DB.
    #[tokio::test]
    async fn ingest_intra_batch_duplicate_no_partial_write() {
        let graph = Arc::new(TemporalGraph::open_in_memory().await.expect("open"));
        let config = PipelineConfig::builder().build().expect("config");
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));

        // Count entities before the failing batch.
        let before_count = graph.list_entities().await.expect("list").len();

        let engine = Engine::new(Arc::clone(&graph), llm, embedder, config);

        let extractor = FixedExtractor {
            entities: vec![
                ExtractedEntity {
                    name: "Dup Entity".to_string(),
                    label: "Entity".to_string(),
                    properties: serde_json::Value::Null,
                },
                ExtractedEntity {
                    name: "Dup Entity".to_string(),
                    label: "Entity".to_string(),
                    properties: serde_json::Value::Null,
                },
            ],
        };

        let _ = engine
            .ingest_with(&extractor, "Dup Entity test.", None, None, None)
            .await;

        // No entity rows should have been written — list count unchanged.
        let after_count = engine.graph.list_entities().await.expect("list").len();
        assert_eq!(
            before_count, after_count,
            "no entities must be written after IntraBatchDuplicate rejection"
        );
    }
}
