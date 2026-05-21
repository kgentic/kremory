use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use metrics::histogram;

use chrono::{DateTime, Utc};

use super::chunker::Chunker;
use super::config::{ContentType, PipelineConfig};
use super::contradiction::TwoPoolDetector;
use super::error::Result;
use super::extraction::NuExtractExtractor;
use super::intelligence::{
    EntityExtractor, EntityResolver, ExtractedEntity, ExtractedFact, ExtractionContext,
    ResolutionResult,
};
use super::provider::{
    ChatProvider, EmbeddingProvider, MockChatProvider, NullEmbeddingProvider, TokenUsage,
};
use super::resolver::{normalize_name, CascadeResolver, UnionFind};
use super::schema::TemporalGraph;
use super::search::SearchFilters;
use super::text_utils;

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
}

/// High-level RQL graph with intelligence pipeline.
/// Wraps TemporalGraph and adds extraction, resolution, and contradiction detection.
pub struct RqlGraph<L: ChatProvider, Emb: EmbeddingProvider> {
    pub(crate) graph: TemporalGraph,
    pub(crate) llm: Arc<L>,
    pub(crate) embedder: Arc<Emb>,
    pub(crate) config: PipelineConfig,
    /// Optional OOV auditor for language-agnostic entity safety net.
    /// When set, runs after each chunk extraction to catch domain terms the LLM missed.
    pub(crate) oov_auditor: Option<text_utils::OovAuditor>,
}

impl<L: ChatProvider, Emb: EmbeddingProvider> RqlGraph<L, Emb> {
    /// Create a new RqlGraph wrapping a TemporalGraph.
    pub fn new(
        graph: TemporalGraph,
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

    /// Access the underlying TemporalGraph for direct queries.
    pub fn graph(&self) -> &TemporalGraph {
        &self.graph
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
        _group_id: Option<&str>,
        content_type: Option<ContentType>,
    ) -> Result<IngestionResult> {
        #[cfg(feature = "ner")]
        {
            use std::sync::OnceLock;
            static GLINER: OnceLock<super::ner::GlinerExtractor> = OnceLock::new();
            if GLINER.get().is_none() {
                let g = super::ner::GlinerExtractor::new().map_err(super::error::RqlError::from)?;
                let _ = GLINER.set(g);
            }
            let extractor = GLINER.get().expect("just initialised");
            return self
                .ingest_with(extractor, text, reference_time, _group_id, content_type)
                .await;
        }
        #[cfg(not(feature = "ner"))]
        {
            let extractor = NuExtractExtractor::new(Arc::clone(&self.llm));
            self.ingest_with(&extractor, text, reference_time, _group_id, content_type)
                .await
        }
    }

    /// Full pipeline with a caller-supplied extractor.
    /// Any type implementing `EntityExtractor` can be used (DefaultExtractor, NuExtractExtractor, etc.).
    pub async fn ingest_with<E: EntityExtractor>(
        &self,
        extractor: &E,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        _group_id: Option<&str>,
        content_type: Option<ContentType>,
    ) -> Result<IngestionResult> {
        let ingest_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);
        let token_usage = TokenUsage::default();

        // 1. Store episode
        let episode_id = self
            .graph
            .insert_episode(text, ref_time, Some("ingest"), None)
            .await?;

        // 2. Chunk
        let chunker = Chunker::new(self.config.chunk.clone());
        let chunks = chunker.split(text, &content_type);

        histogram!("rql.ingest.chunk_count").record(chunks.len() as f64);

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

        // Deduplicate extracted entities by normalized name.
        // sort + dedup_by because dedup_by only removes consecutive duplicates.
        all_entities.sort_by_key(|e| normalize_name(&e.name));
        all_entities.dedup_by(|a, b| normalize_name(&a.name) == normalize_name(&b.name));

        // 4. Resolve entities against existing graph
        let existing_entities = self.graph.list_entities().await?;
        let resolver = CascadeResolver::new(
            Arc::clone(&self.llm),
            self.config.minhash.clone(),
            self.config.entropy.clone(),
        );

        let mut union_find = UnionFind::new();
        let mut upserted_entities: Vec<String> = Vec::new();
        let mut merged_entities: Vec<(String, String)> = Vec::new();

        // Map from extracted entity name → resolved entity ID
        let mut name_to_id: HashMap<String, String> = HashMap::new();

        for extracted in &all_entities {
            let mut resolved_to: Option<String> = None;

            for existing in &existing_entities {
                let result = resolver.resolve(extracted, existing).await?;
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
                existing_id
            } else {
                // New entity — insert
                let entity_id = normalize_name(&extracted.name);
                self.graph
                    .insert_entity(&entity_id, &extracted.label, extracted.properties.clone())
                    .await?;

                // Embed and store embedding
                let embedding = self.embedder.embed(&extracted.name).await?;
                self.graph
                    .set_entity_embedding(&entity_id, &embedding)
                    .await?;

                upserted_entities.push(entity_id.clone());
                entity_id
            };

            name_to_id.insert(normalize_name(&extracted.name), entity_id);
        }

        // 5. Detect contradictions and store facts
        let detector = TwoPoolDetector::new(Arc::clone(&self.llm));
        let mut inserted_fact_ids: Vec<i64> = Vec::new();
        let mut invalidated_fact_ids: Vec<i64> = Vec::new();

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
            let pool_a = self
                .graph
                .get_facts_by_subject_predicate(&subject_id, &fact.predicate)
                .await?;

            // For pool_b, use FTS search on the predicate to find semantically related facts
            let pool_b_hits = self.graph.fts_search_facts(&fact.predicate, 10, &SearchFilters::new()).await?;
            let pool_b: Vec<super::schema::Fact> =
                pool_b_hits.into_iter().map(|h| h.item).collect();

            // Run contradiction detection
            let contradiction_result = detector.detect(fact, &pool_a, &pool_b, &ref_time).await?;

            // Invalidate contradicted facts
            for fact_id in &contradiction_result.contradictions {
                self.graph
                    .invalidate_fact_with_reason(*fact_id, Utc::now(), ref_time)
                    .await?;
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
                    let fact_text = format!("{} {} {}", fact.subject, fact.predicate, fact.object);
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

            // Create episodic edges (MENTIONS) — only for entities that exist in the graph.
            // Facts may reference entities not in the extraction results (e.g., organisations,
            // projects mentioned as objects but not extracted as entities).
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
        }

        histogram!("rql.ingest.total_ms").record(ingest_start.elapsed().as_secs_f64() * 1000.0);
        histogram!("rql.ingest.entity_count").record(upserted_entities.len() as f64);
        histogram!("rql.ingest.fact_count").record(inserted_fact_ids.len() as f64);
        histogram!("rql.ingest.merge_count").record(merged_entities.len() as f64);
        histogram!("rql.ingest.contradiction_count").record(invalidated_fact_ids.len() as f64);

        Ok(IngestionResult {
            episode_id,
            upserted_entities,
            inserted_fact_ids,
            invalidated_fact_ids,
            merged_entities,
            token_usage,
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
    pub async fn ingest_deferred(
        &self,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        _group_id: Option<&str>,
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
        let chunker = Chunker::new(self.config.chunk.clone());
        let chunks = chunker.split(text, &content_type);

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
            return Ok(0);
        }

        // Build a name → entity_id map by resolving against the existing graph.
        let existing_entities = self.graph.list_entities().await?;
        let mut name_to_id: HashMap<String, String> = HashMap::new();
        for entity in &existing_entities {
            name_to_id.insert(normalize_name(&entity.label), entity.id.clone());
            name_to_id.insert(normalize_name(&entity.id), entity.id.clone());
        }
        // Also seed the map with the NER entity names directly so facts that
        // reference them by their original surface form are resolved correctly.
        for name in ner_entity_names {
            let norm = normalize_name(name);
            name_to_id.entry(norm).or_insert_with(|| normalize_name(name));
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

            let pool_b_hits = self.graph.fts_search_facts(&fact.predicate, 10, &SearchFilters::new()).await?;
            let pool_b: Vec<super::schema::Fact> =
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
        Ok(inserted_count)
    }
}

/// Convenience type alias for tests and simple usage (no LLM/embedding).
pub type SimpleGraph = RqlGraph<MockChatProvider, NullEmbeddingProvider>;

impl SimpleGraph {
    /// Open an in-memory graph with null providers and default config.
    pub async fn open_in_memory_simple() -> Result<Self> {
        let graph = TemporalGraph::open_in_memory().await?;
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
    use super::intelligence::{ExtractionContext, ExtractionResult};
    use super::provider::{MockChatProvider, MockEmbeddingProvider};
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
        ) -> super::error::Result<ExtractionResult> {
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

    async fn make_rql_with_mock() -> RqlGraph<MockChatProvider, MockEmbeddingProvider> {
        let graph = TemporalGraph::open_in_memory().await.unwrap();
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
        RqlGraph::new(graph, llm, embedder, config)
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
        let rql = make_rql_with_mock().await;
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
        let rql = make_rql_with_mock().await;
        let extractor = super::extraction::NuExtractExtractor::new(Arc::clone(&rql.llm));
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
        let rql = make_rql_with_mock().await;
        let extractor = super::extraction::NuExtractExtractor::new(Arc::clone(&rql.llm));
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
        let rql = make_rql_with_mock().await;
        let extractor = super::extraction::NuExtractExtractor::new(Arc::clone(&rql.llm));
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
        let rql = make_rql_with_mock().await;
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
        let graph = TemporalGraph::open_in_memory().await.unwrap();
        let config = PipelineConfig::builder().build().unwrap();
        let llm = Arc::new(MockChatProvider::null());
        let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0));
        let rql: RqlGraph<MockChatProvider, MockEmbeddingProvider> =
            RqlGraph::new(graph, llm, embedder, config);

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
}
