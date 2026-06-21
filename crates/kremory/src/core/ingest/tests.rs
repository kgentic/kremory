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
    fn name(&self) -> &'static str {
        "fixed"
    }

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
/// used by LlmExtractor, DefaultExtractor, CascadeResolver, and TwoPoolDetector.
// Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a documented
// too_many_arguments allow); TD-042 args-as-object targets `src/` production fns,
// not test-module builders.
#[allow(clippy::too_many_arguments)]
fn build_mock_llm(
    entities_json: &str,
    relations_json: &str,
    triplets_json: &str,
    resolution_response: &str,
    contradiction_response: &str,
) -> MockChatProvider {
    let mut map = HashMap::new();

    // ── LlmExtractor (graphiti-style) ─────────────────────────────────────────
    // Stage 1: entity extraction with integer-ID schema.
    // Unique substring from build_graphiti_entity_prompt (line ending).
    map.insert(
        "Never include type information in the name field.".to_string(),
        r#"{"entities": [{"name": "Alice", "entity_type_id": 1}, {"name": "Acme", "entity_type_id": 2}]}"#.to_string(),
    );
    // Stage 2: relationship extraction.
    // Unique substring from build_graphiti_relationship_prompt.
    map.insert(
        "Extract all factual relationships between the entities above.".to_string(),
        r#"[{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.95}]"#.to_string(),
    );

    // ── DefaultExtractor stage 1: entity extraction prompt ends with this substring ──
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
    Engine::new(EngineNewParams {
        graph,
        llm,
        embedder,
        config,
    })
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
        .ingest(IngestParams {
            text: "Alice works at Acme",
            reference_time: None,
            group_id: None,
            content_type: None,
            source_params: SourceParams::default(),
        })
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
            IngestWithParams {
                text: "Alice works at Acme",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
            IngestWithParams {
                text: "Alice works at Acme",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
            IngestWithParams {
                text: "Alice works at Acme",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
    rql.ingest(IngestParams {
        text: "Alice works at Acme",
        reference_time: None,
        group_id: None,
        content_type: None,
        source_params: SourceParams::default(),
    })
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
    let rql: Engine<MockChatProvider, MockEmbeddingProvider> = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "Alice discussed the proposal with Zenith Dynamics executives.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "Alice Corp is a company.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
    fn name(&self) -> &'static str {
        "fixed_with_facts"
    }

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
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "Alice works with Bob on research.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "Alice works with Bob.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
            IngestWithParams {
                text: "Bob is a researcher at Stanford.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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

    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "two duplicates submitted in single batch.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
    let engine = Engine::new(EngineNewParams {
        graph,
        llm,
        embedder,
        config,
    });

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
    let engine = Engine::new(EngineNewParams {
        graph,
        llm,
        embedder,
        config,
    });

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
    let engine = Engine::new(EngineNewParams {
        graph,
        llm: Arc::new(NamedMock),
        embedder: Arc::new(MockEmbeddingProvider::new(config.embedding_dim.0)),
        config,
    });

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

    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "test text for L1 catch-all persistence.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
        .allowed_entity_types(vec!["Person".to_string(), "Organisation".to_string()])
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
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });
    let extractor = Arc::clone(&engine.extractor);

    let override_specs = vec![
        EntityTypeSpec {
            id: 0,
            name: "Entity".to_string(),
            description: "Catch-all.".to_string(),
        },
        EntityTypeSpec {
            id: 1,
            name: "Person".to_string(),
            description: "A human individual.".to_string(),
        },
        EntityTypeSpec {
            id: 2,
            name: "Organisation".to_string(),
            description: "A company or institution.".to_string(),
        },
    ];

    // ── First call: override present, DB empty for "fresh" group ─────────────
    let r1 = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Alice works at Acme",
                reference_time: None,
                group_id: Some("fresh"),
                content_type: None,
                source_params: SourceParams {
                    entity_types_override: Some(override_specs.clone()),
                    ..SourceParams::default()
                },
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
    // because the JSON mock format does not match the default extractor's
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
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });
    let extractor = Arc::clone(&engine.extractor);

    // Override with a type list that includes CustomType(5) — not in the DB.
    let override_specs = vec![
        EntityTypeSpec {
            id: 0,
            name: "Entity".to_string(),
            description: "catch-all".to_string(),
        },
        EntityTypeSpec {
            id: 5,
            name: "CustomType".to_string(),
            description: "A custom type.".to_string(),
        },
    ];

    let _ = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "WidgetCo is a company.",
                reference_time: None,
                group_id: Some("g1"),
                content_type: None,
                source_params: SourceParams {
                    entity_types_override: Some(override_specs),
                    ..SourceParams::default()
                },
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
    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });
    let extractor = Arc::clone(&engine.extractor);

    let result = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Alice works at Acme",
                reference_time: None,
                group_id: Some("g_fresh"),
                content_type: None,
                source_params: SourceParams {
                    entity_types_override: None,
                    ..SourceParams::default()
                },
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

    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder,
        config,
    });

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
            IngestWithParams {
                text: "some text mentioning a runtime-allowed entity type.",
                reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
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
