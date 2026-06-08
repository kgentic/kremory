//! L3 — Live-LLM canonical-label assertion (TD-012 regression gate)
//!
//! Verifies that after the structured-output migration (Phase T), real LLM
//! extraction produces canonical entity labels (e.g. "Person", "Organisation")
//! rather than the placeholder "Entity" label that TD-012 surfaced.
//!
//! Uses the Engine layer directly (not the Memory facade) so we can inspect
//! `entity.label` after ingest via `graph.get_entity(id)`.
//!
//! # How to run
//!
//! Requires a live Ollama instance:
//!
//! ```sh
//! OLLAMA_HOST=http://localhost:11434 \
//! OLLAMA_CHAT_MODEL=qwen2.5:14b \
//! cargo test -p kremory --features llm-integration --test label_canonical_assertion -- --ignored --nocapture
//! ```
//!
//! OLLAMA_HOST defaults to `http://localhost:11434`.
//! OLLAMA_CHAT_MODEL defaults to `qwen2.5:14b` (better structured-output compliance
//! than llama3.2:3b for NER).
//!
//! All tests carry `#[ignore]` — they are NOT executed in the standard PR gate.
//! Enable manually pre-release or in a nightly CI job with Ollama access.
//! Consistent with the `llm_integration.rs` gate pattern.

#![cfg(feature = "llm-integration")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::embedding::EmbeddingBuilder;
use kremory::core::config::PipelineConfig;
use kremory::core::extraction::{is_canonical_entity_type, DefaultExtractor};
use kremory::core::ingest::{Engine, SourceParams};
use kremory::core::schema::TemporalGraph;

mod helpers;
use helpers::ollama_adapter::OllamaEmbedderAdapter;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ollama_base_url() -> String {
    std::env::var("OLLAMA_HOST")
        .or_else(|_| std::env::var("OLLAMA_BASE_URL"))
        .unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_chat_model() -> String {
    // Default to qwen2.5:14b — better structured-output compliance than llama3.2:3b.
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string())
}

async fn build_engine(
    dir: &tempfile::TempDir,
) -> (
    Engine<Ollama, OllamaEmbedderAdapter>,
    Arc<Ollama>,
    Arc<TemporalGraph>,
) {
    let base_url = ollama_base_url();
    let chat_model = ollama_chat_model();

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(chat_model)
        .timeout_seconds(60)
        .build()
        .expect("Ollama LLM builder");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder");

    let emb = OllamaEmbedderAdapter(raw_emb);

    let db_path = dir.path().join("l3.db");
    let graph = Arc::new(
        TemporalGraph::open(db_path.to_str().expect("valid UTF-8"))
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig default");

    let engine = Engine::new(Arc::clone(&graph), Arc::clone(&llm), Arc::new(emb), config)
        .expect("Engine::new should succeed in tests");
    (engine, llm, graph)
}

// ── L3.1 — Canonical-label regression on simple entity text ──────────────────

/// L3.1 — Phase 4 smoke regression (TD-012 canary at the Engine layer).
///
/// Ingests "Alice works at OpenAI in San Francisco." and asserts that every
/// extracted entity carries a canonical label (not "Entity", "UNKNOWN", "").
///
/// This is the exact failure mode TD-012 exposed: 57 entities extracted, all
/// labelled "Entity" because the pre-Phase-4 code did not pass structured output
/// schemas to Ollama.  After Phase 4 (StructuredCallBuilder), the FormatSchema
/// arm enforces the schema grammar at the Ollama level.
///
/// To run:
/// ```sh
/// OLLAMA_HOST=http://localhost:11434 OLLAMA_CHAT_MODEL=qwen2.5:14b \
/// cargo test -p kremory --features llm-integration --test label_canonical_assertion \
///   -- label_field_populated_with_canonical_type --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Ollama at $OLLAMA_HOST with $OLLAMA_CHAT_MODEL (default: qwen2.5:14b); enable with --ignored"]
async fn label_field_populated_with_canonical_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, llm, graph) = build_engine(&dir).await;

    let extractor = DefaultExtractor::new(llm);
    let result = engine
        .ingest_with(
            &extractor,
            "Alice works at OpenAI in San Francisco.",
            None,
            None,
            None,
            SourceParams::default(),
        )
        .await
        .expect("ingest must succeed");

    assert!(
        !result.upserted_entities.is_empty(),
        "expected at least one entity from 'Alice works at OpenAI in San Francisco.'"
    );

    // Check every extracted entity label is canonical.
    for entity_id in &result.upserted_entities {
        let entity = graph
            .get_entity(entity_id)
            .await
            .expect("graph.get_entity must succeed")
            .unwrap_or_else(|| panic!("entity '{entity_id}' must exist in graph after ingest"));

        assert!(
            is_canonical_entity_type(&entity.label),
            "entity '{}' has non-canonical label '{}' — TD-012 regression \
             (StructuredCallBuilder schema enforcement failed for Ollama FormatSchema arm)",
            entity.id,
            entity.label,
        );
    }

    // Specifically: "alice" entity should be typed Person.
    let alice_id = result
        .upserted_entities
        .iter()
        .find(|id| id.to_lowercase().contains("alice"))
        .expect("Alice entity must be present after ingest");
    let alice = graph
        .get_entity(alice_id)
        .await
        .expect("get_entity for alice")
        .expect("alice entity must exist");
    assert_eq!(
        alice.label, "Person",
        "Alice must be typed 'Person' — got '{}' (TD-012 regression)",
        alice.label
    );
}

// ── L3.2 — Organisation label on multi-entity text ───────────────────────────

/// L3.2 — Organisation and Location type assertion.
///
/// Ingests a multi-entity sentence and asserts that all entities have canonical
/// labels.  Specifically checks that at least one Organisation and one Location
/// type appears, which TD-012 would have mislabelled as "Entity".
///
/// To run:
/// ```sh
/// OLLAMA_HOST=http://localhost:11434 OLLAMA_CHAT_MODEL=qwen2.5:14b \
/// cargo test -p kremory --features llm-integration --test label_canonical_assertion \
///   -- all_entity_labels_canonical_on_org_location_text --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Ollama at $OLLAMA_HOST with $OLLAMA_CHAT_MODEL (default: qwen2.5:14b); enable with --ignored"]
async fn all_entity_labels_canonical_on_org_location_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, llm, graph) = build_engine(&dir).await;

    let extractor = DefaultExtractor::new(llm);
    let result = engine
        .ingest_with(
            &extractor,
            "Priya works at Google DeepMind in London and collaborates with teams in Singapore.",
            None,
            None,
            None,
            SourceParams::default(),
        )
        .await
        .expect("ingest must succeed");

    assert!(
        !result.upserted_entities.is_empty(),
        "expected entities from a multi-entity text"
    );

    // All labels must be canonical — zero tolerance for placeholder labels.
    let mut non_canonical: Vec<(String, String)> = Vec::new();
    for entity_id in &result.upserted_entities {
        if let Some(entity) = graph
            .get_entity(entity_id)
            .await
            .expect("get_entity must not error")
        {
            if !is_canonical_entity_type(&entity.label) {
                non_canonical.push((entity.id.clone(), entity.label.clone()));
            }
        }
    }

    assert!(
        non_canonical.is_empty(),
        "non-canonical entity labels detected (TD-012 regression): {non_canonical:?}\n\
         All labels must be from ENTITY_TYPE_ALLOWLIST — placeholder 'Entity'/'UNKNOWN' \
         indicate StructuredCallBuilder FormatSchema enforcement failed."
    );
}
