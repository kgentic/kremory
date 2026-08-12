//! L5 — TD-012 Placeholder-Label Canary (env-gated CI gate)
//!
//! A single focused regression test for the TD-012 failure mode: entities
//! extracted with `label="Entity"` (placeholder) instead of a canonical type
//! such as "Person" or "Organisation".
//!
//! This test uses a kremory-eval fixture (mock_interview.txt) — a realistic
//! multi-speaker transcript containing Person/Organisation/Location entities.
//! It runs the IntegerIdLlmExtractor with a real Ollama LLM and asserts that:
//!
//!   1. At least 3 entities are extracted (non-trivial extraction).
//!   2. No entity carries the placeholder label "Entity" or "UNKNOWN".
//!   3. At least one entity is typed "Person".
//!
//! Assertion 2 is the TD-012 canary. If it fires, Phase T (StructuredCallBuilder
//! schema enforcement) has regressed and the FormatSchema arm is no longer
//! constraining Ollama's output.
//!
//! # How to enable in CI
//!
//! Add this step to a nightly job:
//!
//! ```yaml
//! - name: TD-012 canary
//!   env:
//!     OLLAMA_HOST: http://localhost:11434
//!     OLLAMA_CHAT_MODEL: qwen2.5:14b
//!   run: |
//!     cargo test -p kremory --features llm-integration \
//!       --test it td_012_canary:: -- --ignored --nocapture
//! ```
//!
//! The test self-skips if OLLAMA_HOST is not set, so it never breaks
//! environments without Ollama (e.g. developer machines without a local GPU).
//!
//! # How to run manually
//!
//! ```sh
//! OLLAMA_HOST=http://localhost:11434 \
//! OLLAMA_CHAT_MODEL=qwen2.5:14b \
//! cargo test -p kremory --features llm-integration \
//!   --test it td_012_canary:: -- --ignored --nocapture
//! ```

#![cfg(feature = "llm-integration")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::embedding::EmbeddingBuilder;
use kremory::core::config::PipelineConfig;
use kremory::core::extraction::{is_canonical_entity_type, IntegerIdLlmExtractor};
use kremory::core::ingest::{Engine, SourceParams};
use kremory::core::schema::TemporalGraph;

use crate::helpers::ollama_adapter::OllamaEmbedderAdapter;

// ── Canary ────────────────────────────────────────────────────────────────────

/// TD-012 canary — placeholder-label regression gate.
///
/// Uses the `mock_interview.txt` kremory-eval fixture (a realistic interview
/// transcript) as a close-to-production input.  If any entity comes back with
/// `label="Entity"` or `label="UNKNOWN"`, the StructuredCallBuilder schema
/// enforcement has failed.
///
/// Self-skip: if `OLLAMA_HOST` is not set, the test is skipped cleanly.
///
/// Enable with:
/// ```sh
/// OLLAMA_HOST=http://localhost:11434 OLLAMA_CHAT_MODEL=qwen2.5:14b \
/// cargo test -p kremory --features llm-integration --test it td_012_canary:: -- --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Ollama at $OLLAMA_HOST with $OLLAMA_CHAT_MODEL; enable with --ignored"]
async fn td_012_no_placeholder_labels_in_fixture_extraction() {
    // Self-skip when OLLAMA_HOST is not configured.
    let base_url = match std::env::var("OLLAMA_HOST").or_else(|_| std::env::var("OLLAMA_BASE_URL"))
    {
        Ok(url) => url,
        Err(_) => {
            eprintln!(
                "SKIP td_012_canary: OLLAMA_HOST not set — \
                 set OLLAMA_HOST=http://localhost:11434 to enable"
            );
            return;
        }
    };

    let chat_model =
        crate::helpers::chat_model::chat_model_or("qwen2.5:14b");
    eprintln!("td_012_canary: OLLAMA_HOST={base_url} OLLAMA_CHAT_MODEL={chat_model}");

    // Load the kremory-eval mock_interview fixture.
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kremory-eval/fixtures/mock_interview.txt"
    );
    let fixture_text = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {fixture_path}: {e}"));

    // Build Engine with real Ollama.
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(chat_model)
        .timeout_seconds(90)
        .build()
        .expect("Ollama LLM builder");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder");

    let emb = OllamaEmbedderAdapter(raw_emb);

    let dir = tempfile::tempdir().expect("tempdir");
    let graph = Arc::new(
        TemporalGraph::open(dir.path().join("td012.db").to_str().expect("valid UTF-8"))
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig default");

    let extractor = IntegerIdLlmExtractor::new(Arc::clone(&llm));
    let engine = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: Arc::clone(&graph),
        llm,
        embedder: Arc::new(emb),
        config,
        model: None,
    });

    // Run ingest on the fixture.
    let result = engine
        .ingest_with(
            &extractor,
            kremory::core::ingest::IngestWithParams {
                text: &fixture_text,
                reference_time: None,
                declared_reference_time: None,
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
        )
        .await
        .expect("ingest of mock_interview fixture must succeed");

    eprintln!(
        "td_012_canary: extracted {} entities",
        result.upserted_entities.len()
    );

    // Gate 1: at least 3 entities extracted (non-trivial extraction).
    assert!(
        result.upserted_entities.len() >= 3,
        "expected at least 3 entities from mock_interview.txt, got {} — \
         extraction may have completely failed",
        result.upserted_entities.len()
    );

    // Gate 2: zero placeholder labels (TD-012 canary).
    let mut placeholder_entities: Vec<(String, String)> = Vec::new();
    let mut canonical_count = 0usize;
    let mut person_found = false;

    for entity_id in &result.upserted_entities {
        let entity = graph
            .get_entity(entity_id)
            .await
            .expect("get_entity must not error")
            .unwrap_or_else(|| panic!("entity '{entity_id}' must exist after ingest"));

        eprintln!("  entity: id='{}' label='{}'", entity.id, entity.label);

        if is_canonical_entity_type(&entity.label) {
            canonical_count += 1;
            if entity.label == "Person" {
                person_found = true;
            }
        } else {
            placeholder_entities.push((entity.id.clone(), entity.label.clone()));
        }
    }

    // TD-012 canary assertion: zero placeholder labels.
    assert!(
        placeholder_entities.is_empty(),
        "TD-012 REGRESSION: {} entities have placeholder labels: {placeholder_entities:?}\n\
         Expected all labels from ENTITY_TYPE_ALLOWLIST (e.g. Person/Organisation/Location).\n\
         This means StructuredCallBuilder's FormatSchema arm is NOT constraining Ollama output.\n\
         Check: (1) capability_of() returns FormatSchema for the configured model,\n\
               (2) SCHEMA_ENTITY_LIST is wired into the extraction call-site.",
        placeholder_entities.len(),
    );

    eprintln!(
        "td_012_canary: {} canonical labels, 0 placeholders — PASS",
        canonical_count
    );

    // Gate 3: at least one Person entity (mock_interview has multiple named interviewees).
    assert!(
        person_found,
        "expected at least one Person entity from mock_interview.txt \
         (the fixture contains named interviewees) — got {} canonical entities but none \
         with label 'Person'. Check extraction prompt + schema enforcement.",
        canonical_count,
    );
}
