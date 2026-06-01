//! L4 — Label Precision Benchmark (TD-012 quality gate)
//!
//! Asserts that entity extraction achieves label_precision >= 0.75 against
//! the kremory-eval ground truth (mock_interview.txt).
//!
//! Label precision = (entities whose name AND label both match ground truth) /
//!                   (entities expected in ground truth)
//!
//! This gate catches the TD-012 failure mode structurally: even if an LLM
//! correctly identifies entity names, if it outputs placeholder labels
//! ("Entity", "UNKNOWN") for most entities, precision drops to near 0.0 and
//! this benchmark fails.
//!
//! # How to run
//!
//! ```sh
//! OLLAMA_HOST=http://localhost:11434 \
//! OLLAMA_CHAT_MODEL=qwen2.5:14b \
//! cargo test -p kremory --features llm-integration \
//!   --test label_precision_benchmark -- --ignored --nocapture
//! ```
//!
//! All tests carry `#[ignore]` — not in the standard PR gate.
//! Designed to run in nightly CI alongside the L3/L5 canary tests.

#![cfg(feature = "llm-integration")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::embedding::EmbeddingBuilder;
use kremory::core::config::PipelineConfig;
use kremory::core::extraction::{is_canonical_entity_type, DefaultExtractor};
use kremory::core::ingest::{Engine, SourceParams};
use kremory::core::schema::TemporalGraph;
use serde::Deserialize;

mod helpers;
use helpers::ollama_adapter::OllamaEmbedderAdapter;

// ── Ground truth types ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GroundTruthEntity {
    name: String,
    label: String,
}

#[derive(Debug, Deserialize)]
struct DomainGroundTruth {
    entities: Vec<GroundTruthEntity>,
    #[allow(dead_code)]
    #[serde(default)]
    min_relationships: usize,
}

fn load_ground_truth() -> HashMap<String, DomainGroundTruth> {
    let gt_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kremory-eval/fixtures/ground_truth.json"
    );
    let raw = std::fs::read_to_string(gt_path)
        .unwrap_or_else(|e| panic!("failed to read ground_truth.json: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("failed to parse ground_truth.json: {e}"))
}

// ── Label precision metric ────────────────────────────────────────────────────

/// Compute label precision: for each expected entity (name+label pair), check
/// if any extracted entity matches BOTH name (fuzzy) AND label (exact).
///
/// Returns (precision, matched, total_expected).
///
/// `precision = matched / total_expected`
///
/// Using fuzzy name matching (case-insensitive substring) consistent with
/// the existing kremory-eval matching strategy.
fn label_precision(
    extracted: &[(String, String)], // (name, label)
    expected: &[(String, String)],  // (name, label) from ground truth
) -> (f64, usize, usize) {
    let matched = expected
        .iter()
        .filter(|(exp_name, exp_label)| {
            let exp_lower = exp_name.to_lowercase();
            let exp_label_lower = exp_label.to_lowercase();
            extracted.iter().any(|(ext_name, ext_label)| {
                let ext_lower = ext_name.to_lowercase();
                let ext_label_lower = ext_label.to_lowercase();
                // Fuzzy name match: substring in either direction.
                let name_match = ext_lower.contains(exp_lower.as_str())
                    || exp_lower.contains(ext_lower.as_str());
                // Label match: exact (case-insensitive).
                let label_match = ext_label_lower == exp_label_lower;
                name_match && label_match
            })
        })
        .count();
    let total = expected.len();
    let precision = if total == 0 {
        1.0
    } else {
        matched as f64 / total as f64
    };
    (precision, matched, total)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ollama_base_url() -> Option<String> {
    std::env::var("OLLAMA_HOST")
        .or_else(|_| std::env::var("OLLAMA_BASE_URL"))
        .ok()
}

fn ollama_chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string())
}

// ── L4.1 — label_precision >= 0.75 on mock_interview fixture ─────────────────

/// L4.1 — Label precision gate: >= 75% of ground-truth entities must have
/// both name AND label correctly extracted from mock_interview.txt.
///
/// Pre-Phase-4 (TD-012 broken state): label_precision ≈ 0.0 because all
/// entities were labelled "Entity".
/// Post-Phase-4 (target): label_precision >= 0.75.
///
/// To run:
/// ```sh
/// OLLAMA_HOST=http://localhost:11434 OLLAMA_CHAT_MODEL=qwen2.5:14b \
/// cargo test -p kremory --features llm-integration --test label_precision_benchmark \
///   -- label_precision_gte_0_75_on_mock_interview --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Ollama at $OLLAMA_HOST with $OLLAMA_CHAT_MODEL (default: qwen2.5:14b); enable with --ignored"]
async fn label_precision_gte_0_75_on_mock_interview() {
    let base_url = match ollama_base_url() {
        Some(url) => url,
        None => {
            eprintln!("SKIP label_precision_benchmark: OLLAMA_HOST not set");
            return;
        }
    };

    let chat_model = ollama_chat_model();
    eprintln!("label_precision_benchmark: model={chat_model}");

    // Load fixture + ground truth.
    let ground_truth = load_ground_truth();
    let domain = ground_truth
        .get("mock_interview")
        .expect("mock_interview must be in ground_truth.json");

    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kremory-eval/fixtures/mock_interview.txt"
    );
    let fixture_text = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|e| panic!("failed to read fixture: {e}"));

    // Build engine.
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
        TemporalGraph::open(dir.path().join("l4.db").to_str().expect("valid UTF-8"))
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig default");

    let extractor = DefaultExtractor::new(Arc::clone(&llm));
    let engine = Engine::new(Arc::clone(&graph), llm, Arc::new(emb), config);

    // Run extraction.
    let result = engine
        .ingest_with(
            &extractor,
            &fixture_text,
            None,
            None,
            None,
            SourceParams::default(),
        )
        .await
        .expect("ingest of mock_interview fixture must succeed");

    // Collect extracted (name, label) pairs.
    // Use the display name stored in entity.properties["name"] rather than
    // entity.id (the normalised slug).  Slug-vs-display-name comparison
    // systematically under-scores precision for multi-word names where the
    // slug normalisation differs from the ground-truth casing/format.
    let mut extracted: Vec<(String, String)> = Vec::new();
    for entity_id in &result.upserted_entities {
        if let Some(entity) = graph.get_entity(entity_id).await.expect("get_entity") {
            // Prefer the verbatim display name stored at ingest time; fall back to id.
            let display_name = entity
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(&entity.id)
                .to_string();
            eprintln!(
                "  extracted: '{}' (id='{}') label='{}'",
                display_name, entity.id, entity.label
            );
            extracted.push((display_name, entity.label.clone()));
        }
    }

    // Build expected (name, label) pairs from ground truth.
    let expected: Vec<(String, String)> = domain
        .entities
        .iter()
        .map(|e| (e.name.clone(), e.label.clone()))
        .collect();

    // Compute label precision.
    let (precision, matched, total) = label_precision(&extracted, &expected);

    eprintln!(
        "label_precision: {:.1}% ({}/{} ground-truth name+label pairs matched)",
        precision * 100.0,
        matched,
        total
    );

    // Secondary metric: verify no placeholder labels in the extracted set.
    let placeholder_count = extracted
        .iter()
        .filter(|(_, label)| !is_canonical_entity_type(label))
        .count();
    eprintln!(
        "  {} non-canonical labels (TD-012 indicator)",
        placeholder_count
    );
    if placeholder_count > 0 {
        let placeholders: Vec<_> = extracted
            .iter()
            .filter(|(_, label)| !is_canonical_entity_type(label))
            .collect();
        eprintln!("  non-canonical entities: {placeholders:?}");
    }

    // Primary gate: label precision >= 0.75.
    assert!(
        precision >= 0.75,
        "label_precision {:.1}% ({}/{}) is below the 75% threshold.\n\
         Pre-Phase-4 (TD-012 broken state) produces ≈ 0%. Post-Phase-4 target >= 75%.\n\
         {} entities had non-canonical labels (placeholder 'Entity'/'UNKNOWN').\n\
         Check: StructuredCallBuilder FormatSchema arm is active for model '{}'.",
        precision * 100.0,
        matched,
        total,
        placeholder_count,
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string()),
    );
}

// ── Unit tests for the label_precision metric ────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::label_precision;

    #[test]
    fn label_precision_perfect_match() {
        let extracted = vec![
            ("Alice".into(), "Person".into()),
            ("Acme".into(), "Organisation".into()),
        ];
        let expected = vec![
            ("Alice".into(), "Person".into()),
            ("Acme".into(), "Organisation".into()),
        ];
        let (p, matched, total) = label_precision(&extracted, &expected);
        assert_eq!(matched, 2);
        assert_eq!(total, 2);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn label_precision_wrong_label_counts_as_miss() {
        // Name matches but label is "Entity" (placeholder) — should NOT count as a match.
        let extracted = vec![("Alice".into(), "Entity".into())];
        let expected = vec![("Alice".into(), "Person".into())];
        let (p, matched, total) = label_precision(&extracted, &expected);
        assert_eq!(matched, 0, "wrong label must not count as a match");
        assert_eq!(total, 1);
        assert!((p - 0.0).abs() < 1e-9);
    }

    #[test]
    fn label_precision_fuzzy_name_match_with_correct_label() {
        // "Amazon" extracted matches "Amazon Robotics" expected (substring), same label.
        let extracted = vec![("Amazon".into(), "Organisation".into())];
        let expected = vec![("Amazon Robotics".into(), "Organisation".into())];
        let (p, matched, total) = label_precision(&extracted, &expected);
        assert_eq!(matched, 1, "fuzzy name + exact label must match");
        assert_eq!(total, 1);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn label_precision_zero_expected_returns_one() {
        let extracted = vec![("Alice".into(), "Person".into())];
        let expected: Vec<(String, String)> = vec![];
        let (p, matched, total) = label_precision(&extracted, &expected);
        assert_eq!(matched, 0);
        assert_eq!(total, 0);
        assert!((p - 1.0).abs() < 1e-9, "zero expected = perfect precision");
    }
}
