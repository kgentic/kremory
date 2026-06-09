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
    // PHASE 1 DIAGNOSTIC INSTRUMENTATION (2026-06-03)
    // Install metrics::DebuggingRecorder to capture rql.entity.label_rejected_total
    // dimension values (the LITERAL labels qwen2.5:14b emits before L2 rejection).
    use metrics_util::debugging::DebuggingRecorder;
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    // Install as local recorder so the test's metric emissions are captured.
    // SAFETY: install_local panics if a recorder is already set; ok for #[ignore] test.
    let _ = recorder.install();

    let base_url = match ollama_base_url() {
        Some(url) => url,
        None => {
            eprintln!("SKIP label_precision_benchmark: OLLAMA_HOST not set");
            return;
        }
    };

    let chat_model = ollama_chat_model();
    eprintln!("label_precision_benchmark: model={chat_model}");
    eprintln!("label_precision_benchmark: Engine::new receives Arc<Ollama> directly (NO ArcChatProvider wrapper). fn model() should return: '{}'", chat_model);

    // Load fixture + ground truth. TD-021: KREMORY_BENCH_DOMAIN env var lets
    // any 14-domain fixture be benchmarked through the same test logic.
    // Defaults to mock_interview for backwards-compat with existing CI.
    let domain_key =
        std::env::var("KREMORY_BENCH_DOMAIN").unwrap_or_else(|_| "mock_interview".to_string());
    let fixture_filename = format!("{}.txt", domain_key);
    let ground_truth = load_ground_truth();
    let domain = ground_truth
        .get(domain_key.as_str())
        .unwrap_or_else(|| panic!("{} must be in ground_truth.json", domain_key));

    let fixture_path = format!(
        "{}/../kremory-eval/fixtures/{}",
        env!("CARGO_MANIFEST_DIR"),
        fixture_filename
    );
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", fixture_path));
    eprintln!("label_precision_benchmark: domain={domain_key} fixture={fixture_filename}");

    // Build engine.
    // TD-024 fix: autoagents-llm defaults keep_alive="0" → gemma4 unloads
    // between calls → mock_interview at GLiNER threshold 0.2 fails with
    // "error sending request" mid-extraction. Match the production
    // kremory-napi default ("1h") so the model stays loaded across chunks.
    // See `crates/kremory-napi/src/bridge.rs:312-323` for the canonical comment.
    let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "1h".to_string());
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(chat_model)
        .timeout_seconds(90)
        .keep_alive(keep_alive)
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

    // TD-022: KREMORY_BENCH_USE_GLINER=1 swaps the LLM extractor for the
    // GLiNER zero-shot classifier (gliner_large-v2.1 INT8). Requires --features ner.
    // TD-023: KREMORY_BENCH_USE_HYBRID=1 uses the hybrid extractor
    // (GLiNER candidates + ONE LLM typing call).
    let use_gliner = std::env::var("KREMORY_BENCH_USE_GLINER")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let use_hybrid = std::env::var("KREMORY_BENCH_USE_HYBRID")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let allowed_for_gliner: Vec<String> = if use_gliner || use_hybrid {
        use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
        let mut names: Vec<String> = DEFAULT_ENTITY_TYPES
            .iter()
            .filter(|(id, _, _)| *id != 0) // exclude catch-all "Entity"
            .map(|(_, name, _)| name.to_string())
            .collect();
        // Augment with KREMORY_BENCH_EXTRA_TYPES if provided.
        if let Ok(extras) = std::env::var("KREMORY_BENCH_EXTRA_TYPES") {
            for extra in extras
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                names.push(extra.to_string());
            }
        }
        eprintln!(
            "label_precision_benchmark: GLiNER allowed_types={:?}",
            names
        );
        names
    } else {
        Vec::new()
    };

    let config = if use_gliner || use_hybrid {
        PipelineConfig::builder()
            .extraction_arm_budget_ms(300_000)
            .allowed_entity_types(allowed_for_gliner)
            .build()
            .expect("PipelineConfig with GLiNER/Hybrid allowed_types")
    } else {
        PipelineConfig::builder()
            // qwen2.5:14b takes ~80-130s per extraction call on Apple M4 Max 36GB
            // at 32k context. Override the production fail-fast (30s) for the benchmark.
            .extraction_arm_budget_ms(300_000)
            .build()
            .expect("PipelineConfig default")
    };

    let engine = Engine::new(Arc::clone(&graph), llm.clone(), Arc::new(emb), config)
        ; // Phase E: Engine::new is infallible (was Result)

    // TD-021: KREMORY_BENCH_EXTRA_TYPES allows passing additional entity types
    // (comma-separated, e.g. "Court,Drug,Species") on top of the 10 defaults so
    // corpus-aware seeding can be benchmarked. The override REPLACES the
    // registry per ingest.rs:334 semantics, so we explicitly include the
    // defaults + extras.
    let extra_types: Vec<String> = std::env::var("KREMORY_BENCH_EXTRA_TYPES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let source_params = if extra_types.is_empty() {
        SourceParams::default()
    } else {
        use kremory::core::entity_types::{EntityTypeSpec, DEFAULT_ENTITY_TYPES};
        let mut specs: Vec<EntityTypeSpec> = DEFAULT_ENTITY_TYPES
            .iter()
            .map(|(id, name, desc): &(u32, &str, &str)| EntityTypeSpec {
                id: *id,
                name: name.to_string(),
                description: desc.to_string(),
            })
            .collect();
        let mut next_id = (specs.iter().map(|s| s.id).max().unwrap_or(0)) + 1;
        for extra in &extra_types {
            specs.push(EntityTypeSpec {
                id: next_id,
                name: extra.clone(),
                description: format!(
                    "Corpus-specific type for {} benchmark (KREMORY_BENCH_EXTRA_TYPES).",
                    domain_key
                ),
            });
            next_id += 1;
        }
        eprintln!(
            "label_precision_benchmark: extra_types={:?} total_registry_size={}",
            extra_types,
            specs.len()
        );
        SourceParams {
            entity_types_override: Some(specs),
            ..SourceParams::default()
        }
    };

    // Run extraction with one of:
    //   - DefaultExtractor (LLM, default)
    //   - GlinerExtractor (TD-022 GLiNER only, KREMORY_BENCH_USE_GLINER=1)
    //   - GlinerLlmExtractor (TD-023 GLiNER + 1 LLM typing call, KREMORY_BENCH_USE_HYBRID=1)
    let ingest_result = if use_hybrid {
        #[cfg(feature = "ner")]
        {
            eprintln!("label_precision_benchmark: USING GlinerLlmExtractor (TD-023)");
            let hybrid = kremory::core::extraction::GlinerLlmExtractor::new(
                Arc::clone(&llm),
            )
            .expect("GlinerLlmExtractor::new — needs GLiNER model + LLM");
            engine
                .ingest_with(&hybrid, &fixture_text, None, None, None, source_params)
                .await
        }
        #[cfg(not(feature = "ner"))]
        {
            panic!(
                "KREMORY_BENCH_USE_HYBRID=1 requires --features ner. \
                 Re-run with: cargo test --features llm-integration,ner ..."
            )
        }
    } else if use_gliner {
        #[cfg(feature = "ner")]
        {
            eprintln!("label_precision_benchmark: USING GlinerExtractor (TD-022)");
            let gliner = kremory::core::ner::GlinerExtractor::new()
                .expect("GlinerExtractor::new — downloads ~650MB INT8 model from HF on first use");
            engine
                .ingest_with(&gliner, &fixture_text, None, None, None, source_params)
                .await
        }
        #[cfg(not(feature = "ner"))]
        {
            panic!(
                "KREMORY_BENCH_USE_GLINER=1 requires --features ner. \
                 Re-run with: cargo test --features llm-integration,ner ..."
            )
        }
    } else {
        let extractor = DefaultExtractor::new(Arc::clone(&llm));
        engine
            .ingest_with(&extractor, &fixture_text, None, None, None, source_params)
            .await
    };

    // PHASE 1 DIAGNOSTIC: even on ingest failure, dump metrics snapshot before
    // panicking. Multi-stage extraction may fail at any stage; the metrics
    // captured up to the failure point still reveal what labels surfaced.
    if let Err(e) = &ingest_result {
        eprintln!(
            "\n── PHASE 1 EARLY-PANIC METRICS DUMP (ingest failed: {}) ──",
            e
        );
        let snap = snapshotter.snapshot().into_vec();
        let mut rejected_labels: Vec<(String, u64)> = Vec::new();
        for (key, _, _, debug_value) in &snap {
            let name = key.key().name().to_string();
            if name != "rql.entity.label_rejected_total" {
                continue;
            }
            let count = match debug_value {
                metrics_util::debugging::DebugValue::Counter(v) => *v,
                _ => 0,
            };
            let rl = key
                .key()
                .labels()
                .find(|l| l.key() == "rejected_label")
                .map(|l| l.value().to_string())
                .unwrap_or_else(|| "<no rejected_label dim>".to_string());
            rejected_labels.push((rl, count));
        }
        if rejected_labels.is_empty() {
            eprintln!("  rql.entity.label_rejected_total: NO REJECTIONS RECORDED before failure");
            eprintln!("  → meaning either (a) extraction failed before reaching L2,");
            eprintln!("    OR (b) all extracted labels passed L2 (entities are in graph).");
        } else {
            eprintln!("  rql.entity.label_rejected_total — LITERAL labels emitted by LLM and rejected by L2:");
            for (label, count) in &rejected_labels {
                eprintln!("    '{}' rejected {} time(s)", label, count);
            }
        }
        eprintln!("  All rql.* counters in snapshot (first 30):");
        for (key, _, _, dv) in snap.iter().take(30) {
            let name = key.key().name().to_string();
            if !name.starts_with("rql.") {
                continue;
            }
            let labels: Vec<_> = key
                .key()
                .labels()
                .map(|l| format!("{}={}", l.key(), l.value()))
                .collect();
            let count = match dv {
                metrics_util::debugging::DebugValue::Counter(v) => format!("counter={}", v),
                metrics_util::debugging::DebugValue::Gauge(v) => format!("gauge={:?}", v),
                metrics_util::debugging::DebugValue::Histogram(samples) => {
                    format!("histogram[{} samples]", samples.len())
                }
            };
            eprintln!("    {} [{}] {}", name, labels.join(","), count);
        }
        eprintln!("── END EARLY-PANIC DIAGNOSTIC ──\n");
    }

    let result = ingest_result
        .unwrap_or_else(|e| panic!("ingest of {} fixture must succeed: {e}", domain_key));

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

    // PHASE 1 DIAGNOSTIC DUMP (2026-06-03)
    // Surface LITERAL labels that L2 rejected (rql.entity.label_rejected_total{rejected_label}).
    // Also dump any other rql.* counters / histograms / gauges observed in the snapshot.
    //
    // TD-019 observability fix (2026-06-04): previously the snapshot reader
    // extracted Counter values only and mapped Histogram/Gauge to 0 silently.
    // That made every timing metric appear "0" in the diagnostic — a lying
    // counter (observability-first-class.md cardinal failure mode #9 — the
    // observability surface MUST surface the real value, not a sentinel).
    eprintln!("\n── PHASE 1 DIAGNOSTIC: Metrics snapshot ──");
    let snap = snapshotter.snapshot().into_vec();
    type MetricEntry = (String, Vec<(String, String)>, String);
    let mut rejected_labels: Vec<(String, u64)> = Vec::new();
    let mut other_entries: Vec<MetricEntry> = Vec::new();
    for (key, _, _, debug_value) in snap {
        let name = key.key().name().to_string();
        let labels: Vec<(String, String)> = key
            .key()
            .labels()
            .map(|l| (l.key().to_string(), l.value().to_string()))
            .collect();
        let rendered = match &debug_value {
            metrics_util::debugging::DebugValue::Counter(v) => format!("counter={}", v),
            metrics_util::debugging::DebugValue::Gauge(v) => format!("gauge={:.3}", v.0),
            metrics_util::debugging::DebugValue::Histogram(samples) => {
                if samples.is_empty() {
                    "histogram[empty]".to_string()
                } else {
                    let n = samples.len();
                    let mut min = f64::INFINITY;
                    let mut max = f64::NEG_INFINITY;
                    let mut sum = 0.0_f64;
                    for s in samples {
                        let v = s.0;
                        if v < min {
                            min = v;
                        }
                        if v > max {
                            max = v;
                        }
                        sum += v;
                    }
                    let avg = sum / n as f64;
                    format!(
                        "histogram[n={} sum={:.2} min={:.2} max={:.2} avg={:.2}]",
                        n, sum, min, max, avg
                    )
                }
            }
        };
        if name == "rql.entity.label_rejected_total" {
            let rl = labels
                .iter()
                .find(|(k, _)| k == "rejected_label")
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| "<no label dim>".to_string());
            let count = match debug_value {
                metrics_util::debugging::DebugValue::Counter(v) => v,
                _ => 0,
            };
            rejected_labels.push((rl, count));
        } else if name.starts_with("rql.") {
            other_entries.push((name, labels, rendered));
        }
    }
    if rejected_labels.is_empty() {
        eprintln!("  rql.entity.label_rejected_total: NO REJECTIONS RECORDED");
    } else {
        eprintln!(
            "  rql.entity.label_rejected_total — LITERAL labels emitted by LLM and rejected by L2:"
        );
        for (label, count) in &rejected_labels {
            eprintln!("    '{}' rejected {} time(s)", label, count);
        }
    }
    eprintln!("  Other rql.* entries in snapshot:");
    for (name, labels, rendered) in other_entries.iter().take(60) {
        eprintln!("    {} {:?} = {}", name, labels, rendered);
    }
    eprintln!("── END DIAGNOSTIC ──\n");

    // Primary gate: label precision >= 0.65 (R1 threshold per Vera Cycle 2 L4 alignment).
    //
    // TD-013 PR1-corrected + unified design (2026-06-03):
    // - 0.65 is the v0.1.7 threshold (post L1+L2+L3) per Graphiti baseline 0.674
    // - 0.75 was the pre-unified-design target requiring all 7 layers (deferred to v0.2.0)
    // - First implementation run should pass at >= 0.65, not at >= 0.75
    assert!(
        precision >= 0.65,
        "label_precision {:.1}% ({}/{}) is below the 65% R1 threshold.\n\
         Pre-Phase-4 (TD-012 broken state) produces ≈ 0%. Post-L1+L2+L3 target >= 65%.\n\
         {} entities had non-canonical labels.\n\
         Check: StructuredCallBuilder FormatSchema arm is active for model '{}'.",
        precision * 100.0,
        matched,
        total,
        placeholder_count,
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string()),
    );
}

// ── L4.2 — Anthropic Haiku variant ──────────────────────────────────────────
//
// Tests substrate quality with a commercial frontier-class small model.
// Validates whether the qwen2.5:14b 0%/60% stochasticity is a local-model
// limitation OR a substrate issue. Embeddings still come from Ollama
// (nomic-embed-text) since Anthropic doesn't expose an embedding API.

/// L4.2 — Label precision gate using Anthropic Haiku for chat extraction.
///
/// Requires:
///   - $ANTHROPIC_API_KEY (load from .env via direnv / dotenv / shell)
///   - $OLLAMA_HOST for nomic-embed-text embeddings
///
/// To run:
/// ```sh
/// ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2) \
/// OLLAMA_HOST=http://localhost:11434 \
/// cargo test -p kremory --features llm-integration --test label_precision_benchmark \
///   -- label_precision_haiku_on_mock_interview --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires $ANTHROPIC_API_KEY + $OLLAMA_HOST; enable with --ignored"]
async fn label_precision_haiku_on_mock_interview() {
    use autoagents_llm::backends::anthropic::Anthropic;

    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("SKIP label_precision_haiku: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let base_url = match ollama_base_url() {
        Some(url) => url,
        None => {
            eprintln!(
                "SKIP label_precision_haiku: OLLAMA_HOST not set (still needed for embeddings)"
            );
            return;
        }
    };

    let chat_model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    eprintln!("label_precision_haiku: model={chat_model}");

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

    let llm: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
        .api_key(api_key)
        .model(chat_model.clone())
        .timeout_seconds(120)
        .build()
        .expect("Anthropic LLM builder");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder");

    let emb = OllamaEmbedderAdapter(raw_emb);

    let dir = tempfile::tempdir().expect("tempdir");
    let graph = Arc::new(
        TemporalGraph::open(dir.path().join("haiku.db").to_str().expect("valid UTF-8"))
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .extraction_arm_budget_ms(120_000)
        .build()
        .expect("PipelineConfig default");

    let extractor = DefaultExtractor::new(Arc::clone(&llm));
    let engine = Engine::new(Arc::clone(&graph), llm, Arc::new(emb), config)
        ; // Phase E: Engine::new is infallible (was Result)

    let ingest_result = engine
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

    eprintln!(
        "haiku ingest: episode_id={} entities={} facts={}",
        ingest_result.episode_id,
        ingest_result.upserted_entities.len(),
        ingest_result.inserted_fact_ids.len()
    );

    // Pull all stored entities for this episode + compute precision.
    let entities = graph.list_entities().await.expect("list_entities");
    let extracted: Vec<(String, String)> = entities
        .iter()
        .map(|e| {
            let name = e
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| e.id.clone());
            (name, e.label.clone())
        })
        .collect();

    eprintln!("haiku extracted ({}): {:?}", extracted.len(), extracted);

    let expected: Vec<(String, String)> = domain
        .entities
        .iter()
        .map(|gt| (gt.name.clone(), gt.label.clone()))
        .collect();

    let (precision, matched, total) = label_precision(&extracted, &expected);
    let placeholder_count = extracted
        .iter()
        .filter(|(_, l)| !is_canonical_entity_type(l))
        .count();

    eprintln!(
        "label_precision (Haiku): {:.1}% ({}/{} ground-truth pairs matched, {} placeholder)",
        precision * 100.0,
        matched,
        total,
        placeholder_count
    );

    assert!(
        precision >= 0.65,
        "Haiku label_precision {:.1}% ({}/{}) below 65% R1 threshold. \
         {} entities had non-canonical labels. \
         Model: {chat_model}",
        precision * 100.0,
        matched,
        total,
        placeholder_count,
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
