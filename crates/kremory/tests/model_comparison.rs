#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Multi-model entity extraction comparison benchmark.
///
/// Runs all configured models against the same 4 domain fixtures and prints
/// a comparison table of entity recall scores.
///
/// Requires RQL_BENCHMARK_MODELS=1 to opt in (prevents accidental long runs).
/// Set model paths via env vars; any model whose path is unset is skipped.
///
/// Run with:
///   RQL_BENCHMARK_MODELS=1 \
///   RQL_NUEXTRACT_MODEL_PATH=../models/NuExtract-2.0-4B-Q4_K_M.gguf \
///   RQL_PHI4_MODEL_PATH=../models/Phi-4-mini-instruct.Q4_K_M.gguf \
///   RQL_QWEN3B_MODEL_PATH=../models/qwen2.5-3b-instruct-q4_k_m.gguf \
///   RQL_GEMMA4_MODEL_PATH=../models/gemma-4-E2B-it-Q4_K_M.gguf \
///     cargo test --features llm --test model_comparison -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod model_comparison_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::build_llm;
    use autoagents_llamacpp::LlamaCppProvider;
    use kremory::core::config::{ContentType, PipelineConfig};
    use kremory::core::extraction::{
        DefaultExtractor, GroundedNuExtractExtractor, NuExtractExtractor,
    };
    use kremory::core::ingest::Engine;
    use kremory::core::provider::NullEmbeddingProvider;
    use kremory::core::schema::TemporalGraph;

    // ─── Model configuration ──────────────────────────────────────────────────

    #[derive(Debug, Clone, Copy)]
    enum ExtractorKind {
        NuExtract,
        Default,
        DefaultUnconstrained,
        Grounded,
    }

    struct ModelConfig {
        name: &'static str,
        env_var: &'static str,
        extractor: ExtractorKind,
    }

    fn model_configs() -> Vec<ModelConfig> {
        vec![
            ModelConfig {
                name: "NuExtract-4B (template)",
                env_var: "RQL_NUEXTRACT_MODEL_PATH",
                extractor: ExtractorKind::NuExtract,
            },
            ModelConfig {
                name: "NuExtract-4B (grounded)",
                env_var: "RQL_NUEXTRACT_MODEL_PATH",
                extractor: ExtractorKind::Grounded,
            },
            ModelConfig {
                name: "Phi-4-mini (3-stage)",
                env_var: "RQL_PHI4_MODEL_PATH",
                extractor: ExtractorKind::Default,
            },
            ModelConfig {
                name: "Phi-4-mini (3-stage-nogrm)",
                env_var: "RQL_PHI4_MODEL_PATH",
                extractor: ExtractorKind::DefaultUnconstrained,
            },
            ModelConfig {
                name: "Qwen-3B (template)",
                env_var: "RQL_QWEN3B_MODEL_PATH",
                extractor: ExtractorKind::NuExtract,
            },
            ModelConfig {
                name: "Qwen-3B (3-stage)",
                env_var: "RQL_QWEN3B_MODEL_PATH",
                extractor: ExtractorKind::Default,
            },
            ModelConfig {
                name: "Gemma-4-E2B (3-stage)",
                env_var: "RQL_GEMMA4_MODEL_PATH",
                extractor: ExtractorKind::Default,
            },
            ModelConfig {
                name: "Gemma-4-E2B (3-stage-nogrm)",
                env_var: "RQL_GEMMA4_MODEL_PATH",
                extractor: ExtractorKind::DefaultUnconstrained,
            },
            ModelConfig {
                name: "Gemma-4-E2B (template)",
                env_var: "RQL_GEMMA4_MODEL_PATH",
                extractor: ExtractorKind::NuExtract,
            },
        ]
    }

    // ─── Domain fixtures ──────────────────────────────────────────────────────

    struct DomainFixture {
        name: &'static str,
        key: &'static str,
        transcript_path: &'static str,
        entity_types: Vec<String>,
        content_type: ContentType,
    }

    fn fixtures() -> Vec<DomainFixture> {
        vec![
            DomainFixture {
                name: "Mock Interview",
                key: "mock_interview",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/mock_interview.txt"
                ),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Location".into(),
                    "University".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Tech Standup",
                key: "tech_standup",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Service".into(),
                    "Technology".into(),
                    "Tool".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Podcast Interview",
                key: "podcast_interview",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/podcast_interview.txt"
                ),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Species".into(),
                    "Location".into(),
                    "Money".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Sales Call",
                key: "sales_call",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sales_call.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Product".into(),
                    "Money".into(),
                    "Location".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Legal Deposition",
                key: "legal_deposition",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/legal_deposition.txt"
                ),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Location".into(),
                    "Court".into(),
                    "Date".into(),
                ],
                content_type: ContentType::Text,
            },
            DomainFixture {
                name: "Medical Consultation",
                key: "medical_consultation",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/medical_consultation.txt"
                ),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Drug".into(),
                    "Condition".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Academic Lecture",
                key: "academic_lecture",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/academic_lecture.txt"
                ),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Publication".into(),
                    "Theory".into(),
                ],
                content_type: ContentType::Text,
            },
            DomainFixture {
                name: "Customer Support",
                key: "customer_support",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/customer_support.txt"
                ),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Product".into(),
                    "Software".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Board Meeting",
                key: "board_meeting",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/board_meeting.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Location".into(),
                    "Committee".into(),
                ],
                content_type: ContentType::Text,
            },
            DomainFixture {
                name: "News Article",
                key: "news_article",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
                entity_types: vec!["Person".into(), "Organisation".into(), "Location".into()],
                content_type: ContentType::Text,
            },
            DomainFixture {
                name: "Slack Thread",
                key: "slack_thread",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/slack_thread.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Tool".into(),
                    "Product".into(),
                    "Technology".into(),
                ],
                content_type: ContentType::Message,
            },
            DomainFixture {
                name: "Product Review",
                key: "product_review",
                transcript_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/product_review.txt"
                ),
                entity_types: vec!["Organisation".into(), "Product".into(), "Feature".into()],
                content_type: ContentType::Text,
            },
            DomainFixture {
                name: "Short Snippet",
                key: "short_snippet",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/short_snippet.txt"),
                entity_types: vec!["Person".into(), "Organisation".into()],
                content_type: ContentType::Text,
            },
            DomainFixture {
                name: "Long Report",
                key: "long_report",
                transcript_path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/long_report.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Product".into(),
                    "Location".into(),
                ],
                content_type: ContentType::Text,
            },
        ]
    }

    // ─── Ground truth loading ─────────────────────────────────────────────────

    fn load_ground_truth() -> HashMap<String, (Vec<String>, usize)> {
        let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
        let gt_raw = std::fs::read_to_string(gt_path)
            .unwrap_or_else(|e| panic!("failed to read ground_truth.json: {e}"));
        let gt: serde_json::Value = serde_json::from_str(&gt_raw)
            .unwrap_or_else(|e| panic!("failed to parse ground_truth.json: {e}"));

        let mut map = HashMap::new();
        for (key, domain) in gt.as_object().unwrap() {
            let entities: Vec<String> = domain["entities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["name"].as_str().unwrap().to_lowercase())
                .collect();
            let min_rels = domain["min_relationships"].as_u64().unwrap_or(0) as usize;
            map.insert(key.clone(), (entities, min_rels));
        }
        map
    }

    /// Fuzzy match: extracted name contains or is contained by expected name.
    fn fuzzy_match(extracted: &str, expected: &str) -> bool {
        let e = extracted.to_lowercase();
        let x = expected.to_lowercase();
        e == x || e.contains(&x) || x.contains(&e)
    }

    // ─── Per-fixture result ───────────────────────────────────────────────────

    #[derive(Debug)]
    #[allow(dead_code)]
    struct FixtureResult {
        domain_name: String,
        recall: f64,
        precision: f64,
        f1: f64,
        found: usize,
        expected: usize,
        total_extracted: usize,
        elapsed_s: f64,
    }

    // ─── Run one fixture with a NuExtract extractor ───────────────────────────

    async fn run_nuextract(
        llm: Arc<LlamaCppProvider>,
        fixture: &DomainFixture,
        expected_entities: &[String],
    ) -> FixtureResult {
        let transcript = std::fs::read_to_string(fixture.transcript_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture.transcript_path));

        let config = PipelineConfig::builder()
            .min_tokens(50)
            .max_tokens(4000)
            .allowed_entity_types(fixture.entity_types.clone())
            .build()
            .expect("PipelineConfig build failed");

        let embedder = NullEmbeddingProvider {
            dim: config.embedding_dim.0,
        };
        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open graph"),
        );
        let rql = Engine::new(graph, llm.clone(), Arc::new(embedder), config);
        let extractor = NuExtractExtractor::new(llm);

        let start = Instant::now();
        let result = rql
            .ingest_with(
                &extractor,
                &transcript,
                None,
                None,
                Some(fixture.content_type.clone()),
            )
            .await
            .expect("ingest_with() failed");
        let elapsed = start.elapsed();

        compute_fixture_result(
            fixture,
            &result.upserted_entities,
            expected_entities,
            elapsed.as_secs_f64(),
        )
    }

    // ─── Run one fixture with a Default extractor ─────────────────────────────

    async fn run_default(
        llm: Arc<LlamaCppProvider>,
        fixture: &DomainFixture,
        expected_entities: &[String],
    ) -> FixtureResult {
        let transcript = std::fs::read_to_string(fixture.transcript_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture.transcript_path));

        let config = PipelineConfig::builder()
            .min_tokens(50)
            .max_tokens(4000)
            .allowed_entity_types(fixture.entity_types.clone())
            .build()
            .expect("PipelineConfig build failed");

        let embedder = NullEmbeddingProvider {
            dim: config.embedding_dim.0,
        };
        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open graph"),
        );
        let rql = Engine::new(graph, llm.clone(), Arc::new(embedder), config);
        let extractor = DefaultExtractor::new(llm);

        let start = Instant::now();
        let result = rql
            .ingest_with(
                &extractor,
                &transcript,
                None,
                None,
                Some(fixture.content_type.clone()),
            )
            .await
            .expect("ingest_with() failed");
        let elapsed = start.elapsed();

        compute_fixture_result(
            fixture,
            &result.upserted_entities,
            expected_entities,
            elapsed.as_secs_f64(),
        )
    }

    // ─── Run one fixture with an unconstrained Default extractor ───────────────

    async fn run_default_unconstrained(
        llm: Arc<LlamaCppProvider>,
        fixture: &DomainFixture,
        expected_entities: &[String],
    ) -> FixtureResult {
        let transcript = std::fs::read_to_string(fixture.transcript_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture.transcript_path));

        let config = PipelineConfig::builder()
            .min_tokens(50)
            .max_tokens(4000)
            .allowed_entity_types(fixture.entity_types.clone())
            .build()
            .expect("PipelineConfig build failed");

        let embedder = NullEmbeddingProvider {
            dim: config.embedding_dim.0,
        };
        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open graph"),
        );
        let rql = Engine::new(graph, llm.clone(), Arc::new(embedder), config);
        let extractor = DefaultExtractor::new(llm);

        let start = Instant::now();
        let result = rql
            .ingest_with(
                &extractor,
                &transcript,
                None,
                None,
                Some(fixture.content_type.clone()),
            )
            .await
            .expect("ingest_with() failed");
        let elapsed = start.elapsed();

        compute_fixture_result(
            fixture,
            &result.upserted_entities,
            expected_entities,
            elapsed.as_secs_f64(),
        )
    }

    // ─── Run one fixture with a Grounded extractor ────────────────────────────

    async fn run_grounded(
        llm: Arc<LlamaCppProvider>,
        fixture: &DomainFixture,
        expected_entities: &[String],
    ) -> FixtureResult {
        let transcript = std::fs::read_to_string(fixture.transcript_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture.transcript_path));

        let config = PipelineConfig::builder()
            .min_tokens(50)
            .max_tokens(4000)
            .allowed_entity_types(fixture.entity_types.clone())
            .build()
            .expect("PipelineConfig build failed");

        let embedder = NullEmbeddingProvider {
            dim: config.embedding_dim.0,
        };
        let graph = Arc::new(
            TemporalGraph::open_in_memory()
                .await
                .expect("failed to open graph"),
        );
        let rql = Engine::new(graph, llm.clone(), Arc::new(embedder), config);
        let extractor = GroundedNuExtractExtractor::new(llm);

        let start = Instant::now();
        let result = rql
            .ingest_with(
                &extractor,
                &transcript,
                None,
                None,
                Some(fixture.content_type.clone()),
            )
            .await
            .expect("ingest_with() failed");
        let elapsed = start.elapsed();

        compute_fixture_result(
            fixture,
            &result.upserted_entities,
            expected_entities,
            elapsed.as_secs_f64(),
        )
    }

    // ─── Shared recall computation ────────────────────────────────────────────

    fn compute_fixture_result(
        fixture: &DomainFixture,
        extracted: &[String],
        expected_entities: &[String],
        elapsed_s: f64,
    ) -> FixtureResult {
        let mut found = Vec::new();
        let mut missed = Vec::new();
        for expected in expected_entities {
            if extracted.iter().any(|e| fuzzy_match(e, expected)) {
                found.push(expected.clone());
            } else {
                missed.push(expected.clone());
            }
        }

        let tp = found.len() as f64;
        let recall = if expected_entities.is_empty() {
            1.0
        } else {
            tp / expected_entities.len() as f64
        };
        let precision = if extracted.is_empty() {
            0.0
        } else {
            tp / extracted.len() as f64
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!(
            "    {} — R:{:.0}% P:{:.0}% F1:{:.0}% ({}/{} found, {} extracted) in {:.1}s  miss={:?}",
            fixture.name,
            recall * 100.0,
            precision * 100.0,
            f1 * 100.0,
            found.len(),
            expected_entities.len(),
            extracted.len(),
            elapsed_s,
            missed,
        );

        FixtureResult {
            domain_name: fixture.name.to_string(),
            recall,
            precision,
            f1,
            found: found.len(),
            expected: expected_entities.len(),
            total_extracted: extracted.len(),
            elapsed_s,
        }
    }

    // ─── Per-model runner ─────────────────────────────────────────────────────

    #[allow(dead_code)]
    struct ModelResult {
        model_name: String,
        fixture_results: Vec<FixtureResult>,
        overall_recall: f64,
        overall_precision: f64,
        overall_f1: f64,
        total_elapsed_s: f64,
        avg_elapsed_s: f64,
    }

    async fn run_model(
        model_name: &str,
        model_path: &str,
        extractor_kind: ExtractorKind,
        fixtures: &[DomainFixture],
        ground_truth: &HashMap<String, (Vec<String>, usize)>,
    ) -> ModelResult {
        println!("\n  Loading: {model_name} ({model_path})");

        let llm = Arc::new(
            build_llm(model_path, 4096, 512)
                .await
                .unwrap_or_else(|e| panic!("failed to load model {model_name}: {e}")),
        );

        println!("  Running fixtures for: {model_name}");

        let mut fixture_results = Vec::new();
        for fixture in fixtures {
            let (expected, _min_rels) = ground_truth
                .get(fixture.key)
                .unwrap_or_else(|| panic!("no ground truth for '{}'", fixture.key));

            let fr = match extractor_kind {
                ExtractorKind::NuExtract => run_nuextract(llm.clone(), fixture, expected).await,
                ExtractorKind::Default => run_default(llm.clone(), fixture, expected).await,
                ExtractorKind::DefaultUnconstrained => {
                    run_default_unconstrained(llm.clone(), fixture, expected).await
                }
                ExtractorKind::Grounded => run_grounded(llm.clone(), fixture, expected).await,
            };
            fixture_results.push(fr);
        }

        let total_expected: usize = fixture_results.iter().map(|r| r.expected).sum();
        let total_found: usize = fixture_results.iter().map(|r| r.found).sum();
        let total_extracted: usize = fixture_results.iter().map(|r| r.total_extracted).sum();
        let total_elapsed_s: f64 = fixture_results.iter().map(|r| r.elapsed_s).sum();
        let avg_elapsed_s = if fixture_results.is_empty() {
            0.0
        } else {
            total_elapsed_s / fixture_results.len() as f64
        };

        let overall_recall = if total_expected > 0 {
            total_found as f64 / total_expected as f64
        } else {
            1.0
        };
        let overall_precision = if total_extracted > 0 {
            total_found as f64 / total_extracted as f64
        } else {
            0.0
        };
        let overall_f1 = if overall_precision + overall_recall > 0.0 {
            2.0 * overall_precision * overall_recall / (overall_precision + overall_recall)
        } else {
            0.0
        };

        ModelResult {
            model_name: model_name.to_string(),
            fixture_results,
            overall_recall,
            overall_precision,
            overall_f1,
            total_elapsed_s,
            avg_elapsed_s,
        }
    }

    // ─── Comparison table printer ─────────────────────────────────────────────

    fn print_comparison_table(results: &[ModelResult]) {
        const COL_MODEL: usize = 31;
        const COL_METRIC: usize = 8;

        let sep_model = "-".repeat(COL_MODEL);
        let sep_metric = "-".repeat(COL_METRIC);

        println!("\n");
        println!("  {:<COL_MODEL$} | {:>COL_METRIC$} | {:>COL_METRIC$} | {:>COL_METRIC$} | {:>COL_METRIC$} | {:>COL_METRIC$}",
            "Model", "Recall", "Precis.", "F1", "Total(s)", "Avg(s)");
        println!("  {sep_model}-+-{sep_metric}-+-{sep_metric}-+-{sep_metric}-+-{sep_metric}-+-{sep_metric}");

        for mr in results {
            println!(
                "  {:<COL_MODEL$} | {:>5.0}%   | {:>5.0}%   | {:>5.0}%   | {:>6.1}  | {:>6.1} ",
                mr.model_name,
                mr.overall_recall * 100.0,
                mr.overall_precision * 100.0,
                mr.overall_f1 * 100.0,
                mr.total_elapsed_s,
                mr.avg_elapsed_s,
            );
        }

        println!("  {sep_model}-+-{sep_metric}-+-{sep_metric}-+-{sep_metric}-+-{sep_metric}-+-{sep_metric}");
    }

    // ─── Test entry point ─────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_model_comparison() {
        // Gate: opt-in env var required.
        if std::env::var("RQL_BENCHMARK_MODELS").is_err() {
            println!("SKIP: set RQL_BENCHMARK_MODELS=1 to run model comparison benchmark");
            return;
        }

        let ground_truth = load_ground_truth();
        let all_fixtures = fixtures();
        let configs = model_configs();

        println!("\n============================================================");
        println!("  MODEL COMPARISON — Entity Recall Benchmark");
        println!("============================================================");

        let mut model_results: Vec<ModelResult> = Vec::new();

        for config in &configs {
            match std::env::var(config.env_var) {
                Err(_) => {
                    println!("\n  SKIP: {} ({} not set)", config.name, config.env_var);
                }
                Ok(model_path) => {
                    let mr = run_model(
                        config.name,
                        &model_path,
                        config.extractor,
                        &all_fixtures,
                        &ground_truth,
                    )
                    .await;
                    model_results.push(mr);
                }
            }
        }

        if model_results.is_empty() {
            println!("\n  No models were run. Set at least one model path env var.");
            return;
        }

        println!("\n============================================================");
        println!("  MODEL COMPARISON — Recall / Precision / F1 / Latency");
        println!("============================================================");
        print_comparison_table(&model_results);
        println!();
    }
}

#[cfg(not(feature = "llm"))]
#[test]
fn test_model_comparison_feature_not_enabled() {
    println!("SKIP: compile with --features llm to run model comparison benchmark");
}
