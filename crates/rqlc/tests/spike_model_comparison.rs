#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: Multi-model extraction benchmark with latency metrics
///
/// Benchmarks all available models × extractors against 4 domain fixtures.
/// Outputs per-fixture metrics and writes a JSONL metrics file to
/// `rql-core/monitoring/spike-extraction-metrics.jsonl`
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=../models/qwen2.5-3b-instruct-q4_k_m.gguf \
///   RQL_PHI4_MODEL_PATH=../models/Phi-4-mini-instruct.Q4_K_M.gguf \
///   RQL_NUEXTRACT_MODEL_PATH=../models/NuExtract-2.0-4B-Q4_K_M.gguf \
///     cargo test --features llm -p rql-core --test spike_model_comparison -- --nocapture
///
/// Any model whose env var is unset is skipped.

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Instant;

    use rql_core::config::ContentType;
    use rql_core::extraction::{
        DefaultExtractor, GraphitiStyleExtractor, NuExtractExtractor,
    };
    use rql_core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
    use autoagents_llamacpp::LlamaCppProvider;
    use super::common::build_llm;

    struct Fixture {
        name: &'static str,
        key: &'static str,
        path: &'static str,
        content_type: ContentType,
    }

    fn fixtures() -> Vec<Fixture> {
        vec![
            Fixture {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
                content_type: ContentType::Message,
            },
            Fixture {
                name: "Medical Consultation",
                key: "medical_consultation",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/medical_consultation.txt"
                ),
                content_type: ContentType::Message,
            },
            Fixture {
                name: "Legal Deposition",
                key: "legal_deposition",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/legal_deposition.txt"),
                content_type: ContentType::Text,
            },
            Fixture {
                name: "Tech Standup",
                key: "tech_standup",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
                content_type: ContentType::Message,
            },
        ]
    }

    fn load_ground_truth() -> HashMap<String, (Vec<String>, usize)> {
        let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
        let gt_raw = std::fs::read_to_string(gt_path).expect("ground_truth.json");
        let gt: serde_json::Value = serde_json::from_str(&gt_raw).expect("parse ground_truth");
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

    fn fuzzy_match(extracted: &str, expected: &str) -> bool {
        let e = extracted.to_lowercase();
        let x = expected.to_lowercase();
        e == x || e.contains(&x) || x.contains(&e)
    }

    #[derive(serde::Serialize)]
    struct MetricsRecord {
        model: String,
        extractor: String,
        domain: String,
        recall: f64,
        found: usize,
        expected: usize,
        entity_count: usize,
        fact_count: usize,
        elapsed_s: f64,
        json_ok: bool,
        input_chars: usize,
        timestamp: String,
    }

    struct RunResult {
        domain: String,
        recall: f64,
        found: usize,
        expected: usize,
        entity_count: usize,
        fact_count: usize,
        elapsed_s: f64,
        json_ok: bool,
        input_chars: usize,
    }

    #[derive(Clone, Copy)]
    enum ExtractorKind {
        GraphitiStyle,
        NuExtractTemplate,
        Default3Stage,
    }

    impl ExtractorKind {
        fn label(&self) -> &'static str {
            match self {
                Self::GraphitiStyle => "graphiti",
                Self::NuExtractTemplate => "nuextract-template",
                Self::Default3Stage => "default-3stage",
            }
        }
    }

    async fn run_one(
        llm: Arc<LlamaCppProvider>,
        fixture: &Fixture,
        expected_entities: &[String],
        kind: ExtractorKind,
    ) -> RunResult {
        let transcript = std::fs::read_to_string(fixture.path).expect("read fixture");
        let input_chars = transcript.len();

        // Use default broad types (closed vocab with Entity catch-all)
        let default_types: Vec<String> = vec![
            "Person".into(),
            "Organisation".into(),
            "Location".into(),
            "Technology".into(),
            "Product".into(),
            "Event".into(),
            "Date".into(),
        ];

        let ctx = ExtractionContext {
            allowed_entity_types: &default_types,
            allowed_edge_types: &[],
            known_entities: &[],
            excluded_entity_types: &[],
            content_type: fixture.content_type.clone(),
        };

        let start = Instant::now();
        let result = match kind {
            ExtractorKind::GraphitiStyle => {
                let ex = GraphitiStyleExtractor::new(llm);
                ex.extract(&transcript, &ctx).await
            }
            ExtractorKind::NuExtractTemplate => {
                let ex = NuExtractExtractor::new(llm);
                ex.extract(&transcript, &ctx).await
            }
            ExtractorKind::Default3Stage => {
                let ex = DefaultExtractor::new(llm);
                ex.extract(&transcript, &ctx).await
            }
        };
        let elapsed = start.elapsed().as_secs_f64();

        match result {
            Ok(ExtractionResult { entities, facts }) => {
                let extracted_lower: Vec<String> =
                    entities.iter().map(|e| e.name.to_lowercase()).collect();
                let found = expected_entities
                    .iter()
                    .filter(|exp| extracted_lower.iter().any(|ext| fuzzy_match(ext, exp)))
                    .count();
                let recall = if expected_entities.is_empty() {
                    1.0
                } else {
                    found as f64 / expected_entities.len() as f64
                };
                RunResult {
                    domain: fixture.name.to_string(),
                    recall,
                    found,
                    expected: expected_entities.len(),
                    entity_count: entities.len(),
                    fact_count: facts.len(),
                    elapsed_s: elapsed,
                    json_ok: true,
                    input_chars,
                }
            }
            Err(e) => {
                eprintln!("  ERROR {}: {e}", fixture.name);
                RunResult {
                    domain: fixture.name.to_string(),
                    recall: 0.0,
                    found: 0,
                    expected: expected_entities.len(),
                    entity_count: 0,
                    fact_count: 0,
                    elapsed_s: elapsed,
                    json_ok: false,
                    input_chars,
                }
            }
        }
    }

    struct ModelConfig {
        name: &'static str,
        env_var: &'static str,
        kind: ExtractorKind,
    }

    fn model_configs() -> Vec<ModelConfig> {
        vec![
            ModelConfig {
                name: "Qwen-3B",
                env_var: "RQL_QWEN3B_MODEL_PATH",
                kind: ExtractorKind::GraphitiStyle,
            },
            ModelConfig {
                name: "Qwen-3B",
                env_var: "RQL_QWEN3B_MODEL_PATH",
                kind: ExtractorKind::Default3Stage,
            },
            ModelConfig {
                name: "Phi-4-mini",
                env_var: "RQL_PHI4_MODEL_PATH",
                kind: ExtractorKind::GraphitiStyle,
            },
            ModelConfig {
                name: "Phi-4-mini",
                env_var: "RQL_PHI4_MODEL_PATH",
                kind: ExtractorKind::Default3Stage,
            },
            ModelConfig {
                name: "NuExtract-4B",
                env_var: "RQL_NUEXTRACT_MODEL_PATH",
                kind: ExtractorKind::NuExtractTemplate,
            },
            ModelConfig {
                name: "NuExtract-4B",
                env_var: "RQL_NUEXTRACT_MODEL_PATH",
                kind: ExtractorKind::GraphitiStyle,
            },
        ]
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_model_comparison() {
        let gt = load_ground_truth();
        let fixture_list = fixtures();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        // Open metrics file
        let metrics_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring");
        std::fs::create_dir_all(metrics_dir).ok();
        let metrics_path = format!("{}/spike-extraction-metrics.jsonl", metrics_dir);
        let mut metrics_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)
            .expect("open metrics file");

        eprintln!("\nMetrics file: {metrics_path}");

        let mut summary: Vec<(String, String, f64, usize, usize, usize, f64)> = Vec::new();

        for mc in model_configs() {
            let model_path = match std::env::var(mc.env_var) {
                Ok(p) => p,
                Err(_) => {
                    eprintln!(
                        "SKIP {} — {}: {} not set",
                        mc.name,
                        mc.kind.label(),
                        mc.env_var
                    );
                    continue;
                }
            };

            let display_name = format!("{} — {}", mc.name, mc.kind.label());
            eprintln!("\n═══════════════════════════════════════════════════");
            eprintln!("  {display_name}");
            eprintln!("═══════════════════════════════════════════════════");
            eprintln!(
                "{:<22} {:>6} {:>10} {:>6} {:>5} {:>6} {:>8}",
                "Domain", "Recall", "Found/Exp", "Ents", "Facts", "Time", "chars/s"
            );
            eprintln!("{}", "─".repeat(72));

            let llm = Arc::new(
                build_llm(&model_path, 4096, 512)
                    .await
                    .expect("failed to build LlamaCppProvider"),
            );

            let mut total_recall = 0.0;
            let mut total_entities = 0;
            let mut total_facts = 0;
            let mut total_time = 0.0;
            let mut json_ok_count = 0;

            for fixture in &fixture_list {
                let (expected, _) = gt.get(fixture.key).expect("missing ground truth");
                let r = run_one(llm.clone(), fixture, expected, mc.kind).await;

                let throughput = if r.elapsed_s > 0.0 {
                    r.input_chars as f64 / r.elapsed_s
                } else {
                    0.0
                };

                eprintln!(
                    "{:<22} {:>5.0}% {:>5}/{:<4} {:>6} {:>5} {:>5.1}s {:>7.0}",
                    r.domain,
                    r.recall * 100.0,
                    r.found,
                    r.expected,
                    r.entity_count,
                    r.fact_count,
                    r.elapsed_s,
                    throughput
                );

                // Write metrics record
                let record = MetricsRecord {
                    model: mc.name.to_string(),
                    extractor: mc.kind.label().to_string(),
                    domain: r.domain.clone(),
                    recall: r.recall,
                    found: r.found,
                    expected: r.expected,
                    entity_count: r.entity_count,
                    fact_count: r.fact_count,
                    elapsed_s: r.elapsed_s,
                    json_ok: r.json_ok,
                    input_chars: r.input_chars,
                    timestamp: timestamp.clone(),
                };
                if let Ok(json) = serde_json::to_string(&record) {
                    let _ = writeln!(metrics_file, "{json}");
                }

                total_recall += r.recall;
                total_entities += r.entity_count;
                total_facts += r.fact_count;
                total_time += r.elapsed_s;
                if r.json_ok {
                    json_ok_count += 1;
                }
            }

            let avg_recall = total_recall / fixture_list.len() as f64;
            eprintln!("{}", "─".repeat(72));
            eprintln!(
                "AVG: {:.0}% recall | {}/{} JSON | {} ents | {} facts | {:.0}s",
                avg_recall * 100.0,
                json_ok_count,
                fixture_list.len(),
                total_entities,
                total_facts,
                total_time
            );

            summary.push((
                mc.name.to_string(),
                mc.kind.label().to_string(),
                avg_recall,
                total_entities,
                total_facts,
                json_ok_count,
                total_time,
            ));

            drop(llm);
        }

        // ─── Final comparison ────────────────────────────────────────────
        eprintln!("\n═══════════════════════════════════════════════════════════════");
        eprintln!("FINAL COMPARISON — Closed Vocab (defaults + Entity catch-all)");
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!(
            "{:<30} {:>7} {:>6} {:>5} {:>4} {:>7}",
            "Model + Extractor", "Recall", "Ents", "Facts", "JSON", "Time"
        );
        eprintln!("{}", "─".repeat(64));
        for (model, ext, recall, ents, facts, json, time) in &summary {
            eprintln!(
                "{:<30} {:>6.0}% {:>6} {:>5} {:>3}/4 {:>6.0}s",
                format!("{} — {}", model, ext),
                recall * 100.0,
                ents,
                facts,
                json,
                time
            );
        }
        eprintln!("{}", "─".repeat(64));

        if let Some(best) = summary.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) {
            eprintln!(
                "Winner: {} — {} ({:.0}% recall, {:.0}s)",
                best.0,
                best.1,
                best.2 * 100.0,
                best.6
            );
        }
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!("\nMetrics written to: {metrics_path}");
    }
}