#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: Qwen 2.5 3B entity+fact extraction via NuExtract template
///
/// Validates ADR: adr-unified-llm-extraction--background-ingestion-with-open-vocabulary
///
/// Tests:
///   1. Closed vocabulary (allowed_entity_types) — baseline
///   2. Open vocabulary (empty allowed_entity_types) — ADR approach
///   3. Temperature 0.0 (greedy) vs 0.1 — determinism check
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=../models/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_qwen_extraction -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::build_llm;
    use autoagents_llamacpp::LlamaCppProvider;
    use kremory::core::config::ContentType;
    use kremory::core::extraction::{DefaultExtractor, GraphitiStyleExtractor};
    use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};

    struct Fixture {
        name: &'static str,
        key: &'static str,
        path: &'static str,
        entity_types: Vec<String>,
        content_type: ContentType,
    }

    fn spike_fixtures() -> Vec<Fixture> {
        vec![
            Fixture {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Location".into(),
                    "University".into(),
                ],
                content_type: ContentType::Message,
            },
            Fixture {
                name: "Medical Consultation",
                key: "medical_consultation",
                path: concat!(
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
            Fixture {
                name: "Legal Deposition",
                key: "legal_deposition",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/legal_deposition.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Organisation".into(),
                    "Location".into(),
                    "Court".into(),
                    "Date".into(),
                ],
                content_type: ContentType::Text,
            },
            Fixture {
                name: "Tech Standup",
                key: "tech_standup",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
                entity_types: vec![
                    "Person".into(),
                    "Service".into(),
                    "Technology".into(),
                    "Tool".into(),
                ],
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

    struct RunResult {
        domain: String,
        recall: f64,
        found: usize,
        expected: usize,
        entity_count: usize,
        fact_count: usize,
        elapsed_s: f64,
        json_ok: bool,
    }

    #[derive(Clone, Copy)]
    enum ExtractorKind {
        Default3Stage,
        GraphitiStyle,
    }

    /// Run extraction directly via the EntityExtractor trait — no ingestion pipeline,
    /// no resolution, no contradiction. Pure extraction quality measurement.
    async fn run_extraction(
        llm: Arc<LlamaCppProvider>,
        fixture: &Fixture,
        expected_entities: &[String],
        open_vocab: bool,
        kind: ExtractorKind,
    ) -> RunResult {
        let transcript = std::fs::read_to_string(fixture.path).expect("read fixture");

        let entity_types: Vec<String> = if open_vocab {
            vec![]
        } else {
            fixture.entity_types.clone()
        };

        let ctx = ExtractionContext {
            allowed_entity_types: &entity_types,
            allowed_edge_types: &[],
            known_entities: &[],
            excluded_entity_types: &[],
            content_type: fixture.content_type.clone(),
            registry_specs: &[],
        };

        let start = Instant::now();
        let result = match kind {
            ExtractorKind::Default3Stage => {
                let extractor = DefaultExtractor::new(llm);
                extractor.extract(&transcript, &ctx).await
            }
            ExtractorKind::GraphitiStyle => {
                let extractor = GraphitiStyleExtractor::new(llm);
                extractor.extract(&transcript, &ctx).await
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
                }
            }
        }
    }

    fn print_header(mode: &str) {
        eprintln!("\n── {mode} ──\n");
        eprintln!(
            "{:<25} {:>7} {:>12} {:>10} {:>8} {:>7}",
            "Domain", "Recall", "Found/Exp", "Extracted", "Facts", "Time"
        );
        eprintln!("{}", "─".repeat(78));
    }

    fn print_row(r: &RunResult) {
        eprintln!(
            "{:<25} {:>6.0}% {:>5}/{:<5} {:>10} {:>8} {:>6.1}s {}",
            r.domain,
            r.recall * 100.0,
            r.found,
            r.expected,
            r.entity_count,
            r.fact_count,
            r.elapsed_s,
            if r.json_ok { "✓" } else { "✗ JSON" }
        );
    }

    fn print_summary(label: &str, results: &[RunResult]) {
        let avg_recall = results.iter().map(|r| r.recall).sum::<f64>() / results.len() as f64;
        let json_ok = results.iter().filter(|r| r.json_ok).count();
        let total_facts: usize = results.iter().map(|r| r.fact_count).sum();
        let total_entities: usize = results.iter().map(|r| r.entity_count).sum();
        eprintln!("{}", "─".repeat(78));
        eprintln!(
            "{label}: recall {:.0}% | JSON {}/{} | {} entities | {} facts",
            avg_recall * 100.0,
            json_ok,
            results.len(),
            total_entities,
            total_facts
        );
    }

    async fn run_mode(
        llm: Arc<LlamaCppProvider>,
        label: &str,
        fixtures: &[Fixture],
        gt: &HashMap<String, (Vec<String>, usize)>,
        open_vocab: bool,
        kind: ExtractorKind,
    ) -> Vec<RunResult> {
        print_header(label);
        let mut results = Vec::new();
        for fixture in fixtures {
            let (expected_entities, _min_rels) = gt
                .get(fixture.key)
                .unwrap_or_else(|| panic!("no ground truth for {}", fixture.key));
            let r = run_extraction(llm.clone(), fixture, expected_entities, open_vocab, kind).await;
            print_row(&r);
            results.push(r);
        }
        print_summary(label, &results);
        results
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_qwen3b_extraction() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        eprintln!("\n═══════════════════════════════════════════════════");
        eprintln!("SPIKE: Qwen 2.5 3B Entity+Fact Extraction");
        eprintln!("Model: {model_path}");
        eprintln!("Temperature: 0.0 (greedy)");
        eprintln!("═══════════════════════════════════════════════════");

        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider for Qwen 3B"),
        );

        let gt = load_ground_truth();
        let fixtures = spike_fixtures();

        // Mode 1: Default 3-stage (baseline) — closed vocab
        let def_closed = run_mode(
            llm.clone(),
            "Default 3-stage — CLOSED",
            &fixtures,
            &gt,
            false,
            ExtractorKind::Default3Stage,
        )
        .await;

        // Mode 2: Default 3-stage — open vocab
        let def_open = run_mode(
            llm.clone(),
            "Default 3-stage — OPEN",
            &fixtures,
            &gt,
            true,
            ExtractorKind::Default3Stage,
        )
        .await;

        // Mode 3: Graphiti-style prompts — closed vocab
        let gra_closed = run_mode(
            llm.clone(),
            "Graphiti-style — CLOSED",
            &fixtures,
            &gt,
            false,
            ExtractorKind::GraphitiStyle,
        )
        .await;

        // Mode 4: Graphiti-style prompts — open vocab (ADR target)
        let gra_open = run_mode(
            llm.clone(),
            "Graphiti-style — OPEN",
            &fixtures,
            &gt,
            true,
            ExtractorKind::GraphitiStyle,
        )
        .await;

        // ─── Summary ─────────────────────────────────────────────────────
        fn stats(results: &[RunResult]) -> (f64, usize, usize, usize) {
            let avg = results.iter().map(|r| r.recall).sum::<f64>() / results.len() as f64;
            let json_ok = results.iter().filter(|r| r.json_ok).count();
            let entities: usize = results.iter().map(|r| r.entity_count).sum();
            let facts: usize = results.iter().map(|r| r.fact_count).sum();
            (avg, json_ok, entities, facts)
        }

        let (def_c_r, def_c_j, def_c_e, def_c_f) = stats(&def_closed);
        let (def_o_r, def_o_j, def_o_e, def_o_f) = stats(&def_open);
        let (gra_c_r, gra_c_j, gra_c_e, gra_c_f) = stats(&gra_closed);
        let (gra_o_r, gra_o_j, gra_o_e, gra_o_f) = stats(&gra_open);

        eprintln!("\n═══════════════════════════════════════════════════");
        eprintln!("SPIKE RESULTS — Qwen 2.5 3B Extraction Comparison");
        eprintln!("═══════════════════════════════════════════════════");
        eprintln!(
            "{:<35} {:>7} {:>6} {:>8} {:>6}",
            "Mode", "Recall", "JSON", "Entities", "Facts"
        );
        eprintln!("{}", "─".repeat(68));
        eprintln!(
            "{:<35} {:>6.0}% {:>4}/{} {:>8} {:>6}",
            "Default 3-stage — closed",
            def_c_r * 100.0,
            def_c_j,
            def_closed.len(),
            def_c_e,
            def_c_f
        );
        eprintln!(
            "{:<35} {:>6.0}% {:>4}/{} {:>8} {:>6}",
            "Default 3-stage — open",
            def_o_r * 100.0,
            def_o_j,
            def_open.len(),
            def_o_e,
            def_o_f
        );
        eprintln!(
            "{:<35} {:>6.0}% {:>4}/{} {:>8} {:>6}",
            "Graphiti-style — closed",
            gra_c_r * 100.0,
            gra_c_j,
            gra_closed.len(),
            gra_c_e,
            gra_c_f
        );
        eprintln!(
            "{:<35} {:>6.0}% {:>4}/{} {:>8} {:>6}",
            "Graphiti-style — open",
            gra_o_r * 100.0,
            gra_o_j,
            gra_open.len(),
            gra_o_e,
            gra_o_f
        );
        eprintln!("{}", "─".repeat(68));

        let modes: Vec<(&str, f64)> = vec![
            ("Default 3-stage — closed", def_c_r),
            ("Default 3-stage — open", def_o_r),
            ("Graphiti-style — closed", gra_c_r),
            ("Graphiti-style — open", gra_o_r),
        ];
        let best = modes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        eprintln!("Best: {} ({:.0}%)", best.0, best.1 * 100.0);

        let prompt_delta_closed = gra_c_r - def_c_r;
        let prompt_delta_open = gra_o_r - def_o_r;
        eprintln!(
            "Prompt improvement: closed {}{:.0}pp, open {}{:.0}pp",
            if prompt_delta_closed >= 0.0 { "+" } else { "" },
            prompt_delta_closed * 100.0,
            if prompt_delta_open >= 0.0 { "+" } else { "" },
            prompt_delta_open * 100.0
        );
        eprintln!("═══════════════════════════════════════════════════");

        if best.1 < 0.5 {
            eprintln!("\n⚠ Best recall below 50% — consider Phi-4-mini or further prompt tuning");
        }
    }
}
