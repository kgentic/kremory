#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: Candidate A — Unified LLM extraction + proper noun safety net
///
/// Tests the selected architecture: single LLM call (DefaultExtractor 3-stage)
/// followed by scan_proper_nouns() as a recall patch. Measures:
///   - LLM-only recall
///   - LLM + safety net recall (delta)
///   - Per-fixture latency breakdown (LLM vs scanner)
///   - Throughput (chars/s)
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=../models/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_unified_extraction -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::build_llm;
    use kremory::core::config::ContentType;
    use kremory::core::extraction::DefaultExtractor;
    use kremory::core::intelligence::{
        EntityExtractor, ExtractedEntity, ExtractionContext, ExtractionResult,
    };
    use kremory::core::text_utils::scan_proper_nouns;

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
                name: "Medical Consult",
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
            Fixture {
                name: "Sales Call",
                key: "sales_call",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sales_call.txt"),
                content_type: ContentType::Message,
            },
            Fixture {
                name: "Podcast",
                key: "podcast_interview",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/podcast_interview.txt"
                ),
                content_type: ContentType::Message,
            },
            Fixture {
                name: "Board Meeting",
                key: "board_meeting",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/board_meeting.txt"),
                content_type: ContentType::Text,
            },
            Fixture {
                name: "News Article",
                key: "news_article",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
                content_type: ContentType::Text,
            },
        ]
    }

    fn load_ground_truth() -> HashMap<String, Vec<String>> {
        let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
        let gt_raw = std::fs::read_to_string(gt_path).expect("ground_truth.json");
        let gt: serde_json::Value = serde_json::from_str(&gt_raw).expect("parse");
        gt.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                let entities: Vec<String> = v["entities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| e["name"].as_str().unwrap().to_lowercase())
                    .collect();
                (k.clone(), entities)
            })
            .collect()
    }

    fn fuzzy_match(extracted: &str, expected: &str) -> bool {
        let e = extracted.to_lowercase();
        let x = expected.to_lowercase();
        e == x || e.contains(&x) || x.contains(&e)
    }

    fn recall(extracted: &[String], expected: &[String]) -> (f64, usize) {
        let found = expected
            .iter()
            .filter(|exp| extracted.iter().any(|ext| fuzzy_match(ext, exp)))
            .count();
        let r = if expected.is_empty() {
            1.0
        } else {
            found as f64 / expected.len() as f64
        };
        (r, found)
    }

    #[derive(serde::Serialize)]
    struct MetricsRecord {
        fixture: String,
        llm_recall: f64,
        combined_recall: f64,
        recall_delta_pp: f64,
        llm_entities: usize,
        scanner_added: usize,
        combined_entities: usize,
        facts: usize,
        expected: usize,
        llm_ms: f64,
        scanner_ms: f64,
        total_ms: f64,
        input_chars: usize,
        chars_per_sec: f64,
        timestamp: String,
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_unified_with_safety_net() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let fixture_list = fixtures();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        // Metrics file
        let metrics_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring");
        std::fs::create_dir_all(metrics_dir).ok();
        let metrics_path = format!("{}/spike-unified-extraction.jsonl", metrics_dir);
        let mut metrics_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)
            .expect("metrics file");

        let default_types: Vec<String> = vec![
            "Person".into(),
            "Organisation".into(),
            "Location".into(),
            "Technology".into(),
            "Product".into(),
            "Event".into(),
            "Date".into(),
        ];

        eprintln!(
            "\n═══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: Candidate A — Unified LLM + Proper Noun Safety Net");
        eprintln!("Model: {model_path}");
        eprintln!("Extractor: DefaultExtractor::unconstrained (3-stage)");
        eprintln!("Safety net: scan_proper_nouns() post-LLM");
        eprintln!("Vocab: closed defaults + Entity catch-all");
        eprintln!(
            "═══════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>8} {:>8} {:>6} {:>6} {:>6} {:>5} {:>8} {:>8} {:>8}",
            "Domain", "LLM", "LLM+PN", "Delta", "LLMen", "+PN", "Facts", "LLM ms", "PN ms", "Total"
        );
        eprintln!("{}", "─".repeat(95));

        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        let mut totals = (
            0.0f64, 0.0f64, 0usize, 0usize, 0usize, 0.0f64, 0.0f64, 0usize,
        );

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    eprintln!("{:<18} SKIP — no ground truth", fixture.name);
                    continue;
                }
            };

            let text = std::fs::read_to_string(fixture.path).expect("read fixture");
            let input_chars = text.len();

            let ctx = ExtractionContext {
                allowed_entity_types: &default_types,
                allowed_edge_types: &[],
                known_entities: &[],
                excluded_entity_types: &[],
                content_type: fixture.content_type.clone(),
            };

            // Phase 1: LLM extraction
            let llm_start = Instant::now();
            let extractor = DefaultExtractor::new(llm.clone());
            let llm_result = extractor.extract(&text, &ctx).await;
            let llm_ms = llm_start.elapsed().as_secs_f64() * 1000.0;

            let (llm_entities, facts) = match llm_result {
                Ok(ExtractionResult { entities, facts }) => (entities, facts),
                Err(e) => {
                    eprintln!("{:<18} ERROR: {e}", fixture.name);
                    continue;
                }
            };

            let llm_names: Vec<String> =
                llm_entities.iter().map(|e| e.name.to_lowercase()).collect();
            let (llm_recall, _) = recall(&llm_names, expected);

            // Phase 2: Proper noun safety net (diff against LLM output)
            let pn_start = Instant::now();
            let pn_candidates = scan_proper_nouns(&text, &llm_entities);
            let pn_ms = pn_start.elapsed().as_secs_f64() * 1000.0;

            // Combine: LLM entities + scanner additions
            let scanner_added = pn_candidates.len();
            let mut combined: Vec<ExtractedEntity> = llm_entities.clone();
            combined.extend(pn_candidates);
            let combined_names: Vec<String> =
                combined.iter().map(|e| e.name.to_lowercase()).collect();
            let (combined_recall, _) = recall(&combined_names, expected);

            let delta_pp = (combined_recall - llm_recall) * 100.0;
            let total_ms = llm_ms + pn_ms;
            let throughput = if total_ms > 0.0 {
                input_chars as f64 / (total_ms / 1000.0)
            } else {
                0.0
            };

            eprintln!(
                "{:<18} {:>6.0}% {:>6.0}% {:>+5.0}pp {:>6} {:>+5} {:>5} {:>7.0} {:>7.1} {:>7.0}",
                fixture.name,
                llm_recall * 100.0,
                combined_recall * 100.0,
                delta_pp,
                llm_entities.len(),
                scanner_added,
                facts.len(),
                llm_ms,
                pn_ms,
                total_ms
            );

            // Write metrics
            let record = MetricsRecord {
                fixture: fixture.name.to_string(),
                llm_recall,
                combined_recall,
                recall_delta_pp: delta_pp,
                llm_entities: llm_entities.len(),
                scanner_added,
                combined_entities: combined.len(),
                facts: facts.len(),
                expected: expected.len(),
                llm_ms,
                scanner_ms: pn_ms,
                total_ms,
                input_chars,
                chars_per_sec: throughput,
                timestamp: timestamp.clone(),
            };
            if let Ok(json) = serde_json::to_string(&record) {
                let _ = writeln!(metrics_file, "{json}");
            }

            totals.0 += llm_recall;
            totals.1 += combined_recall;
            totals.2 += llm_entities.len();
            totals.3 += scanner_added;
            totals.4 += facts.len();
            totals.5 += llm_ms;
            totals.6 += pn_ms;
            totals.7 += 1;
        }

        let n = totals.7 as f64;
        let avg_llm = totals.0 / n;
        let avg_combined = totals.1 / n;

        eprintln!("{}", "─".repeat(95));
        eprintln!(
            "{:<18} {:>6.0}% {:>6.0}% {:>+5.0}pp {:>6} {:>+5} {:>5} {:>7.0} {:>7.1} {:>7.0}",
            "AVERAGE",
            avg_llm * 100.0,
            avg_combined * 100.0,
            (avg_combined - avg_llm) * 100.0,
            totals.2,
            totals.3,
            totals.4,
            totals.5,
            totals.6,
            totals.5 + totals.6
        );

        eprintln!(
            "\n═══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SUMMARY");
        eprintln!("  LLM-only recall:        {:.0}%", avg_llm * 100.0);
        eprintln!(
            "  LLM + safety net:       {:.0}% ({:+.0}pp from scanner)",
            avg_combined * 100.0,
            (avg_combined - avg_llm) * 100.0
        );
        eprintln!("  LLM latency total:      {:.1}s", totals.5 / 1000.0);
        eprintln!("  Scanner latency total:  {:.1}ms (negligible)", totals.6);
        eprintln!("  Facts extracted:        {}", totals.4);
        eprintln!("  Metrics: {metrics_path}");
        eprintln!(
            "═══════════════════════════════════════════════════════════════════════════════"
        );
    }
}
