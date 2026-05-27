#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Multi-domain extraction benchmark with ground truth validation.
///
/// Each fixture has expected entities in fixtures/ground_truth.json.
/// The test computes entity recall (how many expected entities were found)
/// and reports per-domain scores.
///
/// Requires RQL_NUEXTRACT_MODEL_PATH pointing to NuExtract 2.0-4B GGUF.
///
/// Run with:
///   RQL_NUEXTRACT_MODEL_PATH=../models/NuExtract-2.0-4B-Q4_K_M.gguf \
///     cargo test --features llm --test multi_domain_extraction -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod domain_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::build_llm;
    use autoagents_llamacpp::LlamaCppProvider;
    use kremory::core::config::{ContentType, PipelineConfig};
    use kremory::core::extraction::NuExtractExtractor;
    use kremory::core::ingest::Engine;
    use kremory::core::provider::NullEmbeddingProvider;
    use kremory::core::schema::TemporalGraph;

    struct DomainFixture {
        name: &'static str,
        key: &'static str,
        transcript_path: &'static str,
        entity_types: Vec<String>,
        content_type: ContentType,
    }

    #[derive(Debug)]
    struct BenchmarkResult {
        name: String,
        elapsed_s: f64,
        entities_extracted: Vec<String>,
        facts_inserted: usize,
        expected_entities: Vec<String>,
        found: Vec<String>,
        missed: Vec<String>,
        recall: f64,
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

    async fn run_fixture(
        llm: Arc<LlamaCppProvider>,
        fixture: &DomainFixture,
        expected_entities: &[String],
        min_relationships: usize,
    ) -> BenchmarkResult {
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

        let extracted: Vec<String> = result.upserted_entities.clone();

        // Compute recall against ground truth
        let mut found = Vec::new();
        let mut missed = Vec::new();
        for expected in expected_entities {
            if extracted.iter().any(|e| fuzzy_match(e, expected)) {
                found.push(expected.clone());
            } else {
                missed.push(expected.clone());
            }
        }

        // Extra entities not in ground truth
        let extra: Vec<String> = extracted
            .iter()
            .filter(|e| !expected_entities.iter().any(|x| fuzzy_match(e, x)))
            .cloned()
            .collect();

        let recall = if expected_entities.is_empty() {
            1.0
        } else {
            found.len() as f64 / expected_entities.len() as f64
        };

        // Report
        println!("\n============================================================");
        println!("  {} — {:.1}s", fixture.name, elapsed.as_secs_f64());
        println!("============================================================");
        println!("  Extracted ({}): {:?}", extracted.len(), extracted);
        println!(
            "  Facts inserted: {} (expected >= {})",
            result.inserted_fact_ids.len(),
            min_relationships
        );
        println!("  ---");
        println!(
            "  Ground truth: {} expected entities",
            expected_entities.len()
        );
        println!(
            "  Found:  {}/{} ({:.0}% recall)",
            found.len(),
            expected_entities.len(),
            recall * 100.0
        );
        if !found.is_empty() {
            println!("    hit:  {:?}", found);
        }
        if !missed.is_empty() {
            println!("    miss: {:?}", missed);
        }
        if !extra.is_empty() {
            println!("    extra (not in ground truth): {:?}", extra);
        }

        BenchmarkResult {
            name: fixture.name.to_string(),
            elapsed_s: elapsed.as_secs_f64(),
            entities_extracted: extracted,
            facts_inserted: result.inserted_fact_ids.len(),
            expected_entities: expected_entities.to_vec(),
            found,
            missed,
            recall,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_multi_domain_extraction() {
        let model_path = match std::env::var("RQL_NUEXTRACT_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                println!("SKIP: RQL_NUEXTRACT_MODEL_PATH not set");
                return;
            }
        };

        let llm = Arc::new(
            build_llm(&model_path, 4096, 1024)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        let ground_truth = load_ground_truth();

        println!("\n============================================================");
        println!("  Multi-Domain NuExtract Benchmark");
        println!("  Model: {model_path}");
        println!("============================================================");

        let mut results = Vec::new();
        for fixture in &fixtures() {
            let (expected, min_rels) = ground_truth
                .get(fixture.key)
                .unwrap_or_else(|| panic!("no ground truth for '{}'", fixture.key));
            let r = run_fixture(llm.clone(), fixture, expected, *min_rels).await;
            results.push(r);
        }

        // Summary table
        println!("\n============================================================");
        println!("  SUMMARY");
        println!("============================================================");
        println!(
            "  {:<20} {:>6} {:>8} {:>6} {:>8} {:>7}",
            "Domain", "Time", "Entities", "Facts", "Recall", "Missed"
        );
        println!(
            "  {:-<20} {:->6} {:->8} {:->6} {:->8} {:->7}",
            "", "", "", "", "", ""
        );

        let mut total_expected = 0;
        let mut total_found = 0;

        for r in &results {
            total_expected += r.expected_entities.len();
            total_found += r.found.len();
            println!(
                "  {:<20} {:>5.1}s {:>5}/{:<2} {:>6} {:>7.0}% {:>7}",
                r.name,
                r.elapsed_s,
                r.entities_extracted.len(),
                r.expected_entities.len(),
                r.facts_inserted,
                r.recall * 100.0,
                r.missed.len(),
            );
        }

        let overall_recall = if total_expected > 0 {
            total_found as f64 / total_expected as f64
        } else {
            1.0
        };
        println!(
            "  {:-<20} {:->6} {:->8} {:->6} {:->8} {:->7}",
            "", "", "", "", "", ""
        );
        println!(
            "  {:<20} {:>6} {:>5}/{:<2} {:>6} {:>7.0}% {:>7}",
            "OVERALL",
            "",
            total_found,
            total_expected,
            results.iter().map(|r| r.facts_inserted).sum::<usize>(),
            overall_recall * 100.0,
            results.iter().map(|r| r.missed.len()).sum::<usize>(),
        );
        println!("============================================================");
    }
}

#[cfg(not(feature = "llm"))]
#[test]
fn test_multi_domain_feature_not_enabled() {
    println!("SKIP: compile with --features llm to run multi-domain extraction tests");
}
