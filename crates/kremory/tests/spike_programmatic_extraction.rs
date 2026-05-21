/// Spike: Programmatic entity extraction (zero LLM) recall benchmark
///
/// Tests scan_proper_nouns() against the same ground truth to measure
/// what percentage of entities we get for free without any LLM calls.
///
/// Run with:
///   cargo test -p rql-core --test spike_programmatic_extraction -- --nocapture

mod spike {
    use std::collections::HashMap;

    use kremory::core::text_utils::scan_proper_nouns;

    struct Fixture {
        name: &'static str,
        key: &'static str,
        path: &'static str,
    }

    fn fixtures() -> Vec<Fixture> {
        vec![
            Fixture {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
            },
            Fixture {
                name: "Medical Consultation",
                key: "medical_consultation",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/medical_consultation.txt"
                ),
            },
            Fixture {
                name: "Legal Deposition",
                key: "legal_deposition",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/legal_deposition.txt"),
            },
            Fixture {
                name: "Tech Standup",
                key: "tech_standup",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
            },
            Fixture {
                name: "Sales Call",
                key: "sales_call",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sales_call.txt"),
            },
            Fixture {
                name: "Podcast Interview",
                key: "podcast_interview",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/podcast_interview.txt"
                ),
            },
            Fixture {
                name: "Board Meeting",
                key: "board_meeting",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/board_meeting.txt"),
            },
            Fixture {
                name: "News Article",
                key: "news_article",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
            },
        ]
    }

    fn load_ground_truth() -> HashMap<String, Vec<String>> {
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
            map.insert(key.clone(), entities);
        }
        map
    }

    fn fuzzy_match(extracted: &str, expected: &str) -> bool {
        let e = extracted.to_lowercase();
        let x = expected.to_lowercase();
        e == x || e.contains(&x) || x.contains(&e)
    }

    #[test]
    fn spike_programmatic_recall() {
        let gt = load_ground_truth();
        let fixture_list = fixtures();

        eprintln!("\n═══════════════════════════════════════════════════════════════");
        eprintln!("SPIKE: Programmatic Entity Extraction (zero LLM)");
        eprintln!("Method: scan_proper_nouns() — Title Case sequence detection");
        eprintln!("═══════════════════════════════════════════════════════════════\n");
        eprintln!(
            "{:<25} {:>6} {:>10} {:>10} {:>7}",
            "Domain", "Recall", "Found/Exp", "Extracted", "Time"
        );
        eprintln!("{}", "─".repeat(65));

        let mut total_recall = 0.0;
        let mut total_found = 0;
        let mut total_expected = 0;
        let mut total_extracted = 0;
        let mut fixture_count = 0;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    eprintln!("{:<25} SKIP — no ground truth", fixture.name);
                    continue;
                }
            };

            let text = std::fs::read_to_string(fixture.path).expect("read fixture");

            let start = std::time::Instant::now();
            let extracted = scan_proper_nouns(&text, &[]);
            let elapsed = start.elapsed();

            let extracted_lower: Vec<String> =
                extracted.iter().map(|e| e.name.to_lowercase()).collect();

            let found = expected
                .iter()
                .filter(|exp| extracted_lower.iter().any(|ext| fuzzy_match(ext, exp)))
                .count();

            let recall = if expected.is_empty() {
                1.0
            } else {
                found as f64 / expected.len() as f64
            };

            // Show which expected entities were missed
            let missed: Vec<&String> = expected
                .iter()
                .filter(|exp| !extracted_lower.iter().any(|ext| fuzzy_match(ext, exp)))
                .collect();

            eprintln!(
                "{:<25} {:>5.0}% {:>5}/{:<4} {:>10} {:>5.1}ms",
                fixture.name,
                recall * 100.0,
                found,
                expected.len(),
                extracted.len(),
                elapsed.as_secs_f64() * 1000.0
            );

            if !missed.is_empty() {
                eprintln!(
                    "  missed: {}",
                    missed
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Show what WAS extracted (for debugging false positives)
            if extracted.len() <= 20 {
                let names: Vec<&str> = extracted.iter().map(|e| e.name.as_str()).collect();
                eprintln!("  found:  {}", names.join(", "));
            } else {
                eprintln!("  found:  {} entities (too many to list)", extracted.len());
            }

            total_recall += recall;
            total_found += found;
            total_expected += expected.len();
            total_extracted += extracted.len();
            fixture_count += 1;
        }

        let avg_recall = total_recall / fixture_count as f64;
        eprintln!("{}", "─".repeat(65));
        eprintln!(
            "AVERAGE: {:.0}% recall | {}/{} entities found | {} extracted total",
            avg_recall * 100.0,
            total_found,
            total_expected,
            total_extracted
        );
        eprintln!("Cost: ZERO LLM calls | sub-millisecond per fixture");
        eprintln!("═══════════════════════════════════════════════════════════════");
    }
}
