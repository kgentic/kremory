#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

//! Spike: Programmatic-First Extraction Architecture Validation
//!
//! Validates: OOV-only candidates (language-agnostic) → two-call LLM enrichment.
//!
//! From spike findings (2026-04-08):
//!   - OOV + two-call LLM = 83% recall, best language-agnostic path
//!   - Scanner omitted: English-only (Title Case heuristic)
//!   - PMI omitted: falsified at single-document scale
//!   - Cap at 20-25 candidates: >25 overloads entity confirmation
//!
//! Parts:
//!   Part 1 — OOV pre-LLM recall baseline (zero LLM, fast, runs in CI)
//!   Part 2 — Real LLM validation (requires --features llm + model path)
//!   Part 3 — Atom-level unit tests
//!
//! Run without LLM (CI):
//!   cargo test -p rql-core --test it spike_programmatic_first:: -- --nocapture
//!
//! Run with real LLM (benchmark):
//!   RQL_QWEN3B_MODEL_PATH=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
//!     cargo test --features llm -p rql-core --test it spike_programmatic_first:: -- --nocapture

// ─── Part 1 + 3: No-LLM tests (CI-safe) ─────────────────────────────────────

mod spike {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use kremory::core::extraction::ProgrammaticFirstExtractor;
    use kremory::core::intelligence::{EntityExtractor, ExtractionContext};
    use kremory::core::provider::MockChatProvider;
    use kremory::core::text_utils::{compute_pmi_bigrams, OovAuditor};

    // ── Fixtures ─────────────────────────────────────────────────────────────

    struct Fixture {
        name: &'static str,
        key: &'static str,
        path: &'static str,
    }

    fn all_fixtures() -> Vec<Fixture> {
        vec![
            Fixture {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
            },
            Fixture {
                name: "Medical Consult",
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
            Fixture {
                name: "Academic Lecture",
                key: "academic_lecture",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/academic_lecture.txt"),
            },
            Fixture {
                name: "Customer Support",
                key: "customer_support",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/customer_support.txt"),
            },
            Fixture {
                name: "Slack Thread",
                key: "slack_thread",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/slack_thread.txt"),
            },
            Fixture {
                name: "Product Review",
                key: "product_review",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/product_review.txt"),
            },
            Fixture {
                name: "Short Snippet",
                key: "short_snippet",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/short_snippet.txt"),
            },
            Fixture {
                name: "Long Report",
                key: "long_report",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/long_report.txt"),
            },
        ]
    }

    fn load_ground_truth() -> HashMap<String, Vec<String>> {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/ground_truth.json"
        ))
        .expect("ground_truth.json");
        let gt: serde_json::Value = serde_json::from_str(&raw).expect("parse ground_truth");
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

    fn load_auditor() -> OovAuditor {
        let aff = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .expect("en_US.aff");
        let dic = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .expect("en_US.dic");
        let dict = zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .expect("build dictionary");
        let stops: HashSet<String> = stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        OovAuditor::new(dict, stops)
    }

    fn fuzzy_match(extracted: &str, expected: &str) -> bool {
        let e = extracted.to_lowercase();
        let x = expected.to_lowercase();
        e == x || e.contains(&x) || x.contains(&e)
    }

    fn recall(candidates: &[String], expected: &[String]) -> (usize, f64) {
        let found = expected
            .iter()
            .filter(|exp| candidates.iter().any(|ext| fuzzy_match(ext, exp)))
            .count();
        let r = if expected.is_empty() {
            1.0
        } else {
            found as f64 / expected.len() as f64
        };
        (found, r)
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 1: OOV Pre-LLM Recall Baseline (zero LLM)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn spike_oov_pre_llm_recall() {
        let auditor = load_auditor();
        let gt = load_ground_truth();
        let fixtures = all_fixtures();

        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("SPIKE: OOV Pre-LLM Recall (language-agnostic, zero LLM)");
        eprintln!("Method: extract_candidates() = OOV singles + OOV runs");
        eprintln!("Expected: 10-50% pre-LLM (LLM boost +33-73pp → ~83% post-LLM)");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");
        eprintln!(
            "{:<22} {:>8} {:>10} {:>8}",
            "Domain", "OOV Rcl", "Found/Exp", "Time"
        );
        eprintln!("{}", "─".repeat(52));

        let mut total_oov_recall = 0.0;
        let mut total_candidates = 0;
        let mut total_time_us = 0u128;
        let mut fixture_count = 0;

        for fixture in &fixtures {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    eprintln!("{:<22} SKIP — no ground truth", fixture.name);
                    continue;
                }
            };

            let text = std::fs::read_to_string(fixture.path).expect("read fixture");

            let start = Instant::now();
            let candidates = auditor.extract_candidates(&text, 25);
            let elapsed = start.elapsed();
            let cand_lower: Vec<String> = candidates.iter().map(|c| c.to_lowercase()).collect();
            let (oov_found, oov_recall) = recall(&cand_lower, expected);

            let missed: Vec<&String> = expected
                .iter()
                .filter(|exp| !cand_lower.iter().any(|ext| fuzzy_match(ext, exp)))
                .collect();

            eprintln!(
                "{:<22} {:>6.0}% {:>5}/{:<4} {:>5.1}ms",
                fixture.name,
                oov_recall * 100.0,
                oov_found,
                expected.len(),
                elapsed.as_secs_f64() * 1000.0,
            );

            if !missed.is_empty() && missed.len() <= 8 {
                eprintln!(
                    "  missed: {}",
                    missed
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            } else if !missed.is_empty() {
                eprintln!("  missed: {} entities", missed.len());
            }
            if !candidates.is_empty() && candidates.len() <= 12 {
                eprintln!(
                    "  OOV:    {}",
                    candidates
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            total_oov_recall += oov_recall;
            total_candidates += candidates.len();
            total_time_us += elapsed.as_micros();
            fixture_count += 1;
        }

        let n = fixture_count as f64;
        let avg_oov = total_oov_recall / n * 100.0;
        let avg_cands = total_candidates as f64 / n;
        let avg_time_ms = total_time_us as f64 / n / 1000.0;

        eprintln!("{}", "─".repeat(52));
        eprintln!(
            "{:<22} {:>6.0}% {:>5.0} avg  {:>5.1}ms",
            "AVERAGE", avg_oov, avg_cands, avg_time_ms,
        );
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!(
            "OOV pre-LLM: {:.0}% | {:.0} candidates avg | {:.1}ms avg",
            avg_oov, avg_cands, avg_time_ms,
        );
        eprintln!("LLM boost expected: +33-73pp → ~83% post-LLM");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 2 (mock): Architecture composition proof — runs in CI
    // ═════════════════════════════════════════════════════════════════════════

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn spike_architecture_composition_mock() {
        let auditor = Arc::new(load_auditor());

        // Single fixture to prove composition works (not for recall measurement)
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/tech_standup.txt"
        ))
        .expect("tech_standup.txt");

        let candidates = auditor.extract_candidates(&text, 25);

        let mut mock_map = HashMap::new();
        let mock_entities: Vec<String> = candidates
            .iter()
            .map(|c| {
                format!(
                    "{{\"name\":\"{}\",\"label\":\"Entity\"}}",
                    c.replace('"', "\\\"")
                )
            })
            .collect();
        mock_map.insert(
            "Confirm which are real entities".to_string(),
            format!("{{\"entities\":[{}]}}", mock_entities.join(",")),
        );
        let ent_names: Vec<&String> = candidates.iter().take(4).collect();
        let mut mock_rels = Vec::new();
        for pair in ent_names.windows(2) {
            mock_rels.push(format!(
                "{{\"subject\":\"{}\",\"predicate\":\"related_to\",\"object\":\"{}\",\"is_entity_ref\":true,\"confidence\":0.8}}",
                pair[0].replace('"', "\\\""), pair[1].replace('"', "\\\""),
            ));
        }
        mock_map.insert(
            "Extract relationships between these entities".to_string(),
            format!("{{\"relationships\":[{}]}}", mock_rels.join(",")),
        );

        let mock = Arc::new(MockChatProvider::new(mock_map));
        let extractor =
            ProgrammaticFirstExtractor::new(Arc::clone(&mock), Arc::clone(&auditor), 25);
        let ctx = ExtractionContext::default();
        let result = block_on(extractor.extract(&text, &ctx)).expect("extract should succeed");

        assert!(
            !result.entities.is_empty(),
            "composition broken: 0 entities from {} candidates",
            candidates.len()
        );
        assert!(
            !result.facts.is_empty(),
            "composition broken: 0 relationships"
        );
        eprintln!(
            "Architecture composition: {} candidates → {} entities, {} relationships ✓",
            candidates.len(),
            result.entities.len(),
            result.facts.len()
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 3: Atom-level unit tests
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn spike_oov_catches_domain_terms() {
        let auditor = load_auditor();
        let candidates = auditor.extract_candidates(
            "The meeting was led by Krishnamurthy from the engineering team.",
            25,
        );
        let lower: Vec<String> = candidates.iter().map(|c| c.to_lowercase()).collect();
        assert!(
            lower.iter().any(|c| c.contains("krishnamurthy")),
            "expected 'Krishnamurthy' as OOV candidate in {:?}",
            candidates
        );
    }

    #[test]
    fn spike_oov_skips_dictionary_words() {
        let auditor = load_auditor();
        let candidates =
            auditor.extract_candidates("The quick brown fox jumps over the lazy dog.", 25);
        assert!(
            candidates.is_empty(),
            "common English words should not be OOV candidates, got {:?}",
            candidates
        );
    }

    #[test]
    fn spike_oov_respects_cap() {
        let auditor = load_auditor();
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/long_report.txt"
        ))
        .expect("long_report.txt");
        let candidates = auditor.extract_candidates(&text, 25);
        assert!(
            candidates.len() <= 25,
            "cap=25 but got {} candidates",
            candidates.len()
        );
    }

    #[test]
    fn spike_oov_empty_text() {
        let auditor = load_auditor();
        assert!(auditor.extract_candidates("", 25).is_empty());
    }

    #[test]
    fn spike_pmi_atom_correctness() {
        // PMI is deferred for entity detection (falsified at doc scale),
        // but the atom should still compute correctly.
        let text = "Alice works at Acme Corp. Bob also joined Acme Corp last year.";
        let results = compute_pmi_bigrams(text, 0.0);
        let names: Vec<&str> = results.iter().map(|(s, _)| s.as_str()).collect();
        assert!(
            names
                .iter()
                .any(|b| b.contains("acme") && b.contains("corp")),
            "expected 'acme corp' bigram in {:?}",
            names
        );
        assert!(compute_pmi_bigrams("", 2.0).is_empty());
        assert!(compute_pmi_bigrams("hello", 0.0).is_empty());
    }
}

// ─── Part 2 (real): LLM validation with Qwen 3B ─────────────────────────────

#[cfg(feature = "llm")]
mod spike_real_llm {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use crate::common::build_llm;
    use crate::common::{ArchitectureBenchmarkRecord, JsonlBenchmarkWriter};
    use autoagents_llamacpp::LlamaCppProvider;
    use kremory::core::provider::{chat_msg_system, chat_msg_user, ChatProvider as _};
    use kremory::core::text_utils::OovAuditor;
    use serde::Deserialize;

    // ── Serde models ─────────────────────────────────────────────────────────

    #[derive(Debug, Deserialize, Default)]
    struct EntityOnlyOutput {
        #[serde(default)]
        entities: Vec<RawEntity>,
    }
    #[derive(Debug, Deserialize, Default)]
    struct RelOnlyOutput {
        #[serde(default)]
        relationships: Vec<RawRel>,
    }
    #[derive(Debug, Deserialize)]
    struct RawEntity {
        #[serde(default)]
        name: String,
        #[allow(dead_code)]
        #[serde(default = "default_label")]
        label: String,
    }
    fn default_label() -> String {
        "Entity".to_string()
    }
    #[derive(Debug, Deserialize)]
    struct RawRel {
        #[serde(default)]
        subject: String,
        #[serde(default)]
        predicate: String,
        #[serde(default)]
        object: String,
    }

    // ── Fixtures ─────────────────────────────────────────────────────────────

    struct Fixture {
        name: &'static str,
        key: &'static str,
        path: &'static str,
    }

    fn all_fixtures() -> Vec<Fixture> {
        vec![
            Fixture {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
            },
            Fixture {
                name: "Medical Consult",
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
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/ground_truth.json"
        ))
        .unwrap();
        let gt: serde_json::Value = serde_json::from_str(&raw).unwrap();
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

    fn load_auditor() -> OovAuditor {
        let aff = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .unwrap();
        let dic = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .unwrap();
        let dict = zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .unwrap();
        let stops: HashSet<String> = stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        OovAuditor::new(dict, stops)
    }

    fn fuzzy_match(a: &str, b: &str) -> bool {
        let a = a.to_lowercase();
        let b = b.to_lowercase();
        a == b || a.contains(&b) || b.contains(&a)
    }

    fn recall_ct(ext: &[String], exp: &[String]) -> (usize, f64) {
        let f = exp
            .iter()
            .filter(|e| ext.iter().any(|x| fuzzy_match(x, e)))
            .count();
        (
            f,
            if exp.is_empty() {
                1.0
            } else {
                f as f64 / exp.len() as f64
            },
        )
    }

    fn dedup_ents(e: &[RawEntity]) -> Vec<String> {
        let mut s = HashSet::new();
        e.iter()
            .filter(|e| !e.name.is_empty())
            .filter(|e| s.insert(e.name.to_lowercase()))
            .map(|e| e.name.to_lowercase())
            .collect()
    }

    fn dedup_rels(r: &[RawRel]) -> Vec<(String, String, String)> {
        let mut s = HashSet::new();
        r.iter()
            .filter(|r| !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty())
            .filter(|r| {
                s.insert((
                    r.subject.to_lowercase(),
                    r.predicate.to_lowercase(),
                    r.object.to_lowercase(),
                ))
            })
            .map(|r| (r.subject.clone(), r.predicate.clone(), r.object.clone()))
            .collect()
    }

    fn parse_json<T: for<'de> Deserialize<'de> + Default>(raw: &str) -> T {
        let t = raw.trim();
        if t.is_empty() {
            return T::default();
        }
        if let Ok(v) = serde_json::from_str::<T>(t) {
            return v;
        }
        let r =
            llm_json::repair_json(t, &llm_json::RepairOptions::default()).unwrap_or(t.to_owned());
        if let Ok(v) = serde_json::from_str::<T>(&r) {
            return v;
        }
        if let (Some(s), Some(e)) = (t.find('{'), t.rfind('}')) {
            if e > s {
                let sl = &t[s..=e];
                let r2 = llm_json::repair_json(sl, &llm_json::RepairOptions::default())
                    .unwrap_or(sl.to_owned());
                if let Ok(v) = serde_json::from_str::<T>(&r2) {
                    return v;
                }
            }
        }
        T::default()
    }

    // ── Prompts ──────────────────────────────────────────────────────────────

    const ET: &str = "Person, Organisation, Location, Technology, Product, Event, Date";
    const SYS: &str = "You are a knowledge graph extraction system. Output valid JSON only.";

    fn prompt_ent(text: &str, cands: &[String]) -> String {
        let l = cands
            .iter()
            .map(|c| format!("\"{}\"", c))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Extract all unique entities from the text below.\n\n\
Some candidate entities detected automatically: [{l}]\n\
Confirm which are real entities, correct errors, add any missing.\n\n\
Rules: each entity ONCE, classify as: {ET}, or Entity.\n\n\
Example: Text: \"Alice from Acme Corp met Bob.\"\nCandidates: [\"Alice\", \"Acme Corp\"]\n\
Output: {{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},{{\"name\":\"Bob\",\"label\":\"Person\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\nOutput JSON with \"entities\" array only."
        )
    }

    fn prompt_rel(text: &str, ents: &[String]) -> String {
        let l = ents
            .iter()
            .map(|e| format!("\"{}\"", e))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Extract relationships between these entities from the text.\n\n\
Entities: [{l}]\n\n\
Rules: unique (subject, predicate, object) triples. Use specific predicates.\n\n\
<TEXT>\n{text}\n</TEXT>\n\nOutput JSON with \"relationships\" array only."
        )
    }

    // ── Two-call LLM pattern ─────────────────────────────────────────────────

    async fn two_call(
        llm: &LlamaCppProvider,
        text: &str,
        candidates: &[String],
    ) -> (Vec<String>, Vec<(String, String, String)>, f64, f64) {
        let t1 = Instant::now();
        let msgs1 = vec![
            chat_msg_system(SYS),
            chat_msg_user(&prompt_ent(text, candidates)),
        ];
        let r1 = llm.chat_with_tools(&msgs1, None, None).await;
        let s1 = t1.elapsed().as_secs_f64();

        let o1: EntityOnlyOutput = match r1 {
            Ok(r) => parse_json(&r.text().unwrap_or_default()),
            Err(_) => EntityOnlyOutput::default(),
        };
        let ents = dedup_ents(&o1.entities);

        let t2 = Instant::now();
        let msgs2 = vec![
            chat_msg_system(SYS),
            chat_msg_user(&prompt_rel(text, &ents)),
        ];
        let r2 = llm.chat_with_tools(&msgs2, None, None).await;
        let s2 = t2.elapsed().as_secs_f64();

        let o2: RelOnlyOutput = match r2 {
            Ok(r) => parse_json(&r.text().unwrap_or_default()),
            Err(_) => RelOnlyOutput::default(),
        };
        let rels = dedup_rels(&o2.relationships);

        (ents, rels, s1, s2)
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 2 (real): OOV + Real LLM Benchmark
    // ═════════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_oov_plus_real_llm() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let auditor = load_auditor();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );
        let metrics_writer = JsonlBenchmarkWriter::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring"),
            "architecture-bakeoff-programmatic-first",
        )
        .expect("open unified benchmark writer");

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: OOV Candidates + Real LLM (Programmatic-First Architecture)");
        eprintln!("Flow: OOV extract_candidates(cap=25) → LLM entity typing → LLM relationships");
        eprintln!("Target: ~83% recall (per spike findings)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>5} {:>6} {:>6} {:>8} {:>5}",
            "Domain", "Ents", "Rels", "Cands", "PreR", "PostR", "Time", "Boost"
        );
        eprintln!("{}", "─".repeat(68));

        let mut total_pre_recall = 0.0;
        let mut total_post_recall = 0.0;
        let mut total_ents = 0;
        let mut total_rels = 0;
        let mut total_time = 0.0f64;
        let mut n = 0;

        for fixture in &all_fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };

            let text = std::fs::read_to_string(fixture.path).unwrap();

            // Step 1: OOV candidates (pre-LLM)
            let candidates = auditor.extract_candidates(&text, 25);
            let cand_lower: Vec<String> = candidates.iter().map(|c| c.to_lowercase()).collect();
            let (_pre_found, pre_recall) = recall_ct(&cand_lower, expected);

            // Step 2: Two-call LLM with OOV candidates
            let (ents, rels, s1, s2) = two_call(&llm, &text, &candidates).await;
            let total_sec = s1 + s2;
            let (_post_found, post_recall) = recall_ct(&ents, expected);

            let boost = post_recall - pre_recall;

            eprintln!(
                "{:<18} {:>6} {:>6} {:>5} {:>5.0}% {:>5.0}% {:>6.1}s {:>+4.0}pp",
                fixture.name,
                ents.len(),
                rels.len(),
                candidates.len(),
                pre_recall * 100.0,
                post_recall * 100.0,
                total_sec,
                boost * 100.0,
            );

            // Show what LLM discovered beyond OOV candidates
            let llm_discovers: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    ents.iter().any(|x| fuzzy_match(x, e))
                        && !cand_lower.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            if !llm_discovers.is_empty() {
                eprintln!(
                    "  LLM found: {}",
                    llm_discovers
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            let still_missed: Vec<&String> = expected
                .iter()
                .filter(|e| !ents.iter().any(|x| fuzzy_match(x, e)))
                .collect();
            if !still_missed.is_empty() && still_missed.len() <= 5 {
                eprintln!(
                    "  missed:    {}",
                    still_missed
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            } else if !still_missed.is_empty() {
                eprintln!("  missed:    {} entities", still_missed.len());
            }

            total_pre_recall += pre_recall;
            total_post_recall += post_recall;
            total_ents += ents.len();
            total_rels += rels.len();
            total_time += total_sec;
            n += 1;

            let record = ArchitectureBenchmarkRecord {
                architecture: "programmatic_first_two_call".to_string(),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall: post_recall,
                entity_count: ents.len(),
                expected_entity_count: expected.len(),
                relationship_count: rels.len(),
                relationship_duplicates: None,
                latency_ms: total_sec * 1000.0,
                document_level: false,
                llm_calls: Some(2),
                candidate_count: Some(candidates.len()),
                pipeline_ms: None,
                parser_ok: Some(true),
                stage_label: Some("programmatic_first".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            record.record_metrics();
            let _ = metrics_writer.append(&record);
        }

        let avg_pre = total_pre_recall / n as f64 * 100.0;
        let avg_post = total_post_recall / n as f64 * 100.0;
        let avg_boost = avg_post - avg_pre;

        eprintln!("{}", "─".repeat(68));
        eprintln!(
            "{:<18} {:>6.0} {:>6.0} {:>5} {:>5.0}% {:>5.0}% {:>6.1}s {:>+4.0}pp",
            "AVERAGE",
            total_ents as f64 / n as f64,
            total_rels as f64 / n as f64,
            "",
            avg_pre,
            avg_post,
            total_time / n as f64,
            avg_boost,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "OOV pre-LLM: {:.0}% → Post-LLM: {:.0}% (boost: +{:.0}pp)",
            avg_pre, avg_post, avg_boost,
        );
        eprintln!(
            "Total: {} entities, {} rels | Avg: {:.1}s/fixture",
            total_ents,
            total_rels,
            total_time / n as f64,
        );
        eprintln!("Unified metrics: {}", metrics_writer.path().display());
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 3 (real): Head-to-Head — SingleCall vs SingleCall+OOV vs OOV-first
    //
    // The key question: does OOV audit after SingleCall LLM close the gap
    // without the latency cost of two-call?
    //
    // A: SingleCall LLM only       — 1 call, ~9s, no language-agnostic safety net
    // B: SingleCall LLM + OOV audit — 1 call + <1ms audit, language-agnostic
    // C: OOV-first + two-call       — 2 calls, ~17s, language-agnostic
    // ═════════════════════════════════════════════════════════════════════════

    /// SingleCall prompt (entities + relationships in one pass)
    fn prompt_single_call(text: &str) -> String {
        format!(
            "Extract all unique entities and relationships from the text below.\n\n\
Rules:\n\
- Each entity must appear ONCE (no duplicates)\n\
- Classify entities as: {ET}, or Entity\n\
- Relationships must be unique triples\n\
- Use specific predicates: \"prescribed\", \"works_at\", \"located_in\", \"manages\", etc.\n\n\
Example input: \"Alice from Acme Corp met Bob. Alice manages the platform team.\"\n\
Example output:\n\
{{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},\
{{\"name\":\"Bob\",\"label\":\"Person\"}},{{\"name\":\"platform team\",\"label\":\"Organisation\"}}],\
\"relationships\":[{{\"subject\":\"Alice\",\"predicate\":\"works_at\",\"object\":\"Acme Corp\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"met\",\"object\":\"Bob\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"manages\",\"object\":\"platform team\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with \"entities\" and \"relationships\" arrays. No duplicates."
        )
    }

    /// NuExtract-style combined output
    #[derive(Debug, Deserialize, Default)]
    struct CombinedOutput {
        #[serde(default)]
        entities: Vec<RawEntity>,
        #[serde(default)]
        relationships: Vec<RawRel>,
    }

    /// Run single-call extraction (1 LLM call, entities + relationships)
    async fn single_call(
        llm: &LlamaCppProvider,
        text: &str,
    ) -> (Vec<String>, Vec<(String, String, String)>, f64) {
        let t = Instant::now();
        let msgs = vec![
            chat_msg_system(SYS),
            chat_msg_user(&prompt_single_call(text)),
        ];
        let r = llm.chat_with_tools(&msgs, None, None).await;
        let elapsed = t.elapsed().as_secs_f64();

        let output: CombinedOutput = match r {
            Ok(r) => parse_json(&r.text().unwrap_or_default()),
            Err(_) => CombinedOutput::default(),
        };

        (
            dedup_ents(&output.entities),
            dedup_rels(&output.relationships),
            elapsed,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_head_to_head_comparison() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let auditor = load_auditor();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );
        let metrics_writer = JsonlBenchmarkWriter::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring"),
            "architecture-bakeoff-head-to-head",
        )
        .expect("open unified benchmark writer");

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: Head-to-Head Comparison (all language-agnostic, no scanner)");
        eprintln!("A = SingleCall LLM only (1 call)");
        eprintln!("B = SingleCall LLM + OOV audit after (1 call + <1ms)");
        eprintln!("C = OOV-first + two-call LLM (2 calls)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>6} {:>7} {:>7} {:>7} {:>8} {:>8}",
            "Domain", "A.Rcl", "B.Rcl", "C.Rcl", "A.Ent", "B.Ent", "C.Ent", "A.sec", "C.sec"
        );
        eprintln!("{}", "─".repeat(82));

        let mut totals_a = (0.0f64, 0usize, 0usize, 0.0f64); // recall, ents, rels, time
        let mut totals_b = (0.0f64, 0usize, 0usize); // recall, ents, rels
        let mut totals_c = (0.0f64, 0usize, 0usize, 0.0f64); // recall, ents, rels, time
        let mut n = 0;

        for fixture in &all_fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };

            let text = std::fs::read_to_string(fixture.path).unwrap();

            // ── A: SingleCall LLM only ───────────────────────────────────────
            let (a_ents, a_rels, a_sec) = single_call(&llm, &text).await;
            let (_, a_recall) = recall_ct(&a_ents, expected);

            // ── B: SingleCall + OOV audit ────────────────────────────────────
            // Same LLM output as A, then add OOV candidates the LLM missed
            let mut b_ents = a_ents.clone();
            let oov_candidates = auditor.extract_candidates(&text, 25);
            for oov in &oov_candidates {
                let lower = oov.to_lowercase();
                if !b_ents.iter().any(|e| fuzzy_match(e, &lower)) {
                    b_ents.push(lower);
                }
            }
            let (_, b_recall) = recall_ct(&b_ents, expected);

            // ── C: OOV-first + two-call LLM ─────────────────────────────────
            let oov_cands = auditor.extract_candidates(&text, 25);
            let (c_ents, c_rels, c_s1, c_s2) = two_call(&llm, &text, &oov_cands).await;
            let c_sec = c_s1 + c_s2;
            let (_, c_recall) = recall_ct(&c_ents, expected);

            eprintln!(
                "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>7} {:>7} {:>7} {:>6.1}s {:>6.1}s",
                fixture.name,
                a_recall * 100.0,
                b_recall * 100.0,
                c_recall * 100.0,
                a_ents.len(),
                b_ents.len(),
                c_ents.len(),
                a_sec,
                c_sec,
            );

            // Show what OOV audit uniquely adds to approach B
            let b_unique: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    b_ents.iter().any(|x| fuzzy_match(x, e))
                        && !a_ents.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            if !b_unique.is_empty() {
                eprintln!(
                    "  B adds: {}",
                    b_unique
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            totals_a.0 += a_recall;
            totals_a.1 += a_ents.len();
            totals_a.2 += a_rels.len();
            totals_a.3 += a_sec;
            totals_b.0 += b_recall;
            totals_b.1 += b_ents.len();
            totals_b.2 += 0; // same rels as A
            totals_c.0 += c_recall;
            totals_c.1 += c_ents.len();
            totals_c.2 += c_rels.len();
            totals_c.3 += c_sec;
            n += 1;

            for (
                architecture,
                entity_recall,
                entity_count,
                relationship_count,
                latency_ms,
                llm_calls,
                candidate_count,
            ) in [
                (
                    "single_call_only",
                    a_recall,
                    a_ents.len(),
                    a_rels.len(),
                    a_sec * 1000.0,
                    Some(1),
                    None,
                ),
                (
                    "single_call_plus_oov_post_audit",
                    b_recall,
                    b_ents.len(),
                    a_rels.len(),
                    a_sec * 1000.0,
                    Some(1),
                    Some(oov_candidates.len()),
                ),
                (
                    "programmatic_first_two_call",
                    c_recall,
                    c_ents.len(),
                    c_rels.len(),
                    c_sec * 1000.0,
                    Some(2),
                    Some(oov_cands.len()),
                ),
            ] {
                let record = ArchitectureBenchmarkRecord {
                    architecture: architecture.to_string(),
                    model: "qwen_3b".to_string(),
                    fixture: fixture.name.to_string(),
                    fixture_key: fixture.key.to_string(),
                    entity_recall,
                    entity_count,
                    expected_entity_count: expected.len(),
                    relationship_count,
                    relationship_duplicates: None,
                    latency_ms,
                    document_level: false,
                    llm_calls,
                    candidate_count,
                    pipeline_ms: None,
                    parser_ok: Some(true),
                    stage_label: Some("head_to_head".to_string()),
                    timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
                };
                record.record_metrics();
                let _ = metrics_writer.append(&record);
            }
        }

        let nf = n as f64;
        let avg_a = totals_a.0 / nf * 100.0;
        let avg_b = totals_b.0 / nf * 100.0;
        let avg_c = totals_c.0 / nf * 100.0;

        eprintln!("{}", "─".repeat(82));
        eprintln!(
            "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>7.0} {:>7.0} {:>7.0} {:>6.1}s {:>6.1}s",
            "AVERAGE",
            avg_a,
            avg_b,
            avg_c,
            totals_a.1 as f64 / nf,
            totals_b.1 as f64 / nf,
            totals_c.1 as f64 / nf,
            totals_a.3 / nf,
            totals_c.3 / nf,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "A (SingleCall only):      {:.0}% recall, {:.1}s avg, 1 LLM call",
            avg_a,
            totals_a.3 / nf
        );
        eprintln!(
            "B (SingleCall + OOV):     {:.0}% recall, {:.1}s avg, 1 LLM call + <1ms OOV",
            avg_b,
            totals_a.3 / nf
        );
        eprintln!(
            "C (OOV-first + two-call): {:.0}% recall, {:.1}s avg, 2 LLM calls",
            avg_c,
            totals_c.3 / nf
        );
        eprintln!("B-A delta: {:+.0}pp (OOV audit adds)", avg_b - avg_a);
        eprintln!("Unified metrics: {}", metrics_writer.path().display());
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 4 (real): OOV hints → SingleCall (1 call, OOV pre-injected)
    //
    // D = OOV candidates injected as known_entities hints into SingleCall prompt
    //     Still 1 LLM call. LLM sees hints, types them, extracts relationships.
    //     Compare to A (no hints) and B (hints post-LLM, untyped orphans).
    // ═════════════════════════════════════════════════════════════════════════

    /// SingleCall prompt WITH OOV hints injected
    fn prompt_single_call_with_hints(text: &str, oov_hints: &[String]) -> String {
        let hint_section = if oov_hints.is_empty() {
            String::new()
        } else {
            let list = oov_hints
                .iter()
                .map(|h| format!("\"{}\"", h))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "\nCandidate entities detected programmatically (confirm, type, and include relationships for these): [{}]\n",
                list
            )
        };

        format!(
            "Extract all unique entities and relationships from the text below.\n\n\
Rules:\n\
- Each entity must appear ONCE (no duplicates)\n\
- Classify entities as: {ET}, or Entity\n\
- Relationships must be unique triples\n\
- Use specific predicates: \"prescribed\", \"works_at\", \"located_in\", \"manages\", etc.\n\
{hint_section}\n\
Example input: \"Alice from Acme Corp met Bob. Alice manages the platform team.\"\n\
Example output:\n\
{{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},\
{{\"name\":\"Bob\",\"label\":\"Person\"}},{{\"name\":\"platform team\",\"label\":\"Organisation\"}}],\
\"relationships\":[{{\"subject\":\"Alice\",\"predicate\":\"works_at\",\"object\":\"Acme Corp\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"met\",\"object\":\"Bob\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"manages\",\"object\":\"platform team\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with \"entities\" and \"relationships\" arrays. No duplicates."
        )
    }

    /// SingleCall with OOV hints pre-injected (1 LLM call)
    async fn single_call_with_hints(
        llm: &LlamaCppProvider,
        text: &str,
        oov_hints: &[String],
    ) -> (Vec<String>, Vec<(String, String, String)>, f64) {
        let t = Instant::now();
        let msgs = vec![
            chat_msg_system(SYS),
            chat_msg_user(&prompt_single_call_with_hints(text, oov_hints)),
        ];
        let r = llm.chat_with_tools(&msgs, None, None).await;
        let elapsed = t.elapsed().as_secs_f64();

        let output: CombinedOutput = match r {
            Ok(r) => parse_json(&r.text().unwrap_or_default()),
            Err(_) => CombinedOutput::default(),
        };

        (
            dedup_ents(&output.entities),
            dedup_rels(&output.relationships),
            elapsed,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_single_call_with_oov_hints() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let auditor = load_auditor();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: SingleCall + OOV Hints (1 LLM call, OOV pre-injected)");
        eprintln!("A = SingleCall only (baseline)");
        eprintln!("D = SingleCall with OOV hints in prompt (same 1 call)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>5}",
            "Domain",
            "A.Rcl",
            "D.Rcl",
            "A.Ent",
            "D.Ent",
            "A.Rel",
            "D.Rel",
            "A.sec",
            "D.sec",
            "Hints"
        );
        eprintln!("{}", "─".repeat(88));

        let mut totals_a = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut totals_d = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut n = 0;

        for fixture in &all_fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };

            let text = std::fs::read_to_string(fixture.path).unwrap();

            // OOV candidates (pre-LLM, <1ms)
            let oov_hints = auditor.extract_candidates(&text, 25);

            // A: SingleCall without hints
            let (a_ents, a_rels, a_sec) = single_call(&llm, &text).await;
            let (_, a_recall) = recall_ct(&a_ents, expected);

            // D: SingleCall WITH OOV hints
            let (d_ents, d_rels, d_sec) = single_call_with_hints(&llm, &text, &oov_hints).await;
            let (_, d_recall) = recall_ct(&d_ents, expected);

            eprintln!(
                "{:<18} {:>5.0}% {:>5.0}% {:>6} {:>6} {:>6} {:>6} {:>6.1}s {:>6.1}s {:>5}",
                fixture.name,
                a_recall * 100.0,
                d_recall * 100.0,
                a_ents.len(),
                d_ents.len(),
                a_rels.len(),
                d_rels.len(),
                a_sec,
                d_sec,
                oov_hints.len(),
            );

            // Show what D finds that A misses (OOV hints that worked)
            let d_wins: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    d_ents.iter().any(|x| fuzzy_match(x, e))
                        && !a_ents.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            if !d_wins.is_empty() {
                eprintln!(
                    "  D wins: {}",
                    d_wins
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Show what A finds that D misses (hints hurt?)
            let a_wins: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    a_ents.iter().any(|x| fuzzy_match(x, e))
                        && !d_ents.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            if !a_wins.is_empty() {
                eprintln!(
                    "  A wins: {}",
                    a_wins
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            totals_a.0 += a_recall;
            totals_a.1 += a_ents.len();
            totals_a.2 += a_rels.len();
            totals_a.3 += a_sec;
            totals_d.0 += d_recall;
            totals_d.1 += d_ents.len();
            totals_d.2 += d_rels.len();
            totals_d.3 += d_sec;
            n += 1;
        }

        let nf = n as f64;
        let avg_a = totals_a.0 / nf * 100.0;
        let avg_d = totals_d.0 / nf * 100.0;

        eprintln!("{}", "─".repeat(88));
        eprintln!(
            "{:<18} {:>5.0}% {:>5.0}% {:>6.0} {:>6.0} {:>6.0} {:>6.0} {:>6.1}s {:>6.1}s",
            "AVERAGE",
            avg_a,
            avg_d,
            totals_a.1 as f64 / nf,
            totals_d.1 as f64 / nf,
            totals_a.2 as f64 / nf,
            totals_d.2 as f64 / nf,
            totals_a.3 / nf,
            totals_d.3 / nf,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "A (SingleCall only):       {:.0}% recall, {:.1}s avg, {} rels total",
            avg_a,
            totals_a.3 / nf,
            totals_a.2
        );
        eprintln!(
            "D (SingleCall + OOV hints): {:.0}% recall, {:.1}s avg, {} rels total",
            avg_d,
            totals_d.3 / nf,
            totals_d.2
        );
        eprintln!(
            "D-A delta: {:+.0}pp recall, {:+.1}s latency",
            avg_d - avg_a,
            totals_d.3 / nf - totals_a.3 / nf
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 5 (real): Approach E — SingleCall + OOV audit + targeted typing
    //
    // The hypothesis: B's orphan problem (untyped, no relationships) can be
    // solved with one small targeted LLM call that only types + relates the
    // few OOV entities the LLM missed. This should be much cheaper than C's
    // full two-call pattern because:
    //   - The first SingleCall already did the heavy lifting
    //   - Only 2-5 orphan entities need typing (not 25 candidates)
    //   - Output is tiny: a few type labels + a few relationships
    //
    // E = SingleCall (entities+rels) → OOV audit → typing call for orphans only
    // Compare: A (1 call), B (1 call + untyped), E (1+1 calls, all typed)
    // ═════════════════════════════════════════════════════════════════════════

    /// Prompt for typing OOV orphans + extracting their relationships.
    /// The LLM receives the full text, already-known entities (with types),
    /// and the orphan entities to classify. Output: typed entities + relationships.
    fn prompt_type_orphans(text: &str, orphans: &[String], known: &[(String, String)]) -> String {
        let known_list = known
            .iter()
            .map(|(name, label)| format!("{} ({})", name, label))
            .collect::<Vec<_>>()
            .join(", ");

        let orphan_list = orphans
            .iter()
            .map(|o| format!("\"{}\"", o))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "You have already extracted these entities from the text:\n\
{known_list}\n\n\
A programmatic scan found additional terms that may be entities: [{orphan_list}]\n\n\
For each candidate that IS a real entity in the text:\n\
1. Classify it as: {ET}, or Entity\n\
2. Extract any relationships between it and the known entities above\n\
Discard candidates that are not meaningful entities (typos, common words, abbreviations).\n\n\
Rules:\n\
- Each entity ONCE, no duplicates\n\
- Unique (subject, predicate, object) triples with specific predicates\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with \"entities\" (the confirmed new entities, typed) \
and \"relationships\" (between new and known entities) arrays."
        )
    }

    /// Run targeted orphan-typing call (small output, max_tokens=128)
    async fn type_orphans(
        llm: &LlamaCppProvider,
        text: &str,
        orphans: &[String],
        known: &[(String, String)],
    ) -> (Vec<RawEntity>, Vec<RawRel>, f64) {
        if orphans.is_empty() {
            return (vec![], vec![], 0.0);
        }

        let t = Instant::now();
        let msgs = vec![
            chat_msg_system(SYS),
            chat_msg_user(&prompt_type_orphans(text, orphans, known)),
        ];
        let r = llm.chat_with_tools(&msgs, None, None).await;
        let elapsed = t.elapsed().as_secs_f64();

        let output: CombinedOutput = match r {
            Ok(r) => parse_json(&r.text().unwrap_or_default()),
            Err(_) => CombinedOutput::default(),
        };

        (output.entities, output.relationships, elapsed)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_approach_e_oov_type_orphans() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let auditor = load_auditor();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: Approach E — SingleCall + OOV Audit + Targeted Orphan Typing");
        eprintln!("A = SingleCall only (1 call, baseline)");
        eprintln!("B = SingleCall + OOV audit (1 call + <1ms, untyped orphans)");
        eprintln!("E = SingleCall + OOV audit + typing call (1+1 calls, all typed)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5}",
            "Domain",
            "A.Rcl",
            "B.Rcl",
            "E.Rcl",
            "A.Ent",
            "E.Ent",
            "A.Rel",
            "E.Rel",
            "A.sec",
            "E.sec",
            "Orph"
        );
        eprintln!("{}", "─".repeat(100));

        let mut totals_a = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut totals_b = (0.0f64, 0usize);
        let mut totals_e = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut total_orphans = 0usize;
        let mut n = 0;

        for fixture in &all_fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };

            let text = std::fs::read_to_string(fixture.path).unwrap();

            // ── A: SingleCall LLM only ───────────────────────────────────────
            let (a_ents, a_rels, a_sec) = single_call(&llm, &text).await;
            let (_, a_recall) = recall_ct(&a_ents, expected);

            // ── B: SingleCall + OOV audit (untyped) ─────────────────────────
            let oov_candidates = auditor.extract_candidates(&text, 25);
            let mut b_ents = a_ents.clone();
            for oov in &oov_candidates {
                let lower = oov.to_lowercase();
                if !b_ents.iter().any(|e| fuzzy_match(e, &lower)) {
                    b_ents.push(lower);
                }
            }
            let (_, b_recall) = recall_ct(&b_ents, expected);

            // ── E: SingleCall + OOV audit + typing call for orphans ─────────
            // Find orphans: OOV candidates NOT already in the SingleCall output
            let orphans: Vec<String> = oov_candidates
                .iter()
                .filter(|oov| {
                    let lower = oov.to_lowercase();
                    !a_ents.iter().any(|e| fuzzy_match(e, &lower))
                })
                .cloned()
                .collect();

            // Build known entity list (name, label) from the SingleCall raw output
            // Re-run SingleCall to get raw entities with labels for the known list
            // (we reuse a_ents which are lowercased names; for the prompt we need labels)
            // Approach: re-parse the SingleCall output is wasteful. Instead, build
            // known list from a_ents with "Entity" as fallback label. The LLM already
            // typed them in its output — but we only have lowercased names here.
            // For the typing prompt, approximate labels are fine (the LLM sees full text).
            let known_for_prompt: Vec<(String, String)> = a_ents
                .iter()
                .map(|name| (name.clone(), "Entity".to_string()))
                .collect();

            let (typed_ents_raw, typed_rels_raw, typing_sec) =
                type_orphans(&llm, &text, &orphans, &known_for_prompt).await;

            // Merge: start with A's entities, add typed orphans
            let mut e_ents = a_ents.clone();
            let typed_ent_names = dedup_ents(&typed_ents_raw);
            for name in &typed_ent_names {
                if !e_ents.iter().any(|e| fuzzy_match(e, name)) {
                    e_ents.push(name.clone());
                }
            }

            // Merge relationships: A's rels + typed orphan rels
            let mut e_rels = a_rels.clone();
            let typed_rel_triples = dedup_rels(&typed_rels_raw);
            for rel in &typed_rel_triples {
                let key = (
                    rel.0.to_lowercase(),
                    rel.1.to_lowercase(),
                    rel.2.to_lowercase(),
                );
                if !e_rels
                    .iter()
                    .any(|r| (r.0.to_lowercase(), r.1.to_lowercase(), r.2.to_lowercase()) == key)
                {
                    e_rels.push(rel.clone());
                }
            }

            let e_sec = a_sec + typing_sec;
            let (_, e_recall) = recall_ct(&e_ents, expected);

            eprintln!(
                "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>6} {:>6} {:>6} {:>6} {:>6.1}s {:>6.1}s {:>5}",
                fixture.name,
                a_recall * 100.0,
                b_recall * 100.0,
                e_recall * 100.0,
                a_ents.len(),
                e_ents.len(),
                a_rels.len(),
                e_rels.len(),
                a_sec,
                e_sec,
                orphans.len(),
            );

            // Show what the typing call confirmed as real entities
            if !typed_ent_names.is_empty() {
                eprintln!(
                    "  typed:   {} → {}",
                    orphans.join(", "),
                    typed_ent_names.join(", "),
                );
            } else if !orphans.is_empty() {
                eprintln!(
                    "  discarded: {} (LLM rejected all orphans)",
                    orphans.join(", "),
                );
            }

            // Show new relationships from the typing call
            if !typed_rel_triples.is_empty() {
                for (s, p, o) in &typed_rel_triples {
                    eprintln!("  +rel:    {} → {} → {}", s, p, o);
                }
            }

            // Show what E still misses
            let e_missed: Vec<&String> = expected
                .iter()
                .filter(|e| !e_ents.iter().any(|x| fuzzy_match(x, e)))
                .collect();
            if !e_missed.is_empty() && e_missed.len() <= 5 {
                eprintln!(
                    "  missed:  {}",
                    e_missed
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            totals_a.0 += a_recall;
            totals_a.1 += a_ents.len();
            totals_a.2 += a_rels.len();
            totals_a.3 += a_sec;
            totals_b.0 += b_recall;
            totals_b.1 += b_ents.len();
            totals_e.0 += e_recall;
            totals_e.1 += e_ents.len();
            totals_e.2 += e_rels.len();
            totals_e.3 += e_sec;
            total_orphans += orphans.len();
            n += 1;
        }

        let nf = n as f64;
        let avg_a = totals_a.0 / nf * 100.0;
        let avg_b = totals_b.0 / nf * 100.0;
        let avg_e = totals_e.0 / nf * 100.0;

        eprintln!("{}", "─".repeat(100));
        eprintln!(
            "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>6.0} {:>6.0} {:>6.0} {:>6.0} {:>6.1}s {:>6.1}s {:>5.1}",
            "AVERAGE",
            avg_a,
            avg_b,
            avg_e,
            totals_a.1 as f64 / nf,
            totals_e.1 as f64 / nf,
            totals_a.2 as f64 / nf,
            totals_e.2 as f64 / nf,
            totals_a.3 / nf,
            totals_e.3 / nf,
            total_orphans as f64 / nf,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "A (SingleCall only):           {:.0}% recall, {:.1}s avg, {} rels, 1 LLM call",
            avg_a,
            totals_a.3 / nf,
            totals_a.2
        );
        eprintln!(
            "B (SingleCall + OOV untyped):  {:.0}% recall, {:.1}s avg, {} rels, 1 call + <1ms (orphans UNTYPED)",
            avg_b,
            totals_a.3 / nf,
            totals_a.2
        );
        eprintln!(
            "E (SingleCall + OOV + typing): {:.0}% recall, {:.1}s avg, {} rels, 1+1 calls (orphans TYPED)",
            avg_e,
            totals_e.3 / nf,
            totals_e.2
        );
        eprintln!(
            "E-A delta: {:+.0}pp recall, {:+.1}s latency, {:+} rels",
            avg_e - avg_a,
            totals_e.3 / nf - totals_a.3 / nf,
            totals_e.2 as isize - totals_a.2 as isize
        );
        eprintln!("E vs B:    same recall target but entities are TYPED with RELATIONSHIPS");
        eprintln!("Orphans per fixture: {:.1} avg", total_orphans as f64 / nf);
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════════════════════\n"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 6 (real): Approach F — Additive typing (B's recall + E's rels)
    //
    // E's problem: the typing call rejects valid OOV entities (Kubernetes,
    // Fenbrook, Priya Mehta), losing 3pp recall vs B.
    //
    // Fix: two changes from E:
    //   1. Prompt says "type ALL of these" — no discard instruction
    //   2. Entity list starts from B (A + raw OOV) — typing call can only
    //      ADD types + relationships, never subtract entities
    //
    // F = SingleCall + OOV audit (keep all) + typing call (additive only)
    // Expected: B's recall (82%) + E's relationship gain (+13 rels)
    // ═════════════════════════════════════════════════════════════════════════

    /// Non-rejecting prompt: type ALL orphans, extract relationships.
    /// No "discard" instruction — every candidate gets a type.
    fn prompt_type_all_orphans(
        text: &str,
        orphans: &[String],
        known: &[(String, String)],
    ) -> String {
        let known_list = known
            .iter()
            .map(|(name, label)| format!("{} ({})", name, label))
            .collect::<Vec<_>>()
            .join(", ");

        let orphan_list = orphans
            .iter()
            .map(|o| format!("\"{}\"", o))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "You have already extracted these entities from the text:\n\
{known_list}\n\n\
A programmatic scan found these additional entities: [{orphan_list}]\n\n\
For EACH entity in the list above:\n\
1. Classify it as: {ET}, or Entity\n\
2. Extract any relationships between it and the known entities\n\
Include ALL candidates — they have been confirmed by the scanner.\n\n\
Rules:\n\
- Each entity ONCE, no duplicates\n\
- Unique (subject, predicate, object) triples with specific predicates\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with \"entities\" (all candidates, typed) \
and \"relationships\" (between new and known entities) arrays."
        )
    }

    /// Run non-rejecting orphan-typing call
    async fn type_all_orphans(
        llm: &LlamaCppProvider,
        text: &str,
        orphans: &[String],
        known: &[(String, String)],
    ) -> (Vec<RawEntity>, Vec<RawRel>, f64) {
        if orphans.is_empty() {
            return (vec![], vec![], 0.0);
        }

        let t = Instant::now();
        let msgs = vec![
            chat_msg_system(SYS),
            chat_msg_user(&prompt_type_all_orphans(text, orphans, known)),
        ];
        let r = llm.chat_with_tools(&msgs, None, None).await;
        let elapsed = t.elapsed().as_secs_f64();

        let output: CombinedOutput = match r {
            Ok(r) => parse_json(&r.text().unwrap_or_default()),
            Err(_) => CombinedOutput::default(),
        };

        (output.entities, output.relationships, elapsed)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_approach_f_additive_typing() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let auditor = load_auditor();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: Approach F — Additive Typing (B's recall + typing call rels)");
        eprintln!("A = SingleCall only (baseline)");
        eprintln!("B = SingleCall + OOV audit (untyped orphans, best recall)");
        eprintln!("E = SingleCall + OOV + typing (LLM rejects some → loses recall)");
        eprintln!("F = SingleCall + OOV (keep all) + typing (additive only, no rejection)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5}",
            "Domain",
            "A.Rcl",
            "B.Rcl",
            "F.Rcl",
            "A.Ent",
            "F.Ent",
            "A.Rel",
            "F.Rel",
            "A.sec",
            "F.sec",
            "Orph"
        );
        eprintln!("{}", "─".repeat(100));

        let mut totals_a = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut totals_b = (0.0f64, 0usize);
        let mut totals_f = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut total_orphans = 0usize;
        let mut total_typed = 0usize;
        let mut n = 0;

        for fixture in &all_fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };

            let text = std::fs::read_to_string(fixture.path).unwrap();

            // ── A: SingleCall LLM only ───────────────────────────────────────
            let (a_ents, a_rels, a_sec) = single_call(&llm, &text).await;
            let (_, a_recall) = recall_ct(&a_ents, expected);

            // ── B: SingleCall + OOV audit (untyped) ─────────────────────────
            let oov_candidates = auditor.extract_candidates(&text, 25);
            let mut b_ents = a_ents.clone();
            for oov in &oov_candidates {
                let lower = oov.to_lowercase();
                if !b_ents.iter().any(|e| fuzzy_match(e, &lower)) {
                    b_ents.push(lower);
                }
            }
            let (_, b_recall) = recall_ct(&b_ents, expected);

            // ── F: Additive typing ──────────────────────────────────────────
            // Step 1: Find orphans (same as E)
            let orphans: Vec<String> = oov_candidates
                .iter()
                .filter(|oov| {
                    let lower = oov.to_lowercase();
                    !a_ents.iter().any(|e| fuzzy_match(e, &lower))
                })
                .cloned()
                .collect();

            // Step 2: Build known list for prompt
            let known_for_prompt: Vec<(String, String)> = a_ents
                .iter()
                .map(|name| (name.clone(), "Entity".to_string()))
                .collect();

            // Step 3: Non-rejecting typing call
            let (typed_ents_raw, typed_rels_raw, typing_sec) =
                type_all_orphans(&llm, &text, &orphans, &known_for_prompt).await;

            // Step 4: Start from B's entity list (guarantees B's recall)
            let mut f_ents = b_ents.clone();
            // Add any entities the typing call returned that aren't already in B
            let typed_ent_names = dedup_ents(&typed_ents_raw);
            for name in &typed_ent_names {
                if !f_ents.iter().any(|e| fuzzy_match(e, name)) {
                    f_ents.push(name.clone());
                }
            }

            // Step 5: Merge relationships (A's rels + typing call rels)
            let mut f_rels = a_rels.clone();
            let typed_rel_triples = dedup_rels(&typed_rels_raw);
            for rel in &typed_rel_triples {
                let key = (
                    rel.0.to_lowercase(),
                    rel.1.to_lowercase(),
                    rel.2.to_lowercase(),
                );
                if !f_rels
                    .iter()
                    .any(|r| (r.0.to_lowercase(), r.1.to_lowercase(), r.2.to_lowercase()) == key)
                {
                    f_rels.push(rel.clone());
                }
            }

            let f_sec = a_sec + typing_sec;
            let (_, f_recall) = recall_ct(&f_ents, expected);

            eprintln!(
                "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>6} {:>6} {:>6} {:>6} {:>6.1}s {:>6.1}s {:>5}",
                fixture.name,
                a_recall * 100.0,
                b_recall * 100.0,
                f_recall * 100.0,
                a_ents.len(),
                f_ents.len(),
                a_rels.len(),
                f_rels.len(),
                a_sec,
                f_sec,
                orphans.len(),
            );

            // Show what the typing call returned
            if !typed_ent_names.is_empty() {
                eprintln!("  typed:   {}", typed_ent_names.join(", "),);
                total_typed += typed_ent_names.len();
            }

            // Show new relationships from the typing call
            if !typed_rel_triples.is_empty() {
                for (s, p, o) in &typed_rel_triples {
                    eprintln!("  +rel:    {} → {} → {}", s, p, o);
                }
            }

            // Show what F still misses
            let f_missed: Vec<&String> = expected
                .iter()
                .filter(|e| !f_ents.iter().any(|x| fuzzy_match(x, e)))
                .collect();
            if !f_missed.is_empty() && f_missed.len() <= 5 {
                eprintln!(
                    "  missed:  {}",
                    f_missed
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            totals_a.0 += a_recall;
            totals_a.1 += a_ents.len();
            totals_a.2 += a_rels.len();
            totals_a.3 += a_sec;
            totals_b.0 += b_recall;
            totals_b.1 += b_ents.len();
            totals_f.0 += f_recall;
            totals_f.1 += f_ents.len();
            totals_f.2 += f_rels.len();
            totals_f.3 += f_sec;
            total_orphans += orphans.len();
            n += 1;
        }

        let nf = n as f64;
        let avg_a = totals_a.0 / nf * 100.0;
        let avg_b = totals_b.0 / nf * 100.0;
        let avg_f = totals_f.0 / nf * 100.0;

        eprintln!("{}", "─".repeat(100));
        eprintln!(
            "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>6.0} {:>6.0} {:>6.0} {:>6.0} {:>6.1}s {:>6.1}s {:>5.1}",
            "AVERAGE",
            avg_a,
            avg_b,
            avg_f,
            totals_a.1 as f64 / nf,
            totals_f.1 as f64 / nf,
            totals_a.2 as f64 / nf,
            totals_f.2 as f64 / nf,
            totals_a.3 / nf,
            totals_f.3 / nf,
            total_orphans as f64 / nf,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "A (SingleCall only):           {:.0}% recall, {:.1}s avg, {} rels, 1 LLM call",
            avg_a,
            totals_a.3 / nf,
            totals_a.2
        );
        eprintln!(
            "B (SingleCall + OOV untyped):  {:.0}% recall, {:.1}s avg, {} rels, 1 call + <1ms (orphans UNTYPED)",
            avg_b,
            totals_a.3 / nf,
            totals_a.2
        );
        eprintln!(
            "F (SingleCall + OOV + additive): {:.0}% recall, {:.1}s avg, {} rels, 1+1 calls (orphans TYPED)",
            avg_f,
            totals_f.3 / nf,
            totals_f.2
        );
        eprintln!(
            "F-A delta: {:+.0}pp recall, {:+.1}s latency, {:+} rels",
            avg_f - avg_a,
            totals_f.3 / nf - totals_a.3 / nf,
            totals_f.2 as isize - totals_a.2 as isize
        );
        eprintln!("F vs B:    F ≥ B recall (guaranteed), + typed entities + relationships");
        eprintln!(
            "Orphans: {:.1} avg/fixture | Typed by LLM: {} total",
            total_orphans as f64 / nf,
            total_typed
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════════════════════\n"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Part 7 (real): Phi-4-mini optimised prompts + diagnostic logging
    //
    // Phi-4-mini (3.8B) failed with Qwen-tuned prompts:
    //   - Sales Call: 0% recall (empty output)
    //   - Typing call: 0 typed entities across all fixtures
    //   - 22% slower than Qwen despite being "4.2x faster"
    //
    // Hypothesis: Phi-4-mini needs tighter, more explicit prompts:
    //   - Explicit JSON schema (not just an example)
    //   - Shorter instructions (fewer rules = less confusion)
    //   - Structured delimiters Phi-4-mini was trained on
    //
    // This test logs raw LLM output for diagnosis + uses new prompts.
    // ═════════════════════════════════════════════════════════════════════════

    /// Phi-4-mini optimised SingleCall prompt: tighter, schema-first
    fn prompt_phi4_single_call(text: &str) -> String {
        format!(
            "Extract entities and relationships from this text as JSON.\n\n\
Output format:\n\
{{\"entities\":[{{\"name\":\"...\",\"label\":\"Person|Organisation|Location|Technology|Product|Event|Date|Entity\"}}],\
\"relationships\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]}}\n\n\
Rules: no duplicates, specific predicates (works_at, manages, located_in, prescribed, etc).\n\n\
Text:\n{text}\n\nJSON:"
        )
    }

    /// Phi-4-mini optimised typing prompt: explicit, no ambiguity
    fn prompt_phi4_type_orphans(text: &str, orphans: &[String], known: &[String]) -> String {
        let orphan_list = orphans.join("\", \"");
        let known_list = known.join(", ");

        format!(
            "Known entities already extracted: [{known_list}]\n\n\
New entities to classify: [\"{orphan_list}\"]\n\n\
For each new entity, output its type and any relationships to the known entities.\n\n\
Output format:\n\
{{\"entities\":[{{\"name\":\"...\",\"label\":\"Person|Organisation|Location|Technology|Product|Event|Date|Entity\"}}],\
\"relationships\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]}}\n\n\
Text:\n{text}\n\nJSON:"
        )
    }

    /// SingleCall with raw output logging
    async fn phi4_single_call(
        llm: &LlamaCppProvider,
        text: &str,
        fixture_name: &str,
    ) -> (Vec<String>, Vec<(String, String, String)>, f64) {
        let t = Instant::now();
        let msgs = vec![
            chat_msg_system("You extract entities and relationships as JSON. Output only valid JSON, no commentary."),
            chat_msg_user(&prompt_phi4_single_call(text)),
        ];
        let r = llm.chat_with_tools(&msgs, None, None).await;
        let elapsed = t.elapsed().as_secs_f64();

        match r {
            Ok(ref resp) => {
                let raw_text = resp.text().unwrap_or_default();
                let trimmed = raw_text.trim();
                // Log raw output (truncated) for diagnosis
                let preview = if trimmed.len() > 200 {
                    &trimmed[..200]
                } else {
                    trimmed
                };
                eprintln!(
                    "  [{}] raw({} chars): {}",
                    fixture_name,
                    trimmed.len(),
                    preview
                );

                let output: CombinedOutput = parse_json(&raw_text);
                (
                    dedup_ents(&output.entities),
                    dedup_rels(&output.relationships),
                    elapsed,
                )
            }
            Err(ref e) => {
                eprintln!("  [{}] LLM ERROR: {}", fixture_name, e);
                (vec![], vec![], elapsed)
            }
        }
    }

    /// Typing call with raw output logging
    async fn phi4_type_orphans(
        llm: &LlamaCppProvider,
        text: &str,
        orphans: &[String],
        known: &[String],
        fixture_name: &str,
    ) -> (Vec<RawEntity>, Vec<RawRel>, f64) {
        if orphans.is_empty() {
            return (vec![], vec![], 0.0);
        }

        let t = Instant::now();
        let msgs = vec![
            chat_msg_system("You classify entities and extract relationships as JSON. Output only valid JSON, no commentary."),
            chat_msg_user(&prompt_phi4_type_orphans(text, orphans, known)),
        ];
        let r = llm.chat_with_tools(&msgs, None, None).await;
        let elapsed = t.elapsed().as_secs_f64();

        match r {
            Ok(ref resp) => {
                let raw_text = resp.text().unwrap_or_default();
                let trimmed = raw_text.trim();
                let preview = if trimmed.len() > 300 {
                    &trimmed[..300]
                } else {
                    trimmed
                };
                eprintln!(
                    "  [{}] typing raw({} chars): {}",
                    fixture_name,
                    trimmed.len(),
                    preview
                );

                let output: CombinedOutput = parse_json(&raw_text);
                (output.entities, output.relationships, elapsed)
            }
            Err(ref e) => {
                eprintln!("  [{}] typing LLM ERROR: {}", fixture_name, e);
                (vec![], vec![], elapsed)
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_phi4_optimised_prompts() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let auditor = load_auditor();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: Phi-4-mini Optimised Prompts (Approach F with model-tuned prompts)");
        eprintln!("G = Phi4 SingleCall + OOV (keep all) + Phi4 typing (additive)");
        eprintln!("Raw LLM output logged for diagnosis");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5}",
            "Domain", "A.Rcl", "B.Rcl", "G.Rcl", "G.Ent", "G.Rel", "G.sec", "Orph"
        );
        eprintln!("{}", "─".repeat(72));

        let mut totals_a = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut totals_b = (0.0f64, 0usize);
        let mut totals_g = (0.0f64, 0usize, 0usize, 0.0f64);
        let mut total_orphans = 0usize;
        let mut total_typed = 0usize;
        let mut n = 0;

        for fixture in &all_fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };

            let text = std::fs::read_to_string(fixture.path).unwrap();

            // ── A/G: Phi4 SingleCall ─────────────────────────────────────────
            let (a_ents, a_rels, a_sec) = phi4_single_call(&llm, &text, fixture.name).await;
            let (_, a_recall) = recall_ct(&a_ents, expected);

            // ── B: + OOV audit (untyped) ────────────────────────────────────
            let oov_candidates = auditor.extract_candidates(&text, 25);
            let mut b_ents = a_ents.clone();
            for oov in &oov_candidates {
                let lower = oov.to_lowercase();
                if !b_ents.iter().any(|e| fuzzy_match(e, &lower)) {
                    b_ents.push(lower);
                }
            }
            let (_, b_recall) = recall_ct(&b_ents, expected);

            // ── G: Additive typing with Phi4 prompt ─────────────────────────
            let orphans: Vec<String> = oov_candidates
                .iter()
                .filter(|oov| {
                    let lower = oov.to_lowercase();
                    !a_ents.iter().any(|e| fuzzy_match(e, &lower))
                })
                .cloned()
                .collect();

            let known_names: Vec<String> = a_ents.clone();

            let (typed_ents_raw, typed_rels_raw, typing_sec) =
                phi4_type_orphans(&llm, &text, &orphans, &known_names, fixture.name).await;

            // Start from B's entity list (guarantees B's recall)
            let mut g_ents = b_ents.clone();
            let typed_ent_names = dedup_ents(&typed_ents_raw);
            for name in &typed_ent_names {
                if !g_ents.iter().any(|e| fuzzy_match(e, name)) {
                    g_ents.push(name.clone());
                }
            }

            let mut g_rels = a_rels.clone();
            let typed_rel_triples = dedup_rels(&typed_rels_raw);
            for rel in &typed_rel_triples {
                let key = (
                    rel.0.to_lowercase(),
                    rel.1.to_lowercase(),
                    rel.2.to_lowercase(),
                );
                if !g_rels
                    .iter()
                    .any(|r| (r.0.to_lowercase(), r.1.to_lowercase(), r.2.to_lowercase()) == key)
                {
                    g_rels.push(rel.clone());
                }
            }

            let g_sec = a_sec + typing_sec;
            let (_, g_recall) = recall_ct(&g_ents, expected);

            if !typed_ent_names.is_empty() {
                total_typed += typed_ent_names.len();
            }

            eprintln!(
                "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>6} {:>6} {:>6.1}s {:>5}",
                fixture.name,
                a_recall * 100.0,
                b_recall * 100.0,
                g_recall * 100.0,
                g_ents.len(),
                g_rels.len(),
                g_sec,
                orphans.len(),
            );

            if !typed_rel_triples.is_empty() {
                for (s, p, o) in &typed_rel_triples {
                    eprintln!("  +rel:    {} → {} → {}", s, p, o);
                }
            }

            let g_missed: Vec<&String> = expected
                .iter()
                .filter(|e| !g_ents.iter().any(|x| fuzzy_match(x, e)))
                .collect();
            if !g_missed.is_empty() && g_missed.len() <= 5 {
                eprintln!(
                    "  missed:  {}",
                    g_missed
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            totals_a.0 += a_recall;
            totals_a.1 += a_ents.len();
            totals_a.2 += a_rels.len();
            totals_a.3 += a_sec;
            totals_b.0 += b_recall;
            totals_b.1 += b_ents.len();
            totals_g.0 += g_recall;
            totals_g.1 += g_ents.len();
            totals_g.2 += g_rels.len();
            totals_g.3 += g_sec;
            total_orphans += orphans.len();
            n += 1;
        }

        let nf = n as f64;
        let avg_a = totals_a.0 / nf * 100.0;
        let avg_b = totals_b.0 / nf * 100.0;
        let avg_g = totals_g.0 / nf * 100.0;

        eprintln!("{}", "─".repeat(72));
        eprintln!(
            "{:<18} {:>5.0}% {:>5.0}% {:>5.0}% {:>6.0} {:>6.0} {:>6.1}s {:>5.1}",
            "AVERAGE",
            avg_a,
            avg_b,
            avg_g,
            totals_g.1 as f64 / nf,
            totals_g.2 as f64 / nf,
            totals_g.3 / nf,
            total_orphans as f64 / nf,
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!(
            "Phi4 A (SingleCall only):       {:.0}% recall, {:.1}s avg, {} rels",
            avg_a,
            totals_a.3 / nf,
            totals_a.2
        );
        eprintln!(
            "Phi4 B (+ OOV untyped):         {:.0}% recall, {:.1}s avg",
            avg_b,
            totals_a.3 / nf
        );
        eprintln!(
            "Phi4 G (+ OOV + additive type): {:.0}% recall, {:.1}s avg, {} rels",
            avg_g,
            totals_g.3 / nf,
            totals_g.2
        );
        eprintln!(
            "G-A delta: {:+.0}pp recall, {:+.1}s latency, {:+} rels",
            avg_g - avg_a,
            totals_g.3 / nf - totals_a.3 / nf,
            totals_g.2 as isize - totals_a.2 as isize
        );
        eprintln!(
            "Typed by LLM: {} total from {} orphans",
            total_typed, total_orphans
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════════════════════\n"
        );
    }
}
