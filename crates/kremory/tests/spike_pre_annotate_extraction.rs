#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: Phase 2 — Two-call pre-annotated LLM extraction
///
/// Call 1: Pipeline candidates + text → LLM confirms/classifies entities + finds missed
/// Call 2: Confirmed entities + text → LLM extracts relationships
///
/// Compares: (A) baseline single-call vs (B) two-call pre-annotated
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=../models/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_pre_annotate_extraction -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::{ArchitectureBenchmarkRecord, JsonlBenchmarkWriter};
    use serde::Deserialize;
    use kremory::core::provider::{chat_msg_system, chat_msg_user, ChatProvider as _};
    use super::common::build_llm;
    use kremory::core::text_utils::scan_proper_nouns;
    use unicode_segmentation::UnicodeSegmentation;

    // ─── Serde models ───────────────────────────────────────────────────────

    #[derive(Debug, Deserialize, Default)]
    struct FullOutput {
        #[serde(default)]
        entities: Vec<RawEntity>,
        #[serde(default)]
        relationships: Vec<RawRel>,
    }

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

    // ─── Fixtures ───────────────────────────────────────────────────────────

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
        let gt_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ground_truth.json");
        let gt_raw = std::fs::read_to_string(gt_path).expect("ground_truth.json");
        let gt: serde_json::Value = serde_json::from_str(&gt_raw).expect("parse");
        gt.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                let ents: Vec<String> = v["entities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| e["name"].as_str().unwrap().to_lowercase())
                    .collect();
                (k.clone(), ents)
            })
            .collect()
    }

    fn fuzzy_match(a: &str, b: &str) -> bool {
        let a = a.to_lowercase();
        let b = b.to_lowercase();
        a == b || a.contains(&b) || b.contains(&a)
    }

    fn recall_count(extracted: &[String], expected: &[String]) -> (usize, f64) {
        let found = expected
            .iter()
            .filter(|exp| extracted.iter().any(|ext| fuzzy_match(ext, exp)))
            .count();
        (
            found,
            if expected.is_empty() {
                1.0
            } else {
                found as f64 / expected.len() as f64
            },
        )
    }

    // ─── Pipeline (language-agnostic) ───────────────────────────────────────

    fn load_dictionary() -> zspell::Dictionary {
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
        zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .unwrap()
    }

    fn load_stop_words() -> HashSet<String> {
        stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect()
    }

    fn is_oov(word: &str, dict: &zspell::Dictionary, stops: &HashSet<String>) -> bool {
        if word.len() < 2 {
            return false;
        }
        if word.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
            return false;
        }
        let lower = word.to_lowercase();
        if stops.contains(&lower) {
            return false;
        }
        !dict.check_word(word) && !dict.check_word(&lower)
    }

    /// Top-N pipeline candidates: scanner entities first, then OOV words
    fn top_pipeline_candidates(
        text: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
        max: usize,
    ) -> Vec<String> {
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();

        // Scanner entities first (highest confidence — Title Case multi-word runs)
        let scanner = scan_proper_nouns(text, &[]);
        for e in &scanner {
            if seen.insert(e.name.to_lowercase()) {
                candidates.push(e.name.clone());
            }
        }

        // OOV words second
        for w in text.unicode_words() {
            if is_oov(w, dict, stops) && seen.insert(w.to_lowercase()) {
                candidates.push(w.to_string());
            }
        }

        candidates.truncate(max);
        candidates
    }

    // ─── JSON parsing (with repair) ─────────────────────────────────────────

    fn parse_json<T: for<'de> Deserialize<'de> + Default>(raw: &str) -> T {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return T::default();
        }
        if let Ok(v) = serde_json::from_str::<T>(trimmed) {
            return v;
        }
        let repaired = llm_json::repair_json(trimmed, &llm_json::RepairOptions::default())
            .unwrap_or_else(|_| trimmed.to_owned());
        if let Ok(v) = serde_json::from_str::<T>(&repaired) {
            return v;
        }
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                if end > start {
                    let slice = &trimmed[start..=end];
                    let re = llm_json::repair_json(slice, &llm_json::RepairOptions::default())
                        .unwrap_or_else(|_| slice.to_owned());
                    if let Ok(v) = serde_json::from_str::<T>(&re) {
                        return v;
                    }
                }
            }
        }
        T::default()
    }

    fn dedup_entity_names(entities: &[RawEntity]) -> Vec<String> {
        let mut seen = HashSet::new();
        entities
            .iter()
            .filter(|e| !e.name.is_empty())
            .filter(|e| seen.insert(e.name.to_lowercase()))
            .map(|e| e.name.to_lowercase())
            .collect()
    }

    fn dedup_rels(rels: &[RawRel]) -> Vec<(String, String, String)> {
        let mut seen = HashSet::new();
        rels.iter()
            .filter(|r| !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty())
            .filter(|r| {
                seen.insert((
                    r.subject.to_lowercase(),
                    r.predicate.to_lowercase(),
                    r.object.to_lowercase(),
                ))
            })
            .map(|r| (r.subject.clone(), r.predicate.clone(), r.object.clone()))
            .collect()
    }

    // ─── Prompts ────────────────────────────────────────────────────────────

    const ENTITY_TYPES: &str = "Person, Organisation, Location, Technology, Product, Event, Date";

    /// A: Baseline — single call, extract everything
    fn prompt_baseline(text: &str) -> String {
        format!(
            "Extract all unique entities and relationships from the text below.\n\n\
Rules:\n\
- Each entity must appear ONCE (no duplicates)\n\
- Classify entities as: {ENTITY_TYPES}, or Entity\n\
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
Output a single JSON object with \"entities\" and \"relationships\" arrays."
        )
    }

    /// B Call 1: Entity confirmation — candidates + text → confirmed entities
    fn prompt_entity_confirm(text: &str, candidates: &[String]) -> String {
        let list = candidates
            .iter()
            .map(|c| format!("\"{}\"", c))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Extract all unique entities from the text below.\n\n\
Some candidate entities detected automatically: [{list}]\n\
Confirm which of these are real entities, correct any errors, and add any entities missing from the list.\n\n\
Rules:\n\
- Each entity must appear ONCE (no duplicates)\n\
- Classify each as: {ENTITY_TYPES}, or Entity\n\
- Include entities from the candidate list AND any new ones you find\n\n\
Example input: \"Alice from Acme Corp met Bob at the NYC office.\"\n\
Candidates: [\"Alice\", \"Acme Corp\"]\n\
Example output:\n\
{{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},\
{{\"name\":\"Bob\",\"label\":\"Person\"}},{{\"name\":\"NYC office\",\"label\":\"Location\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with an \"entities\" array. No relationships needed."
        )
    }

    /// B Call 2: Relationship extraction — confirmed entities + text → relationships
    fn prompt_relationships(text: &str, entities: &[String]) -> String {
        let list = entities
            .iter()
            .map(|e| format!("\"{}\"", e))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Extract all relationships between these entities from the text below.\n\n\
Known entities: [{list}]\n\n\
Rules:\n\
- Each relationship is a unique (subject, predicate, object) triple\n\
- Subject and object must be from the entity list above\n\
- Use specific predicates: \"prescribed\", \"works_at\", \"located_in\", \"manages\", \"met\", etc.\n\
- No duplicate triples\n\n\
Example: entities=[\"Alice\", \"Acme Corp\", \"Bob\"]\n\
Text: \"Alice from Acme Corp met Bob.\"\n\
Output: {{\"relationships\":[{{\"subject\":\"Alice\",\"predicate\":\"works_at\",\"object\":\"Acme Corp\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"met\",\"object\":\"Bob\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with a \"relationships\" array."
        )
    }

    // ─── Test ───────────────────────────────────────────────────────────────

    const SYSTEM_PROMPT: &str =
        "You are a knowledge graph extraction system. Output valid JSON only.";

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_pre_annotate_vs_baseline() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let gt = load_ground_truth();
        let dict = load_dictionary();
        let stops = load_stop_words();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 2048)
                .await
                .expect("failed to build LlamaCppProvider"),
        );
        let metrics_writer = JsonlBenchmarkWriter::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring"),
            "architecture-bakeoff-pre-annotate",
        )
        .expect("open unified benchmark writer");

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("PHASE 2: Two-Call Pre-Annotated vs Baseline (Ralph Loop 2)");
        eprintln!("Model: {model_path}");
        eprintln!("A = baseline (1 call: entities+rels)");
        eprintln!("B = pre-annotated (call 1: entity confirm, call 2: relationships)");
        eprintln!("Pipeline: top-20 candidates (scanner + OOV, language-agnostic)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<16} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>5}",
            "Domain",
            "A.Ent",
            "B.Ent",
            "A.Rel",
            "B.Rel",
            "A.Rec",
            "B.Rec",
            "A.sec",
            "B1.sec",
            "B2.sec",
            "Pipe"
        );
        eprintln!("{}", "─".repeat(100));

        let mut sum_a_rec = 0.0f64;
        let mut sum_b_rec = 0.0f64;
        let mut sum_a_rels = 0usize;
        let mut sum_b_rels = 0usize;
        let mut sum_a_ents = 0usize;
        let mut sum_b_ents = 0usize;
        let mut sum_a_sec = 0.0f64;
        let mut sum_b1_sec = 0.0f64;
        let mut sum_b2_sec = 0.0f64;
        let mut n = 0usize;

        for fixture in &fixtures() {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => continue,
            };
            let text = std::fs::read_to_string(fixture.path).unwrap();

            // ── Pipeline ──
            let pipe_start = Instant::now();
            let candidates = top_pipeline_candidates(&text, &dict, &stops, 20);
            let pipe_ms = pipe_start.elapsed().as_secs_f64() * 1000.0;

            // ── A: Baseline single call ──
            let t0 = Instant::now();
            let msgs_a = vec![
                chat_msg_system(SYSTEM_PROMPT),
                chat_msg_user(&prompt_baseline(&text)),
            ];
            let resp_a = llm.chat_with_tools(&msgs_a, None, None).await;
            let a_sec = t0.elapsed().as_secs_f64();
            let a_parser_ok = resp_a.is_ok();

            let out_a: FullOutput = match resp_a {
                Ok(r) => parse_json(&r.text().unwrap_or_default()),
                Err(e) => {
                    eprintln!("{:<16} A ERROR: {e}", fixture.name);
                    continue;
                }
            };
            let a_ents = dedup_entity_names(&out_a.entities);
            let a_rels = dedup_rels(&out_a.relationships);
            let (_, a_rec) = recall_count(&a_ents, expected);

            // ── B Call 1: Entity confirmation ──
            let t1 = Instant::now();
            let msgs_b1 = vec![
                chat_msg_system(SYSTEM_PROMPT),
                chat_msg_user(&prompt_entity_confirm(&text, &candidates)),
            ];
            let resp_b1 = llm.chat_with_tools(&msgs_b1, None, None).await;
            let b1_sec = t1.elapsed().as_secs_f64();
            let b1_parser_ok = resp_b1.is_ok();

            let out_b1: EntityOnlyOutput = match resp_b1 {
                Ok(r) => parse_json(&r.text().unwrap_or_default()),
                Err(e) => {
                    eprintln!("{:<16} B1 ERROR: {e}", fixture.name);
                    continue;
                }
            };
            let b_ents = dedup_entity_names(&out_b1.entities);

            // ── B Call 2: Relationship extraction ──
            let t2 = Instant::now();
            let msgs_b2 = vec![
                chat_msg_system(SYSTEM_PROMPT),
                chat_msg_user(&prompt_relationships(&text, &b_ents)),
            ];
            let resp_b2 = llm.chat_with_tools(&msgs_b2, None, None).await;
            let b2_sec = t2.elapsed().as_secs_f64();
            let b2_parser_ok = resp_b2.is_ok();

            let out_b2: RelOnlyOutput = match resp_b2 {
                Ok(r) => parse_json(&r.text().unwrap_or_default()),
                Err(e) => {
                    eprintln!("{:<16} B2 ERROR: {e}", fixture.name);
                    continue;
                }
            };
            let b_rels = dedup_rels(&out_b2.relationships);
            let (_, b_rec) = recall_count(&b_ents, expected);

            let baseline_record = ArchitectureBenchmarkRecord {
                architecture: "baseline_single_call".to_string(),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall: a_rec,
                entity_count: a_ents.len(),
                expected_entity_count: expected.len(),
                relationship_count: a_rels.len(),
                relationship_duplicates: None,
                latency_ms: a_sec * 1000.0,
                document_level: false,
                llm_calls: Some(1),
                candidate_count: None,
                pipeline_ms: Some(pipe_ms),
                parser_ok: Some(a_parser_ok),
                stage_label: Some("baseline".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            baseline_record.record_metrics();
            let _ = metrics_writer.append(&baseline_record);

            let pre_annotate_record = ArchitectureBenchmarkRecord {
                architecture: "pre_annotate_two_call".to_string(),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall: b_rec,
                entity_count: b_ents.len(),
                expected_entity_count: expected.len(),
                relationship_count: b_rels.len(),
                relationship_duplicates: None,
                latency_ms: (b1_sec + b2_sec) * 1000.0,
                document_level: false,
                llm_calls: Some(2),
                candidate_count: Some(candidates.len()),
                pipeline_ms: Some(pipe_ms),
                parser_ok: Some(b1_parser_ok && b2_parser_ok),
                stage_label: Some("pre_annotate".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            pre_annotate_record.record_metrics();
            let _ = metrics_writer.append(&pre_annotate_record);

            // ── Print row ──
            eprintln!(
                "{:<16} {:>6} {:>6} {:>5} {:>5} {:>4.0}% {:>4.0}% {:>6.1}s {:>6.1}s {:>6.1}s {:>4.0}ms",
                fixture.name,
                a_ents.len(),
                b_ents.len(),
                a_rels.len(),
                b_rels.len(),
                a_rec * 100.0,
                b_rec * 100.0,
                a_sec,
                b1_sec,
                b2_sec,
                pipe_ms,
            );

            // ── Diff: what B finds that A misses and vice versa ──
            let b_wins: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    b_ents.iter().any(|x| fuzzy_match(x, e))
                        && !a_ents.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            let a_wins: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    a_ents.iter().any(|x| fuzzy_match(x, e))
                        && !b_ents.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            if !b_wins.is_empty() {
                eprintln!(
                    "  B finds, A misses: {}",
                    b_wins
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !a_wins.is_empty() {
                eprintln!(
                    "  A finds, B misses: {}",
                    a_wins
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Sample rels
            if !a_rels.is_empty() {
                eprintln!(
                    "  A rels: {}",
                    a_rels
                        .iter()
                        .take(3)
                        .map(|(s, p, o)| format!("({s}->{p}->{o})"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !b_rels.is_empty() {
                eprintln!(
                    "  B rels: {}",
                    b_rels
                        .iter()
                        .take(3)
                        .map(|(s, p, o)| format!("({s}->{p}->{o})"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            eprintln!("  Pipeline: {} candidates", candidates.len());

            sum_a_rec += a_rec;
            sum_b_rec += b_rec;
            sum_a_rels += a_rels.len();
            sum_b_rels += b_rels.len();
            sum_a_ents += a_ents.len();
            sum_b_ents += b_ents.len();
            sum_a_sec += a_sec;
            sum_b1_sec += b1_sec;
            sum_b2_sec += b2_sec;
            n += 1;
        }

        let nf = n as f64;
        eprintln!("{}", "─".repeat(100));
        eprintln!(
            "{:<16} {:>6} {:>6} {:>5} {:>5} {:>4.0}% {:>4.0}% {:>6.1}s {:>6.1}s {:>6.1}s",
            "AVERAGE",
            sum_a_ents / n,
            sum_b_ents / n,
            sum_a_rels / n,
            sum_b_rels / n,
            sum_a_rec / nf * 100.0,
            sum_b_rec / nf * 100.0,
            sum_a_sec / nf,
            sum_b1_sec / nf,
            sum_b2_sec / nf,
        );
        let b_total = sum_b1_sec + sum_b2_sec;
        eprintln!(
            "\nB total time: {:.1}s ({:.1}s entity + {:.1}s rels) vs A total: {:.1}s",
            b_total, sum_b1_sec, sum_b2_sec, sum_a_sec
        );
        eprintln!("Unified metrics: {}", metrics_writer.path().display());
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }
}