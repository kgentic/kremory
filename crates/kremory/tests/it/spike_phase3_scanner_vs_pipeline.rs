#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

//! Spike: Phase 3 — Scanner-only vs Pipeline candidates (both using two-call approach)
//!
//! Both approaches use the proven two-call pattern:
//!   Call 1: candidates + text → LLM confirms entities
//!   Call 2: confirmed entities + text → LLM extracts relationships
//!
//! The question: does OOV+PMI pipeline add value over scanner-only candidates?
//!
//! Run with:
//!   RQL_QWEN3B_MODEL_PATH=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
//!     cargo test --features llm -p rql-core --test it spike_phase3_scanner_vs_pipeline:: -- --nocapture

#[cfg(feature = "llm")]
mod spike {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use crate::common::build_llm;
    use autoagents_llamacpp::LlamaCppProvider;
    use kremory::core::provider::{chat_msg_system, chat_msg_user, ChatProvider as _};
    use kremory::core::text_utils::scan_proper_nouns;
    use serde::Deserialize;
    use unicode_segmentation::UnicodeSegmentation;

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
        #[serde(default = "dl")]
        label: String,
    }
    fn dl() -> String {
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
                name: "Tech Standup",
                key: "tech_standup",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
            },
            Fixture {
                name: "News Article",
                key: "news_article",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
            },
        ]
    }

    fn load_gt() -> HashMap<String, Vec<String>> {
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
                (
                    k.clone(),
                    v["entities"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|e| e["name"].as_str().unwrap().to_lowercase())
                        .collect(),
                )
            })
            .collect()
    }

    fn fmatch(a: &str, b: &str) -> bool {
        let a = a.to_lowercase();
        let b = b.to_lowercase();
        a == b || a.contains(&b) || b.contains(&a)
    }
    fn recall_ct(ext: &[String], exp: &[String]) -> (usize, f64) {
        let f = exp
            .iter()
            .filter(|e| ext.iter().any(|x| fmatch(x, e)))
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

    fn load_dict() -> zspell::Dictionary {
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
    fn load_stops() -> HashSet<String> {
        stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect()
    }
    fn is_oov(w: &str, d: &zspell::Dictionary, s: &HashSet<String>) -> bool {
        if w.len() < 2 {
            return false;
        }
        if w.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
            return false;
        }
        let l = w.to_lowercase();
        if s.contains(&l) {
            return false;
        }
        !d.check_word(w) && !d.check_word(&l)
    }

    /// Scanner-only candidates (no OOV, no PMI)
    fn scanner_candidates(text: &str) -> Vec<String> {
        scan_proper_nouns(text, &[])
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// Full pipeline: scanner + OOV (no PMI — per latency doc, PMI broken at meeting scale)
    fn pipeline_candidates(
        text: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
    ) -> Vec<String> {
        let mut all = HashSet::new();
        // Scanner
        for e in &scan_proper_nouns(text, &[]) {
            all.insert(e.name.clone());
        }
        // OOV singles
        for w in text.unicode_words() {
            if is_oov(w, dict, stops) {
                all.insert(w.to_string());
            }
        }
        // OOV runs (2-4 consecutive)
        let words: Vec<&str> = text.unicode_words().collect();
        let mut run: Vec<&str> = Vec::new();
        for w in &words {
            if is_oov(w, dict, stops) {
                run.push(w);
                if run.len() >= 4 {
                    all.insert(run.join(" "));
                    run.clear();
                }
            } else {
                if run.len() >= 2 {
                    all.insert(run.join(" "));
                }
                run.clear();
            }
        }
        if run.len() >= 2 {
            all.insert(run.join(" "));
        }

        let mut v: Vec<String> = all.into_iter().collect();
        v.truncate(25);
        v
    }

    fn parse_json<T: for<'de> Deserialize<'de> + Default>(raw: &str) -> T {
        let t = raw.trim();
        if t.is_empty() {
            return T::default();
        }
        if let Ok(v) = serde_json::from_str::<T>(t) {
            return v;
        }
        let r = jsonrepair::repair_json(t, &jsonrepair::Options::default()).unwrap_or(t.to_owned());
        if let Ok(v) = serde_json::from_str::<T>(&r) {
            return v;
        }
        if let (Some(s), Some(e)) = (t.find('{'), t.rfind('}')) {
            if e > s {
                let sl = &t[s..=e];
                let r2 = jsonrepair::repair_json(sl, &jsonrepair::Options::default())
                    .unwrap_or(sl.to_owned());
                if let Ok(v) = serde_json::from_str::<T>(&r2) {
                    return v;
                }
            }
        }
        T::default()
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

    const ET: &str = "Person, Organisation, Location, Technology, Product, Event, Date";
    const SYS: &str = "You are a knowledge graph extraction system. Output valid JSON only.";

    fn prompt_ent(text: &str, cands: &[String]) -> String {
        let l = cands
            .iter()
            .map(|c| format!("\"{}\"", c))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Extract all unique entities from the text below.\n\n\
Some candidate entities detected automatically: [{l}]\n\
Confirm which are real entities, correct errors, add any missing.\n\n\
Rules: each entity ONCE, classify as: {ET}, or Entity.\n\n\
Example: Text: \"Alice from Acme Corp met Bob.\"\nCandidates: [\"Alice\", \"Acme Corp\"]\n\
Output: {{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},{{\"name\":\"Bob\",\"label\":\"Person\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\nOutput JSON with \"entities\" array only.")
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_scanner_vs_pipeline() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP");
                return;
            }
        };
        let gt = load_gt();
        let dict = load_dict();
        let stops = load_stops();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("PHASE 3: Scanner-Only vs Pipeline Candidates (both two-call)");
        eprintln!("S = scanner only  |  P = scanner + OOV (language-agnostic pipeline)");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<16} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>5} {:>5}",
            "Domain",
            "S.Ent",
            "P.Ent",
            "S.Rel",
            "P.Rel",
            "S.Rec",
            "P.Rec",
            "S.sec",
            "P.sec",
            "S.can",
            "P.can"
        );
        eprintln!("{}", "─".repeat(100));

        let mut ts = [0.0f64; 4]; // s_rec, p_rec, s_sec, p_sec
        let mut ti = [0usize; 4]; // s_rels, p_rels, s_ents, p_ents
        let mut n = 0usize;

        for f in &fixtures() {
            let exp = match gt.get(f.key) {
                Some(e) => e,
                None => continue,
            };
            let text = std::fs::read_to_string(f.path).unwrap();

            let s_cands = scanner_candidates(&text);
            let p_cands = pipeline_candidates(&text, &dict, &stops);

            let (s_ents, s_rels, s1, s2) = two_call(&*llm, &text, &s_cands).await;
            let s_sec = s1 + s2;
            let (_, s_rec) = recall_ct(&s_ents, exp);

            let (p_ents, p_rels, p1, p2) = two_call(&*llm, &text, &p_cands).await;
            let p_sec = p1 + p2;
            let (_, p_rec) = recall_ct(&p_ents, exp);

            eprintln!(
                "{:<16} {:>6} {:>6} {:>5} {:>5} {:>4.0}% {:>4.0}% {:>6.1}s {:>6.1}s {:>5} {:>5}",
                f.name,
                s_ents.len(),
                p_ents.len(),
                s_rels.len(),
                p_rels.len(),
                s_rec * 100.0,
                p_rec * 100.0,
                s_sec,
                p_sec,
                s_cands.len(),
                p_cands.len()
            );

            let p_wins: Vec<&String> = exp
                .iter()
                .filter(|e| {
                    p_ents.iter().any(|x| fmatch(x, e)) && !s_ents.iter().any(|x| fmatch(x, e))
                })
                .collect();
            let s_wins: Vec<&String> = exp
                .iter()
                .filter(|e| {
                    s_ents.iter().any(|x| fmatch(x, e)) && !p_ents.iter().any(|x| fmatch(x, e))
                })
                .collect();
            if !p_wins.is_empty() {
                eprintln!(
                    "  P finds, S misses: {}",
                    p_wins
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !s_wins.is_empty() {
                eprintln!(
                    "  S finds, P misses: {}",
                    s_wins
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            ts[0] += s_rec;
            ts[1] += p_rec;
            ts[2] += s_sec;
            ts[3] += p_sec;
            ti[0] += s_rels.len();
            ti[1] += p_rels.len();
            ti[2] += s_ents.len();
            ti[3] += p_ents.len();
            n += 1;
        }

        let nf = n as f64;
        eprintln!("{}", "─".repeat(100));
        eprintln!(
            "{:<16} {:>6} {:>6} {:>5} {:>5} {:>4.0}% {:>4.0}% {:>6.1}s {:>6.1}s",
            "AVERAGE",
            ti[2] / n,
            ti[3] / n,
            ti[0] / n,
            ti[1] / n,
            ts[0] / nf * 100.0,
            ts[1] / nf * 100.0,
            ts[2] / nf,
            ts[3] / nf
        );
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }
}
