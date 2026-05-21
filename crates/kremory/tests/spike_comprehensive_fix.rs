#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: Comprehensive Fix — addresses recall degradation, language-agnostic, latency
///
/// Fixes:
/// 1. Candidate cap raised to 50 (was 20-25)
/// 2. OOV-only variant (no scanner — truly language-agnostic)
/// 3. Chunked latency measurement (300-word chunks)
///
/// Tests 3 candidate sources × two-call LLM pattern:
///   S = scanner-only | O = OOV-only | P = pipeline (scanner+OOV)
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_comprehensive_fix -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use serde::Deserialize;
    use autoagents_llamacpp::LlamaCppProvider;
    use kremory::core::provider::{chat_msg_system, chat_msg_user, ChatProvider as _};
    use super::common::build_llm;
    use kremory::core::text_utils::scan_proper_nouns;
    use unicode_segmentation::UnicodeSegmentation;

    #[derive(Debug, Deserialize, Default)]
    struct EntOut {
        #[serde(default)]
        entities: Vec<RE>,
    }
    #[derive(Debug, Deserialize, Default)]
    struct RelOut {
        #[serde(default)]
        relationships: Vec<RR>,
    }
    #[derive(Debug, Deserialize)]
    struct RE {
        #[serde(default)]
        name: String,
        #[allow(dead_code)]
        #[serde(default = "dl")]
        label: String,
    }
    fn dl() -> String {
        "Entity".into()
    }
    #[derive(Debug, Deserialize)]
    struct RR {
        #[serde(default)]
        subject: String,
        #[serde(default)]
        predicate: String,
        #[serde(default)]
        object: String,
    }

    struct Fix {
        name: &'static str,
        key: &'static str,
        path: &'static str,
    }
    fn fixtures() -> Vec<Fix> {
        vec![
            Fix {
                name: "Mock Interview",
                key: "mock_interview",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mock_interview.txt"),
            },
            Fix {
                name: "Medical Consult",
                key: "medical_consultation",
                path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/medical_consultation.txt"
                ),
            },
            Fix {
                name: "Tech Standup",
                key: "tech_standup",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tech_standup.txt"),
            },
            Fix {
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
    fn fm(a: &str, b: &str) -> bool {
        let a = a.to_lowercase();
        let b = b.to_lowercase();
        a == b || a.contains(&b) || b.contains(&a)
    }
    fn recall_ct(ext: &[String], exp: &[String]) -> (usize, f64) {
        let f = exp.iter().filter(|e| ext.iter().any(|x| fm(x, e))).count();
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
        let a = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .unwrap();
        let d = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .unwrap();
        zspell::builder()
            .config_str(&a)
            .dict_str(&d)
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
    fn pj<T: for<'de> Deserialize<'de> + Default>(raw: &str) -> T {
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
    fn de(e: &[RE]) -> Vec<String> {
        let mut s = HashSet::new();
        e.iter()
            .filter(|e| !e.name.is_empty())
            .filter(|e| s.insert(e.name.to_lowercase()))
            .map(|e| e.name.to_lowercase())
            .collect()
    }
    fn dr(r: &[RR]) -> Vec<(String, String, String)> {
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

    // ─── Candidate generators ───────────────────────────────────────────────

    /// OOV-only: no scanner, truly language-agnostic
    fn oov_only_candidates(
        text: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
        max: usize,
    ) -> Vec<String> {
        let mut all = HashSet::new();
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
        v.truncate(max);
        v
    }

    /// Scanner-only: English Title Case heuristic
    fn scanner_only_candidates(text: &str, max: usize) -> Vec<String> {
        let mut v: Vec<String> = scan_proper_nouns(text, &[])
            .iter()
            .map(|e| e.name.clone())
            .collect();
        v.truncate(max);
        v
    }

    /// Pipeline: scanner + OOV combined
    fn pipeline_candidates(
        text: &str,
        dict: &zspell::Dictionary,
        stops: &HashSet<String>,
        max: usize,
    ) -> Vec<String> {
        let mut all = HashSet::new();
        for e in &scan_proper_nouns(text, &[]) {
            all.insert(e.name.clone());
        }
        for w in text.unicode_words() {
            if is_oov(w, dict, stops) {
                all.insert(w.to_string());
            }
        }
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
        v.truncate(max);
        v
    }

    // ─── Two-call LLM pattern ───────────────────────────────────────────────

    const ET: &str = "Person, Organisation, Location, Technology, Product, Event, Date";
    const SYS: &str = "You are a knowledge graph extraction system. Output valid JSON only.";

    fn prompt_ent(text: &str, cands: &[String]) -> String {
        // Compact format: comma-separated, not JSON array
        let list = cands.join(", ");
        format!(
            "Extract all unique entities from the text below.\n\n\
Candidate entities detected automatically: {list}\n\
Include these AND any additional entities you find. Classify each as: {ET}, or Entity.\n\
Each entity must appear ONCE.\n\n\
Example: Text: \"Alice from Acme Corp met Bob.\"\nCandidates: Alice, Acme Corp\n\
Output: {{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},{{\"name\":\"Bob\",\"label\":\"Person\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\nOutput JSON with \"entities\" array. Include ALL entities — candidates plus new ones."
        )
    }

    fn prompt_rel(text: &str, ents: &[String]) -> String {
        let list = ents.join(", ");
        format!(
            "Extract relationships between these entities from the text.\n\n\
Entities: {list}\n\n\
Rules: unique (subject, predicate, object) triples. Use specific predicates.\n\n\
<TEXT>\n{text}\n</TEXT>\n\nOutput JSON with \"relationships\" array."
        )
    }

    struct TwoCallResult {
        entities: Vec<String>,
        rels: Vec<(String, String, String)>,
        ent_sec: f64,
        rel_sec: f64,
        ent_output_len: usize,
        rel_output_len: usize,
    }

    async fn two_call(llm: &LlamaCppProvider, text: &str, candidates: &[String]) -> TwoCallResult {
        let t1 = Instant::now();
        let msgs1 = vec![
            chat_msg_system(SYS),
            chat_msg_user(prompt_ent(text, candidates)),
        ];
        let r1: Result<Box<dyn kremory::core::provider::ChatResponse>, _> =
            llm.chat_with_tools(&msgs1, None, None).await;
        let ent_sec = t1.elapsed().as_secs_f64();
        let raw1 = match r1 {
            Ok(r) => r.text().unwrap_or_default(),
            Err(_) => String::new(),
        };
        let ent_output_len = raw1.len();
        let o1: EntOut = pj(&raw1);
        let ents = de(&o1.entities);

        let t2 = Instant::now();
        let msgs2 = vec![
            chat_msg_system(SYS),
            chat_msg_user(prompt_rel(text, &ents)),
        ];
        let r2: Result<Box<dyn kremory::core::provider::ChatResponse>, _> =
            llm.chat_with_tools(&msgs2, None, None).await;
        let rel_sec = t2.elapsed().as_secs_f64();
        let raw2 = match r2 {
            Ok(r) => r.text().unwrap_or_default(),
            Err(_) => String::new(),
        };
        let rel_output_len = raw2.len();
        let o2: RelOut = pj(&raw2);
        let rels = dr(&o2.relationships);

        TwoCallResult {
            entities: ents,
            rels,
            ent_sec,
            rel_sec,
            ent_output_len,
            rel_output_len,
        }
    }

    // ─── Chunking utility ───────────────────────────────────────────────────

    fn chunk_text(text: &str, max_words: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        words.chunks(max_words).map(|c| c.join(" ")).collect()
    }

    // ─── Main test ──────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_comprehensive_fix() {
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
            build_llm(&model_path, 4096, 2048)
                .await
                .expect("failed to build LlamaCppProvider"),
        );

        let cap = 50; // Fix 1: raised from 20-25

        eprintln!("\n════════════════════════════════════════════════════════════════════════════════════");
        eprintln!("COMPREHENSIVE FIX: 3 candidate sources × two-call × cap={cap}");
        eprintln!("O = OOV-only (language-agnostic) | S = scanner-only | P = scanner+OOV");
        eprintln!("════════════════════════════════════════════════════════════════════════════════════\n");

        eprintln!("{:<16} {:>5} {:>5} {:>5} {:>4} {:>4} {:>4} {:>5} {:>5} {:>5} {:>6} {:>6} {:>6} {:>4} {:>4} {:>4}",
            "Domain", "O.Rec", "S.Rec", "P.Rec", "O.En", "S.En", "P.En", "O.Rel", "S.Rel", "P.Rel", "O.sec", "S.sec", "P.sec", "O.ca", "S.ca", "P.ca");
        eprintln!("{}", "─".repeat(115));

        let mut sum = [[0.0f64; 3]; 4]; // [rec, rels, sec, cands] × [O, S, P]
        let mut sum_ents = [0usize; 3];
        let mut sum_rels = [0usize; 3];
        let mut n = 0usize;

        for f in &fixtures() {
            let exp = match gt.get(f.key) {
                Some(e) => e,
                None => continue,
            };
            let text = std::fs::read_to_string(f.path).unwrap();
            let words: Vec<&str> = text.split_whitespace().collect();

            // Generate candidates
            let o_cands = oov_only_candidates(&text, &dict, &stops, cap);
            let s_cands = scanner_only_candidates(&text, cap);
            let p_cands = pipeline_candidates(&text, &dict, &stops, cap);

            // Pipeline candidate recall (pre-LLM)
            let o_cands_lower: Vec<String> = o_cands.iter().map(|s| s.to_lowercase()).collect();
            let s_cands_lower: Vec<String> = s_cands.iter().map(|s| s.to_lowercase()).collect();
            let p_cands_lower: Vec<String> = p_cands.iter().map(|s| s.to_lowercase()).collect();
            let (_, o_pre) = recall_ct(&o_cands_lower, exp);
            let (_, s_pre) = recall_ct(&s_cands_lower, exp);
            let (_, p_pre) = recall_ct(&p_cands_lower, exp);

            // Two-call LLM for each
            let o = two_call(&*llm, &text, &o_cands).await;
            let s = two_call(&*llm, &text, &s_cands).await;
            let p = two_call(&*llm, &text, &p_cands).await;

            let (_, o_rec) = recall_ct(&o.entities, exp);
            let (_, s_rec) = recall_ct(&s.entities, exp);
            let (_, p_rec) = recall_ct(&p.entities, exp);

            let o_sec = o.ent_sec + o.rel_sec;
            let s_sec = s.ent_sec + s.rel_sec;
            let p_sec = p.ent_sec + p.rel_sec;

            eprintln!("{:<16} {:>4.0}% {:>4.0}% {:>4.0}% {:>4} {:>4} {:>4} {:>5} {:>5} {:>5} {:>5.1}s {:>5.1}s {:>5.1}s {:>4} {:>4} {:>4}",
                f.name,
                o_rec*100.0, s_rec*100.0, p_rec*100.0,
                o.entities.len(), s.entities.len(), p.entities.len(),
                o.rels.len(), s.rels.len(), p.rels.len(),
                o_sec, s_sec, p_sec,
                o_cands.len(), s_cands.len(), p_cands.len());

            // Pre-LLM vs post-LLM recall loss
            eprintln!(
                "  Pre-LLM recall: O={:.0}% S={:.0}% P={:.0}% | Words: {} | Chunks@300: {}",
                o_pre * 100.0,
                s_pre * 100.0,
                p_pre * 100.0,
                words.len(),
                (words.len() + 299) / 300
            );

            // Entity-level diagnostics: what each approach uniquely finds/misses
            let o_unique: Vec<&String> = exp
                .iter()
                .filter(|e| {
                    o.entities.iter().any(|x| fm(x, e)) && !s.entities.iter().any(|x| fm(x, e))
                })
                .collect();
            let s_unique: Vec<&String> = exp
                .iter()
                .filter(|e| {
                    s.entities.iter().any(|x| fm(x, e)) && !o.entities.iter().any(|x| fm(x, e))
                })
                .collect();
            let all_miss: Vec<&String> = exp
                .iter()
                .filter(|e| {
                    !o.entities.iter().any(|x| fm(x, e))
                        && !s.entities.iter().any(|x| fm(x, e))
                        && !p.entities.iter().any(|x| fm(x, e))
                })
                .collect();

            if !o_unique.is_empty() {
                eprintln!(
                    "  OOV-only uniquely finds: {}",
                    o_unique
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !s_unique.is_empty() {
                eprintln!(
                    "  Scanner uniquely finds: {}",
                    s_unique
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !all_miss.is_empty() {
                eprintln!(
                    "  ALL miss: {}",
                    all_miss
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            // Latency breakdown
            eprintln!("  Latency: O(ent={:.1}s rel={:.1}s out={}+{}b) S(ent={:.1}s rel={:.1}s) P(ent={:.1}s rel={:.1}s)",
                o.ent_sec, o.rel_sec, o.ent_output_len, o.rel_output_len,
                s.ent_sec, s.rel_sec, p.ent_sec, p.rel_sec);

            sum[0][0] += o_rec;
            sum[0][1] += s_rec;
            sum[0][2] += p_rec;
            sum_rels[0] += o.rels.len();
            sum_rels[1] += s.rels.len();
            sum_rels[2] += p.rels.len();
            sum_ents[0] += o.entities.len();
            sum_ents[1] += s.entities.len();
            sum_ents[2] += p.entities.len();
            sum[2][0] += o_sec;
            sum[2][1] += s_sec;
            sum[2][2] += p_sec;
            n += 1;
        }

        let nf = n as f64;
        eprintln!("{}", "─".repeat(115));
        eprintln!("{:<16} {:>4.0}% {:>4.0}% {:>4.0}% {:>4} {:>4} {:>4} {:>5} {:>5} {:>5} {:>5.1}s {:>5.1}s {:>5.1}s",
            "AVERAGE",
            sum[0][0]/nf*100.0, sum[0][1]/nf*100.0, sum[0][2]/nf*100.0,
            sum_ents[0]/n, sum_ents[1]/n, sum_ents[2]/n,
            sum_rels[0]/n, sum_rels[1]/n, sum_rels[2]/n,
            sum[2][0]/nf, sum[2][1]/nf, sum[2][2]/nf);

        // ── Chunked latency test (Mock Interview, 300-word chunks) ──────────
        eprintln!("\n── CHUNKED LATENCY (Mock Interview, 300-word chunks) ──────────────────────");
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/mock_interview.txt"
        ))
        .unwrap();
        let chunks = chunk_text(&text, 300);
        eprintln!(
            "Fixture: {} words → {} chunks @ 300 words",
            text.split_whitespace().count(),
            chunks.len()
        );

        let mut chunk_total = 0.0f64;
        for (i, chunk) in chunks.iter().enumerate() {
            let cands = pipeline_candidates(chunk, &dict, &stops, cap);
            let pipe_start = Instant::now();
            let _ = pipeline_candidates(chunk, &dict, &stops, cap);
            let pipe_ms = pipe_start.elapsed().as_secs_f64() * 1000.0;

            let r = two_call(&*llm, chunk, &cands).await;
            let total = r.ent_sec + r.rel_sec;
            chunk_total += total;

            eprintln!(
                "  Chunk {}: {} words, {} cands, ent={:.1}s rel={:.1}s total={:.1}s pipe={:.1}ms",
                i + 1,
                chunk.split_whitespace().count(),
                cands.len(),
                r.ent_sec,
                r.rel_sec,
                total,
                pipe_ms
            );
        }
        eprintln!(
            "  Total: {:.1}s for {} chunks = {:.1}s/chunk avg",
            chunk_total,
            chunks.len(),
            chunk_total / chunks.len() as f64
        );
        eprintln!("  Latency doc baseline: 9.1s/chunk (Qwen 3B single-call)");

        eprintln!("\n════════════════════════════════════════════════════════════════════════════════════");
        eprintln!("Candidate cap: {cap} | Two-call pattern | Language-agnostic filters only");
        eprintln!("O = OOV-only (lang-agnostic) | S = scanner (English) | P = combined");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════════"
        );
    }
}