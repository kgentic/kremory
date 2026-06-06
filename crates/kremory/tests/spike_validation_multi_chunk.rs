#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Validation Spike: Multi-chunk document-level recall
///
/// Tests the full Free Discovery + Programmatic Audit pipeline:
///   Per chunk: SingleCallExtractor → OovAuditor → accumulate known_entities
///   Post-doc: scan_proper_nouns sweep
///
/// Measures document-level entity recall and relationship count.
/// Compares: with OOV audit vs without, cross-chunk accumulation effect.
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_validation_multi_chunk -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::build_llm;
    use kremory::core::config::ContentType;
    use kremory::core::extraction::SingleCallExtractor;
    use kremory::core::intelligence::{EntityExtractor, ExtractedEntity, ExtractionContext};
    use kremory::core::text_utils::{scan_proper_nouns, OovAuditor};

    struct Fix {
        name: &'static str,
        key: &'static str,
        path: &'static str,
    }

    fn fixtures() -> Vec<Fix> {
        vec![
            Fix {
                name: "Long Report",
                key: "long_report",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/long_report.txt"),
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
                name: "News Article",
                key: "news_article",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/news_article.txt"),
            },
            Fix {
                name: "Board Meeting",
                key: "board_meeting",
                path: concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/board_meeting.txt"),
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

    fn fuzzy_match(a: &str, b: &str) -> bool {
        let a = a.to_lowercase();
        let b = b.to_lowercase();
        a == b || a.contains(&b) || b.contains(&a)
    }

    fn recall_detail(
        extracted: &[String],
        expected: &[String],
    ) -> (usize, f64, Vec<String>, Vec<String>) {
        let mut found = Vec::new();
        let mut missed = Vec::new();
        for exp in expected {
            if extracted.iter().any(|ext| fuzzy_match(ext, exp)) {
                found.push(exp.clone());
            } else {
                missed.push(exp.clone());
            }
        }
        let r = if expected.is_empty() {
            1.0
        } else {
            found.len() as f64 / expected.len() as f64
        };
        (found.len(), r, found, missed)
    }

    /// Chunk text into segments of max_words with overlap
    fn chunk_with_overlap(text: &str, max_words: usize, overlap: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() <= max_words {
            return vec![text.to_string()];
        }
        let step = max_words - overlap;
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < words.len() {
            let end = (start + max_words).min(words.len());
            chunks.push(words[start..end].join(" "));
            if end >= words.len() {
                break;
            }
            start += step;
        }
        chunks
    }

    fn load_oov_auditor() -> OovAuditor {
        let aff = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .expect("en_US.aff — run the OOV spike first to download dictionaries");
        let dic = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .expect("en_US.dic");
        let dict = zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .unwrap();
        let stops: HashSet<String> = stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        OovAuditor::new(dict, stops)
    }

    fn norm(s: &str) -> String {
        s.to_lowercase().trim().to_string()
    }

    fn dedup_entities(entities: &mut Vec<ExtractedEntity>) {
        entities.sort_by(|a, b| norm(&a.name).cmp(&norm(&b.name)));
        entities.dedup_by(|a, b| norm(&a.name) == norm(&b.name));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_validation_document_level() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };
        let gt = load_gt();
        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );
        let extractor = SingleCallExtractor::new(Arc::clone(&llm));
        let auditor = load_oov_auditor();

        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("VALIDATION SPIKE: Multi-Chunk Document-Level Recall");
        eprintln!("Pipeline: SingleCallExtractor → OovAuditor → scan_proper_nouns");
        eprintln!("Chunks: 300 words, 50-word overlap");
        eprintln!("Model: {model_path}");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<16} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5} {:>7} {:>5}",
            "Fixture", "Chunks", "NoAud", "+Audit", "N.Rel", "A.Rel", "A.Ent", "Time", "OOV+"
        );
        eprintln!("{}", "─".repeat(75));

        let mut sum_no_audit = 0.0f64;
        let mut sum_with_audit = 0.0f64;
        let mut sum_rels = [0usize; 2];
        let mut n = 0usize;

        for f in &fixtures() {
            let expected = match gt.get(f.key) {
                Some(e) => e,
                None => continue,
            };
            let text = std::fs::read_to_string(f.path).unwrap();
            let chunks = chunk_with_overlap(&text, 300, 50);
            let chunk_count = chunks.len();

            let start = Instant::now();

            // ══ Run A: SingleCallExtractor + cross-chunk accumulation (NO OOV audit) ══
            let mut ents_no_audit: Vec<ExtractedEntity> = Vec::new();
            let mut facts_no_audit = 0usize;
            for chunk in &chunks {
                let ctx = ExtractionContext {
                    known_entities: &ents_no_audit,
                    allowed_entity_types: &[],
                    allowed_edge_types: &[],
                    excluded_entity_types: &[],
                    content_type: ContentType::Text,
                    registry_specs: &[],
                };
                match extractor.extract(chunk, &ctx).await {
                    Ok(result) => {
                        facts_no_audit += result.facts.len();
                        ents_no_audit.extend(result.entities);
                    }
                    Err(e) => eprintln!("  LLM error: {e}"),
                }
            }
            // Post-doc: scanner sweep
            let scanner_adds = scan_proper_nouns(&text, &ents_no_audit);
            ents_no_audit.extend(scanner_adds);
            dedup_entities(&mut ents_no_audit);
            let no_audit_names: Vec<String> = ents_no_audit
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();
            let (_, no_audit_recall, _, _) = recall_detail(&no_audit_names, expected);

            // ══ Run B: SingleCallExtractor + OOV audit + cross-chunk + scanner ══
            let mut ents_with_audit: Vec<ExtractedEntity> = Vec::new();
            let mut facts_with_audit = 0usize;
            let mut oov_adds_total = 0usize;
            for chunk in &chunks {
                let ctx = ExtractionContext {
                    known_entities: &ents_with_audit,
                    allowed_entity_types: &[],
                    allowed_edge_types: &[],
                    excluded_entity_types: &[],
                    content_type: ContentType::Text,
                    registry_specs: &[],
                };
                match extractor.extract(chunk, &ctx).await {
                    Ok(result) => {
                        facts_with_audit += result.facts.len();
                        ents_with_audit.extend(result.entities);
                    }
                    Err(e) => eprintln!("  LLM error: {e}"),
                }
                // OOV audit
                let audit_adds = auditor.audit(chunk, &ents_with_audit);
                oov_adds_total += audit_adds.len();
                ents_with_audit.extend(audit_adds);
            }
            // Post-doc: scanner sweep
            let scanner_adds = scan_proper_nouns(&text, &ents_with_audit);
            ents_with_audit.extend(scanner_adds);
            dedup_entities(&mut ents_with_audit);
            let with_audit_names: Vec<String> = ents_with_audit
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();
            let (found, with_audit_recall, _, missed) = recall_detail(&with_audit_names, expected);

            let elapsed = start.elapsed().as_secs_f64();

            eprintln!(
                "{:<16} {:>6} {:>5.0}% {:>5.0}% {:>5} {:>5} {:>5} {:>6.1}s {:>5}",
                f.name,
                chunk_count,
                no_audit_recall * 100.0,
                with_audit_recall * 100.0,
                facts_no_audit,
                facts_with_audit,
                ents_with_audit.len(),
                elapsed,
                oov_adds_total,
            );

            // Detail: what the audit added
            let audit_unique: Vec<&String> = expected
                .iter()
                .filter(|e| {
                    with_audit_names.iter().any(|x| fuzzy_match(x, e))
                        && !no_audit_names.iter().any(|x| fuzzy_match(x, e))
                })
                .collect();
            if !audit_unique.is_empty() {
                eprintln!(
                    "  OOV audit found: {}",
                    audit_unique
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if !missed.is_empty() {
                eprintln!("  Still missed: {}", missed.join(", "));
            }
            eprintln!("  Recall: {}/{} entities found", found, expected.len());

            sum_no_audit += no_audit_recall;
            sum_with_audit += with_audit_recall;
            sum_rels[0] += facts_no_audit;
            sum_rels[1] += facts_with_audit;
            n += 1;
        }

        let nf = n as f64;
        eprintln!("{}", "─".repeat(75));
        eprintln!(
            "{:<16} {:>6} {:>5.0}% {:>5.0}% {:>5} {:>5}",
            "AVERAGE",
            "",
            sum_no_audit / nf * 100.0,
            sum_with_audit / nf * 100.0,
            sum_rels[0] / n,
            sum_rels[1] / n,
        );
        eprintln!(
            "\n════════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("NoAud = SingleCallExtractor + scanner (no OOV audit)");
        eprintln!("+Audit = SingleCallExtractor + OOV audit + scanner (full pipeline)");
        eprintln!("OOV+ = number of entities added by OOV audit across all chunks");
        eprintln!("Target: ≥95% document-level recall with +Audit");
        eprintln!(
            "════════════════════════════════════════════════════════════════════════════════"
        );
    }
}
