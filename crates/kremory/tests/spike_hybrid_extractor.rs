#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: HybridExtractor benchmark over the shared 14-domain fixture set
///
/// Runs the staged HybridExtractor against the same fixture corpus and unified
/// JSONL schema used by the single-call bakeoff.
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=../models/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_hybrid_extractor -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::{
        fixtures, load_auditor, load_ground_truth, recall, ArchitectureBenchmarkRecord,
        JsonlBenchmarkWriter,
    };
    use kremory::core::config::ContentType;
    use kremory::core::extraction::PromptVersion;
    use kremory::core::hybrid_extractor::{ExtractionConfig, HybridExtractor};
    use kremory::core::intelligence::{EntityExtractor, ExtractionContext};
    use super::common::build_llm;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_hybrid_extractor_bakeoff() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("SKIP: RQL_QWEN3B_MODEL_PATH not set");
                return;
            }
        };

        let prompt_version = match std::env::var("PROMPT_VERSION").as_deref() {
            Ok("v2") => PromptVersion::V2SchemaLight,
            Ok("v3") => PromptVersion::V3SchemaHybrid,
            _ => PromptVersion::V1Rules,
        };
        let gleaning_rounds = std::env::var("GLEANING_ROUNDS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(1);

        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );
        let auditor = Arc::new(load_auditor());
        let extractor = HybridExtractor::new(llm)
            .with_prompt_version(prompt_version)
            .with_auditor(auditor)
            .with_config(ExtractionConfig {
                gleaning_rounds,
                ..ExtractionConfig::default()
            });

        let gt = load_ground_truth();
        let fixture_list = fixtures();
        let metrics_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring");
        let metrics_writer =
            JsonlBenchmarkWriter::new(metrics_dir, "architecture-bakeoff-hybrid-extractor")
                .expect("open hybrid benchmark writer");

        eprintln!(
            "\n═══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: HybridExtractor benchmark");
        eprintln!("Model: {model_path}");
        eprintln!("Prompt: {prompt_version}");
        eprintln!("Gleaning rounds: {gleaning_rounds}");
        eprintln!("Pipeline: Stage1 + Stage2 + Stage3 + Stage4 + Stage5");
        eprintln!(
            "═══════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>5} {:>8} {:>8}",
            "Domain", "Recall", "Ents", "Rels", "Expect", "ms"
        );
        eprintln!("{}", "─".repeat(64));

        let mut total_recall = 0.0f64;
        let mut total_entities = 0usize;
        let mut total_relationships = 0usize;
        let mut total_ms = 0.0f64;
        let mut fixture_count = 0usize;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(value) => value,
                None => {
                    eprintln!("{:<18} SKIP — no ground truth", fixture.name);
                    continue;
                }
            };

            let text = match std::fs::read_to_string(fixture.path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{:<18} SKIP — {error}", fixture.name);
                    continue;
                }
            };

            let start = Instant::now();
            let result = match extractor
                .extract(
                    &text,
                    &ExtractionContext {
                        content_type: ContentType::Text,
                        ..ExtractionContext::default()
                    },
                )
                .await
            {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{:<18} ERROR: {error}", fixture.name);
                    continue;
                }
            };
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            let names: Vec<String> = result
                .entities
                .iter()
                .map(|entity| entity.name.to_lowercase())
                .collect();
            let entity_recall = recall(&names, expected);
            let relationship_count = result.facts.len();

            eprintln!(
                "{:<18} {:>4.0}% {:>6} {:>5} {:>8} {:>7.0}",
                fixture.name,
                entity_recall * 100.0,
                result.entities.len(),
                relationship_count,
                expected.len(),
                latency_ms,
            );

            let record = ArchitectureBenchmarkRecord {
                architecture: format!("hybrid_extractor_{}", prompt_version),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall,
                entity_count: result.entities.len(),
                expected_entity_count: expected.len(),
                relationship_count,
                relationship_duplicates: None,
                latency_ms,
                document_level: false,
                llm_calls: Some(1 + gleaning_rounds + u32::from(!result.entities.is_empty())),
                candidate_count: None,
                pipeline_ms: Some(latency_ms),
                parser_ok: Some(true),
                stage_label: Some("hybrid_extractor_full_pipeline".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            record.record_metrics();
            let _ = metrics_writer.append(&record);

            total_recall += entity_recall;
            total_entities += result.entities.len();
            total_relationships += relationship_count;
            total_ms += latency_ms;
            fixture_count += 1;
        }

        if fixture_count == 0 {
            eprintln!("No fixtures ran — check RQL_QWEN3B_MODEL_PATH and fixture files.");
            return;
        }

        let average_recall = total_recall / fixture_count as f64;

        eprintln!("{}", "─".repeat(64));
        eprintln!(
            "{:<18} {:>4.0}% {:>6} {:>5} {:>8} {:>7.0}",
            "AVERAGE",
            average_recall * 100.0,
            total_entities,
            total_relationships,
            "",
            total_ms,
        );

        eprintln!(
            "\n═══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SUMMARY — HybridExtractor ({prompt_version})");
        eprintln!("  Entity recall: {:.0}%", average_recall * 100.0);
        eprintln!("  Total entities: {total_entities}");
        eprintln!("  Total rels:     {total_relationships}");
        eprintln!("  Total latency:  {:.1}s", total_ms / 1000.0);
        eprintln!("  Fixtures:       {fixture_count}");
        eprintln!("  Metrics:        {}", metrics_writer.path().display());
        eprintln!(
            "═══════════════════════════════════════════════════════════════════════════════"
        );
    }
}