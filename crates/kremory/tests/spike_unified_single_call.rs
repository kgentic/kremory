#![cfg(any())]
//! PARKED 2026-05-18 — D.1a cycle-2 BYOM strict gate removed autoagents-llamacpp from rqlc.
//! Test depends on the concrete LlamaCppProvider. Restore via dedicated spike crate carve-out per `.claude/PARKING_LOT.md` 2026-05-18 entry. See ADR-Phase-D.0 §7.

/// Spike: SingleCallExtractor benchmark — exercises the real extraction pipeline
///
/// Runs SingleCallExtractor (the production extractor) against 8 domain fixtures
/// and measures entity recall + relationship count against ground truth.
///
/// Supports prompt version switching via PROMPT_VERSION env var (v1 or v2).
///
/// Run with:
///   RQL_QWEN3B_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_unified_single_call -- --nocapture
///
///   PROMPT_VERSION=v2 RQL_QWEN3B_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf \
///     cargo test --features llm -p rql-core --test spike_unified_single_call -- --nocapture

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "llm")]
mod spike {
    use std::sync::Arc;
    use std::time::Instant;

    use super::common::{
        dedup_entities, fixtures, load_auditor, load_ground_truth, merge_results, recall,
        ArchitectureBenchmarkRecord, JsonlBenchmarkWriter,
    };
    use kremory::core::config::ContentType;
    use kremory::core::extraction::{PromptVersion, SingleCallExtractor};
    use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
    use autoagents_llamacpp::LlamaCppProvider;
    use super::common::build_llm;

    async fn extract_with_optional_gleaning(
        extractor: &SingleCallExtractor<LlamaCppProvider>,
        text: &str,
        gleaning_rounds: u32,
    ) -> anyhow::Result<(ExtractionResult, u32)> {
        let base_ctx = ExtractionContext::default();
        let mut result = extractor.extract(text, &base_ctx).await?;
        let mut llm_calls = 1u32;

        for _ in 0..gleaning_rounds {
            let glean_ctx = ExtractionContext {
                known_entities: &result.entities,
                allowed_entity_types: &[],
                allowed_edge_types: &[],
                excluded_entity_types: &[],
                content_type: ContentType::Text,
            };
            let gleaned = extractor.extract(text, &glean_ctx).await?;
            llm_calls += 1;

            let merged = merge_results(&result, &gleaned);
            let entity_growth = merged.entities.len().saturating_sub(result.entities.len());
            let fact_growth = merged.facts.len().saturating_sub(result.facts.len());
            result = merged;

            if entity_growth == 0 && fact_growth == 0 {
                break;
            }
        }

        Ok((result, llm_calls))
    }

    // ─── Test ─────────────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spike_single_call_extraction() {
        let model_path = match std::env::var("RQL_QWEN3B_MODEL_PATH") {
            Ok(p) => p,
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
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1);
        let gt = load_ground_truth();
        let fixture_list = fixtures();
        let metrics_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/monitoring");
        let metrics_writer =
            JsonlBenchmarkWriter::new(metrics_dir, "architecture-bakeoff-single-call")
                .expect("open unified benchmark writer");

        let llm = Arc::new(
            build_llm(&model_path, 4096, 512)
                .await
                .expect("failed to build LlamaCppProvider"),
        );
        let extractor = SingleCallExtractor::new(llm).with_prompt_version(prompt_version);
        let auditor = load_auditor();

        eprintln!(
            "\n═══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SPIKE: SingleCallExtractor benchmark family");
        eprintln!("Model: {model_path}");
        eprintln!("Prompt: {prompt_version}");
        eprintln!("max_tokens: 512, temperature: 0.0 (greedy)");
        eprintln!("Gleaning rounds: {gleaning_rounds}");
        eprintln!("Safety net: OovAuditor (en_US Hunspell dictionary subtraction)");
        eprintln!(
            "═══════════════════════════════════════════════════════════════════════════════\n"
        );

        eprintln!(
            "{:<18} {:>6} {:>6} {:>6} {:>6} {:>5} {:>8} {:>8}",
            "Domain", "LLM", "+OOV", "+G+O", "Ents", "Rels", "Expect", "ms"
        );
        eprintln!("{}", "─".repeat(78));

        let mut total_llm_recall = 0.0f64;
        let mut total_combined_recall = 0.0f64;
        let mut total_gleaned_recall = 0.0f64;
        let mut total_entities = 0usize;
        let mut total_audit_added = 0usize;
        let mut total_glean_added = 0usize;
        let mut total_rels = 0usize;
        let mut total_ms = 0.0f64;
        let mut fixture_count = 0usize;

        for fixture in &fixture_list {
            let expected = match gt.get(fixture.key) {
                Some(e) => e,
                None => {
                    eprintln!("{:<18} SKIP — no ground truth", fixture.name);
                    continue;
                }
            };

            let text = match std::fs::read_to_string(fixture.path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{:<18} SKIP — {e}", fixture.name);
                    continue;
                }
            };
            let start = Instant::now();
            let (llm_result, llm_calls) =
                match extract_with_optional_gleaning(&extractor, &text, 0).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("{:<18} ERROR: {e}", fixture.name);
                        continue;
                    }
                };
            let baseline_latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            let glean_start = Instant::now();
            let (gleaned_result, glean_llm_calls) =
                match extract_with_optional_gleaning(&extractor, &text, gleaning_rounds).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("{:<18} ERROR: {e}", fixture.name);
                        continue;
                    }
                };
            let gleaning_latency_ms = glean_start.elapsed().as_secs_f64() * 1000.0;

            let llm_names: Vec<String> = llm_result
                .entities
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();
            let llm_recall = recall(&llm_names, expected);

            let audit_entities = auditor.audit(&text, &llm_result.entities);
            let audit_added = audit_entities.len();

            let mut combined = llm_result.entities.clone();
            combined.extend(audit_entities);
            dedup_entities(&mut combined);
            let combined_names: Vec<String> =
                combined.iter().map(|e| e.name.to_lowercase()).collect();
            let combined_recall = recall(&combined_names, expected);

            let glean_added = gleaned_result
                .entities
                .len()
                .saturating_sub(llm_result.entities.len());
            let glean_audit_entities = auditor.audit(&text, &gleaned_result.entities);
            let mut gleaned_combined = gleaned_result.entities.clone();
            gleaned_combined.extend(glean_audit_entities);
            dedup_entities(&mut gleaned_combined);
            let gleaned_names: Vec<String> = gleaned_combined
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect();
            let gleaned_recall = recall(&gleaned_names, expected);

            let rel_count = llm_result.facts.len();
            let gleaned_rel_count = gleaned_result.facts.len();

            eprintln!(
                "{:<18} {:>4.0}% {:>4.0}% {:>4.0}% {:>6} {:>5} {:>8} {:>7.0}",
                fixture.name,
                llm_recall * 100.0,
                combined_recall * 100.0,
                gleaned_recall * 100.0,
                gleaned_combined.len(),
                gleaned_rel_count,
                expected.len(),
                gleaning_latency_ms,
            );

            let llm_record = ArchitectureBenchmarkRecord {
                architecture: format!("single_call_{}", prompt_version),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall: llm_recall,
                entity_count: llm_result.entities.len(),
                expected_entity_count: expected.len(),
                relationship_count: rel_count,
                relationship_duplicates: None,
                latency_ms: baseline_latency_ms,
                document_level: false,
                llm_calls: Some(llm_calls),
                candidate_count: None,
                pipeline_ms: Some(baseline_latency_ms),
                parser_ok: Some(true),
                stage_label: Some("single_call_only".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            llm_record.record_metrics();
            let _ = metrics_writer.append(&llm_record);

            let oov_record = ArchitectureBenchmarkRecord {
                architecture: format!("single_call_plus_oov_{}", prompt_version),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall: combined_recall,
                entity_count: combined.len(),
                expected_entity_count: expected.len(),
                relationship_count: rel_count,
                relationship_duplicates: None,
                latency_ms: baseline_latency_ms,
                document_level: false,
                llm_calls: Some(llm_calls),
                candidate_count: Some(audit_added),
                pipeline_ms: Some(baseline_latency_ms),
                parser_ok: Some(true),
                stage_label: Some("single_call_plus_oov_audit".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            oov_record.record_metrics();
            let _ = metrics_writer.append(&oov_record);

            let glean_record = ArchitectureBenchmarkRecord {
                architecture: format!("single_call_plus_gleaning_plus_oov_{}", prompt_version),
                model: "qwen_3b".to_string(),
                fixture: fixture.name.to_string(),
                fixture_key: fixture.key.to_string(),
                entity_recall: gleaned_recall,
                entity_count: gleaned_combined.len(),
                expected_entity_count: expected.len(),
                relationship_count: gleaned_rel_count,
                relationship_duplicates: None,
                latency_ms: gleaning_latency_ms,
                document_level: false,
                llm_calls: Some(glean_llm_calls),
                candidate_count: Some(glean_added),
                pipeline_ms: Some(gleaning_latency_ms),
                parser_ok: Some(true),
                stage_label: Some("single_call_plus_gleaning_plus_oov_audit".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            glean_record.record_metrics();
            let _ = metrics_writer.append(&glean_record);

            total_llm_recall += llm_recall;
            total_combined_recall += combined_recall;
            total_gleaned_recall += gleaned_recall;
            total_entities += llm_result.entities.len();
            total_audit_added += audit_added;
            total_glean_added += glean_added;
            total_rels += gleaned_rel_count;
            total_ms += gleaning_latency_ms;
            fixture_count += 1;
        }

        if fixture_count == 0 {
            eprintln!("No fixtures ran — check RQL_QWEN3B_MODEL_PATH and fixture files.");
            return;
        }

        let n = fixture_count as f64;
        let avg_llm = total_llm_recall / n;
        let avg_combined = total_combined_recall / n;
        let avg_gleaned = total_gleaned_recall / n;

        eprintln!("{}", "─".repeat(78));
        eprintln!(
            "{:<18} {:>4.0}% {:>4.0}% {:>4.0}% {:>6} {:>5} {:>8} {:>7.0}",
            "AVERAGE",
            avg_llm * 100.0,
            avg_combined * 100.0,
            avg_gleaned * 100.0,
            total_entities + total_audit_added + total_glean_added,
            total_rels,
            "",
            total_ms,
        );

        eprintln!(
            "\n═══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("SUMMARY — SingleCall benchmark family ({prompt_version})");
        eprintln!("  Single-call recall:              {:.0}%", avg_llm * 100.0);
        eprintln!(
            "  Single-call + OOV audit:         {:.0}% ({:+.0}pp)",
            avg_combined * 100.0,
            (avg_combined - avg_llm) * 100.0
        );
        eprintln!(
            "  Single-call + glean + OOV audit: {:.0}% ({:+.0}pp vs single-call)",
            avg_gleaned * 100.0,
            (avg_gleaned - avg_llm) * 100.0
        );
        eprintln!("  Total entities:  {total_entities} (LLM)");
        eprintln!("  Audit adds:      {total_audit_added}");
        eprintln!("  Glean adds:      {total_glean_added}");
        eprintln!("  Total rels:      {total_rels}");
        eprintln!("  Total latency:   {:.1}s", total_ms / 1000.0);
        eprintln!("  Fixtures:        {fixture_count}");
        eprintln!("  Metrics:         {}", metrics_writer.path().display());
        eprintln!(
            "═══════════════════════════════════════════════════════════════════════════════"
        );
    }
}