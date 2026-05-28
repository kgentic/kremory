#![allow(clippy::unwrap_used, clippy::expect_used)]
/// NER extraction benchmark using GlinerExtractor across 14 domains.
///
/// Compares GLiNER zero-shot NER against the 208-entity ground truth.
/// All tests are #[ignore] (require ONNX model download from HuggingFace Hub).
///
/// Run with:
///   cargo test --features ner --test ner_benchmark -- --include-ignored --nocapture
///
/// Note: model init takes ~2s; the extractor is created once and reused across all domains.
#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "ner")]
mod ner_benchmark_tests {
    use kremory::core::{
        intelligence::{EntityExtractor, ExtractionContext},
        ner::GlinerExtractor,
    };
    use metrics_util::debugging::DebuggingRecorder;
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::time::Instant;

    use super::common::{self, ArchitectureBenchmarkRecord, JsonlBenchmarkWriter};

    // ─── Ground truth types ───────────────────────────────────────────────────

    #[derive(Debug, Deserialize)]
    struct GroundTruthEntity {
        name: String,
        label: String,
    }

    #[derive(Debug, Deserialize)]
    struct DomainGroundTruth {
        entities: Vec<GroundTruthEntity>,
        #[allow(dead_code)]
        min_relationships: usize,
    }

    fn load_ground_truth() -> HashMap<String, DomainGroundTruth> {
        let gt_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../kremory-eval/fixtures/ground_truth.json"
        );
        let raw = std::fs::read_to_string(gt_path)
            .unwrap_or_else(|e| panic!("failed to read ground_truth.json: {e}"));
        serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse ground_truth.json: {e}"))
    }

    // ─── Entity matching ──────────────────────────────────────────────────────

    /// Case-insensitive substring match: returns true when any extracted entity
    /// name contains the expected name or vice versa.
    fn entity_found(extracted_names: &[String], expected_name: &str) -> bool {
        let expected_lower = expected_name.to_lowercase();
        extracted_names.iter().any(|n| {
            let n_lower = n.to_lowercase();
            n_lower.contains(&expected_lower) || expected_lower.contains(&n_lower)
        })
    }

    // ─── Benchmark ────────────────────────────────────────────────────────────

    /// NER extraction benchmark: runs GlinerExtractor against all 14 domain
    /// fixtures and reports per-domain and overall recall against the 208-entity
    /// ground truth.
    ///
    /// The extractor is initialised once (expensive ~2s model load) and reused
    /// across all domains.  Extraction is timed per-domain.
    ///
    /// Requires: `onnx-community/gliner_large-v2.1` downloaded to HF cache.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires onnx-community/gliner_large-v2.1 download (~653 MB)"]
    async fn test_ner_extraction_all_domains() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let ground_truth = load_ground_truth();

        let mut domain_names: Vec<String> = ground_truth.keys().cloned().collect();
        domain_names.sort();

        // Load extractor once — model init is ~2s.
        let init_start = Instant::now();
        let extractor = GlinerExtractor::new().expect("model load failed");
        let init_elapsed = init_start.elapsed();
        eprintln!(
            "\nGlinerExtractor initialised in {:.0}ms",
            init_elapsed.as_secs_f64() * 1000.0
        );

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let eval_fixtures_dir = format!("{manifest_dir}/../kremory-eval/fixtures");

        let mut total_expected = 0usize;
        let mut total_found = 0usize;
        let benchmark_start = Instant::now();
        let jsonl_writer = JsonlBenchmarkWriter::new("logs", "architecture-bakeoff-ner")
            .expect("open unified benchmark writer");

        // Per-domain results accumulated for the summary table.
        let mut rows: Vec<(String, usize, usize, f64, u128)> = Vec::new();

        for domain_key in &domain_names {
            let domain = ground_truth
                .get(domain_key)
                .expect("domain not in ground truth");

            let fixture_path = format!("{eval_fixtures_dir}/{domain_key}.txt");
            let text = std::fs::read_to_string(&fixture_path)
                .unwrap_or_else(|e| panic!("failed to read fixture {fixture_path}: {e}"));

            // Collect unique labels from ground truth for this domain so the
            // benchmark exercises the same types the domain cares about.
            let mut domain_types: Vec<String> = domain
                .entities
                .iter()
                .map(|e| e.label.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            domain_types.sort();
            let ctx = ExtractionContext {
                allowed_entity_types: &domain_types,
                ..ExtractionContext::default()
            };
            let domain_start = Instant::now();
            let result = extractor
                .extract(&text, &ctx)
                .await
                .unwrap_or_else(|e| panic!("extraction failed for domain {domain_key}: {e}"));
            let domain_elapsed_ms = domain_start.elapsed().as_millis();

            let extracted_names: Vec<String> =
                result.entities.iter().map(|e| e.name.clone()).collect();

            let expected = domain.entities.len();
            let found = domain
                .entities
                .iter()
                .filter(|gt| entity_found(&extracted_names, &gt.name))
                .count();

            let recall = if expected == 0 {
                1.0_f64
            } else {
                found as f64 / expected as f64
            };

            total_expected += expected;
            total_found += found;

            // Record per-domain metrics.
            metrics::histogram!(
                "rql.ner_benchmark.domain_recall",
                "domain" => domain_key.clone()
            )
            .record(recall);
            metrics::histogram!(
                "rql.ner_benchmark.domain_extraction_ms",
                "domain" => domain_key.clone()
            )
            .record(domain_elapsed_ms as f64);

            let record = ArchitectureBenchmarkRecord {
                architecture: "ner_first_entities".to_string(),
                model: "gliner_large_v2_1".to_string(),
                fixture: domain_key.clone(),
                fixture_key: domain_key.clone(),
                entity_recall: recall,
                entity_count: extracted_names.len(),
                expected_entity_count: expected,
                relationship_count: 0,
                relationship_duplicates: None,
                latency_ms: domain_elapsed_ms as f64,
                document_level: false,
                llm_calls: Some(0),
                candidate_count: None,
                pipeline_ms: None,
                parser_ok: Some(true),
                stage_label: Some("ner_only".to_string()),
                timestamp: ArchitectureBenchmarkRecord::now_timestamp(),
            };
            record.record_metrics();
            let _ = jsonl_writer.append(&record);

            rows.push((
                domain_key.clone(),
                expected,
                found,
                recall,
                domain_elapsed_ms,
            ));
        }

        let total_elapsed = benchmark_start.elapsed();
        let overall_recall = if total_expected == 0 {
            1.0_f64
        } else {
            total_found as f64 / total_expected as f64
        };

        // Record overall metrics.
        metrics::gauge!("rql.ner_benchmark.overall_recall").set(overall_recall);
        metrics::gauge!("rql.ner_benchmark.total_entities_expected").set(total_expected as f64);
        metrics::gauge!("rql.ner_benchmark.total_entities_found").set(total_found as f64);
        metrics::histogram!("rql.ner_benchmark.total_elapsed_ms")
            .record(total_elapsed.as_millis() as f64);

        // ─── Summary table ────────────────────────────────────────────────────

        let col_domain = 22usize;
        let col_expected = 8usize;
        let col_found = 7usize;
        let col_recall = 8usize;
        let col_time = 8usize;

        eprintln!("\n{:-<64}", "");
        eprintln!("  NER Extraction Benchmark — GlinerExtractor (all domains)");
        eprintln!("{:-<64}", "");
        eprintln!(
            "  {:<col_domain$} | {:>col_expected$} | {:>col_found$} | {:>col_recall$} | {:>col_time$}",
            "Domain", "Expected", "Found", "Recall", "Time"
        );
        eprintln!(
            "  {:-<col_domain$}-+-{:-<col_expected$}-+-{:-<col_found$}-+-{:-<col_recall$}-+-{:-<col_time$}",
            "", "", "", "", ""
        );

        for (domain, expected, found, recall, elapsed_ms) in &rows {
            eprintln!(
                "  {:<col_domain$} | {:>col_expected$} | {:>col_found$} | {:>7.1}% | {:>6}ms",
                domain,
                expected,
                found,
                recall * 100.0,
                elapsed_ms
            );
        }

        eprintln!(
            "  {:-<col_domain$}-+-{:-<col_expected$}-+-{:-<col_found$}-+-{:-<col_recall$}-+-{:-<col_time$}",
            "", "", "", "", ""
        );
        eprintln!(
            "  {:<col_domain$} | {:>col_expected$} | {:>col_found$} | {:>7.1}% | {:>5.1}s",
            "OVERALL",
            total_expected,
            total_found,
            overall_recall * 100.0,
            total_elapsed.as_secs_f64()
        );
        eprintln!("{:-<64}\n", "");

        // ─── Export metrics ───────────────────────────────────────────────────

        let exporter = common::MetricsExporter::new("logs");
        exporter
            .export(&snapshotter, "ner-benchmark-all-domains")
            .expect("metrics export failed");
        eprintln!("Unified metrics: {}", jsonl_writer.path().display());

        // ─── Assertion ────────────────────────────────────────────────────────

        assert!(
            overall_recall >= 0.50,
            "overall NER recall {:.1}% is below 50% threshold ({}/{} found)",
            overall_recall * 100.0,
            total_found,
            total_expected,
        );
    }
}
