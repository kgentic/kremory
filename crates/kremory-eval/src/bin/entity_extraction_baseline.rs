//! Entity extraction baseline runner — Layer B Day 3.
//!
//! Runs `scan_proper_nouns()` against all 14 domain fixtures and writes
//! the results to:
//!   - `crates/kremory-eval/output/entity-extraction-baseline.jsonl` (JSONL per-fixture)
//!   - `crates/kremory-eval/baselines/v0.1.4-diagnostic-entity-extraction.json` (baseline JSON)
//!
//! Run with:
//!   cargo run -p kremory-eval --bin entity_extraction_baseline

use std::path::PathBuf;

use kremory_eval::layer_b::entity_extraction;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let fixtures_dir = PathBuf::from(manifest_dir).join("fixtures");
    let output_dir = PathBuf::from(manifest_dir).join("output");
    let baselines_dir = PathBuf::from(manifest_dir).join("baselines");

    eprintln!("Running entity extraction eval against {}", fixtures_dir.display());

    let report = entity_extraction::run(&fixtures_dir)?;

    // Print summary table
    eprintln!(
        "\n{:<25} {:>9} {:>9} {:>9} {:>4} {:>4} {:>4}",
        "Domain", "Precision", "Recall", "F1", "TP", "FP", "FN"
    );
    eprintln!("{}", "─".repeat(72));
    for f in &report.fixtures {
        eprintln!(
            "{:<25} {:>9.3} {:>9.3} {:>9.3} {:>4} {:>4} {:>4}",
            f.domain, f.precision, f.recall, f.f1, f.tp, f.fp, f.r#fn
        );
    }
    eprintln!("{}", "─".repeat(72));
    eprintln!(
        "{:<25} {:>9.3} {:>9.3} {:>9.3} {:>4} {:>4} {:>4}",
        "OVERALL (macro-avg)",
        report.overall_precision,
        report.overall_recall,
        report.overall_f1,
        report.total_tp,
        report.total_fp,
        report.total_fn,
    );

    // Write JSONL output
    std::fs::create_dir_all(&output_dir)?;
    let jsonl_path = output_dir.join("entity-extraction-baseline.jsonl");
    entity_extraction::write_jsonl(&report, &jsonl_path)?;
    eprintln!("\nJSONL written to {}", jsonl_path.display());

    // Write baseline JSON
    std::fs::create_dir_all(&baselines_dir)?;
    let baseline_path =
        baselines_dir.join("v0.1.4-diagnostic-entity-extraction.json");

    let baseline = serde_json::json!({
        "schema_version": "1",
        "generated_at": report.eval_report.timestamp,
        "kremory_version": report.eval_report.kremory_version,
        "benchmark": "entity-extraction",
        "layer": "B",
        "method": "scan_proper_nouns (zero-LLM, Title Case heuristic)",
        "matching": "fuzzy substring (case-insensitive)",
        "overall": {
            "f1": report.overall_f1,
            "precision": report.overall_precision,
            "recall": report.overall_recall,
            "total_tp": report.total_tp,
            "total_fp": report.total_fp,
            "total_fn": report.total_fn,
            "fixture_count": report.fixtures.len(),
        },
        "fixtures": report.fixtures.iter().map(|f| serde_json::json!({
            "domain": f.domain,
            "fixture_key": f.fixture_key,
            "precision": f.precision,
            "recall": f.recall,
            "f1": f.f1,
            "tp": f.tp,
            "fp": f.fp,
            "fn": f.r#fn,
            "extracted_count": f.extracted_count,
            "expected_count": f.expected_count,
        })).collect::<Vec<_>>(),
    });

    let file = std::fs::File::create(&baseline_path)?;
    serde_json::to_writer_pretty(file, &baseline)?;
    eprintln!("Baseline written to {}", baseline_path.display());

    Ok(())
}
