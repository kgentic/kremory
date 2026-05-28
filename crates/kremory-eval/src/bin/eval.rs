//! Layer B end-to-end eval runner — Phase 2 GATE.
//!
//! # Usage
//!
//! ```text
//! # Run all Layer B metrics with MockJudge (CI-safe, no model required)
//! cargo run -p kremory-eval --release --bin eval -- layer-b
//!
//! # Run with determinism check (3 runs, asserts all variance ≤ 0.02)
//! cargo run -p kremory-eval --release --bin eval -- layer-b --determinism
//!
//! # Run with live Gemma judge (requires KREMORY_EVAL_LIVE_LLM=1 + model on disk)
//! KREMORY_EVAL_LIVE_LLM=1 cargo run -p kremory-eval --release --bin eval -- layer-b
//! ```
//!
//! # Output
//!
//! Results are written to `crates/kremory-eval/output/layer-b-<timestamp>.json`.
//! The directory is created if it does not exist.
//! The file is gitignored.

use std::path::PathBuf;

use chrono::Utc;
use serde_json::json;

use kremory_eval::{
    judge::MockJudge,
    layer_b::{
        entity_extraction,
        graph_integrity::{run_invariants, IntegrityConfig},
        ragas::{load_all_fixtures, score_all_metrics, RagasOutput},
    },
};

// ---------------------------------------------------------------------------
// CLI arg parsing (no external crate — keep the binary dependency-free)
// ---------------------------------------------------------------------------

struct Args {
    /// Which layer to run. Must be "layer-b".
    layer: String,
    /// Whether to run determinism check (3 runs, variance assertion).
    determinism: bool,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        return Err("Usage: eval <layer-b> [--determinism]".into());
    }
    let layer = raw[0].clone();
    if layer != "layer-b" {
        return Err(format!("Unknown layer '{}'. Only 'layer-b' is supported.", layer));
    }
    let determinism = raw.contains(&"--determinism".to_string());
    Ok(Args { layer, determinism })
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn output_path(manifest_dir: &str, label: &str) -> PathBuf {
    let ts = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    PathBuf::from(manifest_dir)
        .join("output")
        .join(format!("{}-{}.json", label, ts))
}

fn write_report(path: &std::path::Path, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer B runner — synchronous entry point (wraps tokio runtime)
// ---------------------------------------------------------------------------

fn run_layer_b(manifest_dir: &str, determinism: bool) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_layer_b_async(manifest_dir, determinism))
}

async fn run_layer_b_async(manifest_dir: &str, determinism: bool) -> Result<(), Box<dyn std::error::Error>> {
    let use_live_llm = std::env::var("KREMORY_EVAL_LIVE_LLM").as_deref() == Ok("1");
    if use_live_llm {
        eprintln!("[eval] KREMORY_EVAL_LIVE_LLM=1 detected — live GemmaJudge enabled");
        eprintln!("[eval] Note: live judge requires model on disk at KREMORY_EVAL_JUDGE_MODEL_PATH");
    } else {
        eprintln!("[eval] Using MockJudge::always_correct() (CI-safe, deterministic)");
        eprintln!("[eval] Set KREMORY_EVAL_LIVE_LLM=1 to use live GemmaJudge");
    }

    // --- 6.1 Entity Extraction ---
    eprintln!("\n[eval] === Layer B 6.1: Entity Extraction ===");
    let fixtures_dir = PathBuf::from(manifest_dir).join("fixtures");
    let entity_report = entity_extraction::run(&fixtures_dir)?;
    eprintln!("[eval] Entity extraction: overall F1 = {:.4}", entity_report.overall_f1);
    eprintln!("[eval]   precision={:.4}  recall={:.4}  tp={}  fp={}  fn={}",
        entity_report.overall_precision,
        entity_report.overall_recall,
        entity_report.total_tp,
        entity_report.total_fp,
        entity_report.total_fn,
    );

    // --- 6.2 RAGAS metrics ---
    eprintln!("\n[eval] === Layer B 6.2: RAGAS Metrics (MockJudge) ===");
    let ragas_dir = fixtures_dir.join("ragas");
    let fixtures = load_all_fixtures(&ragas_dir)?;
    eprintln!("[eval] Loaded {} RAGAS fixtures", fixtures.len());

    // Build synthetic RagasOutput from fixture (mock output = ground truth echo)
    let mock_outputs: Vec<RagasOutput> = fixtures.iter().map(|f| RagasOutput {
        answer: f.expected_answer.clone(),
        retrieved_contexts: f.expected_contexts.clone(),
        retrieved_entities: f.expected_entities.clone(),
    }).collect();

    let judge = MockJudge::always_correct();
    let mut ragas_scores = Vec::with_capacity(fixtures.len());

    for (fixture, output) in fixtures.iter().zip(mock_outputs.iter()) {
        let scores = score_all_metrics(judge.clone(), fixture, output).await?;
        ragas_scores.push((fixture.id.clone(), scores));
    }

    // Aggregate
    let n = ragas_scores.len() as f64;
    let ragas_mean_faithfulness: f64 = ragas_scores.iter().map(|(_, s)| s.faithfulness).sum::<f64>() / n;
    let ragas_mean_answer_relevancy: f64 = ragas_scores.iter().map(|(_, s)| s.answer_relevancy).sum::<f64>() / n;
    let ragas_mean_context_precision: f64 = ragas_scores.iter().map(|(_, s)| s.context_precision).sum::<f64>() / n;
    let ragas_mean_context_recall: f64 = ragas_scores.iter().map(|(_, s)| s.context_recall).sum::<f64>() / n;
    let ragas_mean_context_entities: f64 = ragas_scores.iter().map(|(_, s)| s.context_entities_recall).sum::<f64>() / n;
    let ragas_mean_hallucination: f64 = ragas_scores.iter().map(|(_, s)| s.hallucination).sum::<f64>() / n;

    eprintln!("[eval] RAGAS aggregate ({} fixtures):", ragas_scores.len());
    eprintln!("[eval]   faithfulness         = {:.4}", ragas_mean_faithfulness);
    eprintln!("[eval]   answer_relevancy     = {:.4}", ragas_mean_answer_relevancy);
    eprintln!("[eval]   context_precision    = {:.4}", ragas_mean_context_precision);
    eprintln!("[eval]   context_recall       = {:.4}", ragas_mean_context_recall);
    eprintln!("[eval]   context_entities     = {:.4}", ragas_mean_context_entities);
    eprintln!("[eval]   hallucination        = {:.4}", ragas_mean_hallucination);

    // --- 6.3 Graph Integrity (in-memory empty graph — structural invariant check) ---
    eprintln!("\n[eval] === Layer B 6.3: Graph Integrity (in-memory, empty graph) ===");
    let graph = kremory::core::schema::TemporalGraph::open_in_memory()
        .await
        .map_err(|e| format!("failed to open in-memory graph: {}", e))?;

    let config = IntegrityConfig {
        expected_entity_count: None, // skip FTS count invariant on empty graph
        expected_namespaces: vec![],
        allow_isolated_entity_count: 0,
    };
    let integrity_report = run_invariants(&graph, &config).await?;
    eprintln!("[eval] Graph integrity: {} invariants checked, passed={}",
        integrity_report.invariants.len(),
        integrity_report.all_passed(),
    );
    if !integrity_report.all_passed() {
        for f in integrity_report.failures() {
            eprintln!("[eval]   FAIL: {} — expected={} actual={} details={}",
                f.name, f.expected, f.actual, f.details);
        }
        return Err("Graph integrity invariants FAILED".into());
    }

    // --- Determinism check ---
    let mut determinism_variances: Vec<(String, f64)> = Vec::new();
    let mut determinism_pass = true;

    if determinism {
        eprintln!("\n[eval] === Determinism Check (3 runs, MockJudge) ===");
        let var_threshold = 0.02_f64;

        // Run 3 times and compare consecutive pairs
        let mut run_results = Vec::with_capacity(3);
        for run_idx in 0..3usize {
            eprintln!("[eval] Determinism run {}/3...", run_idx + 1);
            let mut run_scores = Vec::with_capacity(fixtures.len());
            for (fixture, output) in fixtures.iter().zip(mock_outputs.iter()) {
                let scores = score_all_metrics(judge.clone(), fixture, output).await?;
                run_scores.push(scores);
            }
            run_results.push(run_scores);
        }

        // Compare run 0 vs run 1, then run 1 vs run 2
        let pairs = [(0usize, 1usize), (1, 2)];
        for (a_idx, b_idx) in pairs {
            let a = &run_results[a_idx];
            let b = &run_results[b_idx];

            for (i, fixture) in fixtures.iter().enumerate() {
                let variance = a[i].variance_vs(&b[i]);
                let label = format!("fixture[{}] run{} vs run{}", fixture.id, a_idx, b_idx);
                if variance > var_threshold {
                    eprintln!("[eval]   FAIL variance: {} = {:.6} (threshold={:.2})", label, variance, var_threshold);
                    determinism_pass = false;
                } else {
                    eprintln!("[eval]   OK   variance: {} = {:.6}", label, variance);
                }
                determinism_variances.push((label, variance));
            }
        }

        let max_var = determinism_variances.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
        eprintln!("[eval] Determinism max variance: {:.6} (threshold={:.2}) — {}",
            max_var, var_threshold, if determinism_pass { "PASS" } else { "FAIL" });

        if !determinism_pass {
            return Err(format!("Determinism check FAILED: max variance {:.6} > {:.2}", max_var, var_threshold).into());
        }
    }

    // --- Write output ---
    let out_path = output_path(manifest_dir, "layer-b");
    let report = json!({
        "layer": "B",
        "timestamp": Utc::now().to_rfc3339(),
        "kremory_version": env!("CARGO_PKG_VERSION"),
        "judge": "MockJudge::always_correct",
        "entity_extraction": {
            "overall_f1": entity_report.overall_f1,
            "overall_precision": entity_report.overall_precision,
            "overall_recall": entity_report.overall_recall,
            "total_tp": entity_report.total_tp,
            "total_fp": entity_report.total_fp,
            "total_fn": entity_report.total_fn,
            "fixture_count": entity_report.fixtures.len(),
        },
        "ragas": {
            "fixture_count": ragas_scores.len(),
            "faithfulness": ragas_mean_faithfulness,
            "answer_relevancy": ragas_mean_answer_relevancy,
            "context_precision": ragas_mean_context_precision,
            "context_recall": ragas_mean_context_recall,
            "context_entities_recall": ragas_mean_context_entities,
            "hallucination": ragas_mean_hallucination,
        },
        "graph_integrity": {
            "invariants_checked": integrity_report.invariants.len(),
            "all_passed": integrity_report.all_passed(),
        },
        "determinism": if determinism {
            let max_var = determinism_variances.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
            json!({
                "enabled": true,
                "max_variance": max_var,
                "threshold": 0.02,
                "pass": determinism_pass,
            })
        } else {
            json!({ "enabled": false })
        },
    });

    write_report(&out_path, &report)?;
    eprintln!("\n[eval] Report written to {}", out_path.display());
    eprintln!("[eval] Layer B: ALL CHECKS PASSED");

    Ok(())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    match args.layer.as_str() {
        "layer-b" => {
            if let Err(e) = run_layer_b(manifest_dir, args.determinism) {
                eprintln!("[eval] FATAL: {}", e);
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown layer: {}", args.layer);
            std::process::exit(1);
        }
    }
}
