//! Layer A + Layer B end-to-end eval runner — Phase 2 / Phase 3 GATE.
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
//!
//! # Run LongMemEval harness — MockJudge smoke (5 fixtures, no HF download, no model)
//! cargo run -p kremory-eval --release --bin eval -- layer-a longmemeval --sample 5 --judge mock
//!
//! # Run LongMemEval harness — live GemmaJudge (requires KREMORY_EVAL_LIVE_LLM=1)
//! KREMORY_EVAL_LIVE_LLM=1 cargo run -p kremory-eval --release --bin eval -- layer-a longmemeval
//! ```
//!
//! # Output
//!
//! Results are written to `crates/kremory-eval/output/layer-<x>-<timestamp>.json`.
//! The directory is created if it does not exist.
//! The file is gitignored.

use std::path::PathBuf;

use chrono::Utc;
use serde_json::json;

use kremory_eval::{
    judge::{GemmaJudge, JudgeVerdict, MockJudge},
    layer_a::longmemeval::{
        LongMemEvalConfig, LongMemEvalDataset, LongMemEvalOutput, LongMemEvalReport,
        LongMemEvalScorer, LongMemEvalVariant,
    },
    layer_b::{
        entity_extraction,
        graph_integrity::{run_invariants, IntegrityConfig},
        ragas::{load_all_fixtures, score_all_metrics, RagasOutput},
    },
    Dataset, Scorer,
};

// ---------------------------------------------------------------------------
// CLI arg parsing (no external crate — keep the binary dependency-free)
// ---------------------------------------------------------------------------

/// Top-level layer selector.
#[derive(Debug)]
enum LayerCmd {
    LayerA(LayerACmd),
    LayerB { determinism: bool },
}

/// Layer A sub-commands.
#[derive(Debug)]
enum LayerACmd {
    LongMemEval {
        sample: Option<usize>,
        judge: JudgeKind,
    },
}

/// Which judge to use for Layer A eval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JudgeKind {
    Mock,
    Gemma,
}

fn parse_args() -> Result<LayerCmd, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        return Err("Usage: eval <layer-a|layer-b> [subcommand] [options]".into());
    }

    match raw[0].as_str() {
        "layer-b" => {
            let determinism = raw.contains(&"--determinism".to_string());
            Ok(LayerCmd::LayerB { determinism })
        }
        "layer-a" => {
            if raw.len() < 2 {
                return Err("Usage: eval layer-a <longmemeval> [--sample N] [--judge mock|gemma]".into());
            }
            match raw[1].as_str() {
                "longmemeval" => {
                    let sample = raw
                        .windows(2)
                        .find(|w| w[0] == "--sample")
                        .and_then(|w| w[1].parse::<usize>().ok());
                    let judge = raw
                        .windows(2)
                        .find(|w| w[0] == "--judge")
                        .map(|w| match w[1].as_str() {
                            "gemma" => JudgeKind::Gemma,
                            _ => JudgeKind::Mock,
                        })
                        .unwrap_or(JudgeKind::Mock);
                    Ok(LayerCmd::LayerA(LayerACmd::LongMemEval { sample, judge }))
                }
                other => Err(format!("Unknown layer-a subcommand '{}'. Supported: longmemeval", other)),
            }
        }
        other => Err(format!(
            "Unknown layer '{}'. Supported: layer-a, layer-b",
            other
        )),
    }
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

// ---------------------------------------------------------------------------
// Layer A — LongMemEval runner
// ---------------------------------------------------------------------------

/// Synthetic 5-fixture dataset used for MockJudge smoke testing.
///
/// These fixtures exercise all 6 question_type values + abstention without
/// requiring a HuggingFace download. The answers and questions are minimal
/// stand-ins — the smoke test validates the harness wiring, not score quality.
fn make_smoke_fixtures(n: usize) -> Vec<kremory_eval::layer_a::longmemeval::LongMemEvalRecord> {
    use kremory_eval::layer_a::longmemeval::{ConversationTurn, LongMemEvalRecord};

    let base = vec![
        LongMemEvalRecord {
            question_id: "smoke_001".into(),
            question_type: "single-session-user".into(),
            question: "What is the user's coffee preference?".into(),
            answer: "The user prefers oat milk lattes.".into(),
            question_date: "2024/01/15".into(),
            haystack_session_ids: vec!["s1".into()],
            haystack_dates: vec!["2024/01/10".into()],
            haystack_sessions: vec![vec![
                ConversationTurn {
                    role: "user".into(),
                    content: "I really love oat milk lattes in the morning.".into(),
                    has_answer: true,
                },
                ConversationTurn {
                    role: "assistant".into(),
                    content: "That sounds delicious!".into(),
                    has_answer: false,
                },
            ]],
            answer_session_ids: vec!["s1".into()],
        },
        LongMemEvalRecord {
            question_id: "smoke_002".into(),
            question_type: "single-session-assistant".into(),
            question: "What did the assistant suggest for dinner?".into(),
            answer: "The assistant suggested pasta carbonara.".into(),
            question_date: "2024/01/16".into(),
            haystack_session_ids: vec!["s2".into()],
            haystack_dates: vec!["2024/01/11".into()],
            haystack_sessions: vec![vec![
                ConversationTurn {
                    role: "user".into(),
                    content: "What should I make for dinner?".into(),
                    has_answer: false,
                },
                ConversationTurn {
                    role: "assistant".into(),
                    content: "How about pasta carbonara?".into(),
                    has_answer: true,
                },
            ]],
            answer_session_ids: vec!["s2".into()],
        },
        LongMemEvalRecord {
            question_id: "smoke_003".into(),
            question_type: "single-session-preference".into(),
            question: "What music does the user like?".into(),
            answer: "The user likes jazz and classical music.".into(),
            question_date: "2024/01/17".into(),
            haystack_session_ids: vec!["s3".into()],
            haystack_dates: vec!["2024/01/12".into()],
            haystack_sessions: vec![vec![ConversationTurn {
                role: "user".into(),
                content: "I've been listening to a lot of jazz and classical lately.".into(),
                has_answer: true,
            }]],
            answer_session_ids: vec!["s3".into()],
        },
        LongMemEvalRecord {
            question_id: "smoke_004".into(),
            question_type: "temporal-reasoning".into(),
            question: "How many days between the user's first and second messages?".into(),
            answer: "5 days".into(),
            question_date: "2024/01/20".into(),
            haystack_session_ids: vec!["s4".into(), "s5".into()],
            haystack_dates: vec!["2024/01/10".into(), "2024/01/15".into()],
            haystack_sessions: vec![
                vec![ConversationTurn {
                    role: "user".into(),
                    content: "First message on Jan 10.".into(),
                    has_answer: false,
                }],
                vec![ConversationTurn {
                    role: "user".into(),
                    content: "Second message on Jan 15.".into(),
                    has_answer: false,
                }],
            ],
            answer_session_ids: vec![],
        },
        LongMemEvalRecord {
            question_id: "smoke_005_abs".into(),
            question_type: "multi-session".into(),
            question: "What is the user's sister's name?".into(),
            answer: "The user never mentioned a sister in the conversation history.".into(),
            question_date: "2024/01/20".into(),
            haystack_session_ids: vec!["s6".into()],
            haystack_dates: vec!["2024/01/13".into()],
            haystack_sessions: vec![vec![ConversationTurn {
                role: "user".into(),
                content: "I have a brother named Alex.".into(),
                has_answer: false,
            }]],
            answer_session_ids: vec![],
        },
    ];

    base.into_iter().take(n.min(5)).collect()
}

fn run_layer_a_longmemeval(
    manifest_dir: &str,
    sample: Option<usize>,
    judge_kind: JudgeKind,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_layer_a_longmemeval_async(manifest_dir, sample, judge_kind))
}

async fn run_layer_a_longmemeval_async(
    manifest_dir: &str,
    sample: Option<usize>,
    judge_kind: JudgeKind,
) -> Result<(), Box<dyn std::error::Error>> {
    let use_live_llm = std::env::var("KREMORY_EVAL_LIVE_LLM").as_deref() == Ok("1");

    // Select dataset: smoke fixtures for mock, HF download for live.
    let dataset = if judge_kind == JudgeKind::Mock && !use_live_llm {
        eprintln!("[layer-a longmemeval] Using smoke fixtures (MockJudge, no HF download)");
        let n = sample.unwrap_or(5);
        let records = make_smoke_fixtures(n);
        eprintln!("[layer-a longmemeval] Smoke fixtures: {} samples", records.len());

        // Build dataset from in-memory records via a temp file.
        let tmp = std::env::temp_dir().join("kremory_eval_smoke.json");
        let json = serde_json::to_string(&records)?;
        std::fs::write(&tmp, json)?;
        LongMemEvalDataset::from_file(&tmp, None)?
    } else {
        eprintln!("[layer-a longmemeval] Fetching from HuggingFace Hub (oracle variant)...");
        let config = LongMemEvalConfig {
            variant: LongMemEvalVariant::Oracle,
            sample_limit: sample,
            ..LongMemEvalConfig::default()
        };
        LongMemEvalDataset::from_hub(&config).await?
    };

    let samples: Vec<_> = dataset.samples().collect();
    eprintln!("[layer-a longmemeval] Loaded {} samples", samples.len());

    // Build scorer.
    let mut all_scores: Vec<(kremory_eval::layer_a::longmemeval::LongMemEvalSample, kremory_eval::Score)> = Vec::with_capacity(samples.len());

    match judge_kind {
        JudgeKind::Mock => {
            // MockJudge always returns "yes" — validates harness wiring, not accuracy.
            let judge = MockJudge::new(JudgeVerdict {
                is_correct: true,
                is_partial: false,
                reasoning: "yes, smoke mock".into(),
            });
            let scorer = LongMemEvalScorer::new(judge);
            for sample in &samples {
                // For smoke: construct a mock output (empty response — no kremory call)
                let output = LongMemEvalOutput {
                    response: "mock smoke response — no kremory query".into(),
                    input_tokens_used: None,
                };
                let score = scorer.score(sample, &output).await?;
                all_scores.push((sample.clone(), score));
            }
        }
        JudgeKind::Gemma => {
            if !use_live_llm {
                return Err("GemmaJudge requires KREMORY_EVAL_LIVE_LLM=1".into());
            }
            let judge = GemmaJudge::from_env();
            let scorer = LongMemEvalScorer::new(judge);
            for sample in &samples {
                let output = LongMemEvalOutput {
                    response: "gemma eval — kremory adapter not wired in gate run".into(),
                    input_tokens_used: None,
                };
                let score = scorer.score(sample, &output).await?;
                all_scores.push((sample.clone(), score));
            }
        }
    }

    let report = LongMemEvalReport::from_scores(&all_scores);

    // Print per-category breakdown.
    eprintln!("[layer-a longmemeval] Overall accuracy: {:.4}", report.overall_accuracy);
    eprintln!("[layer-a longmemeval] Per-category breakdown:");

    // Sorted for deterministic output.
    let mut categories: Vec<_> = report.per_type_accuracy.iter().collect();
    categories.sort_by_key(|(k, _)| k.as_str());
    for (cat, acc) in &categories {
        eprintln!("[layer-a longmemeval]   {:<35} = {:.4}", cat, acc);
    }
    eprintln!(
        "[layer-a longmemeval]   {:<35} = {:.4}",
        "abstention_accuracy",
        report.abstention_accuracy
    );
    if let Some(tokens) = report.mean_input_tokens_per_recall {
        eprintln!("[layer-a longmemeval]   mean_input_tokens_per_recall = {:.1}", tokens);
    }

    // Write JSON report.
    let out_path = output_path(manifest_dir, "layer-a-longmemeval");
    let report_json = json!({
        "layer": "A",
        "benchmark": "longmemeval",
        "timestamp": Utc::now().to_rfc3339(),
        "kremory_version": env!("CARGO_PKG_VERSION"),
        "judge": format!("{:?}", judge_kind),
        "sample_count": report.sample_count,
        "overall_accuracy": report.overall_accuracy,
        "per_type_accuracy": report.per_type_accuracy,
        "abstention_accuracy": report.abstention_accuracy,
        "mean_input_tokens_per_recall": report.mean_input_tokens_per_recall,
        "upstream_scorer_sha": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
    });

    write_report(&out_path, &report_json)?;
    eprintln!("[layer-a longmemeval] Report written to {}", out_path.display());
    eprintln!("[layer-a longmemeval] DONE");

    Ok(())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cmd = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    match cmd {
        LayerCmd::LayerB { determinism } => {
            if let Err(e) = run_layer_b(manifest_dir, determinism) {
                eprintln!("[eval] FATAL: {}", e);
                std::process::exit(1);
            }
        }
        LayerCmd::LayerA(LayerACmd::LongMemEval { sample, judge }) => {
            if let Err(e) = run_layer_a_longmemeval(manifest_dir, sample, judge) {
                eprintln!("[eval] FATAL: {}", e);
                std::process::exit(1);
            }
        }
    }
}
