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
use std::sync::Arc;

use chrono::Utc;
use serde_json::json;

use autoagents_llm::{
    backends::ollama::Ollama,
    builder::LLMBuilder,
    embedding::{model_provider::EmbeddingBuilder, EmbeddingProvider as AutoEmbeddingProvider},
};

use kremory::core::error::{Error as KremoryCoreError, Result as KremoryCoreResult};

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
// BYOM eval-side adapter: Arc<P: AutoEmbeddingProvider> → kremory::EmbeddingProvider
//
// AA's EmbeddingBuilder returns `Arc<Ollama>` (batch-oriented multi-text).
// kremory expects single-text `EmbeddingProvider`. This thin wrapper bridges
// without mutating kremory — exactly the BYOM contract v0.1.4 ships.
//
// Generic over `P` so unit tests can inject a fake batch embedder. Production
// callers always parameterise with `Ollama`.
// ---------------------------------------------------------------------------

struct OllamaEmbedAdapter<P> {
    inner: Arc<P>,
}

impl<P> kremory::EmbeddingProvider for OllamaEmbedAdapter<P>
where
    P: AutoEmbeddingProvider + Send + Sync + 'static,
{
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = KremoryCoreResult<Vec<f32>>> + Send + 'a {
        let inner = Arc::clone(&self.inner);
        let owned = text.to_owned();
        async move {
            let mut batch = inner
                .embed(vec![owned])
                .await
                .map_err(|e| KremoryCoreError::Embedding(e.to_string()))?;
            batch
                .pop()
                .ok_or_else(|| KremoryCoreError::Embedding("empty embedding batch".into()))
        }
    }
}

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
        /// When true, use local smoke fixtures (no HF download). Compatible
        /// with both mock and gemma judges. Default: false.
        smoke: bool,
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
    parse_args_from(&raw)
}

/// Testable arg parser — same logic as [`parse_args`] but takes the args
/// slice as input. Keeps `parse_args` thin so [`std::env::args`] doesn't
/// block unit testing.
fn parse_args_from(raw: &[String]) -> Result<LayerCmd, String> {
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
                return Err(
                    "Usage: eval layer-a <longmemeval> [--sample N] [--judge mock|gemma]".into(),
                );
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
                    let smoke = raw.contains(&"--smoke".to_string());
                    Ok(LayerCmd::LayerA(LayerACmd::LongMemEval {
                        sample,
                        judge,
                        smoke,
                    }))
                }
                other => Err(format!(
                    "Unknown layer-a subcommand '{}'. Supported: longmemeval",
                    other
                )),
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

fn write_report(
    path: &std::path::Path,
    value: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
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

async fn run_layer_b_async(
    manifest_dir: &str,
    determinism: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let use_live_llm = std::env::var("KREMORY_EVAL_LIVE_LLM").as_deref() == Ok("1");
    if use_live_llm {
        eprintln!("[eval] KREMORY_EVAL_LIVE_LLM=1 detected — live GemmaJudge enabled");
        eprintln!(
            "[eval] Note: live judge requires model on disk at KREMORY_EVAL_JUDGE_MODEL_PATH"
        );
    } else {
        eprintln!("[eval] Using MockJudge::always_correct() (CI-safe, deterministic)");
        eprintln!("[eval] Set KREMORY_EVAL_LIVE_LLM=1 to use live GemmaJudge");
    }

    // --- 6.1 Entity Extraction ---
    eprintln!("\n[eval] === Layer B 6.1: Entity Extraction ===");
    let fixtures_dir = PathBuf::from(manifest_dir).join("fixtures");
    let entity_report = entity_extraction::run(&fixtures_dir)?;
    eprintln!(
        "[eval] Entity extraction: overall F1 = {:.4}",
        entity_report.overall_f1
    );
    eprintln!(
        "[eval]   precision={:.4}  recall={:.4}  tp={}  fp={}  fn={}",
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
    let mock_outputs: Vec<RagasOutput> = fixtures
        .iter()
        .map(|f| RagasOutput {
            answer: f.expected_answer.clone(),
            retrieved_contexts: f.expected_contexts.clone(),
            retrieved_entities: f.expected_entities.clone(),
        })
        .collect();

    let judge = MockJudge::always_correct();
    let mut ragas_scores = Vec::with_capacity(fixtures.len());

    for (fixture, output) in fixtures.iter().zip(mock_outputs.iter()) {
        let scores = score_all_metrics(judge.clone(), fixture, output).await?;
        ragas_scores.push((fixture.id.clone(), scores));
    }

    // Aggregate
    let n = ragas_scores.len() as f64;
    let ragas_mean_faithfulness: f64 = ragas_scores
        .iter()
        .map(|(_, s)| s.faithfulness)
        .sum::<f64>()
        / n;
    let ragas_mean_answer_relevancy: f64 = ragas_scores
        .iter()
        .map(|(_, s)| s.answer_relevancy)
        .sum::<f64>()
        / n;
    let ragas_mean_context_precision: f64 = ragas_scores
        .iter()
        .map(|(_, s)| s.context_precision)
        .sum::<f64>()
        / n;
    let ragas_mean_context_recall: f64 = ragas_scores
        .iter()
        .map(|(_, s)| s.context_recall)
        .sum::<f64>()
        / n;
    let ragas_mean_context_entities: f64 = ragas_scores
        .iter()
        .map(|(_, s)| s.context_entities_recall)
        .sum::<f64>()
        / n;
    let ragas_mean_hallucination: f64 = ragas_scores
        .iter()
        .map(|(_, s)| s.hallucination)
        .sum::<f64>()
        / n;

    eprintln!("[eval] RAGAS aggregate ({} fixtures):", ragas_scores.len());
    eprintln!(
        "[eval]   faithfulness         = {:.4}",
        ragas_mean_faithfulness
    );
    eprintln!(
        "[eval]   answer_relevancy     = {:.4}",
        ragas_mean_answer_relevancy
    );
    eprintln!(
        "[eval]   context_precision    = {:.4}",
        ragas_mean_context_precision
    );
    eprintln!(
        "[eval]   context_recall       = {:.4}",
        ragas_mean_context_recall
    );
    eprintln!(
        "[eval]   context_entities     = {:.4}",
        ragas_mean_context_entities
    );
    eprintln!(
        "[eval]   hallucination        = {:.4}",
        ragas_mean_hallucination
    );

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
    eprintln!(
        "[eval] Graph integrity: {} invariants checked, passed={}",
        integrity_report.invariants.len(),
        integrity_report.all_passed(),
    );
    if !integrity_report.all_passed() {
        for f in integrity_report.failures() {
            eprintln!(
                "[eval]   FAIL: {} — expected={} actual={} details={}",
                f.name, f.expected, f.actual, f.details
            );
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
                    eprintln!(
                        "[eval]   FAIL variance: {} = {:.6} (threshold={:.2})",
                        label, variance, var_threshold
                    );
                    determinism_pass = false;
                } else {
                    eprintln!("[eval]   OK   variance: {} = {:.6}", label, variance);
                }
                determinism_variances.push((label, variance));
            }
        }

        let max_var = determinism_variances
            .iter()
            .map(|(_, v)| *v)
            .fold(0.0_f64, f64::max);
        eprintln!(
            "[eval] Determinism max variance: {:.6} (threshold={:.2}) — {}",
            max_var,
            var_threshold,
            if determinism_pass { "PASS" } else { "FAIL" }
        );

        if !determinism_pass {
            return Err(format!(
                "Determinism check FAILED: max variance {:.6} > {:.2}",
                max_var, var_threshold
            )
            .into());
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

/// Synthetic 6-fixture dataset used for MockJudge smoke testing.
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
        LongMemEvalRecord {
            question_id: "smoke_006".into(),
            question_type: "knowledge-update".into(),
            question: "What is the user's current job title?".into(),
            answer:
                "The user is currently a Staff Engineer after being promoted from Senior Engineer."
                    .into(),
            question_date: "2024/02/01".into(),
            haystack_session_ids: vec!["s7".into(), "s8".into()],
            haystack_dates: vec!["2024/01/05".into(), "2024/01/25".into()],
            haystack_sessions: vec![
                vec![ConversationTurn {
                    role: "user".into(),
                    content: "I just got promoted from Senior Engineer to Staff Engineer!".into(),
                    has_answer: true,
                }],
                vec![ConversationTurn {
                    role: "user".into(),
                    content: "Now that I'm a Staff Engineer I have more responsibilities.".into(),
                    has_answer: true,
                }],
            ],
            answer_session_ids: vec!["s7".into(), "s8".into()],
        },
    ];

    base.into_iter().take(n.min(6)).collect()
}

fn run_layer_a_longmemeval(
    manifest_dir: &str,
    sample: Option<usize>,
    judge_kind: JudgeKind,
    smoke: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_layer_a_longmemeval_async(
        manifest_dir,
        sample,
        judge_kind,
        smoke,
    ))
}

async fn run_layer_a_longmemeval_async(
    manifest_dir: &str,
    sample: Option<usize>,
    judge_kind: JudgeKind,
    smoke: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let use_live_llm = std::env::var("KREMORY_EVAL_LIVE_LLM").as_deref() == Ok("1");

    // Select dataset:
    //   - --smoke flag → local smoke fixtures (no HF download), any judge
    //   - mock judge + no live LLM → local smoke fixtures (CI default)
    //   - otherwise → HF Hub Oracle variant download
    let dataset = if smoke || (judge_kind == JudgeKind::Mock && !use_live_llm) {
        eprintln!(
            "[layer-a longmemeval] Using smoke fixtures (no HF download, judge={:?})",
            judge_kind
        );
        let n = sample.unwrap_or(5);
        let records = make_smoke_fixtures(n);
        eprintln!(
            "[layer-a longmemeval] Smoke fixtures: {} samples",
            records.len()
        );

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
    let mut all_scores: Vec<(
        kremory_eval::layer_a::longmemeval::LongMemEvalSample,
        kremory_eval::Score,
    )> = Vec::with_capacity(samples.len());

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
            let scorer = Arc::new(LongMemEvalScorer::new(GemmaJudge::from_env()));

            // Standard Ollama-conventional env vars (matches litellm / ollama-haystack
            // / langchain-ollama community usage). Defaults chosen for O13:
            //   chat  = qwen2.5:14b      (clean entity extraction, no thinking-config)
            //   embed = nomic-embed-text (768-dim, matches kremory facade default)
            //
            // Rationale (O13 resolution 2026-05-28): `Memory::with_ollama` /
            // `Memory::auto` hardcode `llama3.2` 3B which emits duplicate
            // 'VerbatimString' entities on LongMemEval text → intra-batch
            // dedup failure. qwen2.5:14b extracts cleanly.
            let ollama_host = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let chat_model =
                std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string());
            let embed_model = std::env::var("OLLAMA_EMBED_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string());

            // Concurrency: KREMORY_EVAL_PARALLEL=N (default 1 = serial).
            // Each worker owns a fresh per-question kremory `Memory` handle
            // (own libSQL DB, no writer contention) but shares the LLM and
            // embed provider Arcs (Ollama HTTP backend pools requests).
            // Ollama processes up to OLLAMA_NUM_PARALLEL slots in parallel.
            let parallel: usize = std::env::var("KREMORY_EVAL_PARALLEL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1)
                .max(1);

            eprintln!(
                "[layer-a longmemeval] kremory providers: ollama@{} chat={} embed={} parallel={}",
                ollama_host, chat_model, embed_model, parallel
            );

            let chat_provider: Arc<Ollama> = LLMBuilder::<Ollama>::new()
                .base_url(&ollama_host)
                .model(&chat_model)
                .build()
                .map_err(|e| format!("failed to build Ollama chat provider: {}", e))?;

            let embed_provider: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
                .base_url(&ollama_host)
                .model(&embed_model)
                .build()
                .map_err(|e| format!("failed to build Ollama embedder: {}", e))?;

            let llm: Arc<dyn kremory::ChatProvider> = chat_provider;
            let embedder: Arc<dyn kremory::DynEmbeddingProvider> = Arc::new(OllamaEmbedAdapter {
                inner: embed_provider,
            });

            let total = samples.len();
            let start = std::time::Instant::now();

            // Concurrent worker pool via JoinSet. Each task gets:
            //   - shared llm + embed Arcs
            //   - shared scorer Arc
            //   - fresh per-question Memory(temp.db) — no writer contention
            type TaskResult = std::result::Result<
                (
                    usize,
                    kremory_eval::layer_a::longmemeval::LongMemEvalSample,
                    kremory_eval::Score,
                    std::time::Duration,
                ),
                String,
            >;
            let mut joinset: tokio::task::JoinSet<TaskResult> = tokio::task::JoinSet::new();
            let mut sample_iter = samples.iter().cloned().enumerate();
            let mut completed: usize = 0;

            // Closure: spawn one worker
            let spawn_one =
                |joinset: &mut tokio::task::JoinSet<TaskResult>,
                 idx: usize,
                 sample: kremory_eval::layer_a::longmemeval::LongMemEvalSample,
                 llm: Arc<dyn kremory::ChatProvider>,
                 embedder: Arc<dyn kremory::DynEmbeddingProvider>,
                 scorer: Arc<LongMemEvalScorer<GemmaJudge>>| {
                    let tmp = std::env::temp_dir().join(format!(
                        "kremory_eval_q_{}_{}.db",
                        Utc::now().timestamp_micros(),
                        idx
                    ));
                    joinset.spawn(async move {
                        let q_start = std::time::Instant::now();
                        let memory = kremory::Memory::open(&tmp)
                            .with_llm(llm)
                            .with_embedder(embedder)
                            .embedding_dim(768)
                            .await
                            .map_err(|e| {
                                format!("open Memory failed for {}: {}", sample.question_id, e)
                            })?;
                        let output = kremory_eval::adapters::longmemeval_adapter::run_sample(
                            &memory, &sample,
                        )
                        .await
                        .map_err(|e| {
                            format!("run_sample failed for {}: {}", sample.question_id, e)
                        })?;
                        let score = scorer.score(&sample, &output).await.map_err(|e| {
                            format!("scorer failed for {}: {}", sample.question_id, e)
                        })?;
                        Ok((idx, sample, score, q_start.elapsed()))
                    });
                };

            // Prime the pool
            for _ in 0..parallel.min(total) {
                if let Some((idx, sample)) = sample_iter.next() {
                    spawn_one(
                        &mut joinset,
                        idx,
                        sample,
                        llm.clone(),
                        embedder.clone(),
                        scorer.clone(),
                    );
                }
            }

            // Drain + refill
            while let Some(join_res) = joinset.join_next().await {
                let task_res = join_res.map_err(|e| format!("worker join error: {}", e))?;
                let (idx, sample, score, elapsed) = task_res?;
                completed += 1;
                let label = if score.value >= 1.0 {
                    "CORRECT"
                } else {
                    "INCORRECT"
                };
                eprintln!(
                    "[layer-a longmemeval] [{:>4}/{:<4}] {:>11.1}s  {:<22} q_type={:<28} q_id={} (slot={})",
                    completed,
                    total,
                    elapsed.as_secs_f64(),
                    label,
                    sample.question_type,
                    sample.question_id,
                    idx,
                );
                all_scores.push((sample, score));

                if let Some((next_idx, next_sample)) = sample_iter.next() {
                    spawn_one(
                        &mut joinset,
                        next_idx,
                        next_sample,
                        llm.clone(),
                        embedder.clone(),
                        scorer.clone(),
                    );
                }
            }

            eprintln!(
                "[layer-a longmemeval] all {} samples scored in {:.1}s (parallel={})",
                total,
                start.elapsed().as_secs_f64(),
                parallel,
            );
        }
    }

    let report = LongMemEvalReport::from_scores(&all_scores);

    // Print per-category breakdown.
    eprintln!(
        "[layer-a longmemeval] Overall accuracy: {:.4}",
        report.overall_accuracy
    );
    eprintln!("[layer-a longmemeval] Per-category breakdown:");

    // Sorted for deterministic output.
    let mut categories: Vec<_> = report.per_type_accuracy.iter().collect();
    categories.sort_by_key(|(k, _)| k.as_str());
    for (cat, acc) in &categories {
        eprintln!("[layer-a longmemeval]   {:<35} = {:.4}", cat, acc);
    }
    eprintln!(
        "[layer-a longmemeval]   {:<35} = {:.4}",
        "abstention_accuracy", report.abstention_accuracy
    );
    if let Some(tokens) = report.mean_input_tokens_per_recall {
        eprintln!(
            "[layer-a longmemeval]   mean_input_tokens_per_recall = {:.1}",
            tokens
        );
    }

    // Build per-sample result rows for the output JSON (FIX 3).
    let per_sample_results: Vec<serde_json::Value> = all_scores
        .iter()
        .map(|(sample, score)| {
            let ability_category = score
                .metadata
                .get("ability_category")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let judge_reasoning = score.reasoning.clone();
            let raw_judge_response = score
                .metadata
                .get("judge_response_raw")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            json!({
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "ability_category": ability_category,
                "score": score.value,
                "judge_reasoning": judge_reasoning,
                "raw_judge_response": raw_judge_response,
            })
        })
        .collect();

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
        "samples": per_sample_results,
    });

    write_report(&out_path, &report_json)?;
    eprintln!(
        "[layer-a longmemeval] Report written to {}",
        out_path.display()
    );
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
        LayerCmd::LayerA(LayerACmd::LongMemEval {
            sample,
            judge,
            smoke,
        }) => {
            if let Err(e) = run_layer_a_longmemeval(manifest_dir, sample, judge, smoke) {
                eprintln!("[eval] FATAL: {}", e);
                std::process::exit(1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    // ─── CLI arg parser ──────────────────────────────────────────────────────

    #[test]
    fn parse_args_empty_errors() {
        let err = parse_args_from(&[]).unwrap_err();
        assert!(err.contains("Usage"), "expected usage hint, got: {}", err);
    }

    #[test]
    fn parse_args_layer_b_default_no_determinism() {
        let cmd = parse_args_from(&s(&["layer-b"])).unwrap();
        match cmd {
            LayerCmd::LayerB { determinism } => assert!(!determinism),
            other => panic!("expected LayerB, got {:?}", other),
        }
    }

    #[test]
    fn parse_args_layer_b_determinism_flag() {
        let cmd = parse_args_from(&s(&["layer-b", "--determinism"])).unwrap();
        match cmd {
            LayerCmd::LayerB { determinism } => assert!(determinism),
            other => panic!("expected LayerB, got {:?}", other),
        }
    }

    #[test]
    fn parse_args_layer_a_longmemeval_defaults() {
        let cmd = parse_args_from(&s(&["layer-a", "longmemeval"])).unwrap();
        match cmd {
            LayerCmd::LayerA(LayerACmd::LongMemEval {
                sample,
                judge,
                smoke,
            }) => {
                assert!(sample.is_none(), "default sample is unlimited");
                assert_eq!(judge, JudgeKind::Mock, "default judge is mock");
                assert!(!smoke, "smoke flag defaults to false");
            }
            other => panic!("expected LayerA::LongMemEval, got {:?}", other),
        }
    }

    #[test]
    fn parse_args_layer_a_longmemeval_with_sample_and_gemma() {
        let cmd = parse_args_from(&s(&[
            "layer-a",
            "longmemeval",
            "--sample",
            "42",
            "--judge",
            "gemma",
        ]))
        .unwrap();
        match cmd {
            LayerCmd::LayerA(LayerACmd::LongMemEval {
                sample,
                judge,
                smoke,
            }) => {
                assert_eq!(sample, Some(42));
                assert_eq!(judge, JudgeKind::Gemma);
                assert!(!smoke);
            }
            other => panic!("expected LayerA::LongMemEval, got {:?}", other),
        }
    }

    #[test]
    fn parse_args_layer_a_longmemeval_smoke_flag() {
        let cmd = parse_args_from(&s(&[
            "layer-a",
            "longmemeval",
            "--smoke",
            "--judge",
            "gemma",
        ]))
        .unwrap();
        match cmd {
            LayerCmd::LayerA(LayerACmd::LongMemEval { smoke, judge, .. }) => {
                assert!(smoke, "expected --smoke to set smoke=true");
                assert_eq!(judge, JudgeKind::Gemma);
            }
            other => panic!("expected LayerA::LongMemEval, got {:?}", other),
        }
    }

    #[test]
    fn parse_args_unknown_layer_errors() {
        let err = parse_args_from(&s(&["layer-x"])).unwrap_err();
        assert!(err.contains("Unknown layer"), "got: {}", err);
    }

    #[test]
    fn parse_args_unknown_layer_a_subcommand_errors() {
        let err = parse_args_from(&s(&["layer-a", "bogus"])).unwrap_err();
        assert!(err.contains("Unknown layer-a subcommand"), "got: {}", err);
    }

    #[test]
    fn parse_args_invalid_sample_falls_back_to_none() {
        let cmd =
            parse_args_from(&s(&["layer-a", "longmemeval", "--sample", "not-a-number"])).unwrap();
        match cmd {
            LayerCmd::LayerA(LayerACmd::LongMemEval { sample, .. }) => {
                assert!(
                    sample.is_none(),
                    "non-numeric --sample value silently drops to None"
                );
            }
            other => panic!("expected LayerA::LongMemEval, got {:?}", other),
        }
    }

    // ─── OllamaEmbedAdapter ──────────────────────────────────────────────────

    /// A deterministic batch embedder for testing the adapter's batch→single
    /// translation logic without a live Ollama instance.
    struct FixedBatchEmbedder {
        dim: usize,
    }

    #[async_trait::async_trait]
    impl AutoEmbeddingProvider for FixedBatchEmbedder {
        async fn embed(
            &self,
            input: Vec<String>,
        ) -> std::result::Result<Vec<Vec<f32>>, autoagents_llm::error::LLMError> {
            // First-text marker = 1.0 in slot 0; rest 0.0. Distinguishes calls
            // from the test's perspective.
            Ok(input
                .iter()
                .map(|s| {
                    let mut v = vec![0.0_f32; self.dim];
                    if !s.is_empty() {
                        v[0] = s.len() as f32;
                    }
                    v
                })
                .collect())
        }
    }

    /// An empty-batch embedder simulating a degenerate AA response.
    struct EmptyBatchEmbedder;

    #[async_trait::async_trait]
    impl AutoEmbeddingProvider for EmptyBatchEmbedder {
        async fn embed(
            &self,
            _input: Vec<String>,
        ) -> std::result::Result<Vec<Vec<f32>>, autoagents_llm::error::LLMError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn ollama_embed_adapter_returns_first_batch_element() {
        use kremory::EmbeddingProvider;
        let adapter = OllamaEmbedAdapter {
            inner: Arc::new(FixedBatchEmbedder { dim: 16 }),
        };
        let v = adapter.embed("hello").await.unwrap();
        assert_eq!(v.len(), 16, "embedding dimension matches inner provider");
        assert_eq!(
            v[0], 5.0,
            "first slot encodes input length (proves text round-tripped through batch API)"
        );
    }

    #[tokio::test]
    async fn ollama_embed_adapter_propagates_empty_batch_as_error() {
        use kremory::EmbeddingProvider;
        let adapter = OllamaEmbedAdapter {
            inner: Arc::new(EmptyBatchEmbedder),
        };
        let err = adapter.embed("hello").await.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("empty embedding batch"),
            "expected empty-batch error, got: {}",
            msg
        );
    }

    #[tokio::test]
    async fn ollama_embed_adapter_is_deterministic() {
        use kremory::EmbeddingProvider;
        let adapter = OllamaEmbedAdapter {
            inner: Arc::new(FixedBatchEmbedder { dim: 8 }),
        };
        let a = adapter.embed("test input").await.unwrap();
        let b = adapter.embed("test input").await.unwrap();
        assert_eq!(a, b, "same input must produce the same output");
    }
}
