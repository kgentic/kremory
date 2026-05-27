//! Calibration spike — re-lock gate for v0.1.4 Phase 1.
//!
//! Loads 20 hand-labeled QA pairs and runs each through the Gemma 4 E2B
//! judge via AA LlamaCppProvider. Aggregates agreement rate vs ground-truth
//! verdicts. Pass criterion: >= 80% agreement.
//!
//! Run 3x: cold-start (measure load time), warm-cache (steady state),
//! warm-cache repeat (variance check, tolerance <= 0.02).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct QaPair {
    id: String,
    #[allow(dead_code)]
    category: String,
    #[allow(dead_code)]
    diversity_tags: Vec<String>,
    question: String,
    context: String,
    answer: String,
    ground_truth_verdict: String, // "correct" | "incorrect" | "partial"
    #[allow(dead_code)]
    rationale: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Fixtures {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    generated_at: String,
    pairs: Vec<QaPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JudgeVerdict {
    is_correct: bool,
    is_partial: bool,
    reasoning: String,
}

#[derive(Debug, Serialize)]
struct PairResult {
    qa_id: String,
    expected: String,
    actual: JudgeVerdict,
    agrees: bool,
    latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct RunResult {
    run_number: u32,
    timestamp: String,
    model_path: String,
    pair_count: usize,
    agree_count: usize,
    agreement_rate: f64,
    model_load_ms: f64,
    total_inference_ms: f64,
    avg_inference_ms: f64,
    pairs: Vec<PairResult>,
}

const JUDGE_PROMPT_TEMPLATE: &str = r#"You are an expert evaluator for agent memory systems. Your task is to judge whether the model's ANSWER correctly addresses the QUESTION based ONLY on the provided CONTEXT.

<question>
{question}
</question>

<context>
{context}
</context>

<answer>
{answer}
</answer>

Evaluate the answer using these criteria:
- is_correct=true: the answer is factually correct and supported by the context
- is_correct=false: the answer is incorrect, contradicts the context, or introduces facts not in the context
- is_partial=true: the answer is correct but less specific than the context allows (e.g., "2024" when context says "Q3 2024")

For abstention cases: if the context does NOT contain information to answer the question, then "I don't know" is a CORRECT answer (is_correct=true).

Respond ONLY with a JSON object matching this schema:
{
  "is_correct": <bool>,
  "is_partial": <bool>,
  "reasoning": "<one short sentence explaining your verdict>"
}"#;

fn map_verdict_to_judge_expectation(gt: &str) -> JudgeVerdict {
    match gt {
        "correct" => JudgeVerdict {
            is_correct: true,
            is_partial: false,
            reasoning: String::new(),
        },
        "incorrect" => JudgeVerdict {
            is_correct: false,
            is_partial: false,
            reasoning: String::new(),
        },
        "partial" => JudgeVerdict {
            is_correct: true,
            is_partial: true,
            reasoning: String::new(),
        },
        _ => JudgeVerdict {
            is_correct: false,
            is_partial: false,
            reasoning: String::new(),
        },
    }
}

fn verdicts_agree(expected: &JudgeVerdict, actual: &JudgeVerdict) -> bool {
    // Agreement on the binary is_correct signal is what matters for calibration.
    // is_partial is a softer signal; we accept either value when expected is partial.
    expected.is_correct == actual.is_correct
}

fn parse_judge_output(raw: &str) -> anyhow::Result<JudgeVerdict> {
    let start = raw
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("no JSON object in output"))?;
    let end = raw
        .rfind('}')
        .ok_or_else(|| anyhow::anyhow!("no closing brace"))?;
    if end <= start {
        anyhow::bail!("malformed JSON range");
    }
    let json_slice = &raw[start..=end];
    let parsed: JudgeVerdict = serde_json::from_str(json_slice)
        .map_err(|e| anyhow::anyhow!("JSON parse error: {} (slice: {})", e, json_slice))?;
    Ok(parsed)
}

async fn run_spike(
    fixtures: &Fixtures,
    model_path: &Path,
    run_number: u32,
) -> anyhow::Result<RunResult> {
    use autoagents_llamacpp::{LlamaCppConfigBuilder, LlamaCppProvider};
    use autoagents_llm::chat::{ChatMessage, ChatProvider, MessageType};
    use autoagents_llm::chat::ChatRole;

    // === Model load ===
    let load_start = Instant::now();

    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;

    let config = LlamaCppConfigBuilder::new()
        .model_path(model_path_str)
        .max_tokens(256)
        .temperature(0.0)
        .seed(42)
        .build();

    let provider = LlamaCppProvider::from_config(config)
        .await
        .map_err(|e| anyhow::anyhow!("failed to load model: {}", e))?;

    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // === Inference loop ===
    let mut pairs = Vec::with_capacity(fixtures.pairs.len());
    let total_start = Instant::now();

    for qa in &fixtures.pairs {
        let prompt = JUDGE_PROMPT_TEMPLATE
            .replace("{question}", &qa.question)
            .replace("{context}", &qa.context)
            .replace("{answer}", &qa.answer);

        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: prompt,
        }];

        let inference_start = Instant::now();
        let response = provider
            .chat_with_tools(&messages, None, None)
            .await
            .map_err(|e| anyhow::anyhow!("inference error for {}: {}", qa.id, e))?;
        let latency_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        let raw_output = response.text().unwrap_or_default();
        let actual = match parse_judge_output(&raw_output) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "Failed to parse judge output for {}: {} (raw: {})",
                    qa.id, e, raw_output
                );
                JudgeVerdict {
                    is_correct: false,
                    is_partial: false,
                    reasoning: format!("PARSE_ERROR: {}", e),
                }
            }
        };

        let expected = map_verdict_to_judge_expectation(&qa.ground_truth_verdict);
        let agrees = verdicts_agree(&expected, &actual);

        pairs.push(PairResult {
            qa_id: qa.id.clone(),
            expected: qa.ground_truth_verdict.clone(),
            actual,
            agrees,
            latency_ms,
        });
    }

    let total_inference_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let agree_count = pairs.iter().filter(|p| p.agrees).count();
    let agreement_rate = agree_count as f64 / pairs.len() as f64;
    let avg_inference_ms = total_inference_ms / pairs.len() as f64;

    Ok(RunResult {
        run_number,
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_path: model_path.to_string_lossy().into_owned(),
        pair_count: fixtures.pairs.len(),
        agree_count,
        agreement_rate,
        model_load_ms,
        total_inference_ms,
        avg_inference_ms,
        pairs,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("KREMORY_EVAL_JUDGE_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".cache/huggingface/hub/gemma-4-E2B-it-Q4_K_M.gguf")
        });

    if !model_path.exists() {
        anyhow::bail!(
            "Model not found at {}. Download first or set KREMORY_EVAL_JUDGE_MODEL_PATH.",
            model_path.display()
        );
    }

    let fixtures_path =
        PathBuf::from("crates/kremory-eval/fixtures/calibration-spike/qa-pairs.json");
    let fixtures_raw = std::fs::read_to_string(&fixtures_path)
        .map_err(|e| anyhow::anyhow!("failed to read fixtures at {}: {}", fixtures_path.display(), e))?;
    let fixtures: Fixtures = serde_json::from_str(&fixtures_raw)
        .map_err(|e| anyhow::anyhow!("failed to parse fixtures JSON: {}", e))?;
    println!(
        "Loaded {} QA pairs from {}",
        fixtures.pairs.len(),
        fixtures_path.display()
    );

    let output_dir = PathBuf::from(
        ".ship/sessions/kremory-v014-phase1-20260528-074901/spike-output",
    );
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| anyhow::anyhow!("failed to create output dir: {}", e))?;

    let mut runs: Vec<RunResult> = Vec::with_capacity(3);
    for run_number in 1_u32..=3 {
        println!("\n=== Run {} ===", run_number);
        let result = run_spike(&fixtures, &model_path, run_number).await?;
        println!(
            "  Agreement: {}/{} ({:.1}%)  model_load: {:.0}ms  avg_inference: {:.0}ms",
            result.agree_count,
            result.pair_count,
            result.agreement_rate * 100.0,
            result.model_load_ms,
            result.avg_inference_ms
        );
        let path = output_dir.join(format!("calibration-run-{}.json", run_number));
        std::fs::write(&path, serde_json::to_string_pretty(&result)?)
            .map_err(|e| anyhow::anyhow!("failed to write {}: {}", path.display(), e))?;
        runs.push(result);
    }

    // === Aggregate ===
    let run_1_rate = runs[0].agreement_rate;
    let run_2_rate = runs[1].agreement_rate;
    let run_3_rate = runs[2].agreement_rate;
    let variance = (run_2_rate - run_3_rate).abs();

    let gate_verdict = if run_2_rate >= 0.80 && variance <= 0.02 {
        "PASS"
    } else if run_2_rate >= 0.60 {
        "CONCERN"
    } else {
        "FAIL"
    };

    let aggregate = serde_json::json!({
        "session": "kremory-v014-phase1-20260528-074901",
        "model": "gemma-4-E2B-it-Q4_K_M",
        "model_source": "unsloth/gemma-4-E2B-it-GGUF",
        "runs": runs,
        "summary": {
            "run_1_agreement_rate": run_1_rate,
            "run_2_agreement_rate": run_2_rate,
            "run_3_agreement_rate": run_3_rate,
            "warm_variance": variance,
            "pass_threshold": 0.80,
            "variance_threshold": 0.02,
            "gate_verdict": gate_verdict
        }
    });

    let baseline_path =
        PathBuf::from("crates/kremory-eval/baselines/v0.1.4-judge-calibration.json");
    std::fs::create_dir_all(
        baseline_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("baseline path has no parent"))?,
    )
    .map_err(|e| anyhow::anyhow!("failed to create baselines dir: {}", e))?;
    std::fs::write(&baseline_path, serde_json::to_string_pretty(&aggregate)?)
        .map_err(|e| anyhow::anyhow!("failed to write baseline: {}", e))?;
    println!(
        "\n=== Aggregate baseline written ===\n{}",
        baseline_path.display()
    );

    Ok(())
}
