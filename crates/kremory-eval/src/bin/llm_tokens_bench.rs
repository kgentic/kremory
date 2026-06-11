//! LLM tokens-per-second benchmark (O11y Sprint O0.3).
//!
//! Captures `eval_count` + `prompt_eval_count` + duration data from Ollama
//! `/api/generate` for each model, then derives tokens/sec for both prompt
//! processing and generation. The Ollama API exposes these counts in every
//! response; the autoagents-llm crate's `ChatProvider` trait does NOT surface
//! them, so this binary makes direct HTTP calls.
//!
//! Combined with the per-call latency histograms captured by `O0.2` (via
//! `metrics-util::Snapshotter` in `consistency_check_sweep`), downstream
//! analysis can derive **estimated tokens per LLM call** = latency_ms ×
//! (tokens/sec from this benchmark).
//!
//! ## Governing references
//! - O11y Sprint plan O0.3
//! - Manual Ollama API exploration (2026-06-11 session) — confirmed eval_count
//!   in response body but absent from autoagents-llm `ChatProvider` trait surface
//! - CLAUDE.md Rule 19 (observability first-class)
//!
//! ## Usage
//!
//! ```text
//! OLLAMA_HOST=http://localhost:11434 \
//! LLM_TOKENS_BENCH_MODELS=qwen2.5:14b,gemma4-e2b:latest,gemma4:e4b,qwen3.6:35b-mlx \
//! LLM_TOKENS_BENCH_RUNS=3 \
//! cargo run -p kremory-eval --release --bin llm_tokens_bench
//! ```

use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

const DEFAULT_MODELS: &[&str] = &[
    "qwen2.5:14b",
    "gemma4-e2b:latest",
    "gemma4:e4b",
    "qwen3.6:35b-mlx",
];

/// Two prompt sizes characterise typical kremory LLM call shapes:
/// - SHORT: ~50-token prompt — characteristic of verify_batch (10-candidate batch
///   with thin metadata)
/// - LONG: ~500-token prompt — characteristic of extraction passes (full
///   source_episode + registry + few-shot examples)
const SHORT_PROMPT: &str = "Summarize in 2 sentences: \
Apple announced earnings. Microsoft launched a new product. \
Google reorganised its cloud division. The Federal Reserve raised rates.";

const LONG_PROMPT: &str = "You are an entity-type verification model. Below is a \
candidate entity from a knowledge graph along with the source episode text. \
Decide whether the proposed type is correct.\n\n\
ENTITY: Apple\n\
PROPOSED_TYPE: Organization (id=2)\n\
SOURCE_EPISODE: Apple emailed me yesterday about my developer account suspension, \
which was frustrating because I rely on it for my indie app business. The notification \
arrived in my inbox at around 6am and referenced policy violations I had not been \
informed about previously. I was also working on three other projects at the time, \
including a Microsoft Azure migration for one client, a Google Cloud setup for another, \
and a private side-project using Anthropic's Claude API for evaluating GitHub PRs.\n\n\
ENTITY_TYPE_REGISTRY:\n\
  0 - Entity - Catch-all for unclassified entities\n\
  1 - Person - A named human individual\n\
  2 - Organization - A company, institution, or formal group\n\
  3 - Location - A geographic place or region\n\
  4 - Date - A specific calendar date or time period\n\
  5 - Event - A scheduled occurrence or happening\n\
  6 - Product - A manufactured item or branded good\n\
  7 - Concept - An abstract idea, theory, or notion\n\
  8 - LegalDocument - A contract, ruling, or formal legal text\n\
  9 - Technology - A programming language, framework, or technical tool\n\
 10 - Court - A judicial body or court of law\n\n\
EXAMPLES:\n\
- 'Microsoft' (proposed: Organization) → confirm — Microsoft is canonically a company\n\
- 'Java' (proposed: Location) in a coffee-shop episode → correct (proposed wrong: Java is the coffee variety)\n\
- 'Sherman Act' (proposed: LegalDocument) in a legal episode → confirm\n\n\
Respond in JSON: {\"action\": \"confirm\"|\"correct\"|\"uncertain\", \"new_type_id\": int|null, \"confidence\": float}";

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    eval_count: u64,
    eval_duration: u64, // nanoseconds
    prompt_eval_count: u64,
    prompt_eval_duration: u64,
    total_duration: u64,
    #[allow(dead_code)]
    load_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelMeasurement {
    model: String,
    prompt_kind: String,
    prompt_chars: usize,
    runs: usize,
    eval_tokens_per_sec_mean: f64,
    eval_tokens_per_sec_p50: f64,
    prompt_tokens_per_sec_mean: f64,
    eval_count_mean: f64,
    prompt_eval_count_mean: f64,
    total_duration_ms_mean: f64,
    total_duration_ms_p95: f64,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    binary: &'static str,
    timestamp: String,
    ollama_host: String,
    measurements: Vec<ModelMeasurement>,
}

fn p50(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn p95(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        0.0
    } else {
        let idx = ((v.len() as f64 - 1.0) * 0.95).round() as usize;
        v[idx]
    }
}

async fn measure_one(
    client: &reqwest::Client,
    ollama_host: &str,
    model: &str,
    prompt: &str,
) -> Result<OllamaGenerateResponse> {
    // Quinn MED-01 fix: `keep_alive` is a TOP-LEVEL Ollama API field, not an
    // option. Nesting it inside `options` causes Ollama to silently ignore it
    // (per autoagents-llm OllamaGenerateRequest struct — keep_alive not in
    // options), defaulting to 5-minute keep_alive. That risks reload spikes
    // when sweeping across many models and contradicts
    // `feedback_td024_keepalive_thrash_hurts_precision`.
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "keep_alive": "1h",
        "options": {
            "temperature": 0.0,
            "seed": 42
        }
    });
    let resp = client
        .post(format!("{ollama_host}/api/generate"))
        .json(&body)
        .send()
        .await
        .context("ollama POST")?;
    let parsed: OllamaGenerateResponse = resp.json().await.context("ollama JSON parse")?;
    Ok(parsed)
}

#[tokio::main]
async fn main() -> Result<()> {
    let wall_start = Instant::now();

    let ollama_host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let runs: usize = std::env::var("LLM_TOKENS_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let models: Vec<String> = std::env::var("LLM_TOKENS_BENCH_MODELS")
        .map(|s| s.split(',').map(|m| m.trim().to_string()).collect())
        .unwrap_or_else(|_| DEFAULT_MODELS.iter().map(|s| s.to_string()).collect());

    eprintln!("[llm-tokens-bench] ollama={ollama_host}");
    eprintln!("[llm-tokens-bench] runs_per_model={runs}");
    eprintln!("[llm-tokens-bench] models={models:?}");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("reqwest client")?;

    let mut measurements: Vec<ModelMeasurement> = Vec::new();

    for model in &models {
        for (kind, prompt) in [("short", SHORT_PROMPT), ("long", LONG_PROMPT)] {
            eprintln!("\n[llm-tokens-bench] model={model} kind={kind} (warmup)");
            // Warmup once (especially important for mlx models with cold-load tax).
            if let Err(e) = measure_one(&client, &ollama_host, model, prompt).await {
                eprintln!("  warmup failed: {e:?} — skipping {model}/{kind}");
                continue;
            }

            let mut eval_tps: Vec<f64> = Vec::with_capacity(runs);
            let mut prompt_tps: Vec<f64> = Vec::with_capacity(runs);
            let mut eval_counts: Vec<f64> = Vec::with_capacity(runs);
            let mut prompt_counts: Vec<f64> = Vec::with_capacity(runs);
            let mut totals_ms: Vec<f64> = Vec::with_capacity(runs);

            for i in 0..runs {
                match measure_one(&client, &ollama_host, model, prompt).await {
                    Ok(r) => {
                        let eval_secs = r.eval_duration as f64 / 1e9;
                        let prompt_secs = r.prompt_eval_duration as f64 / 1e9;
                        let etps = if eval_secs > 0.0 {
                            r.eval_count as f64 / eval_secs
                        } else {
                            0.0
                        };
                        let ptps = if prompt_secs > 0.0 {
                            r.prompt_eval_count as f64 / prompt_secs
                        } else {
                            0.0
                        };
                        eval_tps.push(etps);
                        prompt_tps.push(ptps);
                        eval_counts.push(r.eval_count as f64);
                        prompt_counts.push(r.prompt_eval_count as f64);
                        totals_ms.push(r.total_duration as f64 / 1e6);
                        eprintln!(
                            "  run {}: prompt_eval={} t/s ({} tokens), gen={:.1} t/s ({} tokens), total={:.0}ms",
                            i + 1,
                            ptps as u64,
                            r.prompt_eval_count,
                            etps,
                            r.eval_count,
                            r.total_duration as f64 / 1e6
                        );
                    }
                    Err(e) => {
                        eprintln!("  run {} FAILED: {e:?}", i + 1);
                    }
                }
            }

            if eval_tps.is_empty() {
                eprintln!("  no successful runs — skipping {model}/{kind}");
                continue;
            }

            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

            measurements.push(ModelMeasurement {
                model: model.clone(),
                prompt_kind: kind.to_string(),
                prompt_chars: prompt.len(),
                runs: eval_tps.len(),
                eval_tokens_per_sec_mean: mean(&eval_tps),
                eval_tokens_per_sec_p50: p50(eval_tps.clone()),
                prompt_tokens_per_sec_mean: mean(&prompt_tps),
                eval_count_mean: mean(&eval_counts),
                prompt_eval_count_mean: mean(&prompt_counts),
                total_duration_ms_mean: mean(&totals_ms),
                total_duration_ms_p95: p95(totals_ms.clone()),
                notes: vec![],
            });
        }
    }

    // ── Write report ─────────────────────────────────────────────────────────
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("workspace root"))?
        .to_path_buf();

    let date = Utc::now().format("%Y-%m-%d").to_string();
    let outdir = workspace_root
        .join(".ai-docs")
        .join("spikes")
        .join(format!("llm-tokens-bench-{date}"));
    std::fs::create_dir_all(&outdir).context("create outdir")?;

    let report = BenchReport {
        binary: "llm_tokens_bench",
        timestamp: Utc::now().to_rfc3339(),
        ollama_host: ollama_host.clone(),
        measurements: measurements.clone(),
    };

    let json_path = outdir.join("results.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&report)?)
        .context("write results.json")?;
    eprintln!("\n[llm-tokens-bench] Wrote {}", json_path.display());

    let mut md = String::new();
    md.push_str(&format!(
        "# LLM tokens/sec benchmark — {date}\n\n\
         **Binary**: `crates/kremory-eval/src/bin/llm_tokens_bench.rs`\n\
         **Ollama host**: {ollama_host}\n\
         **Runs per (model, prompt-kind) combo**: {runs}\n\n\
         ## Results matrix\n\n\
         | Model | Prompt kind | Prompt chars | Prompt tokens (mean) | Eval tokens (mean) | \
         Prompt eval (t/s mean) | Gen eval (t/s mean) | Gen eval p50 | Total ms (mean) | Total ms p95 |\n\
         |---|---|---|---|---|---|---|---|---|---|\n"
    ));
    for m in &measurements {
        md.push_str(&format!(
            "| {} | {} | {} | {:.0} | {:.0} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} |\n",
            m.model,
            m.prompt_kind,
            m.prompt_chars,
            m.prompt_eval_count_mean,
            m.eval_count_mean,
            m.prompt_tokens_per_sec_mean,
            m.eval_tokens_per_sec_mean,
            m.eval_tokens_per_sec_p50,
            m.total_duration_ms_mean,
            m.total_duration_ms_p95
        ));
    }
    md.push_str("\n## Derived calibrations\n\n");
    md.push_str("For each model + prompt-kind, the wall-clock latency observed by ");
    md.push_str("`consistency_check_sweep` per LLM call can be converted to estimated tokens:\n\n");
    md.push_str("- **Estimated_eval_tokens = latency_ms × eval_tps_mean / 1000**\n");
    md.push_str("- **Estimated_prompt_tokens = latency_ms × prompt_tps_mean / 1000**\n\n");
    md.push_str("Combine with the metrics dump from `consistency_check_sweep` (O0.2) for end-to-end accounting.\n");
    let md_path = outdir.join("results.md");
    std::fs::write(&md_path, md).context("write results.md")?;
    eprintln!("[llm-tokens-bench] Wrote {}", md_path.display());

    eprintln!(
        "\n[llm-tokens-bench] Total wall-clock: {:.1}s",
        wall_start.elapsed().as_secs_f64()
    );

    Ok(())
}
