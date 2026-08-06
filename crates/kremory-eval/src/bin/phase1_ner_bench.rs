//! Phase 1 NER standalone benchmark (O11y Sprint O0.1).
//!
//! Closes the load-bearing gap in `c6-async-gate-feasibility-2026-06-10.md` GAP-001:
//! the previous spike binary used full `Engine::ingest_with` which includes Phase 2
//! LLM extraction (~80-130s per call per qwen2.5:14b per `intelligence.rs:94`),
//! making it impossible to isolate Phase 1 NER wall-clock — the critical ADR-049
//! <100ms p50 hot-path target.
//!
//! This binary calls `ner_singleton()` directly and measures GLiNER alone across
//! synthetic text sizes 300 / 1k / 5k / 25k / 100k characters with N=10 runs each.
//! Captures p50/p95/p99 wall-clock + entity-extraction count + characters/second.
//!
//! ## Governing references
//! - Sprint plan O0.1
//! - ADR-049 §5.5 `ingest_phase1_ner()` split + <100ms p50 hot-path target
//! - `c6-verify-model-local-ladder-benchmark-2026-06-10.md` GAP-001
//! - `prior-art-adoption-audit-2026-06-10.md` Top 5 finding #1 (hot-path measurement)
//!
//! ## Usage
//!
//! ```text
//! cargo run -p kremory-eval --release --features ner --bin phase1_ner_bench
//! ```

use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use kremory::core::config::ContentType;
use kremory::core::intelligence::{EntityExtractor, ExtractionContext};
use kremory::core::ner::ner_singleton;
use serde::Serialize;

// ─── Synthetic text generation ───────────────────────────────────────────────

const BASE_PARAGRAPH: &str = "Apple announced quarterly earnings on Tuesday with CEO Tim Cook \
presenting the results in Cupertino. The company reported strong iPhone sales \
in Asia, particularly in markets like Japan and South Korea. Microsoft and \
Google are facing similar regulatory pressure from the European Commission. \
Dr. Sarah Chen at Stanford University published research about climate change \
affecting the Pacific Northwest forests. The Federal Reserve raised interest \
rates by 25 basis points last month, with Chair Jerome Powell citing inflation \
concerns. Boeing delivered the new 787 aircraft to United Airlines in Seattle. ";

/// Generate a synthetic text of approximately `target_chars` characters by
/// repeating BASE_PARAGRAPH. Used to characterize per-character throughput.
fn synthesize_text(target_chars: usize) -> String {
    let mut out = String::with_capacity(target_chars + BASE_PARAGRAPH.len());
    while out.len() < target_chars {
        out.push_str(BASE_PARAGRAPH);
    }
    // Truncate to roughly target_chars (may overshoot by base length; acceptable).
    out
}

// ─── Statistics ──────────────────────────────────────────────────────────────

fn percentile(mut samples: Vec<u128>, p: f64) -> u128 {
    samples.sort_unstable();
    if samples.is_empty() {
        return 0;
    }
    let idx = ((samples.len() as f64 - 1.0) * p / 100.0).round() as usize;
    samples[idx]
}

#[derive(Debug, Clone, Serialize)]
struct SizeResult {
    target_chars: usize,
    actual_chars: usize,
    runs: usize,
    p50_us: u128,
    p95_us: u128,
    p99_us: u128,
    mean_us: u128,
    min_us: u128,
    max_us: u128,
    entities_per_run: usize,
    chars_per_second: f64,
    /// Target check — flag if p50 exceeds the ADR-049 hot-path threshold.
    p50_under_100ms: bool,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    binary: &'static str,
    timestamp: String,
    runs_per_size: usize,
    sizes: Vec<SizeResult>,
    allowed_entity_types: Vec<String>,
    warmup_us: u128,
    notes: Vec<&'static str>,
}

// ─── Benchmark loop ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let wall_start = Instant::now();

    let runs_per_size: usize = std::env::var("PHASE1_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let sizes: Vec<usize> = std::env::var("PHASE1_BENCH_SIZES")
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|_| vec![300, 1_000, 5_000, 25_000, 100_000]);

    let allowed: Vec<String> = vec![
        "Person".to_string(),
        "Organization".to_string(),
        "Location".to_string(),
        "Date".to_string(),
        "Money".to_string(),
        "Event".to_string(),
    ];

    eprintln!("[phase1-bench] runs_per_size={runs_per_size} sizes={sizes:?}");
    eprintln!("[phase1-bench] allowed_entity_types={allowed:?}");

    // ── Load GLiNER (singleton) ──────────────────────────────────────────────
    eprintln!("[phase1-bench] Loading ner_singleton()...");
    let load_start = Instant::now();
    let extractor = ner_singleton().context("ner_singleton load")?;
    let load_elapsed = load_start.elapsed();
    eprintln!(
        "[phase1-bench] Singleton loaded in {:.2}s (first call, includes model load)",
        load_elapsed.as_secs_f64()
    );

    // ── Warmup ───────────────────────────────────────────────────────────────
    eprintln!("[phase1-bench] Warming up...");
    let warm_text = synthesize_text(300);
    let warm_ctx = ExtractionContext {
        allowed_entity_types: &allowed,
        allowed_edge_types: &[],
        known_entities: &[],
        excluded_entity_types: &[],
        content_type: ContentType::Text,
        registry_specs: &[],
        existing_graph_entities: &[],
        arm_budget_ms: 30_000,
        model: None,
        reference_time: None,
    };
    let warm_start = Instant::now();
    let _ = extractor
        .extract(&warm_text, &warm_ctx)
        .await
        .context("warmup extract")?;
    let warmup_us = warm_start.elapsed().as_micros();
    eprintln!("[phase1-bench] Warmup done in {warmup_us}µs");

    // ── Per-size benchmark loop ──────────────────────────────────────────────
    let mut size_results: Vec<SizeResult> = Vec::new();

    for target in &sizes {
        let text = synthesize_text(*target);
        let actual_chars = text.len();
        eprintln!(
            "\n[phase1-bench] Size target={target}, actual_chars={actual_chars}, runs={runs_per_size}"
        );

        let ctx = ExtractionContext {
            allowed_entity_types: &allowed,
            allowed_edge_types: &[],
            known_entities: &[],
            excluded_entity_types: &[],
            content_type: ContentType::Text,
            registry_specs: &[],
            existing_graph_entities: &[],
            arm_budget_ms: 30_000,
            model: None,
            reference_time: None,
        };

        let mut samples: Vec<u128> = Vec::with_capacity(runs_per_size);
        let mut total_entities = 0usize;

        for run in 0..runs_per_size {
            let start = Instant::now();
            let result = extractor
                .extract(&text, &ctx)
                .await
                .with_context(|| format!("extract run {run} (size {target})"))?;
            let elapsed_us = start.elapsed().as_micros();
            samples.push(elapsed_us);
            total_entities += result.entities.len();

            if run == 0 {
                eprintln!(
                    "  run 1: {}µs ({} entities found)",
                    elapsed_us,
                    result.entities.len()
                );
            }
        }

        let mean_us = samples.iter().sum::<u128>() / samples.len() as u128;
        let p50 = percentile(samples.clone(), 50.0);
        let p95 = percentile(samples.clone(), 95.0);
        let p99 = percentile(samples.clone(), 99.0);
        let min_us = samples.iter().copied().min().unwrap_or(0);
        let max_us = samples.iter().copied().max().unwrap_or(0);
        let entities_per_run = total_entities / runs_per_size;
        // Quinn LOW-03 fix: guard zero-mean to avoid f64::INFINITY which
        // serde_json serialises as `null` — a confusing sentinel for a
        // throughput metric. Zero is the explicit "no data" value.
        let chars_per_second = if mean_us == 0 {
            0.0
        } else {
            (actual_chars as f64) / (mean_us as f64 / 1_000_000.0)
        };
        let p50_under_100ms = p50 < 100_000; // 100ms = 100_000µs

        eprintln!(
            "  results: p50={}µs p95={}µs p99={}µs mean={}µs ({:.1} chars/sec, ~{} entities/run, <100ms target: {})",
            p50,
            p95,
            p99,
            mean_us,
            chars_per_second,
            entities_per_run,
            if p50_under_100ms { "PASS" } else { "FAIL" }
        );

        size_results.push(SizeResult {
            target_chars: *target,
            actual_chars,
            runs: runs_per_size,
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
            mean_us,
            min_us,
            max_us,
            entities_per_run,
            chars_per_second,
            p50_under_100ms,
        });
    }

    // ── Write results to disk ────────────────────────────────────────────────
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("can't determine workspace root"))?
        .to_path_buf();

    let date = Utc::now().format("%Y-%m-%d").to_string();
    let outdir = workspace_root
        .join(".ai-docs")
        .join("spikes")
        .join(format!("phase1-ner-bench-{date}"));
    std::fs::create_dir_all(&outdir).context("create outdir")?;

    let report = BenchReport {
        binary: "phase1_ner_bench",
        timestamp: Utc::now().to_rfc3339(),
        runs_per_size,
        sizes: size_results.clone(),
        allowed_entity_types: allowed.clone(),
        warmup_us,
        notes: vec![
            "GLiNER is closed-vocab; entity counts reflect labels in allowed_entity_types only",
            "Synthetic text = BASE_PARAGRAPH repeated to target size; real episodes may vary",
            "p50 <100ms is the ADR-049 §5.5 hot-path target",
            "Warmup run excluded from sample distribution",
        ],
    };

    let json_path = outdir.join("results.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&report)?)
        .context("write results.json")?;
    eprintln!("\n[phase1-bench] Wrote {}", json_path.display());

    // Markdown summary
    let md_path = outdir.join("results.md");
    let mut md = String::new();
    md.push_str(&format!(
        "# Phase 1 NER standalone benchmark — {date}\n\n\
         Closes ADR-049 GAP-001 (Phase 1 hot-path measurement).\n\n\
         **Binary**: `crates/kremory-eval/src/bin/phase1_ner_bench.rs`\n\
         **Runs per size**: {runs_per_size}\n\
         **Warmup wall-clock**: {warmup_us}µs ({:.2}ms)\n\
         **Singleton load time**: {:.2}s (first call only)\n\n",
        warmup_us as f64 / 1000.0,
        load_elapsed.as_secs_f64()
    ));
    md.push_str("## Results — per-size statistics\n\n");
    md.push_str("| Target chars | Actual chars | p50 (µs) | p95 (µs) | p99 (µs) | Mean (µs) | Chars/sec | Entities/run | <100ms? |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for r in &size_results {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.1} | {} | {} |\n",
            r.target_chars,
            r.actual_chars,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.mean_us,
            r.chars_per_second,
            r.entities_per_run,
            if r.p50_under_100ms {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        ));
    }
    md.push_str("\n## Interpretation\n\n");
    md.push_str(
        "ADR-049 §5.5 claims Phase 1 NER < 100ms p50 hot-path.\n\n\
         **Result by size**:\n",
    );
    for r in &size_results {
        md.push_str(&format!(
            "- {} chars: p50={:.2}ms ({} target)\n",
            r.target_chars,
            r.p50_us as f64 / 1000.0,
            if r.p50_under_100ms {
                "WITHIN"
            } else {
                "EXCEEDS"
            }
        ));
    }
    md.push_str("\n## Notes\n\n");
    for n in &report.notes {
        md.push_str(&format!("- {n}\n"));
    }
    std::fs::write(&md_path, md).context("write results.md")?;
    eprintln!("[phase1-bench] Wrote {}", md_path.display());

    eprintln!(
        "\n[phase1-bench] Total wall-clock: {:.1}s",
        wall_start.elapsed().as_secs_f64()
    );
    Ok(())
}
