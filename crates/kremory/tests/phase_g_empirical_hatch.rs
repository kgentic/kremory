//! Phase G — TD-017 Empirical Hatch Benchmark
//!
//! Real-LLM benchmark measuring PRE-dream vs POST-dream label precision on
//! `legal_deposition` + `mock_interview` fixtures with gemma4-e2b:latest.
//!
//! ## DoD coverage
//!
//! - **G1** PRE-dream + POST-dream precision measured on both fixtures.
//! - **G2** gemma4-e2b PRE-dream precision ≥ Phase A baseline (100% mock / 81.2% legal).
//! - **G3** Cloud-LLM gate: checked at runtime via `OPENAI_API_KEY` /
//!   `ANTHROPIC_API_KEY` env vars. Skipped gracefully if absent (deferred per
//!   §G5 / plan risk R11).
//! - **G4** Per-provider INDEPENDENT pass per ADR-044 §5 — no aggregate-pass.
//!
//! ## How to run
//!
//! ```sh
//! export OLLAMA_BASE_URL=http://localhost:11434
//! export OLLAMA_CHAT_MODEL=gemma4-e2b:latest
//! export OLLAMA_KEEP_ALIVE=1h
//! export KREMORY_BENCH_USE_HYBRID=1
//! export KREMORY_GLINER_THRESHOLD=0.2
//! cargo test -p kremory --features llm-integration,ner \
//!   --test phase_g_empirical_hatch -- --ignored --nocapture
//! ```
//!
//! ## ADR references
//!
//! - ADR-044 §5 (empirical hatch gate, per-provider independent pass)
//! - ADR-046 Option E (2-arm reclassify, drift deferred)
//! - TD-017 (empirical hatch benchmark)
//! - `.ai-docs/plans/v0-1-1-dream-impl-sprint-plan-2026-06-09.md` Phase G DoD

#![cfg(feature = "llm-integration")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::embedding::EmbeddingBuilder;
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::extraction::{is_canonical_entity_type, DefaultExtractor};
use kremory::core::ingest::{DreamPassOpts, Engine, SourceParams};
use kremory::core::schema::TemporalGraph;
use serde::Deserialize;

mod helpers;
use helpers::ollama_adapter::OllamaEmbedderAdapter;

// ── Ground truth types ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GroundTruthEntity {
    name: String,
    label: String,
}

#[derive(Debug, Deserialize)]
struct DomainGroundTruth {
    entities: Vec<GroundTruthEntity>,
    #[allow(dead_code)]
    #[serde(default)]
    min_relationships: usize,
}

fn load_ground_truth() -> HashMap<String, DomainGroundTruth> {
    let gt_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kremory-eval/fixtures/ground_truth.json"
    );
    let raw = std::fs::read_to_string(gt_path)
        .unwrap_or_else(|e| panic!("failed to read ground_truth.json: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("failed to parse ground_truth.json: {e}"))
}

// ── Label precision metric ─────────────────────────────────────────────────────

/// Compute label precision: for each expected entity (name+label), check if
/// any extracted entity matches BOTH name (fuzzy) AND label (exact).
///
/// Returns `(precision, matched, total_expected)`.
fn label_precision(
    extracted: &[(String, String)], // (name, label)
    expected: &[(String, String)],  // (name, label) from ground truth
) -> (f64, usize, usize) {
    let matched = expected
        .iter()
        .filter(|(exp_name, exp_label)| {
            let exp_lower = exp_name.to_lowercase();
            let exp_label_lower = exp_label.to_lowercase();
            extracted.iter().any(|(ext_name, ext_label)| {
                let ext_lower = ext_name.to_lowercase();
                let ext_label_lower = ext_label.to_lowercase();
                let name_match = ext_lower.contains(exp_lower.as_str())
                    || exp_lower.contains(ext_lower.as_str());
                let label_match = ext_label_lower == exp_label_lower;
                name_match && label_match
            })
        })
        .count();
    let total = expected.len();
    let precision = if total == 0 {
        1.0
    } else {
        matched as f64 / total as f64
    };
    (precision, matched, total)
}

// ── Env helpers ───────────────────────────────────────────────────────────────

fn ollama_base_url() -> Option<String> {
    std::env::var("OLLAMA_HOST")
        .or_else(|_| std::env::var("OLLAMA_BASE_URL"))
        .ok()
}

fn ollama_chat_model() -> String {
    // SoT: llm_integration.rs:1-25 — gemma4-e2b:latest is the interactive default.
    // MUST be set explicitly; do not fall back to llama3.2:3b default.
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4-e2b:latest".to_string())
}

// ── PRE/POST dream measurement helper ─────────────────────────────────────────

/// Arguments bundle for [`measure_pre_post`] — avoids clippy too_many_arguments.
struct MeasureArgs<'a, L>
where
    L: kremory::memory::ChatProvider + 'static,
{
    engine: &'a Engine<L, OllamaEmbedderAdapter>,
    llm: Arc<L>,
    graph: &'a Arc<TemporalGraph>,
    fixture_key: &'a str,
    provider_label: &'a str,
    pre_threshold: f64,
    use_hybrid: bool,
}

#[allow(dead_code)]
struct FixtureRunResult {
    fixture: String,
    provider: String,
    pre_precision: f64,
    pre_matched: usize,
    pre_total: usize,
    post_precision: f64,
    post_matched: usize,
    post_total: usize,
    pre_wall_ms: u64,
    post_wall_ms: u64,
    dream_duration_ms: u64,
    dream_entities_reclassified: usize,
    pass_pre: bool,
    pass_post: bool,
}

/// Phase G PRE+POST measurement for a single fixture.
///
/// Ingests the fixture, measures PRE-dream precision, fires
/// `Engine::run_dream_pass_sync`, then measures POST-dream precision against
/// the same graph.
///
/// `args.llm` is passed explicitly (not accessed from `engine.llm`, which is
/// `pub(crate)` and inaccessible from integration-test context).
async fn measure_pre_post<L>(args: MeasureArgs<'_, L>) -> FixtureRunResult
where
    L: kremory::memory::ChatProvider + 'static,
{
    use kremory::core::extraction::GlinerLlmExtractor;

    let MeasureArgs {
        engine,
        llm,
        graph,
        fixture_key,
        provider_label,
        pre_threshold,
        use_hybrid,
    } = args;

    let fixture_filename = format!("{fixture_key}.txt");
    let fixture_path = format!(
        "{}/../kremory-eval/fixtures/{}",
        env!("CARGO_MANIFEST_DIR"),
        fixture_filename
    );
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {fixture_path}: {e}"));

    let ground_truth = load_ground_truth();
    let domain = ground_truth
        .get(fixture_key)
        .unwrap_or_else(|| panic!("{fixture_key} must be in ground_truth.json"));
    let expected: Vec<(String, String)> = domain
        .entities
        .iter()
        .map(|e| (e.name.clone(), e.label.clone()))
        .collect();

    let source_params = SourceParams::default();

    // ── PRE-dream ingest ──────────────────────────────────────────────────────
    let pre_start = std::time::Instant::now();

    let ingest_result = if use_hybrid {
        #[cfg(feature = "ner")]
        {
            let hybrid = GlinerLlmExtractor::new(Arc::clone(&llm))
                .expect("GlinerLlmExtractor::new — needs GLiNER model + LLM");
            engine
                .ingest_with(&hybrid, &fixture_text, None, None, None, source_params)
                .await
                .unwrap_or_else(|e| panic!("PRE-dream ingest of {fixture_key} failed: {e}"))
        }
        #[cfg(not(feature = "ner"))]
        panic!("KREMORY_BENCH_USE_HYBRID=1 requires --features ner")
    } else {
        let extractor = DefaultExtractor::new(Arc::clone(&llm));
        engine
            .ingest_with(&extractor, &fixture_text, None, None, None, source_params)
            .await
            .unwrap_or_else(|e| panic!("PRE-dream ingest of {fixture_key} failed: {e}"))
    };
    let pre_wall_ms = pre_start.elapsed().as_millis() as u64;

    // Collect PRE-dream extracted entities.
    let mut extracted_pre: Vec<(String, String)> = Vec::new();
    for entity_id in &ingest_result.upserted_entities {
        if let Some(entity) = graph.get_entity(entity_id).await.expect("get_entity PRE") {
            let display_name = entity
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(&entity.id)
                .to_string();
            extracted_pre.push((display_name, entity.label.clone()));
        }
    }

    let (pre_precision, pre_matched, pre_total) = label_precision(&extracted_pre, &expected);
    let pre_placeholder = extracted_pre
        .iter()
        .filter(|(_, l)| !is_canonical_entity_type(l))
        .count();

    eprintln!(
        "[G] {fixture_key} PRE-dream: {:.1}% ({}/{}) | {pre_placeholder} non-canonical | {pre_wall_ms}ms",
        pre_precision * 100.0,
        pre_matched,
        pre_total,
    );
    for (n, l) in &extracted_pre {
        eprintln!("    extracted: '{n}' label='{l}'");
    }

    // ── Dream pass ────────────────────────────────────────────────────────────
    let dream_opts = DreamPassOpts {
        include_type_discovery: false, // Pass 0 deferred per Phase D stub
        confidence_threshold: 0.5,
        max_episodes_per_run: None,
        reclassify_high_conf_threshold: 0.7,
    };

    let dream_start = std::time::Instant::now();
    let dream_summary = engine
        .run_dream_pass_sync(dream_opts)
        .await
        .unwrap_or_else(|e| panic!("run_dream_pass_sync for {fixture_key} failed: {e}"));
    let dream_duration_ms = dream_start.elapsed().as_millis() as u64;

    eprintln!(
        "[G] {fixture_key} dream pass: entities_reclassified={} ghost_retried={} duration={}ms",
        dream_summary.entities_reclassified,
        dream_summary.ghost_episodes_retried,
        dream_summary.duration_ms,
    );

    // ── POST-dream precision ──────────────────────────────────────────────────
    // Re-read ALL entities from the graph (including any reclassified by dream).
    let post_start = std::time::Instant::now();
    let all_entities = graph.list_entities().await.expect("list_entities POST");
    let post_wall_ms = post_start.elapsed().as_millis() as u64;

    let extracted_post: Vec<(String, String)> = all_entities
        .iter()
        .map(|e| {
            let name = e
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| e.id.clone());
            (name, e.label.clone())
        })
        .collect();

    let (post_precision, post_matched, post_total) = label_precision(&extracted_post, &expected);
    let post_placeholder = extracted_post
        .iter()
        .filter(|(_, l)| !is_canonical_entity_type(l))
        .count();

    eprintln!(
        "[G] {fixture_key} POST-dream: {:.1}% ({}/{}) | {post_placeholder} non-canonical | query {post_wall_ms}ms",
        post_precision * 100.0,
        post_matched,
        post_total,
    );

    let pass_pre = pre_precision >= pre_threshold;
    // POST-dream: assert no regression (post >= pre) rather than an absolute gate.
    let pass_post = post_precision >= pre_precision;

    eprintln!(
        "[G] {fixture_key} {provider_label}: PRE={:.1}% POST={:.1}% Δ={:+.1}pt | pre-gate ({:.0}%): {} | post-no-regress: {}",
        pre_precision * 100.0,
        post_precision * 100.0,
        (post_precision - pre_precision) * 100.0,
        pre_threshold * 100.0,
        if pass_pre { "PASS" } else { "FAIL" },
        if pass_post { "PASS" } else { "FAIL" },
    );

    FixtureRunResult {
        fixture: fixture_key.to_string(),
        provider: provider_label.to_string(),
        pre_precision,
        pre_matched,
        pre_total,
        post_precision,
        post_matched,
        post_total,
        pre_wall_ms,
        post_wall_ms,
        dream_duration_ms,
        dream_entities_reclassified: dream_summary.entities_reclassified,
        pass_pre,
        pass_post,
    }
}

// ── G1/G2/G4: Main empirical hatch — gemma4-e2b local ────────────────────────

/// Phase G empirical hatch: PRE+POST dream precision on both fixtures,
/// gemma4-e2b:latest, hybrid extractor (KREMORY_BENCH_USE_HYBRID=1 / --features ner).
///
/// ## DoD criteria verified
///
/// - G1: PRE+POST measured on legal_deposition + mock_interview.
/// - G2: gemma4-e2b PRE precision ≥ Phase A baseline (mock=100%, legal=81.2%) at −2pt threshold.
/// - G4: INDEPENDENT per-provider pass (this test covers local provider only).
///
/// ## Thresholds
///
/// Phase A re-run (Amendment #2 2026-06-09, gemma4-e2b:latest, hybrid+ner, threshold=0.2):
/// - mock_interview: 100.0% → G2 gate = 98.0% (−2pt)
/// - legal_deposition: 81.2% → G2 gate = 79.2% (−2pt)
///
/// ## Cloud provider (G3)
///
/// Handled in separate test `phase_g_cloud_llm_gate` (below). If
/// `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is absent, that test skips with
/// an informational note; G3 is filed as deferred (TD-017 follow-up) per
/// plan §G5 and risk R11.
///
/// ## How to run
///
/// ```sh
/// OLLAMA_BASE_URL=http://localhost:11434 \
/// OLLAMA_CHAT_MODEL=gemma4-e2b:latest \
/// OLLAMA_KEEP_ALIVE=1h \
/// KREMORY_BENCH_USE_HYBRID=1 \
/// KREMORY_GLINER_THRESHOLD=0.2 \
/// cargo test -p kremory --features llm-integration,ner \
///   --test phase_g_empirical_hatch -- phase_g_gemma4_e2b_pre_post_dream --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Ollama at $OLLAMA_BASE_URL with gemma4-e2b:latest + nomic-embed-text; enable with --ignored"]
async fn phase_g_gemma4_e2b_pre_post_dream() {
    let base_url = match ollama_base_url() {
        Some(url) => url,
        None => {
            eprintln!(
                "SKIP phase_g_gemma4_e2b_pre_post_dream: OLLAMA_BASE_URL / OLLAMA_HOST not set"
            );
            return;
        }
    };

    let chat_model = ollama_chat_model();
    // Warn if model is not the SoT interactive default.
    if !chat_model.contains("gemma4-e2b") {
        eprintln!(
            "WARNING: OLLAMA_CHAT_MODEL={chat_model} — SoT interactive default is gemma4-e2b:latest \
             (tests/llm_integration.rs:1-25). G2 thresholds assume gemma4-e2b:latest."
        );
    }
    eprintln!("[G] Phase G empirical hatch — provider: gemma4-e2b local (ollama)");
    eprintln!("[G] model={chat_model} base_url={base_url}");

    let use_hybrid = std::env::var("KREMORY_BENCH_USE_HYBRID")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !use_hybrid {
        eprintln!(
            "INFO: KREMORY_BENCH_USE_HYBRID not set — running DefaultExtractor (LLM-only). \
             G2 thresholds assume hybrid+ner config. Set KREMORY_BENCH_USE_HYBRID=1 --features ner for production config."
        );
    }
    eprintln!("[G] hybrid={use_hybrid}");

    let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "1h".to_string());

    // ── Engine setup: gemma4-e2b + nomic-embed-text ───────────────────────────
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(chat_model.clone())
        .timeout_seconds(120)
        .keep_alive(keep_alive)
        .build()
        .expect("Ollama LLM builder");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder");

    // ── Phase G thresholds (ADR-044 §5, Phase A gemma4-e2b Amendment #2 baselines) ──
    // mock_interview Phase A baseline = 100.0% → gate = 98.0% (−2pt)
    // legal_deposition Phase A baseline = 81.2% → gate = 79.2% (−2pt)
    let mock_pre_threshold = 0.98_f64;
    let legal_pre_threshold = 0.792_f64;

    // ── Fixture 1: mock_interview ─────────────────────────────────────────────
    {
        let dir_mock = tempfile::tempdir().expect("tempdir mock");
        let graph_mock = Arc::new(
            TemporalGraph::open(
                dir_mock
                    .path()
                    .join("phase-g-mock.db")
                    .to_str()
                    .expect("valid UTF-8"),
            )
            .await
            .expect("TemporalGraph::open mock"),
        );

        // Build allowed_entity_types for PipelineConfig (hybrid requires explicit list).
        let allowed_for_config: Vec<String> = if use_hybrid {
            DEFAULT_ENTITY_TYPES
                .iter()
                .filter(|(id, _, _)| *id != 0)
                .map(|(_, name, _)| name.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let config_mock = if use_hybrid {
            PipelineConfig::builder()
                .extraction_arm_budget_ms(300_000)
                .allowed_entity_types(allowed_for_config)
                .build()
                .expect("PipelineConfig with allowed_types")
        } else {
            PipelineConfig::builder()
                .extraction_arm_budget_ms(300_000)
                .build()
                .expect("PipelineConfig default")
        };

        let engine_mock = Engine::new(
            Arc::clone(&graph_mock),
            Arc::clone(&llm),
            Arc::new(OllamaEmbedderAdapter(Arc::clone(&raw_emb))),
            config_mock,
        );

        let result_mock = measure_pre_post(MeasureArgs {
            engine: &engine_mock,
            llm: Arc::clone(&llm),
            graph: &graph_mock,
            fixture_key: "mock_interview",
            provider_label: &chat_model,
            pre_threshold: mock_pre_threshold,
            use_hybrid,
        })
        .await;

        // G2: mock PRE precision ≥ 98% (100% − 2pt) per gemma4-e2b:latest Phase A baseline.
        assert!(
            result_mock.pass_pre,
            "[G] G2 FAIL: mock_interview PRE-dream precision {:.1}% ({}/{}) < {:.0}% threshold.\n\
             Model: {chat_model}\n\
             Phase A baseline (gemma4-e2b:latest, hybrid+ner, thr=0.2): 100.0% → G2 gate = 98.0%\n\
             Per ADR-044 §5 per-provider INDEPENDENT pass: gemma4-e2b must independently clear this gate.",
            result_mock.pre_precision * 100.0,
            result_mock.pre_matched,
            result_mock.pre_total,
            mock_pre_threshold * 100.0,
        );

        // POST-dream must not regress vs PRE.
        assert!(
            result_mock.pass_post,
            "[G] POST-dream REGRESSION: mock_interview {:.1}% → {:.1}% (−{:.1}pt). \
             Dream pass must not degrade precision.",
            result_mock.pre_precision * 100.0,
            result_mock.post_precision * 100.0,
            (result_mock.pre_precision - result_mock.post_precision) * 100.0,
        );

        eprintln!(
            "[G] mock_interview PASS — PRE {:.1}% POST {:.1}% Δ{:+.1}pt reclassified={}",
            result_mock.pre_precision * 100.0,
            result_mock.post_precision * 100.0,
            (result_mock.post_precision - result_mock.pre_precision) * 100.0,
            result_mock.dream_entities_reclassified,
        );
    }

    // ── Fixture 2: legal_deposition ───────────────────────────────────────────
    {
        let dir_legal = tempfile::tempdir().expect("tempdir legal");
        let graph_legal = Arc::new(
            TemporalGraph::open(
                dir_legal
                    .path()
                    .join("phase-g-legal.db")
                    .to_str()
                    .expect("valid UTF-8"),
            )
            .await
            .expect("TemporalGraph::open legal"),
        );

        let allowed_for_config_legal: Vec<String> = if use_hybrid {
            DEFAULT_ENTITY_TYPES
                .iter()
                .filter(|(id, _, _)| *id != 0)
                .map(|(_, name, _)| name.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let config_legal = if use_hybrid {
            PipelineConfig::builder()
                .extraction_arm_budget_ms(300_000)
                .allowed_entity_types(allowed_for_config_legal)
                .build()
                .expect("PipelineConfig legal allowed_types")
        } else {
            PipelineConfig::builder()
                .extraction_arm_budget_ms(300_000)
                .build()
                .expect("PipelineConfig legal default")
        };

        let engine_legal = Engine::new(
            Arc::clone(&graph_legal),
            Arc::clone(&llm),
            Arc::new(OllamaEmbedderAdapter(Arc::clone(&raw_emb))),
            config_legal,
        );

        let result_legal = measure_pre_post(MeasureArgs {
            engine: &engine_legal,
            llm: Arc::clone(&llm),
            graph: &graph_legal,
            fixture_key: "legal_deposition",
            provider_label: &chat_model,
            pre_threshold: legal_pre_threshold,
            use_hybrid,
        })
        .await;

        // G2: legal PRE precision ≥ 79.2% (81.2% − 2pt) per gemma4-e2b:latest Phase A baseline.
        assert!(
            result_legal.pass_pre,
            "[G] G2 FAIL: legal_deposition PRE-dream precision {:.1}% ({}/{}) < {:.1}% threshold.\n\
             Model: {chat_model}\n\
             Phase A baseline (gemma4-e2b:latest, hybrid+ner, thr=0.2): 81.2% → G2 gate = 79.2%\n\
             Known misses: CLIFFORD REEVES, DIANA OHLSSON (all-caps GLiNER miss), \
             Feldstein & Moorhouse LLP (GLiNER ampersand limitation — pre-known gaps from Phase A).\n\
             Per ADR-044 §5 per-provider INDEPENDENT pass.",
            result_legal.pre_precision * 100.0,
            result_legal.pre_matched,
            result_legal.pre_total,
            legal_pre_threshold * 100.0,
        );

        // POST-dream must not regress vs PRE.
        assert!(
            result_legal.pass_post,
            "[G] POST-dream REGRESSION: legal_deposition {:.1}% → {:.1}% (−{:.1}pt). \
             Dream pass must not degrade precision.",
            result_legal.pre_precision * 100.0,
            result_legal.post_precision * 100.0,
            (result_legal.pre_precision - result_legal.post_precision) * 100.0,
        );

        eprintln!(
            "[G] legal_deposition PASS — PRE {:.1}% POST {:.1}% Δ{:+.1}pt reclassified={}",
            result_legal.pre_precision * 100.0,
            result_legal.post_precision * 100.0,
            (result_legal.post_precision - result_legal.pre_precision) * 100.0,
            result_legal.dream_entities_reclassified,
        );
    }

    eprintln!("[G] === Phase G gemma4-e2b local gate: PASS ===");
    eprintln!(
        "[G] G4: per-provider INDEPENDENT pass satisfied (gemma4-e2b cleared independently)."
    );
    eprintln!(
        "[G] G3: cloud-LLM gate — see `phase_g_cloud_llm_gate` test (runs if API key present)."
    );
}

// ── G3: Cloud-LLM gate (Anthropic) ───────────────────────────────────────────

/// Phase G cloud-LLM gate: Anthropic provider via `ANTHROPIC_API_KEY`.
///
/// Runs ONLY if `ANTHROPIC_API_KEY` is set in environment. If absent, skips
/// gracefully and logs a deferred notice (TD-017 follow-up per plan §G5 / risk R11).
///
/// Per ADR-044 §5 per-provider INDEPENDENT pass: this test's verdict is
/// INDEPENDENT of `phase_g_gemma4_e2b_pre_post_dream`. Both must pass for
/// full G4 compliance with cloud-LLM enabled.
///
/// ## How to run
///
/// ```sh
/// ANTHROPIC_API_KEY=sk-ant-... \
/// OLLAMA_BASE_URL=http://localhost:11434 \
/// cargo test -p kremory --features llm-integration \
///   --test phase_g_empirical_hatch -- phase_g_cloud_llm_gate --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires $ANTHROPIC_API_KEY or $OPENAI_API_KEY + $OLLAMA_BASE_URL; skips gracefully if absent"]
async fn phase_g_cloud_llm_gate() {
    use autoagents_llm::backends::anthropic::Anthropic;

    let base_url = match ollama_base_url() {
        Some(url) => url,
        None => {
            eprintln!("SKIP phase_g_cloud_llm_gate: OLLAMA_BASE_URL / OLLAMA_HOST not set (needed for embeddings)");
            return;
        }
    };

    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if anthropic_key.is_empty() {
        eprintln!(
            "SKIP phase_g_cloud_llm_gate: ANTHROPIC_API_KEY not set.\n\
             G3 cloud-LLM gate deferred to v0.1.2 per plan §G5 / risk R11.\n\
             When the key is available, re-run this test to complete G3/G4 cloud-provider gate.\n\
             Filed as TD-017 follow-up."
        );
        return;
    }

    let cloud_model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    let provider_label = format!("anthropic:{cloud_model}");
    eprintln!("[G] Phase G cloud-LLM gate — provider: {provider_label}");

    // G2 thresholds apply to cloud provider too (ADR-044 §5 independent pass).
    // mock_interview: ≥ 98%, legal_deposition: ≥ 79.2%
    let mock_pre_threshold = 0.98_f64;
    let legal_pre_threshold = 0.792_f64;

    let llm: Arc<Anthropic> = LLMBuilder::<Anthropic>::new()
        .api_key(anthropic_key)
        .model(cloud_model.clone())
        .timeout_seconds(120)
        .build()
        .expect("Anthropic LLM builder");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder");
    // raw_emb is Arc<Ollama> — clone the Arc to create independent adapters per engine.
    let raw_emb_arc = raw_emb;

    // ── mock_interview ────────────────────────────────────────────────────────
    {
        let dir = tempfile::tempdir().expect("tempdir mock cloud");
        let graph = Arc::new(
            TemporalGraph::open(
                dir.path()
                    .join("phase-g-cloud-mock.db")
                    .to_str()
                    .expect("valid UTF-8"),
            )
            .await
            .expect("TemporalGraph::open cloud mock"),
        );

        let config = PipelineConfig::builder()
            .extraction_arm_budget_ms(120_000)
            .build()
            .expect("PipelineConfig cloud default");

        let engine = Engine::new(
            Arc::clone(&graph),
            Arc::clone(&llm),
            Arc::new(OllamaEmbedderAdapter(Arc::clone(&raw_emb_arc))),
            config,
        );

        // Cloud provider uses DefaultExtractor (no GLiNER hybrid — cloud inference only).
        let result = measure_pre_post(MeasureArgs {
            engine: &engine,
            llm: Arc::clone(&llm),
            graph: &graph,
            fixture_key: "mock_interview",
            provider_label: &provider_label,
            pre_threshold: mock_pre_threshold,
            use_hybrid: false, // cloud = DefaultExtractor (no hybrid)
        })
        .await;

        assert!(
            result.pass_pre,
            "[G] G4 FAIL (cloud): mock_interview PRE {:.1}% ({}/{}) < {:.0}% threshold. \
             Provider: {provider_label} must independently clear G2 gate (ADR-044 §5).",
            result.pre_precision * 100.0,
            result.pre_matched,
            result.pre_total,
            mock_pre_threshold * 100.0,
        );
        assert!(
            result.pass_post,
            "[G] POST-dream REGRESSION (cloud): mock_interview {:.1}% → {:.1}%",
            result.pre_precision * 100.0,
            result.post_precision * 100.0,
        );
        eprintln!(
            "[G] cloud mock_interview PASS — PRE {:.1}% POST {:.1}%",
            result.pre_precision * 100.0,
            result.post_precision * 100.0,
        );
    }

    // ── legal_deposition ──────────────────────────────────────────────────────
    {
        let dir = tempfile::tempdir().expect("tempdir legal cloud");
        let graph = Arc::new(
            TemporalGraph::open(
                dir.path()
                    .join("phase-g-cloud-legal.db")
                    .to_str()
                    .expect("valid UTF-8"),
            )
            .await
            .expect("TemporalGraph::open cloud legal"),
        );

        let config = PipelineConfig::builder()
            .extraction_arm_budget_ms(120_000)
            .build()
            .expect("PipelineConfig cloud legal default");

        let engine = Engine::new(
            Arc::clone(&graph),
            Arc::clone(&llm),
            Arc::new(OllamaEmbedderAdapter(Arc::clone(&raw_emb_arc))),
            config,
        );

        let result = measure_pre_post(MeasureArgs {
            engine: &engine,
            llm: Arc::clone(&llm),
            graph: &graph,
            fixture_key: "legal_deposition",
            provider_label: &provider_label,
            pre_threshold: legal_pre_threshold,
            use_hybrid: false,
        })
        .await;

        assert!(
            result.pass_pre,
            "[G] G4 FAIL (cloud): legal_deposition PRE {:.1}% ({}/{}) < {:.1}% threshold. \
             Provider: {provider_label} must independently clear G2 gate (ADR-044 §5).",
            result.pre_precision * 100.0,
            result.pre_matched,
            result.pre_total,
            legal_pre_threshold * 100.0,
        );
        assert!(
            result.pass_post,
            "[G] POST-dream REGRESSION (cloud): legal_deposition {:.1}% → {:.1}%",
            result.pre_precision * 100.0,
            result.post_precision * 100.0,
        );
        eprintln!(
            "[G] cloud legal_deposition PASS — PRE {:.1}% POST {:.1}%",
            result.pre_precision * 100.0,
            result.post_precision * 100.0,
        );
    }

    eprintln!("[G] === Phase G cloud-LLM gate ({provider_label}): PASS ===");
    eprintln!("[G] G4: {provider_label} cleared G2 thresholds independently — per-provider INDEPENDENT pass satisfied.");
}

// ── Unit tests for label_precision helper (compile/run on every PR) ───────────

#[cfg(test)]
mod unit_tests {
    #![allow(clippy::unwrap_used)]
    use super::label_precision;

    #[test]
    fn perfect_match() {
        let ext = vec![
            ("Alice".into(), "Person".into()),
            ("Acme".into(), "Organisation".into()),
        ];
        let exp = vec![
            ("Alice".into(), "Person".into()),
            ("Acme".into(), "Organisation".into()),
        ];
        let (p, m, t) = label_precision(&ext, &exp);
        assert_eq!(m, 2);
        assert_eq!(t, 2);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn wrong_label_is_miss() {
        let ext = vec![("Alice".into(), "Entity".into())];
        let exp = vec![("Alice".into(), "Person".into())];
        let (p, m, t) = label_precision(&ext, &exp);
        assert_eq!(m, 0, "wrong label must not count as match");
        assert_eq!(t, 1);
        assert!((p - 0.0).abs() < 1e-9);
    }

    #[test]
    fn fuzzy_name_with_correct_label() {
        let ext = vec![("Amazon".into(), "Organisation".into())];
        let exp = vec![("Amazon Robotics".into(), "Organisation".into())];
        let (p, m, t) = label_precision(&ext, &exp);
        assert_eq!(m, 1, "fuzzy name + exact label must match");
        assert_eq!(t, 1);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_expected_is_perfect() {
        let ext = vec![("Alice".into(), "Person".into())];
        let exp: Vec<(String, String)> = vec![];
        let (p, m, t) = label_precision(&ext, &exp);
        assert_eq!(m, 0);
        assert_eq!(t, 0);
        assert!((p - 1.0).abs() < 1e-9);
    }
}
