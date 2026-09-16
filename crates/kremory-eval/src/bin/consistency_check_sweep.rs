//! Dream Pass 4 — τ calibration sweep + RISK-001 acceptance gate (Phase D, v0.1.2).
//!
//! # What this binary does
//!
//! 1. Ingests `fixtures/mis_typed_high_conf.txt` through the full kremory pipeline
//!    (Phase 1 NER with extended entity types) into a temp-file libSQL DB.
//! 2. Queries pre-precision against `fixtures/mis_typed_high_conf_ground_truth.json`.
//! 3. For each τ ∈ {0.3, 0.5, 0.6, 0.7, 0.8}:
//!    - Copies the ingested DB to a scratch file.
//!    - Opens a fresh `TemporalGraph` on the copy.
//!    - Runs `run_consistency_check` with that τ.
//!    - Queries post-precision and computes F1.
//!    - Discards the scratch copy.
//! 4. Picks the τ with the highest F1 (ties broken by precision).
//! 5. Runs the **RISK-001 acceptance gate** N times (default N=5, `--n-runs`):
//!    each run does a *fresh* ingest on its own temp DB, runs consistency_check
//!    with the chosen τ, and records the precision lift. The gate then computes
//!    the sample mean ± SD of those N lifts and asserts `mean - SD >= 0.05` (5pt).
//! 6. Writes sweep results to `.ai-docs/lessons/2026-06-10-v0-1-2-tau-calibration-sweep.md`.
//! 7. Writes RISK-001 verdict (per-run values + mean ± SD) to
//!    `.ai-docs/lessons/2026-06-10-v0-1-2-risk-001-acceptance-verdict.md`.
//!
//! # Usage
//!
//! ```text
//! OLLAMA_HOST=http://localhost:11434 \
//! cargo run -p kremory-eval --release --bin consistency_check_sweep -- --n-runs 5
//! ```
//!
//! `--n-runs <N>` (or env var `RISK001_N_RUNS`) overrides the RISK-001 gate's
//! repeat count; default is 5. `N` is the multi-run aggregation TD-038 asked
//! for — see "RISK-001 gate" section below for why a single-shot measurement
//! is not trustworthy at this threshold.
//!
//! ⚠️ **Standing HITL gate**: every run of this binary performs a REAL ingest
//! through the full kremory pipeline (Phase 1 NER via Ollama, optionally the
//! paid Anthropic verify provider via `KREMORY_VERIFY_PROVIDER=anthropic`).
//! Per this repo's CLAUDE.md, any paid benchmark run — including re-ingesting
//! the bench corpora N times for this gate — is off-limits for autonomous
//! execution. Running the real N=5 sweep (Ollama-only or Anthropic-verify) is
//! a MANUAL step for a human; do not wire this into an autonomous pipeline.
//!
//! Stop condition (per sprint plan §Stop Conditions #1):
//! If RISK-001 gate FAILS (mean - SD < 5pt), binary exits with code 1 and
//! writes FAIL verdict. Do NOT proceed to Phase E without resolving the FAIL.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use autoagents_llm::{
    backends::{anthropic::Anthropic, ollama::Ollama},
    builder::LLMBuilder,
    chat::ChatProvider,
    embedding::{model_provider::EmbeddingBuilder, EmbeddingProvider as AutoEmbeddingProvider},
};
use chrono::Utc;
use serde::Deserialize;

use kremory::{
    core::{
        config::PipelineConfig,
        dream::consistency_check::{
            run_consistency_check, ConsistencyCheckOpts, RunConsistencyCheckParams,
        },
        entity_types::{EntityTypeSpec, DEFAULT_ENTITY_TYPES},
        error::{Error as KremoryCoreError, Result as KremoryCoreResult},
        extraction::IntegerIdLlmExtractor,
        ingest::{Engine, SourceParams},
        schema::TemporalGraph,
    },
    EmbeddingProvider,
};

// ─── OllamaEmbedAdapter ───────────────────────────────────────────────────────
//
// Same pattern as eval.rs — bridges AA batch-embedder to kremory single-text
// EmbeddingProvider. Duplicated here to avoid a shared-lib dep within the eval
// crate (the crate has no lib.rs, only bins + a lib.rs for the library target).

struct OllamaEmbedAdapter<P> {
    inner: Arc<P>,
}

impl<P> EmbeddingProvider for OllamaEmbedAdapter<P>
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

// ─── Ground truth ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GroundTruthEntity {
    name: String,
    entity_type: String,
    #[serde(default)]
    #[allow(dead_code)]
    sentence_index: usize,
    #[serde(default)]
    #[allow(dead_code)]
    dominant_domain_wrong_type: String,
    #[serde(default)]
    #[allow(dead_code)]
    note: String,
}

fn load_ground_truth(fixtures_dir: &Path) -> Result<Vec<GroundTruthEntity>> {
    let path = fixtures_dir.join("mis_typed_high_conf_ground_truth.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read ground truth: {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| "failed to parse ground truth JSON")
}

// ─── Extended entity type registry ───────────────────────────────────────────
//
// The ground truth uses types beyond the default OntoNotes-10 vocabulary.
// We register these as extra types so Phase 1 NER can assign them AND
// consistency_check has their descriptions for the embed-prefilter.
//
// IDs start from 100 to avoid collision with default IDs 0-10.

const EXTRA_TYPES: &[(u32, &str, &str)] = &[
    (100, "Organization", "A business entity, company, brand, or corporate organisation that sells products or services."),
    (101, "Technology", "A programming language, software framework, library, platform, or technical tool used in software development."),
    (102, "Animal", "A living creature of the animal kingdom; a bird, mammal, reptile, fish, or insect."),
    (103, "Product", "A manufactured good, consumer product, substance, or commodity sold to end users."),
    (104, "Plant", "A plant species, herb, flower, tree, or botanical organism."),
    (105, "Language", "A natural human language or dialect spoken by a community of people."),
    (106, "ChemicalElement", "A chemical element, compound, mineral, or substance from the periodic table or chemistry."),
    (107, "NaturalPhenomenon", "A natural event or phenomenon such as a season, weather pattern, geological event, or astronomical occurrence."),
    (108, "Nationality", "A national identity, ethnic group, or cultural demonym tied to a country or region."),
    (109, "Planet", "A planet, moon, star, or celestial body in the solar system or universe."),
];

fn build_source_params() -> SourceParams {
    let mut specs: Vec<EntityTypeSpec> = DEFAULT_ENTITY_TYPES
        .iter()
        .map(|(id, name, desc)| EntityTypeSpec {
            id: *id,
            name: name.to_string(),
            description: desc.to_string(),
        })
        .collect();
    for (id, name, desc) in EXTRA_TYPES {
        specs.push(EntityTypeSpec {
            id: *id,
            name: name.to_string(),
            description: desc.to_string(),
        });
    }
    SourceParams {
        entity_types_override: Some(specs),
        ..SourceParams::default()
    }
}

// ─── Precision computation ────────────────────────────────────────────────────

/// Query all non-catch-all entities from DB, return (name, type_name) pairs.
///
/// Joins entities → entity_types to get the human-readable label.
/// Query all non-catch-all entities from DB, return (name, type_name) pairs.
///
/// Post-Migration-009: `entities.label` was dropped; `entities.id` IS the entity name.
/// Joins entities → entity_types to get the human-readable label.
async fn query_entities(db: &libsql::Connection) -> Result<Vec<(String, String)>> {
    let mut rows = db
        .query(
            "SELECT e.id, et.name \
             FROM entities e \
             JOIN entity_types et ON et.id = e.entity_type_id AND et.group_id = e.group_id \
             WHERE e.entity_type_id != 0",
            (),
        )
        .await
        .context("query_entities: SELECT failed")?;
    let mut result = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .context("query_entities: row iteration failed")?
    {
        let name: String = row.get(0).context("query_entities: id/name column")?;
        let type_name: String = row.get(1).context("query_entities: type_name column")?;
        result.push((name, type_name));
    }
    Ok(result)
}

/// Compute label precision: matched / total_expected.
///
/// Uses fuzzy name matching (case-insensitive substring, either direction)
/// and exact label matching (case-insensitive), consistent with
/// label_precision_benchmark.rs.
fn label_precision(extracted: &[(String, String)], expected: &[GroundTruthEntity]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let matched = expected
        .iter()
        .filter(|gt| {
            let exp_lower = gt.name.to_lowercase();
            let exp_label_lower = gt.entity_type.to_lowercase();
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
    matched as f64 / expected.len() as f64
}

/// Compute F1 on the consistency_check correction task.
///
/// TP = ground truth entity previously wrong that is now correctly typed.
/// FP = entity that was correct and got incorrectly changed.
/// FN = ground truth entity still wrong after the pass.
///
/// The function computes this by comparing pre/post entity lists against GT.
fn compute_f1(
    pre_extracted: &[(String, String)],
    post_extracted: &[(String, String)],
    gt: &[GroundTruthEntity],
) -> (f64, f64, f64) {
    // TP: in GT, was wrong before, is correct after.
    let tp = gt
        .iter()
        .filter(|g| {
            let gn = g.name.to_lowercase();
            let gl = g.entity_type.to_lowercase();
            let was_wrong = !pre_extracted.iter().any(|(n, l)| {
                let nl = n.to_lowercase();
                let ll = l.to_lowercase();
                (nl.contains(gn.as_str()) || gn.contains(nl.as_str())) && ll == gl
            });
            let now_correct = post_extracted.iter().any(|(n, l)| {
                let nl = n.to_lowercase();
                let ll = l.to_lowercase();
                (nl.contains(gn.as_str()) || gn.contains(nl.as_str())) && ll == gl
            });
            was_wrong && now_correct
        })
        .count() as f64;

    // FP: entity was correct before but is now wrong.
    let fp = gt
        .iter()
        .filter(|g| {
            let gn = g.name.to_lowercase();
            let gl = g.entity_type.to_lowercase();
            let was_correct = pre_extracted.iter().any(|(n, l)| {
                let nl = n.to_lowercase();
                let ll = l.to_lowercase();
                (nl.contains(gn.as_str()) || gn.contains(nl.as_str())) && ll == gl
            });
            let now_wrong = !post_extracted.iter().any(|(n, l)| {
                let nl = n.to_lowercase();
                let ll = l.to_lowercase();
                (nl.contains(gn.as_str()) || gn.contains(nl.as_str())) && ll == gl
            });
            was_correct && now_wrong
        })
        .count() as f64;

    // FN: in GT, still wrong after.
    let fn_ = gt
        .iter()
        .filter(|g| {
            let gn = g.name.to_lowercase();
            let gl = g.entity_type.to_lowercase();
            !post_extracted.iter().any(|(n, l)| {
                let nl = n.to_lowercase();
                let ll = l.to_lowercase();
                (nl.contains(gn.as_str()) || gn.contains(nl.as_str())) && ll == gl
            })
        })
        .count() as f64;

    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (f1, precision, recall)
}

// ─── Engine + ingest helpers ──────────────────────────────────────────────────

/// Build an Ollama LLM client.
fn build_llm(ollama_host: &str, model: &str) -> Result<Arc<Ollama>> {
    let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "1h".to_string());
    LLMBuilder::<Ollama>::new()
        .base_url(ollama_host)
        .model(model)
        .timeout_seconds(300) // 5 min — large models can be slow
        .keep_alive(&keep_alive)
        .build()
        .map_err(|e| anyhow::anyhow!("Ollama LLM builder (model={model}): {e}"))
}

/// Build an Anthropic chat client for use as the verify provider.
///
/// Uses the `ANTHROPIC_API_KEY` environment variable. Fails loudly if absent
/// (per [[llm-output-parse-loudly]] — never silently default to no-op).
fn build_anthropic_llm(model: &str) -> Result<Arc<Anthropic>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        anyhow::anyhow!("ANTHROPIC_API_KEY not set — required for anthropic verify provider")
    })?;
    if api_key.is_empty() {
        return Err(anyhow::anyhow!(
            "ANTHROPIC_API_KEY is empty — required for anthropic verify provider"
        ));
    }
    LLMBuilder::<Anthropic>::new()
        .api_key(api_key)
        .model(model)
        .max_tokens(512) // verify calls are short; cap spend
        .timeout_seconds(60)
        .build()
        .map_err(|e| anyhow::anyhow!("Anthropic LLM builder (model={model}): {e}"))
}

/// Build an Ollama embedder + adapter.
fn build_embedder(ollama_host: &str, embed_model: &str) -> Result<Arc<OllamaEmbedAdapter<Ollama>>> {
    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(ollama_host)
        .model(embed_model)
        .build()
        .map_err(|e| anyhow::anyhow!("Ollama embedder builder: {e}"))?;
    Ok(Arc::new(OllamaEmbedAdapter { inner: raw_emb }))
}

async fn ingest_fixture(
    fixture_text: &str,
    db_path: &Path,
    llm: Arc<Ollama>,
    embedder: Arc<OllamaEmbedAdapter<Ollama>>,
) -> Result<Arc<TemporalGraph>> {
    let path_str = db_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("DB path is not valid UTF-8"))?;

    let graph = Arc::new(
        TemporalGraph::open(path_str)
            .await
            .context("TemporalGraph::open failed")?,
    );

    let config = PipelineConfig::builder()
        .extraction_arm_budget_ms(300_000) // 5 min per call — polyseme fixture is short
        .build()
        .context("PipelineConfig::build")?;

    let dyn_emb: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::clone(&embedder) as Arc<dyn kremory::DynEmbeddingProvider>;
    let arc_emb = Arc::new(kremory::ArcEmbedder(dyn_emb));
    let engine = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: Arc::clone(&graph),
        llm: Arc::clone(&llm),
        embedder: arc_emb,
        config,
        model: None,
    });
    let source_params = build_source_params();
    let extractor = IntegerIdLlmExtractor::new(Arc::clone(&llm));

    engine
        .ingest_with(
            &extractor,
            kremory::core::ingest::IngestWithParams {
                text: fixture_text,
                reference_time: None,
                declared_reference_time: None,
                group_id: None,
                content_type: None,
                source_params,
            },
        )
        .await
        .context("ingest_with failed")?;

    Ok(graph)
}

// ─── τ sweep ──────────────────────────────────────────────────────────────────

struct TauResult {
    tau: f32,
    pre_precision: f64,
    post_precision: f64,
    precision_lift: f64,
    f1: f64,
    cc_precision: f64,
    cc_recall: f64,
    scanned: usize,
    flagged: usize,
    corrected: usize,
    // Recorded for the sweep doc even though not used in F1 computation.
    #[allow(dead_code)]
    confirmed: usize,
    #[allow(dead_code)]
    uncertain: usize,
    latency_ms: u64,
}

/// Run one τ point of the calibration sweep.
///
/// `verify_llm` is the model used for consistency_check verification calls.
/// It is intentionally separate from the ingest model so RISK-001 can isolate
/// Dream Pass 4 verification quality from Phase 1 extraction quality.
/// Accepts any `ChatProvider` so Ollama and Anthropic both work.
async fn run_sweep_for_tau(
    base_db_path: &Path,
    tau: f32,
    gt: &[GroundTruthEntity],
    verify_llm: &dyn ChatProvider,
    embedder: Arc<OllamaEmbedAdapter<Ollama>>,
) -> Result<TauResult> {
    // Copy base DB to scratch path for this τ run.
    let scratch_dir = tempfile::tempdir().context("create scratch tempdir")?;
    let scratch_path = scratch_dir.path().join(format!("sweep_tau_{:.2}.db", tau));
    std::fs::copy(base_db_path, &scratch_path)
        .with_context(|| format!("copy base DB to scratch for τ={tau:.2}"))?;

    let scratch_str = scratch_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("scratch DB path is not valid UTF-8"))?;

    let scratch_graph = TemporalGraph::open(scratch_str)
        .await
        .context("open scratch TemporalGraph")?;

    // Pre-precision (on the scratch copy, which is identical to base pre-corrections).
    let pre_entities = query_entities(&scratch_graph.conn)
        .await
        .context("query pre-precision entities")?;
    let pre_precision = label_precision(&pre_entities, gt);

    // Build ArcEmbedder from the shared embedder.
    let dyn_emb: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::clone(&embedder) as Arc<dyn kremory::DynEmbeddingProvider>;
    let arc_embedder: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::new(kremory::ArcEmbedder(dyn_emb));

    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: tau,
        max_candidates_per_run: Some(50),
        verify_model_override: None,
        dry_run: false,
    };

    let summary = run_consistency_check(
        &scratch_graph.conn,
        RunConsistencyCheckParams {
            embedder: arc_embedder.as_ref(),
            llm: verify_llm,
            opts,
        },
    )
    .await
    .context("run_consistency_check failed")?;

    // Post-precision.
    let post_entities = query_entities(&scratch_graph.conn)
        .await
        .context("query post-precision entities")?;
    let post_precision = label_precision(&post_entities, gt);

    let (f1, cc_prec, cc_rec) = compute_f1(&pre_entities, &post_entities, gt);

    Ok(TauResult {
        tau,
        pre_precision,
        post_precision,
        precision_lift: post_precision - pre_precision,
        f1,
        cc_precision: cc_prec,
        cc_recall: cc_rec,
        scanned: summary.scanned,
        flagged: summary.flagged,
        corrected: summary.corrected,
        confirmed: summary.confirmed,
        uncertain: summary.uncertain,
        latency_ms: summary.latency_ms_p50,
    })
}

// ─── RISK-001 gate (TD-038: multi-run mean ± SD hardening) ───────────────────
//
// The gate used to be a single fresh-ingest measurement compared directly
// against the +0.05pt threshold. That is unsound: repeated fresh ingests of
// the SAME fixture show ~0.50pt of pure ingest noise (LLM extraction is not
// deterministic run-to-run even with T=0 + seed pinning — see
// `.ai-docs/lessons/o11y-sprint-retro-2026-06-11.md` TD-H), which is larger
// than the threshold itself. A single unlucky (or lucky) run can flip
// PASS/FAIL on noise alone. TD-038 closes this by running the gate N times
// and requiring the *lower* end of the measured spread to clear the bar.

/// Precision-lift threshold the RISK-001 gate must clear. Absolute, not relative
/// (e.g. 0.05 == 5 percentage points of label precision).
const RISK001_LIFT_THRESHOLD: f64 = 0.05;

/// Default repeat count for the RISK-001 gate (TD-038). Overridable via
/// `--n-runs <N>` or the `RISK001_N_RUNS` env var — see [`parse_n_runs_from`].
const RISK001_DEFAULT_N_RUNS: usize = 5;

/// Sample mean and sample standard deviation (Bessel-corrected, N-1 denominator)
/// over a small in-memory slice of f64 measurements.
///
/// Hand-rolled per this repo's "evaluate a 3p package before hand-rolling" rule's
/// own exemption for genuinely trivial, stable logic (≤~20 lines, no edge-case
/// tail): this is a two-pass mean+variance over at most a handful of precision-lift
/// deltas in `[-1.0, 1.0]`, with no NaN/unit/locale handling required. Pulling in a
/// statistics crate (e.g. `statrs`) for this would be a dependency for arithmetic a
/// short function already expresses correctly.
///
/// Returns `(mean, sample_sd)`. `sample_sd` is `0.0` for `values.len() <= 1` — a
/// single sample has no dispersion estimate, not a divide-by-zero panic.
fn mean_and_sample_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return (mean, 0.0);
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, variance.sqrt())
}

/// RISK-001 gate decision: PASS iff `mean - sd >= threshold`.
///
/// This is the TD-038 fix itself. Gating on `mean` alone (the pre-TD-038
/// behaviour, restated here as what this function deliberately does NOT do)
/// lets a high-variance result pass purely because its single measured mean
/// happened to clear the bar, even though a materially large share of the
/// underlying runs did not. Subtracting the sample SD requires the *worst
/// plausible* run (one SD below the mean) to still clear the threshold.
fn risk001_gate_pass(lift_mean: f64, lift_sd: f64, threshold: f64) -> bool {
    lift_mean - lift_sd >= threshold
}

/// Parse the RISK-001 gate's repeat count from CLI args (`--n-runs <N>` /
/// `--n-runs=<N>`), falling back to the `RISK001_N_RUNS` env var, then to
/// [`RISK001_DEFAULT_N_RUNS`].
///
/// Takes an explicit slice (rather than reading `std::env::args()` directly)
/// so it is unit-testable — same pattern as `eval.rs`'s `parse_args_from`.
fn parse_n_runs_from(raw: &[String], env_override: Option<&str>) -> Result<usize, String> {
    let mut i = 0;
    while i < raw.len() {
        let arg = &raw[i];
        if arg == "--n-runs" {
            let val = raw
                .get(i + 1)
                .ok_or_else(|| "--n-runs requires a value".to_string())?;
            return val
                .parse::<usize>()
                .map_err(|e| format!("--n-runs value {val:?} invalid: {e}"))
                .and_then(require_at_least_one);
        }
        if let Some(val) = arg.strip_prefix("--n-runs=") {
            return val
                .parse::<usize>()
                .map_err(|e| format!("--n-runs value {val:?} invalid: {e}"))
                .and_then(require_at_least_one);
        }
        i += 1;
    }
    if let Some(val) = env_override {
        return val
            .parse::<usize>()
            .map_err(|e| format!("RISK001_N_RUNS value {val:?} invalid: {e}"))
            .and_then(require_at_least_one);
    }
    Ok(RISK001_DEFAULT_N_RUNS)
}

fn require_at_least_one(n: usize) -> Result<usize, String> {
    if n == 0 {
        Err("--n-runs / RISK001_N_RUNS must be >= 1".to_string())
    } else {
        Ok(n)
    }
}

/// One RISK-001 run's measurements (before aggregation).
struct Risk001RunResult {
    run_index: usize,
    pre_precision: f64,
    post_precision: f64,
    precision_lift: f64,
    scanned: usize,
    corrected: usize,
}

/// Aggregated RISK-001 verdict across `n_runs` fresh-ingest repeats (TD-038).
struct Risk001Verdict {
    tau_used: f32,
    n_runs: usize,
    runs: Vec<Risk001RunResult>,
    lift_mean: f64,
    lift_sd: f64,
    pass: bool,
}

/// Run ONE RISK-001 measurement: fresh ingest on its own temp DB, consistency_check
/// at `tau`, pre/post precision + lift. No pass/fail decision here — that only
/// makes sense across the aggregated N runs (see [`run_risk001_gate`]).
///
/// `ingest_llm` handles Phase 1 NER (needs to handle 21-type schema reliably).
/// `verify_llm` handles consistency_check verification calls (the model under test).
/// `verify_llm` is `&dyn ChatProvider` so Ollama and Anthropic both work without boxing.
async fn run_risk001_gate_once(
    fixture_text: &str,
    tau: f32,
    gt: &[GroundTruthEntity],
    ingest_llm: Arc<Ollama>,
    verify_llm: &dyn ChatProvider,
    embedder: Arc<OllamaEmbedAdapter<Ollama>>,
    run_index: usize,
) -> Result<Risk001RunResult> {
    let dir = tempfile::tempdir().context("risk001 tempdir")?;
    let db_path = dir.path().join("risk001.db");

    eprintln!("[risk001] run {run_index}: fresh ingest (τ={tau:.2})...");
    let graph = ingest_fixture(
        fixture_text,
        &db_path,
        Arc::clone(&ingest_llm),
        Arc::clone(&embedder),
    )
    .await
    .context("risk001 ingest")?;

    let pre_entities = query_entities(&graph.conn)
        .await
        .context("risk001 pre-precision query")?;
    let pre_precision = label_precision(&pre_entities, gt);

    eprintln!(
        "[risk001] run {run_index}: pre-precision={:.4} ({}/{})",
        pre_precision,
        (pre_precision * gt.len() as f64) as usize,
        gt.len()
    );

    let dyn_emb2: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::clone(&embedder) as Arc<dyn kremory::DynEmbeddingProvider>;
    let arc_embedder: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::new(kremory::ArcEmbedder(dyn_emb2));

    let opts = ConsistencyCheckOpts {
        embed_prefilter_threshold: tau,
        max_candidates_per_run: Some(50),
        verify_model_override: None,
        dry_run: false,
    };

    let summary = run_consistency_check(
        &graph.conn,
        RunConsistencyCheckParams {
            embedder: arc_embedder.as_ref(),
            llm: verify_llm,
            opts,
        },
    )
    .await
    .context("risk001 run_consistency_check")?;

    eprintln!(
        "[risk001] run {run_index}: consistency_check: scanned={} flagged={} corrected={} confirmed={} uncertain={}",
        summary.scanned, summary.flagged, summary.corrected, summary.confirmed, summary.uncertain
    );

    let post_entities = query_entities(&graph.conn)
        .await
        .context("risk001 post-precision query")?;
    let post_precision = label_precision(&post_entities, gt);

    let precision_lift = post_precision - pre_precision;

    eprintln!(
        "[risk001] run {run_index}: post-precision={:.4} lift={:.4}",
        post_precision, precision_lift,
    );

    Ok(Risk001RunResult {
        run_index,
        pre_precision,
        post_precision,
        precision_lift,
        scanned: summary.scanned,
        corrected: summary.corrected,
    })
}

/// Run the RISK-001 acceptance gate as `n_runs` independent fresh-ingest
/// repeats (TD-038), then gate on `mean(lift) - sample_sd(lift) >= threshold`
/// rather than a single raw measurement.
///
/// ⚠️ Every call to this function performs `n_runs` REAL ingests through the
/// full kremory pipeline. Per this repo's CLAUDE.md standing HITL gate, any
/// paid benchmark run — including re-ingesting the bench corpora N times for
/// this gate — must NOT be invoked autonomously. This is a MANUAL step: a
/// human runs the binary (`cargo run -p kremory-eval --release --bin
/// consistency_check_sweep -- --n-runs 5`) with Ollama (and optionally the
/// paid Anthropic verify provider) reachable.
async fn run_risk001_gate(
    fixture_text: &str,
    tau: f32,
    gt: &[GroundTruthEntity],
    ingest_llm: Arc<Ollama>,
    verify_llm: &dyn ChatProvider,
    embedder: Arc<OllamaEmbedAdapter<Ollama>>,
    n_runs: usize,
) -> Result<Risk001Verdict> {
    let mut runs = Vec::with_capacity(n_runs);
    for run_index in 0..n_runs {
        let run = run_risk001_gate_once(
            fixture_text,
            tau,
            gt,
            Arc::clone(&ingest_llm),
            verify_llm,
            Arc::clone(&embedder),
            run_index,
        )
        .await
        .with_context(|| format!("risk001 run {}/{n_runs}", run_index + 1))?;
        runs.push(run);
    }

    let lifts: Vec<f64> = runs.iter().map(|r| r.precision_lift).collect();
    let (lift_mean, lift_sd) = mean_and_sample_sd(&lifts);
    let pass = risk001_gate_pass(lift_mean, lift_sd, RISK001_LIFT_THRESHOLD);

    eprintln!(
        "[risk001] aggregated over {n_runs} runs: lift_mean={:.4} lift_sd={:.4} mean-sd={:.4} ({}) threshold={:.2}",
        lift_mean,
        lift_sd,
        lift_mean - lift_sd,
        if pass { "PASS" } else { "FAIL" },
        RISK001_LIFT_THRESHOLD,
    );

    Ok(Risk001Verdict {
        tau_used: tau,
        n_runs,
        runs,
        lift_mean,
        lift_sd,
        pass,
    })
}

// ─── Document writers ─────────────────────────────────────────────────────────

fn write_sweep_doc(
    workspace_root: &Path,
    results: &[TauResult],
    best_tau: f32,
    verify_provider_label: &str,
) -> Result<()> {
    let dir = workspace_root.join(".ai-docs").join("lessons");
    std::fs::create_dir_all(&dir).context("create .ai-docs/lessons")?;
    let path = dir.join("2026-06-10-v0-1-2-tau-calibration-sweep.md");

    let mut lines = Vec::new();
    lines.push("# v0.1.2 τ Calibration Sweep — Dream Pass 4".to_string());
    lines.push(format!("\nGenerated: {}", Utc::now().to_rfc3339()));
    lines.push("\n## Summary\n".to_string());
    lines.push(format!(
        "**Chosen τ: {best_tau:.2}** (highest F1, ties broken by cc_precision)\n"
    ));
    lines.push("\n## Per-τ Results\n".to_string());
    lines.push("| τ | pre_prec | post_prec | lift | F1 | cc_prec | cc_rec | scanned | flagged | corrected | latency_ms |".to_string());
    lines.push("|---|---|---|---|---|---|---|---|---|---|---|".to_string());

    for r in results {
        lines.push(format!(
            "| {:.2} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {} | {} | {} |",
            r.tau,
            r.pre_precision,
            r.post_precision,
            r.precision_lift,
            r.f1,
            r.cc_precision,
            r.cc_recall,
            r.scanned,
            r.flagged,
            r.corrected,
            r.latency_ms,
        ));
    }

    lines.push("\n## Notes\n".to_string());
    lines.push("- F1 computed on correction task: TP=was-wrong-now-correct, FP=was-correct-now-wrong, FN=still-wrong-after.".to_string());
    lines.push("- Fixture: `crates/kremory-eval/fixtures/mis_typed_high_conf.txt` (10 polysemous entities, Phase D TD-036).".to_string());
    lines.push(format!(
        "- Verify provider (RISK-001 target): `{verify_provider_label}`."
    ));
    lines.push("- Ingest model: `qwen2.5:14b` (handles 21-type integer enum; EXTRA_TYPES required for fixture GT).".to_string());
    lines.push("- τ precedent: Graphiti NODE_DEDUP_COSINE_MIN_SCORE=0.6 — NOT validated for kremory type-validation space.".to_string());

    std::fs::write(&path, lines.join("\n")).with_context(|| format!("write {}", path.display()))?;
    eprintln!("[sweep] Sweep doc written: {}", path.display());
    Ok(())
}

fn write_risk001_doc(
    workspace_root: &Path,
    verdict: &Risk001Verdict,
    chat_model: &str,
) -> Result<()> {
    let dir = workspace_root.join(".ai-docs").join("lessons");
    std::fs::create_dir_all(&dir).context("create .ai-docs/lessons")?;
    let path = dir.join("2026-06-10-v0-1-2-risk-001-acceptance-verdict.md");

    let status = if verdict.pass { "PASS" } else { "FAIL" };
    let mean_minus_sd = verdict.lift_mean - verdict.lift_sd;
    let mut lines = Vec::new();
    lines.push("# RISK-001 Acceptance Gate Verdict — Dream Pass 4".to_string());
    lines.push(format!("\nGenerated: {}", Utc::now().to_rfc3339()));
    lines.push(format!("\n## Verdict: **{status}**\n"));
    lines.push(format!(
        "| Field | Value |\n|---|---|\n\
         | τ_used | {:.2} |\n\
         | n_runs | {} |\n\
         | lift_mean | {:.4} |\n\
         | lift_sd | {:.4} |\n\
         | mean − sd | {:.4} |\n\
         | threshold | {:.2} |\n\
         | pass | {} |\n\
         | model | {} |",
        verdict.tau_used,
        verdict.n_runs,
        verdict.lift_mean,
        verdict.lift_sd,
        mean_minus_sd,
        RISK001_LIFT_THRESHOLD,
        verdict.pass,
        chat_model,
    ));

    // TD-038: per-run values so a human reviewing this doc can see the raw
    // spread the mean ± SD aggregation was computed from, not just the
    // aggregate.
    lines.push("\n## Per-Run Results (TD-038 multi-run aggregation)\n".to_string());
    lines.push("| run | pre_precision | post_precision | lift | scanned | corrected |".to_string());
    lines.push("|---|---|---|---|---|---|".to_string());
    for r in &verdict.runs {
        lines.push(format!(
            "| {} | {:.4} | {:.4} | {:.4} | {} | {} |",
            r.run_index + 1,
            r.pre_precision,
            r.post_precision,
            r.precision_lift,
            r.scanned,
            r.corrected,
        ));
    }

    if !verdict.pass {
        lines.push("\n## Stop Condition #1 — Action Required\n".to_string());
        lines.push(format!(
            "RISK-001 FAIL: mean − sd = {:.4} < {:.2} required (mean={:.4}, sd={:.4} over {} runs).\n\
             ADR-047 sub-decision (i/ii/iii/iv) revision REQUIRED before Phase E per Stop Condition #1.\n\
             Do NOT proceed to Phase E without resolving this verdict.",
            mean_minus_sd, RISK001_LIFT_THRESHOLD, verdict.lift_mean, verdict.lift_sd, verdict.n_runs
        ));
    } else {
        lines.push("\n## Outcome\n".to_string());
        lines.push(format!(
            "RISK-001 PASS. mean − sd = {:.4} ≥ {:.2} over {} runs (TD-038 multi-run gate).\n\
             Dream Pass 4 delivers a precision lift that clears the threshold even at the\n\
             low end of the measured spread. Proceed to Phase E.",
            mean_minus_sd, RISK001_LIFT_THRESHOLD, verdict.n_runs
        ));
    }

    lines.push("\n## Notes\n".to_string());
    lines.push("- Fixture: `crates/kremory-eval/fixtures/mis_typed_high_conf.txt` (10 polysemous entities, TD-036).".to_string());
    lines.push(
        "- Ground truth: `crates/kremory-eval/fixtures/mis_typed_high_conf_ground_truth.json`."
            .to_string(),
    );
    lines.push("- τ chosen from sweep (highest F1 per D3/D4 DoD).".to_string());
    lines.push(
        "- TD-038: gate is `mean(lift) - sample_sd(lift) >= threshold` across N independent \
         fresh ingests, NOT a single raw measurement — closes the gap where the ~0.50pt \
         ingest-noise floor could flip a single-shot PASS/FAIL on noise alone."
            .to_string(),
    );

    std::fs::write(&path, lines.join("\n")).with_context(|| format!("write {}", path.display()))?;
    eprintln!("[risk001] Verdict doc written: {}", path.display());
    Ok(())
}

// ─── Metrics capture (O11y Sprint O0.2) ──────────────────────────────────────
//
// Wires a `DebuggingRecorder` as the global metrics recorder so every
// `counter!` / `histogram!` emission throughout `kremory` lands in a queryable
// snapshot. Closes the previous gap where `kremory.dream.consistency_check.
// llm_call_latency_ms_histogram` was emitted but routed to /dev/null.
//
// Same recorder type as `crates/kremory/tests/helpers/metrics_capture.rs` but
// installed GLOBALLY rather than thread-locally. The global install is required
// because async kremory code emits metrics inside `.await` continuations that
// may resume on different tokio worker threads — `with_local_recorder` (which
// the test helper uses) would not capture those.

fn install_metrics_recorder() -> metrics_util::debugging::Snapshotter {
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    // Surfaces install-failure with a WARN. If a previous process or test
    // harness already installed a global recorder, this snapshotter would
    // observe ZERO metrics and the downstream JSON dump would silently
    // report `metric_count: 0`. The WARN tells the operator why.
    if metrics::set_global_recorder(recorder).is_err() {
        eprintln!(
            "[sweep] WARN: global metrics recorder already installed — snapshot will be EMPTY. \
             Re-run from a clean process to capture metrics."
        );
    }
    snapshotter
}

#[derive(serde::Serialize)]
struct MetricEntry {
    name: String,
    kind: String,
    labels: Vec<(String, String)>,
    value: serde_json::Value,
}

fn snapshot_to_json(snapshotter: &metrics_util::debugging::Snapshotter) -> Vec<MetricEntry> {
    use metrics_util::debugging::DebugValue;
    let snap = snapshotter.snapshot();
    snap.into_vec()
        .into_iter()
        .map(|(key, kind, _unit, value)| {
            let name = key.key().name().to_string();
            let labels: Vec<(String, String)> = key
                .key()
                .labels()
                .map(|l| (l.key().to_string(), l.value().to_string()))
                .collect();
            let value_json = match value {
                DebugValue::Counter(n) => serde_json::json!({ "counter": n }),
                DebugValue::Gauge(f) => serde_json::json!({ "gauge": f.into_inner() }),
                DebugValue::Histogram(samples) => serde_json::json!({
                    "histogram": {
                        "n": samples.len(),
                        "samples": samples.iter().map(|s| s.into_inner()).collect::<Vec<_>>(),
                    }
                }),
            };
            MetricEntry {
                name,
                kind: format!("{kind:?}"),
                labels,
                value: value_json,
            }
        })
        .collect()
}

// ─── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Install metrics recorder FIRST so the rest of the binary emits into it.
    let snapshotter = install_metrics_recorder();
    let start = std::time::Instant::now();

    // ── Env config ──────────────────────────────────────────────────────────
    let ollama_host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Two-model design: ingest model handles the 21-type registry (needs structured-output
    // reliability with large enum); verify model is the RISK-001 target under test.
    //
    // SoT: tests/llm_integration.rs:1-25.
    // - verify_model: gemma4-e2b:latest (interactive default, RISK-001 target)
    // - ingest_model: qwen2.5:14b (legacy fallback, handles 21-type integer enum reliably)
    //
    // KREMORY_VERIFY_PROVIDER=anthropic routes verify calls to Anthropic instead of Ollama,
    // for running Dream Pass 4 verification with only the frontier model.
    let verify_provider_name =
        std::env::var("KREMORY_VERIFY_PROVIDER").unwrap_or_else(|_| "ollama".to_string());
    let verify_model = std::env::var("KREMORY_VERIFY_MODEL")
        .or_else(|_| std::env::var("OLLAMA_CHAT_MODEL"))
        .unwrap_or_else(|_| {
            if verify_provider_name.to_lowercase() == "anthropic" {
                "claude-haiku-4-5-20251001".to_string()
            } else {
                "gemma4-e2b:latest".to_string()
            }
        });
    let ingest_model =
        std::env::var("OLLAMA_INGEST_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string());
    let embed_model =
        std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    // TD-038: RISK-001 gate repeat count. CLI flag wins over env var wins over default.
    let cli_args: Vec<String> = std::env::args().skip(1).collect();
    let n_runs_env = std::env::var("RISK001_N_RUNS").ok();
    let n_runs = parse_n_runs_from(&cli_args, n_runs_env.as_deref())
        .map_err(|e| anyhow::anyhow!("failed to parse RISK-001 run count: {e}"))?;

    // Human-readable label for docs (never contains the API key).
    let verify_provider_label = format!("{verify_provider_name}/{verify_model}");

    eprintln!("[sweep] ollama={ollama_host} ingest_model={ingest_model} verify_provider={verify_provider_label} embed={embed_model} n_runs={n_runs}");

    // ── Locate fixtures ──────────────────────────────────────────────────────
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures_dir = manifest_dir.join("fixtures");
    // Workspace root is two levels up from kremory-eval manifest.
    let workspace_root = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .ok_or_else(|| anyhow::anyhow!("cannot determine workspace root from manifest dir"))?
        .to_path_buf();

    let fixture_path = fixtures_dir.join("mis_typed_high_conf.txt");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .with_context(|| format!("read fixture: {}", fixture_path.display()))?;
    let gt = load_ground_truth(&fixtures_dir)?;

    eprintln!(
        "[sweep] fixture: {} chars, {} ground-truth entities",
        fixture_text.len(),
        gt.len()
    );

    // ── Build providers ──────────────────────────────────────────────────────
    // Ingest provider: always Ollama (qwen2.5:14b handles the 21-type integer enum reliably).
    // Verify provider: Ollama or Anthropic depending on KREMORY_VERIFY_PROVIDER.
    let ingest_llm = build_llm(&ollama_host, &ingest_model)?;
    let embedder = build_embedder(&ollama_host, &embed_model)?;

    // Build verify LLM and obtain a `&dyn ChatProvider` reference.
    // The owned provider must outlive the sweep + gate calls, so we hold both
    // concrete arcs and dispatch via a trait-object ref (no boxing needed).
    let verify_ollama_opt: Option<Arc<Ollama>> =
        if verify_provider_name.to_lowercase() == "anthropic" {
            None
        } else {
            Some(build_llm(&ollama_host, &verify_model)?)
        };
    let verify_anthropic_opt: Option<Arc<Anthropic>> =
        if verify_provider_name.to_lowercase() == "anthropic" {
            Some(build_anthropic_llm(&verify_model)?)
        } else {
            None
        };
    let verify_llm_ref: &dyn ChatProvider = if let Some(ref a) = verify_anthropic_opt {
        a.as_ref()
    } else if let Some(ref o) = verify_ollama_opt {
        o.as_ref()
    } else {
        return Err(anyhow::anyhow!(
            "no verify LLM provider built — this is a bug"
        ));
    };

    // ── Phase 1 ingest (base DB) ─────────────────────────────────────────────
    // Uses ingest_llm (qwen2.5:14b by default) which reliably handles the
    // 21-type integer enum produced by DEFAULT_ENTITY_TYPES + EXTRA_TYPES.
    eprintln!("[sweep] Ingesting fixture into base DB (model={ingest_model})...");
    let base_dir = tempfile::tempdir().context("base tempdir")?;
    let base_db_path = base_dir.path().join("base.db");

    let base_graph = ingest_fixture(
        &fixture_text,
        &base_db_path,
        Arc::clone(&ingest_llm),
        Arc::clone(&embedder),
    )
    .await
    .context("base ingest")?;

    let pre_entities_base = query_entities(&base_graph.conn)
        .await
        .context("base pre-precision query")?;
    let pre_precision_base = label_precision(&pre_entities_base, &gt);

    eprintln!(
        "[sweep] Base ingest done. pre_precision={:.4} ({}/{}) elapsed={:.1}s",
        pre_precision_base,
        (pre_precision_base * gt.len() as f64) as usize,
        gt.len(),
        start.elapsed().as_secs_f64(),
    );

    // Flush base graph so the file is fully written before we copy it.
    base_graph
        .flush_if_dirty()
        .await
        .context("flush base graph")?;

    // Drop base_graph so the libsql connection is closed before we copy the file.
    drop(base_graph);

    // ── τ sweep ──────────────────────────────────────────────────────────────
    let taus: &[f32] = &[0.3, 0.5, 0.6, 0.7, 0.8];
    let mut sweep_results: Vec<TauResult> = Vec::new();

    for &tau in taus {
        eprintln!("[sweep] Running τ={tau:.2}...");
        let result = run_sweep_for_tau(
            &base_db_path,
            tau,
            &gt,
            verify_llm_ref,
            Arc::clone(&embedder),
        )
        .await
        .with_context(|| format!("sweep τ={tau:.2}"))?;

        eprintln!(
            "[sweep] τ={:.2}: pre={:.4} post={:.4} lift={:.4} F1={:.4} corrected={}",
            result.tau,
            result.pre_precision,
            result.post_precision,
            result.precision_lift,
            result.f1,
            result.corrected,
        );
        sweep_results.push(result);
    }

    // ── Pick best τ ──────────────────────────────────────────────────────────
    let best_idx = sweep_results
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.f1.partial_cmp(&b.f1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.cc_precision
                        .partial_cmp(&b.cc_precision)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_tau = sweep_results[best_idx].tau;
    eprintln!(
        "[sweep] Best τ={:.2} (F1={:.4} cc_precision={:.4})",
        best_tau, sweep_results[best_idx].f1, sweep_results[best_idx].cc_precision,
    );

    // ── Write sweep doc ──────────────────────────────────────────────────────
    write_sweep_doc(
        &workspace_root,
        &sweep_results,
        best_tau,
        &verify_provider_label,
    )?;

    // ── RISK-001 gate (D5, TD-038 multi-run mean ± SD hardening) ─────────────
    eprintln!("[sweep] === RISK-001 Acceptance Gate (n_runs={n_runs}) ===");
    let verdict = run_risk001_gate(
        &fixture_text,
        best_tau,
        &gt,
        Arc::clone(&ingest_llm),
        verify_llm_ref,
        Arc::clone(&embedder),
        n_runs,
    )
    .await
    .context("RISK-001 gate")?;

    write_risk001_doc(&workspace_root, &verdict, &verify_provider_label)?;

    // ── O11y Sprint O0.2: dump all captured metrics to JSON ─────────────────
    let metrics_entries = snapshot_to_json(&snapshotter);
    let safe_model = verify_provider_label.replace(['/', ':'], "__");
    let metrics_path = workspace_root
        .join(".ai-docs")
        .join("lessons")
        .join(format!(
            "2026-06-10-v0-1-2-consistency-check-sweep-metrics-{safe_model}.json"
        ));
    std::fs::write(
        &metrics_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "binary": "consistency_check_sweep",
            "verify_provider": verify_provider_label,
            "wall_clock_s": start.elapsed().as_secs_f64(),
            "metrics": metrics_entries,
            "metric_count": metrics_entries.len(),
        }))?,
    )
    .with_context(|| format!("write metrics: {}", metrics_path.display()))?;
    eprintln!(
        "[sweep] Metrics dumped: {} ({} entries)",
        metrics_path.display(),
        metrics_entries.len()
    );

    eprintln!(
        "[sweep] Total elapsed: {:.1}s",
        start.elapsed().as_secs_f64()
    );

    let mean_minus_sd = verdict.lift_mean - verdict.lift_sd;

    if !verdict.pass {
        eprintln!(
            "\n[sweep] RISK-001 FAIL — mean-sd={:.4} < {:.2} required (mean={:.4} sd={:.4} over {} runs).",
            mean_minus_sd, RISK001_LIFT_THRESHOLD, verdict.lift_mean, verdict.lift_sd, verdict.n_runs
        );
        eprintln!(
            "[sweep] ADR-047 sub-decision revision REQUIRED before Phase E per Stop Condition #1."
        );
        eprintln!("[sweep] Halting with exit code 1.");
        std::process::exit(1);
    }

    eprintln!(
        "\n[sweep] RISK-001 PASS — mean-sd={:.4} ≥ {:.2} (mean={:.4} sd={:.4} over {} runs). Phase E may proceed.",
        mean_minus_sd, RISK001_LIFT_THRESHOLD, verdict.lift_mean, verdict.lift_sd, verdict.n_runs
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (TD-038: multi-run mean ± SD gate hardening)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    // ─── mean_and_sample_sd ────────────────────────────────────────────────

    #[test]
    fn mean_and_sd_known_input_known_output() {
        // deviations from mean=12.0: -2, 0, 2 -> squares 4,0,4 -> sum 8 / (n-1=2) = 4 -> sqrt=2.0
        let (mean, sd) = mean_and_sample_sd(&[10.0, 12.0, 14.0]);
        assert!((mean - 12.0).abs() < 1e-9, "mean={mean}");
        assert!((sd - 2.0).abs() < 1e-9, "sd={sd}");
    }

    #[test]
    fn mean_and_sd_single_value_has_zero_sd() {
        let (mean, sd) = mean_and_sample_sd(&[0.07]);
        assert!((mean - 0.07).abs() < 1e-9, "mean={mean}");
        assert_eq!(sd, 0.0, "a single sample has no dispersion estimate");
    }

    #[test]
    fn mean_and_sd_empty_slice_is_zero_not_panic() {
        let (mean, sd) = mean_and_sample_sd(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(sd, 0.0);
    }

    #[test]
    fn mean_and_sd_identical_values_have_zero_sd() {
        let (mean, sd) = mean_and_sample_sd(&[0.08, 0.08, 0.08, 0.08, 0.08]);
        assert!((mean - 0.08).abs() < 1e-9);
        assert_eq!(sd, 0.0);
    }

    // ─── risk001_gate_pass boundary conditions ────────────────────────────
    //
    // These are the TD-038 acceptance-gate boundary tests requested by the
    // register entry. Both were run against the PRE-FIX gate (a single-shot
    // / SD-less comparison — `mean_lift >= threshold`, exactly what this repo
    // shipped before TD-038) and OBSERVED RED before `risk001_gate_pass` was
    // wired to subtract `lift_sd`:
    //
    // - `gate_sd_zero_exact_threshold_passes`: passed under both the old and
    //   new logic (SD=0 collapses `mean - sd` to `mean`), so it does not by
    //   itself distinguish the fix — kept as the boundary-equality case.
    // - `gate_large_sd_flips_a_mean_only_pass_to_fail`: this is the
    //   discriminating case. Under the pre-fix `mean_lift >= threshold` logic
    //   it asserted PASS (0.06 >= 0.05) and was GREEN. Once inverted to
    //   assert the CORRECT TD-038 behaviour (mean - sd = 0.06 - 0.02 = 0.04 <
    //   0.05 -> FAIL), it went RED against the pre-fix gate — confirming the
    //   test is sensitive to the exact defect TD-038 closes — before
    //   `risk001_gate_pass` was implemented with the `mean - sd` subtraction.

    #[test]
    fn gate_sd_zero_exact_threshold_passes() {
        // mean - sd = 0.05 - 0.0 = 0.05 >= 0.05 threshold -> PASS (boundary equality).
        assert!(risk001_gate_pass(0.05, 0.0, RISK001_LIFT_THRESHOLD));
    }

    #[test]
    fn gate_sd_zero_just_under_threshold_fails() {
        assert!(!risk001_gate_pass(0.0499, 0.0, RISK001_LIFT_THRESHOLD));
    }

    #[test]
    fn gate_large_sd_flips_a_mean_only_pass_to_fail() {
        // mean=0.06 alone clears the 0.05 threshold (this is exactly what the
        // pre-TD-038 single-shot gate would have measured and PASSED on).
        // With sd=0.02, mean - sd = 0.04 < 0.05 -> the aggregated gate FAILS.
        let lift_mean = 0.06;
        let lift_sd = 0.02;
        assert!(
            lift_mean >= RISK001_LIFT_THRESHOLD,
            "sanity: mean alone should have cleared the old gate"
        );
        assert!(
            !risk001_gate_pass(lift_mean, lift_sd, RISK001_LIFT_THRESHOLD),
            "mean - sd = {:.4} must fail the {:.2} threshold even though mean alone passes",
            lift_mean - lift_sd,
            RISK001_LIFT_THRESHOLD
        );
    }

    // ─── parse_n_runs_from ─────────────────────────────────────────────────

    #[test]
    fn parse_n_runs_defaults_to_five_with_no_args_or_env() {
        let n = parse_n_runs_from(&[], None).unwrap();
        assert_eq!(n, RISK001_DEFAULT_N_RUNS);
        assert_eq!(n, 5);
    }

    #[test]
    fn parse_n_runs_cli_flag_space_form() {
        let n = parse_n_runs_from(&s(&["--n-runs", "3"]), None).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn parse_n_runs_cli_flag_equals_form() {
        let n = parse_n_runs_from(&s(&["--n-runs=7"]), None).unwrap();
        assert_eq!(n, 7);
    }

    #[test]
    fn parse_n_runs_env_var_used_when_no_cli_flag() {
        let n = parse_n_runs_from(&[], Some("9")).unwrap();
        assert_eq!(n, 9);
    }

    #[test]
    fn parse_n_runs_cli_flag_overrides_env() {
        let n = parse_n_runs_from(&s(&["--n-runs", "3"]), Some("9")).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn parse_n_runs_zero_is_rejected() {
        let err = parse_n_runs_from(&s(&["--n-runs", "0"]), None).unwrap_err();
        assert!(err.contains(">= 1"), "expected '>= 1' hint, got: {err}");
    }

    #[test]
    fn parse_n_runs_non_numeric_is_rejected() {
        let err = parse_n_runs_from(&s(&["--n-runs", "banana"]), None).unwrap_err();
        assert!(err.contains("invalid"), "expected 'invalid' hint, got: {err}");
    }

    #[test]
    fn parse_n_runs_missing_value_is_rejected() {
        let err = parse_n_runs_from(&s(&["--n-runs"]), None).unwrap_err();
        assert!(
            err.contains("requires a value"),
            "expected 'requires a value' hint, got: {err}"
        );
    }
}
