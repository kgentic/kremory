// ADR-063 spec `dream-adversarial-corpora-and-metrics-2026-07-02.md` §3/§3.1/
// §3.5/§4 — the production-grade metrics harness for Site #2 (proposal-time
// type-novelty gate, `anti_redundancy::check_proposal` + the discover_types
// LLM-verify-band adjudication). Loads the committed adversarial corpus
// (`tests/corpora/site2_type_novelty_adversarial.jsonl`), runs each row
// through the REAL gate decision flow (real gemma4:e4b chat + real
// nomic-embed-text description embeddings via record/replay), and computes
// precision/false-accept safety metrics. Structurally mirrors
// `dream_metrics_harness_site3.rs` (same VCR scaffold shape, same
// smoke-one-before-batch discipline) — adapted for Site #2's different
// decision flow: `check_proposal` -> (optionally) `adjudicate_type_novelty`
// -> `write_gate`, replicating `discover_types.rs`'s per-proposal loop
// exactly rather than calling a single top-level pass function (Site #2 has
// no single "run the whole pass" entry point analogous to
// `type_registry_collapse` / `acronym_nickname_recall` — the gate is inlined
// in `discover_types`'s proposal loop).
//
// ## The Site #2 decision flow (verified against discover_types.rs:359-487)
//
// 1. `check_proposal` (pure, no I/O) -> `GateOutcome::Pass | Redundant |
//    NeedsLlmVerify`.
// 2. `GateOutcome::Pass` -> ACCEPT (novel). `GateOutcome::Redundant` -> REJECT
//    (redundant) — no LLM call, no ambiguity.
// 3. `GateOutcome::NeedsLlmVerify{existing_name, desc_cosine}` -> call
//    `adjudicate_type_novelty` for an `IdentityVerdictItem`, then `write_gate`
//    with `deterministic_signal = DeterministicSignal::from_lexical(
//    names_share_lemma_or_exact(proposal.name, existing_name))`.
//    `WriteDecision::Merge` -> REJECT (redundant); `WriteDecision::Reject` ->
//    ACCEPT (novel); `WriteDecision::PotentialAlias` -> ACCEPT (types have no
//    potential-alias concept — accept-with-log per discover_types.rs:457-485).
//
// ## SAFETY metric (the reason this gate exists)
//
// A "false accept" is a row whose ground truth is `redundant` (the proposal
// IS the same concept as an existing type) but the gate's decision was
// ACCEPT — i.e. a genuine duplicate slipped into the type registry. This is
// the harness's hard, exact-zero safety gate on `clear_redundant` +
// `subtle_redundant` (never loosened). `band_edge_distinct` +
// `clear_novel` rows are ground-truth `novel`; a false REJECT there
// (over-rejection, EDC's over-generalization finding) is tracked as a
// precision/recall quality signal, NOT part of the hard safety gate — a
// false reject merely discards a valid new type (recoverable on a future
// cycle), whereas a false accept silently merges a genuinely distinct
// concept into the wrong slot (destructive-adjacent for downstream typing).
//
// smoke-one-before-batch (hard rule): `smoke_one_metrics_harness_site2` runs
// ONE representative row (`s2-001`, Firm/Company, `clear_redundant`) through
// the full scoring pipeline FIRST and asserts sane wiring, before the full
// corpus batch runs.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::dream::{
    adjudicate_type_novelty, check_proposal, names_share_lemma_or_exact,
    AdjudicateTypeNoveltyParams, CheckProposalParams, GateOutcome, DESC_COSINE_THRESHOLD,
};
use kremory::core::entity_types::EntityTypeSpec;
use kremory::core::provider::{DynEmbeddingProvider, RecordReplayChatProvider};
use kremory::core::{write_gate, DeterministicSignal, WriteDecision, WriteGateInputs};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

// ─── VCR scaffold (mirrors dream_metrics_harness_site3.rs) ───────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum VcrMode {
    Record,
    Replay,
}

fn resolve_vcr_mode() -> VcrMode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => VcrMode::Record,
        Ok("replay") | Err(_) => VcrMode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn chat_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!("dream_metrics_harness_site2_{name}.json"))
}

fn embedding_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!(
            "dream_metrics_harness_site2_{name}.embeddings.json"
        ))
}

/// Description-cosine is a SEMANTIC comparison (spec §4.1/§4.3) — Site #2,
/// like Site #3, REQUIRES a real embedder (cosine gates decide Pass vs
/// NeedsLlmVerify vs Redundant). Mirrors
/// `dream_metrics_harness_site3.rs::RecordReplayEmbedder` exactly.
struct RecordReplayEmbedder {
    inner: Option<Arc<dyn DynEmbeddingProvider>>,
    cache: std::sync::Mutex<HashMap<String, Vec<f32>>>,
    path: std::path::PathBuf,
}

impl RecordReplayEmbedder {
    fn record(inner: Arc<dyn DynEmbeddingProvider>, path: std::path::PathBuf) -> Self {
        Self {
            inner: Some(inner),
            cache: std::sync::Mutex::new(HashMap::new()),
            path,
        }
    }

    fn replay(path: std::path::PathBuf) -> Self {
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "embedding cassette must load ({}): {e} — re-record via KREMORY_VCR=record",
                path.display()
            )
        });
        let map: HashMap<String, Vec<f32>> =
            serde_json::from_str(&raw).expect("embedding cassette must be valid JSON");
        Self {
            inner: None,
            cache: std::sync::Mutex::new(map),
            path,
        }
    }

    fn flush(&self) {
        let map = self.cache.lock().expect("embedding cache lock");
        let json = serde_json::to_string_pretty(&*map).expect("serialize embedding cassette");
        std::fs::write(&self.path, json).expect("write embedding cassette");
    }
}

impl kremory::EmbeddingProvider for RecordReplayEmbedder {
    async fn embed<'a>(&'a self, text: &'a str) -> kremory::CoreResult<Vec<f32>> {
        if let Some(v) = self.cache.lock().expect("cache lock").get(text).cloned() {
            return Ok(v);
        }
        match &self.inner {
            Some(inner) => {
                let v = inner.embed_dyn(text).await?;
                self.cache
                    .lock()
                    .expect("cache lock")
                    .insert(text.to_string(), v.clone());
                Ok(v)
            }
            None => Err(kremory::CoreError::Embedding(format!(
                "embedding cassette MISS for {text:?} — re-record via KREMORY_VCR=record"
            ))),
        }
    }
}

/// Real nomic embedder bridge (record mode only) — mirrors
/// `dream_metrics_harness_site3.rs::OllamaEmbedderAdapter` exactly, INCLUDING
/// the `search_document:` task prefix (TD-097 root cause).
struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    async fn embed<'a>(&'a self, text: &'a str) -> kremory::CoreResult<Vec<f32>> {
        use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
        let prefixed = format!("search_document: {text}");
        let mut vecs = AlLmEmbeddingProvider::embed(&*self.0, vec![prefixed])
            .await
            .map_err(|e| kremory::CoreError::Embedding(e.to_string()))?;
        vecs.pop().ok_or_else(|| {
            kremory::CoreError::Embedding("OllamaEmbedderAdapter: empty embed vec".to_string())
        })
    }
}

/// Build the (chat provider, embedder) pair for one row's isolated call, in
/// the given VCR mode, cassette-tagged by the row id.
async fn build_providers(
    mode: VcrMode,
    cassette_tag: &str,
) -> (
    Arc<RecordReplayChatProvider>,
    Arc<RecordReplayEmbedder>,
    String,
) {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // gemma4:e4b + think:false — this project's benchmarked deferred-quality
    // dream model (F1 85.7, local-model-benchmark-2026-06-24 /
    // project_kremory_validated_model_findings_2026-06-24). Mirrors the Site
    // #3 harness's model choice exactly — same pass, same tier.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    let chat_cassette = chat_cassette_path(cassette_tag);
    let provider: Arc<RecordReplayChatProvider> = match mode {
        VcrMode::Record => {
            let real: Arc<Ollama> = LLMBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model(&chat_model)
                .think(false)
                .timeout_seconds(180)
                .keep_alive("1h")
                .build()
                .expect("Ollama LLM builder must succeed (KREMORY_VCR=record needs Ollama)");
            Arc::new(RecordReplayChatProvider::record(
                real,
                chat_cassette,
                chat_model.clone(),
            ))
        }
        VcrMode::Replay => Arc::new(
            RecordReplayChatProvider::replay(chat_cassette).unwrap_or_else(|e| {
                panic!(
                    "replay cassette must load for tag={cassette_tag}: {e} — \
                     record it via KREMORY_VCR=record"
                )
            }),
        ),
    };

    let emb_path = embedding_cassette_path(cassette_tag);
    let embedder: Arc<RecordReplayEmbedder> = match mode {
        VcrMode::Record => {
            use autoagents_llm::backends::ollama::Ollama as OllamaEmb;
            use autoagents_llm::embedding::EmbeddingBuilder;
            let raw_nomic: Arc<OllamaEmb> = EmbeddingBuilder::<OllamaEmb>::new()
                .base_url(&base_url)
                .model("nomic-embed-text")
                .build()
                .expect("nomic embedder must build (KREMORY_VCR=record needs Ollama)");
            let nomic: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_nomic));
            Arc::new(RecordReplayEmbedder::record(nomic, emb_path))
        }
        VcrMode::Replay => Arc::new(RecordReplayEmbedder::replay(emb_path)),
    };

    (provider, embedder, chat_model)
}

// ─── Corpus ───────────────────────────────────────────────────────────────────

/// Ground truth, resolved from the corpus row's `ground_truth` field
/// (`"novel"` | `"redundant"` — Site #2's corpus has no `"uncertain"` rows,
/// unlike Site #3/#5's borderline categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundTruth {
    Novel,
    Redundant,
}

/// A single named-type definition (proposal or existing) parsed from the
/// corpus row.
struct TypeDef {
    name: String,
    description: String,
}

struct Row {
    id: String,
    proposal: TypeDef,
    existing: TypeDef,
    category: String,
    ground_truth: GroundTruth,
}

fn load_corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("site2_type_novelty_adversarial.jsonl");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read corpus at {path:?}: {e}"));

    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("corpus line {line_no} failed to parse as JSON: {e}"));

        let id = raw["id"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no}: missing 'id'"))
            .to_string();

        let parse_typedef = |field: &str| -> TypeDef {
            let obj = &raw[field];
            let name = obj["name"]
                .as_str()
                .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing '{field}.name'"))
                .to_string();
            let description = obj["description"]
                .as_str()
                .unwrap_or_else(|| {
                    panic!("corpus line {line_no} ({id}): missing '{field}.description'")
                })
                .to_string();
            TypeDef { name, description }
        };

        let proposal = parse_typedef("proposal");
        let existing = parse_typedef("existing");

        let category = raw["category"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'category'"))
            .to_string();

        let ground_truth = match raw["ground_truth"].as_str() {
            Some("novel") => GroundTruth::Novel,
            Some("redundant") => GroundTruth::Redundant,
            other => panic!(
                "corpus line {line_no} (id={id}): unrecognized 'ground_truth' value {other:?}"
            ),
        };

        rows.push(Row {
            id,
            proposal,
            existing,
            category,
            ground_truth,
        });
    }
    rows
}

// ─── Per-row run ──────────────────────────────────────────────────────────────

/// The harness's own decision label for one row: "accept" (proposal treated
/// as novel — the type would be registered) or "reject" (proposal treated as
/// redundant — the type would NOT be registered), plus which gate stage
/// produced it (for diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateDecision {
    Accept,
    Reject,
}

#[derive(Debug, Clone)]
struct RowOutcome {
    row_id: String,
    category: String,
    ground_truth: GroundTruth,
    decision: GateDecision,
    /// Which stage produced the decision — "check_proposal:pass",
    /// "check_proposal:redundant", "write_gate:merge",
    /// "write_gate:reject", "write_gate:potential_alias".
    stage: String,
    needed_llm_verify: bool,
    row_wall_clock_ms: f64,
}

/// Run one corpus row through the REAL Site #2 decision flow, replicating
/// `discover_types.rs:359-487` exactly.
async fn run_one_row(row: &Row, mode: VcrMode) -> Result<RowOutcome, String> {
    let row_start = std::time::Instant::now();

    let (provider, emb_vcr, chat_model) = build_providers(mode, &row.id).await;
    if mode == VcrMode::Record {
        for text in [
            row.proposal.description.as_str(),
            row.existing.description.as_str(),
        ] {
            kremory::EmbeddingProvider::embed(&*emb_vcr, text)
                .await
                .map_err(|e| {
                    format!("prewarm embed failed for row {} text {text:?}: {e}", row.id)
                })?;
        }
    }

    let proposal_desc_emb = kremory::EmbeddingProvider::embed(&*emb_vcr, &row.proposal.description)
        .await
        .map_err(|e| format!("embed proposal desc failed for row {}: {e}", row.id))?;
    let existing_desc_emb = kremory::EmbeddingProvider::embed(&*emb_vcr, &row.existing.description)
        .await
        .map_err(|e| format!("embed existing desc failed for row {}: {e}", row.id))?;

    let existing_spec = EntityTypeSpec {
        id: 1,
        name: row.existing.name.clone(),
        description: row.existing.description.clone(),
    };
    let existing_type_embeddings = vec![(existing_spec, existing_desc_emb)];

    let gid = format!("site2-corpus-{}", row.id);

    let outcome = match check_proposal(CheckProposalParams {
        proposal_name: &row.proposal.name,
        proposal_desc_emb: &proposal_desc_emb,
        existing_type_embeddings: &existing_type_embeddings,
        namespace: &gid,
        model: &chat_model,
    }) {
        GateOutcome::Redundant { .. } => (GateDecision::Reject, "check_proposal:redundant", false),
        GateOutcome::Pass => (GateDecision::Accept, "check_proposal:pass", false),
        GateOutcome::NeedsLlmVerify {
            existing_name,
            desc_cosine,
        } => {
            let existing_desc = existing_type_embeddings
                .iter()
                .find(|(spec, _)| spec.name == existing_name)
                .map(|(spec, _)| spec.description.clone())
                .unwrap_or_default();
            let deterministic = names_share_lemma_or_exact(&row.proposal.name, &existing_name);

            let verdict = adjudicate_type_novelty(AdjudicateTypeNoveltyParams {
                llm: &*provider,
                model_id: &chat_model,
                proposal_name: &row.proposal.name,
                proposal_desc: &row.proposal.description,
                existing_name: &existing_name,
                existing_desc: &existing_desc,
                group_id: &gid,
            })
            .await;

            let decision = write_gate(WriteGateInputs {
                cosine: desc_cosine,
                merge_threshold: DESC_COSINE_THRESHOLD,
                deterministic_signal: DeterministicSignal::from_lexical(deterministic),
                llm_verdict: verdict,
                min_confidence_floor: None,
            });

            match decision {
                WriteDecision::Merge => (GateDecision::Reject, "write_gate:merge", true),
                WriteDecision::Reject => (GateDecision::Accept, "write_gate:reject", true),
                WriteDecision::PotentialAlias => {
                    (GateDecision::Accept, "write_gate:potential_alias", true)
                }
            }
        }
    };

    if mode == VcrMode::Record {
        provider
            .flush()
            .map_err(|e| format!("provider.flush() failed for row {}: {e}", row.id))?;
        emb_vcr.flush();
    }

    let row_wall_clock_ms = row_start.elapsed().as_secs_f64() * 1000.0;

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory::tests::dream_metrics_harness_site2",
            row_id = %row.id,
            proposal = %row.proposal.name,
            existing = %row.existing.name,
            category = %row.category,
            ground_truth = ?row.ground_truth,
            decision = ?outcome.0,
            stage = %outcome.1,
            "per-row scoring detail"
        );
    }

    Ok(RowOutcome {
        row_id: row.id.clone(),
        category: row.category.clone(),
        ground_truth: row.ground_truth,
        decision: outcome.0,
        stage: outcome.1.to_string(),
        needed_llm_verify: outcome.2,
        row_wall_clock_ms,
    })
}

// ─── Metrics computation ──────────────────────────────────────────────────────

/// Precision here is precision of "novel" (accept) decisions: of everything
/// the gate ACCEPTED, how many were genuinely novel? Recall: of everything
/// genuinely novel, how many did the gate accept?
///
/// SAFETY metric — `false_accepts`: rows whose ground truth is `redundant`
/// but the gate decision was ACCEPT (a genuine duplicate slipped through).
/// This is the hard, never-loosened gate.
#[derive(Debug, Clone, Default, serde::Serialize)]
struct CategoryMetrics {
    n: usize,
    tp: usize,
    fp: usize,
    fn_: usize,
    tn: usize,
    false_accepts: usize,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
}

fn compute_metrics(outcomes: &[&RowOutcome]) -> CategoryMetrics {
    let mut m = CategoryMetrics {
        n: outcomes.len(),
        ..Default::default()
    };

    for o in outcomes {
        let truly_novel = match o.ground_truth {
            GroundTruth::Novel => true,
            GroundTruth::Redundant => false,
        };
        let gate_says_novel = o.decision == GateDecision::Accept;

        match (gate_says_novel, truly_novel) {
            (true, true) => m.tp += 1,
            (true, false) => {
                m.fp += 1;
                m.false_accepts += 1;
            }
            (false, true) => m.fn_ += 1,
            (false, false) => m.tn += 1,
        }
    }

    let flagged_novel = m.tp + m.fp;
    let genuinely_novel_total = m.tp + m.fn_;

    m.precision = if flagged_novel == 0 {
        None
    } else {
        Some(m.tp as f64 / flagged_novel as f64)
    };
    m.recall = if genuinely_novel_total == 0 {
        None
    } else {
        Some(m.tp as f64 / genuinely_novel_total as f64)
    };
    m.f1 = match (m.precision, m.recall) {
        (Some(p), Some(r)) if (p + r) > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };

    m
}

// ─── o11y snapshot helpers (kept lightweight — Site #2 has no single top-level
// pass function/report to cross-check against, unlike Site #3/#5; latency is
// still worth capturing for the LLM-verify-band path) ────────────────────────

fn sum_counter(snapshotter: &Snapshotter, metric_name: &str) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != metric_name {
                return None;
            }
            if let DebugValue::Counter(n) = value {
                Some(n)
            } else {
                None
            }
        })
        .sum()
}

fn histogram_samples(snapshotter: &Snapshotter, metric_name: &str) -> Vec<f64> {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != metric_name {
                return None;
            }
            if let DebugValue::Histogram(samples) = value {
                Some(samples.into_iter().map(f64::from).collect::<Vec<f64>>())
            } else {
                None
            }
        })
        .flatten()
        .collect()
}

fn percentile(samples: &[f64], p: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("latency ms samples are never NaN"));
    let rank = ((p * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[rank])
}

#[derive(Debug, Clone, Default, serde::Serialize)]
struct LatencyMetrics {
    vcr_mode: String,
    llm_call_count: usize,
    llm_call_total_ms: f64,
    llm_call_ms_p50: Option<f64>,
    llm_call_ms_p95: Option<f64>,
    llm_call_ms_p99: Option<f64>,
    row_wall_clock_ms_p50: Option<f64>,
    row_wall_clock_ms_p95: Option<f64>,
    row_wall_clock_ms_p99: Option<f64>,
    total_harness_wall_clock_ms: f64,
    rows_needing_llm_verify: usize,
}

fn compute_latency_metrics(
    snapshotter: &Snapshotter,
    outcomes: &[RowOutcome],
    vcr_mode: VcrMode,
) -> LatencyMetrics {
    let llm_samples = histogram_samples(
        snapshotter,
        "kremory.identity.llm_call_latency_ms_histogram",
    );
    let row_wall_clock_ms: Vec<f64> = outcomes.iter().map(|o| o.row_wall_clock_ms).collect();

    LatencyMetrics {
        vcr_mode: match vcr_mode {
            VcrMode::Record => "record (real Ollama latency)".to_string(),
            VcrMode::Replay => "replay (cassette lookup — NOT real LLM latency)".to_string(),
        },
        llm_call_count: llm_samples.len(),
        llm_call_total_ms: llm_samples.iter().sum(),
        llm_call_ms_p50: percentile(&llm_samples, 0.50),
        llm_call_ms_p95: percentile(&llm_samples, 0.95),
        llm_call_ms_p99: percentile(&llm_samples, 0.99),
        row_wall_clock_ms_p50: percentile(&row_wall_clock_ms, 0.50),
        row_wall_clock_ms_p95: percentile(&row_wall_clock_ms, 0.95),
        row_wall_clock_ms_p99: percentile(&row_wall_clock_ms, 0.99),
        total_harness_wall_clock_ms: row_wall_clock_ms.iter().sum(),
        rows_needing_llm_verify: outcomes.iter().filter(|o| o.needed_llm_verify).count(),
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
struct VolumeMetrics {
    n_rows: usize,
    accepts: usize,
    rejects: usize,
    llm_verify_band_hits: usize,
}

fn compute_volume_metrics(outcomes: &[RowOutcome]) -> VolumeMetrics {
    VolumeMetrics {
        n_rows: outcomes.len(),
        accepts: outcomes
            .iter()
            .filter(|o| o.decision == GateDecision::Accept)
            .count(),
        rejects: outcomes
            .iter()
            .filter(|o| o.decision == GateDecision::Reject)
            .count(),
        llm_verify_band_hits: outcomes.iter().filter(|o| o.needed_llm_verify).count(),
    }
}

// ─── Report emission ─────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct MetricsReport {
    site: String,
    overall: CategoryMetrics,
    per_category: std::collections::BTreeMap<String, CategoryMetrics>,
    n_flagged_novel: usize,
    safety_false_accepts: usize,
    false_accept_rows: Vec<String>,
    hard_gate: String,
    latency: LatencyMetrics,
    volume: VolumeMetrics,
}

fn print_and_write_report(report: &MetricsReport) {
    eprintln!("\n── Site #2 metrics harness — full report ──────────────────────────────");
    eprintln!(
        "  OVERALL: n={} precision={:?} recall={:?} f1={:?}",
        report.overall.n, report.overall.precision, report.overall.recall, report.overall.f1,
    );
    for (cat, m) in &report.per_category {
        eprintln!(
            "  [{cat}] n={} tp={} fp={} fn={} tn={} false_accepts={} precision={:?} recall={:?} f1={:?}",
            m.n, m.tp, m.fp, m.fn_, m.tn, m.false_accepts, m.precision, m.recall, m.f1,
        );
    }
    eprintln!(
        "\n  n_flagged_novel={} safety_false_accepts={} hard_gate={}",
        report.n_flagged_novel, report.safety_false_accepts, report.hard_gate,
    );
    if !report.false_accept_rows.is_empty() {
        eprintln!("  FALSE ACCEPT ROWS (safety failure — genuine duplicate slipped through):");
        for r in &report.false_accept_rows {
            eprintln!("    *** {r} ***");
        }
    }

    eprintln!(
        "\n  LATENCY [{}]: llm_call_count={} llm_call_total_ms={:.1} \
         llm_call_ms p50={:?} p95={:?} p99={:?} | row_wall_clock_ms p50={:?} p95={:?} p99={:?} \
         | total_harness_wall_clock_ms={:.1} | rows_needing_llm_verify={}",
        report.latency.vcr_mode,
        report.latency.llm_call_count,
        report.latency.llm_call_total_ms,
        report.latency.llm_call_ms_p50,
        report.latency.llm_call_ms_p95,
        report.latency.llm_call_ms_p99,
        report.latency.row_wall_clock_ms_p50,
        report.latency.row_wall_clock_ms_p95,
        report.latency.row_wall_clock_ms_p99,
        report.latency.total_harness_wall_clock_ms,
        report.latency.rows_needing_llm_verify,
    );
    eprintln!(
        "  VOLUME: n_rows={} accepts={} rejects={} llm_verify_band_hits={}",
        report.volume.n_rows,
        report.volume.accepts,
        report.volume.rejects,
        report.volume.llm_verify_band_hits,
    );

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("site2_metrics.json");
    let json = serde_json::to_string_pretty(report).expect("serialize metrics report");
    std::fs::write(&out_path, json)
        .unwrap_or_else(|e| panic!("failed to write metrics report to {out_path:?}: {e}"));
    eprintln!("  metrics JSON written to {out_path:?}");
}

// ─── smoke-one-before-batch ───────────────────────────────────────────────────

/// smoke-one-before-batch (hard rule): run ONLY corpus row `s2-001`
/// (Firm/Company, `clear_redundant`) through the full scoring pipeline first,
/// confirm the plant->run->score wiring is sane, THEN proceed to the full
/// corpus run.
#[tokio::test]
#[ignore = "dream_metrics_harness_site2: requires Ollama in record mode, or a committed cassette \
            in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test dream_metrics_harness_site2 -- --ignored \
            --nocapture smoke_one_metrics_harness_site2"]
async fn smoke_one_metrics_harness_site2() {
    let mode = resolve_vcr_mode();
    let corpus = load_corpus();
    let smoke_row = corpus
        .iter()
        .find(|r| r.id == "s2-001")
        .expect("corpus must contain s2-001 (Firm/Company)");
    assert_eq!(smoke_row.proposal.name, "Firm");
    assert_eq!(smoke_row.category, "clear_redundant");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let outcome = run_one_row(smoke_row, mode)
        .await
        .unwrap_or_else(|e| panic!("smoke-one row failed: {e}"));

    eprintln!(
        "[smoke-one] row_id={} proposal={} existing={} category={} ground_truth={:?} \
         decision={:?} stage={} needed_llm_verify={}",
        outcome.row_id,
        smoke_row.proposal.name,
        smoke_row.existing.name,
        outcome.category,
        outcome.ground_truth,
        outcome.decision,
        outcome.stage,
        outcome.needed_llm_verify,
    );

    assert_eq!(
        outcome.decision,
        GateDecision::Reject,
        "smoke-one: Firm/Company (clear redundant, near-identical description) should be \
         rejected as redundant — got decision={:?} stage={}",
        outcome.decision,
        outcome.stage,
    );

    let _ = snapshotter; // reserved for future o11y cross-check parity with Site #3/#5

    eprintln!("[smoke-one] PASS — proceeding to the full corpus run is safe.");
}

// ─── Full corpus run — the binding metrics harness ───────────────────────────

#[tokio::test]
#[ignore = "dream_metrics_harness_site2: requires Ollama in record mode, or a committed cassette \
            in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test dream_metrics_harness_site2 -- --ignored \
            --nocapture full_corpus_site2_metrics"]
async fn full_corpus_site2_metrics() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kremory=debug")),
        )
        .with_test_writer()
        .try_init();

    let mode = resolve_vcr_mode();
    let corpus = load_corpus();
    assert_eq!(corpus.len(), 26, "corpus must have exactly 26 rows");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut outcomes: Vec<RowOutcome> = Vec::with_capacity(corpus.len());
    for row in &corpus {
        let outcome = run_one_row(row, mode)
            .await
            .unwrap_or_else(|e| panic!("row {} failed: {e}", row.id));
        outcomes.push(outcome);
    }

    // ── Per-category metrics ──────────────────────────────────────────────
    let mut categories: std::collections::BTreeMap<String, Vec<&RowOutcome>> =
        std::collections::BTreeMap::new();
    for o in &outcomes {
        categories.entry(o.category.clone()).or_default().push(o);
    }
    let per_category: std::collections::BTreeMap<String, CategoryMetrics> = categories
        .iter()
        .map(|(cat, os)| (cat.clone(), compute_metrics(os)))
        .collect();

    // ── Overall metrics ────────────────────────────────────────────────────
    let all_refs: Vec<&RowOutcome> = outcomes.iter().collect();
    let overall = compute_metrics(&all_refs);

    // ── SAFETY gate (exact-zero): clear_redundant + subtle_redundant rows
    // must NEVER be accepted as novel — that would silently register a
    // genuine duplicate type. ─────────────────────────────────────────────
    let safety_categories = ["clear_redundant", "subtle_redundant"];
    let mut safety_false_accepts = 0usize;
    for cat in safety_categories {
        if let Some(m) = per_category.get(cat) {
            safety_false_accepts += m.false_accepts;
        }
    }
    let false_accept_rows: Vec<String> = outcomes
        .iter()
        .filter(|o| o.ground_truth == GroundTruth::Redundant && o.decision == GateDecision::Accept)
        .map(|o| {
            let row = corpus
                .iter()
                .find(|r| r.id == o.row_id)
                .expect("row lookup");
            format!(
                "{} ({} / {}, category={}, stage={})",
                o.row_id, row.proposal.name, row.existing.name, o.category, o.stage
            )
        })
        .collect();

    let n_flagged_novel = overall.tp + overall.fp;

    // ── HARD GATE: safety_false_accepts == 0 is the binding assertion
    // (never loosened). Precision is reported for quality visibility but NOT
    // hard-failed in this smoke, per the task brief. ──────────────────────
    let hard_gate = if safety_false_accepts == 0 {
        "PASS"
    } else {
        "FAIL"
    }
    .to_string();

    let latency = compute_latency_metrics(&snapshotter, &outcomes, mode);
    let volume = compute_volume_metrics(&outcomes);

    let report = MetricsReport {
        site: "site2_type_novelty".to_string(),
        overall: overall.clone(),
        per_category,
        n_flagged_novel,
        safety_false_accepts,
        false_accept_rows: false_accept_rows.clone(),
        hard_gate: hard_gate.clone(),
        latency,
        volume,
    };
    print_and_write_report(&report);

    // ── Per-row rationale dump for any FP/FN, so a failing gate is
    // immediately diagnosable without re-running with KREMORY_DEBUG=1 ──────
    for row in &corpus {
        let outcome = outcomes.iter().find(|o| o.row_id == row.id).unwrap();
        let truly_novel = outcome.ground_truth == GroundTruth::Novel;
        let gate_says_novel = outcome.decision == GateDecision::Accept;
        if truly_novel != gate_says_novel {
            eprintln!(
                "  *** MISCLASSIFIED [{}] {} vs {} — truly_novel={} decision={:?} stage={} ***",
                outcome.category,
                row.proposal.name,
                row.existing.name,
                truly_novel,
                outcome.decision,
                outcome.stage,
            );
        }
    }

    eprintln!(
        "\n  sum counter kremory.identity.verdict_parse_fail_total={}",
        sum_counter(&snapshotter, "kremory.identity.verdict_parse_fail_total")
    );

    // ── Hard assertion — the SAFETY gate, exact-zero, never loosened ──────
    assert_eq!(
        safety_false_accepts, 0,
        "SAFETY GATE FAILED: {safety_false_accepts} false accept(s) among clear_redundant / \
         subtle_redundant rows (ground truth = genuinely REDUNDANT concepts, but the gate \
         ACCEPTED them as novel). Exact-zero gate, never loosened. See false_accept_rows above \
         for exactly which corpus row(s) caused it."
    );

    eprintln!(
        "\n══ GATE: {hard_gate} (safety_false_accepts={safety_false_accepts}, \
         precision={:?}) ═══════════════",
        overall.precision,
    );
}
