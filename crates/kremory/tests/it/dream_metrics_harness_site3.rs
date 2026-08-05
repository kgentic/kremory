// ADR-063 spec `dream-adversarial-corpora-and-metrics-2026-07-02.md` §3/§3.1/
// §3.5/§4 — the production-grade metrics harness for Site #3 (type-registry
// post-hoc collapse). Loads the committed adversarial corpus
// (`tests/corpora/site3_type_collapse_adversarial.jsonl`, 119 rows), runs each
// row through the REAL pass (real gemma4:e4b chat + real nomic-embed-text
// description embeddings via record/replay, mirrors
// `type_registry_collapse_s3_spike.rs` exactly), and computes precision/
// recall/F1 + Wilson 95% CI per §3.1's denominator definition, cross-checked
// against the pass's own o11y counters (§3.5). Structurally mirrors
// `dream_metrics_harness.rs` (Site #5) — same VCR scaffold shape, same
// CategoryMetrics/LatencyMetrics/VolumeMetrics report shapes, same
// smoke-one-before-batch discipline — adapted for Site #3's different plant
// shape (entity_types, not entities) and different report shape
// (`TypeRegistryCollapseReport` has no per-decision breakdown fields, unlike
// Site #5's `AcronymNicknameRecallReport` — see `run_one_row`'s decision-
// derivation doc comment for how this harness recovers the per-pair decision).
//
// KEY DESIGN CONSTRAINT (mirrors Site #5's isolation rationale exactly):
// `type_registry_collapse`'s pairwise loop is O(n^2) over every non-catch-all
// `entity_types` row in a `group_id` (`type_registry_collapse.rs`'s nested
// `for i .. for j` loop). Planting all 119 corpus rows (238 types) into ONE
// group would produce thousands of pairs and risk spurious cross-row
// nominations corrupting the measurement. Each row is therefore ISOLATED into
// its own `group_id` (a fresh 2-type registry: type `a` (desc_a), type `b`
// (desc_b), no co-occurrence concept for types), and `type_registry_collapse`
// is run once per row on that group alone — up to 119 separate adjudication
// calls (many rows resolve via cosine alone, no LLM call needed), but no
// manual cross-row collision audit required.
//
// smoke-one-before-batch (hard rule): `smoke_one_metrics_harness_site3` runs
// ONE representative row (`s3-001`, Company/company, `trivial_duplicate`)
// through the full scoring pipeline FIRST and asserts sane wiring, before the
// 119-row batch runs.
//
// Denominator mapping (spec §3.1, mirrors Site #5's convention exactly):
//   - Precision denominator = pairs the pass's write_gate calls "same"
//     (Merge ∪ PotentialAlias). Precision = (of those, how many are
//     genuinely the same concept per `same_concept`).
//   - Recall denominator = genuinely-same-concept pairs (`same_concept=true`)
//     that were NOMINATED-OR-AUTO-MERGED at all (i.e. reached a decision).
//     Recall = (of those, how many did the pass call "same", i.e. NOT Reject
//     and NOT silently unexamined).
//   - Categories (verified 2026-07-03): trivial_duplicate, hard_true_collapse,
//     semantic_near_dup_zero_lexical, unicode_casing_variant [same_concept=
//     true]; distinct_lemma_collision, distinct_unrelated, band_edge_moderate
//     [same_concept=false]; every one of these feeds the hard precision/
//     recall/safety gate directly.
//   - One EXCEPTION (added 2026-07-03, product-owner reclassification, see
//     `site3_corpus_v2_changelog.md` v2.1): `borderline_type_identity`
//     (`same_concept="uncertain"`, row `s3-088` Suspenders/Suspender) is a
//     genuinely-ambiguous entity-TYPE-identity call — mirrors Site #5's
//     `uncertain_thin_context` convention exactly. `GroundTruth::Uncertain`
//     is EXCLUDED from precision/recall AND from the zero-false-merge safety
//     gate, reported descriptively only. The literal zero-false-merge gate
//     is UNCHANGED for every clear-cut distinct category.

#![cfg(all(feature = "test-utils", feature = "llm-smoke"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use kremory::core::dream::{
    type_registry_collapse, wilson_lower_upper, TypeRegistryCollapseParams,
};
use kremory::core::provider::{DynEmbeddingProvider, RecordReplayChatProvider};
use kremory::core::schema::TemporalGraph;

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

// ─── VCR scaffold (mirrors dream_metrics_harness.rs + type_registry_collapse_s3_spike.rs) ──

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
        .join(format!("dream_metrics_harness_site3_{name}.json"))
}

fn embedding_cassette_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(format!(
            "dream_metrics_harness_site3_{name}.embeddings.json"
        ))
}

/// Description-cosine is a SEMANTIC comparison (spec §4.1/§4.3) — Site #3,
/// unlike Site #5, REQUIRES a real embedder (cosine gates decide auto-merge
/// vs LLM-verify-band vs no-candidate). Mirrors
/// `type_registry_collapse_s3_spike.rs::RecordReplayEmbedder` exactly.
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
/// `type_registry_collapse_s3_spike.rs::OllamaEmbedderAdapter` exactly,
/// INCLUDING the `search_document:` task prefix (TD-097 root cause).
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
    // project_kremory_validated_model_findings_2026-06-24). Mirrors the S3
    // spike's model choice exactly — same pass, same tier.
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

// ─── Fixture helpers (mirrors type_registry_collapse_s3_spike.rs) ────────────

/// Insert a custom `entity_types` row at a fresh id above the seeded default
/// vocabulary (mirrors `type_registry_collapse_s3_spike.rs::insert_custom_type`).
#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
async fn insert_custom_type(graph: &TemporalGraph, group_id: &str, name: &str, desc: &str) {
    let mut rows = graph
        .conn
        .query(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM entity_types WHERE group_id = ?1",
            libsql::params![group_id.to_string()],
        )
        .await
        .expect("query next entity_type id");
    let row = rows.next().await.expect("row read").expect("row present");
    let next_id: i64 = row.get(0).expect("next id at index 0");
    let new_id = next_id.max(100);
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types (id, group_id, name, description) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![
                new_id,
                group_id.to_string(),
                name.to_string(),
                desc.to_string()
            ],
        )
        .await
        .expect("insert custom entity_type");
}

async fn type_exists(conn: &libsql::Connection, group_id: &str, name: &str) -> bool {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1 AND name = ?2",
            libsql::params![group_id, name],
        )
        .await
        .expect("query type exists");
    let n: i64 = rows
        .next()
        .await
        .expect("row")
        .expect("row present")
        .get(0)
        .expect("count col");
    n > 0
}

/// Read back the `identity_verdict_audit.decision` for a specific
/// `(a, b)` pair (order-insensitive), if any row was written. Types have no
/// nomination-vs-not distinction outside this table + the report's terminal
/// counts (see `run_one_row`'s decision-derivation doc comment).
#[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption (mirrors type_registry_collapse_s3_spike.rs::insert_custom_type)
async fn read_audit_decision(
    conn: &libsql::Connection,
    group_id: &str,
    a: &str,
    b: &str,
) -> Option<String> {
    let mut rows = conn
        .query(
            "SELECT decision FROM identity_verdict_audit WHERE group_id = ?1 AND \
             ((candidate_a = ?2 AND candidate_b = ?3) OR (candidate_a = ?3 AND candidate_b = ?2)) \
             ORDER BY id DESC LIMIT 1",
            libsql::params![group_id.to_string(), a.to_string(), b.to_string()],
        )
        .await
        .expect("audit query");
    let row = rows.next().await.expect("row read")?;
    let decision: String = row.get(0).expect("decision col");
    Some(decision)
}

// ─── Ground truth ─────────────────────────────────────────────────────────────

/// Ground truth, resolved from the corpus row's `same_concept` field
/// (`true` | `false` | `"uncertain"`).
///
/// `Same` / `Different` feed the hard precision/recall/safety gate. `Uncertain`
/// (the `borderline_type_identity` category, `same_concept="uncertain"` —
/// added 2026-07-03 for row `s3-088` Suspenders/Suspender, a product-owner
/// judgment call that entity-TYPE identity is genuinely ambiguous here) is
/// EXCLUDED from the hard gate entirely — ground truth itself is ambiguous by
/// design — and reported descriptively only. Mirrors Site #5's
/// `uncertain_thin_context` / `same_entity="uncertain"` convention exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundTruth {
    Same,
    Different,
    Uncertain,
}

/// Parsed corpus row.
struct Row {
    id: String,
    a: String,
    b: String,
    desc_a: String,
    desc_b: String,
    category: String,
    ground_truth: GroundTruth,
}

/// Load + classify the committed corpus.
fn load_corpus() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("site3_type_collapse_adversarial.jsonl");
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
        let a = raw["a"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'a'"))
            .to_string();
        let b = raw["b"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'b'"))
            .to_string();
        let desc_a = raw["desc_a"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'desc_a'"))
            .to_string();
        let desc_b = raw["desc_b"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'desc_b'"))
            .to_string();
        let category = raw["category"]
            .as_str()
            .unwrap_or_else(|| panic!("corpus line {line_no} ({id}): missing 'category'"))
            .to_string();

        let same_concept_raw = &raw["same_concept"];
        let ground_truth = if same_concept_raw.as_str() == Some("uncertain") {
            GroundTruth::Uncertain
        } else if same_concept_raw.as_bool() == Some(true) {
            GroundTruth::Same
        } else if same_concept_raw.as_bool() == Some(false) {
            GroundTruth::Different
        } else {
            panic!(
                "corpus line {line_no} (id={id}): unrecognized 'same_concept' value \
                 {same_concept_raw:?}"
            );
        };

        rows.push(Row {
            id,
            a,
            b,
            desc_a,
            desc_b,
            category,
            ground_truth,
        });
    }
    rows
}

// ─── Per-row isolated plant + run ────────────────────────────────────────────

/// Outcome of running one corpus row through the real pass in its own
/// isolated 2-type group.
#[derive(Debug, Clone)]
struct RowOutcome {
    row_id: String,
    category: String,
    ground_truth: GroundTruth,
    /// The pass's actual per-pair decision ("merge" | "potential_alias" |
    /// "reject"), or `None` if the pair was never examined at all (should
    /// never happen — every isolated 2-type group always yields exactly 1
    /// pair per `report.pairs_examined`; kept as `Option` defensively).
    decision: Option<String>,
    candidates_nominated: usize,
    pairs_examined: usize,
    row_wall_clock_ms: f64,
}

/// Plant one row's two entity_types into a fresh isolated `group_id`, run
/// `type_registry_collapse` on that group alone, and derive the pass's
/// per-pair decision.
///
/// Decision derivation (Site #3's report shape differs from Site #5's — no
/// per-decision breakdown fields on `TypeRegistryCollapseReport`, only
/// terminal aggregate counts):
///   - `report.merges_applied > 0` → "merge" (auto-merge row 1 OR an
///     LLM-authorized row-5/row-6 merge — either way the type disappeared).
///   - else, read `identity_verdict_audit.decision` for this exact (a, b)
///     pair: "potential_alias" if a row was written (LLM-verify band,
///     PotentialAlias outcome — spec §5.2 audits only LLM-touched
///     decisions); if NO row exists, the pair was either (a) nominated then
///     cleanly Rejected (no audit row per spec §2.2/§5.2), or (b) never
///     nominated at all (cosine < 0.70, spec §4.3 row 4 — no candidate, no
///     LLM call, no audit row either). Distinguish (a) from (b) via
///     `report.candidates_nominated`: >0 means this isolated single-pair
///     group's one pair WAS nominated, so no-audit-row + no-merge here
///     unambiguously means "reject"; ==0 means the cosine gate never
///     nominated it at all, i.e. "not examined by the LLM-verify band" —
///     reported as `None` (structurally equivalent to Site #5's
///     NotNominated, but every category in this corpus is expected to reach
///     SOME decision per spec's PASS bar, so a `None` here on a Same/Different
///     scored row is itself diagnostic signal, not an expected outcome).
async fn run_one_row(
    row: &Row,
    mode: VcrMode,
) -> Result<(RowOutcome, TemporalGraph, String), String> {
    let row_start = std::time::Instant::now();
    let graph = TemporalGraph::open_in_memory()
        .await
        .map_err(|e| format!("open_in_memory failed for row {}: {e}", row.id))?;
    let gid = format!("site3-corpus-{}", row.id);

    insert_custom_type(&graph, &gid, &row.a, &row.desc_a).await;
    insert_custom_type(&graph, &gid, &row.b, &row.desc_b).await;

    let (provider, emb_vcr, chat_model) = build_providers(mode, &row.id).await;
    if mode == VcrMode::Record {
        for text in [row.desc_a.as_str(), row.desc_b.as_str()] {
            kremory::EmbeddingProvider::embed(&*emb_vcr, text)
                .await
                .map_err(|e| {
                    format!("prewarm embed failed for row {} text {text:?}: {e}", row.id)
                })?;
        }
    }
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();

    let report = type_registry_collapse(
        &*provider,
        TypeRegistryCollapseParams {
            conn: &graph.conn,
            group_id: &gid,
            embedder: Some(embedder.as_ref()),
            model_id: &chat_model,
        },
    )
    .await
    .map_err(|e| format!("type_registry_collapse failed for row {}: {e}", row.id))?;

    if mode == VcrMode::Record {
        provider
            .flush()
            .map_err(|e| format!("provider.flush() failed for row {}: {e}", row.id))?;
        emb_vcr.flush();
    }

    let decision: Option<String> = if report.merges_applied > 0 {
        Some("merge".to_string())
    } else if let Some(audited) = read_audit_decision(&graph.conn, &gid, &row.a, &row.b).await {
        Some(audited) // "potential_alias" (the only non-merge decision that writes a row)
    } else if report.candidates_nominated > 0 {
        Some("reject".to_string()) // nominated + adjudicated, neither merged nor aliased
    } else {
        None // cosine < 0.70 — never reached the LLM-verify band at all
    };

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory::tests::dream_metrics_harness_site3",
            row_id = %row.id,
            a = %row.a,
            b = %row.b,
            category = %row.category,
            ground_truth = ?row.ground_truth,
            decision = ?decision,
            pairs_examined = report.pairs_examined,
            candidates_nominated = report.candidates_nominated,
            merges_applied = report.merges_applied,
            "per-row scoring detail"
        );
    }

    let row_wall_clock_ms = row_start.elapsed().as_secs_f64() * 1000.0;

    Ok((
        RowOutcome {
            row_id: row.id.clone(),
            category: row.category.clone(),
            ground_truth: row.ground_truth,
            decision,
            candidates_nominated: report.candidates_nominated,
            pairs_examined: report.pairs_examined,
            row_wall_clock_ms,
        },
        graph,
        gid,
    ))
}

// ─── Metrics computation (§3/§3.1) — mirrors dream_metrics_harness.rs ────────

#[derive(Debug, Clone, Default, serde::Serialize)]
struct CategoryMetrics {
    n: usize,
    tp: usize,
    fp: usize,
    fn_: usize,
    tn: usize,
    false_merges: usize,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
    precision_ci_low: Option<f64>,
    precision_ci_high: Option<f64>,
}

fn decision_says_same(decision: &Option<String>) -> bool {
    matches!(decision.as_deref(), Some("merge") | Some("potential_alias"))
}

fn decision_is_merge(decision: &Option<String>) -> bool {
    matches!(decision.as_deref(), Some("merge"))
}

/// Compute precision/recall/F1/Wilson-CI per §3.1's denominator definition.
/// Every row in this corpus has ground truth `Same`/`Different`/`Uncertain`.
/// `Uncertain` rows (the `borderline_type_identity` category) are excluded
/// from tp/fp/fn/tn entirely (mirrors Site #5's `compute_metrics` `continue`
/// carve-out for `GroundTruth::Uncertain`/`NotNominated`) — ground truth
/// itself is ambiguous by design for those rows, so they cannot contribute to
/// a precision/recall measurement.
fn compute_metrics(outcomes: &[&RowOutcome]) -> CategoryMetrics {
    let mut m = CategoryMetrics {
        n: outcomes.len(),
        ..Default::default()
    };

    for o in outcomes {
        let truly_same = match o.ground_truth {
            GroundTruth::Same => true,
            GroundTruth::Different => false,
            GroundTruth::Uncertain => continue,
        };
        let pass_says_same = decision_says_same(&o.decision);

        match (pass_says_same, truly_same) {
            (true, true) => m.tp += 1,
            (true, false) => {
                m.fp += 1;
                if decision_is_merge(&o.decision) {
                    m.false_merges += 1;
                }
            }
            (false, true) => m.fn_ += 1,
            (false, false) => m.tn += 1,
        }
    }

    let flagged_same = m.tp + m.fp;
    let genuinely_same_total = m.tp + m.fn_;

    m.precision = if flagged_same == 0 {
        None
    } else {
        Some(m.tp as f64 / flagged_same as f64)
    };
    m.recall = if genuinely_same_total == 0 {
        None
    } else {
        Some(m.tp as f64 / genuinely_same_total as f64)
    };
    m.f1 = match (m.precision, m.recall) {
        (Some(p), Some(r)) if (p + r) > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    if flagged_same > 0 {
        let (lo, hi) = wilson_lower_upper(m.tp, flagged_same);
        m.precision_ci_low = Some(lo);
        m.precision_ci_high = Some(hi);
    }

    m
}

/// Merge-only precision (stricter, destructive-write-only measure): of the
/// pairs the pass actually MERGED (not PotentialAlias), what fraction are
/// genuinely the same concept?
fn compute_merge_only_precision(outcomes: &[&RowOutcome]) -> (usize, usize, Option<f64>) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    for o in outcomes {
        let truly_same = match o.ground_truth {
            GroundTruth::Same => true,
            GroundTruth::Different => false,
            GroundTruth::Uncertain => continue,
        };
        if !decision_is_merge(&o.decision) {
            continue;
        }
        if truly_same {
            tp += 1;
        } else {
            fp += 1;
        }
    }
    let precision = if tp + fp == 0 {
        None
    } else {
        Some(tp as f64 / (tp + fp) as f64)
    };
    (tp, fp, precision)
}

// ─── o11y cross-check (§3.5) ─────────────────────────────────────────────────

const SITE: &str = "site3_type_registry";

/// Sum a named counter, optionally filtered by `site` label. Takes the
/// `Snapshotter` (not a `Snapshot` — `Snapshot` is not `Clone`) so each call
/// takes its own fresh snapshot.
fn sum_counter(snapshotter: &Snapshotter, metric_name: &str, site_filter: Option<&str>) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != metric_name {
                return None;
            }
            if let Some(site) = site_filter {
                let labels: HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("site").copied() != Some(site) {
                    return None;
                }
            }
            if let DebugValue::Counter(n) = value {
                Some(n)
            } else {
                None
            }
        })
        .sum()
}

/// Sum the `kremory.identity.write_gate_decision_total{site,decision}`
/// counter for one specific `(site, decision)` label pair.
/// TD-180/TD-182: every `kremory.test.cassette_miss_total` bucket, as
/// `(cassette, count)`, so a stale-cassette run names the files rather than
/// reporting a mysteriously degraded metric.
///
/// Reads the CAUSE. The alternative — hoping a downstream quality metric
/// notices — is what failed on 2026-08-05: recall fell 1.0 -> 0.339 and every
/// gate stayed green because all of them were precision/safety-shaped.
fn cassette_miss_rows(snapshotter: &Snapshotter) -> Vec<(String, u64)> {
    let mut rows: Vec<(String, u64)> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.test.cassette_miss_total" {
                return None;
            }
            let labels: HashMap<&str, &str> = key.labels().map(|l| (l.key(), l.value())).collect();
            let cassette = labels.get("cassette").copied().unwrap_or("<unknown>").to_string();
            match value {
                DebugValue::Counter(n) if n > 0 => Some((cassette, n)),
                _ => None,
            }
        })
        .collect();
    rows.sort();
    rows
}

fn sum_write_gate_decision_counter(snapshotter: &Snapshotter, site: &str, decision: &str) -> u64 {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != "kremory.identity.write_gate_decision_total" {
                return None;
            }
            let labels: HashMap<&str, &str> = key.labels().map(|l| (l.key(), l.value())).collect();
            if labels.get("site").copied() == Some(site)
                && labels.get("decision").copied() == Some(decision)
            {
                if let DebugValue::Counter(n) = value {
                    return Some(n);
                }
            }
            None
        })
        .sum()
}

/// Collect all recorded samples for a named histogram, optionally filtered
/// by `site` label.
fn histogram_samples(
    snapshotter: &Snapshotter,
    metric_name: &str,
    site_filter: Option<&str>,
) -> Vec<f64> {
    snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter_map(|(composite_key, _, _, value)| {
            let key = composite_key.key();
            if key.name() != metric_name {
                return None;
            }
            if let Some(site) = site_filter {
                let labels: HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("site").copied() != Some(site) {
                    return None;
                }
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

/// Latency metrics. `llm_call_ms_*` sourced from the pass's own histogram
/// `kremory.identity.llm_call_latency_ms_histogram{site=site3_type_registry}`
/// — but unlike Site #5 (every pair always calls the LLM), MANY Site #3 pairs
/// resolve via cosine + lexical pre-filter alone (auto-merge, or cos<0.70
/// no-candidate) with ZERO LLM calls — so `llm_call_count` is expected to be
/// well below `row_wall_clock_ms`'s row count. `embed_call_count` is tracked
/// separately since EVERY row makes exactly 2 embedding calls (one per type
/// description) regardless of whether adjudication fires.
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
    embed_calls_expected: usize,
}

fn compute_latency_metrics(
    snapshotter: &Snapshotter,
    outcomes: &[RowOutcome],
    vcr_mode: VcrMode,
) -> LatencyMetrics {
    let llm_samples = histogram_samples(
        snapshotter,
        "kremory.identity.llm_call_latency_ms_histogram",
        Some(SITE),
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
        embed_calls_expected: outcomes.len() * 2,
    }
}

/// Volume metrics — pairs examined / candidates nominated / nomination rate,
/// cross-checked against the pass's own counters by `crosscheck_o11y`.
#[derive(Debug, Clone, Default, serde::Serialize)]
struct VolumeMetrics {
    pairs_examined: usize,
    candidates_nominated: usize,
    nomination_rate: Option<f64>,
    merges_applied: usize,
    potential_aliases: usize,
    rejects: usize,
    not_examined: usize,
}

fn compute_volume_metrics(outcomes: &[RowOutcome]) -> VolumeMetrics {
    let pairs_examined: usize = outcomes.iter().map(|o| o.pairs_examined).sum();
    let candidates_nominated: usize = outcomes.iter().map(|o| o.candidates_nominated).sum();
    let merges_applied = outcomes
        .iter()
        .filter(|o| decision_is_merge(&o.decision))
        .count();
    let potential_aliases = outcomes
        .iter()
        .filter(|o| o.decision.as_deref() == Some("potential_alias"))
        .count();
    let rejects = outcomes
        .iter()
        .filter(|o| o.decision.as_deref() == Some("reject"))
        .count();
    let not_examined = outcomes.iter().filter(|o| o.decision.is_none()).count();

    VolumeMetrics {
        pairs_examined,
        candidates_nominated,
        nomination_rate: if pairs_examined == 0 {
            None
        } else {
            Some(candidates_nominated as f64 / pairs_examined as f64)
        },
        merges_applied,
        potential_aliases,
        rejects,
        not_examined,
    }
}

/// Cross-check the pass's own o11y counters (§3.5) against the harness's
/// observed decisions. A divergence is a FAIL — the counter must not lie.
fn crosscheck_o11y(snapshotter: &Snapshotter, outcomes: &[RowOutcome]) -> Result<(), String> {
    let observed_examined: u64 = outcomes.iter().map(|o| o.pairs_examined as u64).sum();
    let counter_examined = sum_counter(
        snapshotter,
        "kremory.dream.type_registry_collapse.pairs_examined_total",
        None,
    );
    if observed_examined != counter_examined {
        return Err(format!(
            "o11y divergence: kremory.dream.type_registry_collapse.pairs_examined_total = \
             {counter_examined}, but harness observed {observed_examined} pairs examined \
             summed across report.pairs_examined per row"
        ));
    }

    let observed_nominated: u64 = outcomes.iter().map(|o| o.candidates_nominated as u64).sum();
    let counter_nominated = sum_counter(
        snapshotter,
        "kremory.identity.candidate_nominated_total",
        Some(SITE),
    );
    if observed_nominated != counter_nominated {
        return Err(format!(
            "o11y divergence: kremory.identity.candidate_nominated_total{{site={SITE}}} = \
             {counter_nominated}, but harness observed {observed_nominated} nominations \
             summed across report.candidates_nominated per row"
        ));
    }

    let observed_merges = outcomes
        .iter()
        .filter(|o| decision_is_merge(&o.decision))
        .count() as u64;
    let counter_merges = sum_counter(
        snapshotter,
        "kremory.dream.type_registry_collapse.merges_applied_total",
        None,
    );
    if observed_merges != counter_merges {
        return Err(format!(
            "o11y divergence: kremory.dream.type_registry_collapse.merges_applied_total = \
             {counter_merges}, but harness observed {observed_merges} merge decisions"
        ));
    }

    for decision_label in ["merge", "potential_alias", "reject"] {
        let observed = outcomes
            .iter()
            .filter(|o| o.decision.as_deref() == Some(decision_label))
            .count() as u64;
        let counter_labeled = sum_write_gate_decision_counter(snapshotter, SITE, decision_label);
        if observed != counter_labeled {
            return Err(format!(
                "o11y divergence: kremory.identity.write_gate_decision_total{{site={SITE},\
                 decision={decision_label}}} = {counter_labeled}, but harness observed \
                 {observed} \"{decision_label}\" decisions"
            ));
        }
    }

    // Harness-owned false-merge counter — never in production source,
    // incremented here whenever a genuinely-DIFFERENT-concept row (ground
    // truth `same_concept=false`) lands on "merge".
    for o in outcomes {
        if o.ground_truth == GroundTruth::Different && decision_is_merge(&o.decision) {
            metrics::counter!(
                "dream.corpus.false_merge_total",
                "site" => SITE,
                "category" => o.category.clone()
            )
            .increment(1);
        }
    }

    Ok(())
}

// ─── Report emission ─────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct MetricsReport {
    site: String,
    overall: CategoryMetrics,
    per_category: std::collections::BTreeMap<String, CategoryMetrics>,
    merge_only_precision: Option<f64>,
    merge_only_tp: usize,
    merge_only_fp: usize,
    n_flagged_same: usize,
    borderline_type_identity_n: usize,
    borderline_type_identity_rows: Vec<String>,
    safety_false_merges: usize,
    false_merge_rows: Vec<String>,
    hard_gate: String,
    latency: LatencyMetrics,
    volume: VolumeMetrics,
}

fn print_and_write_report(report: &MetricsReport) {
    eprintln!("\n── Site #3 metrics harness — full report ──────────────────────────────");
    eprintln!(
        "  OVERALL: n={} precision={:?} recall={:?} f1={:?} ci=[{:?},{:?}]",
        report.overall.n,
        report.overall.precision,
        report.overall.recall,
        report.overall.f1,
        report.overall.precision_ci_low,
        report.overall.precision_ci_high,
    );
    for (cat, m) in &report.per_category {
        eprintln!(
            "  [{cat}] n={} tp={} fp={} fn={} tn={} false_merges={} precision={:?} \
             recall={:?} f1={:?} ci=[{:?},{:?}]",
            m.n,
            m.tp,
            m.fp,
            m.fn_,
            m.tn,
            m.false_merges,
            m.precision,
            m.recall,
            m.f1,
            m.precision_ci_low,
            m.precision_ci_high,
        );
    }
    eprintln!(
        "\n  merge_only_precision={:?} (tp={} fp={})",
        report.merge_only_precision, report.merge_only_tp, report.merge_only_fp,
    );
    eprintln!(
        "\n  n_flagged_same={} borderline_type_identity_n={} safety_false_merges={} hard_gate={}",
        report.n_flagged_same,
        report.borderline_type_identity_n,
        report.safety_false_merges,
        report.hard_gate,
    );
    if !report.false_merge_rows.is_empty() {
        eprintln!("  FALSE MERGE ROWS:");
        for r in &report.false_merge_rows {
            eprintln!("    *** {r} ***");
        }
    }

    eprintln!(
        "\n  LATENCY [{}]: llm_call_count={} llm_call_total_ms={:.1} \
         llm_call_ms p50={:?} p95={:?} p99={:?} | row_wall_clock_ms p50={:?} p95={:?} p99={:?} \
         | total_harness_wall_clock_ms={:.1} | embed_calls_expected={}",
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
        report.latency.embed_calls_expected,
    );
    eprintln!(
        "  VOLUME: pairs_examined={} candidates_nominated={} nomination_rate={:?} \
         merges_applied={} potential_aliases={} rejects={} not_examined={}",
        report.volume.pairs_examined,
        report.volume.candidates_nominated,
        report.volume.nomination_rate,
        report.volume.merges_applied,
        report.volume.potential_aliases,
        report.volume.rejects,
        report.volume.not_examined,
    );

    // ── TD-180 item 4 / TD-182 item 3: OUTPUT is not the BASELINE ────────
    //
    // This harness used to write straight over `tests/corpora/site3_metrics.json`
    // — the very file it is judged against. A degraded run therefore rewrote
    // its own expectation, silently. On 2026-08-05 a stale-cassette run
    // rewrote site3's `recall` from 1.0 to 0.339 and only a hand `git checkout`
    // stopped that being committed as the new normal.
    //
    // Output now always goes to the gitignored workspace `target/`. The
    // committed baseline is rewritten ONLY under an explicit opt-in, so
    // updating an expectation is a deliberate act with a visible diff.
    let json = serde_json::to_string_pretty(report).expect("serialize metrics report");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("site3_metrics.json");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create {parent:?}: {e}"));
    }
    std::fs::write(&out_path, &json)
        .unwrap_or_else(|e| panic!("failed to write metrics report to {out_path:?}: {e}"));
    eprintln!("  metrics JSON written to {out_path:?} (run output)");

    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpora")
        .join("site3_metrics.json");
    if std::env::var("KREMORY_UPDATE_BASELINE").is_ok() {
        std::fs::write(&baseline_path, &json).unwrap_or_else(|e| {
            panic!("failed to update baseline {baseline_path:?}: {e}")
        });
        eprintln!("  BASELINE UPDATED at {baseline_path:?} (KREMORY_UPDATE_BASELINE set)");
    } else {
        eprintln!(
            "  baseline at {baseline_path:?} left UNTOUCHED \
             (set KREMORY_UPDATE_BASELINE=1 to rewrite it deliberately)"
        );
    }
}

// ─── smoke-one-before-batch ───────────────────────────────────────────────────

/// smoke-one-before-batch (hard rule): run ONLY corpus row `s3-001`
/// (Company/company, `trivial_duplicate`) through the full scoring pipeline
/// first, confirm the per-row plant->run->score wiring is sane, THEN proceed
/// to the full 119-row corpus run.
#[tokio::test]
#[ignore = "dream_metrics_harness_site3: requires Ollama in record mode, or a committed cassette \
            in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test it dream_metrics_harness_site3:: -- --ignored \
            --nocapture smoke_one_metrics_harness_site3"]
async fn smoke_one_metrics_harness_site3() {
    let mode = resolve_vcr_mode();
    let corpus = load_corpus();
    let smoke_row = corpus
        .iter()
        .find(|r| r.id == "s3-001")
        .expect("corpus must contain s3-001 (Company/company)");
    assert_eq!(smoke_row.a, "Company");
    assert_eq!(smoke_row.category, "trivial_duplicate");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let (outcome, graph, gid) = run_one_row(smoke_row, mode)
        .await
        .unwrap_or_else(|e| panic!("smoke-one row failed: {e}"));

    eprintln!(
        "[smoke-one] row_id={} a={} b={} category={} ground_truth={:?} decision={:?} \
         pairs_examined={} candidates_nominated={} group_id={gid}",
        outcome.row_id,
        smoke_row.a,
        smoke_row.b,
        outcome.category,
        outcome.ground_truth,
        outcome.decision,
        outcome.pairs_examined,
        outcome.candidates_nominated,
    );

    assert_eq!(
        outcome.pairs_examined, 1,
        "smoke-one fixture has exactly 2 non-catch-all types -> 1 pair"
    );
    assert_eq!(
        outcome.decision.as_deref(),
        Some("merge"),
        "smoke-one: Company/company (trivial duplicate, exact-lemma-match + near-identical \
         description) should auto-merge via write_gate row 1 (cosine>=0.85 + lexical match, \
         no LLM call needed) — got decision={:?}",
        outcome.decision,
    );

    // Confirm the destructive write actually happened — exactly one of the
    // two types must have disappeared (the loser was remapped + removed).
    let company_survived = type_exists(&graph.conn, &gid, &smoke_row.a).await;
    let company_lower_survived = type_exists(&graph.conn, &gid, &smoke_row.b).await;
    assert_ne!(
        company_survived, company_lower_survived,
        "smoke-one: exactly one of Company/company must survive a merge — \
         company_survived={company_survived} company_lower_survived={company_lower_survived}"
    );

    crosscheck_o11y(&snapshotter, std::slice::from_ref(&outcome))
        .unwrap_or_else(|e| panic!("smoke-one o11y cross-check FAILED: {e}"));

    eprintln!("[smoke-one] PASS — proceeding to full 119-row corpus run is safe.");
}

// ─── Full corpus run — the binding metrics harness ───────────────────────────

#[tokio::test]
#[ignore = "dream_metrics_harness_site3: requires Ollama in record mode, or a committed cassette \
            in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features test-utils,llm-smoke --test it dream_metrics_harness_site3:: -- --ignored \
            --nocapture full_corpus_site3_metrics"]
async fn full_corpus_site3_metrics() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kremory=debug")),
        )
        .with_test_writer()
        .try_init();

    let mode = resolve_vcr_mode();
    let corpus = load_corpus();
    assert_eq!(corpus.len(), 119, "corpus must have exactly 119 rows");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut outcomes: Vec<RowOutcome> = Vec::with_capacity(corpus.len());
    for row in &corpus {
        let (outcome, _graph, _gid) = run_one_row(row, mode)
            .await
            .unwrap_or_else(|e| panic!("row {} failed: {e}", row.id));
        outcomes.push(outcome);
    }

    // ── o11y cross-check (§3.5) — divergence is a FAIL, never a soft warning ──
    crosscheck_o11y(&snapshotter, &outcomes)
        .unwrap_or_else(|e| panic!("o11y cross-check FAILED: {e}"));

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

    // ── Overall (site-level) metrics ──────────────────────────────────────
    let all_refs: Vec<&RowOutcome> = outcomes.iter().collect();
    let overall = compute_metrics(&all_refs);
    let (merge_only_tp, merge_only_fp, merge_only_precision) =
        compute_merge_only_precision(&all_refs);

    // ── SAFETY gate (task brief, exact-zero): distinct_lemma_collision +
    // distinct_unrelated (+ band_edge_moderate, also genuinely distinct) must
    // NEVER merge. A PotentialAlias outcome on band_edge_moderate is fine
    // (deferred, audit-only, no destructive write) — only an actual MERGE on
    // any `same_concept=false` row is a false merge. ──────────────────────
    let safety_categories = [
        "distinct_lemma_collision",
        "distinct_unrelated",
        "band_edge_moderate",
    ];
    let mut safety_false_merges = 0usize;
    let mut false_merge_rows: Vec<String> = Vec::new();
    for cat in safety_categories {
        if let Some(m) = per_category.get(cat) {
            safety_false_merges += m.false_merges;
        }
    }
    for o in &outcomes {
        if o.ground_truth == GroundTruth::Different && decision_is_merge(&o.decision) {
            let row = corpus
                .iter()
                .find(|r| r.id == o.row_id)
                .expect("row lookup");
            false_merge_rows.push(format!(
                "{} ({} / {}, category={})",
                o.row_id, row.a, row.b, o.category
            ));
        }
    }

    // ── borderline_type_identity — descriptive report only, never gated ────
    let borderline_outcomes: Vec<&RowOutcome> = outcomes
        .iter()
        .filter(|o| o.ground_truth == GroundTruth::Uncertain)
        .collect();
    let borderline_type_identity_rows: Vec<String> = borderline_outcomes
        .iter()
        .map(|o| {
            let row = corpus
                .iter()
                .find(|r| r.id == o.row_id)
                .expect("row lookup");
            format!(
                "{} ({} / {}) — decision={:?} pass_says_same={}",
                o.row_id,
                row.a,
                row.b,
                o.decision,
                decision_says_same(&o.decision)
            )
        })
        .collect();

    let n_flagged_same = overall.tp + overall.fp;

    // ── HARD GATE (spec §4.1 honest fallback): point-estimate aggregate
    // precision >= 0.90 AND false_merges == 0 on every genuinely-distinct
    // (same_concept=false) row. Wilson lower-CI reported as a quality signal,
    // NOT the binding gate (mirrors Site #5's N-limited-corpus discipline). ──
    let point_precision = overall.precision.unwrap_or(0.0);
    let hard_gate_pass = point_precision >= 0.90 && safety_false_merges == 0;
    let hard_gate = if hard_gate_pass { "PASS" } else { "FAIL" }.to_string();

    let latency = compute_latency_metrics(&snapshotter, &outcomes, mode);
    let volume = compute_volume_metrics(&outcomes);

    let report = MetricsReport {
        site: "site3_type_registry".to_string(),
        overall: overall.clone(),
        per_category,
        merge_only_precision,
        merge_only_tp,
        merge_only_fp,
        n_flagged_same,
        borderline_type_identity_n: borderline_outcomes.len(),
        borderline_type_identity_rows: borderline_type_identity_rows.clone(),
        safety_false_merges,
        false_merge_rows: false_merge_rows.clone(),
        hard_gate: hard_gate.clone(),
        latency,
        volume,
    };
    print_and_write_report(&report);

    eprintln!(
        "\n── borderline_type_identity (descriptive only, NOT part of the gate) — n={} ──",
        borderline_type_identity_rows.len()
    );
    for line in &borderline_type_identity_rows {
        eprintln!("  {line}");
    }

    // ── Per-row rationale dump for any FP/FN, so a failing gate is
    // immediately diagnosable without re-running with KREMORY_DEBUG=1 ──────
    // (GroundTruth::Uncertain rows are skipped — no truly_same fact exists.)
    for row in &corpus {
        let outcome = outcomes.iter().find(|o| o.row_id == row.id).unwrap();
        let truly_same = match outcome.ground_truth {
            GroundTruth::Same => Some(true),
            GroundTruth::Different => Some(false),
            GroundTruth::Uncertain => None,
        };
        let pass_says_same = decision_says_same(&outcome.decision);
        if let Some(truly_same) = truly_same {
            if truly_same != pass_says_same {
                eprintln!(
                    "  *** MISCLASSIFIED [{}] {} / {} — truly_same={} decision={:?} ***",
                    outcome.category, row.a, row.b, truly_same, outcome.decision,
                );
            }
        }
    }

    // ── TD-180/TD-182: fail on the CAUSE, before any downstream metric ────
    //
    // A cassette MISS returns Err from the provider, which the adjudication
    // path deliberately defaults to no-verdict — so a stale cassette becomes a
    // silent NON-MERGE, i.e. a false negative, not an error. On 2026-08-05 that
    // cost 67 of 119 cassettes and dropped recall 1.0 -> 0.339 with every gate
    // below still GREEN, because they are all precision/safety-shaped.
    //
    // Worse, the `warn!` naming the cause is destroyed by the test PASSING:
    // libtest captures per-test output and discards it on success, so grepping
    // afterwards returns empty and empty reads as healthy.
    //
    // Assert the cause directly. This cannot be captured away.
    let cassette_misses = cassette_miss_rows(&snapshotter);
    assert!(
        cassette_misses.is_empty(),
        "CASSETTE MISS GATE FAILED: {} recorded miss(es). Every metric below is measuring a \
         pass that never received an LLM verdict, NOT model quality. Re-record with \
         KREMORY_VCR=record against live Ollama (serially — record mode is bounded by one \
         local model server). Offending cassettes:\n{}",
        cassette_misses.len(),
        cassette_misses
            .iter()
            .map(|(c, n)| format!("  {n:>4} miss(es)  {c}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    // ── Hard assertions ───────────────────────────────────────────────────
    assert_eq!(
        safety_false_merges, 0,
        "SAFETY GATE FAILED: {safety_false_merges} false merge(s) among distinct_lemma_collision \
         / distinct_unrelated / band_edge_moderate (safety categories, ground truth = genuinely \
         DIFFERENT concepts). Literal-equality gate, never loosened. See false_merge_rows above \
         for exactly which corpus row(s) caused it."
    );
    assert!(
        overall.precision.is_some(),
        "aggregate precision undefined (no positive 'same' calls at all) — the corpus or the \
         pass regressed; see per-row report above"
    );
    assert!(
        point_precision >= 0.90,
        "PRECISION GATE FAILED: point-estimate precision {point_precision:.4} < 0.90 bar — \
         n_flagged_same={n_flagged_same}, tp={} fp={}. Wilson 95% CI=[{:?}, {:?}]. See per-row \
         misclassification report above.",
        overall.tp,
        overall.fp,
        overall.precision_ci_low,
        overall.precision_ci_high,
    );

    // ── TD-182: RECALL floor — the gate above is one-sided ────────────────
    //
    // Every assertion before this point detects OVER-merging. None detects
    // UNDER-merging, so a pass that simply stops merging IMPROVES them all
    // (precision 0.949 -> 1.0, false-merges 3 -> 0) while losing two thirds of
    // the merges the feature exists to make. That is exactly what happened on
    // 2026-08-05 and it shipped as GREEN.
    //
    // Floor is 0.90 against a 2026-07-03 baseline of 1.0 — loose enough for
    // ordinary model drift, tight enough that the 0.339 collapse is impossible
    // to miss. If this fires, FIND OUT WHY; do not lower it.
    let measured_recall = overall
        .recall
        .expect("recall is defined whenever any genuinely-same pair exists in the corpus");
    assert!(
        measured_recall >= 0.90,
        "RECALL GATE FAILED: {measured_recall:.4} < 0.90 (2026-07-03 baseline: 1.0) — \
         tp={} fn_={}. The precision/safety gates above CANNOT see this: a pass that stops \
         merging improves every one of them. Check the cassette-miss gate first; a stale \
         cassette becomes a silent non-merge.",
        overall.tp,
        overall.fn_,
    );

    eprintln!(
        "\n══ GATE: {hard_gate} (precision={point_precision:.4} >= 0.90, \
         false_merges={safety_false_merges}) ═══════════════"
    );
}
