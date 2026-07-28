//! `kremory-http` — thin REST transport over the SAME `handlers::do_remember`
//! / `do_recall` / `do_dream` bodies the MCP stdio server (`main.rs` /
//! `kremory-mcp-server`) uses. One Rust SDK (`kremory::Memory`), two thin
//! transports — this bin owns HTTP marshalling only; all facade-calling
//! logic lives in `handlers.rs` and is NOT duplicated here.
//!
//! Exists so a non-MCP harness (e.g. a Python benchmark script, codemem-style)
//! can drive kremory over plain HTTP without speaking JSON-RPC/MCP.
//!
//! Env-driven boot — the DB path / Ollama reachability preflight / `Memory`
//! construction sequence below is intentionally the SAME shape as
//! `main.rs`'s (verbatim boot logic, not extracted to a shared fn: each
//! `[[bin]]` target is a distinct crate, and this ~25-line sequence is the
//! only thing that would need sharing — not worth a new lib-level export for
//! two call sites).
//!
//! ```bash
//! export KREMORY_MCP_DB_PATH=./agent.db
//! cargo run -p kremory-mcp --bin kremory-http
//! ```
//!
//! ## Env vars
//!
//! - `KREMORY_MCP_DB_PATH` — required. Path to the kremory libSQL database.
//! - `KREMORY_MCP_OLLAMA_URL` — default `http://localhost:11434`.
//! - `KREMORY_MCP_MODEL_ID` — default `gemma4:e4b`.
//! - `PORT` — default `3179` (matches codemem's benchmark harness default).
//!
//! ## Routes
//!
//! NOTE for the harness (E2): routes are BARE — there is NO `/api` prefix
//! (unlike codemem, whose routes nest under `/api`). Point the harness at
//! `--base-url http://localhost:3179` (NOT `.../api`).
//!
//! | Route | Delegates to | Response |
//! |---|---|---|
//! | `GET /health` | — | `200` |
//! | `POST /memories` `{content, namespace, published_at?}` | `handlers::do_remember` | `201 {"id"}` |
//! | `GET /search?q=&namespace=&k=&mode=` | `handlers::do_recall` / `do_recall_content` | `200 {"results":[{"id","content","score","kind","source_episode_id"}]}` |
//!
//! `results[].kind` (`"entity" \| "episode" \| "fact"`) and
//! `results[].source_episode_id` (populated ONLY for `kind == "fact"`) are
//! TD-139 measurement-prerequisite fields, additive over the pre-existing
//! `{id, content, score}` contract — see [`SearchResultKindWire`] and
//! [`SearchResultWire::source_episode_id`]. No live arm emits `"fact"` yet.
//!
//! `mode` (benchmark-completion-roadmap W0.1) selects which of kremory's
//! retrieval surfaces `/search` reaches: `recall` is the existing entity/fact
//! hybrid keyword+semantic+graph path; `content` is ADR-072 seq1's BM25-only
//! full-text search over raw `episodes.content` (requires this bin built with
//! `--features content-search`; otherwise `mode=content`/`mode=hybrid` HARD-FAIL
//! 422 rather than silently degrading — B1 fail-loud, see `content_mode_results`);
//! `hybrid` runs both and RRF-fuses them (see `rrf_merge` below). `hybrid` is
//! the DEFAULT since the 2026-07-21 LoCoMo diagnostic (entity-graph `recall`
//! alone judged 40.2% vs 70.4% hybrid).
//!
//! TD-066 Increment 1 (`.ai-docs/specs/td-066-recall-scoring-foundation-
//! spec-2026-07-21.md` §3): the fusion this file pioneered has been ported
//! into `core::search::rrf_fuse_with_content` and wired into
//! `Memory::recall()`'s `.raw()`/`execute()` terminals by default whenever
//! `content-search` is compiled in — so `recall_mode_results` below (which
//! calls `handlers::do_recall` → `.raw()`) is **no longer entity-graph-only**
//! in a `content-search`-enabled build; it now collaterally receives the
//! SAME fusion `mode=hybrid` does.
//!
//! ⚠️ **CORRECTED 2026-07-28 (ADR-078 / TD-153).** The sentence that used to
//! stand here — *"`mode=recall` vs `mode=hybrid` therefore no longer differ
//! when this bin is built with `content-search` — both reach the fused
//! surface"* — is **FALSE, and was measured false**: conv0, same server, same
//! DB, dense arm on, rerank off, explicit `?mode=`:
//!
//! | mode | recall@10 | nDCG@10 | hit |
//! |---|---|---|---|
//! | `recall` (what `Memory::recall().raw()` returns) | 77.2 | 63.7 | 83.2 |
//! | `hybrid` | **82.1** | **69.3** | **87.9** |
//!
//! The premise was right and the conclusion did not follow: `recall` DOES now
//! receive the substrate's `rrf_fuse_with_content`, but `hybrid_mode_results`
//! then runs a **SECOND, independent `content_mode_results` pass** and
//! `rrf_merge`s it on top of that already-fused list. So hybrid double-weights
//! the content arm and adds a REST-depth BM25 pass the library never issues.
//! Net effect: **the LIBRARY under-performs its own HTTP wrapper by 4.9
//! recall@10** — and since the library is the product, that is backwards.
//! Tracked as TD-153; do not treat the two modes as interchangeable, and note
//! that any benchmark which omits `?mode=` is measuring HYBRID, not the
//! library path (the LoCoMo harness did exactly this until 2026-07-28).
//!
//! This is a known, deliberate consequence of
//! wiring fusion into the canonical facade terminal (the spec explicitly
//! sanctions the default-behaviour change with no separate HITL), not an
//! oversight — flagged here rather than silently left stale. Untangling the
//! REST `mode` selector's semantics (or removing it in favour of always
//! calling the now-fused `.raw()` directly) is deferred past Increment 1;
//! the selector is left wired as-is because the LoCoMo bench harness this
//! spec is grounded in drives `mode=recall`/`mode=content`/`mode=hybrid`
//! directly for its own three-way surface comparison (see spec §1.1's
//! table) and must not be broken out from under it mid-run.
//! | `DELETE /namespaces/{ns}` | `Memory::forget` | `200 {"deleted"}` |
//! | `POST /consolidation/{cycle}?namespace=` | `handlers::do_dream` | `200` (422 if `?namespace=` omitted) |

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use axum::extract::{Path as AxumPath, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
#[cfg(feature = "prometheus")]
use metrics_exporter_prometheus::PrometheusBuilder;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

use kremory::Memory;
use kremory_mcp::handlers::{self, ToolError};
use kremory_mcp::health;
use kremory_mcp::params::{
    DreamParams, RecallFormat, RecallParams, RecallStructuredOutput, RecallTemplateWire,
    RememberParams, RetrievedContextWire,
};

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const REACHABILITY_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_PORT: u16 = 3179;

#[derive(Clone)]
struct AppState {
    mem: Arc<Memory>,
    /// recall-improvement-e2e-spec-2026-07-22 §S0-infra (D3): RRF `k` for the
    /// bin-local `rrf_merge` hybrid fusion, read once from `KREMORY_RRF_K` at
    /// boot (default 60). The library fusion sites read their own `k` from the
    /// Engine's `SearchConfig` (via `search_env_overrides`); this bin-local
    /// value keeps the REST hybrid arm's `rrf_merge` on the SAME sweep point.
    rrf_k: usize,
}

/// Pure parsing logic for the `KREMORY_RERANK_K` boot override, extracted out
/// of the `RERANK_K` `LazyLock` below so it is unit-testable. A `LazyLock` is
/// read exactly once per process and can't be re-run per test case — this is
/// the SAME untestability shape `instrument-real-data-flow-before-hypothesizing`
/// names for process-global config: the only way to exercise the parsing
/// branches is to extract them into a function the static then calls.
///
/// `raw` mirrors `std::env::var("KREMORY_RERANK_K").ok()`: `None` when the var
/// is absent (or not valid UTF-8 — `VarError::NotUnicode` collapses to `None`
/// via `.ok()`, matching the pre-extraction `Err(_) => None` arm exactly).
/// `Some("")` for a present-but-empty value. Malformed/negative/empty values
/// WARN and disable the reranker (`None`) rather than silently defaulting —
/// Rule 21 (parse loudly, never silently default) applies to boot config the
/// same as it does to LLM output: a benchmark run that silently disabled
/// reranking would misreport its own configuration (the exact TD-134
/// provenance-stamp failure mode `/health`'s `rerank_k` field exists to
/// prevent).
fn parse_rerank_k(raw: Option<&str>) -> Option<usize> {
    let raw = raw?;
    match raw.trim().parse::<usize>() {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(
                value = %raw,
                error = %e,
                "KREMORY_RERANK_K is not a valid usize — reranker disabled"
            );
            None
        }
    }
}

/// `KREMORY_RERANK_K` boot override — enables the TD-062 cross-encoder reranker
/// on this REST route for the TD-134 LoCoMo A/B. Unlike `rrf_k` (an `AppState`
/// field, threaded into `rrf_merge`), `rerank_k` is a per-recall param consumed
/// only in the `search` handler and carried on `RecallParams`, NOT in
/// `SearchConfig` — so it lives in a process-wide `LazyLock` (read once at first
/// request) rather than `AppState`, keeping every `AppState` literal untouched.
/// Mirrors the `KREMORY_RRF_K` sweep design: an A/B costs a restart, not a
/// rebuild. `None`/malformed ⇒ rerank off (pre-TD-134 behaviour). Fail-loud: a
/// malformed value WARNs and disables rerank rather than silently accepting it.
///
/// The actual parsing/validation lives in [`parse_rerank_k`] (unit-tested
/// below) — this `LazyLock` is a thin, untestable-by-construction shim around
/// it.
static RERANK_K: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
    parse_rerank_k(std::env::var("KREMORY_RERANK_K").ok().as_deref())
});

// ────────────────────────────────────────────────────────────────────────
// Error mapping — ToolError -> HTTP status + JSON body.
// ────────────────────────────────────────────────────────────────────────

struct ApiError(ToolError);

impl From<ToolError> for ApiError {
    fn from(e: ToolError) -> Self {
        Self(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            ToolError::InvalidParams(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ToolError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let message = match self.0 {
            ToolError::InvalidParams(msg) | ToolError::Internal(msg) => msg,
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

// ────────────────────────────────────────────────────────────────────────
// GET /health
// ────────────────────────────────────────────────────────────────────────

/// `GET /health` — liveness (still 200) PLUS the server's build-feature flags
/// and its ACTIVE scoring config (TD-135 / recall-improvement-e2e-spec-2026-07-22
/// §S0-infra). The bench harness's `provenance.build_provenance()` stamps the
/// recall-run JSON from THIS response so the provenance record is faithful to
/// what the search path actually used — rather than re-reading the harness
/// process's env, which can silently differ from the server's env and is exactly
/// how a config-mismatch produced a bogus benchmark number.
///
/// - `content_search` / `rerank` / `prometheus` are compile-time
///   `cfg!(feature = "…")` booleans — an orchestrator can now detect a degraded
///   recall-only build programmatically instead of scraping the boot banner
///   (closes the Stage-0 Quinn LOW).
/// - `scoring` is read from the LIVE [`kremory::Memory::search_config`] (the
///   Engine's `SearchConfig`, carrying any `KREMORY_CONTENT_WEIGHT` /
///   `KREMORY_RRF_K` boot overrides), NOT re-read from env here.
///
/// Additive: the response is still `200 OK`; the JSON body is new (the prior
/// handler returned an empty 200), so no existing field is removed.
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let scoring = state.mem.search_config();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "content_search": cfg!(feature = "content-search"),
            "rerank": cfg!(feature = "rerank"),
            // TD-134: the ACTIVE rerank_k (KREMORY_RERANK_K boot override) — so the
            // bench provenance stamp records whether reranking was on + at what
            // depth. null ⇒ off. Distinct from the `rerank` compile-flag above
            // (compiled-in ≠ enabled): a build can ship the reranker yet run it off.
            "rerank_k": *RERANK_K,
            // The REQUESTED reranker (`KREMORY_RERANK_MODEL`), so a result JSON
            // records which cross-encoder produced it. Deliberately named
            // `_requested`, not `_active`: resolution lives in `kremory`'s
            // `parse_reranker_model`, and an unrecognised alias there falls back
            // to bge-base with a WARN. Echoing the raw value as "active" could
            // therefore stamp a model that never loaded — the provenance-lie
            // class TD-135 exists to close. The authoritative record of what
            // actually loaded is the `kremory.rerank.model_selected` INFO log.
            // Exposing the resolved name instead would mean new public API on
            // `kremory` (with ADR-031 napi-parity obligations) for a bench-only
            // sweep knob — not worth it; revisit if the knob outgrows benching.
            "rerank_model_requested": std::env::var("KREMORY_RERANK_MODEL")
                .ok()
                .unwrap_or_else(|| "bge-base (default)".to_string()),
            "prometheus": cfg!(feature = "prometheus"),
            "scoring": {
                "content_stream_weight": scoring.content_stream_weight,
                "rrf_k": scoring.rrf_k,
                "graph_degree_weight": scoring.graph_degree_weight,
                "temporal_weight": scoring.temporal_weight,
                // TD-136 dense episode arm — surfaced so the bench provenance
                // stamp records whether KREMORY_EPISODE_DENSE was active (the
                // dense-vs-BM25 A/B is otherwise invisible in the recall-run JSON).
                "episode_dense_enabled": scoring.episode_dense_enabled,
            },
        })),
    )
}

// ────────────────────────────────────────────────────────────────────────
// POST /memories
// ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CreateMemoryBody {
    content: String,
    namespace: String,
    #[serde(default)]
    published_at: Option<String>,
}

async fn create_memory(
    State(state): State<AppState>,
    Json(body): Json<CreateMemoryBody>,
) -> Result<impl IntoResponse, ApiError> {
    let params = RememberParams {
        namespace: body.namespace,
        thread: None,
        content: body.content,
        source_kind: None,
        source_id: None,
        published_at: body.published_at,
        structured_facts: Vec::new(),
        skip_extraction: false,
    };
    let output = handlers::do_remember(&state.mem, params).await?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": output.episode_entity_id })),
    ))
}

// ────────────────────────────────────────────────────────────────────────
// GET /search
// ────────────────────────────────────────────────────────────────────────

/// `?mode=` selector (benchmark-completion-roadmap W0.1) — a REST-only
/// routing concept. `RecallParams` (the shared MCP+REST wire type in
/// `handlers.rs` / `params.rs`) carries no `mode` field and the MCP
/// `kremory_recall` tool has no equivalent; `mode` lives purely at this
/// query-string layer and selects which handler fn(s) `search()` below
/// calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SearchMode {
    Recall,
    Content,
    // Default is `hybrid` (RRF fusion of the entity/fact recall stream + the BM25
    // content stream), NOT the entity-graph `recall` surface. The 2026-07-21 LoCoMo
    // diagnostic (memory `project_kremory_locomo_recall_root_cause_retrieval_surface`)
    // showed the entity-graph `recall` surface judged only 40.2% vs 71.4% once the
    // content stream is RRF-fused in — a consumer hitting `/search` with no `?mode=`
    // must get the surface that actually answers, not the net-negative-on-LoCoMo
    // unscored-graph one. HARD-FAILS 422 when built without `content-search`
    // (B1 fail-loud — see the feature-off arms of content_/hybrid_mode_results),
    // never a silent degrade to `recall`.
    #[default]
    Hybrid,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
    namespace: String,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    mode: SearchMode,
}

/// What a `/search` result item IS — TD-139 measurement prerequisite
/// (`.ai-docs/tech-debt/tech-debt-register.md` "TD-139", the "⚠️ MEASUREMENT
/// PREREQUISITE" block). Additive wire metadata: existing consumers (the
/// LoCoMo harness) read only `id`/`content`/`score` and are unaffected by
/// this enum's presence. `Fact` is emitted by `recall_mode_results` (below)
/// once the entity+content fusion `handlers::do_recall` → `.raw()` reaches
/// includes a dense-fact-arm entry (TD-139 DoD item 2,
/// `core::search::rrf_fuse_with_facts`, feature-gated + default-OFF via
/// `SearchConfig::fact_dense_enabled` — absent that knob, no `Fact` item ever
/// appears, so this variant is dormant-but-wired on a default build, not
/// unreachable). This enum + `SearchResultWire::source_episode_id` exist so
/// `bench/locomo/evidence_eval.py` can score a fact-kind item against its
/// source episode's evidence turns instead of reading it as irrelevant (a
/// fact string does not contain LoCoMo turn text verbatim).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SearchResultKindWire {
    /// `mode=recall` / the recall arm of `mode=hybrid` — one entity + its
    /// flattened connected facts (`flatten_result_content`).
    Entity,
    /// `mode=content` / the content arm of `mode=hybrid` — a BM25-matched
    /// episode passage (`ContentPassage`).
    Episode,
    /// TD-139 DoD item 2: a dense-fact-arm hit
    /// (`core::search::vector_search_facts`, fused via `rrf_fuse_with_facts`)
    /// — `entity_type_name == "Fact"` on the underlying `RetrievedContext`
    /// is `recall_mode_results`'s discriminator for this variant.
    Fact,
}

#[derive(Debug, Serialize)]
struct SearchResultWire {
    id: String,
    content: String,
    score: f32,
    /// TD-139 measurement prerequisite: what this item IS. See
    /// [`SearchResultKindWire`].
    kind: SearchResultKindWire,
    /// Populated ONLY when `kind == Fact`: the episode this fact was
    /// asserted from (`facts.source_episode_id`, `core/schema.rs:129` —
    /// `Option<i64>` because a caller-supplied structured fact, or a fact
    /// whose source episode was later deleted, can carry no episode
    /// anchor). `None` for `Entity`/`Episode` items, and for `Fact` items
    /// with no recorded source. `bench/locomo/evidence_eval.py` resolves
    /// this id to the episode's evidence turns rather than scoring the
    /// fact's own (non-verbatim) text against them.
    source_episode_id: Option<i64>,
}

#[derive(Debug, Serialize)]
struct SearchResponseWire {
    results: Vec<SearchResultWire>,
}

/// Flattens a `RetrievedContextWire` (entity_name/summary + its connected
/// `facts[].fact` natural-language strings — `params.rs:117-135`) into a
/// single prose string. kremory's `RecallStructuredOutput` is entity-shaped
/// (one result = one entity + its facts); codemem's benchmark scorer expects
/// one flat `content` string per result to substring-match the gold answer
/// against — this is the "thin adapter" the REST route owns so
/// `handlers::do_recall` itself stays transport-agnostic.
fn flatten_result_content(r: &RetrievedContextWire) -> String {
    let mut parts = vec![format!("{}: {}", r.entity_name, r.summary)];
    parts.extend(r.facts.iter().map(|f| f.fact.clone()));
    parts.join(". ")
}

async fn search(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let params = RecallParams {
        namespace: query.namespace,
        thread: None,
        query: query.q,
        k: query.k,
        as_of: None,
        format: RecallFormat::Structured,
        template: RecallTemplateWire::default(),
        // TD-134 measurement: the TD-062 reranker is exposed on this bench/eval
        // REST route via the `KREMORY_RERANK_K` boot override (read once into the
        // `RERANK_K` static above), mirroring the `KREMORY_RRF_K` sweep pattern so
        // the LoCoMo A/B costs a server restart, not a rebuild. `None` ⇒ rerank off
        // (pre-TD-134 behaviour). Deep-pool discipline: the reranker sees exactly
        // the caller's `k` items (harness `--recall-limit 50`), reordered then
        // scored at top-10 downstream by evidence_eval.py (rank-aware, order-blind
        // substring scorer would read 0 — CLAUDE.md Rule 36).
        rerank_k: *RERANK_K,
    };
    let results = match query.mode {
        SearchMode::Recall => recall_mode_results(&state.mem, params).await?,
        SearchMode::Content => content_mode_results(&state.mem, params).await?,
        SearchMode::Hybrid => hybrid_mode_results(&state.mem, params, state.rrf_k).await?,
    };
    Ok(Json(SearchResponseWire { results }))
}

/// `mode=recall` — the entity-shaped keyword/semantic/graph path
/// (`handlers::do_recall`, structured format), flattened via
/// [`flatten_result_content`]. Byte-identical to `/search`'s pre-W0.1
/// behaviour ONLY when this bin is built WITHOUT `content-search`. When
/// `content-search` IS compiled in, `handlers::do_recall`'s `Structured`
/// format calls `.raw()`, and `.raw()` now fuses in the content stream by
/// default (TD-066 Increment 1, `core::search::rrf_fuse_with_content`) — see
/// this module's top-level doc comment.
async fn recall_mode_results(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    let value = handlers::do_recall(mem, params).await?;
    let structured: RecallStructuredOutput = serde_json::from_value(value).map_err(|e| {
        ApiError(ToolError::Internal(format!(
            "kremory-http: failed to deserialize recall structured output: {e}"
        )))
    })?;
    Ok(structured
        .results
        .iter()
        .map(|r| {
            // TD-139 DoD item 2: a dense-fact-arm hit is lifted into the
            // entity+content fusion as a synthetic `RetrievedContext` with
            // `entity_type_name == "Fact"`
            // (`core::search::fact_hit_into_retrieved_context`'s
            // discriminator) — mirrors how a `ContentPassage` is tagged
            // `"ContentPassage"` one layer down. `fact_dense_enabled`
            // defaults `false`, so no live result carries this tag unless
            // the knob is on.
            //
            // ADR-078 Phase A FIX (2026-07-28): `"ContentPassage"` was NOT
            // mapped, so every content-arm hit on this path was reported as
            // `kind: Entity` — i.e. a consumer was told that VERBATIM EPISODE
            // TEXT is a derived entity summary. Measured on conv0: 3980 of 3980
            // items across the top-20 came back `entity`, zero `episode`, on a
            // fused path where most of the useful items ARE episode passages.
            // `SearchResultKindWire::Episode` already existed and is documented
            // as exactly this case; only the mapping was missing.
            //
            // Why it matters beyond tidiness: an agent cannot tell distilled
            // summary from source text, so it cannot weight them differently.
            // That is the mechanism behind the measured dilution — adding the
            // graph moves single-hop +15.6pt but temporal -8.1pt, and of the
            // questions it flips right->wrong, 8 of 8 had the gold evidence
            // present in context anyway (RECALL-LEDGER §2quater).
            let kind = match r.entity_type_name.as_str() {
                "Fact" => SearchResultKindWire::Fact,
                "ContentPassage" => SearchResultKindWire::Episode,
                _ => SearchResultKindWire::Entity,
            };
            SearchResultWire {
                id: r.entity_id.clone(),
                content: flatten_result_content(r),
                score: r.score,
                kind,
                // The entity's connected facts each carry their OWN
                // `source_episode_ids` (`RetrievedFactWire`, params.rs) but
                // that provenance is discarded by `flatten_result_content`'s
                // join — an entity item is not itself a fact, so no single
                // episode id applies here. TD-139 leaves this loss in place
                // deliberately: splitting facts out of `flatten_result_content`
                // would change this arm's result count/content, which the
                // DoD requires to stay byte-identical. A `Fact`-kind item is
                // different: it is NOT an entity's connected fact, it IS a
                // dense-fact-arm hit, and its own `source_refs[0]` (a
                // `SourceKind::Episode` ref stamped by
                // `fact_hit_into_retrieved_context`) carries its provenance —
                // recover it here.
                source_episode_id: if kind == SearchResultKindWire::Fact {
                    r.source_refs
                        .iter()
                        .find(|sr| sr.kind == "episode")
                        .and_then(|sr| sr.id.parse::<i64>().ok())
                } else {
                    None
                },
            }
        })
        .collect())
}

/// `mode=content` — ADR-072 seq1 BM25-only full-text search over raw
/// `episodes.content` (`handlers::do_recall_content`), adapted into the SAME
/// `{id, content, score}` wire contract `mode=recall` uses:
/// `ContentPassage::episode_id` (stringified) -> `id`,
/// `ContentPassage::snippet` (now the FULL matched-episode body, not a
/// 32-token FTS5 excerpt — see `ContentPassage` doc) -> `content`.
#[cfg(feature = "content-search")]
async fn content_mode_results(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    let passages = handlers::do_recall_content(mem, params).await?;
    Ok(passages
        .into_iter()
        .map(|p| SearchResultWire {
            id: p.episode_id.to_string(),
            content: p.snippet,
            score: p.score,
            kind: SearchResultKindWire::Episode,
            // `id` already IS the episode id for this arm (stringified);
            // `source_episode_id` is reserved for `Fact` items whose `id` is
            // the FACT's own id, not an episode's — see doc comment on
            // `SearchResultWire::source_episode_id`.
            source_episode_id: None,
        })
        .collect())
}

/// Feature-off HARD-FAIL for `mode=content`: this bin was not built with
/// `--features content-search`, so there is no BM25 stream to serve. B1
/// fail-loud (2026-07-22): a silent fallback to `mode=recall` here served a
/// ~40%-surface entity-only result under an explicit content/hybrid request —
/// exactly the LoCoMo 13.9% disaster class (a fabricated garbage baseline that
/// LOOKED like a real answer). We now REFUSE the request (`InvalidParams`/422)
/// instead of degrading silently. Recall-mode requests are unaffected (they hit
/// the un-gated `recall_mode_results`). To serve content/hybrid, rebuild WITH
/// `--features content-search`.
#[cfg(not(feature = "content-search"))]
async fn content_mode_results(
    _mem: &Memory,
    _params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    tracing::error!(
        "mode=content requested but kremory-http was built WITHOUT the `content-search` \
         feature — refusing to serve a silently-degraded recall-only result (B1 fail-loud). \
         Rebuild with `--features content-search`."
    );
    Err(ApiError(ToolError::InvalidParams(
        "mode=content requires the `content-search` feature, but this server was built \
         without it. Rebuild with `--features content-search`."
            .to_string(),
    )))
}

/// `mode=hybrid` (the DEFAULT) — runs BOTH `mode=recall` and `mode=content`
/// and RRF-fuses them via [`rrf_merge`]. The Arch-1a fusion/fairness decision
/// (benchmark-completion-roadmap) was resolved by the 2026-07-21 LoCoMo
/// diagnostic: RRF fusion over the entity/fact recall stream + the BM25 content
/// stream. Note (measured): the two streams have disjoint id-spaces
/// (entity-ids vs episode-ids), so RRF ≈ the prior `naive_merge` on this
/// benchmark (both 70.4%); the lift over `recall`-only (40.2%) is from ADDING
/// the content stream.
///
/// DONE (TD-066 Increment 1, `.ai-docs/specs/td-066-recall-scoring-
/// foundation-spec-2026-07-21.md` §3): this fusion has been ported into
/// `core::search::rrf_fuse_with_content` and wired into `Memory::recall()`'s
/// `.raw()`/`execute()` terminals — the library `recall()` + MCP
/// `kremory_recall` now reach it too. This fn + [`rrf_merge`] are kept as
/// the REST bin's own (now-parallel, not load-bearing for the library/MCP
/// surfaces) implementation rather than deleted/refactored into a thin
/// caller of the core fn — see this module's top-level doc comment for why
/// (the LoCoMo bench harness drives `mode=recall`/`mode=content`/
/// `mode=hybrid` directly for a three-way surface comparison; collapsing
/// them mid-run risks breaking that comparison).
#[cfg(feature = "content-search")]
async fn hybrid_mode_results(
    mem: &Memory,
    params: RecallParams,
    rrf_k: usize,
) -> Result<Vec<SearchResultWire>, ApiError> {
    let k = params.k;
    let recall = recall_mode_results(mem, params.clone()).await?;
    let content = content_mode_results(mem, params).await?;
    // Cap the fused output. Without this, hybrid returned `recall ∪ content`
    // (~122 memories on LoCoMo conv0: recall's ~73 unioned with content's ~50)
    // regardless of the consumer's requested `k` — a context-budget flood.
    // `rrf_merge` orders by relevance, so truncating the tail drops noise
    // WITHOUT dropping surfaced answers (the answer ranks near the top). Honour
    // `k` when given; absent it, bound hybrid to at most its largest single arm
    // so the union never floods past what either mode alone would return.
    let cap = k.unwrap_or_else(|| recall.len().max(content.len()));
    let mut merged = rrf_merge(recall, content, rrf_k);
    merged.truncate(cap);
    Ok(merged)
}

/// Feature-off HARD-FAIL for `mode=hybrid` (the DEFAULT mode) — same B1
/// fail-loud rationale as `content_mode_results`'s feature-off arm. Because
/// hybrid is the default, a silent fallback here was the exact path that
/// produced the 13.9% LoCoMo baseline. Refuse the request rather than degrade.
#[cfg(not(feature = "content-search"))]
async fn hybrid_mode_results(
    _mem: &Memory,
    _params: RecallParams,
    _rrf_k: usize,
) -> Result<Vec<SearchResultWire>, ApiError> {
    tracing::error!(
        "mode=hybrid requested but kremory-http was built WITHOUT the `content-search` \
         feature — refusing to serve a silently-degraded recall-only result (B1 fail-loud). \
         Rebuild with `--features content-search`."
    );
    Err(ApiError(ToolError::InvalidParams(
        "mode=hybrid requires the `content-search` feature, but this server was built \
         without it. Rebuild with `--features content-search`."
            .to_string(),
    )))
}

/// NAIVE BASELINE FUSION — real fusion/fairness decision deferred to Arch-1a
/// post-diagnostic per benchmark-completion-roadmap. Union by `id`
/// (first-seen wins across the two ranked lists), rank-interleaved
/// (`recall[0], content[0], recall[1], content[1], ...`) — NOT an RRF or any
/// score-aware fusion.
#[cfg(feature = "content-search")]
/// RRF (Reciprocal Rank Fusion) of two ranked result streams — replaces the v0
/// `naive_merge` rank-interleave (which was explicitly a placeholder: "not
/// kremory's answer to hybrid ranking").
///
/// Rationale (`.ai-docs/research/v011-recall-redesign/W2-hybrid-scoring.md` +
/// the 2026-07-21 LoCoMo diagnostic, memory
/// `project_kremory_locomo_recall_root_cause_retrieval_surface`): RRF is the
/// tune-free dominant fusion across Elastic/Weaviate/Graphiti; kremory already
/// uses RRF_K=60 in `search.rs`. The naive 1:1 interleave DILUTED the content
/// stream — judged LoCoMo recall 70.4% for naive-hybrid vs 71.4% content-only;
/// RRF recovers to 71.4% (matches content-only) without letting the
/// LoCoMo-net-negative entity-graph stream dominate. `score(d) = Σ_list
/// 1/(RRF_K + rank_list(d))`, rank 1-based; dedup by id (a result present in
/// both streams accrues both contributions). Deterministic: fused-score desc,
/// then id asc on ties.
fn rrf_merge(
    a: Vec<SearchResultWire>,
    b: Vec<SearchResultWire>,
    rrf_k: usize,
) -> Vec<SearchResultWire> {
    // recall-improvement-e2e-spec-2026-07-22 §S0-infra (D3): `k` is now the
    // boot-read `KREMORY_RRF_K` value (AppState.rrf_k), NOT a hardcoded
    // `const RRF_K = 60.0`, so this bin-local hybrid-fusion sweep site tracks
    // the same k as the library fusion sites.
    let rrf_k = rrf_k as f32;
    let mut fused: std::collections::HashMap<String, SearchResultWire> =
        std::collections::HashMap::with_capacity(a.len() + b.len());
    for list in [a, b] {
        for (rank, r) in list.into_iter().enumerate() {
            let contrib = 1.0 / (rrf_k + (rank as f32) + 1.0);
            fused
                .entry(r.id.clone())
                .and_modify(|e| e.score += contrib)
                .or_insert_with(|| SearchResultWire {
                    score: contrib,
                    ..r
                });
        }
    }
    let mut merged: Vec<SearchResultWire> = fused.into_values().collect();
    merged.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.id.cmp(&y.id))
    });
    merged
}

// ────────────────────────────────────────────────────────────────────────
// DELETE /namespaces/{ns}
// ────────────────────────────────────────────────────────────────────────

async fn delete_namespace(
    State(state): State<AppState>,
    AxumPath(ns): AxumPath<String>,
) -> Result<impl IntoResponse, ApiError> {
    let namespace = kremory::Namespace::new(ns);
    let deleted = state
        .mem
        .forget()
        .in_namespace(namespace)
        .execute()
        .await
        .map_err(ToolError::from)?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

// ────────────────────────────────────────────────────────────────────────
// POST /consolidation/{cycle}
// ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ConsolidationQuery {
    /// FRICTION (flagged per task instructions): codemem's own harness posts
    /// `/consolidation/{cycle}` with NO namespace at all (global consolidation
    /// in codemem's data model) — but `kremory::Memory::dream()` is always
    /// namespace-scoped (ADR-048 three-signal local-first consistency check
    /// operates per-namespace). There is no lossless mapping from codemem's
    /// global-consolidation semantics to kremory's namespace-scoped `dream()`.
    ///
    /// FAIL-LOUD (Quinn HIGH): this param is REQUIRED, not defaulted. An
    /// earlier draft fell back to a single `"default"` namespace when omitted
    /// — but dream idempotency is keyed on `(namespace, batch_id)`, so every
    /// benchmark consolidation call would share `(default, <cycle>)` and only
    /// the FIRST would execute; the rest silently no-op and dream never
    /// touches real data. Missing `?namespace=` is now a loud 422 (see
    /// `run_consolidation`) rather than a silent no-op. `cycle` is threaded
    /// through as `batch_id` so repeated same-`(namespace, cycle)` calls are
    /// idempotent (`DreamParams::batch_id`).
    namespace: Option<String>,
}

async fn run_consolidation(
    State(state): State<AppState>,
    AxumPath(cycle): AxumPath<String>,
    Query(query): Query<ConsolidationQuery>,
) -> Result<impl IntoResponse, ApiError> {
    // Fail loud (422) on a missing namespace rather than silently defaulting
    // to a shared scope that would collapse dream's `(namespace, batch_id)`
    // idempotency into a single-run no-op for the rest of the benchmark.
    let namespace = query.namespace.filter(|ns| !ns.is_empty()).ok_or_else(|| {
        ApiError(ToolError::InvalidParams(
            "namespace query param required — consolidation is namespace-scoped in kremory \
             (POST /consolidation/{cycle}?namespace=<ns>)"
                .to_string(),
        ))
    })?;
    let params = DreamParams {
        namespace,
        thread: None,
        batch_id: Some(cycle),
    };
    let _summary = handlers::do_dream(&state.mem, params).await?;
    Ok(StatusCode::OK)
}

// ────────────────────────────────────────────────────────────────────────
// Router — factored out of `main` so the in-process test module below can
// build the exact same route table over a mock-provider `Memory`.
// ────────────────────────────────────────────────────────────────────────

/// Collapse a request path to its top-level route group (`/namespaces/conv0` →
/// `/namespaces`) so the latency metric's `route` label has bounded cardinality
/// (recall-v2 o11y / TD-132 — the per-conversation namespace is a cardinality
/// bomb if used raw).
fn route_label(path: &str) -> String {
    match path.split('/').nth(1) {
        Some(seg) if !seg.is_empty() => format!("/{seg}"),
        _ => "/".to_owned(),
    }
}

/// Per-request latency + TTFB observability (TD-132). For these non-streaming
/// JSON handlers request-duration IS the TTFB. Dual-emit per ratified ADR-D1:
/// a `metrics` histogram (rendered at `/metrics` when the `prometheus` feature
/// installs a recorder; a no-op otherwise) AND an always-on `tracing::info` so
/// recall latency is visible in the server log regardless. The metric follows
/// the canonical `kremory_core_*` scheme + `_seconds` base unit (R3).
async fn track_request_latency(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let route = route_label(&path);
    let start = std::time::Instant::now();
    let response = next.run(req).await;
    let latency_s = start.elapsed().as_secs_f64();
    let status = response.status().as_u16();
    metrics::histogram!(
        "kremory_core_http_request_duration_seconds",
        "route" => route.clone(),
        "method" => method.to_string(),
        "status" => status.to_string(),
    )
    .record(latency_s);
    tracing::info!(
        target: "kremory.http.request",
        %method,
        path = %path,
        route = %route,
        status,
        latency_ms = latency_s * 1000.0,
        "kremory.http.request_complete"
    );
    response
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/memories", post(create_memory))
        .route("/search", get(search))
        .route("/namespaces/{ns}", delete(delete_namespace))
        .route("/consolidation/{cycle}", post(run_consolidation))
        // Outermost app layer — times the WHOLE request (TD-132).
        .layer(middleware::from_fn(track_request_latency))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ────────────────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let mut env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // KREMORY_DEBUG=1: surface the raw-payload `tracing::debug!` dumps the
    // extraction parsers already emit (parsers.rs) — without this directive
    // the default `info` filter (or a `RUST_LOG` that doesn't mention this
    // target) silently drops them, which is exactly the gap that made the
    // 2026-07-20 empty-fact-graph bug hard to diagnose. This only adds
    // VERBOSITY; it never gates the always-on `rql.extraction.*` C2 alarm
    // metrics/warnings, which fire regardless of KREMORY_DEBUG.
    if std::env::var("KREMORY_DEBUG").is_ok() {
        match "kremory.extraction.parsers=debug".parse() {
            Ok(directive) => env_filter = env_filter.add_directive(directive),
            Err(e) => {
                tracing::warn!(error = %e, "failed to parse KREMORY_DEBUG env-filter directive")
            }
        }
    }
    let subscriber = FmtSubscriber::builder()
        .with_writer(std::io::stderr)
        .with_env_filter(env_filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow!("failed to install tracing subscriber: {e}"))?;

    // B1 fail-loud (2026-07-22): announce the content-search capability at boot so
    // an operator can NEVER unknowingly run a degraded (recall-only) server behind
    // the default `mode=hybrid`. Without the feature, content/hybrid requests now
    // hard-fail (422) rather than silently degrading — this banner is the paired
    // startup signal so the cause is visible before the first request.
    #[cfg(feature = "content-search")]
    tracing::info!(
        "kremory-http built WITH `content-search` — mode=content/hybrid available."
    );
    #[cfg(not(feature = "content-search"))]
    tracing::warn!(
        "kremory-http built WITHOUT `content-search` — mode=content and mode=hybrid \
         (the DEFAULT) will HARD-FAIL (422). Only mode=recall is servable. Rebuild with \
         `--features content-search` for hybrid/content recall."
    );

    let db_path = std::env::var("KREMORY_MCP_DB_PATH").map_err(|_| {
        anyhow!(
            "KREMORY_MCP_DB_PATH is required — set it to the path of the kremory \
             libSQL database (e.g. ./agent.db)"
        )
    })?;
    let ollama_url =
        std::env::var("KREMORY_MCP_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model_id = std::env::var("KREMORY_MCP_MODEL_ID").ok();
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    tracing::info!(
        db_path = %db_path,
        ollama_url = %ollama_url,
        port,
        model_id = model_id.as_deref().unwrap_or("gemma4:e4b (default)"),
        "kremory-http booting"
    );

    // Fail loud BEFORE constructing Memory — see `health.rs` module docs.
    health::check_reachable(&ollama_url, REACHABILITY_TIMEOUT)
        .await
        .map_err(|reason| {
            anyhow!(
                "Ollama unreachable at {ollama_url} ({reason}). Start Ollama \
                 (`ollama serve`) or set KREMORY_MCP_OLLAMA_URL to a reachable endpoint."
            )
        })?;

    // Benchmark-execution-spec §Phase-2: when an OpenAI-compatible cloud chat
    // endpoint is configured (e.g. Groq gpt-oss), use it for EXTRACTION (fast +
    // quality-comparable to the peers' gpt-4o-mini) while keeping embeddings on
    // local Ollama (Groq has no embeddings endpoint). Ollama must still be up
    // for the embedder — the reachability check above already enforced that.
    // Falls back to all-Ollama when the cloud env vars are absent.
    let chat_base_url = std::env::var("KREMORY_MCP_CHAT_BASE_URL").ok();
    let chat_api_key = std::env::var("KREMORY_MCP_CHAT_API_KEY").ok();
    let mem = match (chat_base_url, chat_api_key) {
        (Some(base_url), Some(key)) => {
            let chat_model = model_id
                .clone()
                .unwrap_or_else(|| "openai/gpt-oss-120b".to_string());
            tracing::info!(
                chat_base_url = %base_url,
                chat_model = %chat_model,
                "kremory-http: OpenAI-compatible cloud chat (extraction) + local Ollama embed"
            );
            kremory::facade::providers::with_openai_compatible_chat_ollama_embed(
                kremory::facade::providers::OpenAiCompatibleParams {
                    chat_base_url: &base_url,
                    api_key: &key,
                    chat_model: &chat_model,
                    ollama_url: &ollama_url,
                    path: &db_path,
                },
            )
            .await
            .with_context(|| format!("failed to open kremory Memory (cloud chat) at {db_path}"))?
        }
        _ => kremory::facade::providers::with_ollama_at_model(ollama_url, model_id, &db_path)
            .await
            .with_context(|| format!("failed to open kremory Memory at {db_path}"))?,
    };

    // TD-136 (dense episode retrieval): one-shot maintenance subcommand
    // `kremory-http backfill-episode-embeddings [batch_size]` — embed + store
    // `episodes.embedding` for the existing corpus (populating `episodes_vec_idx`
    // from Migration 026), then exit WITHOUT starting the HTTP server. Run this
    // against a COPY of the DB before measuring the dense arm. Requires the
    // `content-search` feature (the column + fn only exist there).
    if std::env::args().nth(1).as_deref() == Some("backfill-episode-embeddings") {
        #[cfg(feature = "content-search")]
        {
            let batch_size: usize = std::env::args()
                .nth(2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            tracing::info!(
                batch_size,
                "kremory-http: running episode-embedding backfill (TD-136), then exiting"
            );
            let stats = mem
                .backfill_episode_embeddings(batch_size)
                .await
                .context("episode-embedding backfill failed")?;
            tracing::info!(
                embedded = stats.embedded,
                failed = stats.failed,
                "kremory-http: episode-embedding backfill complete"
            );
            // Stdout line for the orchestrator to scrape (stderr carries tracing).
            println!(
                "backfill-episode-embeddings: embedded={} failed={}",
                stats.embedded, stats.failed
            );
            return Ok(());
        }
        #[cfg(not(feature = "content-search"))]
        {
            anyhow::bail!(
                "backfill-episode-embeddings requires a `content-search` build — \
                 rebuild with `--features content-search`"
            );
        }
    }

    // TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143): one-shot
    // maintenance subcommand `kremory-http reembed-episode-embeddings
    // [batch_size]` — re-embed EVERY episode's `content` (overwriting any
    // vector already stored), then exit WITHOUT starting the HTTP server.
    // Unlike `backfill-episode-embeddings` above (which only fills
    // NULL-embedding gaps and can NEVER touch a row that already has a
    // vector), this is the remedy for an embedding-CONFIG change — flipping
    // `KREMORY_EMBED_TASK_PREFIX`, swapping the embedder model, or changing
    // the embedding dimension all make every EXISTING stored embedding stale.
    // Run this against a COPY of the DB before measuring. Requires the
    // `content-search` feature (the column + fn only exist there).
    if std::env::args().nth(1).as_deref() == Some("reembed-episode-embeddings") {
        #[cfg(feature = "content-search")]
        {
            let batch_size: usize = std::env::args()
                .nth(2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            tracing::info!(
                batch_size,
                "kremory-http: running full episode-embedding re-embed (TD-143), then exiting"
            );
            let stats = mem
                .reembed_all_episode_embeddings(batch_size)
                .await
                .context("episode-embedding re-embed failed")?;
            tracing::info!(
                embedded = stats.embedded,
                failed = stats.failed,
                "kremory-http: episode-embedding re-embed complete"
            );
            // Stdout line for the orchestrator to scrape (stderr carries tracing).
            println!(
                "reembed-episode-embeddings: embedded={} failed={}",
                stats.embedded, stats.failed
            );
            return Ok(());
        }
        #[cfg(not(feature = "content-search"))]
        {
            anyhow::bail!(
                "reembed-episode-embeddings requires a `content-search` build — \
                 rebuild with `--features content-search`"
            );
        }
    }

    // TD-112 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-112): one-shot
    // maintenance subcommand `kremory-http reembed-entity-embeddings
    // [batch_size]` — re-embed EVERY entity's display name (overwriting any
    // vector already stored), then exit WITHOUT starting the HTTP server.
    // Sibling of `reembed-episode-embeddings` above — remedies BOTH a live
    // correctness bug (entity embeddings go stale after a dream-phase
    // merge/alias) and de-confounds a clean TD-143 A/B (that A/B previously
    // only re-embedded episodes, leaving entity/fact arms in a mismatched
    // task space — see TD-112's register entry). Requires the
    // `content-search` feature.
    if std::env::args().nth(1).as_deref() == Some("reembed-entity-embeddings") {
        #[cfg(feature = "content-search")]
        {
            let batch_size: usize = std::env::args()
                .nth(2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            tracing::info!(
                batch_size,
                "kremory-http: running full entity-embedding re-embed (TD-112), then exiting"
            );
            let stats = mem
                .reembed_all_entity_embeddings(batch_size)
                .await
                .context("entity-embedding re-embed failed")?;
            tracing::info!(
                embedded = stats.embedded,
                failed = stats.failed,
                "kremory-http: entity-embedding re-embed complete"
            );
            println!(
                "reembed-entity-embeddings: embedded={} failed={}",
                stats.embedded, stats.failed
            );
            return Ok(());
        }
        #[cfg(not(feature = "content-search"))]
        {
            anyhow::bail!(
                "reembed-entity-embeddings requires a `content-search` build — \
                 rebuild with `--features content-search`"
            );
        }
    }

    // TD-112: one-shot maintenance subcommand `kremory-http
    // reembed-fact-embeddings [batch_size]` — re-embed EVERY fact's `subject
    // predicate object` triple text (overwriting any vector already stored),
    // then exit WITHOUT starting the HTTP server. Run entity re-embed FIRST
    // when both are needed (facts resolve their subject/object text from the
    // CURRENT entity rows — `reembed-all-embeddings` below does this in the
    // right order). Requires the `content-search` feature.
    if std::env::args().nth(1).as_deref() == Some("reembed-fact-embeddings") {
        #[cfg(feature = "content-search")]
        {
            let batch_size: usize = std::env::args()
                .nth(2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            tracing::info!(
                batch_size,
                "kremory-http: running full fact-embedding re-embed (TD-112), then exiting"
            );
            let stats = mem
                .reembed_all_fact_embeddings(batch_size)
                .await
                .context("fact-embedding re-embed failed")?;
            tracing::info!(
                embedded = stats.embedded,
                failed = stats.failed,
                "kremory-http: fact-embedding re-embed complete"
            );
            println!(
                "reembed-fact-embeddings: embedded={} failed={}",
                stats.embedded, stats.failed
            );
            return Ok(());
        }
        #[cfg(not(feature = "content-search"))]
        {
            anyhow::bail!(
                "reembed-fact-embeddings requires a `content-search` build — \
                 rebuild with `--features content-search`"
            );
        }
    }

    // TD-112: convenience subcommand `kremory-http reembed-all-embeddings
    // [batch_size]` — runs all three bulk re-embeds (episode, entity, fact)
    // in sequence, in the dependency-correct order (entities BEFORE facts,
    // since fact text is resolved from entity rows), then exits. This is
    // what a clean full-corpus TD-143 A/B actually needs — re-embedding
    // episodes alone leaves entity/fact arms in a mismatched task space (the
    // exact confound TD-112 was filed to close). Requires `content-search`.
    if std::env::args().nth(1).as_deref() == Some("reembed-all-embeddings") {
        #[cfg(feature = "content-search")]
        {
            let batch_size: usize = std::env::args()
                .nth(2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            tracing::info!(
                batch_size,
                "kremory-http: running full episode+entity+fact re-embed (TD-112/TD-143), \
                 then exiting"
            );
            let episodes = mem
                .reembed_all_episode_embeddings(batch_size)
                .await
                .context("episode-embedding re-embed failed")?;
            let entities = mem
                .reembed_all_entity_embeddings(batch_size)
                .await
                .context("entity-embedding re-embed failed")?;
            let facts = mem
                .reembed_all_fact_embeddings(batch_size)
                .await
                .context("fact-embedding re-embed failed")?;
            tracing::info!(
                episodes_embedded = episodes.embedded,
                episodes_failed = episodes.failed,
                entities_embedded = entities.embedded,
                entities_failed = entities.failed,
                facts_embedded = facts.embedded,
                facts_failed = facts.failed,
                "kremory-http: full episode+entity+fact re-embed complete"
            );
            println!(
                "reembed-all-embeddings: episodes(embedded={} failed={}) \
                 entities(embedded={} failed={}) facts(embedded={} failed={})",
                episodes.embedded,
                episodes.failed,
                entities.embedded,
                entities.failed,
                facts.embedded,
                facts.failed
            );
            return Ok(());
        }
        #[cfg(not(feature = "content-search"))]
        {
            anyhow::bail!(
                "reembed-all-embeddings requires a `content-search` build — \
                 rebuild with `--features content-search`"
            );
        }
    }

    // TD-132 / ratified ADR-D2: the CONSUMER binary installs the recorder (the
    // `kremory` lib never does). A Prometheus PULL exporter (HTTP is scrapeable;
    // the sibling stdio bin uses a different channel — hence the `prometheus`
    // feature gate). This makes the already-emitted `kremory_core_*` metrics
    // (tokens, cost, request latency, recall/ingest/dream histograms) actually
    // render at `/metrics` instead of firing into silence (R1's #1 finding).
    #[cfg(feature = "prometheus")]
    let prom_handle = PrometheusBuilder::new()
        .install_recorder()
        .context("failed to install Prometheus recorder")?;

    // recall-improvement-e2e-spec-2026-07-22 §S0-infra (R2): read the
    // search-fusion SWEEP knobs at boot so weight/`k` sweeps cost a restart, not
    // a rebuild. `KREMORY_RRF_K` feeds the bin-local `rrf_merge` (hybrid arm);
    // the library fusion sites (`context.rs` entity RRF + `rrf_fuse_with_content`
    // content fusion) read the SAME env at Memory construction via
    // `facade::providers::search_env_overrides`. `KREMORY_CONTENT_WEIGHT` is
    // consumed by the library — read here only for the authoritative boot
    // banner. Fail-loud: a malformed `KREMORY_RRF_K` WARNs + falls back to 60
    // (never silently accepted as garbage).
    let rrf_k: usize = match std::env::var("KREMORY_RRF_K") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    value = %raw,
                    error = %e,
                    "KREMORY_RRF_K is not a valid usize — falling back to default 60"
                );
                60
            }
        },
        Err(_) => 60,
    };
    let content_weight_display =
        std::env::var("KREMORY_CONTENT_WEIGHT").unwrap_or_else(|_| "1.0 (default)".to_string());
    tracing::info!(
        rrf_k,
        content_stream_weight = %content_weight_display,
        "kremory-http: search-fusion sweep point — rrf_k applied to bin-local rrf_merge; \
         content_stream_weight applied to the Engine SearchConfig via search_env_overrides"
    );

    let app = build_router(AppState {
        mem: Arc::new(mem),
        rrf_k,
    });

    #[cfg(feature = "prometheus")]
    let app = app.route(
        "/metrics",
        get(move || {
            let handle = prom_handle.clone();
            async move { handle.render() }
        }),
    );

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("failed to bind 0.0.0.0:{port}"))?;

    tracing::info!(
        port,
        "kremory-http ready — serving GET /health, POST /memories, GET /search, \
         DELETE /namespaces/{{ns}}, POST /consolidation/{{cycle}}"
    );

    axum::serve(listener, app).await?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────
// Tests — REST-transport coverage. These live INSIDE the bin (not a
// `tests/*.rs` integration file) because `flatten_result_content`,
// `build_router`, `AppState` and the handler fns are all bin-private and an
// integration file links only against the LIB target. The mock-provider
// `Memory` path mirrors `tests/handler_roundtrip.rs::mock_memory`.
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
    #[cfg(feature = "content-search")]
    use kremory::core::provider::MockEmbeddingProvider;
    use kremory::{ChatProvider, DynEmbeddingProvider};
    use kremory_mcp::params::{RetrievedFactWire, SourceRefWire};
    #[cfg(feature = "content-search")]
    use std::collections::HashMap;
    use tower::ServiceExt as _;

    // ─── flatten_result_content (benchmark-load-bearing) ─────────────────

    fn fact_wire(fact: &str) -> RetrievedFactWire {
        RetrievedFactWire {
            fact: fact.to_string(),
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
            object_is_entity: false,
            valid_at: "2026-01-01T00:00:00+00:00".into(),
            invalid_at: None,
            recorded_at: "2026-01-01T00:00:00+00:00".into(),
            expired_at: None,
            confidence: 1.0,
            source_episode_ids: vec![1],
            score: 0.5,
        }
    }

    fn context_wire(
        name: &str,
        summary: &str,
        facts: Vec<RetrievedFactWire>,
    ) -> RetrievedContextWire {
        RetrievedContextWire {
            entity_id: name.to_lowercase(),
            entity_name: name.to_string(),
            summary: summary.to_string(),
            score: 0.9,
            incomplete: false,
            entity_type_id: 0,
            entity_type_name: "Entity".into(),
            namespace: None,
            source_refs: Vec::<SourceRefWire>::new(),
            facts,
        }
    }

    #[test]
    fn flatten_result_content_includes_summary_and_every_fact() {
        let ctx = context_wire(
            "Ada Lovelace",
            "a mathematician",
            vec![
                fact_wire("Ada Lovelace wrote the first algorithm"),
                fact_wire("Ada Lovelace collaborated with Charles Babbage"),
            ],
        );
        let content = flatten_result_content(&ctx);
        // The entity name + summary line must be present.
        assert!(
            content.contains("Ada Lovelace") && content.contains("a mathematician"),
            "flattened content must carry entity name + summary: {content:?}"
        );
        // EVERY connected fact's natural-language string must survive — the
        // benchmark substring scorer relies on this.
        assert!(
            content.contains("Ada Lovelace wrote the first algorithm"),
            "fact 1 must appear in flattened content: {content:?}"
        );
        assert!(
            content.contains("Ada Lovelace collaborated with Charles Babbage"),
            "fact 2 must appear in flattened content: {content:?}"
        );
    }

    #[test]
    fn flatten_result_content_with_no_facts_is_the_summary_line() {
        let ctx = context_wire("Grace Hopper", "a computer scientist", Vec::new());
        let content = flatten_result_content(&ctx);
        assert_eq!(content, "Grace Hopper: a computer scientist");
    }

    // ─── SearchResultWire::kind / source_episode_id (TD-139 measurement
    // prerequisite) — pure serialization, no live arm emits `Fact` yet ────

    /// Entity and Episode items (the two kinds every live arm emits today)
    /// serialize `source_episode_id` as `null` — proves the ADDITIVE fields
    /// don't perturb the existing `{id, content, score}` shape consumers
    /// (the LoCoMo harness) already read.
    #[test]
    fn search_result_wire_entity_and_episode_kinds_have_no_source_episode_id() {
        let entity = SearchResultWire {
            id: "e1".into(),
            content: "Ada Lovelace: a mathematician".into(),
            score: 0.9,
            kind: SearchResultKindWire::Entity,
            source_episode_id: None,
        };
        let episode = SearchResultWire {
            id: "42".into(),
            content: "Ada Lovelace wrote the first algorithm".into(),
            score: 0.7,
            kind: SearchResultKindWire::Episode,
            source_episode_id: None,
        };
        let ej = serde_json::to_value(&entity).expect("entity serializes");
        let pj = serde_json::to_value(&episode).expect("episode serializes");
        assert_eq!(ej["kind"], "entity", "entity kind: {ej}");
        assert!(ej["source_episode_id"].is_null(), "entity: {ej}");
        assert_eq!(pj["kind"], "episode", "episode kind: {pj}");
        assert!(pj["source_episode_id"].is_null(), "episode: {pj}");
    }

    /// A Fact-kind item — not yet emitted by any live path (TD-139 DoD item
    /// 2 is a later, separately-measured change) — carries its source
    /// episode id on the wire. This is the shape `bench/locomo/
    /// evidence_eval.py`'s fact-resolution path (Part B) depends on: proves
    /// the wire CAN carry the provenance before the arm that would populate
    /// it exists.
    #[test]
    fn search_result_wire_fact_kind_serializes_with_source_episode_id() {
        let fact = SearchResultWire {
            id: "fact-7".into(),
            content: "Caroline attended LGBTQ_support_group".into(),
            score: 0.8,
            kind: SearchResultKindWire::Fact,
            source_episode_id: Some(42),
        };
        let json = serde_json::to_value(&fact).expect("fact serializes");
        assert_eq!(json["kind"], "fact", "fact kind: {json}");
        assert_eq!(
            json["source_episode_id"], 42,
            "fact source_episode_id must round-trip: {json}"
        );
    }

    // ─── in-process HTTP round-trip over a mock-provider Memory ──────────

    async fn mock_memory() -> Arc<Memory> {
        let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        let mem = Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("in-memory Memory must build");
        Arc::new(mem)
    }

    /// Pin a mode-(c) fact directly through the shared handler so the entity
    /// is recall-findable with mock providers (TD-113 stamps the FTS name +
    /// embedding at PIN time — no live LLM / enrichment seam needed, same as
    /// `handler_roundtrip.rs::recall_structured_surfaces_pinned_fact...`).
    async fn pin_fact(mem: &Memory, namespace: &str, subject: &str) {
        let params = RememberParams {
            namespace: namespace.to_string(),
            thread: None,
            content: format!("{subject} wrote the first algorithm"),
            source_kind: Some(kremory_mcp::params::SourceKindWire::Note),
            source_id: Some("doc-1".into()),
            published_at: None,
            structured_facts: vec![kremory_mcp::params::StructuredFactWire {
                subject: subject.to_string(),
                predicate: "wrote".into(),
                object: "the first algorithm".into(),
                valid_at: None,
                invalid_at: None,
            }],
            skip_extraction: true,
        };
        handlers::do_remember(mem, params)
            .await
            .expect("pin must succeed");
    }

    async fn body_json(body: Body) -> serde_json::Value {
        let bytes = to_bytes(body, 1 << 20).await.expect("read body");
        serde_json::from_slice(&bytes).expect("body is JSON")
    }

    #[tokio::test]
    async fn http_roundtrip_memories_search_delete_consolidation() {
        let mem = mock_memory().await;
        let router = build_router(AppState {
            mem: mem.clone(),
            rrf_k: 60,
        });
        let ns = "ns-http";

        // POST /memories → 201 + non-empty {id}. (Extraction path + mock LLM
        // means this specific episode won't itself be recall-findable, but the
        // route contract — 201 + an id — is what's asserted here.)
        let post = Request::builder()
            .method("POST")
            .uri("/memories")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "content": "Ada Lovelace wrote the first algorithm.",
                    "namespace": ns,
                }))
                .unwrap(),
            ))
            .unwrap();
        let resp = router.clone().oneshot(post).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_json(resp.into_body()).await;
        assert!(
            json["id"].as_str().is_some_and(|s| !s.is_empty()),
            "POST /memories must return a non-empty id: {json}"
        );

        // Seed a recall-findable pinned fact so GET /search has a deterministic
        // hit with mock providers, then assert the flattened `content` carries
        // the fact text (the benchmark substring path, end-to-end through the
        // route + adapter).
        pin_fact(&mem, ns, "Zephyrine").await;
        // `mode=recall` is explicit here (not the default `hybrid`): under a
        // default-features build (no `content-search`), B1 fail-loud makes
        // `hybrid`/`content` a hard 422, so the servable roundtrip mode is
        // `recall`. `recall` returns the pinned entity/fact regardless of the
        // `content-search` feature, keeping this route-contract test
        // config-agnostic. The 422 fail-loud contract is asserted separately in
        // `hybrid_without_content_search_feature_hard_fails`.
        let search = Request::builder()
            .method("GET")
            .uri(format!("/search?q=Zephyrine&namespace={ns}&k=10&mode=recall"))
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");
        assert!(
            results.iter().any(|r| {
                r["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("Zephyrine"))
            }),
            "GET /search results[].content must contain the pinned subject: {json}"
        );
        // Each result carries the id/content/score contract shape.
        for r in results {
            assert!(
                r["id"].as_str().is_some(),
                "result.id must be a string: {r}"
            );
            assert!(
                r["content"].as_str().is_some(),
                "result.content must be a string: {r}"
            );
            assert!(
                r["score"].as_f64().is_some(),
                "result.score must be numeric: {r}"
            );
            // TD-139: `mode=recall` items carry no per-fact episode provenance
            // (the connected facts' own `source_episode_ids` are discarded by
            // `flatten_result_content`'s join — see the doc comment on
            // `SearchResultWire::source_episode_id`).
            //
            // ⚠️ AMENDED 2026-07-28 (ADR-078 / TD-154). This previously asserted
            // `kind == "entity"` for EVERY `mode=recall` item — which **encoded
            // the bug as the contract**. In a `content-search` build the fused
            // recall path also returns content passages, and one of them
            // (`"Episode #2: Zephyrine wrote the first algorithm"`) was being
            // reported as an `entity`. The mapper simply never matched
            // `"ContentPassage"`. The correct invariant is not "everything is an
            // entity" — it is "nothing on this path is a FACT unless the
            // dense-fact arm is on", which the sibling tests
            // `http_search_fact_dense_arm_{off,on}_*` already pin.
            assert!(
                matches!(r["kind"].as_str(), Some("entity") | Some("episode")),
                "mode=recall result.kind must be entity or episode (never fact with \
                 the dense-fact arm off): {r}"
            );
            assert!(
                r["source_episode_id"].is_null(),
                "mode=recall result.source_episode_id must be null (only Fact items carry it): {r}"
            );
        }

        // POST /consolidation/{cycle} WITHOUT ?namespace= → 422 (Quinn HIGH).
        let no_ns = Request::builder()
            .method("POST")
            .uri("/consolidation/creative")
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(no_ns).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "missing namespace must be a loud 422, not a silent no-op default"
        );
        let json = body_json(resp.into_body()).await;
        assert!(
            json["error"]
                .as_str()
                .is_some_and(|e| e.to_lowercase().contains("namespace")),
            "422 body must name the missing namespace param: {json}"
        );

        // POST /consolidation/{cycle}?namespace=ns → 200.
        let with_ns = Request::builder()
            .method("POST")
            .uri(format!("/consolidation/creative?namespace={ns}"))
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(with_ns).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // DELETE /namespaces/{ns} → 200 + {deleted}.
        let del = Request::builder()
            .method("DELETE")
            .uri(format!("/namespaces/{ns}"))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(del).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        assert!(
            json["deleted"].is_u64(),
            "DELETE must return a numeric deleted count: {json}"
        );
    }

    #[tokio::test]
    async fn health_returns_200() {
        let mem = mock_memory().await;
        let router = build_router(AppState { mem, rrf_k: 60 });
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// TD-135 / recall-improvement-e2e-spec-2026-07-22 §S0-infra: `GET /health`
    /// must report the server's ACTIVE scoring config + build-feature flags so
    /// the bench harness (`provenance.build_provenance`) stamps a FAITHFUL
    /// provenance record instead of re-reading the harness process's env (which
    /// can silently differ from the server's — the config-mismatch that produced
    /// a bogus benchmark number).
    ///
    /// The load-bearing assertion: a value set via the `KREMORY_CONTENT_WEIGHT`
    /// / `KREMORY_RRF_K` boot override appears in `/health`'s `scoring` block —
    /// proving the handler reads the LIVE `SearchConfig` (via
    /// `Memory::search_config`, which `open_graph` populates from these env
    /// overrides at construction) rather than a hardcoded default. nextest runs
    /// each test in its own process, so this env mutation is isolated (same
    /// pattern as `providers::search_env_overrides_apply_*`).
    #[tokio::test]
    async fn health_reports_active_scoring_config_and_features() {
        std::env::set_var("KREMORY_CONTENT_WEIGHT", "2.5");
        std::env::set_var("KREMORY_RRF_K", "42");
        // `open_graph` (reached through the builder in `mock_memory`) applies the
        // overrides to the Engine's `SearchConfig` at construction.
        let mem = mock_memory().await;
        std::env::remove_var("KREMORY_CONTENT_WEIGHT");
        std::env::remove_var("KREMORY_RRF_K");

        let router = build_router(AppState { mem, rrf_k: 42 });
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;

        // Build-feature booleans reflect the compiled build (cfg!-evaluated).
        assert_eq!(
            json["content_search"].as_bool(),
            Some(cfg!(feature = "content-search")),
            "/health content_search must reflect the compiled feature: {json}"
        );
        assert_eq!(
            json["rerank"].as_bool(),
            Some(cfg!(feature = "rerank")),
            "/health rerank must reflect the compiled feature: {json}"
        );
        assert_eq!(
            json["prometheus"].as_bool(),
            Some(cfg!(feature = "prometheus")),
            "/health prometheus must reflect the compiled feature: {json}"
        );

        // Scoring block reflects the LIVE (env-overridden) SearchConfig — NOT a
        // hardcoded default. This is the faithfulness guarantee TD-135 hinges on.
        let scoring = &json["scoring"];
        assert_eq!(
            scoring["content_stream_weight"].as_f64(),
            Some(2.5),
            "content_stream_weight must be the KREMORY_CONTENT_WEIGHT override (2.5), \
             proving /health reads the live SearchConfig: {json}"
        );
        assert_eq!(
            scoring["rrf_k"].as_u64(),
            Some(42),
            "rrf_k must be the KREMORY_RRF_K override (42): {json}"
        );
        // The remaining post-RRF axes must be present + numeric (defaults here).
        assert!(
            scoring["graph_degree_weight"].as_f64().is_some(),
            "graph_degree_weight must be present + numeric: {json}"
        );
        assert!(
            scoring["temporal_weight"].as_f64().is_some(),
            "temporal_weight must be present + numeric: {json}"
        );
    }

    /// B1 fail-loud invariant (2026-07-22): on a build WITHOUT `content-search`,
    /// an explicit `mode=hybrid`/`mode=content` request MUST hard-fail (422),
    /// never silently degrade to recall-only. Silent degradation here served a
    /// ~40%-surface answer under the DEFAULT `hybrid` mode and produced the
    /// LoCoMo 13.9% garbage baseline. Only compiled/relevant when the feature is
    /// OFF (with it ON, hybrid/content are servable and return 200 — covered by
    /// `http_search_mode_content_returns_bm25_passage`).
    #[cfg(not(feature = "content-search"))]
    #[tokio::test]
    async fn hybrid_without_content_search_feature_hard_fails() {
        let mem = mock_memory().await;
        let router = build_router(AppState { mem, rrf_k: 60 });
        let ns = "ns-http-faildude";
        for mode in ["hybrid", "content"] {
            let req = Request::builder()
                .method("GET")
                .uri(format!("/search?q=anything&namespace={ns}&k=10&mode={mode}"))
                .body(Body::empty())
                .unwrap();
            let resp = router.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNPROCESSABLE_ENTITY,
                "mode={mode} without the content-search feature must hard-fail \
                 422 (B1 fail-loud), not silently degrade to recall"
            );
            let json = body_json(resp.into_body()).await;
            assert!(
                json["error"]
                    .as_str()
                    .is_some_and(|e| e.to_lowercase().contains("content-search")),
                "422 body must name the missing content-search feature: {json}"
            );
        }
    }

    // ─── mode=content / mode=hybrid (benchmark-completion-roadmap W0.1) ──

    /// `GET /search?mode=content` reaches ADR-072 seq1's BM25 FTS5 stream
    /// (`handlers::do_recall_content`) rather than the entity/fact path —
    /// asserts a passage carrying the pinned subject's snippet comes back
    /// through the SAME `{id, content, score}` wire contract `mode=recall`
    /// uses.
    /// ADR-078 Phase A regression: on the FUSED recall path, a content-derived
    /// item must be reported as `kind: episode`, not `entity`.
    ///
    /// `recall_mode_results` matched only `entity_type_name == "Fact"` and
    /// defaulted everything else to `Entity`, while `core::search` tags a fused
    /// content passage `"ContentPassage"` (`search.rs:1995`) and
    /// `SearchResultKindWire::Episode` is documented as exactly that case. The
    /// mapping was simply missing, so a consumer was told VERBATIM EPISODE TEXT
    /// is a derived entity summary. Measured on conv0 before the fix: 3980 of
    /// 3980 items across the top-20 came back `entity`, zero `episode`.
    ///
    /// This drives the real HTTP path end-to-end rather than calling the mapper
    /// with a hand-built `RetrievedContext` — a pure-function test would assert
    /// the arithmetic of a mapping while remaining blind to whether the fused
    /// path reaches it at all (the TD-140 failure mode).
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_recall_mode_labels_content_items_as_episode() {
        let mem = mock_memory().await;
        let router = build_router(AppState {
            mem: mem.clone(),
            rrf_k: 60,
        });
        let ns = "ns-http-kind-fix";

        pin_fact(&mem, ns, "Zephyrine").await;

        // `mode=recall` (NOT `mode=content`) — the fused path the library's own
        // `recall()` uses and the one every benchmark measures.
        let search = Request::builder()
            .method("GET")
            .uri(format!("/search?q=Zephyrine&namespace={ns}&k=10&mode=recall"))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");
        assert!(!results.is_empty(), "expected fused results: {json}");

        let kinds: Vec<&str> = results.iter().filter_map(|r| r["kind"].as_str()).collect();
        assert_eq!(
            kinds.len(),
            results.len(),
            "every result must carry a `kind`: {json}"
        );
        assert!(
            kinds.contains(&"episode"),
            "the fused recall path must label content-derived items `episode`, not \
             collapse everything to `entity` — a consumer cannot otherwise tell \
             verbatim source text from a derived summary. got kinds={kinds:?}: {json}"
        );
    }

    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_mode_content_returns_bm25_passage() {
        let mem = mock_memory().await;
        let router = build_router(AppState {
            mem: mem.clone(),
            rrf_k: 60,
        });
        let ns = "ns-http-content";

        pin_fact(&mem, ns, "Zephyrine").await;

        let search = Request::builder()
            .method("GET")
            .uri(format!(
                "/search?q=Zephyrine&namespace={ns}&k=10&mode=content"
            ))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");
        assert!(
            results.iter().any(|r| {
                r["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("Zephyrine"))
            }),
            "mode=content results[].content must carry the pinned subject's BM25 snippet: {json}"
        );
        for r in results {
            assert!(
                r["id"].as_str().is_some(),
                "result.id must be a string: {r}"
            );
            assert!(
                r["score"].as_f64().is_some(),
                "result.score must be numeric: {r}"
            );
            // TD-139 measurement prerequisite: `mode=content` items are
            // EPISODE-kind (a BM25-matched passage), never Fact.
            assert_eq!(
                r["kind"].as_str(),
                Some("episode"),
                "mode=content result.kind must be \"episode\": {r}"
            );
            assert!(
                r["source_episode_id"].is_null(),
                "mode=content result.source_episode_id must be null (episode's own id is \
                 already `id`; only Fact items carry source_episode_id): {r}"
            );
        }
    }

    /// Real-substrate regression for the content-search AND->OR fallback
    /// ladder (`TemporalGraph::content_search`, `core/search.rs`).
    ///
    /// The sibling test above (`http_search_mode_content_returns_bm25_passage`)
    /// queries with a SINGLE word (`q=Zephyrine`) — AND-joining one token is
    /// indistinguishable from OR-joining one token, so that test cannot
    /// detect a regression in the AND->OR ladder (it was GREEN even when the
    /// underlying substrate silently returned empty for every multi-word,
    /// natural-language query — the exact failure the LoCoMo benchmark
    /// harness hit: 100% empty content-mode recalls on real questions).
    ///
    /// This test drives the REAL ingest pipeline (`pin_fact` -> `do_remember`,
    /// same production path `insert_episode_with_group` uses — not a
    /// hand-built fixture) then queries with a multi-word sentence whose
    /// tokens ("Did"/"invent"/"anything"/"remarkable") are ABSENT from the
    /// pinned content except "Zephyrine" — an AND-only match is impossible by
    /// construction, so a non-empty result here can only have come from the
    /// OR-fallback rung. Without the fallback (pre-fix `content_search`),
    /// this asserts and fails.
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_mode_content_natural_language_query_uses_or_fallback() {
        let mem = mock_memory().await;
        let router = build_router(AppState {
            mem: mem.clone(),
            rrf_k: 60,
        });
        let ns = "ns-http-content-nl";

        // Pinned content: "Zephyrine wrote the first algorithm" (see `pin_fact`).
        pin_fact(&mem, ns, "Zephyrine").await;

        let question = "Did Zephyrine invent anything remarkable";
        let search = Request::builder()
            .method("GET")
            .uri(format!(
                "/search?q={question}&namespace={ns}&k=10&mode=content",
                question = question.replace(' ', "%20")
            ))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");
        assert!(
            !results.is_empty(),
            "mode=content must rescue a multi-word natural-language query via the \
             AND->OR fallback ladder — an AND-only match is impossible here (query \
             tokens 'invent'/'anything'/'remarkable' are absent from the pinned \
             content). Empty here reproduces the LoCoMo-benchmark content-search bug: {json}"
        );
        assert!(
            results.iter().any(|r| {
                r["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("Zephyrine"))
            }),
            "OR-fallback result must still carry the pinned subject's snippet: {json}"
        );
    }

    // ─── TD-139 DoD item 3: dense fact arm end-to-end (real recall path) ──
    //
    // `pin_fact`'s caller-supplied `structured_facts` + `skip_extraction`
    // path does NOT populate `facts.embedding` — only the LLM-extraction
    // insert paths do (`ingest_with.rs:~1774`, `deferred.rs:~411`). So this
    // section drives REAL LLM-extraction ingest (a staged `MockChatProvider`,
    // mirrors `crates/kremory/tests/facade_fact_persistence_mock.rs::
    // staged_mock`) — the ONLY way to get a real, embedded fact through the
    // real `remember()` → deferred Phase 2 → `ingest_with` path, then queries
    // it back through the real `/search` REST route. Per the TD-140 lesson
    // (`instrument-real-data-flow-before-hypothesizing` §3), a hand-fed pure
    // function cannot prove the arm is actually WIRED — only a real
    // end-to-end run can.

    /// Staged mock for `IntegerIdLlmExtractor`'s 3 prompt stages (substring-
    /// keyed on prompt BOILERPLATE, content-agnostic — same keys
    /// `facade_fact_persistence_mock.rs::staged_mock` uses), producing ONE
    /// fact triple: `Priya relocated_to Berlin`. The embedded fact text
    /// (`ingest_with.rs`'s `format!("{subject} {predicate} {object}")`) is
    /// therefore exactly [`FACT_DENSE_QUERY`] — querying with that identical
    /// string gives `MockEmbeddingProvider` (hash-based: same text -> same
    /// vector) a PERFECT cosine match, deterministically surfacing this fact
    /// as the dense arm's top hit without a real semantic model.
    #[cfg(feature = "content-search")]
    fn fact_dense_staged_mock() -> MockChatProvider {
        let mut map = HashMap::new();
        map.insert(
            "Each entity must appear exactly once".to_string(),
            r#"{"entities":[{"name":"Priya","entity_type_id":1},{"name":"Berlin","entity_type_id":2}]}"#
                .to_string(),
        );
        map.insert(
            "Output a JSON array of relationship name strings.".to_string(),
            r#"["relocated_to"]"#.to_string(),
        );
        map.insert(
            "Output a concise JSON array of objects with".to_string(),
            r#"[{"subject":"Priya","predicate":"relocated_to","object":"Berlin","is_entity_ref":true,"confidence":0.95}]"#
                .to_string(),
        );
        map.insert(
            "Are these two entities".to_string(),
            "\"different\"".to_string(),
        );
        map.insert(
            "Output a JSON array of index numbers".to_string(),
            "[]".to_string(),
        );
        MockChatProvider::new(map)
    }

    #[cfg(feature = "content-search")]
    const FACT_DENSE_QUERY: &str = "Priya relocated_to Berlin";
    #[cfg(feature = "content-search")]
    const FACT_DENSE_NS: &str = "ns-fact-dense";

    /// Builds a `Memory` via the staged-mock LLM extraction path, `remember`s
    /// one episode, waits for Phase 2 to land the embedded fact, and returns
    /// `(mem, episode_id)`.
    #[cfg(feature = "content-search")]
    async fn build_fact_dense_mem(fact_dense_enabled: bool) -> (Arc<Memory>, i64) {
        let llm: Arc<dyn ChatProvider> = Arc::new(fact_dense_staged_mock());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));
        let mem = kremory::Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .with_fact_dense_enabled(fact_dense_enabled)
            .await
            .expect("Memory::open with fact-dense knob");

        let commit = handlers::do_remember(
            &mem,
            RememberParams {
                namespace: FACT_DENSE_NS.to_string(),
                thread: None,
                content: "Priya relocated to Berlin last year.".to_string(),
                source_kind: Some(kremory_mcp::params::SourceKindWire::Chat),
                source_id: Some("mock-session".into()),
                published_at: None,
                structured_facts: Vec::new(),
                skip_extraction: false,
            },
        )
        .await
        .expect("do_remember with real LLM-extraction path");

        let episode_id: i64 = commit
            .episode_entity_id
            .parse()
            .expect("episode_entity_id must parse to an i64 rowid");
        mem.wait_for_processing(episode_id, std::time::Duration::from_secs(30))
            .await
            .expect("wait_for_processing must complete (Phase 2 lands the embedded fact)");

        (Arc::new(mem), episode_id)
    }

    /// TD-139 DoD item 3: with `fact_dense_enabled = true`, the fact
    /// surfaces through the REAL `/search?mode=recall` route as
    /// `kind: "fact"` with the CORRECT `source_episode_id` — driven through
    /// the real `Memory` → `fuse_content_stream` → `rrf_fuse_with_facts` →
    /// `handlers::do_recall` → `recall_mode_results` chain, not a hand-fed
    /// pure function.
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_fact_dense_arm_on_surfaces_kind_fact_with_source_episode_id() {
        let (mem, episode_id) = build_fact_dense_mem(true).await;
        let router = build_router(AppState { mem, rrf_k: 60 });

        let search = Request::builder()
            .method("GET")
            .uri(format!(
                "/search?q={q}&namespace={FACT_DENSE_NS}&k=10&mode=recall",
                q = FACT_DENSE_QUERY.replace(' ', "%20")
            ))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");

        let fact_hit = results
            .iter()
            .find(|r| r["kind"].as_str() == Some("fact"))
            .unwrap_or_else(|| {
                panic!("fact_dense_enabled=true must surface a kind:\"fact\" result: {json}")
            });
        assert_eq!(
            fact_hit["source_episode_id"].as_i64(),
            Some(episode_id),
            "kind:\"fact\" result must carry the CORRECT source_episode_id: {fact_hit}"
        );
    }

    /// TD-139 DoD item 3, the gate half: with `fact_dense_enabled = false`
    /// (the default), the SAME fact — embedded identically, same query — is
    /// NEVER surfaced as `kind: "fact"`. Proves the knob gates the arm rather
    /// than the arm always firing regardless of config.
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_fact_dense_arm_off_never_surfaces_kind_fact() {
        let (mem, _episode_id) = build_fact_dense_mem(false).await;
        let router = build_router(AppState { mem, rrf_k: 60 });

        let search = Request::builder()
            .method("GET")
            .uri(format!(
                "/search?q={q}&namespace={FACT_DENSE_NS}&k=10&mode=recall",
                q = FACT_DENSE_QUERY.replace(' ', "%20")
            ))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");
        assert!(
            !results.iter().any(|r| r["kind"].as_str() == Some("fact")),
            "fact_dense_enabled=false (default) must NEVER surface a kind:\"fact\" result: {json}"
        );
    }

    /// `GET /search` with no `?mode=` now defaults to `mode=hybrid` (RRF fusion
    /// of the entity/fact recall stream + the BM25 content stream), changed from
    /// the pre-W0.1 `recall` default per the 2026-07-21 LoCoMo diagnostic
    /// (entity-graph `recall` judged 40.2% vs 71.4% hybrid). Guard: the default
    /// arm reaches the fused surface and still finds a pinned entity.
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_default_mode_is_hybrid() {
        let mem = mock_memory().await;
        let router = build_router(AppState {
            mem: mem.clone(),
            rrf_k: 60,
        });
        let ns = "ns-http-default-mode";

        pin_fact(&mem, ns, "Ada").await;

        let search = Request::builder()
            .method("GET")
            .uri(format!("/search?q=Ada&namespace={ns}&k=10"))
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(search).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        let results = json["results"].as_array().expect("results array");
        assert!(
            results
                .iter()
                .any(|r| r["content"].as_str().is_some_and(|c| c.contains("Ada"))),
            "omitted ?mode= must default to hybrid and still find the pinned entity: {json}"
        );
    }

    /// Test-only `SearchResultWire` builder for the `rrf_merge` tests below,
    /// which exercise fusion arithmetic and are indifferent to `kind`/
    /// `source_episode_id` — both new TD-139 fields default to the values an
    /// `Entity`-arm result would carry (`rrf_merge`'s `..r` spread passes
    /// them through unchanged regardless).
    #[cfg(feature = "content-search")]
    fn sr(id: &str, content: &str, score: f32) -> SearchResultWire {
        SearchResultWire {
            id: id.into(),
            content: content.into(),
            score,
            kind: SearchResultKindWire::Entity,
            source_episode_id: None,
        }
    }

    /// `rrf_merge` — Reciprocal Rank Fusion. A result in BOTH streams accrues
    /// both `1/(k+rank)` contributions and outranks single-stream hits; ties
    /// break by id asc; the first-inserted (recall stream, processed first)
    /// copy wins the content dedup while the content-stream duplicate only adds
    /// to the fused score.
    #[cfg(feature = "content-search")]
    #[test]
    fn rrf_merge_fuses_by_reciprocal_rank() {
        let recall = vec![sr("a", "recall-a", 0.9), sr("b", "recall-b", 0.8)];
        let content = vec![sr("b", "content-b", 0.7), sr("c", "content-c", 0.6)];
        let merged = rrf_merge(recall, content, 60);
        let ids: Vec<&str> = merged.iter().map(|r| r.id.as_str()).collect();
        // b ∈ both → 1/(60+2)+1/(60+1) ≈ 0.0325 (top). a (recall rank0) 1/61 ≈
        // 0.01639 edges c (content rank1) 1/62 ≈ 0.01613; id-asc tiebreak is
        // moot here since the scores differ.
        assert_eq!(ids, vec!["b", "a", "c"], "RRF fused order: {ids:?}");
        // recall's copy is inserted first (recall list processed first); the
        // content duplicate only adds to the score via `and_modify`.
        assert_eq!(
            merged.iter().find(|r| r.id == "b").unwrap().content,
            "recall-b",
            "first-inserted (recall) copy wins the dedup; content dup only adds score"
        );
        // the dual-stream hit must strictly outrank both single-stream hits.
        assert!(
            merged[0].id == "b" && merged[0].score > merged[1].score,
            "dual-stream result must outrank single-stream: {merged:?}"
        );
    }

    /// recall-improvement-e2e-spec-2026-07-22 §S0-infra (D3/R4a): the bin-local
    /// `rrf_merge` (fusion site 3 of 3) reads its RRF `k` from the argument
    /// (boot `KREMORY_RRF_K` → `AppState.rrf_k`), NOT a hardcoded const. A
    /// different `k` must produce different fused scores on identical input,
    /// proving the value flows through rather than being ignored.
    #[cfg(feature = "content-search")]
    #[test]
    fn rrf_merge_reads_k_argument_not_const() {
        let input = || (vec![sr("x", "x", 0.9)], vec![sr("y", "y", 0.8)]);
        let (ra, rb) = input();
        let k60 = rrf_merge(ra, rb, 60);
        let (ra, rb) = input();
        let k1 = rrf_merge(ra, rb, 1);
        // rank-0 contribution is 1/(k+0+1): k=60 → 1/61 ≈ 0.0164; k=1 → 1/2 = 0.5.
        let score_x_k60 = k60.iter().find(|r| r.id == "x").unwrap().score;
        let score_x_k1 = k1.iter().find(|r| r.id == "x").unwrap().score;
        assert!(
            (score_x_k60 - (1.0 / 61.0)).abs() < f32::EPSILON,
            "k=60 must yield 1/61 for a rank-0 hit, got {score_x_k60}"
        );
        assert!(
            (score_x_k1 - 0.5).abs() < f32::EPSILON,
            "k=1 must yield 1/2 for a rank-0 hit, got {score_x_k1}"
        );
        assert!(
            score_x_k1 > score_x_k60,
            "a smaller k must raise the fused score — proves rrf_merge reads its k arg"
        );
    }

    // ── parse_rerank_k (KREMORY_RERANK_K boot override) ───────────────────
    //
    // Gap named in the TD-066/TD-062 reranker hardening pass: the boot
    // override was previously inline in the `RERANK_K` `LazyLock`, which is
    // read exactly once per process and therefore untestable by
    // construction — a malformed `KREMORY_RERANK_K` would silently disable
    // reranking and no test could ever have caught it before it reached
    // `/health`'s `rerank_k` provenance stamp. `parse_rerank_k` is the
    // extracted pure function; these tests exercise every branch directly.

    /// Absent env var, and every value `usize::from_str` accepts, parse
    /// exactly as documented on `parse_rerank_k` / the `RERANK_K` doc
    /// comment. `"0"` is PINNED here, not rejected: `usize::from_str` has no
    /// notion of "zero is invalid" and `parse_rerank_k` adds no additional
    /// range check, so `KREMORY_RERANK_K=0` boots with reranking "requested"
    /// at k=0 (a real, if unusual, config — `apply_rerank_with` then reranks
    /// an empty head and is a behavioural no-op, see
    /// `apply_rerank_real_path_tests` in `kremory::facade::recall`). If this
    /// assertion ever changes, it means `parse_rerank_k` grew a deliberate
    /// "reject zero" rule — update this pin, don't just delete it.
    #[test]
    fn parse_rerank_k_accepts_absent_and_valid_values() {
        assert_eq!(
            parse_rerank_k(None),
            None,
            "absent KREMORY_RERANK_K must disable reranking"
        );
        assert_eq!(
            parse_rerank_k(Some("50")),
            Some(50),
            "a plain valid usize must parse through"
        );
        assert_eq!(
            parse_rerank_k(Some(" 50 ")),
            Some(50),
            "surrounding whitespace must be trimmed before parsing (operators hand-editing \
             a .env file routinely leave stray whitespace)"
        );
        assert_eq!(
            parse_rerank_k(Some("0")),
            Some(0),
            "PINNED: \"0\" parses to Some(0), not None — parse_rerank_k performs no \
             zero-rejection range check, only usize parsing"
        );
    }

    /// Malformed, negative, and empty values must all disable reranking
    /// (`None`) with a WARN, never silently accept garbage or panic. Rule 21
    /// (parse LLM/config output loudly, never silently default) — a
    /// misconfigured boot override must fail toward "reranking off"
    /// (pre-TD-134 behaviour), the same safe default as the var being
    /// entirely absent.
    #[test]
    fn parse_rerank_k_rejects_malformed_negative_and_empty_values() {
        assert_eq!(
            parse_rerank_k(Some("abc")),
            None,
            "non-numeric value must disable reranking, not panic"
        );
        assert_eq!(
            parse_rerank_k(Some("-1")),
            None,
            "negative value must disable reranking — usize has no negative representation"
        );
        assert_eq!(
            parse_rerank_k(Some("")),
            None,
            "empty (but present) value must disable reranking, matching the malformed-value \
             path rather than being treated as a distinct absent-value case"
        );
    }
}
