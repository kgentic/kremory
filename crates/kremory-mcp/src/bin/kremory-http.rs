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
//! | `GET /search?q=&namespace=&k=&mode=` | `handlers::do_recall` / `do_recall_content` | `200 {"results":[{"id","content","score"}]}` |
//!
//! `mode` (benchmark-completion-roadmap W0.1) selects which of kremory's
//! retrieval surfaces `/search` reaches: `recall` (default, byte-identical
//! to pre-W0.1 `/search`) is the existing hybrid keyword+semantic+graph path;
//! `content` is ADR-072 seq1's BM25-only full-text search over raw
//! `episodes.content` (requires this bin built with `--features
//! content-search`, else degrades to `recall` with a warning); `hybrid` runs
//! both and naively unions them (see `naive_merge` below — a real
//! fusion/fairness decision is deferred to Arch-1a post-diagnostic).
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
}

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

async fn health_check() -> StatusCode {
    StatusCode::OK
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
    #[default]
    Recall,
    Content,
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

#[derive(Debug, Serialize)]
struct SearchResultWire {
    id: String,
    content: String,
    score: f32,
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
    };
    let results = match query.mode {
        SearchMode::Recall => recall_mode_results(&state.mem, params).await?,
        SearchMode::Content => content_mode_results(&state.mem, params).await?,
        SearchMode::Hybrid => hybrid_mode_results(&state.mem, params).await?,
    };
    Ok(Json(SearchResponseWire { results }))
}

/// `mode=recall` (the default) — the existing entity-shaped hybrid
/// keyword/semantic/graph path (`handlers::do_recall`, structured format),
/// flattened via [`flatten_result_content`]. Byte-identical to `/search`'s
/// pre-W0.1 behaviour.
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
        .map(|r| SearchResultWire {
            id: r.entity_id.clone(),
            content: flatten_result_content(r),
            score: r.score,
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
        })
        .collect())
}

/// Feature-off degrade for `mode=content`: this bin was not built with
/// `--features content-search`, so there is no BM25 stream to serve.
/// Falls back to `mode=recall` with a loud warning rather than a hard
/// error — the harness is expected to build this bin WITH the feature when
/// it wants content/hybrid modes; this is a defensive fallback so a
/// feature-off build still answers `/search` instead of 500ing.
#[cfg(not(feature = "content-search"))]
async fn content_mode_results(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    tracing::warn!(
        "mode=content requested but kremory-http was built without the `content-search` \
         feature; falling back to mode=recall"
    );
    recall_mode_results(mem, params).await
}

/// `mode=hybrid` — runs BOTH `mode=recall` and `mode=content` and naively
/// unions them via [`naive_merge`]. NAIVE BASELINE FUSION — real
/// fusion/fairness decision deferred to Arch-1a post-diagnostic per
/// benchmark-completion-roadmap; do not read this as kremory's answer to
/// hybrid ranking.
#[cfg(feature = "content-search")]
async fn hybrid_mode_results(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    let recall = recall_mode_results(mem, params.clone()).await?;
    let content = content_mode_results(mem, params).await?;
    Ok(naive_merge(recall, content))
}

/// Feature-off degrade for `mode=hybrid` — same rationale as
/// `content_mode_results`'s feature-off arm.
#[cfg(not(feature = "content-search"))]
async fn hybrid_mode_results(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    tracing::warn!(
        "mode=hybrid requested but kremory-http was built without the `content-search` \
         feature; falling back to mode=recall"
    );
    recall_mode_results(mem, params).await
}

/// NAIVE BASELINE FUSION — real fusion/fairness decision deferred to Arch-1a
/// post-diagnostic per benchmark-completion-roadmap. Union by `id`
/// (first-seen wins across the two ranked lists), rank-interleaved
/// (`recall[0], content[0], recall[1], content[1], ...`) — NOT an RRF or any
/// score-aware fusion.
#[cfg(feature = "content-search")]
fn naive_merge(a: Vec<SearchResultWire>, b: Vec<SearchResultWire>) -> Vec<SearchResultWire> {
    let mut seen = std::collections::HashSet::with_capacity(a.len() + b.len());
    let mut merged = Vec::with_capacity(a.len() + b.len());
    let mut ia = a.into_iter();
    let mut ib = b.into_iter();
    loop {
        let ra = ia.next();
        let rb = ib.next();
        if ra.is_none() && rb.is_none() {
            break;
        }
        if let Some(x) = ra {
            if seen.insert(x.id.clone()) {
                merged.push(x);
            }
        }
        if let Some(x) = rb {
            if seen.insert(x.id.clone()) {
                merged.push(x);
            }
        }
    }
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

    let app = build_router(AppState { mem: Arc::new(mem) });

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
    use kremory::{ChatProvider, DynEmbeddingProvider};
    use kremory_mcp::params::{RetrievedFactWire, SourceRefWire};
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
        let router = build_router(AppState { mem: mem.clone() });
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
        let search = Request::builder()
            .method("GET")
            .uri(format!("/search?q=Zephyrine&namespace={ns}&k=10"))
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
        let router = build_router(AppState { mem });
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ─── mode=content / mode=hybrid (benchmark-completion-roadmap W0.1) ──

    /// `GET /search?mode=content` reaches ADR-072 seq1's BM25 FTS5 stream
    /// (`handlers::do_recall_content`) rather than the entity/fact path —
    /// asserts a passage carrying the pinned subject's snippet comes back
    /// through the SAME `{id, content, score}` wire contract `mode=recall`
    /// uses.
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_mode_content_returns_bm25_passage() {
        let mem = mock_memory().await;
        let router = build_router(AppState { mem: mem.clone() });
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
        let router = build_router(AppState { mem: mem.clone() });
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

    /// `GET /search` with no `?mode=` is `mode=recall` — byte-identical to
    /// pre-W0.1 `/search` behaviour. Regression guard for the W0.1 addition:
    /// the default arm must not have drifted when `content-search` is
    /// compiled in.
    #[cfg(feature = "content-search")]
    #[tokio::test]
    async fn http_search_default_mode_is_recall() {
        let mem = mock_memory().await;
        let router = build_router(AppState { mem: mem.clone() });
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
            "omitted ?mode= must default to recall and still find the pinned entity: {json}"
        );
    }

    /// `naive_merge` — dedupes by `id` (first-seen-in-the-interleave wins)
    /// and interleaves by rank (`recall[0], content[0], recall[1],
    /// content[1], ...`), not any score-aware fusion.
    #[cfg(feature = "content-search")]
    #[test]
    fn naive_merge_dedupes_by_id_and_interleaves_by_rank() {
        let recall = vec![
            SearchResultWire {
                id: "a".into(),
                content: "recall-a".into(),
                score: 0.9,
            },
            SearchResultWire {
                id: "b".into(),
                content: "recall-b".into(),
                score: 0.8,
            },
        ];
        let content = vec![
            SearchResultWire {
                id: "b".into(),
                content: "content-b".into(),
                score: 0.7,
            },
            SearchResultWire {
                id: "c".into(),
                content: "content-c".into(),
                score: 0.6,
            },
        ];
        let merged = naive_merge(recall, content);
        let ids: Vec<&str> = merged.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["a", "b", "c"],
            "expected deduped, rank-interleaved id order: {ids:?}"
        );
        // "b" is duplicated across both lists; the interleave reaches
        // content's "b" (content[0], processed in round 1 alongside
        // recall[0]) BEFORE it reaches recall's own "b" (recall[1],
        // round 2) — so content's copy wins the dedup, not recall's.
        assert_eq!(
            merged.iter().find(|r| r.id == "b").unwrap().content,
            "content-b",
            "first-seen-in-the-interleave copy must win the dedup"
        );
    }
}
