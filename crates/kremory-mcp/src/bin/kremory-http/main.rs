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
//! - `KREMORY_MCP_HOST` — bind address, default `127.0.0.1` (loopback). This
//!   binary has NO inbound authentication and serves `DELETE /namespaces/{ns}`,
//!   so binding a non-loopback address exposes unauthenticated namespace erasure
//!   to anyone who can reach the port. Set it only behind an authenticating proxy.
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
//! additive fields, layered over the pre-existing
//! `{id, content, score}` contract — see [`SearchResultKindWire`] and
//! [`SearchResultWire::source_episode_id`].
//!
//! ⚠️ **`"fact"` is emitted ONLY when `SearchConfig::fact_dense_enabled` is on,
//! which is DEFAULT-OFF** (see the note at `SearchResultKindWire` below, and
//! RECALL-LEDGER §4.3 — the fact-dense arm measured −0.7 nDCG / −1.3 MRR, hence
//! off by default). This line previously read "No live arm emits `fact` yet",
//! which was stale once the dense-fact arm was wired in and actively
//! misleading: it reads as "facts can never reach a consumer" when the truth
//! is "not on the shipped default".
//!
//! **Consequence worth knowing before designing any benchmark**: under the
//! default config the harness receives ZERO `kind == "fact"` rows, so facts —
//! and everything carried on them, including `valid_at` — are invisible to
//! EVERY bench scorer (substring, `evidence_eval`, and `qa_eval --structured`
//! alike). Measured on a full conv0 run: 25 `episode` + 25 `entity`
//! provenance rows per question, 0 `fact`. A lever whose effect lives on a fact
//! cannot be measured without turning this knob on first.
//!
//! `mode` selects which of kremory's
//! retrieval surfaces `/search` reaches: `recall` is the existing entity/fact
//! hybrid keyword+semantic+graph path; `content` is the BM25-only
//! full-text search over raw `episodes.content` (requires this bin built with
//! `--features content-search`; otherwise `mode=content`/`mode=hybrid` HARD-FAIL
//! 422 rather than silently degrading — B1 fail-loud, see `content_mode_results`);
//! `hybrid` runs both and RRF-fuses them (see `rrf_merge` below). `hybrid` is
//! the DEFAULT since a LoCoMo diagnostic run (entity-graph `recall`
//! alone judged 40.2% vs 70.4% hybrid).
//!
//! The fusion this file pioneered has been ported
//! into `core::search::rrf_fuse_with_content` and wired into
//! `Memory::recall()`'s `.raw()`/`execute()` terminals by default whenever
//! `content-search` is compiled in — so `recall_mode_results` below (which
//! calls `handlers::do_recall` → `.raw()`) is **no longer entity-graph-only**
//! in a `content-search`-enabled build; it now collaterally receives the
//! SAME fusion `mode=hybrid` does.
//!
//! ⚠️ **CORRECTION.** The sentence that used to
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
//! Do not treat the two modes as interchangeable, and note
//! that any benchmark which omits `?mode=` is measuring HYBRID, not the
//! library path (an early LoCoMo harness run made exactly this mistake).
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
//! | `DELETE /namespaces/{ns}` | `Memory::forget` | `200 {entities,facts,episodes,edges,is_empty}` (TD-247) |
//! | `POST /consolidation/{cycle}?namespace=` | `handlers::do_dream` | `200` (422 if `?namespace=` omitted) |

mod fusion;
mod handlers;
mod wire;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::Router;
#[cfg(feature = "prometheus")]
use metrics_exporter_prometheus::PrometheusBuilder;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

use kremory_mcp::health;

use crate::handlers::{create_memory, delete_namespace, health_check, run_consolidation, search};
use crate::wire::AppState;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const REACHABILITY_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_PORT: u16 = 3179;

// ────────────────────────────────────────────────────────────────────────
// Router — factored out of `main` so the in-process test module below can
// build the exact same route table over a mock-provider `Memory`.
// ────────────────────────────────────────────────────────────────────────

/// Collapse a request path to its top-level route group (`/namespaces/conv0` →
/// `/namespaces`) so the latency metric's `route` label has bounded cardinality
/// — the per-conversation namespace is a cardinality
/// bomb if used raw.
fn route_label(path: &str) -> String {
    match path.split('/').nth(1) {
        Some(seg) if !seg.is_empty() => format!("/{seg}"),
        _ => "/".to_owned(),
    }
}

/// Per-request latency + TTFB observability. For these non-streaming
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
        // Outermost app layer — times the WHOLE request.
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
    // target) silently drops them, which is exactly the gap that made an
    // empty-fact-graph bug hard to diagnose. This only adds
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

    // B1 fail-loud: announce the content-search capability at boot so
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

    // Dense episode retrieval: one-shot maintenance subcommand
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
                "kremory-http: running episode-embedding backfill, then exiting"
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

    // One-shot
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
                "kremory-http: running full episode-embedding re-embed, then exiting"
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

    // One-shot
    // maintenance subcommand `kremory-http reembed-entity-embeddings
    // [batch_size]` — re-embed EVERY entity's display name (overwriting any
    // vector already stored), then exit WITHOUT starting the HTTP server.
    // Sibling of `reembed-episode-embeddings` above — remedies BOTH a live
    // correctness bug (entity embeddings go stale after a dream-phase
    // merge/alias) and de-confounds a clean episode-re-embed A/B (that A/B previously
    // only re-embedded episodes, leaving entity/fact arms in a mismatched
    // task space). Requires the
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
                "kremory-http: running full entity-embedding re-embed, then exiting"
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

    // One-shot maintenance subcommand `kremory-http
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
                "kremory-http: running full fact-embedding re-embed, then exiting"
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

    // Convenience subcommand `kremory-http reembed-all-embeddings
    // [batch_size]` — runs all three bulk re-embeds (episode, entity, fact)
    // in sequence, in the dependency-correct order (entities BEFORE facts,
    // since fact text is resolved from entity rows), then exits. This is
    // what a clean full-corpus re-embed A/B actually needs — re-embedding
    // episodes alone leaves entity/fact arms in a mismatched task space.
    // Requires `content-search`.
    if std::env::args().nth(1).as_deref() == Some("reembed-all-embeddings") {
        #[cfg(feature = "content-search")]
        {
            let batch_size: usize = std::env::args()
                .nth(2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            tracing::info!(
                batch_size,
                "kremory-http: running full episode+entity+fact re-embed, \
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

    // The CONSUMER binary installs the recorder (the
    // `kremory` lib never does). A Prometheus PULL exporter (HTTP is scrapeable;
    // the sibling stdio bin uses a different channel — hence the `prometheus`
    // feature gate). This makes the already-emitted `kremory_core_*` metrics
    // (tokens, cost, request latency, recall/ingest/dream histograms) actually
    // render at `/metrics` instead of firing into silence (R1's #1 finding).
    #[cfg(feature = "prometheus")]
    let prom_handle = PrometheusBuilder::new()
        .install_recorder()
        .context("failed to install Prometheus recorder")?;

    // Read the
    // search-fusion SWEEP knobs at boot so weight/`k` sweeps cost a restart, not
    // a rebuild. `KREMORY_RRF_K` feeds the bin-local `rrf_merge` (hybrid arm);
    // the library fusion sites (`context.rs` entity RRF + `rrf_fuse_with_content`
    // content fusion) read the SAME env at Memory construction via
    // `facade::providers::search_env_overrides`. `KREMORY_CONTENT_WEIGHT` is
    // consumed by the library — read here only for the authoritative boot
    // banner. Fail-loud: a malformed `KREMORY_RRF_K` WARNs + falls back to 1
    // (never silently accepted as garbage). Default flipped 60 -> 1
    // alongside `SearchConfig::default().rrf_k` — measured win on
    // the full LoCoMo corpus; kept in sync here so this bin-local hybrid-fusion
    // sweep site still tracks the library default, per this fn's own doc
    // comment below ("tracks the same k as the library fusion sites").
    let rrf_k: usize = match std::env::var("KREMORY_RRF_K") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    value = %raw,
                    error = %e,
                    "KREMORY_RRF_K is not a valid usize — falling back to default 1"
                );
                1
            }
        },
        Err(_) => 1,
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

    // Loopback by DEFAULT. This bound `0.0.0.0` — every interface — while serving
    // `DELETE /namespaces/{ns}` straight onto `Memory::forget` with NO inbound
    // authentication anywhere in this binary. (The only key here,
    // `KREMORY_MCP_CHAT_API_KEY`, is an OUTBOUND credential for the chat
    // provider.) That combination is unauthenticated remote erasure of a
    // namespace by anyone who can reach the port — on a laptop, that is anyone on
    // the coffee-shop wifi.
    //
    // kremory is local-first by design, so loopback is also the honest default
    // rather than a restriction. Binding wider is now a deliberate act, and the
    // operator who takes it is told what they are exposing.
    let host = std::env::var("KREMORY_MCP_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    if host != "127.0.0.1" && host != "localhost" && host != "::1" {
        tracing::warn!(
            %host,
            "kremory-http is binding a NON-LOOPBACK address and has NO inbound \
             authentication. DELETE /namespaces/{{ns}} erases a namespace. Put it \
             behind a reverse proxy that authenticates, or bind loopback."
        );
    }
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .with_context(|| format!("failed to bind {host}:{port}"))?;

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
mod tests;
