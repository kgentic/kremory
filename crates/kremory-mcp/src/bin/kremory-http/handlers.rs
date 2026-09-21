use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use kremory::Memory;
use kremory_mcp::handlers::{self, ToolError};
use kremory_mcp::params::{
    DreamParams, RecallFormat, RecallParams, RecallStructuredOutput, RecallTemplateWire,
    RecallTextOutput, RememberParams, RetrievedContextWire,
};

use crate::fusion::{fused_cap, rrf_merge};
use crate::wire::{
    AppState, ApiError, ConsolidationQuery, CreateMemoryBody, SearchMode, SearchQuery,
    SearchResponseWire, SearchResultKindWire, SearchResultWire,
};

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
/// config must be parsed loudly, never silently defaulted, the same as it
/// applies to LLM output: a benchmark run that silently disabled
/// reranking would misreport its own configuration (exactly the
/// provenance-stamp failure mode `/health`'s `rerank_k` field exists to
/// prevent).
pub(crate) fn parse_rerank_k(raw: Option<&str>) -> Option<usize> {
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

/// `KREMORY_RERANK_K` boot override — enables the cross-encoder reranker
/// on this REST route for offline A/B sweeps. Unlike `rrf_k` (an `AppState`
/// field, threaded into `rrf_merge`), `rerank_k` is a per-recall param consumed
/// only in the `search` handler and carried on `RecallParams`, NOT in
/// `SearchConfig` — so it lives in a process-wide `LazyLock` (read once at first
/// request) rather than `AppState`, keeping every `AppState` literal untouched.
/// Mirrors the `KREMORY_RRF_K` sweep design: an A/B costs a restart, not a
/// rebuild. `None`/malformed ⇒ rerank off by default. Fail-loud: a
/// malformed value WARNs and disables rerank rather than silently accepting it.
///
/// The actual parsing/validation lives in [`parse_rerank_k`] (unit-tested
/// below) — this `LazyLock` is a thin, untestable-by-construction shim around
/// it.
static RERANK_K: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
    parse_rerank_k(std::env::var("KREMORY_RERANK_K").ok().as_deref())
});

// ────────────────────────────────────────────────────────────────────────
// GET /health
// ────────────────────────────────────────────────────────────────────────

/// `GET /health` — liveness (still 200) PLUS the server's build-feature flags
/// and its ACTIVE scoring config. The bench harness's `provenance.build_provenance()` stamps the
/// recall-run JSON from THIS response so the provenance record is faithful to
/// what the search path actually used — rather than re-reading the harness
/// process's env, which can silently differ from the server's env and is exactly
/// how a config-mismatch produced a bogus benchmark number.
///
/// - `content_search` / `rerank` / `prometheus` are compile-time
///   `cfg!(feature = "…")` booleans — an orchestrator can now detect a degraded
///   recall-only build programmatically instead of scraping the boot banner
///   (closing a gap where an operator previously had no programmatic signal).
/// - `scoring` is read from the LIVE [`kremory::Memory::search_config`] (the
///   Engine's `SearchConfig`, carrying any `KREMORY_CONTENT_WEIGHT` /
///   `KREMORY_RRF_K` boot overrides), NOT re-read from env here.
///
/// Additive: the response is still `200 OK`; the JSON body is new (the prior
/// handler returned an empty 200), so no existing field is removed.
/// Identity of the running binary, read from its OWN executable.
///
/// Every field is observed at runtime, so a stale or mismatched value is not
/// representable: `current_exe()` is the artefact actually executing, and its
/// mtime + size pin it to a specific build. `crate_version` is compile-time and
/// pins the source revision's declared version.
///
/// Deliberately NOT a git sha. A sha would have to be injected at build time and
/// would then assert what the builder BELIEVED, whereas mtime+size are properties
/// of the file that is running — the observe-over-declare split. The harness pairs
/// this with its own `harness_git_sha` so a mismatch between the two is visible
/// rather than silently collapsed into one number.
///
/// Every field is best-effort: a filesystem that refuses to stat the executable
/// must degrade to an explicit `"unknown"`, never to a silently-absent key that a
/// downstream stamp would render as "no problem".
fn build_identity() -> serde_json::Value {
    let exe = std::env::current_exe().ok();
    let meta = exe.as_ref().and_then(|p| std::fs::metadata(p).ok());
    let mtime_unix = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    serde_json::json!({
        "crate_version": env!("CARGO_PKG_VERSION"),
        "exe": exe.as_ref().map_or_else(
            || "unknown".to_string(),
            |p| p.display().to_string(),
        ),
        "exe_mtime_unix": mtime_unix,
        "exe_size_bytes": meta.as_ref().map(std::fs::Metadata::len),
        "features": {
            "content_search": cfg!(feature = "content-search"),
            "rerank": cfg!(feature = "rerank"),
            "prometheus": cfg!(feature = "prometheus"),
        },
    })
}

pub(crate) async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let scoring = state.mem.search_config();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "content_search": cfg!(feature = "content-search"),
            "rerank": cfg!(feature = "rerank"),
            // The ACTIVE rerank_k (KREMORY_RERANK_K boot override) — so the
            // bench provenance stamp records whether reranking was on + at what
            // depth. null ⇒ off. Distinct from the `rerank` compile-flag above
            // (compiled-in ≠ enabled): a build can ship the reranker yet run it off.
            "rerank_k": *RERANK_K,
            // The REQUESTED reranker (`KREMORY_RERANK_MODEL`), so a result JSON
            // records which cross-encoder produced it. Deliberately named
            // `_requested`, not `_active`: resolution lives in `kremory`'s
            // `parse_reranker_model`, and an unrecognised alias there falls back
            // to bge-base with a WARN. Echoing the raw value as "active" could
            // therefore stamp a model that never loaded — exactly the
            // provenance-lie this field exists to avoid. The authoritative record of what
            // actually loaded is the `kremory.rerank.model_selected` INFO log.
            // Exposing the resolved name instead would mean new public API on
            // `kremory` (with napi-parity obligations) for a bench-only
            // sweep knob — not worth it; revisit if the knob outgrows benching.
            "rerank_model_requested": std::env::var("KREMORY_RERANK_MODEL")
                .ok()
                .unwrap_or_else(|| "bge-base (default)".to_string()),
            "prometheus": cfg!(feature = "prometheus"),
            // The identity of THE BINARY THAT IS SERVING, observed by the
            // process from its own executable file.
            //
            // The bench provenance stamp previously recorded only
            // `git rev-parse HEAD` run in the HARNESS's working tree
            // (`bench/locomo/provenance.py:218`), which says nothing about which
            // binary answered the requests — the tree can have moved on, or back,
            // or be a different checkout entirely. `.context/td186a-variance/`
            // already records a run stamped with a commit authored TWELVE HOURS
            // AFTER the binary was built.
            //
            // That mis-attribution is normally recoverable by re-running. The paid
            // grader is NOT: it runs ONCE, at the end, by design (~$4.60/config),
            // so its provenance is the one stamp that can never be corrected after
            // the fact.
            //
            // OBSERVED, not declared (contract-first §"derive > observe > declare"):
            // a binary reporting its own `current_exe` mtime+size cannot be wrong
            // about which artefact is running, whereas any value passed in from the
            // environment is an assertion by whoever launched it. This is the same
            // reasoning as `rerank_model_requested` above refusing to claim
            // "active".
            "build": build_identity(),
            "scoring": {
                "content_stream_weight": scoring.content_stream_weight,
                "rrf_k": scoring.rrf_k,
                "graph_degree_weight": scoring.graph_degree_weight,
                "temporal_weight": scoring.temporal_weight,
                // Dense episode arm — surfaced so the bench provenance
                // stamp records whether KREMORY_EPISODE_DENSE was active (the
                // dense-vs-BM25 A/B is otherwise invisible in the recall-run JSON).
                "episode_dense_enabled": scoring.episode_dense_enabled,
            },
        })),
    )
}

pub(crate) async fn create_memory(
    State(state): State<AppState>,
    Json(body): Json<CreateMemoryBody>,
) -> Result<impl IntoResponse, ApiError> {
    let params = RememberParams {
        namespace: body.namespace,
        thread: None,
        content: body.content,
        source_kind: None,
        source_id: body.source_id,
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

/// Flattens a `RetrievedContextWire` (entity_name/summary + its connected
/// `facts[].fact` natural-language strings — `params.rs:117-135`) into a
/// single prose string. kremory's `RecallStructuredOutput` is entity-shaped
/// (one result = one entity + its facts); codemem's benchmark scorer expects
/// one flat `content` string per result to substring-match the gold answer
/// against — this is the "thin adapter" the REST route owns so
/// `handlers::do_recall` itself stays transport-agnostic.
pub(crate) fn flatten_result_content(r: &RetrievedContextWire) -> String {
    let mut parts = vec![format!("{}: {}", r.entity_name, r.summary)];
    parts.extend(r.facts.iter().map(|f| f.fact.clone()));
    parts.join(". ")
}

pub(crate) async fn search(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Result<impl IntoResponse, ApiError> {
    // `format=text` returns kremory's OWN
    // prompt-ready rendering instead of the flattened rows. It is the
    // rendering the MCP tool surface has always defaulted to.
    //
    // `format=text` previously read `query.mode` ONLY to gate
    // `mode=content` (422) below, and otherwise ALWAYS retrieved via a
    // single `handlers::do_recall(format: Text)` call regardless of whether
    // the caller asked for `mode=recall` or `mode=hybrid` — i.e. `mode` was
    // silently ignored for every OTHER value. Every REST caller requesting
    // `format=text` under the default `mode=hybrid` therefore silently got
    // `mode=recall`-quality retrieval (measured on conv0: 77.2 vs 82.1
    // recall@10, 63.9 vs 69.6 nDCG@10 — hybrid is the better arm on every
    // metric).
    //
    // Retrieval and rendering are now separated by construction: `mode` is
    // matched HERE, exactly as it is below for `format=structured`, to fetch
    // the item set ([`fetch_recall_items`] / [`hybrid_items_for_render`]) —
    // the SAME functions (or their retrieval halves) `format=structured`
    // uses for the same `mode`, so the two formats cannot drift apart again.
    // `format=text` only picks the LAST step (render vs serialize); it
    // cannot bypass the `mode` dispatch, because there is no other code path
    // left that reaches this response.
    if query.format == Some(RecallFormat::Text) {
        let template = query.template;
        let params = RecallParams {
            namespace: query.namespace,
            thread: None,
            query: query.q,
            k: query.k,
            as_of: None,
            format: RecallFormat::Structured,
            template,
            rerank_k: *RERANK_K,
        };
        let items = match query.mode {
            SearchMode::Content => {
                return Err(ApiError(ToolError::InvalidParams(
                    "format=text renders the entity/fact recall surface and has no \
                     meaning for mode=content, which returns BM25 passages. Use \
                     mode=recall or mode=hybrid."
                        .to_string(),
                )));
            }
            SearchMode::Recall => fetch_recall_items(&state.mem, params).await?,
            SearchMode::Hybrid => {
                hybrid_items_for_render(&state.mem, params, state.rrf_k).await?
            }
        };
        let block = render_prompt_block(&items, template);
        return Ok(Json(serde_json::to_value(RecallTextOutput { block }).map_err(
            |e| {
                ApiError(ToolError::Internal(format!(
                    "kremory-http: failed to serialize format=text block: {e}"
                )))
            },
        )?));
    }

    let params = RecallParams {
        namespace: query.namespace,
        thread: None,
        query: query.q,
        k: query.k,
        as_of: None,
        format: RecallFormat::Structured,
        template: query.template,
        // The cross-encoder reranker is exposed on this bench/eval
        // REST route via the `KREMORY_RERANK_K` boot override (read once into the
        // `RERANK_K` static above), mirroring the `KREMORY_RRF_K` sweep pattern so
        // an A/B costs a server restart, not a rebuild. `None` ⇒ rerank off
        // by default. Deep-pool discipline: the reranker sees exactly
        // the caller's `k` items (harness `--recall-limit 50`), reordered then
        // scored at top-10 downstream by evidence_eval.py — a rank-aware scorer
        // is required here, since an order-blind substring scorer cannot
        // detect a reordering-only change at all.
        rerank_k: *RERANK_K,
    };
    let results = match query.mode {
        SearchMode::Recall => recall_mode_results(&state.mem, params).await?,
        SearchMode::Content => content_mode_results(&state.mem, params).await?,
        SearchMode::Hybrid => hybrid_mode_results(&state.mem, params, state.rrf_k).await?,
    };
    // Serialised through `Value` so both arms of this handler share one return
    // type — the `format=text` arm above returns kremory's rendered string
    // rather than a results array. Existing `format=structured` callers see a
    // byte-identical `{"results":[…]}` body.
    Ok(Json(serde_json::to_value(SearchResponseWire { results }).map_err(
        |e| {
            ApiError(ToolError::Internal(format!(
                "kremory-http: failed to serialize search results: {e}"
            )))
        },
    )?))
}

/// Fetch the entity/fact recall arm's item SET (`handlers::do_recall`,
/// structured format) WITHOUT flattening it into the `format=structured`
/// wire shape — the retrieval half of `recall_mode_results`, extracted so
/// the `format=text` path can consume the same full-fidelity items
/// [`retrieved_context_wire_to_search_result`] flattens for `mode=recall`.
/// `RecallStructuredOutput.results: Vec<RetrievedContextWire>` already IS
/// this full-fidelity shape — see `params.rs`.
async fn fetch_recall_items(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<RetrievedContextWire>, ApiError> {
    let value = handlers::do_recall(mem, params).await?;
    let structured: RecallStructuredOutput = serde_json::from_value(value).map_err(|e| {
        ApiError(ToolError::Internal(format!(
            "kremory-http: failed to deserialize recall structured output: {e}"
        )))
    })?;
    Ok(structured.results)
}

/// Flatten one entity/fact recall-arm item into the `format=structured`
/// wire shape. Extracted verbatim from `recall_mode_results`'s former
/// inline closure — same logic, same output, now also reusable by
/// [`fetch_recall_items`]'s other consumer.
fn retrieved_context_wire_to_search_result(r: &RetrievedContextWire) -> SearchResultWire {
    // A dense-fact-arm hit is lifted into the
    // entity+content fusion as a synthetic `RetrievedContext` with
    // `entity_type_name == "Fact"`
    // (`core::search::fact_hit_into_retrieved_context`'s
    // discriminator) — mirrors how a `ContentPassage` is tagged
    // `"ContentPassage"` one layer down. `fact_dense_enabled`
    // defaults `false`, so no live result carries this tag unless
    // the knob is on.
    //
    // FIXED: `"ContentPassage"` was previously NOT
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
        // episode id applies here. This loss is left in place
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
}

/// `mode=recall` — the entity-shaped keyword/semantic/graph path
/// (`handlers::do_recall`, structured format), flattened via
/// [`flatten_result_content`]. Byte-identical to `/search`'s pre-W0.1
/// behaviour ONLY when this bin is built WITHOUT `content-search`. When
/// `content-search` IS compiled in, `handlers::do_recall`'s `Structured`
/// format calls `.raw()`, and `.raw()` now fuses in the content stream by
/// default (`core::search::rrf_fuse_with_content`) — see
/// this module's top-level doc comment.
///
/// Retrieval (`fetch_recall_items`) and flattening
/// (`retrieved_context_wire_to_search_result`) are now two named steps —
/// this fn's OUTPUT is unchanged, but the retrieval step is shared with the
/// `format=text` rendering path so `mode` genuinely selects the item set
/// regardless of `format`.
async fn recall_mode_results(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<SearchResultWire>, ApiError> {
    let items = fetch_recall_items(mem, params).await?;
    Ok(items
        .iter()
        .map(retrieved_context_wire_to_search_result)
        .collect())
}

/// `mode=content` — BM25-only full-text search over raw
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
/// fail-loud: a silent fallback to `mode=recall` here served a
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

/// Lift one BM25 `ContentPassage` (`mode=content`'s raw arm) into this
/// crate's `RetrievedContextWire` DTO shape — so the `format=text`
/// rendering path can carry a content-arm item with the SAME full-fidelity
/// type the entity/fact arm's [`fetch_recall_items`] already returns, and
/// fuse both through one [`rrf_merge`] + one renderer.
///
/// Mirrors `kremory::core::search::content_passage_into_retrieved_context`
/// (`crates/kremory/src/core/search.rs`, module-private — not reachable from
/// this crate) field-for-field, EXCEPT `score` (see below): `entity_id =
/// episode_id.to_string()`, `entity_name = "Episode #{episode_id}"`,
/// `summary = snippet`, empty `facts`, `entity_type_name = "ContentPassage"`
/// (the SAME discriminator `retrieved_context_wire_to_search_result` already
/// keys its `kind` mapping off of).
///
/// **Deliberately NOT folded into the [`kremory::memory::RenderableContext`]
/// trait that replaced `render_entities_wire`/`render_edge_summary_wire`/
/// `render_temporal_facts_wire`.** That trait is READ-ONLY (accessors over an
/// already-built value); this fn is a CONSTRUCTOR, a different shape of
/// duplication the trait doesn't address. Two independent, structural reasons
/// it can't be unified even so: (1) `content_passage_into_retrieved_context`
/// is a private fn in `core::search` — this crate cannot call it regardless
/// of any trait; (2) `score` is NOT copy-paste-identical — core's version
/// hardcodes `score: 0.0` as a documented placeholder because its ONE call
/// site is a step inside `rrf_fuse_with_content`'s fusion loop, which
/// overwrites it with the real RRF-fused score immediately after
/// construction; this fn's caller does no such fusion step, so it must bake
/// in the real `passage.score` at construction time or the score would ship
/// as a permanent `0.0`. Sharing one body would require threading that
/// call-site difference through — out of scope here. This is recorded as
/// knowingly-remaining duplication rather than
/// silently left as-is.
#[cfg(feature = "content-search")]
fn content_passage_into_context_wire(
    passage: kremory::memory::ContentPassage,
) -> RetrievedContextWire {
    RetrievedContextWire {
        entity_id: passage.episode_id.to_string(),
        entity_name: format!("Episode #{}", passage.episode_id),
        summary: passage.snippet,
        score: passage.score,
        incomplete: false,
        entity_type_id: 0,
        entity_type_name: "ContentPassage".to_string(),
        namespace: None,
        source_refs: vec![passage.source_ref.into()],
        facts: Vec::new(),
    }
}

/// Fetch the BM25 content arm's item SET in the SAME full-fidelity
/// `RetrievedContextWire` shape [`fetch_recall_items`] returns —
/// the retrieval half of `content_mode_results`, lifted via
/// [`content_passage_into_context_wire`] instead of flattened into
/// [`SearchResultWire`]. `content_mode_results` itself is UNCHANGED and
/// remains the sole producer of `format=structured&mode=content`'s wire
/// output — this is a parallel fetch for the `format=text` rendering path
/// only, never substituted into the structured path (whose `content` field
/// is the BARE snippet, not `"Episode #N: {snippet}"` — merging the two
/// would change that byte-for-byte contract).
#[cfg(feature = "content-search")]
async fn fetch_content_items(
    mem: &Memory,
    params: RecallParams,
) -> Result<Vec<RetrievedContextWire>, ApiError> {
    let passages = handlers::do_recall_content(mem, params).await?;
    Ok(passages
        .into_iter()
        .map(content_passage_into_context_wire)
        .collect())
}

/// `mode=hybrid` (the DEFAULT) — runs BOTH `mode=recall` and `mode=content`
/// and RRF-fuses them via [`rrf_merge`]. The fusion/fairness decision
/// was resolved by an earlier LoCoMo
/// diagnostic run: RRF fusion over the entity/fact recall stream + the BM25 content
/// stream. Note (measured): the two streams have disjoint id-spaces
/// (entity-ids vs episode-ids), so RRF ≈ the prior `naive_merge` on this
/// benchmark (both 70.4%); the lift over `recall`-only (40.2%) is from ADDING
/// the content stream.
///
/// This fusion has been ported into
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
    // Absent an explicit `k`, the bound is the FULL deduped union,
    // matching the library's `core::search::rrf_fuse_with_content`. It
    // previously read `recall.len().max(content.len())` —
    // an expression that assumes the two arms OVERLAP so `union ≈ max`. They
    // do not: `rrf_merge`'s own doc records the streams as having DISJOINT
    // id-spaces (entity-ids vs episode-ids), so `.max()` silently dropped
    // `min(recall, content)` distinct results with no error and no warning.
    //
    // Note the direction of travel: `.max()` ORIGINATED here and was specified
    // INTO the library ("matching the REST fix's fallback"). The library was
    // later fixed and nothing propagated back to the source.
    //
    // The bench never hit this (`bench/common/kremory_client.py:145` always
    // sends `k`); consumers of `GET /search?mode=hybrid` — the DEFAULT mode —
    // did.
    let cap = fused_cap(k, recall.len(), content.len());
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

/// `format=text`'s `mode=hybrid` item set — the retrieval-only
/// twin of [`hybrid_mode_results`], operating on the full-fidelity
/// `RetrievedContextWire` shape ([`fetch_recall_items`] +
/// [`fetch_content_items`]) instead of the flattened `format=structured`
/// wire shape, so the RENDERED item set (and its RELATIVE ORDER — the
/// fused score `rrf_merge` computes) matches
/// `format=structured&mode=hybrid`'s BY CONSTRUCTION: same params, same
/// underlying fetches, same generic [`rrf_merge`]/[`fused_cap`] the
/// structured path uses.
#[cfg(feature = "content-search")]
async fn hybrid_items_for_render(
    mem: &Memory,
    params: RecallParams,
    rrf_k: usize,
) -> Result<Vec<RetrievedContextWire>, ApiError> {
    let k = params.k;
    let recall = fetch_recall_items(mem, params.clone()).await?;
    let content = fetch_content_items(mem, params).await?;
    let cap = fused_cap(k, recall.len(), content.len());
    let mut merged = rrf_merge(recall, content, rrf_k);
    merged.truncate(cap);
    Ok(merged)
}

/// Feature-off HARD-FAIL twin of [`hybrid_items_for_render`] — same B1
/// fail-loud rationale as [`hybrid_mode_results`]'s feature-off arm.
#[cfg(not(feature = "content-search"))]
async fn hybrid_items_for_render(
    _mem: &Memory,
    _params: RecallParams,
    _rrf_k: usize,
) -> Result<Vec<RetrievedContextWire>, ApiError> {
    tracing::error!(
        "mode=hybrid (format=text) requested but kremory-http was built WITHOUT the \
         `content-search` feature — refusing to serve a silently-degraded recall-only \
         rendering (B1 fail-loud). Rebuild with `--features content-search`."
    );
    Err(ApiError(ToolError::InvalidParams(
        "mode=hybrid requires the `content-search` feature, but this server was built \
         without it. Rebuild with `--features content-search`."
            .to_string(),
    )))
}

/// Render an already-computed `format=text` item set into kremory's
/// prompt-ready text block — the `format=text` twin of
/// [`SearchResponseWire`]'s JSON serialization for `format=structured`.
/// Retrieval and rendering are two SEPARATE steps by construction: this fn
/// takes the item set `mode` already selected ([`fetch_recall_items`] /
/// [`hybrid_items_for_render`]) and does nothing else — it cannot see or
/// re-derive `mode`, so there is no code path left where a rendering choice
/// could silently substitute a different item set.
///
/// Calls `kremory::memory::render_entities` / `render_edge_summary` /
/// `render_temporal_facts` directly — these are generic over
/// `kremory::memory::RenderableContext`, which [`RetrievedContextWire`]
/// implements (`conversions.rs`), so this crate no longer needs its own
/// hand-mirrored copies of the three renderer bodies. Only the multi-
/// namespace `[ns:{group_id}]` prefix behaves differently here, and it does
/// so as a DATA fact rather than a second code path: `/search` takes exactly
/// one `namespace` per request (`RecallParams::namespace: String`), so
/// `RetrievedContextWire::namespace_group_id` always returns the SAME value
/// across one response's items (see that impl's doc comment) — the shared
/// renderer's multi-namespace branch is live code that structurally cannot
/// fire here, not a dropped feature.
pub(crate) fn render_prompt_block(items: &[RetrievedContextWire], template: RecallTemplateWire) -> String {
    if items.is_empty() {
        return String::new();
    }
    match template {
        RecallTemplateWire::Entities => kremory::memory::render_entities(items),
        RecallTemplateWire::EdgeSummary => kremory::memory::render_edge_summary(items),
        RecallTemplateWire::TemporalFacts => kremory::memory::render_temporal_facts(items),
    }
}

// ────────────────────────────────────────────────────────────────────────
// DELETE /namespaces/{ns}
// ────────────────────────────────────────────────────────────────────────

pub(crate) async fn delete_namespace(
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
    // Full per-table outcome (TD-247) — `deleted: 0` on a real erasure was
    // misleading (shared-entity preservation routinely leaves `entities: 0`
    // while facts/episodes/edges were removed; see `ForgetOutcome::is_empty`
    // for the honest "did anything happen?" check). This was deliberately
    // deferred until "after the crate ships" — it has (0.8.0 and 0.9.0 are
    // both live) — so this widens the response to match, mirroring the shape
    // already shipped for `kremory-napi`'s `forget()` (`JsForgetOutcome`).
    Ok(Json(serde_json::json!({
        "entities": deleted.entities,
        "facts": deleted.facts,
        "episodes": deleted.episodes,
        "edges": deleted.edges,
        "is_empty": deleted.is_empty(),
    })))
}

pub(crate) async fn run_consolidation(
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

