use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use kremory::Memory;
use kremory_mcp::handlers::ToolError;
use kremory_mcp::params::{RecallFormat, RecallTemplateWire};

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) mem: Arc<Memory>,
    /// RRF `k` for the
    /// bin-local `rrf_merge` hybrid fusion, read once from `KREMORY_RRF_K` at
    /// boot (default 60). The library fusion sites read their own `k` from the
    /// Engine's `SearchConfig` (via `search_env_overrides`); this bin-local
    /// value keeps the REST hybrid arm's `rrf_merge` on the SAME sweep point.
    pub(crate) rrf_k: usize,
}

// ────────────────────────────────────────────────────────────────────────
// Error mapping — ToolError -> HTTP status + JSON body.
// ────────────────────────────────────────────────────────────────────────

pub(crate) struct ApiError(pub(crate) ToolError);

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
// POST /memories
// ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct CreateMemoryBody {
    pub(crate) content: String,
    pub(crate) namespace: String,
    #[serde(default)]
    pub(crate) published_at: Option<String>,
    /// Conversation/thread key. Forwarded to `RememberRequest`, which
    /// maps a bare source id to `SourceKind::Chat` — i.e. `.from_chat(id)`.
    ///
    /// Two episodes posted with the SAME `source_id` are two turns of one
    /// conversation, which is what makes prior-turn replay reachable over this
    /// transport. Omitted (the previous behaviour) means every episode gets a
    /// fresh uuid and nothing can replay, so this is strictly additive.
    #[serde(default)]
    pub(crate) source_id: Option<String>,
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
pub(crate) enum SearchMode {
    Recall,
    Content,
    // Default is `hybrid` (RRF fusion of the entity/fact recall stream + the BM25
    // content stream), NOT the entity-graph `recall` surface. A LoCoMo
    // diagnostic run (memory `project_kremory_locomo_recall_root_cause_retrieval_surface`)
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
pub(crate) struct SearchQuery {
    pub(crate) q: String,
    pub(crate) namespace: String,
    #[serde(default)]
    pub(crate) k: Option<usize>,
    #[serde(default)]
    pub(crate) mode: SearchMode,
    /// Output rendering — `structured` (default) returns the entity-shaped
    /// rows; `text` returns kremory's prompt-ready rendering of the same
    /// recall, chosen by [`SearchQuery::template`].
    ///
    /// This endpoint previously HARDCODED
    /// `RecallFormat::Structured` and discarded the template, so a REST caller
    /// could not reach the prompt-ready rendering at all — even though the MCP
    /// tool surface has always exposed both and **defaults to `Text` +
    /// `TemporalFacts`**. The consequence was not merely ergonomic: the LoCoMo
    /// harness drives THIS endpoint, so every number this project has published
    /// measured the rendering MCP agents do NOT get. Measured worth of the
    /// rendering: ~+4pt answerability, and +10.8pt on temporal questions
    /// specifically, on identical retrieval.
    ///
    /// The default stays `structured` deliberately: a REST caller may be an
    /// application, not a model, and picking a rendering FOR the consumer is
    /// the mistake this field exists to undo. Offer the choice; do not impose one.
    ///
    /// ⚠️ `Option<_>` and NOT `#[serde(default)]`, because `RecallFormat`'s own
    /// `#[default]` is **`Text`** — right for the MCP tool, whose consumers are
    /// models, and wrong here, where omitting the param has always meant
    /// structured rows. Deriving this surface's default from the type's would
    /// have silently flipped an existing endpoint's contract — the exact
    /// "default changed without anyone deciding" failure this whole change set
    /// exists to fix. Two surfaces, two appropriate defaults, both explicit.
    /// (Caught by the pre-existing `/search` tests, which is why they drive the
    /// real HTTP path rather than the handler's internals.)
    pub(crate) format: Option<RecallFormat>,
    /// Which prompt-ready rendering to use when `format=text`. Ignored
    /// otherwise. Mirrors the MCP tool's `template` param exactly.
    #[serde(default)]
    pub(crate) template: RecallTemplateWire,
}

/// What a `/search` result item IS. Additive wire metadata: existing
/// consumers (the LoCoMo harness) read only `id`/`content`/`score` and are
/// unaffected by this enum's presence. `Fact` is emitted by
/// `recall_mode_results` (below)
/// once the entity+content fusion `handlers::do_recall` → `.raw()` reaches
/// includes a dense-fact-arm entry (
/// `core::search::rrf_fuse_with_facts`, feature-gated + default-OFF via
/// `SearchConfig::fact_dense_enabled` — absent that knob, no `Fact` item ever
/// appears, so this variant is dormant-but-wired on a default build, not
/// unreachable). This enum + `SearchResultWire::source_episode_id` exist so
/// `bench/locomo/evidence_eval.py` can score a fact-kind item against its
/// source episode's evidence turns instead of reading it as irrelevant (a
/// fact string does not contain LoCoMo turn text verbatim).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SearchResultKindWire {
    /// `mode=recall` / the recall arm of `mode=hybrid` — one entity + its
    /// flattened connected facts (`flatten_result_content`).
    Entity,
    /// `mode=content` / the content arm of `mode=hybrid` — a BM25-matched
    /// episode passage (`ContentPassage`).
    Episode,
    /// A dense-fact-arm hit
    /// (`core::search::vector_search_facts`, fused via `rrf_fuse_with_facts`)
    /// — `entity_type_name == "Fact"` on the underlying `RetrievedContext`
    /// is `recall_mode_results`'s discriminator for this variant.
    Fact,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResultWire {
    pub(crate) id: String,
    pub(crate) content: String,
    pub(crate) score: f32,
    /// What this item IS. See
    /// [`SearchResultKindWire`].
    pub(crate) kind: SearchResultKindWire,
    /// Populated ONLY when `kind == Fact`: the episode this fact was
    /// asserted from (`facts.source_episode_id`, `core/schema.rs:129` —
    /// `Option<i64>` because a caller-supplied structured fact, or a fact
    /// whose source episode was later deleted, can carry no episode
    /// anchor). `None` for `Entity`/`Episode` items, and for `Fact` items
    /// with no recorded source. `bench/locomo/evidence_eval.py` resolves
    /// this id to the episode's evidence turns rather than scoring the
    /// fact's own (non-verbatim) text against them.
    pub(crate) source_episode_id: Option<i64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResponseWire {
    pub(crate) results: Vec<SearchResultWire>,
}

// ────────────────────────────────────────────────────────────────────────
// POST /consolidation/{cycle}
// ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct ConsolidationQuery {
    /// FRICTION (flagged per task instructions): codemem's own harness posts
    /// `/consolidation/{cycle}` with NO namespace at all (global consolidation
    /// in codemem's data model) — but `kremory::Memory::dream()` is always
    /// namespace-scoped (the three-signal local-first consistency check
    /// operates per-namespace). There is no lossless mapping from codemem's
    /// global-consolidation semantics to kremory's namespace-scoped `dream()`.
    ///
    /// FAIL-LOUD: this param is REQUIRED, not defaulted. An
    /// earlier draft fell back to a single `"default"` namespace when omitted,
    /// which would have pointed every benchmark conversation at one shared
    /// scope. Missing `?namespace=` is a loud 422 (see `run_consolidation`).
    ///
    /// ⚠️ **CORRECTION — the original rationale here was WRONG.** It
    /// argued the 422 was needed because *"dream idempotency is keyed on
    /// `(namespace, batch_id)`, so every benchmark consolidation call would
    /// share `(default, <cycle>)` and only the FIRST would execute; the rest
    /// silently no-op"*. **That idempotency does not exist.** `batch_id` is
    /// never read on the awaited path (`facade/dream.rs::execute_blocking`),
    /// so repeated calls re-run in full — they do not no-op. The 422 remains
    /// CORRECT, but for the simpler reason stated above (namespace scoping),
    /// not for the mechanism the old comment invoked. Left in place because
    /// requiring the namespace is right either way.
    pub(crate) namespace: Option<String>,
}

