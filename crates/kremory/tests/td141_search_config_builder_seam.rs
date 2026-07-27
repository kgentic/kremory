#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-141 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-141) — end-to-end
//! tests for the `MemoryBuilder` `SearchConfig` builder seam.
//!
//! Before this seam, `content_stream_weight` / `rrf_k` / `episode_dense_enabled`
//! reached `Engine` ONLY via the `KREMORY_CONTENT_WEIGHT` / `KREMORY_RRF_K` /
//! `KREMORY_EPISODE_DENSE` process-wide env vars — untunable from library-consumer
//! code and untestable end-to-end (TD-140's note: "the sibling knobs ship with
//! no behavioural test because they are env-only — there is no builder hook on
//! `Memory` to set `SearchConfig`"). These tests exercise the seam TD-141 adds:
//! each per-knob `MemoryBuilder::with_*` setter, its precedence over env
//! (design decision (a): explicit programmatic config > env override >
//! default), and that leaving the knobs unset reproduces today's
//! byte-identical defaults — all observed via the REAL public accessor
//! `Memory::search_config()` (not a hand-shaped model of the config), driven
//! through the REAL `MemoryBuilder::into_future` construction path across
//! multiple type-states (LLM builder, no-LLM custom-extractor builder, and the
//! `.with_sink()` `BackgroundIngestorGraphHandle` path) — per
//! `instrument-real-data-flow-before-hypothesizing` §3 ("test the real system,
//! not a model of it"), since this seam threads through 7 distinct
//! `GraphOpenParams` / `OpenGraphParams` / `OpenEngineHandleParams`
//! construction sites in `facade/builder.rs`.
//!
//! Env-var tests mutate process env directly: safe because nextest isolates
//! every test in its own process (TD-109), mirroring the existing precedent in
//! `facade::providers::search_env_override_tests::search_env_overrides_apply_default_and_failloud`.

use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::{DynEmbeddingProvider, Memory};
use std::sync::Arc;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td141_search_config_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Clears the three TD-141 env knobs so a test doesn't inherit stray state
/// from the calling shell/CI environment. Safe per-test because nextest runs
/// each test in its own process (TD-109) — no cross-test leakage.
fn clear_search_env() {
    std::env::remove_var("KREMORY_CONTENT_WEIGHT");
    std::env::remove_var("KREMORY_RRF_K");
    std::env::remove_var("KREMORY_EPISODE_DENSE");
}

// ── Minimal BYOE extractor for the no-LLM path test ──────────────────────────

struct NullExtractor;

impl EntityExtractor for NullExtractor {
    fn name(&self) -> &'static str {
        "null-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: vec![],
            facts: vec![],
        })
    }
}

// ── Minimal sink for the `.with_sink()` BackgroundIngestorGraphHandle path ──

/// Only the two methods `EnrichmentEventSink` requires with no default
/// (`on_community_updated`, `on_batch_phase2_complete`) — every
/// `IngestEventSink` method has a `{}` default (TD-133 D3), so the parent
/// trait needs no explicit impl body.
struct NullSink;

impl kremory::core::sink::IngestEventSink for NullSink {}

impl kremory::memory::events::EnrichmentEventSink for NullSink {
    fn on_community_updated(&self, _community_id: &str, _member_count: usize) {}
    fn on_batch_phase2_complete(&self, _event: kremory::memory::events::BatchPhase2Complete) {}
}

// ── 1. Each setter is independently observable in the live SearchConfig ─────

#[tokio::test]
async fn with_content_stream_weight_reaches_live_search_config() {
    clear_search_env();
    let mem = Memory::open(unique_db("content_weight"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_content_stream_weight(2.5)
        .await
        .expect("build with with_content_stream_weight");

    assert_eq!(
        mem.search_config().content_stream_weight,
        2.5,
        "programmatic with_content_stream_weight(2.5) must reach the live SearchConfig"
    );
}

#[tokio::test]
async fn with_rrf_k_reaches_live_search_config() {
    clear_search_env();
    let mem = Memory::open(unique_db("rrf_k"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_rrf_k(7)
        .await
        .expect("build with with_rrf_k");

    assert_eq!(
        mem.search_config().rrf_k,
        7,
        "programmatic with_rrf_k(7) must reach the live SearchConfig"
    );
}

#[tokio::test]
async fn with_episode_dense_enabled_reaches_live_search_config() {
    clear_search_env();
    let mem = Memory::open(unique_db("episode_dense"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_episode_dense_enabled(true)
        .await
        .expect("build with with_episode_dense_enabled");

    assert!(
        mem.search_config().episode_dense_enabled,
        "programmatic with_episode_dense_enabled(true) must reach the live SearchConfig"
    );
}

// ── 2. Unset knobs reproduce today's byte-identical defaults ────────────────

#[tokio::test]
async fn unset_knobs_match_documented_defaults() {
    clear_search_env();
    let mem = Memory::open(unique_db("defaults"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("build with no search overrides");

    let cfg = mem.search_config();
    assert_eq!(
        cfg.content_stream_weight, 1.0,
        "default content_stream_weight must stay 1.0 (equal-weight fusion) — byte-identical to pre-TD-141"
    );
    assert_eq!(
        cfg.rrf_k, 60,
        "default rrf_k must stay 60 (Cormack et al.) — byte-identical to pre-TD-141"
    );
    assert!(
        !cfg.episode_dense_enabled,
        "default episode_dense_enabled must stay false (BM25-only) — byte-identical to pre-TD-141"
    );
}

// ── 3. Precedence: explicit programmatic config wins over env override ──────

#[tokio::test]
async fn explicit_config_wins_over_env_override() {
    // Env sets DIFFERENT values than the programmatic overrides below — if
    // precedence were wrong (env winning, or a stale mid-priority order), the
    // assertions below catch it.
    std::env::set_var("KREMORY_CONTENT_WEIGHT", "9.9");
    std::env::set_var("KREMORY_RRF_K", "999");
    std::env::set_var("KREMORY_EPISODE_DENSE", "true");

    let mem = Memory::open(unique_db("precedence_explicit_wins"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_content_stream_weight(2.5)
        .with_rrf_k(7)
        .with_episode_dense_enabled(false)
        .await
        .expect("build with explicit overrides + conflicting env");

    let cfg = mem.search_config();
    assert_eq!(
        cfg.content_stream_weight, 2.5,
        "explicit with_content_stream_weight(2.5) must win over KREMORY_CONTENT_WEIGHT=9.9"
    );
    assert_eq!(
        cfg.rrf_k, 7,
        "explicit with_rrf_k(7) must win over KREMORY_RRF_K=999"
    );
    assert!(
        !cfg.episode_dense_enabled,
        "explicit with_episode_dense_enabled(false) must win over KREMORY_EPISODE_DENSE=true"
    );

    clear_search_env();
}

/// The sparse-overlay design decision (see `SearchConfigOverrides` rustdoc in
/// `core/config.rs`): setting ONE knob programmatically must NOT clobber env
/// overrides for the OTHER two knobs. This is the behaviour a monolithic
/// `Option<SearchConfig>` could not express (TD-141's literal design decision
/// (b) sketch) — the reason this implementation deviates from that shape.
#[tokio::test]
async fn unset_knob_still_honours_its_own_env_override() {
    std::env::set_var("KREMORY_RRF_K", "13");
    std::env::remove_var("KREMORY_CONTENT_WEIGHT");
    std::env::remove_var("KREMORY_EPISODE_DENSE");

    let mem = Memory::open(unique_db("precedence_sparse_overlay"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        // Only content_stream_weight is set programmatically — rrf_k and
        // episode_dense_enabled are left to env / default.
        .with_content_stream_weight(3.0)
        .await
        .expect("build with one explicit knob + one env knob");

    let cfg = mem.search_config();
    assert_eq!(
        cfg.content_stream_weight, 3.0,
        "the explicitly-set knob must reflect the programmatic value"
    );
    assert_eq!(
        cfg.rrf_k, 13,
        "the UNSET knob must still honour its own env override (KREMORY_RRF_K=13) — \
         a sibling knob being set programmatically must not clobber it"
    );
    assert!(
        !cfg.episode_dense_enabled,
        "the UNSET, no-env knob must fall through to its default (false)"
    );

    clear_search_env();
}

// ── 4. The seam also reaches the no-LLM custom-extractor path ───────────────

#[tokio::test]
async fn with_rrf_k_reaches_live_search_config_on_no_llm_custom_extractor_path() {
    clear_search_env();
    let ext = Arc::new(NullExtractor);

    let mem = Memory::open(unique_db("no_llm_path"))
        .with_embedder(null_embedder())
        .with_extractor(ext)
        .with_rrf_k(11)
        .await
        .expect("build on the no-LLM custom-extractor path (open_graph_no_llm)");

    assert_eq!(
        mem.search_config().rrf_k,
        11,
        "with_rrf_k must reach the live SearchConfig via open_graph_no_llm too"
    );
}

// ── 5. The seam also reaches the `.with_sink()` BackgroundIngestorGraphHandle path ──

#[tokio::test]
async fn with_content_stream_weight_reaches_live_search_config_on_sink_path() {
    clear_search_env();
    let mem = Memory::open(unique_db("sink_path"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_event_sink(Arc::new(NullSink))
        .with_content_stream_weight(4.25)
        .await
        .expect("build on the .with_sink() BackgroundIngestorGraphHandle path");

    assert_eq!(
        mem.search_config().content_stream_weight,
        4.25,
        "with_content_stream_weight must reach the live SearchConfig through the \
         BackgroundIngestorGraphHandle path (Memory::search_config delegates to the \
         EngineGraphHandle delegate opened via the first of two open_engine_handle calls)"
    );

    drop(mem);
}
