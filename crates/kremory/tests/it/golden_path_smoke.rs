//! Golden-path smoke test — Tier 2 of the 3-tier test pyramid (v0.2.4 Component 2).
//!
//! Governing spec: `.ai-docs/specs/v0-2-4-test-infra-o11y-harness-arch-spec-2026-06-15.md` §9 P3.
//!
//! ONE test drives the entire consumer journey through the PUBLIC `Memory`
//! facade. It is the v0.2.3 PUSH GATE. Two run modes share this body, selected
//! by the `KREMORY_VCR` env var (§4.4):
//!
//!   * `KREMORY_VCR=record` → LIVE: build the real Ollama provider
//!     (`gemma4-e2b:latest`, `.keep_alive("1h")`), wrap it in
//!     `RecordReplayChatProvider::record(...)` so a single live run refreshes the
//!     committed cassette. Requires Ollama + the model pulled. After
//!     `wait_for_processing` returns `Ok(())` the test calls `provider.flush()`
//!     (NEW-202, MANDATORY) before any cassette-entry assertion, because the
//!     Phase-2 cassette write fires on the background worker thread and
//!     flush-on-`Drop` is non-deterministic.
//!   * `KREMORY_VCR=replay` OR unset → REPLAY: build
//!     `RecordReplayChatProvider::replay(cassette)` so CI runs deterministically
//!     with NO Ollama. A missing cassette is a LOUD error (the decorator errors
//!     in `replay()`); recording the cassette is P4.
//!
//! # Assertions are STRUCTURAL INVARIANTS ONLY (RISK-001, §8/§9 P3)
//!
//! The LIVE/record path runs the real, nondeterministic `gemma4-e2b:latest`
//! (80% precision). Asserting exact entity names or exact counts would flake on
//! EVERY cassette re-record. This test asserts ONLY:
//!   1. `episode_processing_status == "Verified"` (pipeline reached terminal state);
//!   2. sink captured `>= 1` `on_entity_extracted` event (Phase-2 ran);
//!   3. `recall.raw().len() >= 1` (recall is non-empty).
//!
//! It contains NO `== "<name>"` against an entity name and NO `.len() == N`
//! against entities/recall — only `>=`.
//!
//! # Phase-2 oracle is the EnrichmentEventSink, NOT trace output (ASMP-001, §6.4)
//!
//! kremory's Phase 2 runs on a dedicated worker thread + its own
//! `new_multi_thread().worker_threads(1)` runtime. The per-test thread-local
//! tracing subscriber (`init_test_log`, §6) cannot reach that thread, so trace
//! output is a caller-thread debugging aid only. Phase-2 correctness is asserted
//! via the captured sink events — the synchronous, cross-thread callback surface.

#![cfg(feature = "llm-smoke")]
// Test files use expect/unwrap/panic as intentional assertion mechanisms.
// Consistent with the project-wide test convention (see tests/llm_integration.rs:55-58,
// tests/sink_trait_shape.rs, background_integration.rs et al).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use kremory::core::error::IngestStatus;
use kremory::core::provider::RecordReplayChatProvider;
use kremory::core::sink::{
    ContradictionDetected, IngestEventSink, IngestionError, OnEdgeAddedParams,
};
use kremory::memory::events::{BatchPhase2Complete, EnrichmentEventSink};
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use crate::support::test_log::init_test_log;

// ── Capturing EnrichmentEventSink (PRIMARY Phase-2 assertion surface, §6.4) ───

/// Captured Phase-2 sink events. Callbacks fire synchronously on whichever
/// thread emits them (including the background worker thread), so a
/// `Mutex`-guarded `Vec` is the only reliable cross-thread Phase-2 oracle.
#[derive(Default)]
struct CapturingSink {
    entities: Mutex<Vec<(String, String)>>,
    edges: Mutex<Vec<(String, String, String)>>,
    stages: Mutex<Vec<IngestStatus>>,
    batches: Mutex<Vec<String>>,
}

impl CapturingSink {
    fn entity_count(&self) -> usize {
        self.entities.lock().unwrap().len()
    }
}

impl IngestEventSink for CapturingSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.entities
            .lock()
            .unwrap()
            .push((entity_id.to_owned(), name.to_owned()));
    }
    fn on_edge_added(&self, params: OnEdgeAddedParams<'_>) {
        let OnEdgeAddedParams {
            from_entity_id: from,
            to_entity_id: to,
            predicate,
        } = params;
        self.edges
            .lock()
            .unwrap()
            .push((from.to_owned(), to.to_owned(), predicate.to_owned()));
    }
    fn on_contradiction(&self, _event: ContradictionDetected) {}
    fn on_dedup_merge(&self, _surviving_id: &str, _absorbed_id: &str) {}
    fn on_stage_change(&self, stage: IngestStatus) {
        self.stages.lock().unwrap().push(stage);
    }
    fn on_ingestion_error(&self, _event: IngestionError) {}
}

impl EnrichmentEventSink for CapturingSink {
    fn on_community_updated(&self, _community_id: &str, _member_count: usize) {}
    fn on_batch_phase2_complete(&self, event: BatchPhase2Complete) {
        self.batches.lock().unwrap().push(event.batch_id);
    }
}

// ── Mode selection (§4.4) ─────────────────────────────────────────────────────

/// Resolved run mode for the smoke test.
enum SmokeMode {
    /// `KREMORY_VCR=record`: LIVE Ollama wrapped in `record(...)`. The bool
    /// signals the test body to call `provider.flush()` after Phase-2 completes.
    Live,
    /// `KREMORY_VCR=replay` or unset: deterministic `replay(...)`, no Ollama.
    Replay,
}

fn resolve_mode() -> SmokeMode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => SmokeMode::Live,
        Ok("replay") | Err(_) => SmokeMode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("golden_path_smoke.json")
}

// ── Real-Ollama construction (LIVE / record only) — mirrors llm_integration.rs ─

/// base_url from OLLAMA_BASE_URL, default localhost.
fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Chat model from OLLAMA_CHAT_MODEL, default `gemma4-e2b:latest` (model SoT,
/// llm_integration.rs:1-53). The `:latest` tag is REQUIRED so `capability_of()`
/// routes to the `FormatSchema` arm (provider.rs:156 colon-routing footgun).
fn ollama_chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4-e2b:latest".to_string())
}

/// Build the real Ollama chat provider (LIVE/record only). `.keep_alive("1h")`
/// per feedback_td024 — keep_alive thrash hurts precision.
fn real_ollama_chat() -> Arc<dyn ChatProvider> {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model(ollama_chat_model())
        .keep_alive("1h")
        .timeout_seconds(60)
        .build()
        .expect("real Ollama chat provider must build (KREMORY_VCR=record requires Ollama)");
    llm as Arc<dyn ChatProvider>
}

/// Build the real Ollama embedder (LIVE/record only) — `nomic-embed-text`, 768-dim.
fn real_ollama_embedder() -> Arc<dyn DynEmbeddingProvider> {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::embedding::EmbeddingBuilder;

    let raw: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model("nomic-embed-text")
        .build()
        .expect("real Ollama embedder must build (KREMORY_VCR=record requires Ollama)");
    Arc::new(OllamaEmbedderAdapter(raw))
}

/// Minimal local adapter bridging `autoagents_llm::embedding::EmbeddingProvider`
/// (batch `Vec<String>` → `Vec<Vec<f32>>`) → kremory's single-string
/// `EmbeddingProvider`. Mirrors `tests/helpers/ollama_adapter.rs` (kept local so
/// this binary needs no `mod helpers;`).
struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    async fn embed(&self, text: &str) -> kremory::CoreResult<Vec<f32>> {
        use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
        let mut vecs = AlLmEmbeddingProvider::embed(&*self.0, vec![text.to_string()])
            .await
            .map_err(|e| kremory::CoreError::Embedding(e.to_string()))?;
        vecs.pop().ok_or_else(|| {
            kremory::CoreError::Embedding(
                "OllamaEmbedderAdapter: embed returned empty vec".to_string(),
            )
        })
    }
}

// ── The golden path ───────────────────────────────────────────────────────────

/// Golden-path smoke test (§9 P3). Current-thread tokio flavor (plain
/// `#[tokio::test]`) per §6.3 — do NOT add `flavor = "multi_thread"`.
#[tokio::test]
async fn golden_path_smoke() {
    let _log = init_test_log("golden_path_smoke");

    let mode = resolve_mode();
    let cassette = cassette_path();

    // 1) LLM provider + embedder, mode-selected.
    //    LIVE wraps real Ollama in record(...) so one run refreshes the cassette.
    //    REPLAY reads the committed cassette (loud error if missing — that's P4).
    let (provider, embedder, embedding_dim): (
        Arc<RecordReplayChatProvider>,
        Arc<dyn DynEmbeddingProvider>,
        usize,
    ) = match mode {
        SmokeMode::Live => {
            let real = real_ollama_chat();
            let rec = Arc::new(RecordReplayChatProvider::record(
                real,
                cassette.clone(),
                ollama_chat_model(),
            ));
            (rec, real_ollama_embedder(), 768)
        }
        SmokeMode::Replay => {
            let rep = Arc::new(
                RecordReplayChatProvider::replay(cassette.clone())
                    .expect("replay cassette must load (record it via KREMORY_VCR=record — P4)"),
            );
            let emb: Arc<dyn DynEmbeddingProvider> =
                Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 });
            (rep, emb, 384)
        }
    };

    // The decorator is wired into the facade as `Arc<dyn ChatProvider>`; we keep a
    // typed `Arc<RecordReplayChatProvider>` handle too so we can call `flush()`.
    let llm: Arc<dyn ChatProvider> = provider.clone();

    // 2) Build Memory through the public facade.
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = Memory::open(dir.path().join("golden_path.db"))
        .with_llm(llm)
        .with_embedder(embedder)
        .embedding_dim(embedding_dim)
        .default_namespace(Namespace::new("golden-path-smoke"))
        .await
        .expect("Memory::open must succeed");

    // 3) Wire the capturing EnrichmentEventSink (PRIMARY Phase-2 oracle, §6.4).
    let sink = Arc::new(CapturingSink::default());

    // 4) Ingest one fixed golden episode WITH the sink. The terminal `.await`
    //    drives Phase 1 + schedules Phase 2; `wait_for_processing` below blocks
    //    on Phase 2 completion.
    let commit = mem
        .remember("Alice works at Acme Corp in London.")
        .from_chat("golden-path-session")
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await
        .expect("remember must succeed");

    // 5) Block until Phase 2 reaches a terminal state. episode_entity_id is the
    //    i64 rowid as a string on the ingest path (facade/mod.rs:544-545).
    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("episode_entity_id must be a parseable i64 rowid");

    mem.wait_for_processing(episode_id, Duration::from_secs(60))
        .await
        .expect("wait_for_processing must return Ok (Phase 2 reached Verified)");

    // 6) LIVE/record ONLY: flush the cassette AFTER wait_for_processing returns
    //    and BEFORE any cassette-entry assertion (NEW-202 — MANDATORY). The
    //    Phase-2 cassette write fires on the worker thread; the shared
    //    Arc<Mutex<CassetteState>> guarantees this test-thread flush observes it.
    if matches!(mode, SmokeMode::Live) {
        provider
            .flush()
            .expect("provider.flush() must succeed in LIVE/record mode");
    }

    // 7) STRUCTURAL-INVARIANT assertions ONLY (RISK-001). No exact name / count.

    // 7a) Pipeline reached the terminal Verified state.
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");
    let status = read_status(&tg.conn, episode_id).await;
    assert_eq!(
        status, "Verified",
        "episode_processing_status must be 'Verified' after wait_for_processing returned Ok"
    );

    // 7b) Phase 2 ran and produced at least one entity (count >= 1, NOT exact).
    assert!(
        sink.entity_count() >= 1,
        "sink must capture >= 1 on_entity_extracted event (Phase-2 oracle, §6.4); got {}",
        sink.entity_count()
    );

    // 7c) Recall returns at least one hit (non-empty, NOT an exact count / name).
    let hits = mem
        .recall("Alice")
        .raw()
        .await
        .expect("recall().raw() must succeed");
    assert!(
        !hits.is_empty(),
        "recall.raw() must return at least one hit (len >= 1, structural invariant)"
    );

    drop(dir);
}

/// Read `episode_processing_status` for a given episode id (one targeted SELECT,
/// not the API under test — mirrors wait_for_processing.rs:69-87).
async fn read_status(conn: &libsql::Connection, episode_id: i64) -> String {
    let mut rows = conn
        .query(
            "SELECT episode_processing_status FROM episodes WHERE id = ?1",
            libsql::params![episode_id],
        )
        .await
        .expect("status query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<String>(0)
        .expect("status col")
}
