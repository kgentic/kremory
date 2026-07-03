//! Whole-project end-to-end test — real ingest → dream reconciliation (VCR).
//!
//! Governing docs: ADR-063 (§3 Site #5 acronym/nickname, §4 Site #3 type-registry
//! collapse), `.ai-docs/specs/dream-adversarial-corpora-and-metrics-2026-07-02.md`.
//!
//! WHY THIS TEST EXISTS (the integration seam it closes):
//! The dream identity passes (Site #3 + Site #5, enabled by default as of
//! 2026-07-03) are deeply validated in isolation by the metrics harnesses
//! (`dream_metrics_harness{,_site3}.rs`) — but those PLANT entities directly into
//! the graph. NO existing test proves a real `remember()`-ingested episode flows
//! all the way through phase-1 (embed+NER) + phase-2 (LLM relationships) and is
//! then correctly reconciled by `mem.dream()`. THIS test drives the ENTIRE public
//! pipeline — `remember()` → phase1/phase2 → `recall()` → `mem.dream()` — on
//! ingest-produced entities, with the real gemma4:e4b model (`.think(false)`) +
//! real nomic embedder recorded once to VCR cassettes and replayed deterministically.
//!
//! # Two run modes (mirrors golden_path_smoke §4.4 + dream_e2e_real_llm), by `KREMORY_VCR`:
//!   * `KREMORY_VCR=record` → LIVE: real Ollama chat wrapped in
//!     `RecordReplayChatProvider::record(...)` AND real nomic wrapped in
//!     `RecordReplayEmbedder::record(...)`; one run refreshes BOTH committed
//!     cassettes (chat + embeddings). `provider.flush()` + `emb_vcr.flush()` are
//!     MANDATORY after the last background write and before assertions (NEW-202).
//!   * `KREMORY_VCR=replay` OR unset → REPLAY: fully deterministic, NO Ollama.
//!     BOTH cassettes are replayed — the chat cassette AND the REAL recorded nomic
//!     vectors — so recall (embedding-driven) and the Site #3 type-collapse cosine
//!     gate reproduce faithfully offline. A missing cassette entry is a LOUD error.
//!
//! # Determinism note (multi-episode ingest):
//! The episodes are DISTINCT, non-overlapping domains on purpose. Overlapping
//! (semantically-similar) facts across episodes trigger an ingest-time
//! contradiction-detection LLM call whose retrieved-context ordering is NOT yet
//! VCR-deterministic (an un-hardened ingest-path gap, distinct from the already
//! hardened dream path). Distinct domains avoid that call, so multi-episode replay
//! is byte-stable. Verified: 2× replay produces identical dream-summary counters.
//!
//! # Assertions are STRUCTURAL INVARIANTS ONLY (RISK-001, mirrors golden_path).
//! gemma4:e4b is nondeterministic; asserting exact entity names or merge counts
//! would flake on every cassette re-record. This test asserts only:
//!   1. every ingested episode reaches `episode_processing_status == "Verified"`;
//!   2. phase-2 produced >= 1 entity (sink oracle);
//!   3. `recall().raw()` is non-empty after ingest;
//!   4. `mem.dream()` returns `Ok` with a well-formed `DreamSummary`;
//!   5. SAFETY — dream did NOT catastrophically over-merge: the entity set is not
//!      collapsed below the count of genuinely-distinct concepts (structural
//!      zero-false-merge floor), and recall still serves results post-dream.

#![cfg(feature = "llm-smoke")]
// Test files use expect/unwrap/panic as intentional assertion mechanisms
// (project-wide test convention — see golden_path_smoke.rs:42-46).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod support;

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

use support::test_log::init_test_log;

// ── Capturing sink (phase-2 oracle, mirrors golden_path §6.4) ─────────────────

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

// ── Mode selection (mirrors golden_path §4.4) ─────────────────────────────────

enum Mode {
    Live,
    Replay,
}

fn resolve_mode() -> Mode {
    match std::env::var("KREMORY_VCR").as_deref() {
        Ok("record") => Mode::Live,
        Ok("replay") | Err(_) => Mode::Replay,
        Ok(other) => panic!("KREMORY_VCR must be record|replay, got {other:?}"),
    }
}

fn cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("whole_project_e2e.json")
}

fn embedding_cassette_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join("whole_project_e2e.embeddings.json")
}

/// Embedding record/replay (TD-093), copied from `dream_e2e_real_llm.rs`. Recall
/// and the Site #3 type-collapse gate are SEMANTIC (cosine) — a null/hash embedder
/// gives spurious results, so replay must reuse the REAL nomic vectors captured at
/// record time. record: delegate to real nomic + memoise each text→vector; replay:
/// look up offline (loud MISS → re-record).
struct RecordReplayEmbedder {
    /// `Some` in record mode (real nomic), `None` in replay.
    inner: Option<Arc<dyn DynEmbeddingProvider>>,
    cache: Mutex<std::collections::HashMap<String, Vec<f32>>>,
    path: std::path::PathBuf,
}

impl RecordReplayEmbedder {
    fn record(inner: Arc<dyn DynEmbeddingProvider>, path: std::path::PathBuf) -> Self {
        Self {
            inner: Some(inner),
            cache: Mutex::new(std::collections::HashMap::new()),
            path,
        }
    }

    fn replay(path: std::path::PathBuf) -> Self {
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "embedding cassette must load ({}): {e} — re-record via KREMORY_VCR=record (TD-093)",
                path.display()
            )
        });
        let map: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_str(&raw).expect("embedding cassette must be valid JSON");
        Self {
            inner: None,
            cache: Mutex::new(map),
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
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        async move {
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
                    "embedding cassette MISS for {text:?} — re-record via KREMORY_VCR=record (TD-093)"
                ))),
            }
        }
    }
}

fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_chat_model() -> String {
    // Default to the project's benchmarked chat model (facade default + metrics
    // harness): `gemma4:e4b` with reasoning disabled. NOT golden_path's
    // `gemma4-e2b:latest` (that small model is "UNUSABLE — 0 ents" for real
    // extraction, llm_integration.rs model table).
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string())
}

fn real_ollama_chat() -> Arc<dyn ChatProvider> {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;

    // `.think(false)`: kremory extraction is structured-output, not reasoning.
    // Reasoning ON blows past the extraction ladder's 30s per-arm ttft budget
    // (gemma4:e4b 44s/F1-75 thinking-on → 16s/F1-84 think:false, benchmark
    // 2026-06-24). Mirrors dream_metrics_harness::build_provider + facade
    // `with_ollama`. `.timeout_seconds(180)` matches the harness.
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model(ollama_chat_model())
        .think(false)
        .keep_alive("1h")
        .timeout_seconds(180)
        .build()
        .expect("real Ollama chat provider must build (KREMORY_VCR=record requires Ollama)");
    llm as Arc<dyn ChatProvider>
}

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

struct OllamaEmbedderAdapter(Arc<autoagents_llm::backends::ollama::Ollama>);

impl kremory::EmbeddingProvider for OllamaEmbedderAdapter {
    async fn embed(&self, text: &str) -> kremory::CoreResult<Vec<f32>> {
        use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
        // nomic-embed-text REQUIRES a task prefix (TD-097): without it, short
        // strings collapse to near-identical vectors (cosine ~1.0), breaking the
        // semantic gates. `search_document:` is nomic's document-embedding prefix.
        // RecordReplayEmbedder caches under the ORIGINAL `text`, so replay lookup
        // is unaffected — the prefix is internal to the nomic call.
        let prefixed = format!("search_document: {text}");
        let mut vecs = AlLmEmbeddingProvider::embed(&*self.0, vec![prefixed])
            .await
            .map_err(|e| kremory::CoreError::Embedding(e.to_string()))?;
        vecs.pop().ok_or_else(|| {
            kremory::CoreError::Embedding(
                "OllamaEmbedderAdapter: embed returned empty vec".to_string(),
            )
        })
    }
}

// ── Ingest corpus ─────────────────────────────────────────────────────────────
//
// DISTINCT-DOMAIN episodes (smoke-one-before-batch: the single-episode smoke is
// proven deterministic first; this is the batch step). Each episode is about a
// SEPARATE, non-overlapping domain (a company, a river, a scientist, a recipe) so
// that ingest fires NO cross-episode contradiction call — that LLM call's
// retrieved-context ordering is the source of multi-episode replay
// non-determinism (an un-hardened ingest-path VCR gap, distinct from the already
// hardened dream path). The extra entities give the dream phase real work to do
// (Pass-0 type discovery + reclassify). Ground truth is NOT hard-asserted
// (extraction is nondeterministic) — the test asserts structural invariants +
// zero false merge and LOGS whatever the dream passes actually did.
const EPISODES: &[(&str, &str)] = &[
    (
        "e2e-company",
        "Acme Corporation, founded by Jane Smith in Ohio, manufactures industrial robots \
         for automotive assembly lines.",
    ),
    (
        "e2e-river",
        "The Amazon River flows over six thousand kilometres through Brazil and Peru before \
         it empties into the Atlantic Ocean.",
    ),
    (
        "e2e-scientist",
        "Marie Curie, a physicist born in Warsaw, was awarded Nobel Prizes in both Physics \
         and Chemistry for her research on radioactivity.",
    ),
    (
        "e2e-recipe",
        "The banana bread recipe calls for two cups of flour, a teaspoon of baking soda, \
         and three ripe bananas mashed into the batter.",
    ),
];

// ── The whole-project pipeline ────────────────────────────────────────────────

#[tokio::test]
#[ignore = "whole_project_e2e: requires Ollama in record mode, or the committed cassette \
            in replay mode. Run explicitly: KREMORY_VCR=record cargo test -p kremory \
            --features llm-smoke --test whole_project_e2e -- --ignored --nocapture"]
async fn whole_project_ingest_to_dream() {
    let _log = init_test_log("whole_project_e2e");

    let mode = resolve_mode();
    let cassette = cassette_path();

    // 1) LLM chat + embedder, both VCR-backed so replay faithfully reproduces the
    //    recorded pipeline (real nomic vectors → recall + Site #3 cosine gate work
    //    deterministically offline). Mirrors dream_e2e_real_llm.rs (chat + emb VCR).
    let (provider, emb_vcr): (Arc<RecordReplayChatProvider>, Arc<RecordReplayEmbedder>) = match mode
    {
        Mode::Live => {
            let real_chat = real_ollama_chat();
            let rec = Arc::new(RecordReplayChatProvider::record(
                real_chat,
                cassette.clone(),
                ollama_chat_model(),
            ));
            let emb = Arc::new(RecordReplayEmbedder::record(
                real_ollama_embedder(),
                embedding_cassette_path(),
            ));
            (rec, emb)
        }
        Mode::Replay => {
            let rep = Arc::new(
                RecordReplayChatProvider::replay(cassette.clone())
                    .expect("replay cassette must load (record it via KREMORY_VCR=record)"),
            );
            let emb = Arc::new(RecordReplayEmbedder::replay(embedding_cassette_path()));
            (rep, emb)
        }
    };

    let llm: Arc<dyn ChatProvider> = provider.clone();
    let embedder: Arc<dyn DynEmbeddingProvider> = emb_vcr.clone();
    let embedding_dim: usize = 768;

    // 2) Build Memory through the public facade.
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = Memory::open(dir.path().join("whole_project_e2e.db"))
        .with_llm(llm)
        .with_embedder(embedder)
        .embedding_dim(embedding_dim)
        .default_namespace(Namespace::new("whole-project-e2e"))
        .await
        .expect("Memory::open must succeed");

    let sink = Arc::new(CapturingSink::default());

    // 3) Ingest every adversarial episode through the REAL pipeline (phase1+phase2).
    for (session, text) in EPISODES {
        let commit = mem
            .remember(*text)
            .from_chat(*session)
            .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
            .await
            .unwrap_or_else(|e| panic!("remember({session}) must succeed: {e:?}"));

        let episode_id: i64 = commit
            .episode_entity_id
            .parse()
            .expect("episode_entity_id must be a parseable i64 rowid");

        mem.wait_for_processing(episode_id, Duration::from_secs(90))
            .await
            .unwrap_or_else(|e| {
                panic!("wait_for_processing({session}) must reach Verified: {e:?}")
            });

        // ASSERT 1: pipeline reached the terminal Verified state for this episode.
        let tg = mem
            .temporal_graph_for_test()
            .expect("temporal_graph must be set on the builder path");
        let status = read_status(&tg.conn, episode_id).await;
        assert_eq!(
            status, "Verified",
            "episode {session} must be 'Verified' after wait_for_processing returned Ok"
        );
    }

    // ASSERT 2: phase-2 ran and produced at least one entity PER episode. Each of
    // the EPISODES.len() distinct-domain episodes names >= 1 clear entity, so a
    // floor of EPISODES.len() (not a trivial >= 1) catches a phase-2 extraction
    // that silently degrades to firing on only a subset of episodes.
    assert!(
        sink.entity_count() >= EPISODES.len(),
        "sink must capture >= {} on_entity_extracted events (one per distinct-domain \
         episode); got {}",
        EPISODES.len(),
        sink.entity_count()
    );

    // Entity population BEFORE dream (structural zero-false-merge baseline).
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set");
    let entities_before = count_entities(&tg.conn).await;

    // ASSERT 3: recall is non-empty after ingest (pipeline serves reads e2e).
    let hits_before = mem
        .recall("Acme")
        .raw()
        .await
        .expect("recall().raw() must succeed after ingest");
    assert!(
        !hits_before.is_empty(),
        "recall.raw() must return >= 1 hit after ingest (structural invariant)"
    );

    // 4) Run the dream phase on ingest-produced entities. DreamOpts::default()
    //    now enables Site #3 (type-registry collapse) + Site #5 (acronym/nickname
    //    recall). This is the integration seam under test.
    let summary = mem
        .dream()
        .await
        .expect("mem.dream() must succeed end-to-end on ingest-produced entities");

    // record mode ONLY: flush the chat cassette AFTER dream's background LLM calls
    // complete and BEFORE assertions (NEW-202, mirrors golden_path + dream_e2e).
    if matches!(mode, Mode::Live) {
        provider
            .flush()
            .expect("provider.flush() must succeed in record mode");
        // Persist the embedding cassette too (TD-093) so replay is fully offline.
        emb_vcr.flush();
    }

    eprintln!(
        "[whole-project-e2e] entities_before={entities_before} \
         types_discovered={} aliases_resolved={} entities_reclassified={} \
         canonicalization_merges={} cross_episode_merges={} \
         acronym_nickname_merges={} type_registry_merges={} \
         consistency_check_corrected={} warnings={:?} duration_ms={}",
        summary.types_discovered.len(),
        summary.aliases_resolved,
        summary.entities_reclassified,
        summary.canonicalization_merges,
        summary.cross_episode_merges,
        summary.acronym_nickname_merges,
        summary.type_registry_merges,
        summary.consistency_check_corrected,
        summary.warnings,
        summary.duration_ms,
    );

    // ASSERT 4: no dream pass FAILED. Every pass templates its failure warning as
    // "Dream <pass> failed: {e}" (facade/dream.rs — e.g. the acronym_nickname_recall
    // and type_registry_collapse Err branches). Matching that exact code-emitted
    // template (NOT a guessed "panic"/"corrupt" blocklist) means a real pass failure
    // — LLM error, timeout, DB error on ANY of the 5 passes incl. the two enabled
    // identity passes — fails this assert instead of silently passing. A benign
    // degraded-but-Ok notice (which does not contain "failed:") is still allowed.
    assert!(
        !summary
            .warnings
            .iter()
            .any(|w| w.to_lowercase().contains("failed:")),
        "a dream pass reported failure (warnings contain a 'failed:' template): {:?}",
        summary.warnings
    );

    // ASSERT 5 (SAFETY — structural over-merge floor, dual guard).
    //
    // `DreamSummary` now surfaces the Site #3 (type_registry_merges) and Site #5
    // (acronym_nickname_merges) merge counts (ADR-063 §3/§4 observability), so this
    // test observes the two enabled identity passes directly. `merge_actions` sums
    // every DreamSummary merge/alias counter — L7 alias-resolution, canonicalize,
    // consolidation (honest-zero), AND Site #3/#5 — so an over-merge by ANY pass is
    // covered. Deep per-pair Site #3/#5 CORRECTNESS is still the metrics harnesses'
    // job (dream_metrics_harness{,_site3}.rs); this is the structural over-merge
    // ceiling. Second guard: the 4 episodes are non-overlapping domains, so a correct
    // dream keeps >= one surviving entity per domain — a catastrophic collapse below
    // EPISODES.len() is a false-merge failure.
    let entities_after = count_entities(&tg.conn).await;
    let merge_actions = summary.aliases_resolved
        + summary.canonicalization_merges
        + summary.cross_episode_merges
        + summary.acronym_nickname_merges
        + summary.type_registry_merges;
    assert!(
        (merge_actions as i64) <= entities_before,
        "dream reported {merge_actions} total merge/alias actions but only \
         {entities_before} entities existed before dream — impossible without corruption"
    );
    assert!(
        entities_after >= EPISODES.len() as i64,
        "dream over-merged: {entities_before} entities across {} distinct domains \
         collapsed to {entities_after} (< one per domain) — a false-merge failure",
        EPISODES.len(),
    );

    // ASSERT 6: the graph still serves reads after dream (not destroyed).
    let hits_after = mem
        .recall("Acme")
        .raw()
        .await
        .expect("recall().raw() must succeed after dream");
    assert!(
        !hits_after.is_empty(),
        "recall.raw() must still return >= 1 hit after dream (graph not destroyed)"
    );

    drop(dir);
}

/// Count rows in the `entities` table (targeted SELECT, not the API under test).
async fn count_entities(conn: &libsql::Connection) -> i64 {
    let mut rows = conn
        .query("SELECT COUNT(*) FROM entities", ())
        .await
        .expect("entity count query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

/// Read `episode_processing_status` for a given episode id (mirrors golden_path).
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
