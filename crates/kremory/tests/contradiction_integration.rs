//! Real-LLM contradiction + bi-temporal supersession integration test (Suite Item 2).
//!
//! Closes the gap in `.ai-docs/research/real-llm-coverage-inventory-2026-06-29.md` §4
//! Item 2: `background_integration.rs::rql_graph_contradiction_invalidates_superseded_fact`
//! proves contradiction + supersession with a SCRIPTED LLM (deterministic JSON queue).
//! This test runs the SAME pipeline with a REAL Ollama model so the full path is
//! exercised on natural language: prose → real LLM fact extraction → real LLM
//! contradiction classification (TwoPoolDetector) → `invalidate_fact` → `invalid_at` set.
//!
//! # Why the `Engine::ingest_with` path (not the `Memory` facade)
//!
//! The facade `Memory::remember(...).await` defers fact/relationship extraction
//! (Phase 2b) to the background worker and returns an `EpisodeCommit` with no
//! `invalidated_fact_ids`. Asserting contradiction through the facade requires racing
//! Phase-2b completion. `Engine::ingest_with` runs the full pipeline SYNCHRONOUSLY and
//! returns `IngestionResult { inserted_fact_ids, invalidated_fact_ids }` directly — the
//! exact surface the inventory's DONE bar names, and the surface the scripted test uses.
//!
//! # Why a literal-valued fact ("app go_live Friday")
//!
//! The contradiction resolver compares FACTS — triples with the SAME (subject, predicate)
//! and a changed object. Entity→entity relationships ("Alice works at Acme Corp", both
//! nodes) become graph EDGES, not literal facts, and never enter the fact pool. So this
//! mirrors the proven scripted shape `(app, go_live, friday→monday)`.
//!
//! # Model
//!
//! `gemma4:e4b` with `reasoning(false)` — validated-best extraction config
//! (`facade/providers.rs`; F1 84 / recall 90%). `gemma4-e2b` extracts 0 entities and is
//! UNUSABLE. Override via `OLLAMA_CHAT_MODEL`.
//!
//! # Gate
//!
//! `#[ignore]`. Requires live Ollama with `gemma4:e4b` + (mock embeddings used, so no
//! embed model needed):
//!   cargo test -p kremory --features llm-integration --test contradiction_integration \
//!       -- --ignored --nocapture

#![cfg(feature = "llm-integration")]
// Test files use expect/unwrap/panic as intentional assertion mechanisms.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use chrono::Utc;

use kremory::core::config::PipelineConfig;
use kremory::core::extraction::IntegerIdLlmExtractor;
use kremory::core::ingest::{Engine, EngineNewParams, IngestWithParams, SourceParams};
use kremory::core::provider::MockEmbeddingProvider;
use kremory::core::schema::TemporalGraph;

/// Chat model from `OLLAMA_CHAT_MODEL`, default `qwen2.5:14b`.
///
/// Contradiction detection needs RELATIONSHIP (triple) extraction, not just entity
/// extraction. `qwen2.5:14b` is the model that `label_precision_benchmark.rs` (which
/// asserts `min_relationships`) and the LongMemEval harness default to — it reliably
/// produces Stage-2 relationship triples. `gemma4:e4b` is validated for *entity*
/// precision (the model ladder is entity-F1) but extracts ~0 relationships on short
/// prose, so it cannot exercise the fact-level contradiction resolver.
fn ollama_chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string())
}

/// base_url from `OLLAMA_BASE_URL`, default localhost.
fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Ingest a literal-valued fact and a contradicting update through a real LLM;
/// assert the contradiction resolver invalidates the prior fact (bi-temporal
/// supersession: `invalid_at` + `expired_at` set, prior fact never deleted).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn real_llm_contradiction_supersedes_prior_fact() {
    let graph = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory"),
    );
    // TD-167: contradiction detection is now DEFAULT-OFF (it supersedes
    // set-valued facts). This test asserts the capability, so it opts in.
    let config = PipelineConfig::builder()
        .contradiction_detection_enabled(true)
        .build()
        .expect("PipelineConfig");
    let dim = config.embedding_dim.0;

    let model = ollama_chat_model();
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(ollama_base_url())
        .model(&model)
        // kremory extraction is structured-output, not reasoning — disabling thinking
        // BOTH raises quality and cuts latency (validated 2026-06-24).
        .reasoning(false)
        .keep_alive("1h")
        .timeout_seconds(120)
        .build()
        .expect("real Ollama chat provider (needs gemma4:e4b pulled)");

    // Mock embeddings: the contradiction LLM call fires off pool_a (same subject+predicate
    // exact match), which does not depend on embedding similarity. Recall quality is a
    // separate suite item. This isolates the contradiction-detection capability.
    let embedder = Arc::new(MockEmbeddingProvider::new(dim));

    let engine = Engine::new(EngineNewParams {
        graph: Arc::clone(&graph),
        llm: Arc::clone(&llm),
        embedder,
        config,
        model: Some(model),
    });
    let extractor = IntegerIdLlmExtractor::new(Arc::clone(&llm));

    // ── Utterance 1: establish (Sarah Martinez, works_at, Acme Corporation) ─────
    // RICH prose, not a single sentence: real-LLM relationship (Stage-2) extraction
    // is context-hungry — sparse prose yields entities but 0 triples (verified vs
    // `label_precision_benchmark`, which extracts 10 facts from the rich mock_interview
    // corpus but 0 from 1-sentence inputs). The repeated "works at Acme Corporation"
    // phrasing biases the extractor toward a stable `works_at` predicate so utterance 2
    // lands in the same (subject, predicate) contradiction pool.
    let r1 = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Sarah Martinez joined Acme Corporation in 2019. \
                       Sarah Martinez works at Acme Corporation as a senior backend engineer, \
                       where she leads the payments platform team. \
                       Sarah Martinez has worked at Acme Corporation for several years.",
                reference_time: Some(Utc::now()),
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
        )
        .await
        .expect("ingest utterance 1");

    eprintln!(
        "[contradiction] u1: inserted_facts={} invalidated={}",
        r1.inserted_fact_ids.len(),
        r1.invalidated_fact_ids.len()
    );
    assert!(
        !r1.inserted_fact_ids.is_empty(),
        "utterance 1 should extract >=1 fact (Sarah Martinez / works_at / Acme Corporation); \
         got 0. The real LLM may not have produced a relationship triple — inspect with \
         --nocapture. r1 = {r1:?}"
    );
    assert!(
        r1.invalidated_fact_ids.is_empty(),
        "utterance 1 has no prior state, so nothing should be invalidated"
    );

    // ── Utterance 2: contradict — same subject+predicate, new value (Monday) ────
    let r2 = engine
        .ingest_with(
            &extractor,
            IngestWithParams {
                text: "Sarah Martinez has now left Acme Corporation. \
                       As of this month, Sarah Martinez works at Globex Industries, \
                       where she leads the cloud infrastructure division. \
                       Sarah Martinez no longer works at Acme Corporation — she works at Globex Industries now.",
                reference_time: Some(Utc::now()),
                group_id: None,
                content_type: None,
                source_params: SourceParams::default(),
            },
        )
        .await
        .expect("ingest utterance 2");

    eprintln!(
        "[contradiction] u2: inserted_facts={} invalidated={}",
        r2.inserted_fact_ids.len(),
        r2.invalidated_fact_ids.len()
    );

    // PRIMARY oracle: the contradiction resolver invalidated the prior (Friday) fact.
    // `invalidated_fact_ids` is populated by `invalidate_fact_with_reason`, which sets
    // both `invalid_at` and `expired_at` — so a non-empty list IS the supersession proof.
    assert!(
        !r2.invalidated_fact_ids.is_empty(),
        "utterance 2 should invalidate the prior Friday fact via contradiction detection; \
         got 0 invalidated. Either the LLM extracted a different (subject, predicate) for \
         the two utterances (no pool overlap), or the TwoPoolDetector classifier did not \
         flag Friday→Monday as a contradiction. Inspect with --nocapture. r2 = {r2:?}"
    );

    // SECONDARY oracle: the invalidated fact still exists in history (supersede never
    // deletes) with both bi-temporal invalidation columns set.
    let invalidated_id = r2.invalidated_fact_ids[0];
    let active = graph
        .facts_at(Utc::now())
        .await
        .expect("facts_at(now) must succeed");
    assert!(
        !active.is_empty(),
        "at least one active fact (the surviving Monday fact) should remain"
    );
    let subject = active[0].subject_id.clone();
    let history = graph
        .entity_history(&subject)
        .await
        .expect("entity_history must succeed");
    let invalidated = history
        .iter()
        .find(|f| f.id == invalidated_id)
        .expect("invalidated fact must still exist in entity_history — supersede never deletes");
    assert!(
        invalidated.invalid_at.is_some() && invalidated.expired_at.is_some(),
        "superseded fact must carry invalid_at + expired_at; got: {invalidated:?}"
    );

    eprintln!(
        "[contradiction] PASS — real LLM superseded fact#{invalidated_id} ({} {} {:?}); \
         {} active fact(s) remain",
        invalidated.subject_id,
        invalidated.predicate,
        invalidated.object_value,
        active.len(),
    );
}
