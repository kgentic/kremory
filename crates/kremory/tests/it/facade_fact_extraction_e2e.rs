//! Facade front-door fact-extraction E2E (DoD for the ADR-056 extractor-default fix).
//!
//! Verifies the PRODUCTION consumer path — `Memory::open().with_llm().with_embedder()` →
//! `open_graph` → `Engine::new` (default extractor = IntegerIdLlmExtractor since ADR-056) —
//! actually extracts relationship facts from real prose via a real LLM. This is the end-to-end
//! link the `ingest_with` probe did NOT cover (the probe called the extractor directly; this
//! drives `remember()` through the two-phase facade scheduling).
//!
//! Oracle: the `on_edge_added` Phase-2b sink callback (fires per relationship written),
//! polled after `remember().await` to tolerate inline-vs-deferred Phase-2b timing.
//!
//! Model: `OLLAMA_CHAT_MODEL` (default `qwen2.5:14b` — the proven fact extractor: 16 facts on
//! mock_interview). `gemma4:e4b` (the `Memory::with_ollama` default) fact-extraction via
//! IntegerId is tracked as a separate follow-up.
//!
//!   OLLAMA_HOST=http://localhost:11434 OLLAMA_CHAT_MODEL=qwen2.5:14b \
//!     cargo test -p kremory --features llm-integration --test it facade_fact_extraction_e2e:: \
//!       -- --ignored --nocapture

#![cfg(feature = "llm-integration")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::embedding::EmbeddingBuilder;

use kremory::core::error::IngestStatus;
use kremory::core::sink::{
    ContradictionDetected, IngestEventSink, IngestionError, OnEdgeAddedParams,
};
use kremory::memory::events::{BatchPhase2Complete, EnrichmentEventSink};
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use crate::helpers::ollama_adapter::OllamaEmbedderAdapter;

#[derive(Default)]
struct EdgeCapturingSink {
    entities: Mutex<Vec<(String, String)>>,
    edges: Mutex<Vec<(String, String, String)>>,
    complete: Mutex<bool>,
}

impl IngestEventSink for EdgeCapturingSink {
    fn on_entity_extracted(&self, entity_id: &str, name: &str) {
        self.entities
            .lock()
            .unwrap()
            .push((entity_id.to_owned(), name.to_owned()));
    }
    fn on_edge_added(&self, p: OnEdgeAddedParams<'_>) {
        self.edges.lock().unwrap().push((
            p.from_entity_id.to_owned(),
            p.to_entity_id.to_owned(),
            p.predicate.to_owned(),
        ));
    }
    fn on_contradiction(&self, _e: ContradictionDetected) {}
    fn on_dedup_merge(&self, _s: &str, _a: &str) {}
    fn on_stage_change(&self, stage: IngestStatus) {
        if matches!(stage, IngestStatus::Complete) {
            *self.complete.lock().unwrap() = true;
        }
    }
    fn on_ingestion_error(&self, _e: IngestionError) {}
}

impl EnrichmentEventSink for EdgeCapturingSink {
    fn on_community_updated(&self, _c: &str, _m: usize) {}
    fn on_batch_phase2_complete(&self, _e: BatchPhase2Complete) {}
}

fn base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}
fn chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string())
}

/// The production facade path extracts relationship facts from real prose.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn facade_remember_extracts_facts_via_real_llm() {
    let dir = tempfile::tempdir().expect("tempdir");

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(base_url())
        .model(chat_model())
        .reasoning(false)
        .keep_alive("1h")
        .timeout_seconds(120)
        .build()
        .expect("ollama chat provider");
    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(base_url())
        .model("nomic-embed-text")
        .build()
        .expect("ollama embedder");
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_emb));

    // PROD facade path: Memory::open().with_llm() → open_graph → Engine::new (IntegerId default).
    // `.with_model_id(...)` is REQUIRED for fact extraction: raw `.with_llm()` leaves
    // model_id None → capability detection picks the PromptOnly arm → structured triplet
    // extraction (stages 2-3) degrades and facts come back empty (entities still land).
    // Consumer footgun — `with_ollama_at_model` threads this automatically; raw `with_llm`
    // does not. See builder.rs:492.
    let mem = Memory::open(dir.path().join("kremory.db"))
        .with_llm(llm as Arc<dyn ChatProvider>)
        .with_model_id(chat_model())
        .with_embedder(emb)
        .embedding_dim(768)
        .default_namespace(Namespace::new("e2e"))
        .await
        .expect("Memory::open facade must succeed");

    let sink = Arc::new(EdgeCapturingSink::default());

    // Use the SAME rich corpus the Engine `ingest_with` probe used (mock_interview → 16
    // facts). Running it through the FACADE isolates facade-vs-engine: if facts land here,
    // the facade works and any prior 0 was prose-shape; if not, it's a real facade gap.
    let corpus = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kremory-eval/fixtures/mock_interview.txt"),
    )
    .expect("read mock_interview.txt");

    mem.remember(corpus)
        .from_chat("e2e-session")
        .with_event_sink(sink.clone() as Arc<dyn EnrichmentEventSink>)
        .await
        .expect("remember must succeed");

    // Wait for Phase-2b (fact/relationship extraction) to complete, tolerating
    // inline-vs-deferred timing.
    for _ in 0..120 {
        if *sink.complete.lock().unwrap() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let entities = sink.entities.lock().unwrap().clone();
    assert!(
        !entities.is_empty(),
        "facade should extract >=1 entity from the prose"
    );

    // DoD oracle: query the FACTS table directly (not the on_edge_added mention-edge
    // callback, which fires for episodic entity mentions and would pass on mentions alone).
    // The probe counted `inserted_fact_ids`; the facade equivalent is `facts_at`.
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (run with --features llm-integration,test-utils)");
    // Retry-poll facts_at in case Phase-2b fact writes lag the Complete signal slightly.
    let mut facts = graph.facts_at(chrono::Utc::now()).await.expect("facts_at");
    for _ in 0..20 {
        if !facts.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        facts = graph.facts_at(chrono::Utc::now()).await.expect("facts_at");
    }

    // Diagnostic: facts INCLUDING expired/invalidated (entity_history) vs ACTIVE only
    // (facts_at). If history >> active, facts were written then immediately invalidated /
    // expired (a dedup/contradiction bug); if both ~0, facts were never written.
    let mut history_total = 0usize;
    for (eid, name) in &entities {
        for key in [eid.as_str(), name.as_str()] {
            if let Ok(h) = graph.entity_history(key).await {
                history_total += h.len();
            }
        }
    }

    eprintln!(
        "[facade-e2e] entities={} active_facts={} facts_in_history(incl expired)~={} phase2b_complete={}",
        entities.len(),
        facts.len(),
        history_total,
        *sink.complete.lock().unwrap()
    );
    for f in &facts {
        eprintln!(
            "[facade-e2e]   fact#{}: {} --{}--> obj_id={:?} obj_value={:?}",
            f.id, f.subject_id, f.predicate, f.object_id, f.object_value
        );
    }

    // The DoD assertion: the production facade actually extracts SEMANTIC relationship facts
    // (was 0 under the graphiti LlmExtractor default before ADR-056). A fact whose predicate
    // is a real relationship (not a bare episodic "mention") must be present.
    let semantic = facts
        .iter()
        .filter(|f| f.predicate != "mention" && !f.predicate.is_empty())
        .count();
    assert!(
        semantic >= 1,
        "facade should extract >=1 semantic relationship fact (e.g. works_at / manages / \
         reports_to) via the real LLM through the production path (IntegerId default). \
         Got {semantic} semantic facts among {total} total. Facts: {facts:?}",
        total = facts.len()
    );

    // ADR-057 correctness oracle: the prior `semantic >= 1` count passed even when
    // every fact's subject was CORRUPTED to one wrong entity ("amazon robotics") by
    // the cosine-only L4 over-merge — the count is blind to subject correctness.
    // mock_interview is Ria's interview: the LLM reliably extracts `Ria` as the
    // subject of her own facts (verified spike_td080_self_loop). So a correct graph
    // MUST have ≥1 fact whose subject is the normalized protagonist name "ria".
    // Under the pre-fix bug this is `amazon robotics` for ALL facts → this fails.
    let ria_is_a_subject = facts.iter().any(|f| f.subject_id == "ria");
    let distinct_subjects: std::collections::BTreeSet<&str> =
        facts.iter().map(|f| f.subject_id.as_str()).collect();
    assert!(
        ria_is_a_subject,
        "subject corruption (ADR-057): expected the protagonist 'ria' to be the subject of \
         >=1 fact, but no fact has subject_id=='ria'. This is the L4 over-merge signature — \
         all subjects collapsed to one wrong entity. Distinct subjects seen: {distinct_subjects:?}. \
         Facts: {facts:?}"
    );
}
