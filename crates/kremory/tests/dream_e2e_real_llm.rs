#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Real-LLM validation of the reconciled dream pass chain — minimal fixture.
//!
//! Governing spec: `.ai-docs/specs/dream-phase-reconciliation-v2-2026-06-30.md`
//! (§D3 canonical 5-pass ordering). Phase 6 DoD E2E, pulled forward to validate
//! Phases 1-3 against a REAL model.
//!
//! Design note: this plants a MINIMAL fixture (two catch-all entities with
//! semantically clear names) directly via the graph rather than ingesting a
//! corpus. That isolates the dream passes to ONE real `mem.dream()` call (~tens
//! of seconds) instead of paying for real-LLM ingest extraction of a corpus —
//! same smoke-one-first discipline applied to the fixture, not just the build.
//!
//! Proves, with a real model, that `mem.dream()` (restructured in Phase 1, wired
//! in Phases 2-3):
//!   1. runs the full §D3 chain (all pass counters fire),
//!   2. does REAL work — reclassify types the catch-all entities
//!      (`entities_reclassified >= 1`), the load-bearing "not running blind" check,
//!   3. hard-fails no pass (empty `warnings`).
//!
//! Run: `cargo test -p kremory --features llm-integration,test-utils --test dream_e2e_real_llm -- --ignored --nocapture`
//! Requires: Ollama at localhost:11434 with `gemma4:e4b` + `nomic-embed-text`.
//!
//! Tier: currently tier-3 (`llm-integration` + `#[ignore]`, live-Ollama only).
//! Promotion to the project's tier-2 (`--features llm-smoke`, offline via a
//! KREMORY_VCR cassette like `golden_path_smoke.rs`) is tracked as TD-093 — do
//! NOT naively re-gate on `llm-smoke` without recording a cassette first (it
//! would fail wherever Ollama is absent).

#![cfg_attr(not(feature = "llm-integration"), allow(dead_code, unused_imports))]

use std::sync::Arc;

use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

#[cfg(feature = "llm-integration")]
mod helpers;

/// Plant a catch-all (`entity_type_id = 0`, `Phase1Ner`) entity whose `id` is a
/// semantically clear name — reclassify's `catch_all_cascade` arm selects it and
/// a real LLM can type it. Mirrors `tests/phase_e_reclassify.rs::insert_entity`.
async fn plant_catch_all(graph: &TemporalGraph, id: &str, group_id: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, group_id, entity_type_id, entity_type_source, ner_confidence, \
              recorded_at, updated_at, entity_type_assigned_at) \
             VALUES (?1, ?2, 0, 'Phase1Ner', 0.9, ?3, ?3, ?3)",
            libsql::params![id.to_string(), group_id.to_string(), now],
        )
        .await
        .expect("plant catch-all entity");
}

/// Map of entity id → entity_type_id for the group.
async fn type_ids(graph: &TemporalGraph, group_id: &str) -> Vec<(String, i64)> {
    graph
        .list_entities_in_group(group_id)
        .await
        .expect("list entities")
        .iter()
        .map(|e| (e.id.clone(), i64::from(e.entity_type_id)))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real LLM — run explicitly with --features llm-integration,test-utils --ignored"]
#[cfg(feature = "llm-integration")]
async fn dream_e2e_real_llm_five_pass_chain() {
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use autoagents_llm::embedding::EmbeddingBuilder;
    use metrics_util::debugging::DebuggingRecorder;

    use helpers::ollama_adapter::OllamaEmbedderAdapter;

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Dream is a QUALITY pass — use gemma4:e4b (deferred-quality default). Overridable.
    let chat_model =
        std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(&chat_model)
        .timeout_seconds(180)
        .keep_alive("1h")
        .build()
        .expect("Ollama LLM builder must succeed");

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("Ollama embedder builder must succeed");
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_emb));

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("dream-e2e");
    let mem = Memory::open(dir.path().join("dream_e2e.db"))
        .with_llm(llm as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        .embedding_dim(768)
        .default_namespace(ns.clone())
        .await
        .expect("Memory::open must succeed");

    // Ensure the namespace/group exists + default entity types are seeded, then
    // plant catch-all entities the reclassify pass can type.
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();
    for name in ["Albert Einstein", "Marie Curie"] {
        plant_catch_all(&graph, name, &gid).await;
    }

    let before = type_ids(&graph, &gid).await;
    eprintln!("[dream-e2e] before dream (id, type_id): {before:?}");

    // Run the FULL 5-pass chain via the path Phases 1-3 modified.
    let summary = mem
        .dream()
        .await
        .expect("mem.dream() must succeed end-to-end with a real LLM");

    let after = type_ids(&graph, &gid).await;
    eprintln!(
        "[dream-e2e] after dream (id, type_id): {after:?} | duration={}ms \
         entities_reclassified={} types_discovered={} warnings={:?}",
        summary.duration_ms,
        summary.entities_reclassified,
        summary.types_discovered.len(),
        summary.warnings,
    );

    // (1) Whole §D3 chain executed — every pass counter fired via mem.dream().
    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    for expected in [
        "kremory.dream.passes_continued_past_reclassify_total", // Phase 1 restructure
        "kremory.dream.aliases_resolved_total",                 // Phase 2 aliases
        "kremory.dream.canonicalization_merges_total",          // Phase 2 canonicalize
        "kremory.dream.consistency_check.scanned_total",        // Phase 3 (core invocation)
        "kremory.dream.consistency_check_corrected_total",      // Phase 3 (facade)
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "expected dream counter `{expected}` absent after real-LLM mem.dream(); \
             the pass did not run. Counters seen: {names:?}",
        );
    }

    // (2) TEETH — reclassify did REAL work: at least one clear-name catch-all
    // entity was typed (entity_type_id moved off the 0 catch-all). This is the
    // "not running blind" check the mock tests cannot provide.
    assert!(
        summary.entities_reclassified >= 1,
        "reclassify must type >=1 clear-name catch-all entity via mem.dream() with \
         a real LLM; before={before:?} after={after:?}",
    );

    // (3) No pass may hard-fail — each non-fatal failure pushes a "…failed…" warning.
    let failures: Vec<&String> = summary
        .warnings
        .iter()
        .filter(|w| w.contains("failed"))
        .collect();
    assert!(
        failures.is_empty(),
        "no dream pass may fail in the real-LLM E2E; pass failures: {failures:?}",
    );
}
