//! LLM Integration Tests — kremory v0.1.0 BYOM facade (Tier 2)
//!
//! All tests require a live Ollama instance at OLLAMA_BASE_URL (default:
//! http://localhost:11434) with `nomic-embed-text` (embedding) and a JSON-capable
//! chat model pulled.
//!
//! Chat model selection: set `OLLAMA_CHAT_MODEL` env var (default: `llama3.2:3b`).
//! `gemma4-e2b` is NOT supported — it does not follow structured JSON prompts.
//! Recommended: `llama3.2:3b`, `llama3.1:8b`, or `qwen2.5:14b`.
//!
//! Gate: all tests carry `#[ignore]`.  Invoke manually before publish:
//!   cargo test -p kremory --features llm-integration --test llm_integration -- --ignored
//!
//! Decision D3 (strategy): #[ignore] is sufficient gating; the feature flag
//! is retained for build-matrix control without adding test-discovery noise.

#![cfg(feature = "llm-integration")]
// Test files use expect/unwrap/panic as intentional assertion mechanisms.
// Consistent with the project-wide test convention (see b1_observability.rs,
// background_integration.rs et al).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod helpers;

use std::sync::Arc;
use std::time::Duration;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::embedding::EmbeddingBuilder;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

use helpers::metrics_capture::MetricsCapture;
use helpers::ollama_adapter::OllamaEmbedderAdapter;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns base_url from OLLAMA_BASE_URL env var, or "http://localhost:11434".
fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Returns chat model from OLLAMA_CHAT_MODEL env var, or "llama3.2:3b".
///
/// gemma4-e2b is NOT supported — it does not follow structured JSON prompts.
fn ollama_chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "llama3.2:3b".to_string())
}

/// Build a `Memory` instance with real Ollama providers wired to the given tempdir.
async fn build_mem(
    dir: &tempfile::TempDir,
    ns: &str,
) -> kremory::memory::Result<Memory> {
    let base_url = ollama_base_url();

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(&ollama_chat_model())
        .timeout_seconds(60)
        .build()
        .map_err(|e| kremory::memory::MemoryError::Other(e.to_string()))?;

    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .map_err(|e| kremory::memory::MemoryError::Other(e.to_string()))?;

    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_emb));

    Memory::open(dir.path().join("kremory.db"))
        .with_llm(llm as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        // nomic-embed-text outputs 768-dim vectors; must match the SQLite vector index.
        .embedding_dim(768)
        .default_namespace(Namespace::new(ns))
        .await
}

// ── LLM.2.1 — Entity extraction: Alice / Bob / Stanford ──────────────────────

/// LLM.2.1 — Submit one episode containing named entities. Verify that recall
/// surfaces results mentioning those entities.
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `llama3.2:3b`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn entity_extraction_alice_bob_stanford() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = build_mem(&dir, "test-llm-2-1")
        .await
        .expect("Memory::open must succeed");

    // Submit with no_wait so Phase 2 enrichment runs in background → run_id is Some.
    let commit = mem
        .remember("Alice met Bob at Stanford in 2020.")
        .from_chat("test-session-1")
        .no_wait()
        .await
        .expect("remember must succeed");

    assert!(
        !commit.episode_entity_id.is_empty(),
        "episode_entity_id must be non-empty"
    );

    // Block until Phase 2 enrichment completes (entities extracted + graph updated).
    // 120s timeout — local Ollama with a small model (e.g. llama3.2:3b) can take
    // 30–90s for a full extraction+resolution pipeline under CPU contention.
    let enrichment_status = mem.await_enrichment(&commit, Duration::from_secs(120))
        .await
        .expect("await_enrichment must succeed");
    assert!(
        matches!(enrichment_status, kremory::core::error::IngestStatus::Complete),
        "enrichment must complete successfully; got: {:?}",
        enrichment_status
    );

    // Recall entities matching "Alice".
    let results = mem
        .recall("Alice")
        .raw()
        .await
        .expect("recall must succeed");

    // At least 2 results mentioning Alice or Bob.
    let alice_or_bob = results.iter().filter(|r| {
        let name = r.entity_name.to_lowercase();
        let summary = r.summary.to_lowercase();
        name.contains("alice") || name.contains("bob")
            || summary.contains("alice") || summary.contains("bob")
    });
    assert!(
        alice_or_bob.count() >= 2,
        "expected >=2 results matching alice|bob; got: {:#?}",
        results
    );

    // At least 1 result mentioning Stanford.
    let stanford = results.iter().any(|r| {
        r.entity_name.to_lowercase().contains("stanford")
            || r.summary.to_lowercase().contains("stanford")
    });
    assert!(stanford, "expected >=1 result mentioning stanford; got: {:#?}", results);
}

// ── LLM.2.2 — Dedup invariant: same text submitted twice ─────────────────────

/// LLM.2.2 — Submitting identical content twice must be handled by the dedup
/// path. The second submit either returns `Err(CoreError::Duplicate{..})` or an
/// `Ok` commit whose enrichment status is `Complete`. Recall must return ≤3
/// alice results (no phantom duplication).
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `llama3.2:3b`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn dedup_invariant_same_text_twice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = build_mem(&dir, "test-llm-2-2")
        .await
        .expect("Memory::open must succeed");

    let content = "Alice is a researcher at MIT.";

    let c1 = mem
        .remember(content)
        .from_chat("dedup-session")
        .await
        .expect("first remember must succeed");

    assert!(!c1.episode_entity_id.is_empty(), "c1 episode_entity_id must be non-empty");

    // Second submit with identical content — must deduplicate.
    let second_result = mem
        .remember(content)
        .from_chat("dedup-session")
        .await;

    match second_result {
        Err(e) => {
            // Dedup path: CoreError::Duplicate propagated as MemoryError::Core.
            let msg = format!("{e:?}");
            assert!(
                msg.contains("Duplicate") || msg.contains("duplicate"),
                "expected Duplicate error on second submit, got: {e:?}"
            );
        }
        Ok(c2) => {
            // Some graph implementations may return Ok and mark the episode as
            // already complete. In that case assert enrichment finishes cleanly.
            let status = if c2.run_id.is_some() {
                mem.await_enrichment(&c2, Duration::from_secs(120))
                    .await
                    .expect("await_enrichment on second commit must succeed")
            } else {
                kremory::IngestStatus::Complete
            };
            assert_eq!(
                status,
                kremory::IngestStatus::Complete,
                "second commit status must be Complete, got: {status:?}"
            );
        }
    }

    // Recall should not return runaway duplicates.
    let results = mem
        .recall("Alice")
        .raw()
        .await
        .expect("recall must succeed");

    assert!(
        results.len() <= 3,
        "expected <=3 alice results after dedup, got {} results: {:#?}",
        results.len(),
        results
    );
}

// ── LLM.2.3 — Concurrent ingest: 3 independent Memory instances ──────────────

/// LLM.2.3 — Spawn 3 independent `Memory` instances (each with its own tempdir
/// and SQLite file), submit one episode each concurrently via `tokio::join!`.
/// Assert all 3 episode_entity_id strings are non-empty and distinct.
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `llama3.2:3b`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn concurrent_ingest_three_episodes_no_panic() {
    let dir1 = tempfile::tempdir().expect("tempdir-1");
    let dir2 = tempfile::tempdir().expect("tempdir-2");
    let dir3 = tempfile::tempdir().expect("tempdir-3");

    let base_url = ollama_base_url();

    let build = |path: std::path::PathBuf, ns: &'static str| {
        let url = base_url.clone();
        let chat_model = ollama_chat_model();
        async move {
            let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
                .base_url(&url)
                .model(&chat_model)
                .timeout_seconds(60)
                .build()
                .map_err(|e| kremory::memory::MemoryError::Other(e.to_string()))?;
            let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
                .base_url(&url)
                .model("nomic-embed-text")
                .build()
                .map_err(|e| kremory::memory::MemoryError::Other(e.to_string()))?;
            let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(OllamaEmbedderAdapter(raw_emb));
            Memory::open(path)
                .with_llm(llm as Arc<dyn ChatProvider>)
                .with_embedder(emb)
                // nomic-embed-text outputs 768-dim vectors; must match the SQLite vector index.
                .embedding_dim(768)
                .default_namespace(Namespace::new(ns))
                .await
        }
    };

    let (m1, m2, m3) = tokio::join!(
        build(dir1.path().join("kremory.db"), "concurrent-1"),
        build(dir2.path().join("kremory.db"), "concurrent-2"),
        build(dir3.path().join("kremory.db"), "concurrent-3"),
    );
    let m1 = m1.expect("Memory-1 open must succeed");
    let m2 = m2.expect("Memory-2 open must succeed");
    let m3 = m3.expect("Memory-3 open must succeed");

    // Use no_wait() so the background path returns a UUID-based episode_entity_id.
    // Blocking ingest returns the DB row ID (starts at 1 per fresh DB), which
    // would be "1" for all three separate Memory instances and fail the distinctness check.
    let (c1, c2, c3) = tokio::join!(
        m1.remember("Carol is a distributed systems engineer.").from_chat("s1").no_wait(),
        m2.remember("Dave leads the platform reliability team.").from_chat("s2").no_wait(),
        m3.remember("Eve specialises in Byzantine fault-tolerant consensus.").from_chat("s3").no_wait(),
    );

    let c1 = c1.expect("Carol ingest must succeed");
    let c2 = c2.expect("Dave ingest must succeed");
    let c3 = c3.expect("Eve ingest must succeed");

    assert!(!c1.episode_entity_id.is_empty(), "c1 episode_entity_id must be non-empty");
    assert!(!c2.episode_entity_id.is_empty(), "c2 episode_entity_id must be non-empty");
    assert!(!c3.episode_entity_id.is_empty(), "c3 episode_entity_id must be non-empty");

    let ids: std::collections::HashSet<&str> = [
        c1.episode_entity_id.as_str(),
        c2.episode_entity_id.as_str(),
        c3.episode_entity_id.as_str(),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        ids.len(),
        3,
        "expected 3 distinct episode_entity_id values; got: {:?}",
        ids
    );
}

// ── LLM.2.4 — Recall ranks Alice episodes above unrelated content ─────────────

/// LLM.2.4 — Submit 10 Alice-related episodes and 1 unrelated Tokyo episode.
/// Recall "Alice" must return ≥8 results whose entity_name or summary contains
/// "alice". Tokyo-specific content should rank lower.
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `llama3.2:3b`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn recall_ranks_alice_episodes_above_unrelated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = build_mem(&dir, "test-llm-2-4")
        .await
        .expect("Memory::open must succeed");

    let alice_episodes = [
        "Alice founded a machine learning consultancy in 2019.",
        "Alice published a paper on transformer architectures.",
        "Alice mentors junior engineers at her firm.",
        "Alice gave a keynote at NeurIPS 2023.",
        "Alice collaborates with Bob on reinforcement learning research.",
        "Alice's consultancy focuses on healthcare AI applications.",
        "Alice received a best paper award at ICML.",
        "Alice is based in San Francisco and travels to NYC frequently.",
        "Alice co-authored a book on practical deep learning.",
        "Alice is fluent in Python, Rust, and Julia.",
    ];

    for (i, content) in alice_episodes.iter().enumerate() {
        mem.remember(*content)
            .from_chat(format!("alice-session-{i}"))
            .await
            .unwrap_or_else(|e| panic!("alice episode {i} must succeed: {e}"));
    }

    // One unrelated Tokyo episode.
    mem.remember("The Tokyo skyline is famous for Mount Fuji views.")
        .from_chat("tokyo-session")
        .await
        .expect("tokyo episode must succeed");

    let results = mem
        .recall("Alice")
        .raw()
        .await
        .expect("recall must succeed");

    let alice_hits = results.iter().filter(|r| {
        r.entity_name.to_lowercase().contains("alice")
            || r.summary.to_lowercase().contains("alice")
    });
    assert!(
        alice_hits.count() >= 8,
        "expected >=8 results containing 'alice'; got: {:#?}",
        results
    );

    // Conditional ranking check: if Tokyo appears, it should not be in the top 3.
    let tokyo_rank = results.iter().enumerate().find(|(_, r)| {
        r.entity_name.to_lowercase().contains("tokyo")
            || r.summary.to_lowercase().contains("tokyo")
    });
    if let Some((rank, _)) = tokyo_rank {
        assert!(
            rank >= 3,
            "tokyo result appeared in top-3 (rank {rank}) when recalling 'Alice'; results: {:#?}",
            results
        );
    }
}

// ── LLM.2.5 — Dream phase returns NotImplemented (F-01 LOCKED) ───────────────

/// LLM.2.5 — Submit 5 episodes about Alice/Bob/project Gamma, then call
/// `mem.dream()`. Assert that it returns `Err(MemoryError::NotImplemented)`
/// per F-01 contract (ADR-007 §3). This is an integration-level smoke test
/// verifying the typed-error contract holds end-to-end after real ingest.
///
/// Dream consolidation ships in v0.1.1. This test guards against accidental
/// silent-Ok or panic regressions at the facade level.
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `llama3.2:3b`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn dream_phase_not_implemented() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = build_mem(&dir, "test-llm-2-5")
        .await
        .expect("Memory::open must succeed");

    let episodes = [
        "Alice and Bob are co-leads on project Gamma.",
        "Project Gamma aims to build a real-time knowledge graph engine.",
        "Bob handles infrastructure; Alice owns the research direction for Gamma.",
        "Alice presented Gamma's progress to the steering committee last week.",
        "Bob integrated the new vector index into the Gamma pipeline.",
    ];

    for (i, content) in episodes.iter().enumerate() {
        mem.remember(*content)
            .from_chat(format!("dream-session-{i}"))
            .await
            .unwrap_or_else(|e| panic!("episode {i} must succeed: {e}"));
    }

    // F-01 contract: dream() must return Err(NotImplemented) — not Ok, not panic.
    let result = mem.dream().await;
    match result {
        Err(kremory::memory::MemoryError::NotImplemented { feature, available_in, adr_ref }) => {
            assert!(
                !feature.is_empty(),
                "NotImplemented.feature must be non-empty"
            );
            assert!(
                available_in.contains("v0.1.1"),
                "NotImplemented.available_in must reference v0.1.1; got: {available_in}"
            );
            assert!(
                adr_ref.contains("ADR-007"),
                "NotImplemented.adr_ref must reference ADR-007; got: {adr_ref}"
            );
        }
        Err(other) => panic!(
            "dream() must return Err(NotImplemented) per F-01; got: {other:?}"
        ),
        Ok(summary) => panic!(
            "dream() must return Err(NotImplemented) per F-01; got Ok: {summary:?}"
        ),
    }
}

// ── G_NS — Namespace isolation: entities in ns-alpha must not bleed into ns-beta ─

/// G_NS — Namespace isolation gate.
///
/// Ingests content into two distinct namespaces using a deterministic mock LLM
/// that extracts "ALPHA" from alpha-namespace content and "BETA" from beta-namespace
/// content. Asserts that:
///   1. Recall in ns-beta for "ALPHA" returns no ns-alpha content.
///   2. Entity sets in ns-alpha and ns-beta are non-empty and disjoint.
///   3. Each entity carries the correct `group_id` matching its namespace.
///
/// Does NOT require a live Ollama instance — uses `MockChatProvider` +
/// `DeterministicEmbeddingProvider`. Still marked `#[ignore]` per integration
/// test convention; run with `--ignored` before publish.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "namespace-isolation integration test — run with --ignored"]
#[cfg(feature = "llm-integration")]
async fn namespace_isolation() {
    use kremory::core::config::PipelineConfig;
    use kremory::core::provider::{DeterministicEmbeddingProvider, MockChatProvider};
    use kremory::core::schema::TemporalGraph;
    use kremory::memory::engine_handle::EngineGraphHandle;
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::GraphHandle;
    use kremory::{ChatProvider, DynEmbeddingProvider, Namespace};
    use metrics_util::debugging::DebuggingRecorder;
    use std::collections::HashMap;

    // Install a no-op metrics recorder so histogram!() calls don't panic.
    let recorder = DebuggingRecorder::new();
    let _metrics_guard = metrics::set_default_local_recorder(&recorder);

    // Build a mock LLM whose extraction response depends on the content substring:
    //   prompt containing "alpha-content" → entity ALPHA (Person)
    //   prompt containing "beta-content"  → entity BETA  (Person)
    //
    // NuExtractExtractor matches on "# Template:" and expects {"entities": [...], "relationships": [...]}.
    // MockChatProvider::new matches on the last user message substring — the
    // full input text is embedded in the prompt, so these keys reliably select
    // the right response per ingest call.
    let mut responses: HashMap<String, String> = HashMap::new();
    responses.insert(
        "alpha-content".to_string(),
        serde_json::json!({
            "entities": [{"name": "ALPHA", "label": "Person"}],
            "relationships": []
        })
        .to_string(),
    );
    responses.insert(
        "beta-content".to_string(),
        serde_json::json!({
            "entities": [{"name": "BETA", "label": "Person"}],
            "relationships": []
        })
        .to_string(),
    );

    // with_config takes Arc<dyn ChatProvider + Send + Sync> for the engine's internal LLM —
    // this drives extraction. The _provider arg on graph_ingest_episode is ignored by
    // EngineGraphHandle, so we pass a separate null mock there (avoids coercion boilerplate).
    let mock_llm: Arc<dyn ChatProvider + Send + Sync> = Arc::new(MockChatProvider::new(responses));
    // Null provider for the ignored _provider parameter of graph_ingest_episode.
    let noop_provider: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(DeterministicEmbeddingProvider::new(384));

    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("ns-iso.db");

    // Keep a direct Arc<TemporalGraph> reference so we can call list_entities_in_group
    // after ingestion without going through the facade.
    let graph = Arc::new(
        TemporalGraph::open(db_path.to_str().expect("valid UTF-8 path"))
            .await
            .expect("TemporalGraph opens"),
    );

    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig default");

    let handle = EngineGraphHandle::with_config(
        Arc::clone(&graph),
        Arc::clone(&mock_llm),
        embedder,
        config,
    );

    let ns_alpha = Namespace::new("ns-alpha");
    let ns_beta = Namespace::new("ns-beta");

    let alpha_source = SourceRef {
        kind: SourceKind::Chat,
        id: "alpha-src-1".to_string(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    };
    let beta_source = SourceRef {
        kind: SourceKind::Chat,
        id: "beta-src-1".to_string(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    };

    let opts = SubmitOpts {
        enrich_per_episode: true,
        run_in_background: false,
    };

    // Write to ns-alpha — mock engine LLM returns ALPHA entity for "alpha-content" prompt
    handle
        .graph_ingest_episode(
            &ns_alpha,
            &alpha_source,
            "alpha-content about ALPHA",
            &[],
            Arc::clone(&noop_provider), // _provider arg is ignored by EngineGraphHandle
            None,
            opts.clone(),
            None,
        )
        .await
        .expect("ns-alpha ingest OK");

    // Write to ns-beta — mock engine LLM returns BETA entity for "beta-content" prompt
    handle
        .graph_ingest_episode(
            &ns_beta,
            &beta_source,
            "beta-content about BETA",
            &[],
            Arc::clone(&noop_provider), // _provider arg is ignored by EngineGraphHandle
            None,
            opts,
            None,
        )
        .await
        .expect("ns-beta ingest OK");

    // Query ns-beta for "ALPHA" — must NOT surface ns-alpha entities
    let beta_results = handle
        .graph_search(
            &ns_beta,
            "ALPHA",
            &SearchOpts { limit: Some(20), as_of: None, source_kind: None },
        )
        .await
        .expect("graph_search ns-beta OK");

    assert!(
        beta_results
            .iter()
            .all(|r| !r.summary.contains("alpha-content")
                && !r.entity_name.eq_ignore_ascii_case("ALPHA")),
        "namespace bleed: ns-beta query returned ns-alpha content; results: {beta_results:#?}"
    );

    // Verify entity sets are non-empty and disjoint via TemporalGraph::list_entities_in_group
    let alpha_entities = graph
        .list_entities_in_group("ns-alpha")
        .await
        .expect("list_entities_in_group ns-alpha");

    let beta_entities = graph
        .list_entities_in_group("ns-beta")
        .await
        .expect("list_entities_in_group ns-beta");

    assert!(
        !alpha_entities.is_empty(),
        "ns-alpha must have entities after ingest; got empty list"
    );
    assert!(
        !beta_entities.is_empty(),
        "ns-beta must have entities after ingest; got empty list"
    );

    // Entity ID sets must be disjoint across namespaces
    let alpha_ids: std::collections::HashSet<&str> =
        alpha_entities.iter().map(|e| e.id.as_str()).collect();
    let beta_ids: std::collections::HashSet<&str> =
        beta_entities.iter().map(|e| e.id.as_str()).collect();

    assert!(
        alpha_ids.is_disjoint(&beta_ids),
        "entity IDs overlap across namespaces; alpha={alpha_ids:?}, beta={beta_ids:?}"
    );

    // Each entity must carry its namespace as group_id
    assert!(
        alpha_entities
            .iter()
            .all(|e| e.group_id.as_deref() == Some("ns-alpha")),
        "not all ns-alpha entities have group_id='ns-alpha'; entities: {alpha_entities:#?}"
    );
    assert!(
        beta_entities
            .iter()
            .all(|e| e.group_id.as_deref() == Some("ns-beta")),
        "not all ns-beta entities have group_id='ns-beta'; entities: {beta_entities:#?}"
    );
}

// ── LLM.3 — Token observability: kremory_core_tokens_total counter ────────────

/// LLM.3 — Wrap a real Ollama embedder in `TokenTrackingEmbedder`. Call `embed`
/// once inside `metrics::with_local_recorder`. Assert that
/// `kremory_core_tokens_total` counter is present in the snapshot.
///
/// Uses a plain `#[test]` (no outer tokio runtime) so that the inner
/// `new_current_thread` runtime can call `block_on` without hitting the
/// "cannot start a runtime from within a runtime" panic.  The thread-local
/// recorder installed by `metrics::with_local_recorder` stays active on the
/// single thread throughout the `block_on` call — identical pattern to
/// `b1_observability.rs::token_tracking_embedder_emits_tokens_total_counter`.
///
/// `#[ignore]`: requires live Ollama with nomic-embed-text.
#[test]
#[ignore]
#[cfg(feature = "llm-integration")]
fn token_tracking_embedder_counter_increments() {
    use kremory::TokenTrackingEmbedder;
    use kremory::core::provider::EmbeddingProvider as _;

    let capture = MetricsCapture::new();

    let base_url = ollama_base_url();
    let raw_emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model("nomic-embed-text")
        .build()
        .expect("EmbeddingBuilder must succeed");

    let adapter = OllamaEmbedderAdapter(raw_emb);
    let tracked = TokenTrackingEmbedder::new(adapter, "ollama", "nomic-embed-text");

    // Single-threaded runtime keeps the thread-local recorder active.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&capture.recorder, || {
        rt.block_on(async {
            tracked
                .embed("Alice met Bob at Stanford.")
                .await
                .expect("embed must succeed");
        });
    });

    assert!(
        capture.has_counter("kremory_core_tokens_total"),
        "kremory_core_tokens_total counter must be present after embed; counters: {:?}",
        capture.counter_names()
    );
}
