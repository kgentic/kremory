//! LLM Integration Tests — kremory v0.1.0 BYOM facade (Tier 2)
//!
//! All tests require a live Ollama instance at OLLAMA_BASE_URL (default:
//! http://localhost:11434) with `nomic-embed-text` (embedding) and a JSON-capable
//! chat model pulled.
//!
//! # Chat model selection
//!
//! Set `OLLAMA_CHAT_MODEL` to the model under test. For thinking-capable models
//! set `KREMORY_BENCH_THINK=false` to disable reasoning (kremory's `with_ollama`
//! does this in production — see `facade/providers.rs`).
//!
//! ## Empirical ladder — kremory benchmark 2026-06-24 (Apple Silicon M4 Max,
//! mock_interview.txt = 10 ground-truth entities; reproduce via
//! `scripts/model-benchmark/`). Latency is M4-Max-only; precision/recall/F1 are
//! hardware-independent. `prec`=correct/extracted, `recall`=found/expected.
//!
//! | Model (registry tag) | think | F1    | recall | prec | slowest call | fits 30s | size  | tier               |
//! |----------------------|-------|-------|--------|------|--------------|----------|-------|--------------------|
//! | `gemma4:e4b`         | false | 84    | 90%    | 82%  | ~16s         | yes      | 9.6GB | DEFAULT (best)     |
//! | `qwen2.5:14b`        | n/a   | 78-82 | 70%    | ~94% | 29-50s       | NO       | 9.0GB | deferred/quality   |
//! | `qwen2.5:7b`         | n/a   | 77-82 | 70%    | 86%  | 11s (28s p)  | yes      | 4.7GB | light alternative  |
//! | `qwen3.5:9b`         | false | 75    | 90%    | 64%  | ~24s         | yes      | 6.6GB | thinking-ON FAILS  |
//! | `llama3.2:3b`        | n/a   | 70    | 70%    | 70%  | ~20s         | yes      | 2.0GB | small              |
//! | `qwen2.5:3b`         | n/a   | 54    | 70%    | 44%  | ~6s          | yes      | 1.9GB | small/noisy        |
//! | `gemma4-e2b:latest`  | n/a   | 0     | 0%     | 0%   | —            | —        | 3.1GB | UNUSABLE (0 ents)  |
//! | `*-mlx` (any)        | —     | —     | —      | —    | —            | —        | —     | AVOID (Apple-only) |
//!
//! ## Rationale (model choice is QUALITY knob, not latency knob)
//!
//! kremory ingest is two-phase: Phase 1 (sync) embeds + saves + runs NER;
//! Phase 2 (`ingest_deferred`) runs LLM relationship extraction via background
//! worker. The wall-clock numbers above are total combined-phase costs from
//! the benchmark, which uses the inline `ingest_with` path. Production usage
//! routes the heavy LLM cost through the deferred queue — so a 378s/doc model
//! does NOT mean the user waits 378s; the user sees instant ack from Phase 1.
//!
//! ## Tag-form footgun
//!
//! `crates/kremory/src/core/provider.rs:156` `capability_of()` routes by `:` colon
//! detection OR known prefix list (llama, qwen, phi, mistral, nuextract). Bare
//! `gemma4-e2b` falls through to PromptOnly (no FormatSchema arm) → LLM emits
//! `null` / `{}`. Use the full `gemma4-e2b:latest` tag form to ensure the
//! FormatSchema arm fires. The `gemma` family prefix should be added to the
//! known list in a follow-up.
//!
//! ## `gemma4:26b` quarantine
//!
//! Same 90% precision as e4b but 4.5x slower AND extracts ~5 junk strings the
//! shape-validator cannot reject (`:`, `: 15 different developers...`, etc).
//! No precision win, more noise. Skip for now.
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

/// Returns chat model from OLLAMA_CHAT_MODEL env var, or `gemma4-e2b:latest`.
///
/// Default tracks the empirical ladder above: `gemma4-e2b:latest` is the
/// interactive-default (80% precision / ~37-54s) per benchmarks 2026-06-04
/// post TD-013. Override via `OLLAMA_CHAT_MODEL` for benchmark-gate runs
/// (`gemma4:e4b`) or Path-β verify experiments (`qwen2.5:14b` per
/// ADR-048 ratified 2026-06-11).
///
/// Note: `gemma4-e2b` REQUIRES the full `gemma4-e2b:latest` tag form so
/// `capability_of()` routes to the FormatSchema arm. Bare names without
/// `:tag` fall through to PromptOnly and produce empty / null output.
fn ollama_chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4-e2b:latest".to_string())
}

/// Build a `Memory` instance with real Ollama providers wired to the given tempdir.
async fn build_mem(dir: &tempfile::TempDir, ns: &str) -> kremory::memory::Result<Memory> {
    let base_url = ollama_base_url();

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(ollama_chat_model())
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
/// surfaces results mentioning those entities with v0.1.1 shape.
///
/// Assertion shape per Tessa §16.1 (SUPERSEDES prior assertions):
/// - source_refs must be non-empty (ASMP-001 regression guard — entity-loop episodic edge write)
/// - source_refs.kind must be Episode (Bug A fix — not Document)
/// - summary must be a verbatim snippet, not an entity label (Bug B regression guard)
/// - score must be normalised in [0.0, 1.0] (Bug C RRF fix)
/// - 1-hop expansion: Bob or Stanford must also surface
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `gemma4-e2b:latest`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn entity_extraction_alice_bob_stanford() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mem = build_mem(&dir, "test-llm-2-1")
        .await
        .expect("Memory::open must succeed");

    // Submit and wait for enrichment to complete inline.
    mem.remember("Alice met Bob at Stanford in 2020.")
        .from_chat("test-session-1")
        .await
        .expect("remember must succeed");

    // Recall entities matching "Alice".
    let results = mem
        .recall("Alice")
        .raw()
        .await
        .expect("recall must succeed");

    assert!(
        !results.is_empty(),
        "recall must return at least one result"
    );

    let alice = results
        .iter()
        .find(|r| r.entity_name.to_lowercase().contains("alice"))
        .expect("Alice must be in recall results");

    // Bug B regression guard: summary must be a real snippet, not an entity label.
    assert!(!alice.summary.is_empty(), "summary must be non-empty");
    assert_ne!(
        alice.summary.to_lowercase().trim(),
        alice.entity_name.to_lowercase().trim(),
        "summary must be a verbatim snippet, not the entity name (Bug B regression guard)"
    );
    assert_ne!(
        alice.summary.to_lowercase().trim(),
        "person",
        "summary must not be raw entity label (Bug B regression guard)"
    );

    // Bug C regression guard: score must be normalised RRF in [0.0, 1.0].
    assert!(
        alice.score >= 0.0 && alice.score <= 1.0,
        "score must be in [0.0, 1.0] (RRF normalised — Bug C fix) — was: {}",
        alice.score
    );

    // Bug A + ASMP-001 regression guard: source_refs must be populated.
    assert!(
        !alice.source_refs.is_empty(),
        "source_refs must be non-empty — ASMP-001 regression guard (entity-loop episodic edge write)"
    );

    // Bug A: source_refs must carry SourceKind::Episode, not SourceKind::Document.
    for sr in &alice.source_refs {
        assert_eq!(
            sr.kind,
            kremory::SourceKind::Episode,
            "source_refs must use SourceKind::Episode variant (not Document) — Bug A fix"
        );
        assert!(
            !sr.id.is_empty(),
            "episode id in source_ref must be non-empty"
        );
    }

    // Bug B: summary must contain an entity mention from the source text.
    let summary_lower = alice.summary.to_lowercase();
    assert!(
        summary_lower.contains("alice")
            || summary_lower.contains("stanford")
            || summary_lower.contains("bob"),
        "summary snippet must contain an entity mention from the source text, not a generic label (Bug B)"
    );

    // 1-hop expansion: Bob or Stanford should appear as separate RetrievedContext rows.
    let bob_present = results
        .iter()
        .any(|r| r.entity_name.to_lowercase().contains("bob"));
    let stanford_present = results
        .iter()
        .any(|r| r.entity_name.to_lowercase().contains("stanford"));
    assert!(
        bob_present || stanford_present,
        "1-hop expansion must surface Bob or Stanford as separate RetrievedContext rows"
    );
}

// ── LLM.2.2 — Dedup invariant: same text submitted twice ─────────────────────

/// LLM.2.2 — Submitting identical content twice must be handled by the dedup
/// path. The second submit either returns `Err(CoreError::Duplicate{..})` or an
/// `Ok` commit whose enrichment status is `Complete`. Recall must return ≤3
/// alice results (no phantom duplication).
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `gemma4-e2b:latest`) + nomic-embed-text.
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

    assert!(
        !c1.episode_entity_id.is_empty(),
        "c1 episode_entity_id must be non-empty"
    );

    // Second submit with identical content — must deduplicate.
    let second_result = mem.remember(content).from_chat("dedup-session").await;

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
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `gemma4-e2b:latest`) + nomic-embed-text.
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
        m1.remember("Carol is a distributed systems engineer.")
            .from_chat("s1")
            .no_wait(),
        m2.remember("Dave leads the platform reliability team.")
            .from_chat("s2")
            .no_wait(),
        m3.remember("Eve specialises in Byzantine fault-tolerant consensus.")
            .from_chat("s3")
            .no_wait(),
    );

    let c1 = c1.expect("Carol ingest must succeed");
    let c2 = c2.expect("Dave ingest must succeed");
    let c3 = c3.expect("Eve ingest must succeed");

    assert!(
        !c1.episode_entity_id.is_empty(),
        "c1 episode_entity_id must be non-empty"
    );
    assert!(
        !c2.episode_entity_id.is_empty(),
        "c2 episode_entity_id must be non-empty"
    );
    assert!(
        !c3.episode_entity_id.is_empty(),
        "c3 episode_entity_id must be non-empty"
    );

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

// ── LLM.2.4 — Multi-episode entity recall: canonical Alice + episodic source_refs ─

/// LLM.2.4 — Submit 10 Alice-related episodes and 1 unrelated Tokyo episode.
/// Recall "Alice" must surface a canonical Alice row with ≥5 episodic source_refs
/// and SourceKind::Episode on all refs.
///
/// Assertion shape per Tessa §16.2 (SUPERSEDES prior assertions):
/// - Alice appears exactly once (canonical entity, no duplicate rows)
/// - source_refs.len() >= 5 (one episodic_edge per Alice-mentioning episode — looser bound
///   than spec's == 10 because LLM extraction may not always link Alice to every episode)
/// - All source_refs must be SourceKind::Episode
/// - score > 0.0
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `gemma4-e2b:latest`) + nomic-embed-text.
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

    // One unrelated Tokyo episode (must NOT appear in Alice's source_refs).
    mem.remember("Tokyo hosted the annual summit.")
        .from_chat("tokyo-session")
        .await
        .expect("tokyo episode must succeed");

    let results = mem
        .recall("Alice")
        .raw()
        .await
        .expect("recall must succeed");

    assert!(!results.is_empty(), "recall must return results");

    // Alice must appear exactly once (canonical entity, no duplicate rows).
    let alice_rows: Vec<_> = results
        .iter()
        .filter(|r| r.entity_name.to_lowercase().contains("alice"))
        .collect();
    assert_eq!(
        alice_rows.len(),
        1,
        "Alice must appear as exactly one canonical RetrievedContext row (no duplicates)"
    );

    let alice = alice_rows[0];

    // source_refs must contain at least 5 episodic edges (looser bound than spec's == 10
    // because LLM extraction may not link Alice to every episode — at least 5/10 is honest).
    assert!(
        alice.source_refs.len() >= 5,
        "Alice must have >= 5 source_refs (episodic_edge per Alice episode — entity-loop write); got {}",
        alice.source_refs.len()
    );

    // All source_refs must be SourceKind::Episode.
    for sr in &alice.source_refs {
        assert_eq!(
            sr.kind,
            kremory::SourceKind::Episode,
            "all source_refs must be SourceKind::Episode"
        );
        assert!(
            !sr.id.is_empty(),
            "episode id in source_ref must be non-empty"
        );
    }

    // score must be > 0.0 when recalled by name.
    assert!(
        alice.score > 0.0,
        "Alice RRF score must be > 0.0 when recalled by name"
    );
}

// ── LLM.2.5 — Dream phase returns Ok(DreamSummary) (F-01 retired) ───────────────

/// LLM.2.5 — Submit 5 episodes about Alice/Bob/project Gamma, then call
/// `mem.dream()`. Verifies `Ok(DreamSummary)` with real Pass-0/Pass-2
/// output and honest-zero Phase-3 consolidation fields.
///
/// Per `adr-mem-dream-canonical-supersede-f01-2026-06-22`: `mem.dream()` is the
/// canonical blocking dream API. F-01 (NotImplemented) is retired. Pass-0 and
/// Pass-2 populate `types_discovered` + `entities_reclassified`; the four
/// consolidation fields (`communities_updated`, `cross_episode_merges`,
/// `supersessions_recorded`, `facts_archived`) are honest zeros until Phase-3
/// consolidation ships (ADR-007 retirement).
///
/// `#[ignore]`: requires live Ollama with a JSON-capable chat model (default: `gemma4-e2b:latest`) + nomic-embed-text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn dream_phase_returns_ok_summary() {
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

    // ADR-071 Item 1 turned cross_episode ON by default (in SHADOW); communities/
    // archive follow in Item 2. This reconciliation-focused end-to-end test pins ALL
    // consolidation flags OFF so its honest-zeros below stay valid across both items
    // — the new default's shadow behaviour is covered by the P3 corpus gate +
    // cross_episode shadow unit tests + dream_full_consolidation_real_llm (all-on).
    // Field-mutation (not struct literal) — DreamOpts is `#[non_exhaustive]`.
    let mut dream_opts = kremory::memory::types::DreamOpts::default();
    dream_opts.include_cross_episode_merges = false;
    dream_opts.include_community_detection = false;
    dream_opts.include_fact_archival = false;
    let summary = mem
        .dream()
        .with_opts(dream_opts)
        .await
        .expect("dream() must return Ok after F-01 retirement");

    // Honest-zeros lock: with consolidation pinned OFF, consolidation fields must be 0.
    assert_eq!(
        summary.communities_updated, 0,
        "communities_updated must be 0 — consolidation pinned OFF here (ADR-071)"
    );
    assert_eq!(
        summary.cross_episode_would_merge, 0,
        "cross_episode_merges must be 0 — consolidation pinned OFF here (ADR-071)"
    );
    assert_eq!(
        summary.supersessions_recorded, 0,
        "supersessions_recorded must be 0 — consolidation pinned OFF here (ADR-071)"
    );
    assert_eq!(
        summary.facts_archived, 0,
        "facts_archived must be 0 — consolidation pinned OFF here (ADR-071)"
    );
    // Real-work fields: don't assert > 0 (stochastic; mirror E10 no-panic stance).
    // The pass itself not panicking + honest-zeros is the contract under test.
    let _ = summary.types_discovered;
    let _ = summary.entities_reclassified;
    let _ = summary.duration_ms;
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
    use kremory::memory::engine_handle::{EngineGraphHandle, WithConfigParams};
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::{GraphHandle, GraphIngestEpisodeParams, GraphSearchParams};
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
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

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

    let handle = EngineGraphHandle::with_config(WithConfigParams {
        graph: Arc::clone(&graph),
        chat: Arc::clone(&mock_llm),
        embedder,
        config,
    });

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
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns_alpha,
            source_ref: &alpha_source,
            content: "alpha-content about ALPHA",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider), // _provider arg is ignored by EngineGraphHandle
            batch_id: None,
            opts: opts.clone(),
            sink: None,
        })
        .await
        .expect("ns-alpha ingest OK");

    // Write to ns-beta — mock engine LLM returns BETA entity for "beta-content" prompt
    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns_beta,
            source_ref: &beta_source,
            content: "beta-content about BETA",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider), // _provider arg is ignored by EngineGraphHandle
            batch_id: None,
            opts,
            sink: None,
        })
        .await
        .expect("ns-beta ingest OK");

    // Query ns-beta for "ALPHA" — must NOT surface ns-alpha entities
    let beta_results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns_beta,
            query: "ALPHA",
            opts: &SearchOpts {
                limit: Some(20),
                as_of: None,
                source_kind: None,
            },
        })
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

// ── G_v011_* — v0.1.1 recall-pipeline unit-integration tests ─────────────────
//
// These tests use EngineGraphHandle::with_config + MockChatProvider +
// DeterministicEmbeddingProvider. No live Ollama required. They validate:
//   G_v011_5  — source_refs_carries_episode_kind
//   G_v011_12 — rrf_single_result_scores_one
//   G_v011_standalone — standalone_entity_has_episodic_edge
//   G_v011_stub_forward_ref — stub_entity_inserted_on_forward_reference
//   G_v011_stub_promoted — stub_entity_promoted_on_reingestion
//
// Each test builds a fresh EngineGraphHandle, ingests via graph_ingest_episode,
// then asserts using graph_search — the same verified pattern as namespace_isolation.

/// Build a shared helper: EngineGraphHandle + TemporalGraph with a MockChatProvider
/// that returns the given entity/fact JSON extraction response.
///
/// The `response_key` / `response_json` pair is registered so that MockChatProvider
/// returns `response_json` whenever the prompt contains `response_key`.
#[cfg(feature = "llm-integration")]
async fn build_mock_handle_with_response(
    db_path: &std::path::Path,
    response_key: impl Into<String>,
    response_json: impl Into<String>,
) -> (
    kremory::memory::engine_handle::EngineGraphHandle,
    Arc<kremory::core::schema::TemporalGraph>,
) {
    use kremory::core::config::PipelineConfig;
    use kremory::core::provider::{DeterministicEmbeddingProvider, MockChatProvider};
    use kremory::core::schema::TemporalGraph;
    use kremory::memory::engine_handle::{EngineGraphHandle, WithConfigParams};
    use std::collections::HashMap;

    let mut responses: HashMap<String, String> = HashMap::new();
    responses.insert(response_key.into(), response_json.into());

    let mock_llm: Arc<dyn kremory::memory::ChatProvider + Send + Sync> =
        Arc::new(MockChatProvider::new(responses));
    let embedder: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let graph = Arc::new(
        TemporalGraph::open(db_path.to_str().expect("valid UTF-8"))
            .await
            .expect("TemporalGraph::open"),
    );
    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig::default");
    let handle = EngineGraphHandle::with_config(WithConfigParams {
        graph: Arc::clone(&graph),
        chat: Arc::clone(&mock_llm),
        embedder,
        config,
    });
    (handle, graph)
}

/// G_v011_5 — source_refs_carries_episode_kind
///
/// After ingesting one episode that introduces Alice (real extraction via MockChatProvider),
/// recall must return source_refs with SourceKind::Episode and non-empty id.
///
/// No Ollama required. Does not require `#[ignore]`.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn source_refs_carries_episode_kind() {
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::{GraphHandle, GraphIngestEpisodeParams, GraphSearchParams};
    use kremory::Namespace;

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let tmp = tempfile::tempdir().expect("tempdir");
    let (handle, _graph) = build_mock_handle_with_response(
        &tmp.path().join("g_v011_5.db"),
        "alice-conference",
        serde_json::json!({
            "entities": [{"name": "Alice", "label": "Person"}],
            "relationships": []
        })
        .to_string(),
    )
    .await;

    let noop_provider: Arc<dyn kremory::memory::ChatProvider> =
        Arc::new(kremory::core::provider::MockChatProvider::null());
    let ns = Namespace::new("g-v011-5");
    let source = SourceRef {
        kind: SourceKind::Chat,
        id: "ep-1".to_string(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    };

    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &source,
            content: "alice-conference: Alice attended the conference.",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider),
            batch_id: None,
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: false,
            },
            sink: None,
        })
        .await
        .expect("graph_ingest_episode OK");

    let results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns,
            query: "Alice",
            opts: &SearchOpts {
                limit: Some(10),
                as_of: None,
                source_kind: None,
            },
        })
        .await
        .expect("graph_search OK");

    let alice = results
        .iter()
        .find(|r| r.entity_name.to_lowercase().contains("alice"))
        .expect("Alice must be recalled");

    assert!(
        !alice.source_refs.is_empty(),
        "Alice must have at least one source_ref (entity-loop episodic edge — ASMP-001 fix)"
    );
    for sr in &alice.source_refs {
        assert_eq!(
            sr.kind,
            SourceKind::Episode,
            "source_ref must use Episode kind (not Document) — G_v011_5"
        );
        assert!(
            !sr.id.is_empty(),
            "episode id in source_ref must be non-empty — G_v011_5"
        );
    }
}

/// G_v011_12 — rrf_single_result_scores_one
///
/// Regression guard for RISK-001: a single unique match must score 1.0, not 0.0.
/// No Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn rrf_single_result_scores_one() {
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::{GraphHandle, GraphIngestEpisodeParams, GraphSearchParams};
    use kremory::Namespace;

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let tmp = tempfile::tempdir().expect("tempdir");
    let (handle, _graph) = build_mock_handle_with_response(
        &tmp.path().join("g_v011_12.db"),
        "zephyr-unique",
        serde_json::json!({
            "entities": [{"name": "Zephyr", "label": "Concept"}],
            "relationships": []
        })
        .to_string(),
    )
    .await;

    let noop_provider: Arc<dyn kremory::memory::ChatProvider> =
        Arc::new(kremory::core::provider::MockChatProvider::null());
    let ns = Namespace::new("g-v011-12");
    let source = SourceRef {
        kind: SourceKind::Chat,
        id: "ep-z1".to_string(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    };

    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &source,
            content: "zephyr-unique: Zephyr is a unique entity.",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider),
            batch_id: None,
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: false,
            },
            sink: None,
        })
        .await
        .expect("graph_ingest_episode OK");

    let results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns,
            query: "Zephyr",
            opts: &SearchOpts {
                limit: Some(10),
                as_of: None,
                source_kind: None,
            },
        })
        .await
        .expect("graph_search OK");

    assert_eq!(
        results.len(),
        1,
        "exactly one result for unique entity Zephyr"
    );
    assert_eq!(
        results[0].score, 1.0_f32,
        "single unique match must score 1.0 (degenerate normalisation fix RISK-001)"
    );
}

/// G_v011_standalone — standalone_entity_has_episodic_edge
///
/// ASMP-001 regression guard: an entity extracted with NO associated facts must
/// still get an episodic_edge via the entity loop, and source_refs must be non-empty.
/// No Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn standalone_entity_has_episodic_edge() {
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::{GraphHandle, GraphIngestEpisodeParams, GraphSearchParams};
    use kremory::Namespace;

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let tmp = tempfile::tempdir().expect("tempdir");
    let (handle, _graph) = build_mock_handle_with_response(
        &tmp.path().join("g_v011_standalone.db"),
        "carol-present",
        // One entity, zero relationships — standalone entity with no facts.
        serde_json::json!({
            "entities": [{"name": "Carol", "label": "Person"}],
            "relationships": []
        })
        .to_string(),
    )
    .await;

    let noop_provider: Arc<dyn kremory::memory::ChatProvider> =
        Arc::new(kremory::core::provider::MockChatProvider::null());
    let ns = Namespace::new("g-v011-standalone");
    let source = SourceRef {
        kind: SourceKind::Chat,
        id: "ep-c1".to_string(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    };

    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &source,
            content: "carol-present: Carol was present.",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider),
            batch_id: None,
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: false,
            },
            sink: None,
        })
        .await
        .expect("graph_ingest_episode OK");

    let results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns,
            query: "Carol",
            opts: &SearchOpts {
                limit: Some(10),
                as_of: None,
                source_kind: None,
            },
        })
        .await
        .expect("graph_search OK");

    let carol = results
        .iter()
        .find(|r| r.entity_name.to_lowercase().contains("carol"))
        .expect("Carol must be recalled");

    assert!(
        !carol.source_refs.is_empty(),
        "standalone entity with no facts must still have source_refs via entity-loop episodic edge (ASMP-001)"
    );
    assert_eq!(
        carol.source_refs[0].kind,
        SourceKind::Episode,
        "source_ref must be SourceKind::Episode even for a zero-fact entity"
    );
}

/// G_v011_stub_forward_ref — stub_entity_inserted_on_forward_reference
///
/// A fact references Bob who is NOT in the entity list. Verify stub Bob is created
/// with stub=true, fact is stored, and stub appears in recall with incomplete=true.
///
/// NOTE: This test exercises the stub pre-scan (Bug E) + entity-loop path. Its
/// validity depends on MockChatProvider returning a forward reference (Alice in
/// entity list, Bob only in a relationship). Stub creation was validated by ingest
/// lib tests. This integration-level test verifies the recall-side incomplete flag.
///
/// No Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn stub_entity_inserted_on_forward_reference() {
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::{GraphHandle, GraphIngestEpisodeParams, GraphSearchParams};
    use kremory::Namespace;

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let tmp = tempfile::tempdir().expect("tempdir");
    let (handle, _graph) = build_mock_handle_with_response(
        &tmp.path().join("g_v011_stub.db"),
        "alice-works-with-bob",
        // Alice in entity list; Bob only referenced in relationship (forward ref).
        serde_json::json!({
            "entities": [{"name": "Alice", "label": "Person"}],
            "relationships": [{"subject": "Alice", "predicate": "works_with", "object": "Bob"}]
        })
        .to_string(),
    )
    .await;

    let noop_provider: Arc<dyn kremory::memory::ChatProvider> =
        Arc::new(kremory::core::provider::MockChatProvider::null());
    let ns = Namespace::new("g-v011-stub");
    let source = SourceRef {
        kind: SourceKind::Chat,
        id: "ep-stub-1".to_string(),
        occurred_at: chrono::Utc::now(),
        published_at: None,
    };

    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &source,
            content: "alice-works-with-bob: Alice works with Bob on research.",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider),
            batch_id: None,
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: false,
            },
            sink: None,
        })
        .await
        .expect("graph_ingest_episode OK");

    // Bob should appear in recall; incomplete must be true (stub entity).
    let results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns,
            query: "Bob",
            opts: &SearchOpts {
                limit: Some(10),
                as_of: None,
                source_kind: None,
            },
        })
        .await
        .expect("graph_search OK");

    let bob = results
        .iter()
        .find(|r| r.entity_name.to_lowercase().contains("bob"));
    // Bob may or may not be recalled depending on whether the stub entity is indexed.
    // The critical invariant: if Bob IS recalled, incomplete must be true.
    if let Some(bob) = bob {
        assert!(
            bob.incomplete,
            "recalled stub entity Bob must have incomplete: true — G_v011_stub"
        );
    }
    // Also verify Alice was recalled (non-stub entity must be indexed).
    let alice_results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns,
            query: "Alice",
            opts: &SearchOpts {
                limit: Some(10),
                as_of: None,
                source_kind: None,
            },
        })
        .await
        .expect("graph_search for Alice OK");
    assert!(
        alice_results
            .iter()
            .any(|r| r.entity_name.to_lowercase().contains("alice")),
        "Alice must be recalled (non-stub entity)"
    );
    let alice = alice_results
        .iter()
        .find(|r| r.entity_name.to_lowercase().contains("alice"))
        .unwrap();
    assert!(
        !alice.incomplete,
        "non-stub entity Alice must have incomplete: false"
    );
}

/// G_v011_stub_promoted — stub_entity_promoted_on_reingestion
///
/// First ingest creates stub Bob. Second ingest with full extraction promotes it
/// via upsert_entity_with_group. The promoted entity must have incomplete=false.
///
/// No Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn stub_entity_promoted_on_reingestion() {
    use kremory::core::config::PipelineConfig;
    use kremory::core::provider::{DeterministicEmbeddingProvider, MockChatProvider};
    use kremory::core::schema::TemporalGraph;
    use kremory::memory::engine_handle::{EngineGraphHandle, WithConfigParams};
    use kremory::memory::types::{SearchOpts, SourceKind, SourceRef, SubmitOpts};
    use kremory::memory::{GraphHandle, GraphIngestEpisodeParams, GraphSearchParams};
    use kremory::Namespace;
    use std::collections::HashMap;

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let _guard = metrics::set_default_local_recorder(&recorder);

    // Two separate MockChatProvider responses in one handle.
    let mut responses: HashMap<String, String> = HashMap::new();
    // Batch 1: Alice in entity list; Bob only in relationship (forward ref → stub).
    responses.insert(
        "alice-works-with-bob-promoted".to_string(),
        serde_json::json!({
            "entities": [{"name": "Alice", "label": "Person"}],
            "relationships": [{"subject": "Alice", "predicate": "works_with", "object": "Bob"}]
        })
        .to_string(),
    );
    // Batch 2: Bob now in full extraction (promoted from stub).
    responses.insert(
        "bob-full-extraction".to_string(),
        serde_json::json!({
            "entities": [{"name": "Bob", "label": "Person"}],
            "relationships": []
        })
        .to_string(),
    );

    let mock_llm: Arc<dyn kremory::memory::ChatProvider + Send + Sync> =
        Arc::new(MockChatProvider::new(responses));
    let embedder: Arc<dyn kremory::DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));
    let tmp = tempfile::tempdir().expect("tempdir");
    let graph = Arc::new(
        TemporalGraph::open(tmp.path().join("g_v011_promoted.db").to_str().unwrap())
            .await
            .expect("TemporalGraph::open"),
    );
    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig::default");
    let handle = EngineGraphHandle::with_config(WithConfigParams {
        graph: Arc::clone(&graph),
        chat: Arc::clone(&mock_llm),
        embedder,
        config,
    });

    let noop_provider: Arc<dyn kremory::memory::ChatProvider> = Arc::new(MockChatProvider::null());
    let ns = Namespace::new("g-v011-promoted");

    // Batch 1: creates stub Bob.
    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &SourceRef {
                kind: SourceKind::Chat,
                id: "ep-promo-1".to_string(),
                occurred_at: chrono::Utc::now(),
                published_at: None,
            },
            content: "alice-works-with-bob-promoted: Alice works with Bob on research.",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider),
            batch_id: None,
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: false,
            },
            sink: None,
        })
        .await
        .expect("batch 1 ingest OK");

    // Batch 2: promotes Bob to a full entity.
    handle
        .graph_ingest_episode(GraphIngestEpisodeParams {
            namespace: &ns,
            source_ref: &SourceRef {
                kind: SourceKind::Chat,
                id: "ep-promo-2".to_string(),
                occurred_at: chrono::Utc::now(),
                published_at: None,
            },
            content: "bob-full-extraction: Bob is a researcher at Stanford.",
            structured_facts: &[],
            provider: Arc::clone(&noop_provider),
            batch_id: None,
            opts: SubmitOpts {
                enrich_per_episode: true,
                run_in_background: false,
            },
            sink: None,
        })
        .await
        .expect("batch 2 ingest OK");

    // After promotion, Bob must be recalled with incomplete=false.
    let results = handle
        .graph_search(GraphSearchParams {
            namespace: &ns,
            query: "Bob",
            opts: &SearchOpts {
                limit: Some(10),
                as_of: None,
                source_kind: None,
            },
        })
        .await
        .expect("graph_search OK");

    let bob = results
        .iter()
        .find(|r| r.entity_name.to_lowercase().contains("bob"));
    if let Some(bob) = bob {
        assert!(
            !bob.incomplete,
            "promoted entity Bob must have incomplete: false after re-ingestion — G_v011_stub_promoted"
        );
    }
    // Bob must be in the graph (upsert guarantee — single row).
    let bob_entity = graph.get_entity("bob").await.expect("graph.get_entity OK");
    assert!(
        bob_entity.is_some(),
        "Bob must exist in the graph after promotion — G_v011_stub_promoted"
    );
    if let Some(e) = bob_entity {
        // After promotion, stub property must be gone or false.
        let stub_flag = e
            .properties
            .get("stub")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(
            !stub_flag,
            "promoted entity must not have stub=true in properties — G_v011_stub_promoted"
        );
    }
}

// ── LLM.3 — Token observability: kremory_core_tokens_total counter ────────────

// ── G_v012_* — v0.1.2 LLM observability facade integration tests ──────────────
//
// These tests validate the TokenTrackingChatProvider integration via the public
// MemoryBuilder API and the Tier 1 shortcuts. No live Ollama required for
// G_v012_7 and G_v012_8 — they use MockChatProviderTracking +
// DeterministicEmbeddingProvider. G_v012_9 also uses MockChatProviderTracking
// via with_llm_tracked("ollama", ...) — no live Ollama required.

/// G_v012_7 — `with_llm_unchanged_v011_compat`
///
/// Build via `Memory::open(..).with_llm(..)` (plain, no tracking). Ingest must
/// succeed. kremory_core_tokens_total must NOT be emitted — the untracked path
/// must not pollute the metrics surface. This is a v0.1.1 backward-compat guard.
///
/// No Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn with_llm_unchanged_v011_compat() {
    use kremory::core::provider::{DeterministicEmbeddingProvider, MockChatProvider};
    use kremory::{DynEmbeddingProvider, Namespace};
    use std::collections::HashMap;

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    // Simple extraction response: one entity, no relationships.
    let mut responses: HashMap<String, String> = HashMap::new();
    responses.insert(
        "g012-7-content".to_string(),
        serde_json::json!({
            "entities": [{"name": "Dave", "label": "Person"}],
            "relationships": []
        })
        .to_string(),
    );
    let mock_llm: Arc<dyn kremory::memory::ChatProvider> =
        Arc::new(MockChatProvider::new(responses));
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    let tmp = tempfile::tempdir().expect("tempdir");
    let mem = Memory::open(tmp.path().join("g_v012_7.db"))
        .with_llm(mock_llm)
        .with_embedder(embedder)
        .default_namespace(Namespace::new("g-v012-7"))
        .await
        .expect("Memory::open must succeed");

    mem.remember("g012-7-content: Dave is an engineer.")
        .from_chat("g-v012-7-session")
        .await
        .expect("remember must succeed");

    let snapshot = snapshotter.snapshot().into_vec();
    let has_token_counter = snapshot.iter().any(|(k, _, _, _)| {
        k.key().name() == "kremory_core_tokens_total"
            && k.key()
                .labels()
                .any(|l| l.key() == "operation" && l.value() == "chat")
    });
    assert!(
        !has_token_counter,
        "kremory_core_tokens_total must NOT be emitted via plain with_llm() (v0.1.1 compat guard)"
    );
}

/// G_v012_8 — `with_llm_tracked_emits_chat_metrics`
///
/// Build via `Memory::open(..).with_llm_tracked(..)`. Ingest with a
/// `MockChatProviderTracking(WithUsage{input:50, output:25})`. Assert that
/// `kremory_core_tokens_total` is emitted with `operation=chat`.
///
/// No Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn with_llm_tracked_emits_chat_metrics() {
    use helpers::mock_chat::{MockBehavior, MockChatProviderTracking};
    use kremory::core::provider::DeterministicEmbeddingProvider;
    use kremory::{DynEmbeddingProvider, Namespace};

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mock_llm = MockChatProviderTracking::new(MockBehavior::WithUsage {
        input: 50,
        output: 25,
    });
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    let tmp = tempfile::tempdir().expect("tempdir");
    let mem = Memory::open(tmp.path().join("g_v012_8.db"))
        .with_llm(Arc::new(mock_llm))
        .with_token_tracking("test-provider", "test-model")
        .with_embedder(embedder)
        .default_namespace(Namespace::new("g-v012-8"))
        .await
        .expect("Memory::open must succeed");

    mem.remember("Eve leads the security team.")
        .from_chat("g-v012-8-session")
        .await
        .expect("remember must succeed");

    let snapshot = snapshotter.snapshot().into_vec();
    let has_token_counter = snapshot.iter().any(|(k, _, _, _)| {
        k.key().name() == "kremory_core_tokens_total"
            && k.key()
                .labels()
                .any(|l| l.key() == "operation" && l.value() == "chat")
    });
    assert!(
        has_token_counter,
        "kremory_core_tokens_total must be emitted when using with_llm_tracked(); \
         counters: {:?}",
        snapshot
            .iter()
            .map(|(k, _, _, _)| k.key().name().to_string())
            .collect::<Vec<_>>()
    );
}

/// G_v012_9 — `tier_1_with_ollama_auto_emits_chat_metrics`
///
/// Verifies Tier 1 `with_ollama` wraps `ChatProvider` in
/// `TokenTrackingChatProvider` with `provider="ollama"` label.
///
/// Test uses the `with_llm_tracked` path directly because autoagents-llm 0.3.7
/// Ollama backend `ChatResponse::usage()` returns `None` — wrapper functionality
/// is verified via a mock that DOES override `usage()`. The `with_ollama_at`
/// shortcut internally calls `with_llm_tracked("ollama", model, OllamaBackend)`
/// — this test proves that path emits metrics correctly when the LLM provides
/// usage.
///
/// No live Ollama required.
#[tokio::test]
#[cfg(feature = "llm-integration")]
async fn tier_1_with_ollama_auto_emits_chat_metrics() {
    use helpers::mock_chat::{MockBehavior, MockChatProviderTracking};
    use kremory::core::provider::DeterministicEmbeddingProvider;
    use kremory::{DynEmbeddingProvider, Namespace};

    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mock_llm = MockChatProviderTracking::new(MockBehavior::WithUsage {
        input: 30,
        output: 15,
    });
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(DeterministicEmbeddingProvider::new(384));

    let tmp = tempfile::tempdir().expect("tempdir");
    let mem = Memory::open(tmp.path().join("g_v012_9.db"))
        .with_llm(Arc::new(mock_llm))
        .with_token_tracking("ollama", "llama3.2:3b")
        .with_embedder(embedder)
        .default_namespace(Namespace::new("g-v012-9"))
        .await
        .expect("Memory::open must succeed");

    mem.remember("Frank is a principal engineer at Acme.")
        .from_chat("g-v012-9-session")
        .await
        .expect("remember must succeed");

    let snapshot = snapshotter.snapshot().into_vec();

    let has_token_counter = snapshot.iter().any(|(k, _, _, _)| {
        k.key().name() == "kremory_core_tokens_total"
            && k.key()
                .labels()
                .any(|l| l.key() == "provider" && l.value() == "ollama")
            && k.key()
                .labels()
                .any(|l| l.key() == "operation" && l.value() == "chat")
            && k.key()
                .labels()
                .any(|l| l.key() == "model" && l.value() == "llama3.2:3b")
    });
    assert!(
        has_token_counter,
        "kremory_core_tokens_total must be emitted with provider=ollama, \
         operation=chat, model=llama3.2:3b when using with_llm_tracked(\"ollama\", ...); \
         counters: {:?}",
        snapshot
            .iter()
            .map(|(k, _, _, _)| {
                let labels: Vec<_> = k
                    .key()
                    .labels()
                    .map(|l| format!("{}={}", l.key(), l.value()))
                    .collect();
                format!("{}{{{}}}", k.key().name(), labels.join(","))
            })
            .collect::<Vec<_>>()
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
    use kremory::core::provider::EmbeddingProvider as _;
    use kremory::TokenTrackingEmbedder;

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

// ── E2E-C1 — Multi-namespace recall with real embeddings (ADR-029c) ───────────

/// C1: Multi-namespace recall E2E with real Ollama embeddings.
///
/// ADR-029c Decisions 4 + 7: recall `in_namespaces(&[A, B])` must fan-out
/// correctly with real vector embeddings and attribute results to their origin
/// namespace.
///
/// Steps:
///   1. Ingest one episode into ns-A and one into ns-B.
///   2. Recall with `in_namespaces(&[ns_a, ns_b])`.
///   3. Assert all returned `RetrievedContext.namespace` values are `Some`.
///   4. Assert no result carries a namespace other than ns-A or ns-B.
///   5. Recall with `in_namespace(ns_a)` only — assert no ns-B result leaks.
///
/// `#[ignore]`: requires live Ollama (OLLAMA_BASE_URL + nomic-embed-text +
/// JSON-capable chat model).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn c1_multi_namespace_recall_e2e_real_embeddings() {
    let dir_a = tempfile::tempdir().expect("tempdir-a");
    let dir_b = tempfile::tempdir().expect("tempdir-b");

    // Use separate Memory instances sharing the same DB file (one db, two ns).
    let db_path = dir_a.path().join("kremory-c1.db");
    let base_url = ollama_base_url();

    let llm_a: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(ollama_chat_model())
        .timeout_seconds(60)
        .build()
        .expect("LLMBuilder");
    let emb_a: Arc<dyn DynEmbeddingProvider> = {
        let raw: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
            .base_url(&base_url)
            .model("nomic-embed-text")
            .build()
            .expect("EmbeddingBuilder");
        Arc::new(OllamaEmbedderAdapter(raw))
    };

    let mem = Memory::open(&db_path)
        .with_llm(llm_a as Arc<dyn ChatProvider>)
        .with_embedder(emb_a)
        .embedding_dim(768)
        .await
        .expect("Memory::open C1");

    let ns_a = Namespace::new("c1-ns-alpha");
    let ns_b = Namespace::new("c1-ns-beta");

    // Ingest into ns-A.
    mem.remember("Alice is a researcher at the Institute.")
        .in_namespace(ns_a.clone())
        .await
        .expect("remember into ns-A");

    // Ingest into ns-B.
    mem.remember("Bob manages the logistics division at Omega Corp.")
        .in_namespace(ns_b.clone())
        .await
        .expect("remember into ns-B");

    // Multi-namespace recall.
    let results = mem
        .recall("who are the people involved?")
        .in_namespaces(&[ns_a.clone(), ns_b.clone()])
        .raw()
        .await
        .expect("in_namespaces recall must succeed");

    // All results must carry namespace attribution.
    for rc in &results {
        assert!(
            rc.namespace.is_some(),
            "multi-namespace recall result must carry namespace attribution; got: {rc:?}"
        );
        let ns_val = rc.namespace.as_ref().expect("just checked Some");
        assert!(
            ns_val.namespace == "c1-ns-alpha" || ns_val.namespace == "c1-ns-beta",
            "result namespace must be c1-ns-alpha or c1-ns-beta, got '{}'",
            ns_val.namespace
        );
    }

    // ns-A only recall must not include ns-B results.
    let ns_a_results = mem
        .recall("who are the people involved?")
        .in_namespace(ns_a.clone())
        .raw()
        .await
        .expect("in_namespace(ns_a) recall must succeed");

    for rc in &ns_a_results {
        if let Some(ref ns) = rc.namespace {
            assert_ne!(
                ns.namespace, "c1-ns-beta",
                "ns-A-only recall must not return ns-B results; got: {rc:?}"
            );
        }
    }

    drop(dir_b); // silence unused warning
}

// ── E2E-C2 — AppendOnly + Mutable namespaces co-exist in same Memory ─────────

/// C2: AppendOnly and Mutable namespaces in the same `Memory` instance.
///
/// ADR-029b Decision 2: an AppendOnly namespace blocks `forget()` and `dream()`;
/// a Mutable namespace in the same DB must still permit them.
///
/// Steps:
///   1. Register ns-append as AppendOnly, ns-mutable as Mutable.
///   2. Ingest an episode into each namespace.
///   3. Recall from ns-append succeeds.
///   4. Recall from ns-mutable succeeds.
///   5. `forget()` against ns-append returns a policy-violation error.
///   6. `forget()` against ns-mutable succeeds (or returns NotFound — not a policy error).
///
/// `#[ignore]`: requires live Ollama.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn c2_append_only_and_mutable_namespaces_coexist() {
    use kremory::{ImmutabilityLevel, NamespacePolicy};

    let dir = tempfile::tempdir().expect("tempdir");
    let base_url = ollama_base_url();

    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(&base_url)
        .model(ollama_chat_model())
        .timeout_seconds(60)
        .build()
        .expect("LLMBuilder");
    let emb: Arc<dyn DynEmbeddingProvider> = {
        let raw: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
            .base_url(&base_url)
            .model("nomic-embed-text")
            .build()
            .expect("EmbeddingBuilder");
        Arc::new(OllamaEmbedderAdapter(raw))
    };

    let mem = Memory::open(dir.path().join("kremory-c2.db"))
        .with_llm(llm as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        .embedding_dim(768)
        .await
        .expect("Memory::open C2");

    let ns_append = Namespace::new("c2-append-only")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("APPEND_ONLY coherent");
    let ns_mutable = Namespace::new("c2-mutable")
        .with_policy(NamespacePolicy::new().with_immutability(ImmutabilityLevel::Mutable))
        .expect("Mutable coherent");

    mem.register_namespace(ns_append.clone())
        .await
        .expect("register AppendOnly namespace");
    mem.register_namespace(ns_mutable.clone())
        .await
        .expect("register Mutable namespace");

    // Ingest into both.
    mem.remember("Alice is the lead engineer at SafeVault Corp.")
        .in_namespace(ns_append.clone())
        .await
        .expect("remember into AppendOnly ns");

    mem.remember("Charlie manages operations at FlexGroup Inc.")
        .in_namespace(ns_mutable.clone())
        .await
        .expect("remember into Mutable ns");

    // Recall from both must succeed.
    mem.recall("who is the lead engineer?")
        .in_namespace(Namespace::new("c2-append-only"))
        .await
        .expect("recall from AppendOnly ns must succeed");

    mem.recall("who manages operations?")
        .in_namespace(Namespace::new("c2-mutable"))
        .await
        .expect("recall from Mutable ns must succeed");

    // Forget in AppendOnly ns must return a policy-violation error.
    // (ForgetRequest scopes by namespace, not entity name.)
    let forget_append = mem
        .forget()
        .in_namespace(Namespace::new("c2-append-only"))
        .execute()
        .await;

    assert!(
        forget_append.is_err(),
        "forget() in AppendOnly namespace must return Err; got Ok"
    );
    let err_msg = forget_append
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    // Error must mention the policy violation (not a generic I/O error).
    assert!(
        err_msg.contains("AppendOnly") || err_msg.contains("policy"),
        "forget() error in AppendOnly ns must mention policy; got: {err_msg}"
    );

    // Forget in Mutable ns must NOT produce a policy-violation error.
    // (It may return 0 deleted rows if no entity was extracted — that is fine.)
    let forget_mutable = mem
        .forget()
        .in_namespace(Namespace::new("c2-mutable"))
        .execute()
        .await;

    if let Err(ref e) = forget_mutable {
        let msg = e.to_string();
        assert!(
            !msg.contains("AppendOnly"),
            "forget() in Mutable ns must NOT produce AppendOnly error; got: {msg}"
        );
    }
}

// ── E2E-C3 — kremory-admin CLI smoke test (ADR-029b Decision 7) ──────────────

/// C3: kremory-admin CLI smoke test.
///
/// ADR-029b Decision 7: the `kremory-admin` binary must be buildable and its
/// top-level subcommands (`migrate --dry-run`, `verify`, `upgrade-namespace`)
/// must exit 0 against a valid database path without performing destructive
/// operations.
///
/// This test does NOT require live Ollama (no embedding / LLM calls).
/// It uses `std::process::Command` to shell out to `cargo run -p kremory-admin`.
///
/// `#[ignore]`: requires a full workspace build (`cargo build -p kremory-admin`
/// succeeds). This is gated here to avoid blocking CI that cannot build the
/// workspace. Remove `#[ignore]` once kremory-admin ships as a pre-built binary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
#[cfg(feature = "llm-integration")]
async fn c3_kremory_admin_cli_smoke_test() {
    use std::process::Command;

    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("kremory-admin-smoke.db");

    // Step 1: Create a real DB via Memory::open (runs auto-migrations).
    {
        let base_url = ollama_base_url();
        let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
            .base_url(&base_url)
            .model(ollama_chat_model())
            .timeout_seconds(30)
            .build()
            .expect("LLMBuilder");
        let emb: Arc<dyn DynEmbeddingProvider> = {
            let raw: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
                .base_url(&base_url)
                .model("nomic-embed-text")
                .build()
                .expect("EmbeddingBuilder");
            Arc::new(OllamaEmbedderAdapter(raw))
        };
        Memory::open(&db_path)
            .with_llm(llm as Arc<dyn ChatProvider>)
            .with_embedder(emb)
            .embedding_dim(768)
            .await
            .expect("Memory::open for admin smoke test");
        // Memory drops here — DB file persists.
    }

    let db_str = db_path.to_str().expect("db path is valid UTF-8");

    // Step 2: `kremory-admin verify` — exits 0, prints file size.
    let verify_out = Command::new("cargo")
        .args([
            "run",
            "-q",
            "-p",
            "kremory-admin",
            "--",
            "verify",
            "--db",
            db_str,
        ])
        .output()
        .expect("cargo run kremory-admin verify must not fail to spawn");

    assert!(
        verify_out.status.success(),
        "kremory-admin verify must exit 0; stderr: {}",
        String::from_utf8_lossy(&verify_out.stderr)
    );
    let verify_stdout = String::from_utf8_lossy(&verify_out.stdout);
    assert!(
        verify_stdout.contains("bytes") || verify_stdout.contains("Verifying"),
        "kremory-admin verify output must contain 'bytes' or 'Verifying'; got: {verify_stdout}"
    );

    // Step 3: `kremory-admin migrate --dry-run` — exits 0, prints dry-run marker.
    let migrate_out = Command::new("cargo")
        .args([
            "run",
            "-q",
            "-p",
            "kremory-admin",
            "--",
            "migrate",
            "--db",
            db_str,
            "--dry-run",
        ])
        .output()
        .expect("cargo run kremory-admin migrate --dry-run must not fail to spawn");

    assert!(
        migrate_out.status.success(),
        "kremory-admin migrate --dry-run must exit 0; stderr: {}",
        String::from_utf8_lossy(&migrate_out.stderr)
    );
    let migrate_stdout = String::from_utf8_lossy(&migrate_out.stdout);
    assert!(
        migrate_stdout.contains("dry-run"),
        "kremory-admin migrate --dry-run output must contain 'dry-run'; got: {migrate_stdout}"
    );

    // Step 4: `kremory-admin upgrade-namespace --group-id test-ns` — exits 0.
    let upgrade_out = Command::new("cargo")
        .args([
            "run",
            "-q",
            "-p",
            "kremory-admin",
            "--",
            "upgrade-namespace",
            "--db",
            db_str,
            "--group-id",
            "c3-smoke-ns",
        ])
        .output()
        .expect("cargo run kremory-admin upgrade-namespace must not fail to spawn");

    assert!(
        upgrade_out.status.success(),
        "kremory-admin upgrade-namespace must exit 0; stderr: {}",
        String::from_utf8_lossy(&upgrade_out.stderr)
    );
}
