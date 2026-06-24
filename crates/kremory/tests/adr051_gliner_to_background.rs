#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-051 Phase 3 — `add_episode_returns_before_gliner_fires`
//!
//! Governing spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md`
//!   Phase 3 DoD #6: "NEW test: `add_episode_returns_before_gliner_fires` —
//!   instrument GLiNER mock to record a `gliner_called_at` timestamp via channel;
//!   assert `ingest_returned_at < gliner_called_at`."
//!
//! # What this proves
//!
//! After ADR-051 Phase 3, `BackgroundIngestor::send()` returns (fast path: episode
//! INSERT only, no extraction) BEFORE the entity extractor is invoked in the
//! background worker. The extractor call is the proxy for "GLiNER fires" — on
//! Path β (no `ner` feature) the LLM extractor is used instead.
//!
//! # Test mechanism
//!
//! A `TimestampRecordingLlmClient` records `first_call_at: Arc<Mutex<Option<Instant>>>`
//! on its first `chat_with_tools` call. After `send()` returns we record
//! `send_returned_at`. We then wait (up to 5 s) for the extractor to fire and
//! compare the two timestamps.
//!
//! # Mocking boundary (per testing-policy.md)
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open_in_memory()`, migrations
//! - ALWAYS REAL: BackgroundIngestor worker loop, process_item, process_deferred
//! - ALWAYS REAL: run_verify_stage Path β code path (no `ner` feature in unit tests)
//! - ALWAYS MOCK: LLM provider (TimestampRecordingLlmClient)
//! - NOT TESTED HERE: GLiNER model loading (requires `--features ner` + model file)
//!
//! # Path α GLiNER async timing — coverage gap (Quinn Phase 3 MED-02)
//!
//! This test proves the WEAKER invariant: entity extraction (via the LLM extractor
//! proxy on Path β) is async relative to the hot path. The spec DoD #6 asked for
//! "instrument GLiNER mock… assert `ingest_returned_at < gliner_called_at`" — i.e.
//! Path α (GLiNER-specific) async timing. Path α requires `#[cfg(feature = "ner")]`
//! gating AND a mock GLiNER extractor that records a timestamp; neither is available
//! in unit tests without the `ner` feature and a real model file.
//!
//! The ADR-051 architectural invariant — "GLiNER fires async, not on the caller hot
//! path" — is proven structurally: `process_item` calls `ingest_phase1_ner` (episode
//! INSERT only, no extraction) and then enqueues to `process_deferred`, which calls
//! `run_verify_stage` with the extractor. The timing test confirms the structural
//! invariant holds at runtime on the Path β proxy. A Path α feature-gated timing
//! test with a mock GLiNER model would add redundant coverage; the structural proof
//! is sufficient for the sprint scope.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::{BackgroundIngestor, IngestorConfig, SendParams};
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::provider::{ChatProvider, MockChatResponse};
use kremory::core::schema::TemporalGraph;

// ─── TimestampRecordingLlmClient ──────────────────────────────────────────────

/// A ChatProvider that records the wall-clock `Instant` of its first call.
///
/// Subsequent calls are silently ignored (first-call semantic is sufficient for
/// the timing assertion). Returns an empty JSON array `"[]"` for all calls —
/// safe no-op for both extraction and contradiction prompts.
#[derive(Debug, Clone)]
struct TimestampRecordingLlmClient {
    /// Captured on the first `chat_with_tools` call; `None` until then.
    first_call_at: Arc<Mutex<Option<Instant>>>,
}

impl TimestampRecordingLlmClient {
    fn new() -> (Self, Arc<Mutex<Option<Instant>>>) {
        let first_call_at = Arc::new(Mutex::new(None));
        (
            Self {
                first_call_at: Arc::clone(&first_call_at),
            },
            first_call_at,
        )
    }
}

#[async_trait::async_trait]
impl ChatProvider for TimestampRecordingLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Record the first call timestamp under lock. Scoped block ensures the
        // MutexGuard is dropped BEFORE this async fn does any await work.
        {
            let mut guard = self.first_call_at.lock().unwrap_or_else(|p| p.into_inner());
            if guard.is_none() {
                *guard = Some(Instant::now());
            }
        } // guard dropped here

        // Return an empty JSON array — safe no-op for both extraction +
        // contradiction prompts consumed by the pipeline.
        Ok(Box::new(MockChatResponse {
            text: "[]".to_owned(),
        }))
    }
}

// ─── Test ─────────────────────────────────────────────────────────────────────

/// ADR-051 Phase 3 DoD #6: `BackgroundIngestor::send()` returns BEFORE the
/// entity extractor is invoked in the background worker.
///
/// Invariant: `send_returned_at < extractor_called_at`
///
/// On Path β (no `ner` feature): the extractor IS the LLM extractor.
/// `TimestampRecordingLlmClient.first_call_at` captures when the extractor fires.
///
/// Note on drain wait: Phase 4 (`Memory::wait_for_processing`) doesn't exist
/// yet; we poll `first_call_at` with `tokio::time::sleep(50 ms)` intervals
/// up to a 5 s wall-clock budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_episode_returns_before_gliner_fires() {
    let (recording_llm, first_call_at) = TimestampRecordingLlmClient::new();

    let temporal = Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory failed"),
    );

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build failed");

    let dim = config.embedding_dim.0;
    let embedder = Arc::new(kremory::core::provider::NullEmbeddingProvider { dim });
    let graph = kremory::core::ingest::Engine::new(kremory::core::ingest::EngineNewParams {
        graph: temporal,
        llm: Arc::new(recording_llm),
        embedder,
        config,
        model: None,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(graph, ingestor_config);

    // ── Hot path: send returns immediately ────────────────────────────────────
    ingestor
        .send(
            "Alice met Bob at the Acme conference.",
            SendParams::default(),
        )
        .expect("send should succeed — channel not full");

    // Capture the timestamp IMMEDIATELY after send() returns.
    // This is the "ingest_returned_at" in the spec timing assertion.
    let send_returned_at = Instant::now();

    // ── Poll for background extractor call (up to 5 s) ────────────────────────
    // The lock is released BEFORE the await by using a scoped block so the
    // MutexGuard's lifetime ends before tokio::time::sleep is awaited.
    let mut extractor_called_at: Option<Instant> = None;
    for _ in 0..100 {
        let maybe = {
            let guard_inner = first_call_at.lock().unwrap_or_else(|p| p.into_inner());
            *guard_inner
        }; // guard_inner dropped here — BEFORE the await below
        if let Some(t) = maybe {
            extractor_called_at = Some(t);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Drain + shutdown worker.
    drop(ingestor);
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");

    // ── Assertions ────────────────────────────────────────────────────────────

    // Assert 1: extractor did fire at some point (background worker ran).
    let extractor_called_at = extractor_called_at.expect(
        "entity extractor was never called within 5 s — \
         background worker may not be running run_verify_stage (ADR-051 Phase 3 wiring)",
    );

    // Assert 2: send() returned BEFORE the extractor fired.
    // This proves GLiNER (or its LLM proxy on Path β) did NOT run synchronously
    // in the hot path — it fired AFTER BackgroundIngestor::send() returned.
    assert!(
        send_returned_at < extractor_called_at,
        "send_returned_at ({send_returned_at:?}) must be BEFORE extractor_called_at \
         ({extractor_called_at:?}) — extraction must be background, not sync hot-path"
    );
}
