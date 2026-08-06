//! TD-187 wiring proof: does `Memory::remember().published_at()` — the
//! public, documented API — actually reach the stage-3 triplet extraction
//! prompt?
//!
//! TD-187 wired `ExtractionContext::reference_time` (a caller-DECLARED
//! anchor sourced from `SourceRef::published_at`, never wall-clock) into
//! `build_triplet_prompt`'s date-grounding block
//! (`core/extraction/graphiti.rs`). The unit tests living alongside
//! `build_triplet_prompt` prove the RENDERING is correct given `Some(ts)` —
//! but they hand-construct `TripletPromptParams` directly, so they prove
//! nothing about whether the value actually reaches that call site from the
//! public API. This test drives the real path end-to-end — through
//! `Memory::remember(text).published_at(ts).await`, no test-only shortcuts —
//! with a prompt-CAPTURING `ChatProvider` and inspects what the extractor
//! actually sent the model.
//!
//! Per the implementer's own flag at review time: `declared_reference_time`
//! is threaded on the ENGINE-INLINE ingest path (`memory/engine_handle.rs`
//! lines ~274 and ~391 — both the same-process-spawn "background" branch AND
//! the blocking inline branch of `EngineGraphHandle::graph_ingest_episode`
//! set it from `source_ref.published_at`), but the SEPARATE OS-thread
//! `BackgroundIngestor` deferred path
//! (`memory/background_ingestor_handle.rs:213-221` →
//! `core::background::IngestRequest`) constructs
//! `IngestRequest { reference_time: None, .. }` with NO
//! `declared_reference_time` field on the struct at all — that path is
//! untouched by TD-187 and always extracts with `reference_time: None`.
//!
//! `Memory::remember(text).published_at(ts).await` — no `.no_wait()` on the
//! request, no `.with_sink()` on the builder — resolves `SubmitOpts {
//! run_in_background: false, .. }`, which routes to `EngineGraphHandle`'s
//! INLINE branch regardless of whether the top-level handle is a bare
//! `EngineGraphHandle` or a `BackgroundIngestorGraphHandle` (its
//! `run_in_background == false` arm also delegates to `EngineGraphHandle`).
//! So THIS is the exact path a normal consumer drives, and the one under
//! test here.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

use kremory::core::provider::{
    ChatMessage, ChatProvider, ChatResponse, LLMError, MockChatResponse, MockEmbeddingProvider,
    StructuredOutputFormat, Tool,
};
use kremory::{DynEmbeddingProvider, Memory, Namespace};

// ─── CapturingChatProvider ────────────────────────────────────────────────────

/// Captures every prompt (all messages' content, concatenated) sent through
/// `chat_with_tools`, across every call the ingest pipeline makes (entity
/// extraction, relation-name extraction, triplet extraction, resolution,
/// contradiction — whichever fire). Always responds with an EMPTY string,
/// which every `StructuredCallBuilder` fallback arm treats as "nothing
/// extracted" and returns `Ok` for (see
/// `core/extraction/structured.rs::parse_response_to_value` — an empty
/// response short-circuits to `Value::Object(Map::new())` before any
/// arm-specific parsing), so the pipeline runs to completion without
/// needing per-stage scripted JSON. We only care what was SENT, not what
/// came back.
#[derive(Debug, Clone, Default)]
struct CapturingChatProvider {
    prompts: Arc<Mutex<Vec<String>>>,
}

impl CapturingChatProvider {
    fn new() -> Self {
        Self::default()
    }

    fn snapshot(&self) -> Vec<String> {
        self.prompts
            .lock()
            .expect("CapturingChatProvider prompts mutex poisoned")
            .clone()
    }
}

#[async_trait::async_trait]
impl ChatProvider for CapturingChatProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
        let combined = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        self.prompts
            .lock()
            .expect("CapturingChatProvider prompts mutex poisoned")
            .push(combined);
        Ok(Box::new(MockChatResponse {
            text: String::new(),
        }))
    }
}

fn fixed_published_at() -> DateTime<Utc> {
    "2019-03-15T12:00:00Z"
        .parse::<DateTime<Utc>>()
        .expect("fixed test timestamp must parse")
}

/// Poll the captured prompts until at least one call has landed or the
/// deadline passes. Tolerates inline-vs-deferred Phase 2 timing without
/// hardcoding an assumption about which one `remember()` uses on this build
/// (mirrors the polling pattern already used in
/// `facade_fact_persistence_mock.rs`).
async fn wait_for_any_prompt(capturing: &CapturingChatProvider) -> Vec<String> {
    let mut prompts = capturing.snapshot();
    for _ in 0..100 {
        if !prompts.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        prompts = capturing.snapshot();
    }
    prompts
}

// ─── Positive: .published_at() reaches the prompt ────────────────────────────

/// TD-187 wiring proof (positive): `Memory::remember(text).published_at(ts)`
/// — the real public facade path — must produce a captured prompt
/// containing the date-grounding line `build_triplet_prompt` renders when
/// `reference_time` is `Some`.
///
/// If this fails, `declared_reference_time` is NOT reaching the prompt on
/// the path a normal consumer drives — i.e. TD-187 is dead on arrival for
/// the default (non-`.no_wait()`) `remember()` call, regardless of what the
/// hand-constructed unit tests in `graphiti.rs` prove about rendering in
/// isolation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remember_published_at_reaches_stage3_triplet_prompt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let capturing = Arc::new(CapturingChatProvider::new());
    let llm: Arc<dyn ChatProvider> = capturing.clone();
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));

    let mem = Memory::open(dir.path().join("kremory.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(384)
        .default_namespace(Namespace::new("td187-wiring-positive"))
        .await
        .expect("Memory::open facade");

    let ts = fixed_published_at();

    let commit = mem
        .remember("Alice joined the company last week.")
        .published_at(ts)
        .await
        .expect("remember must succeed");

    // If the episode_entity_id is a parseable rowid (inline path), wait for
    // terminal status explicitly. Either way `wait_for_any_prompt` below is
    // the real oracle — this is defensive, not load-bearing.
    if let Ok(episode_id) = commit.episode_entity_id.parse::<i64>() {
        let _ = mem
            .wait_for_processing(episode_id, Duration::from_secs(30))
            .await;
    }

    let prompts = wait_for_any_prompt(&capturing).await;

    assert!(
        !prompts.is_empty(),
        "no chat_with_tools call was captured at all — the extraction \
         pipeline never fired; cannot evaluate TD-187 wiring"
    );

    assert!(
        prompts
            .iter()
            .any(|p| p.contains("The source document is dated 2019-03-15.")),
        "TD-187: expected at least one captured prompt (stage-3 triplet \
         extraction) to carry the caller-declared published_at anchor via \
         the PUBLIC path Memory::remember(text).published_at(ts).await — got \
         {} captured prompt(s), none contained the date-grounding line. \
         Captured prompts:\n{prompts:#?}",
        prompts.len()
    );
}

// ─── Negative twin: no .published_at() ⇒ no date-grounding block ─────────────

/// TD-187 wiring proof (negative twin): the SAME path WITHOUT
/// `.published_at()` must render byte-identically to pre-TD-187 — no
/// date-grounding line in ANY captured prompt. Guards against a wiring bug
/// that always threads `Some` (e.g. silently defaulting to wall-clock)
/// rather than strictly the caller-declared value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remember_without_published_at_omits_date_grounding_block() {
    let dir = tempfile::tempdir().expect("tempdir");
    let capturing = Arc::new(CapturingChatProvider::new());
    let llm: Arc<dyn ChatProvider> = capturing.clone();
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));

    let mem = Memory::open(dir.path().join("kremory.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(384)
        .default_namespace(Namespace::new("td187-wiring-negative"))
        .await
        .expect("Memory::open facade");

    let commit = mem
        .remember("Alice joined the company last week.")
        .await
        .expect("remember must succeed");

    if let Ok(episode_id) = commit.episode_entity_id.parse::<i64>() {
        let _ = mem
            .wait_for_processing(episode_id, Duration::from_secs(30))
            .await;
    }

    let prompts = wait_for_any_prompt(&capturing).await;

    assert!(
        !prompts.is_empty(),
        "no chat_with_tools call was captured at all — the extraction \
         pipeline never fired; cannot evaluate TD-187 wiring"
    );

    assert!(
        !prompts
            .iter()
            .any(|p| p.contains("The source document is dated")),
        "TD-187: without .published_at(), NO captured prompt should contain \
         the date-grounding line — a caller-declared anchor must never be \
         synthesized from wall-clock or any other implicit source. Captured \
         prompts:\n{prompts:#?}"
    );
}
