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
//! Per the implementer's own flag at review time (the gap was real when
//! written): `declared_reference_time` was threaded on the ENGINE-INLINE
//! ingest path (`memory/engine_handle.rs` lines ~274 and ~391 — both the
//! same-process-spawn "background" branch AND the blocking inline branch of
//! `EngineGraphHandle::graph_ingest_episode` set it from
//! `source_ref.published_at`), but the SEPARATE OS-thread `BackgroundIngestor`
//! deferred path (`memory/background_ingestor_handle.rs:213-221` →
//! `core::background::IngestRequest`) constructed
//! `IngestRequest { reference_time: None, .. }` with NO
//! `declared_reference_time` field on the struct at all.
//!
//! **FIXED 2026-08-20 (TD-187 Gap 1 + Gap 2).** `IngestRequest`,
//! `DeferredRequest`, and `SendParams` (`core/background/mod.rs`,
//! `core/background/ingestor.rs`) now all carry `declared_reference_time`,
//! threaded end to end: `background_ingestor_handle.rs`'s batched and
//! un-batched branches now set it from `source_ref.published_at`; the
//! Phase 1 → Phase 2 handoff in `deferred_pipeline.rs` propagates it from
//! `IngestRequest` onto `DeferredRequest`; the final `ingest_deferred` call
//! (`deferred_pipeline.rs:~683`) now reads it from `req.declared_reference_time`
//! instead of hardcoding `None`. `remember_with_sink_and_no_wait_reaches_stage3_triplet_prompt`
//! below proves this end to end on the real background path — see that
//! test's own doc comment for why it asserts the branch was taken BEFORE
//! asserting the outcome.
//!
//! `Memory::remember(text).published_at(ts).await` — no `.no_wait()` on the
//! request, no `.with_event_sink()` on the builder — resolves `SubmitOpts {
//! run_in_background: false, .. }`, which routes to `EngineGraphHandle`'s
//! INLINE branch regardless of whether the top-level handle is a bare
//! `EngineGraphHandle` or a `BackgroundIngestorGraphHandle` (its
//! `run_in_background == false` arm also delegates to `EngineGraphHandle`).
//! So THIS is the exact path a normal consumer drives, and the one under
//! test in the first two tests below. The third test below drives the
//! OTHER path — `.with_event_sink()` on the builder AND `.no_wait()` on the
//! request — which is the one Gap 1/Gap 2 actually lived on.

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

// ─── Gap 1 + Gap 2: the SEPARATE OS-thread BackgroundIngestor path ───────────

/// Records every `on_stage_change` call, keeping the LATEST stage.
/// `deferred_pipeline.rs`'s `process_item` fires this at entry to Phase 1, on
/// the background OS thread — the INLINE `EngineGraphHandle` path never
/// calls it. So a non-empty count is proof the request actually crossed onto
/// the `BackgroundIngestor` path, not just that a fact eventually got
/// persisted (which the inline path would also do).
///
/// Tracking the LATEST stage (not just a count) matters because `Memory::open`
/// itself fires `warm_schema_caches` (`extraction/structured.rs`) — 11
/// trivial "ok" prompts sent to warm the schema cache, `#[cfg(not(test))]`-gated
/// so it runs in integration test binaries. On the background path those
/// warmup calls land BEFORE the worker thread even starts Phase 1, so
/// polling "is any prompt captured yet" alone races the warmup batch and
/// returns too early — before real extraction has run. Polling until the
/// sink reaches a TERMINAL stage (`Complete`/`Failed`) is what actually
/// proves Phase 1 + Phase 2 finished on the background thread.
#[derive(Clone, Default)]
struct StageChangeCountingSink {
    count: Arc<std::sync::atomic::AtomicUsize>,
    latest: Arc<Mutex<Option<kremory::core::error::IngestStatus>>>,
}

impl StageChangeCountingSink {
    fn count(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn is_terminal(&self) -> bool {
        matches!(
            *self.latest.lock().expect("StageChangeCountingSink poisoned"),
            Some(kremory::core::error::IngestStatus::Complete)
                | Some(kremory::core::error::IngestStatus::Failed(_))
        )
    }
}

impl kremory::core::sink::IngestEventSink for StageChangeCountingSink {
    fn on_stage_change(&self, stage: kremory::core::error::IngestStatus) {
        self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *self.latest.lock().expect("StageChangeCountingSink poisoned") = Some(stage);
    }
}

impl kremory::memory::events::EnrichmentEventSink for StageChangeCountingSink {
    fn on_community_updated(&self, _community_id: &str, _member_count: usize) {}
    fn on_batch_phase2_complete(&self, _event: kremory::memory::events::BatchPhase2Complete) {}
}

/// TD-187 Gap 1 + Gap 2 wiring proof: the SAME date-grounding assertion as
/// `remember_published_at_reaches_stage3_triplet_prompt` above, but driven
/// through `.with_event_sink()` (builder) + `.no_wait()` (request) — the ONLY
/// combination that reaches `BackgroundIngestor`'s OS-thread deferred path
/// (`background_ingestor_handle.rs`'s un-batched branch → `ingestor.rs::send`
/// → `deferred_pipeline.rs`'s `process_item` then `process_deferred`).
///
/// Per R-05: a test that only asserts the final prompt content, without
/// first proving the intended branch actually ran, silently re-tests the
/// INLINE path and passes for the wrong reason if routing regresses — the
/// inline path would produce the same prompt content via a completely
/// different, already-fixed code path, making a routing regression
/// invisible. `StageChangeCountingSink` closes that hole: it can ONLY
/// receive calls from the background dispatch path, so asserting
/// `sink.count() > 0` FIRST proves the branch, and only then is the
/// date-grounding assertion evidence about Gap 1 + Gap 2 specifically.
///
/// Manually verified RED before the fix: temporarily reverting
/// `deferred_pipeline.rs`'s `ingest_deferred(..)` call back to
/// `declared_reference_time: None` (the pre-fix state) makes this test fail
/// on the date-grounding assertion while `sink.count() > 0` still holds —
/// i.e. it fails for the RIGHT reason (grounding didn't reach the prompt),
/// not because the branch wasn't exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remember_with_sink_and_no_wait_reaches_stage3_triplet_prompt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let capturing = Arc::new(CapturingChatProvider::new());
    let llm: Arc<dyn ChatProvider> = capturing.clone();
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(384));
    let sink = StageChangeCountingSink::default();

    let mem = Memory::open(dir.path().join("kremory.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(384)
        .with_event_sink(Arc::new(sink.clone()))
        .default_namespace(Namespace::new("td187-wiring-background"))
        .await
        .expect("Memory::open facade with .with_event_sink()");

    let ts = fixed_published_at();

    let _commit = mem
        .remember("Alice joined the company last week.")
        .published_at(ts)
        .no_wait()
        .await
        .expect("remember must succeed on the background path");

    // Poll for a TERMINAL stage, not merely "any prompt captured" — the
    // latter races `warm_schema_caches`'s 11 warmup "ok" prompts, which land
    // synchronously at `Memory::open` time, before the background worker
    // thread has even started Phase 1 (see `StageChangeCountingSink`'s doc
    // comment).
    for _ in 0..100 {
        if sink.is_terminal() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Assert the branch was taken BEFORE asserting the outcome (R-05).
    assert!(
        sink.count() > 0,
        "on_stage_change never fired — the request did not cross onto the \
         BackgroundIngestor OS-thread path at all (.with_event_sink() + .no_wait() \
         should force it); this test proves nothing about Gap 1/Gap 2 if the \
         branch wasn't exercised"
    );
    assert!(
        sink.is_terminal(),
        "background pipeline never reached a terminal stage (Complete/Failed) \
         within the poll deadline — sink.count()={}; cannot evaluate Gap 1/Gap 2 \
         without knowing extraction actually finished",
        sink.count()
    );

    let prompts = capturing.snapshot();

    assert!(
        !prompts.is_empty(),
        "background branch confirmed taken (sink.count()={}), but no \
         chat_with_tools call was captured — the extraction pipeline never \
         fired on the background path",
        sink.count()
    );

    assert!(
        prompts
            .iter()
            .any(|p| p.contains("The source document is dated 2019-03-15.")),
        "TD-187 Gap 1/Gap 2: background branch confirmed taken \
         (sink.count()={}), but no captured prompt on the BackgroundIngestor \
         OS-thread path carried the caller-declared published_at anchor — \
         declared_reference_time is not reaching ingest_deferred's \
         ExtractionContext on this path. Captured prompts:\n{prompts:#?}",
        sink.count()
    );
}
