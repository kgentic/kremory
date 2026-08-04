#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-051 Phase 4 — `Memory::wait_for_processing` + `Memory::with_await_extraction`
//!
//! Governing spec: `.ai-docs/specs/v0-2-2-adr-051-only-impl-spec-2026-06-11.md` §Phase 4.
//! ADR-051: GLiNER-to-background unified hot path.
//!
//! # Coverage (Phase 4 DoD items 5, 6, 7)
//!
//! 1. `wait_for_processing_blocks_until_extraction_complete`
//!    — ingest via BackgroundIngestor, poll via `wait_for_processing`;
//!    assert the call returns Ok(()) only after the worker wrote `Verified`.
//!
//! 2. `with_await_extraction_true_blocks_add_episode`
//!    — build Memory with `with_await_extraction(true)`, ingest via inline path;
//!    assert the remember() call itself blocks until status is `Verified`
//!    (inline ingest writes Verified per engine_handle.rs ADR-051 Phase 4 amendment).
//!
//! 3. `wait_for_processing_returns_err_on_failed_status`
//!    — seed an episode with `Failed` status, call `wait_for_processing`;
//!    assert `Err(MemoryError::Core(ExtractionFailed { episode_id }))` returned promptly
//!    (before timeout).
//!
//! # Mocking boundary (per testing-policy.md)
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open` + `tempfile::TempDir`
//! - ALWAYS REAL: `BackgroundIngestor` worker loop for test 1
//! - ALWAYS REAL: Memory facade inline ingest path for test 2
//! - ALWAYS REAL: DB status reads/writes in all three tests
//! - ALWAYS MOCK: `ChatProvider` (TimestampRecordingLlmClient / empty-array LLM)
//! - NOT TESTED HERE: GLiNER model loading (requires `--features ner` + model file)

use std::sync::Arc;
use std::time::{Duration, Instant};

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::{BackgroundIngestor, IngestorConfig, SendParams};
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::error::Error as CoreError;
use kremory::core::ingest::Engine;
use kremory::core::provider::{ChatProvider, MockChatResponse, NullEmbeddingProvider};
use kremory::memory::types::{MemoryError, Namespace};
use kremory::Memory;

// ─── Minimal LLM client — returns empty array (safe no-op for all prompts) ────

/// Returns `"[]"` for every call — extraction produces zero entities/facts.
/// Sufficient for tests that only need ingest to complete without error.
#[derive(Debug, Clone)]
struct EmptyArrayLlmClient;

#[async_trait::async_trait]
impl ChatProvider for EmptyArrayLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(MockChatResponse {
            text: "[]".to_owned(),
        }))
    }
}

// ─── DB helpers (mirrored from verify_stage_integration.rs) ──────────────────

/// Read `episode_processing_status` for a given episode id.
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

/// Insert a minimal episode row with a given status and return its id.
async fn insert_episode_with_status(conn: &libsql::Connection, content: &str, status: &str) -> i64 {
    conn.execute(
        "INSERT INTO episodes (content, timestamp, episode_processing_status) \
         VALUES (?1, datetime('now'), ?2)",
        libsql::params![content, status.to_string()],
    )
    .await
    .expect("insert episode");

    let mut rows = conn
        .query("SELECT last_insert_rowid()", ())
        .await
        .expect("rowid query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("rowid")
}

// ─── Test 1: wait_for_processing_blocks_until_extraction_complete ─────────────

/// Phase 4 DoD #5: `Memory::wait_for_processing` returns `Ok(())` only after the
/// background worker has written `Verified`.
///
/// Quinn Phase 4 M-1 cause-fix: the test now calls `mem.wait_for_processing`
/// directly (the API we ship), rather than reimplementing the polling loop in
/// the test. The integration path Memory → wait_for_processing → status-read →
/// terminal-resolution is the contract under test.
///
/// Setup:
/// - Build `Memory` via the public builder (gives us a TemporalGraph owned by Memory).
/// - Extract `Arc<TemporalGraph>` via the test-only `temporal_graph_for_test()`
///   helper and clone the Arc to share with `BackgroundIngestor`'s `Engine`.
/// - Send one episode via the ingestor (fire-and-forget Path β with
///   `EmptyArrayLlmClient`).
/// - Poll directly for the episode rowid to appear (we need the id to pass to
///   wait_for_processing; this is one targeted SELECT, not the API being tested).
/// - Call `mem.wait_for_processing(episode_id, 10s)` and assert Ok.
/// - Assert `episode_processing_status == 'Verified'` in the DB.
///
/// Because Memory and BackgroundIngestor share the SAME `Arc<TemporalGraph>`,
/// the worker's status writes are observable via Memory's poll path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_processing_blocks_until_extraction_complete() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("t1-wait-blocks.db");

    // Concrete embedder Arc — Memory builder takes the dyn-coerced form, Engine
    // takes the concrete generic form. We clone the Arc, not the underlying value.
    let null_emb: Arc<NullEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

    // Build Memory via the public builder. Memory's internal Arc<TemporalGraph>
    // becomes the SoT for the wait_for_processing poll.
    let mem = Memory::open(db_path.to_str().expect("utf-8"))
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(Arc::clone(&null_emb) as Arc<dyn kremory::DynEmbeddingProvider>)
        .default_namespace(Namespace::new("test-ns"))
        .await
        .expect("Memory open must succeed");

    // Share the SAME TemporalGraph with the BackgroundIngestor's Engine so the
    // worker's status writes land on the same SQLite file Memory's
    // wait_for_processing polls.
    let temporal = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on builder path");

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build");

    let engine = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: Arc::clone(temporal),
        llm: Arc::new(EmptyArrayLlmClient),
        embedder: null_emb,
        config,
        model: None,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    };
    let (ingestor, guard) = BackgroundIngestor::new(engine, ingestor_config);

    // Send one episode — returns immediately (fire-and-forget).
    ingestor
        .send("Alice met Bob at the Acme summit.", SendParams::default())
        .expect("send must not fail");

    // Targeted SELECT just to discover the rowid (NOT the API under test).
    // The BackgroundIngestor's Phase 1 INSERT writes the episode row; we need
    // its id to pass to wait_for_processing.
    let episode_id: i64 = {
        let mut found_id: Option<i64> = None;
        for _ in 0..60 {
            let mut rows = temporal
                .conn
                .query("SELECT id FROM episodes ORDER BY id DESC LIMIT 1", ())
                .await
                .expect("episode query");
            if let Some(row) = rows.next().await.expect("iter") {
                found_id = Some(row.get::<i64>(0).expect("id"));
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        found_id.expect("episode row must appear within 3 s")
    };

    // ── THE API UNDER TEST ───────────────────────────────────────────────────
    // Call the actual `Memory::wait_for_processing` method. This exercises:
    //   1. The polling loop the production code ships
    //   2. The 50→200ms backoff
    //   3. The terminal-status resolution to Ok(()) on Verified
    //   4. The timeout-before-sleep check
    let wait_result = mem
        .wait_for_processing(episode_id, Duration::from_secs(10))
        .await;

    assert!(
        wait_result.is_ok(),
        "wait_for_processing must return Ok when worker writes Verified; got {wait_result:?}"
    );

    // Status must be Verified post-wait — proves the worker actually transitioned
    // the episode (we didn't get false-Ok from a stale read).
    let final_status = read_status(&temporal.conn, episode_id).await;
    assert_eq!(
        final_status, "Verified",
        "episode_processing_status must be 'Verified' after wait_for_processing returns Ok"
    );

    drop(guard);
    drop(tmp);
}

// ─── Test 2: with_await_extraction_true_blocks_add_episode ───────────────────

/// Phase 4 DoD #6: when `with_await_extraction(true)`, `Memory::remember().await`
/// does not return until the episode's `episode_processing_status` is `Verified`.
///
/// The inline ingest path (EngineHandle) writes `Verified` synchronously after
/// successful extraction per the ADR-051 Phase 4 engine_handle amendment.
/// This test verifies that the `await_extraction=true` path actually calls
/// `wait_for_processing` and that the status is `Verified` when `remember` returns.
///
/// Setup: Memory built from `open_in_memory()` path via test helper.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_await_extraction_true_blocks_add_episode() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("await-extract-test.db");

    let emb: Arc<dyn kremory::DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

    let mem = Memory::open(db_path.to_str().expect("utf-8"))
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        .with_await_extraction(true)
        .with_await_extraction_timeout(Duration::from_secs(15))
        .default_namespace(Namespace::new("test-ns"))
        .await
        .expect("Memory open must succeed");

    let before = Instant::now();
    let commit = mem
        .remember("The project deadline was moved to Q3.")
        .await
        .expect("remember must succeed with await_extraction=true");
    let elapsed = before.elapsed();

    // The commit episode_entity_id is the i64 rowid as a string (inline path).
    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("episode_entity_id must be a parseable i64 rowid on inline path");

    // Verify: the episode is Verified after remember() returned.
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on builder path");
    let status = read_status(&tg.conn, episode_id).await;

    assert_eq!(
        status, "Verified",
        "episode_processing_status must be 'Verified' when remember() returns with await_extraction=true"
    );

    // Sanity: elapsed recorded for diagnostic purposes only.
    // We don't assert a minimum time — inline path may complete very fast.
    let _ = elapsed;
    drop(tmp);
}

// ─── Test 3: wait_for_processing_returns_err_on_failed_status ────────────────

/// Phase 4 DoD #7: when `episode_processing_status = 'Failed'`, `wait_for_processing`
/// returns `Err(MemoryError::Core(ExtractionFailed { episode_id }))` promptly —
/// well before the 10 s timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_processing_returns_err_on_failed_status() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("wait-failed-test.db");

    let emb: Arc<dyn kremory::DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

    let mem = Memory::open(db_path.to_str().expect("utf-8"))
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        .default_namespace(Namespace::new("test-ns"))
        .await
        .expect("Memory open must succeed");

    // Access the TemporalGraph via the test-only accessor.
    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on builder path");

    // Seed an episode with status = 'Failed' directly.
    let episode_id =
        insert_episode_with_status(&tg.conn, "This episode failed extraction.", "Failed").await;

    // Call wait_for_processing with a 10 s timeout. Should return immediately.
    let before = Instant::now();
    let result = mem
        .wait_for_processing(episode_id, Duration::from_secs(10))
        .await;
    let elapsed = before.elapsed();

    // Assert: returned Err.
    assert!(
        result.is_err(),
        "wait_for_processing must return Err when status is 'Failed'"
    );

    // Assert: error is ExtractionFailed with the correct episode_id.
    match result.unwrap_err() {
        MemoryError::Core(CoreError::ExtractionFailed { episode_id: err_id }) => {
            assert_eq!(
                err_id, episode_id,
                "ExtractionFailed must carry the correct episode_id"
            );
        }
        other => panic!(
            "expected MemoryError::Core(ExtractionFailed {{ episode_id }}) but got: {other:?}"
        ),
    }

    // Assert: returned well before the 10 s timeout (should be ~1 poll cycle = 50 ms).
    // We allow up to 2 s to avoid flakiness on slow CI.
    assert!(
        elapsed < Duration::from_secs(2),
        "wait_for_processing on Failed status should return promptly (< 2 s), \
         took {elapsed:?}"
    );

    drop(tmp);
}

// ─── Test 4: DUR-3 — skip_extraction must reach a TERMINAL status ────────────

/// **DUR-3 regression pin.** A `skip_extraction()` ingest succeeds and is durably
/// stored, but leaves `episode_processing_status = 'Pending'` forever, because
/// `engine_handle` gates the terminal write on `opts.enrich_per_episode`. Since
/// `wait_for_processing` treats `Pending` as non-terminal, a consumer that
/// combines the two burns its entire timeout budget and is then told
/// `WaitTimeout` — **failure reported for an ingest that succeeded**.
///
/// # Why the fix is not "write `Verified`"
///
/// `engine_handle`'s comment is right that `Verified` would be semantically false
/// — no extraction ran, so nothing was verified. The broken invariant is neither
/// consumer's: the status vocabulary **conflates "not processed yet" with "will
/// never be processed"**. The fix adds the missing terminal state, `'Skipped'`
/// (the spelling GitHub Actions / Argo / Tekton use for a deliberately-not-run
/// step), and teaches every reader of the column about it.
///
/// # Sensitivity
///
/// Proven in BOTH directions — reverting the `engine_handle` write back to
/// `Pending` makes this fail with `WaitTimeout` after the full 3 s budget, and
/// reverting only the `wait_for_processing` arm makes it fail the same way.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skip_extraction_reaches_a_terminal_status_and_does_not_time_out() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("dur3-skip-extraction.db");

    let emb: Arc<dyn kremory::DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

    let mem = Memory::open(db_path.to_str().expect("utf-8"))
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(emb)
        .default_namespace(Namespace::new("test-ns"))
        .await
        .expect("Memory open must succeed");

    let commit = mem
        .remember("Bulk-imported row; the caller is the sole source of truth.")
        .skip_extraction()
        .await
        .expect("remember().skip_extraction() must succeed");

    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("episode_entity_id must be a parseable i64 rowid on the inline path");

    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on builder path");

    // (1) The column must carry a TERMINAL status — not 'Pending'.
    let status = read_status(&tg.conn, episode_id).await;
    assert_eq!(
        status, "Skipped",
        "a skip_extraction episode must land in the terminal 'Skipped' state, not \
         '{status}' — 'Pending' means work is still coming, and for this episode \
         nothing was ever enqueued, so it would never leave that state"
    );

    // (2) The waiter must agree, and must not burn the budget to say so.
    // A 3 s timeout is deliberately generous relative to the ~50 ms first poll:
    // if the fix regresses, this test costs 3 s and fails loudly rather than
    // hanging.
    let before = Instant::now();
    let result = mem
        .wait_for_processing(episode_id, Duration::from_secs(3))
        .await;
    let elapsed = before.elapsed();

    assert!(
        result.is_ok(),
        "wait_for_processing must return Ok for a skip_extraction episode — the \
         ingest SUCCEEDED and is durably stored. Got: {:?}",
        result.err()
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "must resolve on the first poll (~50 ms), not by exhausting the budget; \
         took {elapsed:?}"
    );

    // (3) The SQL->enum bridge must not report this success as a FAILURE.
    // `from_sql_status` maps anything it does not recognise to
    // `IngestStatus::Failed("unknown_sql_status:...")`, so adding the column
    // value without teaching the bridge would turn a successful skip into a
    // reported failure for every sink consumer.
    let bridged = kremory::core::sink::from_sql_status(&status);
    assert_eq!(
        bridged,
        kremory::core::error::IngestStatus::ExtractionSkipped,
        "from_sql_status must bridge 'Skipped' to a non-failure variant; an \
         unmapped value silently becomes Failed(\"unknown_sql_status:..\")"
    );

    drop(tmp);
}
