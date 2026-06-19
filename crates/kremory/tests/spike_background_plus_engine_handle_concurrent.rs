#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Spike C: BackgroundIngestor + EngineGraphHandle concurrent WAL safety spike
//!
//! Empirically validates the production concurrency shape described in
//! `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §3.3`:
//!
//! > The builder must construct two separate `Engine` instances — one for
//! > `BackgroundIngestor` (background path) and one for `EngineGraphHandle`
//! > (search/dream/inline path). Both engines open the same libSQL database file.
//!
//! ## Difference from Spike B
//!
//! Spike B tested TWO `BackgroundIngestor` instances. The production shape is:
//! - ONE `BackgroundIngestor` (background-path writes: episodes + entities via LLM pipeline)
//! - ONE `EngineGraphHandle` (non-ingest writes: inline ingest `run_in_background=false`,
//!   plus UPDATE-class writes via `graph_assert_entity_type`)
//!
//! This spike tests the EXACT production concurrency shape: BackgroundIngestor
//! enqueuing episodes (OS-thread writes) while EngineGraphHandle performs inline
//! ingest (tokio-task writes) on the SAME libSQL database file simultaneously.
//!
//! ## Risks addressed
//!
//! - R-03b (arch spec §7): "Concurrent NON-INGEST writes from EngineGraphHandle
//!   while BackgroundIngestor ingests"
//!
//! ## Failure modes tested
//!
//! 1. "Database is locked" — libSQL WAL cannot serialize concurrent writers
//! 2. Episode count mismatch — rows silently dropped on write conflict
//! 3. Episode ID collision — autoincrement assigns duplicate PK
//! 4. Worker thread panic — WAL error surfaces as background thread panic

use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use chrono::Utc;
use kremory::core::background::{BackgroundIngestor, IngestGuard, IngestorConfig, SendParams};
use kremory::core::config::{ContentType, PipelineConfig};
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::ingest::Engine;
use kremory::core::provider::{ChatProvider, MockChatResponse, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::engine_handle::{EngineGraphHandle, WithConfigParams};
use kremory::memory::graph::{GraphHandle, GraphIngestEpisodeParams};
use kremory::memory::types::{Namespace, SourceKind, SourceRef, SubmitOpts};

// ---------------------------------------------------------------------------
// EmptyArrayLlmClient — matches Spike B; returns "[]" for all LLM calls.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: build an EngineGraphHandle on the given DB path
// ---------------------------------------------------------------------------

async fn build_engine_handle(db_path: &str) -> EngineGraphHandle {
    let temporal = Arc::new(
        TemporalGraph::open(db_path)
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build");

    let chat: Arc<dyn ChatProvider + Send + Sync> = Arc::new(EmptyArrayLlmClient);
    let null_emb: Arc<dyn kremory::core::provider::DynEmbeddingProvider> =
        Arc::new(NullEmbeddingProvider { dim: 384 });

    EngineGraphHandle::with_config(WithConfigParams {
        graph: temporal,
        chat,
        embedder: null_emb,
        config,
    })
}

// ---------------------------------------------------------------------------
// Helper: build one Engine + BackgroundIngestor pair on the given DB path
// ---------------------------------------------------------------------------

async fn build_engine_ingestor(
    db_path: &str,
    thread_name: &str,
) -> (BackgroundIngestor, IngestGuard) {
    let temporal = Arc::new(
        TemporalGraph::open(db_path)
            .await
            .expect("TemporalGraph::open"),
    );

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build");

    let null_emb = Arc::new(NullEmbeddingProvider { dim: 384 });

    let engine = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: Arc::clone(&temporal),
        llm: Arc::new(EmptyArrayLlmClient),
        embedder: null_emb,
        config,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        thread_name: thread_name.to_string(),
        ..IngestorConfig::default()
    };

    BackgroundIngestor::new(engine, ingestor_config)
}

// ---------------------------------------------------------------------------
// Spike C: BackgroundIngestor + EngineGraphHandle concurrent writes
// ---------------------------------------------------------------------------

/// Spike C: ONE BackgroundIngestor (OS-thread writes) + ONE EngineGraphHandle
/// (inline tokio-task writes) on the SAME libSQL database concurrently.
///
/// This is the EXACT production shape for `BackgroundIngestorGraphHandle`:
/// - `BackgroundIngestor` owns the background ingest Engine
/// - `EngineGraphHandle` owns a second Engine for inline/search/dream paths
/// - Both write to the same libSQL WAL database
///
/// Asserts:
/// 1. All operations succeed — no "Database is locked" errors
/// 2. All 10 episodes are persisted (5 from BackgroundIngestor + 5 from EngineGraphHandle)
/// 3. No episode ID collision
/// 4. Worker thread does NOT panic (spawn_blocking join succeeds)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn spike_c_background_ingestor_plus_engine_handle_concurrent() {
    // ── Setup: single DB path shared by both Engine instances ─────────────────
    let tmp_dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let db_path = tmp_dir
        .join(format!("kremory-spike-c-bg-plus-engine-{nanos}.db"))
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // ── Construct BackgroundIngestor (owns Engine A) ──────────────────────────
    let (ingestor, guard) = build_engine_ingestor(&db_path, "spike-c-bg-worker").await;

    // ── Construct EngineGraphHandle (owns Engine B via with_config, same DB path) ──
    // EngineGraphHandle::with_config opens a new libSQL connection on the same DB.
    let engine_handle = Arc::new(build_engine_handle(&db_path).await);

    // ── Enqueue 5 background episodes via BackgroundIngestor ─────────────────
    let mut bg_send_errors: Vec<String> = vec![];
    for i in 0..5_usize {
        let result = ingestor.send(
            format!("Spike C background episode {i}: concurrent with EngineGraphHandle"),
            SendParams {
                reference_time: None,
                group_id: Some("spike-c-group".to_string()),
                content_type: Some(ContentType::Text),
            },
        );
        if let Err(e) = result {
            bg_send_errors.push(format!("bg{i}: {e:?}"));
        }
    }

    assert!(
        bg_send_errors.is_empty(),
        "BackgroundIngestor send errors: {:?}",
        bg_send_errors
    );

    // ── Run 5 inline ingest calls via EngineGraphHandle concurrently ──────────
    // These happen while BackgroundIngestor's OS thread is also writing.
    // run_in_background=false triggers the inline path — a write path that
    // calls engine.ingest() on the tokio runtime, i.e., same DB as the bg worker.
    let provider: Arc<dyn ChatProvider> = Arc::new(EmptyArrayLlmClient);
    let mut inline_results = Vec::new();

    for i in 0..5_usize {
        let h = Arc::clone(&engine_handle);
        let p = Arc::clone(&provider);
        let source_ref = SourceRef {
            kind: SourceKind::Episode,
            id: format!("spike-c-inline-{i}"),
            occurred_at: Utc::now(),
            published_at: None,
        };
        let ns = Namespace::new("spike-c-group");
        let content = format!("Spike C inline episode {i}: concurrent with BackgroundIngestor");

        // Spawn each inline call as a tokio task so they can overlap with the
        // background OS-thread writes.
        let task = tokio::spawn(async move {
            h.graph_ingest_episode(GraphIngestEpisodeParams {
                namespace: &ns,
                source_ref: &source_ref,
                content: &content,
                structured_facts: &[],
                provider: p,
                batch_id: None,
                opts: SubmitOpts {
                    enrich_per_episode: false, // skip LLM — just Phase 1
                    run_in_background: false,  // inline path
                },
                sink: None,
            })
            .await
        });
        inline_results.push(task);
    }

    // Collect all inline results
    let mut inline_errors: Vec<String> = vec![];
    for (i, task) in inline_results.into_iter().enumerate() {
        match task.await {
            Ok(Ok(_commit)) => {}
            Ok(Err(e)) => inline_errors.push(format!("inline{i}: ingest error: {e:?}")),
            Err(e) => inline_errors.push(format!("inline{i}: task panic: {e:?}")),
        }
    }

    assert!(
        inline_errors.is_empty(),
        "EngineGraphHandle inline ingest errors (concurrent with BackgroundIngestor): {:?}",
        inline_errors
    );

    // ── Drain BackgroundIngestor: drop handle then shutdown ───────────────────
    drop(ingestor);
    // Drop engine_handle to release Engine B connection
    drop(engine_handle);

    // Join the BackgroundIngestor worker via spawn_blocking.
    // If the worker panicked (WAL lock error), join() returns Err and
    // spawn_blocking propagates it as JoinError.
    let shutdown_result = tokio::task::spawn_blocking(move || {
        guard.shutdown();
    })
    .await;

    shutdown_result.expect(
        "BackgroundIngestor worker must complete without panic (no WAL lock errors \
         from concurrent EngineGraphHandle writes)",
    );

    // ── Verify DB state via a fresh read-only connection ──────────────────────
    // Both paths are fully stopped. Open a third TemporalGraph to inspect final state.
    let verification_graph = Arc::new(
        TemporalGraph::open(&db_path)
            .await
            .expect("verification TemporalGraph::open"),
    );

    let conn = verification_graph.conn_for_test();

    // Count all episodes regardless of group_id.
    let mut count_rows = conn
        .query("SELECT COUNT(*) FROM episodes", libsql::params![])
        .await
        .expect("SELECT COUNT(*) FROM episodes");
    let count_row = count_rows
        .next()
        .await
        .expect("count row iteration")
        .expect("count row present");
    let episode_count: i64 = count_row.get(0).expect("count column");
    let episode_count = episode_count as usize;

    // Assert all 10 episodes persisted: 5 via BackgroundIngestor + 5 via inline.
    // < 10 means WAL conflict caused silent drops or a duplicate-PK rollback.
    assert_eq!(
        episode_count, 10,
        "Expected exactly 10 episodes (5 background + 5 inline); got {episode_count}. \
         If <10: WAL conflict caused silent drops or duplicate-PK rollback. \
         Two-Engine concurrent write (BackgroundIngestor + EngineGraphHandle) is NOT safe."
    );

    // ── Secondary: verify unique episode IDs ─────────────────────────────────
    let mut rows = conn
        .query("SELECT id FROM episodes ORDER BY id", libsql::params![])
        .await
        .expect("SELECT id FROM episodes");

    let mut episode_ids: Vec<i64> = vec![];
    while let Some(row) = rows.next().await.expect("row iteration") {
        let id: i64 = row.get(0).expect("id column");
        episode_ids.push(id);
    }

    let has_duplicates = episode_ids.windows(2).any(|w| w[0] == w[1]);
    assert!(
        !has_duplicates,
        "Episode IDs must be unique (no autoincrement collision between paths). \
         IDs: {episode_ids:?}"
    );

    assert_eq!(
        episode_ids.len(),
        10,
        "Expected 10 unique episode IDs; got {} IDs: {episode_ids:?}",
        episode_ids.len()
    );

    // ── PASS ──────────────────────────────────────────────────────────────────
    println!(
        "SPIKE C PASS: BackgroundIngestor (OS-thread) + EngineGraphHandle (tokio inline) \
         safely wrote 10 episodes concurrently to the same libSQL WAL DB. \
         No lock errors, no ID collisions. Episode IDs: {episode_ids:?}"
    );
}
