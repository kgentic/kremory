#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Spike B: Two-Engine concurrent WAL safety spike
//!
//! Empirically validates the architectural claim in
//! `.ai-docs/specs/v0-2-3-followup-dual-path-consolidation-arch-spec-2026-06-15.md §3.3`:
//!
//! > The builder must construct two separate `Engine` instances — one for
//! > `BackgroundIngestor` (background path) and one for `EngineGraphHandle`
//! > (search/dream/inline path). Both engines open the same libSQL database file;
//! > concurrent reads are safe (libSQL WAL mode); write serialization is enforced
//! > by the fact that the ingest write path only runs on the background OS thread.
//!
//! This spike targets Risk R-03 (arch spec §7):
//!
//! > Two `Engine` instances on same libSQL database cause WAL conflicts or schema
//! > divergence — Probability MED, Impact HIGH
//!
//! Two concrete failure modes tested:
//! 1. "Database is locked" errors from concurrent writers
//! 2. Episode ID sequence collisions (same autoincrement ID assigned twice)
//!
//! ## Concurrency shape
//!
//! This spike tests two `BackgroundIngestor` instances (each with their own Engine)
//! writing concurrently to the same DB. This is the maximum-stress case — the actual
//! production shape is one `BackgroundIngestor` (background writes) + one
//! `EngineGraphHandle` (read-primary, writes only on `run_in_background=false`).
//!
//! If two concurrent BackgroundIngestors on the same DB are safe, the production
//! shape (one writer + one read-primary) is necessarily safe.

use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::{BackgroundIngestor, IngestorConfig};
use kremory::core::config::{ContentType, PipelineConfig};
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::ingest::Engine;
use kremory::core::provider::{ChatProvider, MockChatResponse, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;

// ---------------------------------------------------------------------------
// EmptyArrayLlmClient — same pattern as sink_wiring_integration.rs
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
// Helper: build one Engine + BackgroundIngestor pair on the given DB path
// ---------------------------------------------------------------------------

async fn build_engine_ingestor(
    db_path: &str,
    thread_name: &str,
) -> (BackgroundIngestor, kremory::core::background::IngestGuard) {
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
        config: config,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        thread_name: thread_name.to_string(),
        ..IngestorConfig::default()
    };

    BackgroundIngestor::new(engine, ingestor_config)
}

// ---------------------------------------------------------------------------
// Spike B: two Engine instances, same DB path, concurrent ingest
// ---------------------------------------------------------------------------

/// Spike B: Two Engine instances on the same libSQL DB write concurrently.
///
/// Asserts:
/// 1. Both ingestors succeed without "Database is locked" panics on the workers
/// 2. All 10 episodes are present in the DB after both workers drain (no silent drops)
/// 3. No episode ID collision (unique IDs for all 10 rows)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn spike_b_two_engine_concurrent_wal_safety() {
    // ── Setup: single DB path shared by both engines ──────────────────────────
    let tmp_dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let db_path = tmp_dir
        .join(format!("kremory-spike-b-two-engine-{nanos}.db"))
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // ── Construct two Engine+BackgroundIngestor pairs on the SAME DB path ─────
    let (ingestor_a, guard_a) = build_engine_ingestor(&db_path, "spike-b-worker-a").await;
    let (ingestor_b, guard_b) = build_engine_ingestor(&db_path, "spike-b-worker-b").await;

    // ── Enqueue 5 episodes on each ingestor ──────────────────────────────────
    // group_id = None means no namespace isolation; all episodes go to the
    // default group. This maximises write contention between the two workers.
    let mut send_errors_a: Vec<String> = vec![];
    for i in 0..5 {
        let result = ingestor_a.send(
            format!("Episode A{i}: concurrent writes to libSQL WAL from two Engine instances"),
            None,                              // reference_time
            Some("spike-b-group".to_string()), // group_id
            Some(ContentType::Text),           // content_type
        );
        if let Err(e) = result {
            send_errors_a.push(format!("A{i}: {e:?}"));
        }
    }

    let mut send_errors_b: Vec<String> = vec![];
    for i in 0..5 {
        let result = ingestor_b.send(
            format!("Episode B{i}: libSQL WAL concurrency under two separate Engine instances"),
            None,                              // reference_time
            Some("spike-b-group".to_string()), // group_id
            Some(ContentType::Text),           // content_type
        );
        if let Err(e) = result {
            send_errors_b.push(format!("B{i}: {e:?}"));
        }
    }

    // Assert: no send errors (channel not full, worker alive at send time)
    assert!(
        send_errors_a.is_empty(),
        "Ingestor A send errors: {:?}",
        send_errors_a
    );
    assert!(
        send_errors_b.is_empty(),
        "Ingestor B send errors: {:?}",
        send_errors_b
    );

    // ── Drain: drop senders, then shutdown both workers ────────────────────────
    // Drop the ingestor clones first — closes the sender side of the work channel.
    drop(ingestor_a);
    drop(ingestor_b);

    // Join both workers concurrently via spawn_blocking.
    // During this join, both OS threads are processing their queued work
    // and writing to the same libSQL DB. WAL conflicts would surface as
    // panics inside the worker thread, causing JoinHandle::join() to return
    // Err. We check this via the spawn_blocking result.
    let (result_a, result_b) = tokio::join!(
        tokio::task::spawn_blocking(move || {
            guard_a.shutdown(); // sets stop flag + join()
        }),
        tokio::task::spawn_blocking(move || {
            guard_b.shutdown(); // sets stop flag + join()
        })
    );

    // If a worker panicked due to WAL lock error, join() returns an Err
    // and spawn_blocking propagates it as JoinError.
    result_a.expect("Worker A must complete without panic (no WAL lock errors)");
    result_b.expect("Worker B must complete without panic (no WAL lock errors)");

    // ── Verify DB state via a fresh read-only connection ─────────────────────
    // Both workers are now fully stopped. Open a third TemporalGraph connection
    // to inspect the final DB state without interference.
    let verification_graph = Arc::new(
        TemporalGraph::open(&db_path)
            .await
            .expect("verification TemporalGraph::open"),
    );

    // Use raw SQL to count ALL episodes in the DB (no group_id filter).
    // The `send()` API passes group_id to the Engine as a group_id for entity/fact
    // association, but the episode row itself is stored with the group_id from the
    // IngestRequest — which may be NULL if the pipeline doesn't translate it.
    // We query all episodes to verify the total count (WAL safety) regardless of
    // group_id assignment.
    let conn = verification_graph.conn_for_test();
    let mut count_rows = conn
        .query("SELECT COUNT(*) FROM episodes", libsql::params![])
        .await
        .expect("SELECT COUNT(*) FROM episodes");
    let count_row = count_rows
        .next()
        .await
        .expect("count row iteration")
        .expect("count row");
    let episode_count: i64 = count_row.get(0).expect("count column");
    let episode_count = episode_count as usize;

    // Assert: all 10 episodes present (5 from A + 5 from B).
    // < 10 means: WAL conflict caused silent drops OR ID collision caused
    // duplicate-PK constraint failure with silent rollback.
    assert_eq!(
        episode_count, 10,
        "Expected exactly 10 episodes (5 from worker A + 5 from worker B); got {episode_count}. \
         If <10: WAL conflict caused silent drops OR episode ID collision caused \
         duplicate-PK constraint with rollback (episode rows stored with group_id=NULL \
         when no namespace is set — queried all episodes regardless of group_id). \
         Two-Engine concurrent write is NOT safe."
    );

    // ── Secondary: query episode IDs directly for uniqueness check ───────────
    let mut rows = conn
        .query("SELECT id FROM episodes ORDER BY id", libsql::params![])
        .await
        .expect("SELECT id FROM episodes");

    let mut episode_ids: Vec<i64> = vec![];
    while let Some(row) = rows.next().await.expect("row iteration") {
        let id: i64 = row.get(0).expect("id column");
        episode_ids.push(id);
    }

    // All IDs must be unique (sorted above, so check for adjacent duplicates)
    let has_duplicates = episode_ids.windows(2).any(|w| w[0] == w[1]);
    assert!(
        !has_duplicates,
        "Episode IDs must be unique (no autoincrement collision). IDs: {episode_ids:?}"
    );

    assert_eq!(
        episode_ids.len(),
        10,
        "Expected 10 unique episode IDs; got {} IDs: {episode_ids:?}",
        episode_ids.len()
    );

    // ── PASS ──────────────────────────────────────────────────────────────────
    println!(
        "SPIKE B PASS: Two Engine instances on same libSQL DB (WAL mode) safely \
         wrote 10 concurrent episodes. No lock errors, no ID collisions. \
         Episode IDs: {episode_ids:?}"
    );
}
