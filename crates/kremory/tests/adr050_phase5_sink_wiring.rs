#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-050 Phase 5 — sink events for crash-resume.
//!
//! Governing spec: `.ai-docs/specs/v0-2-4-impl-spec-2026-06-16.md` Phase 5.
//! Governing ADR:  ADR-050 — dream-pass crash-safety + idempotency cluster.
//!
//! # Test inventory (2 required per Phase 5 DoD §3)
//!
//! 1. `sink_skipped_idempotent_fires_on_second_run` — run same episode twice
//!    via `run_verify_stage`; `RecordingSink` observes `SkippedIdempotent`
//!    on second run (idempotency HIT path).
//!
//! 2. `sink_worker_resumed_fires_on_checkpoint_boot` — inject a non-null
//!    `op_checkpoints` row; start `BackgroundIngestor`; `RecordingSink` observes
//!    `on_worker_resumed` with correct `op_name`.
//!
//! # Mocking boundary (per testing-policy.md)
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open` + `tempfile::TempDir`
//! - ALWAYS REAL: `stage3_write` SQL, idempotency key writes, op_checkpoints reads
//! - ALWAYS MOCK: `EntityExtractorDyn` (inline structs below)
//! - ALWAYS MOCK: `ChatProvider` (`EmptyArrayLlmClient`)
//! - ALWAYS MOCK: `EmbeddingProvider` (`NullEmbeddingProvider`)
//! - RECORDING: `RecordingSink` from `tests/helpers/recording_sink.rs`

mod helpers;

use helpers::recording_sink::{RecordingSink, SinkEvent};

use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::background::verify_stage::run_verify_stage;
use kremory::core::background::{BackgroundIngestor, DeferredRequest, IngestorConfig};
use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::DEFAULT_ENTITY_TYPES;
use kremory::core::ingest::Engine;
use kremory::core::intelligence::{
    EntityExtractorDyn, ExtractedEntity, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::{MockChatResponse, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;

// ---------------------------------------------------------------------------
// EmptyArrayLlmClient
// ---------------------------------------------------------------------------

/// Returns `"[]"` for every call — zero entities/facts; sufficient for
/// `sink_worker_resumed` which only needs the worker_loop to boot.
#[derive(Debug, Clone)]
struct EmptyArrayLlmClient;

use kremory::core::provider::ChatProvider;

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
// MockExtractorReturnsEntities (local copy — same pattern as verify_stage_integration.rs)
// ---------------------------------------------------------------------------

/// Mock extractor returning a fixed set of entity candidates.
struct MockExtractorReturnsEntities {
    entities: Vec<ExtractedEntity>,
}

impl EntityExtractorDyn for MockExtractorReturnsEntities {
    fn name(&self) -> &'static str {
        "mock-phase5-entities"
    }

    fn extract_dyn<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = kremory::core::error::Result<ExtractionResult>>
                + Send
                + 'a,
        >,
    > {
        let entities = self.entities.clone();
        Box::pin(async move {
            Ok(ExtractionResult {
                entities,
                facts: Vec::new(),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn open_graph(tag: &str) -> (Arc<TemporalGraph>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("adr050-phase5-{tag}.db"));
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("TemporalGraph::open");
    (Arc::new(graph), tmp)
}

async fn seed_entity_types(conn: &libsql::Connection) {
    for (id, name, desc) in [
        (0i64, "Entity", "Catch-all"),
        (1i64, "Person", "A human individual"),
        (2i64, "Organization", "A company or group"),
    ] {
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (id, group_id, name, description) \
             VALUES (?1, 'default', ?2, ?3)",
            libsql::params![id, name.to_string(), desc.to_string()],
        )
        .await
        .expect("seed entity_types");
    }
}

async fn insert_pending_episode(conn: &libsql::Connection, content: &str) -> i64 {
    conn.execute(
        "INSERT INTO episodes (content, timestamp, episode_processing_status) \
         VALUES (?1, datetime('now'), 'Pending')",
        libsql::params![content],
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

/// Drain a `BackgroundIngestor` cleanly: drop sender → sleep → shutdown guard.
///
/// The 300 ms sleep mirrors the `sink_wiring_integration.rs` L3 pattern:
/// gives the background worker time to drain the deferred queue before the
/// stop flag is set by `guard.shutdown()`.
async fn drain_ingestor(
    ingestor: BackgroundIngestor,
    guard: kremory::core::background::IngestGuard,
) {
    drop(ingestor);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    tokio::task::spawn_blocking(move || guard.shutdown())
        .await
        .expect("guard.shutdown() panicked");
}

// ---------------------------------------------------------------------------
// Test 1: sink_skipped_idempotent_fires_on_second_run
// ---------------------------------------------------------------------------

/// Runs the same episode through `run_verify_stage` twice.
///
/// First run: entity is new → written to DB + idempotency key written → `RecordingSink`
/// records `StageChange(Extracting)`, `EntityExtracted`, `StageChange(Verified)`.
///
/// Second run: idempotency HIT → `stage3_write` fires `on_stage_change(SkippedIdempotent)`.
/// `RecordingSink` must contain at least one `SkippedIdempotent` event and the
/// `skipped_idempotent_count()` helper must return ≥ 1.
///
/// ADR-050 Phase 5, arch spec §3.1.1.
#[tokio::test]
async fn sink_skipped_idempotent_fires_on_second_run() {
    let (graph, _tmp) = open_graph("skipped-idem").await;
    seed_entity_types(&graph.conn).await;

    let episode_id = insert_pending_episode(&graph.conn, "Alice works at Acme Corp.").await;

    let extractor = MockExtractorReturnsEntities {
        entities: vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::json!({ "confidence": 0.9 }),
        }],
    };

    let sink = RecordingSink::new();

    // ── First run: MISS path — entity written + idempotency key written ────────
    let first_result = run_verify_stage(
        &DeferredRequest {
            text: "Alice works at Acme Corp.".to_string(),
            reference_time: None,
            group_id: None,
            content_type: None,
            episode_id,
            ner_entity_names: vec!["Alice".to_string()],
            batch_id: None,
        },
        &extractor,
        None,
        &graph,
        Some(&sink),
    )
    .await;
    assert!(
        first_result.is_ok(),
        "first run must succeed; got: {:?}",
        first_result.err()
    );

    // Snapshot after first run — must NOT contain SkippedIdempotent yet.
    let first_events = sink.snapshot();
    assert!(
        !first_events
            .iter()
            .any(|e| matches!(e, SinkEvent::SkippedIdempotent)),
        "SkippedIdempotent must NOT fire on first run (MISS path); events: {first_events:?}"
    );

    // ── Second run: same content → HIT path → SkippedIdempotent fires ─────────
    //
    // Episode is already in 'Verified' state from the first run.
    // `run_verify_stage` updates Pending→Extracting; the episode_processing_status
    // UPDATE from 'Verified' back to 'Extracting' may warn but must not block.
    // The critical assertion is that SkippedIdempotent fires for the "Alice" entity.
    //
    // We insert a fresh Pending episode (same content) so the status transition
    // succeeds cleanly — idempotency HIT is on the entity, not the episode.
    let episode_id_2 = insert_pending_episode(&graph.conn, "Alice works at Acme Corp.").await;

    let second_result = run_verify_stage(
        &DeferredRequest {
            text: "Alice works at Acme Corp.".to_string(),
            reference_time: None,
            group_id: None,
            content_type: None,
            episode_id: episode_id_2,
            ner_entity_names: vec!["Alice".to_string()],
            batch_id: None,
        },
        &extractor,
        None,
        &graph,
        Some(&sink),
    )
    .await;
    assert!(
        second_result.is_ok(),
        "second run must succeed (idempotency is not an error); got: {:?}",
        second_result.err()
    );

    // ── Assert SkippedIdempotent fired ────────────────────────────────────────
    let skipped_count = sink.skipped_idempotent_count();
    assert!(
        skipped_count >= 1,
        "RecordingSink must record ≥1 SkippedIdempotent event on second run; \
         got {skipped_count}. Full events: {:?}",
        sink.snapshot()
    );
}

// ---------------------------------------------------------------------------
// Test 2: sink_worker_resumed_fires_on_checkpoint_boot
// ---------------------------------------------------------------------------

/// Pre-seeds an `op_checkpoints` row then starts `BackgroundIngestor`.
///
/// On boot, `worker_loop` queries `op_checkpoints` for `op_name='verify_stage'`.
/// A non-null cursor → fires `on_worker_resumed(cursor, "verify_stage")` via the sink.
///
/// `RecordingSink.worker_resumed_events()` must contain exactly one entry with
/// `op_name = "verify_stage"` and `from_cursor = "42"`.
///
/// ADR-050 Phase 5, arch spec §3.1.2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_worker_resumed_fires_on_checkpoint_boot() {
    let (temporal, _tmp) = open_graph("worker-resumed").await;
    let conn = &temporal.conn;

    // Pre-seed op_checkpoints with a non-null cursor — simulates a prior crash.
    let now_epoch = chrono::Utc::now().timestamp();
    conn.execute(
        "INSERT OR REPLACE INTO op_checkpoints \
         (op_name, op_run_id, cursor, updated_at) \
         VALUES ('verify_stage', 'test_run_001', '42', ?1)",
        libsql::params![now_epoch],
    )
    .await
    .expect("pre-seed op_checkpoints must succeed");

    let sink = RecordingSink::new();

    let config = PipelineConfig::builder()
        .allowed_entity_types(
            DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(_, name, _)| name.to_string())
                .collect(),
        )
        .build()
        .expect("PipelineConfig build");

    let null_emb: Arc<NullEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
    let engine = Engine::new(kremory::core::ingest::EngineNewParams {
        graph: Arc::clone(&temporal),
        llm: Arc::new(EmptyArrayLlmClient),
        embedder: null_emb,
        config: config,
    });

    let ingestor_config = IngestorConfig {
        deferred_extraction_enabled: true,
        ..IngestorConfig::default()
    }
    .with_sink(sink.clone());

    let (ingestor, guard) = BackgroundIngestor::new(engine, ingestor_config);

    // Allow worker_loop to boot and read op_checkpoints before we drain.
    // No episode sent — we only need the boot path to fire.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    drain_ingestor(ingestor, guard).await;

    // ── Assert on_worker_resumed fired ────────────────────────────────────────
    let resumed_events = sink.worker_resumed_events();
    assert_eq!(
        resumed_events.len(),
        1,
        "RecordingSink must record exactly 1 WorkerResumed event; \
         got {}. Full events: {:?}",
        resumed_events.len(),
        sink.snapshot()
    );

    let (from_cursor, op_name) = &resumed_events[0];
    assert_eq!(
        from_cursor, "42",
        "WorkerResumed from_cursor must match seeded cursor; got: {from_cursor:?}"
    );
    assert_eq!(
        op_name, "verify_stage",
        "WorkerResumed op_name must be 'verify_stage'; got: {op_name:?}"
    );

    // Confirm no SkippedIdempotent — no episodes were processed.
    let skipped = sink.skipped_idempotent_count();
    assert_eq!(
        skipped, 0,
        "No SkippedIdempotent expected (no episodes sent); got {skipped}"
    );
}
