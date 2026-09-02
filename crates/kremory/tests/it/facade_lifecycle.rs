#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.8b — Memory async lifecycle tests: handles, polling, cancel.
//!
//! Verifies `status_of`, `await_enrichment`, `await_batch`, `cancel`,
//! `cancel_dream` signatures + error paths that fire before graph calls.

use chrono::Utc;
use kremory::{
    DreamHandle, DynEmbeddingProvider, EpisodeCommit, IngestStatus, Memory, MemoryError, Namespace,
};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kremory_facade_lifecycle_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

async fn open_with_ns() -> Memory {
    Memory::open(unique_db_path("open_with_ns"))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("tests"))
        .await
        .expect("builder should succeed")
}

fn inline_commit() -> EpisodeCommit {
    EpisodeCommit {
        run_id: None,
        episode_entity_id: "entity-abc".into(),
        committed_at: Utc::now(),
        stub_entities_inserted: 0,
        dense_embedded: None,
    }
}

fn background_commit() -> EpisodeCommit {
    EpisodeCommit {
        run_id: Some(Uuid::new_v4()),
        episode_entity_id: "entity-xyz".into(),
        committed_at: Utc::now(),
        stub_entities_inserted: 0,
        dense_embedded: None,
    }
}

fn dream_handle() -> DreamHandle {
    DreamHandle {
        run_id: Uuid::new_v4(),
        namespace: Namespace::new("tests"),
        submitted_at: Utc::now(),
        batch_id: None,
    }
}

/// `status_of` with an `EpisodeCommit` that has no `run_id` (inline Phase 2)
/// returns `Err(MemoryError::Other)` before touching the graph.
#[tokio::test]
async fn status_of_no_run_id_errors() {
    let mem = open_with_ns().await;
    let err = mem
        .status_of(&inline_commit())
        .await
        .expect_err("should fail");
    assert!(
        matches!(err, MemoryError::Other(_)),
        "expected Other error for missing run_id, got: {err}"
    );
}

/// `await_enrichment` with no `run_id` errors before graph call.
#[tokio::test]
async fn await_enrichment_no_run_id_errors() {
    let mem = open_with_ns().await;
    let err = mem
        .await_enrichment(&inline_commit(), Duration::from_secs(5))
        .await
        .expect_err("should fail");
    assert!(matches!(err, MemoryError::Other(_)));
}

/// `cancel` with no `run_id` returns `Err(MemoryError::Other)`.
#[tokio::test]
async fn cancel_no_run_id_errors() {
    let mem = open_with_ns().await;
    let err = mem.cancel(&inline_commit()).await.expect_err("should fail");
    assert!(matches!(err, MemoryError::Other(_)));
}

/// `status_of` with a `run_id` set reaches the real `EngineGraphHandle`.
/// Unknown run_id → `Ok(IngestStatus::Complete)` (idempotent — already done or never ran).
#[tokio::test]
async fn status_of_with_run_id_reaches_engine_graph_handle() {
    let mem = open_with_ns().await;
    let result = mem.status_of(&background_commit()).await;
    // EngineGraphHandle returns Complete for unknown run_ids (idempotent).
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
    assert!(
        matches!(result.unwrap(), IngestStatus::Complete),
        "expected Complete for unknown run_id"
    );
}

/// `cancel_dream` with a `DreamHandle` reaches the real `EngineGraphHandle`.
/// Unknown run_id → idempotent Ok (no task to cancel, no panic).
#[tokio::test]
async fn cancel_dream_reaches_engine_graph_handle() {
    let mem = open_with_ns().await;
    let result = mem.cancel_dream(&dream_handle()).await;
    assert!(
        result.is_ok(),
        "cancel_dream with unknown run_id should be idempotent Ok, got: {result:?}"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}
