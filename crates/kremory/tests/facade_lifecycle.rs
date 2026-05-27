//! A.8b — Memory async lifecycle tests: handles, polling, cancel.
//!
//! Verifies `status_of`, `await_enrichment`, `await_batch`, `cancel`,
//! `cancel_dream` signatures + error paths that fire before graph calls.

use chrono::Utc;
use kremory::{DreamHandle, DynEmbeddingProvider, EpisodeCommit, Memory, MemoryError, Namespace};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

async fn open_with_ns() -> Memory {
    Memory::open("/tmp/test.db")
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
    }
}

fn background_commit() -> EpisodeCommit {
    EpisodeCommit {
        run_id: Some(Uuid::new_v4()),
        episode_entity_id: "entity-xyz".into(),
        committed_at: Utc::now(),
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

/// `status_of` with a `run_id` set reaches the graph stub (panics in StubGraphHandle).
/// Verified via `#[should_panic]` — confirms run_id check passes before graph.
#[tokio::test]
#[should_panic(expected = "StubGraphHandle::graph_ingest_status")]
async fn status_of_with_run_id_reaches_graph_stub() {
    let mem = open_with_ns().await;
    let _ = mem.status_of(&background_commit()).await;
}

/// `cancel_dream` with a `DreamHandle` reaches the graph stub.
/// Verified via `#[should_panic]` — confirms routing is wired.
#[tokio::test]
#[should_panic(expected = "StubGraphHandle::graph_cancel")]
async fn cancel_dream_reaches_graph_stub() {
    let mem = open_with_ns().await;
    let _ = mem.cancel_dream(&dream_handle()).await;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}
