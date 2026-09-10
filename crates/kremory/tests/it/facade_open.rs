#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A.8a — Memory builder open/construction tests.
//!
//! Verifies the type-state builder pipeline compiles and returns `Memory`
//! without calling into the graph substrate.

use kremory::{DynEmbeddingProvider, Memory, MemoryBuilder, Namespace, NoEmbedder, NoLlm, WithLlm};
use std::sync::Arc;

/// Unique per-call DB path. Tests run in parallel; sharing a single path
/// caused intermittent `database is locked` flakes.
fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_facade_open_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

/// Helper: build a Memory backed by Ollama defaults + real EngineGraphHandle.
async fn open_test_memory(tag: &str) -> Memory {
    Memory::with_ollama(unique_db_path(tag))
        .await
        .expect("Memory::with_ollama should succeed in test-utils builds")
}

/// Builder type-state: `Memory::open` returns `MemoryBuilder<NoLlm, NoEmbedder>`.
/// This test asserts the type-state signature by binding to that type.
#[test]
fn builder_initial_state_is_nollm_noemb() {
    let _b: MemoryBuilder<NoLlm, NoEmbedder> = Memory::open("/tmp/test.db");
}

/// After `.with_llm()`, state advances to `MemoryBuilder<WithLlm, NoEmbedder>`.
#[test]
fn builder_after_with_llm_is_withlm_noemb() {
    // The type assertion happens at compile time; runtime just verifies no panic.
    let b0: MemoryBuilder<NoLlm, NoEmbedder> = Memory::open("/tmp/test.db");
    let _b1: MemoryBuilder<WithLlm, NoEmbedder> = b0.with_llm(make_null_llm());
}

/// Full builder pipeline: NoLlm → WithLlm → WithEmbedder → Memory.
#[tokio::test]
async fn builder_full_pipeline_returns_memory() {
    let _mem: Memory = Memory::open(unique_db_path("full_pipeline"))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder pipeline should succeed in test-utils builds");
}

/// `Memory::with_ollama` Tier 1 shortcut compiles and succeeds in test mode.
#[tokio::test]
async fn tier1_with_ollama_succeeds_in_test_mode() {
    let _mem = open_test_memory("tier1").await;
}

/// `Memory` is Clone (cheap Arc clone).
#[tokio::test]
async fn memory_is_clone() {
    let mem = open_test_memory("is_clone").await;
    let _clone = mem.clone();
}

/// default_namespace is preserved on the builder and used by ops.
/// Verified indirectly: `ForgetRequest` with a default_namespace does not
/// return `MissingNamespace`.
#[tokio::test]
async fn builder_default_namespace_flows_to_forget() {
    let mem = Memory::open(unique_db_path("default_ns_forget"))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("acme"))
        .await
        .expect("builder should succeed");

    // ForgetRequest stub returns Ok(0) — no graph call — namespace present = no error.
    let count = mem.forget().execute().await.expect("forget should succeed");
    assert!(count.is_empty(), "nothing to erase; got {count:?}");
}

/// B1 (public-docs-and-api-surface-audit quality-review) — `Memory` must
/// implement `Debug` (C-DEBUG). Before the fix, `println!("{:?}", mem)` /
/// `dbg!(mem)` did not compile at all for the type consumers interact with
/// most; this test proves it now compiles AND reports sensible,
/// presence-only info for the opaque trait-object fields.
#[tokio::test]
async fn memory_debug_reports_configured_state() {
    let mem = Memory::open(unique_db_path("debug_memory"))
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("debug-ns"))
        .await
        .expect("builder should succeed");

    let rendered = format!("{mem:?}");
    assert!(rendered.contains("Memory"), "got: {rendered}");
    assert!(
        rendered.contains("llm_configured: true"),
        "expected llm presence to be reported; got: {rendered}"
    );
    assert!(
        rendered.contains("debug-ns") || rendered.contains("Namespace"),
        "expected default_namespace to be visible; got: {rendered}"
    );
}

/// B1 — `MemoryBuilder<L, E>` must implement `Debug` too (same guideline,
/// same trait-object-field constraint), for any type-state combination —
/// including the initial `<NoLlm, NoEmbedder>` state before any provider is
/// wired, which is exactly when a confused consumer reaches for `dbg!`.
#[test]
fn memory_builder_debug_reports_unconfigured_state() {
    let b: MemoryBuilder<NoLlm, NoEmbedder> = Memory::open("/tmp/kremory-debug-builder-test.db");
    let rendered = format!("{b:?}");
    assert!(rendered.contains("MemoryBuilder"), "got: {rendered}");
    assert!(
        rendered.contains("llm_configured: false"),
        "expected unconfigured llm to be reported; got: {rendered}"
    );
    assert!(
        rendered.contains("embedder_configured: false"),
        "expected unconfigured embedder to be reported; got: {rendered}"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}
