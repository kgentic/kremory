//! A.8a — Memory builder open/construction tests.
//!
//! Verifies the type-state builder pipeline compiles and returns `Memory`
//! without calling into the graph substrate.

use kremory::{DynEmbeddingProvider, Memory, MemoryBuilder, Namespace, NoEmb, NoLlm, WithLlm};
use std::sync::Arc;

/// Helper: build a Memory backed by Ollama defaults + real EngineGraphHandle.
async fn open_test_memory() -> Memory {
    Memory::with_ollama("/tmp/kremory-test.db")
        .await
        .expect("Memory::with_ollama should succeed in test-utils builds")
}

/// Builder type-state: `Memory::open` returns `MemoryBuilder<NoLlm, NoEmb>`.
/// This test asserts the type-state signature by binding to that type.
#[test]
fn builder_initial_state_is_nollm_noemb() {
    let _b: MemoryBuilder<NoLlm, NoEmb> = Memory::open("/tmp/test.db");
}

/// After `.with_llm()`, state advances to `MemoryBuilder<WithLlm, NoEmb>`.
#[test]
fn builder_after_with_llm_is_withlm_noemb() {
    // The type assertion happens at compile time; runtime just verifies no panic.
    let b0: MemoryBuilder<NoLlm, NoEmb> = Memory::open("/tmp/test.db");
    let _b1: MemoryBuilder<WithLlm, NoEmb> = b0.with_llm(make_null_llm());
}

/// Full builder pipeline: NoLlm → WithLlm → WithEmb → Memory.
#[tokio::test]
async fn builder_full_pipeline_returns_memory() {
    let _mem: Memory = Memory::open("/tmp/test.db")
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .await
        .expect("builder pipeline should succeed in test-utils builds");
}

/// `Memory::with_ollama` Tier 1 shortcut compiles and succeeds in test mode.
#[tokio::test]
async fn tier1_with_ollama_succeeds_in_test_mode() {
    let _mem = open_test_memory().await;
}

/// `Memory` is Clone (cheap Arc clone).
#[tokio::test]
async fn memory_is_clone() {
    let mem = open_test_memory().await;
    let _clone = mem.clone();
}

/// default_namespace is preserved on the builder and used by ops.
/// Verified indirectly: `ForgetRequest` with a default_namespace does not
/// return `MissingNamespace`.
#[tokio::test]
async fn builder_default_namespace_flows_to_forget() {
    let mem = Memory::open("/tmp/test.db")
        .with_llm(make_null_llm())
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("acme"))
        .await
        .expect("builder should succeed");

    // ForgetRequest stub returns Ok(0) — no graph call — namespace present = no error.
    let count = mem.forget().execute().await.expect("forget should succeed");
    assert_eq!(count, 0, "stub forget returns 0 deleted");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}
