#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Consumer-facing compile-time snippet tests.
//!
//! Tests here MUST:
//!   - Always compile (no `#[ignore]`, no feature gate)
//!   - Never make network calls
//!   - Validate that the public API surface is usable as documented
//!
//! These run in every `cargo test -p kremory` invocation, catching API
//! regressions before they reach users.

use kremory::{DynEmbeddingProvider, Memory, Namespace};
use std::sync::Arc;

// ── Tier 2: always-on compile test ────────────────────────────────────────────

/// Verify that the `Memory::open` → `with_llm` → `with_embedder` builder
/// pipeline compiles with Ollama-compatible types.
///
/// No network connection is made.  The builder opens (or creates) a local
/// libSQL DB at the given path and wires an `EngineGraphHandle`.
#[tokio::test]
async fn memory_open_with_ollama_providers_compiles() -> anyhow::Result<()> {
    // Construct providers through the same path a real consumer would use —
    // but with null/mock implementations so no Ollama instance is required.
    let llm: Arc<dyn kremory::memory::ChatProvider> =
        Arc::new(kremory::core::provider::MockChatProvider::null());

    let emb: Arc<dyn DynEmbeddingProvider> =
        Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 });

    // Full builder pipeline with explicit type annotations matching the
    // documented consumer pattern.
    let mem: Memory = Memory::open("/tmp/kremory-consumer-snippet-test.db")
        .with_llm(llm)
        .with_embedder(emb)
        .default_namespace(Namespace::new("snippets"))
        .await?;

    // Verify the handle is Clone (cheap Arc clone).
    let _clone = mem.clone();

    Ok(())
}
