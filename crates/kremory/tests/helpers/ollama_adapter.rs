#![allow(clippy::unwrap_used, clippy::expect_used)]
//! `OllamaEmbedderAdapter` — bridges the `autoagents_llm::embedding::EmbeddingProvider`
//! trait (batch `Vec<String>` → `Vec<Vec<f32>>`) to kremory's single-string
//! `EmbeddingProvider` trait (`&str` → `Vec<f32>`).
//!
//! Used only in `tests/llm_integration.rs` under `#[cfg(feature = "llm-integration")]`.

use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::embedding::EmbeddingProvider as AlLmEmbeddingProvider;
use kremory::CoreError;
use kremory::EmbeddingProvider;

/// Newtype wrapping an `Arc<Ollama>` to implement kremory's `EmbeddingProvider`.
///
/// Used by `tests/llm_integration.rs` (requires `llm-integration` feature).
/// `dead_code` is suppressed here because the struct is only referenced by
/// test binaries that opt in to Ollama integration — cargo does not see those
/// usages when compiling test binaries that include this helper module without
/// the feature flag.
///
/// `#[allow(dead_code)]` (not `#[expect]`) is the right tool: with feature
/// gating, the struct IS used in some compilations (llm-integration feature
/// enabled) and not in others; `#[expect]` would fire `unfulfilled-lint-
/// expectations` in the compiles where it IS used.  Per-cfg gating is the
/// documented escape hatch for cross-compile-target test helpers.
/// Quinn Phase 6 review MED Rule-8.
#[allow(dead_code)]
pub struct OllamaEmbedderAdapter(pub Arc<Ollama>);

impl EmbeddingProvider for OllamaEmbedderAdapter {
    async fn embed(&self, text: &str) -> kremory::CoreResult<Vec<f32>> {
        let mut vecs = AlLmEmbeddingProvider::embed(&*self.0, vec![text.to_string()])
            .await
            .map_err(|e| CoreError::Embedding(e.to_string()))?;

        vecs.pop().ok_or_else(|| {
            CoreError::Embedding("OllamaEmbedderAdapter: embed returned empty vec".to_string())
        })
    }
}
