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
