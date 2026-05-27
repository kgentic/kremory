// File: crates/kremory/src/memory/llm_adapters.rs
//
// Bridges autoagents-llm batch embedder to kremory's single-text EmbeddingProvider.
//
// This adapter belongs in `kremory::memory` (the facade layer), not in `kremory::core`:
// it depends on `autoagents-llm` types, which would introduce an upward dependency
// into the atomic-design topology (ADR-007 + ADR-005 §9) if placed in core.

use std::sync::Arc;

use crate::core::error::Result;
use crate::core::provider::EmbeddingProvider;

/// Bridges autoagents-llm batch embedder to kremory's single-text `EmbeddingProvider`.
///
/// `autoagents_llm::embedding::EmbeddingProvider` is batch-oriented:
/// `embed(Vec<String>) -> Result<Vec<Vec<f32>>>`. kremory's `Engine` expects a
/// single-text `EmbeddingProvider`. This adapter bridges the two by submitting
/// a one-element batch and extracting the first result.
pub struct AutoagentsEmbedderAdapter<P> {
    inner: Arc<P>,
}

impl<P> AutoagentsEmbedderAdapter<P> {
    /// Wrap a batch embedding provider in the single-text adapter.
    pub fn new(inner: P) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<P> EmbeddingProvider for AutoagentsEmbedderAdapter<P>
where
    P: autoagents_llm::embedding::EmbeddingProvider + Send + Sync + 'static,
{
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send + 'a {
        let inner = Arc::clone(&self.inner);
        let owned = text.to_owned();
        async move {
            let mut batch = inner
                .embed(vec![owned])
                .await
                .map_err(|e| crate::core::error::Error::Embedding(e.to_string()))?;
            batch.pop().ok_or_else(|| {
                crate::core::error::Error::Embedding("empty batch result".into())
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::error::LLMError;

    struct FixedBatchEmbedder {
        dim: usize,
    }

    #[async_trait::async_trait]
    impl autoagents_llm::embedding::EmbeddingProvider for FixedBatchEmbedder {
        async fn embed(&self, input: Vec<String>) -> std::result::Result<Vec<Vec<f32>>, LLMError> {
            Ok(input
                .iter()
                .map(|_| vec![0.5_f32; self.dim])
                .collect())
        }
    }

    #[tokio::test]
    async fn adapter_returns_correct_dimension() {
        let adapter = AutoagentsEmbedderAdapter::new(FixedBatchEmbedder { dim: 16 });
        let result = adapter.embed("hello").await.expect("embed should succeed");
        assert_eq!(result.len(), 16, "dimension must match the inner provider's output");
    }

    #[tokio::test]
    async fn adapter_is_deterministic() {
        let adapter = AutoagentsEmbedderAdapter::new(FixedBatchEmbedder { dim: 8 });
        let a = adapter.embed("test input").await.expect("first embed");
        let b = adapter.embed("test input").await.expect("second embed");
        assert_eq!(a, b, "same input must produce the same output");
    }
}
