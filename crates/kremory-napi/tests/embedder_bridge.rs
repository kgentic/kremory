//! BYOM embedder bridge tests live in `src/bridge.rs` under `#[cfg(test)]` (unit
//! tests, run via `cargo test -p kremory-napi --lib`).
//!
//! Integration tests for napi crates must NOT import `kremory_napi` directly —
//! the compiled rlib includes `#[napi]`-generated code that references `_napi_*`
//! symbols only available when the .node binary is loaded by a Node.js runtime.
//! Attempting to link those symbols in a standalone test binary produces
//! `ld: symbol(s) not found` linker errors.
//!
//! This file is kept as a placeholder to document that pattern. It contains one
//! structural test that uses no napi symbols and serves as a compile-time
//! sanity check that `kremory::DynEmbeddingProvider` and `kremory::CoreError`
//! are reachable from external crates (as a consumer would use them).
//!
//! The 10 behavioural dimensions (dispatch, error propagation, dim mismatch,
//! correct dim, call counting, lifecycle, concurrency, token counter, multiple
//! calls, into_arc) are fully covered in `src/bridge.rs::tests`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use kremory::{CoreError, DynEmbeddingProvider};

/// A minimal DynEmbeddingProvider impl — confirms the trait is implementable
/// from an external crate without requiring any napi symbols.
struct ExternalMock;

impl DynEmbeddingProvider for ExternalMock {
    fn embed_dyn<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, CoreError>> + Send + 'a>> {
        Box::pin(std::future::ready(Ok(vec![0.5f32; 4])))
    }

    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        None
    }
}

/// Confirms `DynEmbeddingProvider` is object-safe and can be wrapped in
/// `Arc<dyn DynEmbeddingProvider>` from an external crate.
#[tokio::test]
async fn external_dyn_embedding_provider_is_object_safe() {
    let provider: Arc<dyn DynEmbeddingProvider> = Arc::new(ExternalMock);
    let result = provider.embed_dyn("test").await;
    let Ok(vec) = result else {
        panic!("ExternalMock embed_dyn must succeed: {:?}", result.err());
    };
    assert_eq!(vec.len(), 4, "ExternalMock must return 4-dim vec");
    assert!(provider.last_usage_tokens_dyn().is_none());
}
