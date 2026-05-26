//! B.1 — Observability RED gate.
//!
//! Verifies that `TokenTrackingEmbedder` exists and emits the canonical
//! `kremory_core_tokens_total` counter after an embed call.
//!
//! Uses `metrics_util::debugging::DebuggingRecorder` with
//! `metrics::with_local_recorder` to capture counter increments without
//! installing a global recorder (library-safe pattern per ADR D2).
//!
//! These tests fail until B.1 GREEN adds:
//! - `kremory::core::embedding` module with `TokenTrackingEmbedder`
//! - `EmbeddingProvider::last_usage_tokens()` default method
//! - Counter emission: `kremory_core_tokens_total{provider, model, operation, direction}`

use kremory::core::embedding::TokenTrackingEmbedder;
use kremory::core::provider::{EmbeddingProvider, NullEmbeddingProvider};
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

/// `TokenTrackingEmbedder` wrapping `NullEmbeddingProvider` emits
/// `kremory_core_tokens_total` counter after a successful embed call.
///
/// Uses `tokio::runtime::Builder::new_current_thread` to pin the async execution
/// to a single thread, keeping `metrics::with_local_recorder`'s thread-local
/// recorder active throughout the async embed call.
#[test]
fn token_tracking_embedder_emits_tokens_total_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter: Snapshotter = recorder.snapshotter();

    let inner = NullEmbeddingProvider { dim: 4 };
    let tracker = TokenTrackingEmbedder::new(inner, "local", "null-4d");

    // Single-threaded runtime keeps the thread-local recorder active.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");

    metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let _vec = tracker
                .embed("hello world")
                .await
                .expect("embed must succeed");
        });
    });

    let snapshot = snapshotter.snapshot();
    let counter_names: Vec<String> = snapshot
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();

    assert!(
        counter_names
            .iter()
            .any(|n| n == "kremory_core_tokens_total"),
        "kremory_core_tokens_total counter must be emitted after embed; got: {:?}",
        counter_names
    );
}

/// `EmbeddingProvider` trait has `last_usage_tokens()` default method returning `None`.
/// Verifies ADR D10 trait extension compiles.
#[tokio::test]
async fn embedding_provider_has_last_usage_tokens_default() {
    let provider = NullEmbeddingProvider { dim: 4 };
    // Default returns None — just verifying the method exists and compiles
    let tokens: Option<u64> = provider.last_usage_tokens();
    assert!(
        tokens.is_none(),
        "NullEmbeddingProvider.last_usage_tokens() must return None"
    );
}

/// `TokenTrackingEmbedder` is Send + Sync — required for Arc<dyn EmbeddingProvider>.
#[test]
fn token_tracking_embedder_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TokenTrackingEmbedder<NullEmbeddingProvider>>();
}
