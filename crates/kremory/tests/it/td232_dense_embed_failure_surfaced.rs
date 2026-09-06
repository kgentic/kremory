#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-232 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-232) — proves
//! `EpisodeCommit::dense_embedded` actually surfaces a real embed failure on
//! the real inline ingest path, rather than asserting the field exists in
//! isolation. Before this fix, `maybe_embed_episode` (`core/ingest/mod.rs`)
//! swallowed an embedder error into a `tracing::warn!` + a metrics counter and
//! `remember()` returned `Ok` with no way to tell — this is the regression
//! test for that silent degrade.
//!
//! Three cases: a document under the embedder's (simulated) context window
//! succeeds and reports `Some(true)`; one over it fails and reports
//! `Some(false)` while the ingest as a WHOLE still succeeds (BM25/entities are
//! unaffected — the existing graceful-degrade behaviour is intentionally
//! UNCHANGED, only made observable); the dense arm disabled by config reports
//! `Some(true)` (nothing wrong — a deliberate choice, not a failure).

use kremory::{DynEmbeddingProvider, Memory, Namespace};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td232_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> std::sync::Arc<dyn kremory::memory::ChatProvider> {
    std::sync::Arc::new(kremory::core::provider::MockChatProvider::null())
}

/// Simulates a real embedder's context-window rejection (e.g. nomic-embed-text
/// HTTP 400 on oversized input, per the aidocs-trial fit check's D-2 finding)
/// WITHOUT a real network call — rejects any text over `max_chars`.
struct WindowLimitedEmbeddingProvider {
    dim: usize,
    max_chars: usize,
}

impl kremory::core::provider::EmbeddingProvider for WindowLimitedEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        let over_window = text.len() > self.max_chars;
        async move {
            if over_window {
                Err(kremory::CoreError::Embedding(
                    "simulated context-window rejection (TD-232 test)".to_string(),
                ))
            } else {
                Ok(vec![1.0_f32; dim])
            }
        }
    }
}

#[tokio::test]
async fn short_document_reports_dense_embedded_true() {
    let embedder: std::sync::Arc<dyn DynEmbeddingProvider> =
        std::sync::Arc::new(WindowLimitedEmbeddingProvider {
            dim: 16,
            max_chars: 100,
        });

    let mem = Memory::open(unique_db("short_ok"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .embedding_dim(16)
        .with_episode_dense_enabled(true)
        .default_namespace(Namespace::new("td232-short"))
        .await
        .expect("build memory with dense episode arm on");

    let commit = mem
        .remember("well under the simulated window")
        .skip_extraction()
        .await
        .expect("remember must succeed for a short document");

    assert_eq!(
        commit.dense_embedded,
        Some(true),
        "a document under the embedder's window must report dense_embedded == Some(true)"
    );
}

#[tokio::test]
async fn oversized_document_ingest_still_succeeds_but_reports_dense_embedded_false() {
    let embedder: std::sync::Arc<dyn DynEmbeddingProvider> =
        std::sync::Arc::new(WindowLimitedEmbeddingProvider {
            dim: 16,
            max_chars: 50,
        });

    let mem = Memory::open(unique_db("oversized_fail"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .embedding_dim(16)
        .with_episode_dense_enabled(true)
        .default_namespace(Namespace::new("td232-oversized"))
        .await
        .expect("build memory with dense episode arm on");

    let long_text = "x".repeat(500); // well over max_chars=50
    let commit = mem.remember(&long_text).skip_extraction().await.expect(
        "TD-232: ingest must NOT abort just because the dense arm failed \
             — BM25/entities are unaffected by an embed failure",
    );

    assert_eq!(
        commit.dense_embedded,
        Some(false),
        "TD-232 regression: an episode over the embedder's context window must \
         report dense_embedded == Some(false), not silently succeed with no signal"
    );
}

#[tokio::test]
async fn dense_arm_disabled_by_config_reports_dense_embedded_true() {
    // Would fail if attempted (max_chars=0), proving this path returns
    // Some(true) because the embed was never ATTEMPTED — not because it
    // silently succeeded.
    let embedder: std::sync::Arc<dyn DynEmbeddingProvider> =
        std::sync::Arc::new(WindowLimitedEmbeddingProvider {
            dim: 16,
            max_chars: 0,
        });

    let mem = Memory::open(unique_db("disabled_by_config"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .embedding_dim(16)
        .with_episode_dense_enabled(false) // deliberate opt-out
        .default_namespace(Namespace::new("td232-disabled"))
        .await
        .expect("build memory with dense episode arm explicitly OFF");

    let commit = mem
        .remember("irrelevant — the embedder is never called")
        .skip_extraction()
        .await
        .expect("remember must succeed when the dense arm is disabled by config");

    assert_eq!(
        commit.dense_embedded,
        Some(true),
        "a deliberately-disabled dense arm must report Some(true) — \
         'nothing went wrong', not 'unknown' and not 'failed'"
    );
}
