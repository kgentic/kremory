#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-232 / TD-234 — proves `kremory::split_for_embedding` actually closes the
//! loop it exists for: a document that silently loses its dense arm when
//! `remember()`d whole (TD-232's regression) must NOT lose it once the caller
//! pre-splits with this helper first.
//!
//! Sibling to `td232_dense_embed_failure_surfaced.rs`, which pins the failure
//! this file pins the fix for — same `WindowLimitedEmbeddingProvider`
//! simulated-window fixture (duplicated locally per this codebase's own
//! per-file-fixture convention, not shared, to keep each test file's failure
//! mode legible on its own).

use kremory::{split_for_embedding, DynEmbeddingProvider, Memory, Namespace};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td234_{}_{}_{}.db",
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

/// Simulates a real embedder's context-window rejection, matching
/// `td232_dense_embed_failure_surfaced.rs`'s fixture exactly.
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
                    "simulated context-window rejection (TD-234 test)".to_string(),
                ))
            } else {
                Ok(vec![1.0_f32; dim])
            }
        }
    }
}

/// Regression pin — unchanged from TD-232: the whole-document path still
/// fails on an over-window document. If this ever stops failing, the
/// simulated fixture (or the underlying bug) has drifted and the "closes the
/// loop" claim below would no longer mean anything.
#[tokio::test]
async fn baseline_whole_document_over_window_still_reports_dense_embedded_false() {
    let embedder: std::sync::Arc<dyn DynEmbeddingProvider> =
        std::sync::Arc::new(WindowLimitedEmbeddingProvider {
            dim: 16,
            max_chars: 50,
        });
    let mem = Memory::open(unique_db("baseline_whole"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .embedding_dim(16)
        .with_episode_dense_enabled(true)
        .default_namespace(Namespace::new("td234-baseline"))
        .await
        .expect("build memory with dense episode arm on");

    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(20); // ~940 chars
    let commit = mem
        .remember(&long_text)
        .skip_extraction()
        .await
        .expect("remember must succeed even when the dense arm fails");

    assert_eq!(
        commit.dense_embedded,
        Some(false),
        "baseline must still fail whole — this is the exact condition split_for_embedding exists to route around"
    );
}

/// The actual proof — pre-split the SAME oversized text with
/// `split_for_embedding`, loop `remember()` per chunk, and confirm EVERY
/// chunk reports `dense_embedded == Some(true)`. This is the real, live
/// ingest path a consumer would run, not a unit test of the splitter alone.
#[tokio::test]
async fn pre_split_document_reports_dense_embedded_true_for_every_chunk() {
    let embedder: std::sync::Arc<dyn DynEmbeddingProvider> =
        std::sync::Arc::new(WindowLimitedEmbeddingProvider {
            dim: 16,
            max_chars: 50,
        });
    let mem = Memory::open(unique_db("presplit"))
        .with_llm(null_llm())
        .with_embedder(embedder)
        .embedding_dim(16)
        .with_episode_dense_enabled(true)
        .default_namespace(Namespace::new("td234-presplit"))
        .await
        .expect("build memory with dense episode arm on");

    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(20); // ~940 chars
    let chunks = split_for_embedding(&long_text, 50);

    assert!(
        chunks.len() > 1,
        "the fixture text must actually need splitting for this test to prove anything; got {} chunk(s)",
        chunks.len()
    );

    let mut episode_ids = Vec::new();
    for chunk in &chunks {
        let commit = mem
            .remember(chunk)
            .skip_extraction()
            .await
            .expect("remember must succeed for a pre-split chunk");
        assert_eq!(
            commit.dense_embedded,
            Some(true),
            "TD-234: a pre-split chunk within the embedder's window must embed \
             successfully; got a failure for chunk {chunk:?}"
        );
        episode_ids.push(commit.episode_entity_id);
    }

    // Sanity: every chunk really did land as its own episode (not silently
    // merged or dropped) — otherwise "every chunk succeeded" could be
    // vacuously true over a shrunk set.
    let unique: std::collections::HashSet<_> = episode_ids.iter().collect();
    assert_eq!(
        unique.len(),
        chunks.len(),
        "each chunk must persist as its own distinct episode; got {} unique ids for {} chunks",
        unique.len(),
        chunks.len()
    );
}
