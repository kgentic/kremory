#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! `Memory::reembed_all_episode_embeddings` — public-API-only companions to
//! the in-crate overwrite proof
//! (`facade::mod::reembed_all_episode_embeddings_tests`, which needs
//! `pub(crate)` `vector_search_episodes` to assert the stored vector actually
//! moved). This file covers the two DoD requirements reachable from the
//! public surface:
//! - prefix parity: with `embed_task_prefix_enabled` on, the text handed to
//!   the embedder on the re-embed path is document-prefixed — reusing the
//!   same recording-embedder pattern as `td143_embed_task_prefix_call_sites.rs`.
//! - idempotency: running it twice against a stable corpus + embedder leaves
//!   a consistent tally with zero failures.

use kremory::{DynEmbeddingProvider, Memory, Namespace};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td143_reembed_all_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

/// Records every raw string handed to `.embed()` — same recording pattern as
/// `td143_embed_task_prefix_call_sites.rs`'s `RecordingEmbeddingProvider`.
/// Always returns a well-formed non-zero vector.
struct RecordingEmbeddingProvider {
    dim: usize,
    texts: Mutex<Vec<String>>,
}

impl RecordingEmbeddingProvider {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            texts: Mutex::new(Vec::new()),
        }
    }

    fn texts(&self) -> Vec<String> {
        self.texts.lock().expect("recording mutex poisoned").clone()
    }
}

impl kremory::core::provider::EmbeddingProvider for RecordingEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        self.texts
            .lock()
            .expect("recording mutex poisoned")
            .push(text.to_string());
        let dim = self.dim;
        async move { Ok(vec![1.0_f32; dim]) }
    }
}

// ── Prefix parity: TD-143 document-prefix applies on the re-embed path ──────

#[tokio::test]
async fn reembed_all_applies_document_prefix_when_enabled() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(384));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("prefix"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        .with_embed_task_prefix_enabled(true)
        .default_namespace(Namespace::new("td143-reembed-prefix"))
        .await
        .expect("build memory with dense episode arm + task prefix on");

    mem.remember("kremory td143 reembed prefix probe")
        .skip_extraction()
        .await
        .expect("remember must persist the episode");

    // Ingest-time embed already happened (and is already prefixed per the
    // sibling call-site test) — clear so this assertion is scoped to the
    // re-embed call specifically.
    recording.texts.lock().expect("mutex poisoned").clear();

    let stats = mem
        .reembed_all_episode_embeddings(256)
        .await
        .expect("reembed_all_episode_embeddings must succeed");
    assert_eq!(stats.embedded, 1);
    assert_eq!(stats.failed, 0);

    let texts = recording.texts();
    assert!(
        texts
            .iter()
            .any(|t| t == "search_document: kremory td143 reembed prefix probe"),
        "reembed_all_episode_embeddings must document-prefix the text it \
         hands to the embedder when embed_task_prefix_enabled is on; \
         recorded texts: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("search_query: ")),
        "a re-embed WRITE call must never receive the query prefix; \
         recorded texts: {texts:?}"
    );
}

// ── Idempotency: running it twice is stable, no failures ────────────────────

#[tokio::test]
async fn reembed_all_is_idempotent_across_two_runs() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(384));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("idempotent"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        .default_namespace(Namespace::new("td143-reembed-idempotent"))
        .await
        .expect("build memory with dense episode arm");

    for i in 0..3 {
        mem.remember(format!("idempotent probe episode {i}"))
            .skip_extraction()
            .await
            .expect("remember must persist each episode");
    }

    let first = mem
        .reembed_all_episode_embeddings(2) // batch_size smaller than corpus — forces multiple pages
        .await
        .expect("first reembed_all_episode_embeddings run must succeed");
    assert_eq!(first.embedded, 3, "all three episodes must be re-embedded");
    assert_eq!(first.failed, 0);

    let second = mem
        .reembed_all_episode_embeddings(2)
        .await
        .expect("second reembed_all_episode_embeddings run must succeed");
    assert_eq!(
        second.embedded, 3,
        "a second run over the SAME corpus must re-embed the SAME three \
         episodes again — this is a full re-embed, not a gap-fill, so the \
         tally must be stable across runs, not zero"
    );
    assert_eq!(second.failed, 0, "no failures on either run");
}
