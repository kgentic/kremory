#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143) — call-site
//! level proof that `search_document: ` / `search_query: ` land on the RIGHT
//! side (write vs read) of the real ingest/recall paths when
//! `embed_task_prefix_enabled` is on, via a recording embedder that captures
//! the ACTUAL string handed to `.embed()` — not a hand-shaped model of the
//! call site.
//!
//! Covers:
//! - WRITE side: `core::ingest::Engine::maybe_embed_episode`
//!   (`core/ingest/mod.rs`), reached via `Memory::remember(..).skip_extraction()`.
//! - QUERY side: `facade::recall::fuse_content_stream`'s dense episode + fact
//!   arms (`facade/recall.rs`), AND `core::context::Engine::contextualize`'s
//!   entity-graph arm (`core/context.rs`) — reached via `Memory::recall(..)`.
//! - Asymmetry: the SAME raw query text gets a DIFFERENT prefix than the SAME
//!   raw document text.
//!
//! The pure-function correctness of the prefix strings themselves (default-off
//! passthrough, exact prefix literal, asymmetry) is unit-tested directly in
//! `core::embed_prefix`'s own `#[cfg(test)]` module — this file proves the
//! WIRING (config → call site → embedder) on the real end-to-end path.

use kremory::{DynEmbeddingProvider, Memory, Namespace};
use std::sync::{Arc, Mutex};

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td143_call_sites_{}_{}_{}.db",
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

/// Records every raw string handed to `.embed()` — the load-bearing evidence
/// for "assert the ACTUAL string handed to the embedder" (TD-143 DoD). Always
/// returns a well-formed non-zero vector so callers' zero-magnitude guards
/// (e.g. `block_resolution_candidates`'s ANN-arm skip) never short-circuit.
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

// ── WRITE side: episode ingest embed gets `search_document: ` ───────────────

#[tokio::test]
async fn episode_ingest_embed_gets_search_document_prefix() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(768));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("write_side"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        .with_embed_task_prefix_enabled(true)
        .default_namespace(Namespace::new("td143-write-side"))
        .await
        .expect("build memory with dense episode arm + task prefix on");

    mem.remember("kremory td143 write side probe")
        .skip_extraction()
        .await
        .expect("remember with skip_extraction must persist the episode + embedding");

    let texts = recording.texts();
    assert!(
        texts
            .iter()
            .any(|t| t == "search_document: kremory td143 write side probe"),
        "episode ingest-time embed must be document-prefixed; recorded texts: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("search_query: ")),
        "a WRITE-side call must never receive the query prefix; recorded texts: {texts:?}"
    );
}

/// Default-off byte-identical guard on the SAME real write path — the recorded
/// text equals the raw content, no prefix at all.
#[tokio::test]
async fn episode_ingest_embed_is_unprefixed_when_knob_off() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(768));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("write_side_default"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        // embed_task_prefix_enabled left UNSET (default false).
        .default_namespace(Namespace::new("td143-write-side-default"))
        .await
        .expect("build memory with dense episode arm, task prefix default off");

    mem.remember("kremory td143 default off probe")
        .skip_extraction()
        .await
        .expect("remember with skip_extraction must persist the episode + embedding");

    let texts = recording.texts();
    assert!(
        texts.iter().any(|t| t == "kremory td143 default off probe"),
        "default-off must be byte-identical: bare text, no prefix; recorded texts: {texts:?}"
    );
    assert!(
        texts
            .iter()
            .all(|t| !t.starts_with("search_document: ") && !t.starts_with("search_query: ")),
        "default-off must never emit either prefix; recorded texts: {texts:?}"
    );
}

// ── QUERY side: recall-time embeds get `search_query: ` ─────────────────────

#[tokio::test]
async fn recall_query_embed_gets_search_query_prefix() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(768));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("query_side"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        .with_fact_dense_enabled(true)
        .with_embed_task_prefix_enabled(true)
        .default_namespace(Namespace::new("td143-query-side"))
        .await
        .expect("build memory with dense episode + fact arms + task prefix on");

    // No data need be ingested: every gated dense arm embeds the query BEFORE
    // running the (possibly-empty) vector search — the call-site wiring under
    // test fires regardless of corpus content.
    let _ = mem.recall("kremory td143 query side probe").await;

    let texts = recording.texts();
    assert!(
        texts
            .iter()
            .any(|t| t == "search_query: kremory td143 query side probe"),
        "recall-time embed (entity graph arm, always-on; and/or the dense \
         episode/fact arms) must be query-prefixed; recorded texts: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("search_document: ")),
        "a QUERY-side call must never receive the document prefix; recorded texts: {texts:?}"
    );
}

// ── Asymmetry: the SAME raw text diverges by side, on the REAL path ─────────

#[tokio::test]
async fn write_and_query_sides_diverge_for_the_same_raw_text_on_the_real_path() {
    let recording = Arc::new(RecordingEmbeddingProvider::new(768));
    let embedder_dyn: Arc<dyn DynEmbeddingProvider> = recording.clone();

    let mem = Memory::open(unique_db("asymmetry"))
        .with_llm(null_llm())
        .with_embedder(embedder_dyn)
        .with_episode_dense_enabled(true)
        .with_embed_task_prefix_enabled(true)
        .default_namespace(Namespace::new("td143-asymmetry"))
        .await
        .expect("build memory with dense episode arm + task prefix on");

    let shared_text = "asymmetric probe text";

    mem.remember(shared_text)
        .skip_extraction()
        .await
        .expect("remember with skip_extraction must persist the episode + embedding");
    let _ = mem.recall(shared_text).await;

    let texts = recording.texts();
    assert!(
        texts
            .iter()
            .any(|t| t == &format!("search_document: {shared_text}")),
        "the WRITE of `{shared_text}` must be document-prefixed; recorded texts: {texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t == &format!("search_query: {shared_text}")),
        "the QUERY of `{shared_text}` must be query-prefixed; recorded texts: {texts:?}"
    );
}
