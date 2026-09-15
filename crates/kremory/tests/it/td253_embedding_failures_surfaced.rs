#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-253 — `remember()` must surface an embedder failure on `EpisodeCommit`
//! instead of silently persisting a NULL-embedding row and returning `Ok`.
//!
//! Uses a pure-Rust always-erroring embedder (mirrors the `AlwaysErrEmbedder`
//! control the Python binding spike used to prove the underlying bug is
//! language-independent) — no mock LLM extraction quirks involved, just the
//! embedder failing every call.
//!
//! Exercises the `.with_facts()` / pinned-entity swallow site
//! (`core/ingest/pipeline/pre_pinned.rs`'s `make_pinned_entity_recallable`,
//! threaded through `write_pre_pinned_facts`) — the path that needs no LLM
//! extraction to trigger deterministically. See the ADR "TD-253 — surface
//! embedder failures via EpisodeCommit.embedding_failures" for the other two
//! synchronous sites (fact-triple embedding, forward-reference stub entities)
//! and why the separate `BackgroundIngestor` path cannot be covered here.

use std::sync::Arc;

use kremory::core::error::Result as CoreResult;
use kremory::core::provider::EmbeddingProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

/// Always fails — the pure-Rust control for "the embedder is down/erroring".
struct AlwaysErrEmbedder;

impl EmbeddingProvider for AlwaysErrEmbedder {
    async fn embed(&self, _text: &str) -> CoreResult<Vec<f32>> {
        Err(kremory::core::error::Error::Other(anyhow::anyhow!(
            "AlwaysErrEmbedder: simulated embedder outage"
        )))
    }
}

fn always_err_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(AlwaysErrEmbedder)
}

#[tokio::test]
async fn remember_surfaces_pinned_entity_embedding_failure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("td253");

    let mem = Memory::open(dir.path().join("t.db"))
        .default_namespace(ns.clone())
        .with_llm(null_llm())
        .with_embedder(always_err_embedder())
        .await
        .expect("open");

    let commit = mem
        .remember("Priya works at Northwind.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "priya".into(),
            predicate: "works_at".into(),
            object: "Northwind".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await
        .expect(
            "remember() must still return Ok — TD-253 is about VISIBILITY, \
             not making embed failures fatal (the dense_embedded precedent)",
        );

    // THE POINT: the failure is no longer invisible. Before this fix, `commit`
    // carried no signal at all that the embedder errored.
    //
    // Only the SUBJECT is a pinned entity here, not the object: the public
    // `StructuredFact` → `PrePinnedFact` translation
    // (`memory/engine_handle.rs`) always sends `sf.object` to `object_value`
    // (a literal), never `object_id` — so `write_pre_pinned_facts`'s
    // object-pin branch (which would ALSO call `make_pinned_entity_recallable`)
    // is unreachable from this public API. That is a separate, pre-existing
    // fact about the surface, not something this test asserts a view on.
    assert_eq!(
        commit.embedding_failures,
        vec!["priya".to_string()],
        "the pinned subject's failure must be named exactly once: {:?}",
        commit.embedding_failures
    );

    // NOTE: this test does not additionally assert "still recallable" via
    // `recall()` — with `AlwaysErrEmbedder` wired as the ONLY embedder,
    // `recall()`'s own query-vector embed call fails too, which would test
    // the embedder double, not the ingest write. The write-not-aborted claim
    // (the `dense_embedded`-style precedent this ADR extends) is already
    // proven above: `remember()` returned `Ok` at all despite the embed
    // failure, and `inserted_fact_ids`/entity persistence are covered by the
    // pre-existing `with_facts_integration` suite.

    mem.close().await.expect("close");
}

#[tokio::test]
async fn remember_reports_no_embedding_failures_on_a_working_embedder() {
    // Sensitivity control: prove the field is not just always non-empty by
    // construction — a WORKING embedder must report zero failures.
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("td253-control");

    let mem = Memory::open(dir.path().join("t.db"))
        .default_namespace(ns.clone())
        .with_llm(null_llm())
        .with_embedder(Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 }))
        .await
        .expect("open");

    let commit = mem
        .remember("Priya works at Northwind.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "priya".into(),
            predicate: "works_at".into(),
            object: "Northwind".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await
        .expect("remember");

    assert!(
        commit.embedding_failures.is_empty(),
        "a working embedder must report zero failures, got {:?}",
        commit.embedding_failures
    );

    mem.close().await.expect("close");
}
