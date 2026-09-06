#![cfg(feature = "content-search")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-068 extension — `recall().as_of()` now scopes episode/content search,
//! not just the fact 1-hop expansion.
//!
//! `.ai-docs/adrs/adr-068-as-of-temporal-recall-2026-07-03.md` Decision 3
//! scoped `as_of` to the fact stream only, reasoning "entities carry no
//! temporal columns" — true for entities, but never actually true for
//! episodes: `episodes.timestamp` has carried resolved world-time since TD-187
//! and was already selected by both `content_search` (BM25) and
//! `vector_search_episodes` (dense) — it was simply never filtered on. That
//! gap went unnoticed because content-search was a minor arm when ADR-068
//! was written and became the default-on primary arm three weeks later
//! (ADR-078). See TD register entries for this session's fix.
//!
//! Sibling to `facade_as_of_warn.rs` (the original fact-side ADR-068 suite,
//! whose helper/naming conventions this file mirrors) and
//! `adr072_seq1_content_search.rs` / `td066_increment1_content_fusion.rs`
//! (the content-search producer path this file drives).
//!
//! Episode world-time (`occurred_at`/`episodes.timestamp`) has no public
//! setter — `IngestRequest` hardcodes it to `Utc::now()` at construction
//! (verified: 8 construction sites in `facade/remember.rs`, none accept a
//! caller-supplied occurred-at). So these tests control `.as_of()` relative
//! to a real wall-clock timestamp captured immediately BEFORE the ingest
//! call, rather than controlling the episode's own timestamp directly — the
//! only lever the public API actually exposes.
//!
//! Lower-level SQL-generation coverage (both `content_search` and BOTH
//! `vector_search_episodes_with_index` / `_brute_force` called directly,
//! plus a boundary-inclusive pin) lives in `core::search::tests` — this file
//! proves the same fix is reachable end-to-end through the real public
//! surface a consumer actually uses.

use std::sync::Arc;

use chrono::Utc;
use kremory::core::provider::MockChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(MockChatProvider::null())
}

/// Always returns the same non-zero vector regardless of input text — used
/// to isolate the dense episode arm from BM25 (see
/// `as_of_scopes_dense_episode_arm_when_lexical_arm_would_miss` below).
/// Mirrors the established `RecordingEmbeddingProvider`/
/// `WindowLimitedEmbeddingProvider` pattern (`td143_reembed_all_episode_
/// embeddings.rs`, `td232_dense_embed_failure_surfaced.rs`) rather than
/// inventing a new shape.
struct AlwaysMatchEmbeddingProvider {
    dim: usize,
}

impl kremory::core::provider::EmbeddingProvider for AlwaysMatchEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> impl std::future::Future<Output = kremory::CoreResult<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move { Ok(vec![1.0_f32; dim]) }
    }
}

async fn make_memory(ns: &str, embedder: Arc<dyn DynEmbeddingProvider>) -> Memory {
    Memory::open(":memory:")
        .with_llm(null_llm())
        .with_embedder(embedder)
        .default_namespace(Namespace::new(ns))
        .await
        .expect("Memory must build")
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// `as_of` set to a moment BEFORE the episode was ingested must exclude it
/// from `.content()` (the BM25-only terminal — `adr072_seq1_content_search.
/// rs`'s own terminal, unaffected by this fix until now).
#[tokio::test]
async fn as_of_before_ingest_excludes_episode_from_content_terminal() {
    let mem = make_memory("adr068-content-before", null_embedder()).await;

    let t_before_ingest = Utc::now();
    mem.remember("Xenobia explored the ancient ruins in Peru.")
        .skip_extraction()
        .await
        .expect("episode must commit");

    let passages = mem
        .recall("Xenobia ruins")
        .as_of(t_before_ingest)
        .content()
        .await
        .expect("content recall must succeed");

    assert!(
        passages.is_empty(),
        "as_of before the episode existed must exclude it from content search; \
         got: {passages:?}"
    );
}

/// `as_of` set to a moment AFTER the episode was ingested must include it.
#[tokio::test]
async fn as_of_after_ingest_includes_episode_in_content_terminal() {
    let mem = make_memory("adr068-content-after", null_embedder()).await;

    mem.remember("Xenobia explored the ancient ruins in Peru.")
        .skip_extraction()
        .await
        .expect("episode must commit");
    let t_after_ingest = Utc::now();

    let passages = mem
        .recall("Xenobia ruins")
        .as_of(t_after_ingest)
        .content()
        .await
        .expect("content recall must succeed");

    assert_eq!(
        passages.len(),
        1,
        "as_of after the episode was ingested must include it; got: {passages:?}"
    );
    assert!(passages[0].snippet.to_lowercase().contains("xenobia"));
}

/// Regression pin — no `.as_of()` call at all (the default, `None`) must be
/// unaffected: content search still returns present-day results, matching
/// `adr072_seq1_content_search.rs`'s pre-existing behaviour (which predates
/// this fix and must not regress).
#[tokio::test]
async fn as_of_none_is_unaffected_for_content_terminal_regression_pin() {
    let mem = make_memory("adr068-content-none", null_embedder()).await;

    mem.remember("Xenobia explored the ancient ruins in Peru.")
        .skip_extraction()
        .await
        .expect("episode must commit");

    let passages = mem
        .recall("Xenobia ruins")
        .content()
        .await
        .expect("content recall must succeed");

    assert_eq!(
        passages.len(),
        1,
        "as_of=None must be unaffected — episode must still be present; got: {passages:?}"
    );
}

/// Proves the fix reaches the DENSE episode arm specifically, not just BM25.
///
/// The query shares ZERO tokens with the episode content, so `content_search`
/// (lexical) returns nothing for it — any hit that reaches `.raw()` for this
/// episode can only have come from `vector_search_episodes` (the dense arm).
/// `AlwaysMatchEmbeddingProvider` makes the dense arm match regardless of
/// semantic content, isolating "does the dense arm respect as_of" from "does
/// the dense arm rank correctly" — the same isolation technique
/// `td066_increment1_content_fusion.rs` uses for the fusion boundary itself.
#[tokio::test]
async fn as_of_scopes_dense_episode_arm_when_lexical_arm_would_miss() {
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(AlwaysMatchEmbeddingProvider { dim: 384 });
    let mem = make_memory("adr068-dense-isolate", embedder).await;

    let t_before_ingest = Utc::now();
    let commit = mem
        .remember("Quokka migration patterns differ across the archipelago.")
        .skip_extraction()
        .await
        .expect("episode must commit");
    let episode_id: i64 = commit
        .episode_entity_id
        .parse()
        .expect("commit must carry a parseable rowid (inline Phase 1 path)");
    let t_after_ingest = Utc::now();

    // Zero token overlap with the episode content — BM25 alone would return
    // nothing for this query regardless of as_of.
    let query = "orbital telescope calibration procedure";

    // as_of BEFORE ingest: dense arm must also exclude it (this is the
    // behaviour under test — before the fix, the dense arm ignored as_of
    // entirely and would have returned this episode here).
    let raw_before = mem
        .recall(query)
        .as_of(t_before_ingest)
        .raw()
        .await
        .expect("raw recall must succeed");
    assert!(
        !raw_before
            .iter()
            .any(|r| r.entity_id == episode_id.to_string()),
        "dense arm must respect as_of before ingest even though BM25 has \
         nothing to say about this query; got: {raw_before:?}"
    );

    // as_of AFTER ingest: dense arm must include it, proving the exclusion
    // above was genuinely as_of-driven and not just "the dense arm never
    // matches this query".
    let raw_after = mem
        .recall(query)
        .as_of(t_after_ingest)
        .raw()
        .await
        .expect("raw recall must succeed");
    assert!(
        raw_after
            .iter()
            .any(|r| r.entity_id == episode_id.to_string()),
        "dense arm must include the episode once as_of is after ingest; \
         got: {raw_after:?}"
    );
}
