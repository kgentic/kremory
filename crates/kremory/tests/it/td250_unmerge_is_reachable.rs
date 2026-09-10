#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-250 — an entity merge IS producible through the public API, so `unmerge` can
//! be exercised end to end.
//!
//! The register recorded the opposite twice, for two different wrong reasons, and
//! both were measurement failures rather than reasoning failures:
//!
//! 1. "consolidation correctly declines on thin evidence" — the gate is not a
//!    similarity threshold. It is `shares_structure`: two spellings must share a
//!    neighbour OR an identical `(predicate, object)` fact, else they are treated
//!    as homonyms. Every corpus tried gave the two spellings DIFFERENT facts,
//!    which is exactly what that gate reads as "two people with the same name".
//! 2. "the pair is never even considered" — wrong in the other direction. It is
//!    admitted, then rejected by the gate above.
//!
//! What actually hid it: `dream()` defaults to `CrossEpisodeMode::Shadow`, where
//! `cross_episode_merged` is ALWAYS `0` by construction and `cross_episode_would_merge`
//! carries the decision. `what_can_be_undone.rs` asserted on the former, so its
//! tripwire watched a number the default configuration pins at zero.
//!
//! The recipe, and what this file locks down:
//!   - two spellings of one subject, in ≥ 2 distinct episodes
//!   - sharing an IDENTICAL `(predicate, object)` fact — the corroboration
//!   - `dream().cross_episode(Apply)` — or nothing commits
//!
//! DETERMINISTIC, zero-LLM: admission is label + shingle Jaccard, corroboration is
//! SQL. No embeddings are consulted on this path at all.

use std::sync::Arc;

use kremory::facade::MutationKind;
use kremory::memory::types::CrossEpisodeMode;
use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

fn fact(subject: &str, predicate: &str, object: &str) -> StructuredFact {
    StructuredFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        valid_from: None,
        valid_to: None,
        memory_type: None,
    }
}

/// Two spellings, four episodes, one shared `(predicate, object)` between them.
async fn seed(mem: &Memory, ns: &Namespace) {
    for (s, p, o) in [
        ("Margarethe Solberg", "works_at", "Nordvik"),
        ("margarethe solberg", "works_at", "Nordvik"),
        ("Margarethe Solberg", "role", "auditor"),
        ("margarethe solberg", "role", "auditor"),
    ] {
        mem.remember(format!("{s} {p} {o}."))
            .in_namespace(ns.clone())
            .with_facts(vec![fact(s, p, o)])
            .skip_extraction()
            .await
            .unwrap();
    }
}

async fn open_mem(dir: &tempfile::TempDir, ns: &Namespace) -> Memory {
    Memory::open(dir.path().join("merge.db").to_str().unwrap())
        .default_namespace(ns.clone())
        .with_llm(Arc::new(kremory::core::provider::MockChatProvider::null()))
        .with_embedder(Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
            as Arc<dyn DynEmbeddingProvider>)
        .await
        .unwrap()
}

/// The full round trip: merge through the public API, name it, reverse it.
#[tokio::test]
async fn a_merge_is_producible_and_unmerge_reverses_it() {
    let dir = tempfile::tempdir().unwrap();
    let ns = Namespace::new("aliases");
    let mem = open_mem(&dir, &ns).await;
    seed(&mem, &ns).await;

    let summary = mem
        .dream()
        .in_namespace(ns.clone())
        .cross_episode(CrossEpisodeMode::Apply)
        .execute()
        .await
        .unwrap();
    assert_eq!(
        summary.cross_episode_merged, 1,
        "Apply mode must COMMIT the merge the corroboration gate approved"
    );

    let merges = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .kind(MutationKind::EntityMerge)
        .await
        .unwrap();
    assert_eq!(merges.len(), 1, "the merge must be listed: {merges:?}");
    let record = &merges[0];
    assert!(
        record.summary.contains("margarethe solberg"),
        "the summary must name the loser: {}",
        record.summary
    );

    // THE POINT: the mutation_id is enough to reverse it. `unmerge` was recorded as
    // publicly unreachable for exactly as long as nobody could produce a merge.
    let outcome = mem.unmerge(record.mutation_id).execute().await.unwrap();
    assert_eq!(outcome.restored_entity, "margarethe solberg");

    let after = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .kind(MutationKind::EntityMerge)
        .include_undone(true)
        .await
        .unwrap();
    assert!(after[0].undone, "the reversal must be recorded on the row");
}

/// The default is SHADOW, and that is why the gap looked permanent: `merged` is
/// pinned at `0` there, so a test asserting on it can never observe a merge.
#[tokio::test]
async fn dream_defaults_to_shadow_so_merged_is_zero_and_would_merge_carries_the_decision() {
    let dir = tempfile::tempdir().unwrap();
    let ns = Namespace::new("aliases-shadow");
    let mem = open_mem(&dir, &ns).await;
    seed(&mem, &ns).await;

    let summary = mem.dream().in_namespace(ns.clone()).execute().await.unwrap();
    assert_eq!(
        summary.cross_episode_would_merge, 1,
        "the op DECIDED to merge — shadow mode observes, it does not decline"
    );
    assert_eq!(
        summary.cross_episode_merged, 0,
        "...and committed nothing, which is what shadow means"
    );
    assert!(
        mem.list_mutations()
            .in_namespace(ns.clone())
            .kind(MutationKind::EntityMerge)
            .include_undone(true)
            .await
            .unwrap()
            .is_empty(),
        "a shadow decision must leave NO mutation behind"
    );
}
