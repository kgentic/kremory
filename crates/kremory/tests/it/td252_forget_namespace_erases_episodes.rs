#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-252 — namespace-wide `forget()` must erase the SOURCE TEXT, not only the
//! graph derived from it.
//!
//! `forget().in_namespace(ns)` deleted entities, facts and `episodic_edges` and
//! left every `episodes` row standing. The episode body is the personal data —
//! the graph is a derivative of it — and content search returned it verbatim
//! afterwards. Measured before the fix:
//!
//! ```text
//! entities removed = 1
//! recall passages after forget = 1
//!   text: Dana Fitzwilliam complained about a late delivery.
//! ```
//!
//! TD-246 built this cascade for the `by_source_id` path and it was never brought
//! across, so the two scopes of the same verb meant different things.
//!
//! DETERMINISTIC, zero-LLM. Fast tier, no VCR (`llm-test-pyramid-vcr-seams`).

use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

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

async fn open_mem(dir: &tempfile::TempDir, ns: &Namespace) -> Memory {
    let path = dir.path().join("erase.db");
    Memory::open(path.to_str().unwrap())
        .default_namespace(ns.clone())
        .with_llm(stub_llm())
        .with_embedder(null_embedder())
        .await
        .unwrap()
}

/// The one that matters: after erasing a namespace, its text must not come back
/// from content search.
#[tokio::test]
async fn namespace_forget_removes_the_episode_text_not_just_the_graph() {
    let dir = tempfile::tempdir().unwrap();
    let ns = Namespace::new("subject-access");
    let mem = open_mem(&dir, &ns).await;

    mem.remember("Dana Fitzwilliam complained about a late delivery.")
        .in_namespace(ns.clone())
        .with_facts(vec![fact("dana", "complained_about", "late delivery")])
        .skip_extraction()
        .await
        .unwrap();

    let before = mem
        .recall("late delivery")
        .in_namespace(ns.clone())
        .content()
        .await
        .unwrap();
    assert!(
        !before.is_empty(),
        "precondition: the episode must be recallable before erasure — a test \
         that erases nothing proves nothing"
    );

    let outcome = mem
        .forget()
        .in_namespace(ns.clone())
        .execute()
        .await
        .unwrap();
    assert_eq!(
        outcome.episodes, 1,
        "the episode row IS the personal data; erasing only the derived graph is \
         not erasure. got {outcome:?}"
    );

    let after = mem
        .recall("late delivery")
        .in_namespace(ns.clone())
        .content()
        .await
        .unwrap();
    assert!(
        after.is_empty(),
        "erased text must not be recallable; content search returned {:?}",
        after.iter().map(|p| &p.snippet).collect::<Vec<_>>()
    );
}

/// The outcome is per table, so an erasure that pins every entity still reports
/// what it did (TD-247) — the defect that made `entities > 0` a lie.
#[tokio::test]
async fn forget_outcome_reports_every_table_not_just_entities() {
    let dir = tempfile::tempdir().unwrap();
    let ns = Namespace::new("counts");
    let mem = open_mem(&dir, &ns).await;

    for (text, s, p, o) in [
        ("Dana signed the service agreement.", "dana", "signed", "service agreement"),
        ("Dana complained about a late delivery.", "dana", "complained_about", "late delivery"),
    ] {
        mem.remember(text)
            .in_namespace(ns.clone())
            .with_facts(vec![fact(s, p, o)])
            .skip_extraction()
            .await
            .unwrap();
    }

    let outcome = mem
        .forget()
        .in_namespace(ns.clone())
        .execute()
        .await
        .unwrap();

    assert!(!outcome.is_empty(), "something was erased; got {outcome:?}");
    assert_eq!(outcome.episodes, 2, "both episodes: {outcome:?}");
    assert_eq!(outcome.facts, 2, "both facts: {outcome:?}");
    assert!(
        outcome.entities >= 1,
        "the subject entity is unshared here, so it goes too: {outcome:?}"
    );
}
