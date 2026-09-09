#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-244 — `supersede(fact_id)` must be reachable from the public recall path.
//!
//! Before this, `Memory::supersede` took a `fact_id` that NOTHING on the public
//! surface returned. The API was documented, exercised internally with hand-made
//! ids, and impossible for a consumer to call. The gap survived review and was
//! found by writing a beginner example.
//!
//! This test drives the CONSUMER journey rather than the internals: recall a
//! fact, take the id off the result, supersede it, observe it closed. If
//! `fact_id` ever stops being populated, this fails.

use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

#[tokio::test]
async fn supersede_is_reachable_from_recall() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("td244");

    let mem = Memory::open(dir.path().join("t.db"))
        .default_namespace(ns.clone())
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("open");

    mem.remember("Dana works at Northwind.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "dana".into(),
            predicate: "works_at".into(),
            object: "Northwind".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await
        .expect("remember");

    // The consumer's ONLY handle on a fact is what recall gives back.
    let facts: Vec<_> = mem
        .recall("dana")
        .in_namespace(ns.clone())
        .raw()
        .await
        .expect("recall")
        .into_iter()
        .flat_map(|c| c.facts)
        .collect();

    let target = facts
        .iter()
        .find(|f| f.predicate == "works_at")
        .expect("the fact just written should be recallable");

    // THE POINT: an id is present, so supersede can be called at all.
    let fact_id = target
        .fact_id
        .expect("TD-244: recall must expose fact_id, or supersede is unreachable");

    // `.at(..)` is REQUIRED by design — `execute()` refuses rather than
    // defaulting to now, because a missing bound is a caller bug. Note
    // `.close_now()` does NOT supply it: that flag runs the retirement sweep
    // for already-past bounds, a separate concern from setting the bound.
    mem.supersede(fact_id)
        .in_namespace(ns.clone())
        .at(chrono::Utc::now())
        .with_reason("she moved on")
        .close_now()
        .execute()
        .await
        .expect("supersede should accept an id obtained from recall");

    // And it took effect. A superseded fact is no longer "true now", so it
    // drops out of a default recall — that absence IS the proof the bound
    // landed, applied via an id the consumer obtained from recall.
    //
    // (Asserting it is still *returned* would be wrong: default recall answers
    // "what is true now", and after supersession this is not.)
    let after: Vec<_> = mem
        .recall("dana")
        .in_namespace(ns.clone())
        .raw()
        .await
        .expect("recall after")
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.predicate == "works_at")
        .collect();

    assert!(
        after.is_empty(),
        "superseding via the recalled id should drop the fact from a now-scoped \
         recall; still got {:?}",
        after.iter().map(|f| (&f.object, f.invalid_at)).collect::<Vec<_>>()
    );

    mem.close().await.expect("close");
}
