#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-246 — `forget().by_source_id(..)` must work when a subject appears in
//! MORE THAN ONE source. That is the only case that matters; a single-source
//! graph is a demo.
//!
//! Before the fix this raised `FOREIGN KEY constraint failed`, so kremory had no
//! working right-to-erasure path for any realistic graph. The cause was a third
//! foreign key into `episodes(id)` — `facts.source_episode_id` — that the delete
//! cascade never handled. With one source the entity is unshared and gets deleted,
//! taking its facts with it, and the bug hides. With two sources the entity is
//! PINNED by shared-entity preservation, its fact survives, and the dangling
//! reference breaks the constraint.
//!
//! Both halves are asserted. Over-deletion is also a failure — it destroys data
//! you were obliged to keep, and nobody notices.

use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

const ERASE_ME: &str = "support-chat-4417";
const KEEP_ME: &str = "signed-contract-2026-03";

async fn predicates_for(mem: &Memory, ns: Namespace) -> Vec<String> {
    mem.recall("dana")
        .in_namespace(ns)
        .raw()
        .await
        .expect("recall")
        .into_iter()
        .flat_map(|c| c.facts)
        .map(|f| f.predicate)
        .collect()
}

#[tokio::test]
async fn erasure_is_surgical_across_two_sources() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("customer-records");

    let mem = Memory::open(dir.path().join("records.db"))
        .default_namespace(ns.clone())
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("open");

    // Two sources, SAME subject — this is what pins the entity and exposes the bug.
    mem.remember("Support chat with Dana.")
        .in_namespace(ns.clone())
        .from_chat(ERASE_ME)
        .with_facts(vec![StructuredFact {
            subject: "dana".into(),
            predicate: "complained_about".into(),
            object: "late delivery".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await
        .expect("remember chat");

    mem.remember("Signed contract with Dana.")
        .in_namespace(ns.clone())
        .from_document(KEEP_ME)
        .with_facts(vec![StructuredFact {
            subject: "dana".into(),
            predicate: "signed".into(),
            object: "service agreement".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await
        .expect("remember contract");

    let before = predicates_for(&mem, ns.clone()).await;
    assert!(
        before.iter().any(|p| p == "complained_about") && before.iter().any(|p| p == "signed"),
        "setup: both sources' facts should exist before erasure, got {before:?}"
    );

    // The erasure itself. Before the fix this returned a FK violation.
    mem.forget()
        .in_namespace(ns.clone())
        .by_source_id(ERASE_ME)
        .execute()
        .await
        .expect("TD-246: forget by_source_id must succeed on a multi-source graph");

    let after = predicates_for(&mem, ns.clone()).await;

    // Half one — the erased source is gone.
    assert!(
        !after.iter().any(|p| p == "complained_about"),
        "ERASURE INCOMPLETE: the erased source's fact survived: {after:?}"
    );

    // Half two — the half people forget to write.
    assert!(
        after.iter().any(|p| p == "signed"),
        "OVER-DELETION: erasing one source also destroyed the other source's fact, \
         which you were obliged to keep: {after:?}"
    );

    mem.close().await.expect("close");
}
