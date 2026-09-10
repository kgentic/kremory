//! **Runs offline.** A user invokes their right to erasure, and you have to prove
//! it was surgical.
//!
//! ```text
//! cargo run --example gdpr_erasure_by_source
//! ```
//!
//! ## The problem this solves
//!
//! Under GDPR Article 17 a person can require you to delete their data. For a
//! memory system that is harder than `DELETE FROM`, because facts extracted from
//! one conversation are mixed in with facts from every other conversation.
//!
//! kremory tracks the SOURCE each fact came from, so erasure is scoped to a source
//! rather than guessed at by matching on a name. `mem.forget().by_source_id(..)`
//! removes what that source contributed and leaves everything else intact.
//!
//! ## The assertion that matters
//!
//! Deleting too much is also a failure — it destroys data you were obliged to keep,
//! and you will not notice. So this example proves BOTH halves:
//!
//!   - the erased source's facts are gone, and
//!   - a second source's facts, about the SAME subject, still survive.
//!
//! The second assertion is the one people forget to write.

use std::future::Future;
use std::sync::Arc;

use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace, StructuredFact};

const DEMO_DIM: usize = 16;

/// Deterministic stand-in embedder — see `offline_remember_recall.rs`.
struct DemoEmbedder {
    dim: usize,
}

impl EmbeddingProvider for DemoEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a {
        let dim = self.dim;
        async move {
            let mut v = vec![0f32; dim];
            for (i, b) in text.bytes().enumerate() {
                v[i % dim] += f32::from(b) / 255.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= norm;
            }
            Ok(v)
        }
    }
}

/// Never invoked — every write here calls `.skip_extraction()`.
struct NoExtraction;

impl EntityExtractor for NoExtraction {
    fn name(&self) -> &'static str {
        "no-extraction"
    }

    async fn extract(
        &self,
        _text: &str,
        _ctx: &ExtractionContext<'_>,
    ) -> CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: Vec::new(),
            facts: Vec::new(),
        })
    }
}

/// Every predicate currently recalled for Dana. A plain `async fn` rather than a
/// closure — an async closure over a borrow needs lifetime gymnastics that would
/// teach the reader nothing about kremory.
async fn predicates_for(mem: &Memory, ns: Namespace) -> anyhow::Result<Vec<String>> {
    Ok(mem
        .recall("dana")
        .in_namespace(ns)
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .map(|f| f.predicate)
        .collect())
}

const ERASE_ME: &str = "support-chat-4417";
const KEEP_ME: &str = "signed-contract-2026-03";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("customer-records");

    let mem = Memory::open(dir.path().join("records.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // Two sources, both about the same person. Only one is being erased.
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
        .await?;

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
        .await?;

    let before = predicates_for(&mem, ns.clone()).await?;
    println!("before erasure : {before:?}");
    assert!(
        before.iter().any(|p| p == "complained_about"),
        "setup failed — the support-chat fact should exist before erasure"
    );

    // ── The erasure ─────────────────────────────────────────────────────────
    println!("\nerasing source {ERASE_ME} …");
    // `.execute()` is required — like every other mutating request in kremory,
    // nothing happens until you call the terminal. It returns how many rows went.
    let removed = mem
        .forget()
        .in_namespace(ns.clone())
        .by_source_id(ERASE_ME)
        .execute()
        .await?;
    // The outcome is per TABLE, and reading only one number is how you conclude
    // nothing happened. `entities: 0` is CORRECT here — Dana appears in the
    // surviving contract too, so her entity is deliberately PINNED while the
    // chat's facts and its episode are erased around her. Never write
    // `if removed > 0 { ... }` against a single count.
    println!(
        "  erased: entities={} facts={} episodes={} edges={}",
        removed.entities, removed.facts, removed.episodes, removed.edges
    );
    println!("  (entities=0 is correct — Dana is shared, so she is pinned)");
    assert_eq!(
        removed.entities, 0,
        "Dana is shared with the surviving contract, so her entity must be pinned"
    );
    assert!(
        removed.episodes > 0,
        "erasure must remove the SOURCE TEXT, not only the graph derived from it"
    );

    let after = predicates_for(&mem, ns.clone()).await?;
    println!("after erasure  : {after:?}");

    // Half one: the erased source is gone.
    assert!(
        !after.iter().any(|p| p == "complained_about"),
        "ERASURE INCOMPLETE — the support-chat fact survived: {after:?}"
    );

    // Half two, the one people forget: everything else survived.
    assert!(
        after.iter().any(|p| p == "signed"),
        "OVER-DELETION — erasing {ERASE_ME} also destroyed data from {KEEP_ME}, \
         which you were obliged to keep: {after:?}"
    );

    println!("\nThe support chat is gone. The signed contract — same person, different");
    println!("source — is untouched. Erasure scoped by SOURCE, not by matching a name.");

    mem.close().await?;
    Ok(())
}
