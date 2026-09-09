//! **Runs offline.** Delete a single fact, change your mind, put it back.
//!
//! ```text
//! cargo run --example deleting_and_restoring
//! ```
//!
//! ## The problem this solves
//!
//! Erasure (`forget()`) removes everything a SOURCE contributed — right for a
//! data-subject request, far too blunt for "that one line is wrong". Correction
//! (`supersede()`) closes a fact on the world clock, which is right when it USED
//! to be true. Neither fits the third case: **this should never have been
//! recorded at all.**
//!
//! `delete_fact()` is that case, and it is reversible.
//!
//! ## The chain, and where each id comes from
//!
//! This is worth reading closely, because each step hands you the handle for the
//! next and the handles come from different places:
//!
//! ```text
//!   recall()            → RetrievedFact::fact_id      (the fact to remove)
//!   delete_fact(id)     → DeleteFactOutcome           (does NOT carry a mutation id)
//!   list_mutations()    → MutationRecord::mutation_id (the handle for the undo)
//!   undo_delete_fact(m) → restored
//! ```
//!
//! Note the asymmetry in the middle: the delete does not hand back the id its
//! own undo requires, so you look it up. Every mutating call in the crate
//! behaves this way, and for an undo UI you would be listing anyway — but it is
//! a real step, and assuming otherwise is how people get stuck.

use std::future::Future;
use std::sync::Arc;

use kremory::{
    CoreResult, EmbeddingProvider, EntityExtractor, ExtractionContext, ExtractionResult, Memory,
    Namespace, StructuredFact,
};

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

/// Every (predicate, object) currently recalled for Rosa.
async fn current(mem: &Memory, ns: &Namespace) -> anyhow::Result<Vec<(String, String)>> {
    let mut v: Vec<(String, String)> = mem
        .recall("rosa")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .map(|f| (f.predicate, f.object))
        .collect();
    v.sort();
    Ok(v)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("hr");

    let mem = Memory::open(dir.path().join("hr.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // One correct record, and one that should never have been entered.
    mem.remember("Rosa is a structural engineer.")
        .in_namespace(ns.clone())
        .with_facts(vec![
            fact("rosa", "role", "structural engineer"),
            fact("rosa", "home_address", "14 Bellhaven Row"),
        ])
        .skip_extraction()
        .await?;

    println!("recorded      : {:?}", current(&mem, &ns).await?);

    // ── Find the offending fact ─────────────────────────────────────────────
    //
    // `fact_id` on a recalled fact is what makes this reachable at all. It was
    // added because `supersede` needed it; `delete_fact` needs the same handle.
    let target = mem
        .recall("rosa")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .find(|f| f.predicate == "home_address")
        .ok_or_else(|| anyhow::anyhow!("expected the home_address fact to be recallable"))?;

    let fact_id = target
        .fact_id
        .ok_or_else(|| anyhow::anyhow!("recall must expose fact_id for delete_fact to be callable"))?;

    // ── Delete it ───────────────────────────────────────────────────────────
    // No `.in_namespace(..)` here: a fact id is a globally-unique integer, so it
    // already identifies its namespace. (`delete_entity` DOES take one, because
    // entity ids are namespace-scoped strings — the asymmetry is deliberate.)
    let deleted = mem.delete_fact(fact_id).execute().await?;
    println!("deleted       : fact_id {}", deleted.fact_id);

    let after_delete = current(&mem, &ns).await?;
    println!("after delete  : {after_delete:?}");
    assert!(
        !after_delete.iter().any(|(p, _)| p == "home_address"),
        "the deleted fact should be gone; got {after_delete:?}"
    );
    // The neighbouring fact must be untouched — a delete that takes more than
    // it was asked for is the failure people do not test for.
    assert!(
        after_delete.iter().any(|(p, _)| p == "role"),
        "deleting one fact must not disturb another; got {after_delete:?}"
    );

    // ── Change your mind ────────────────────────────────────────────────────
    //
    // `DeleteFactOutcome` gives back the fact id, NOT the mutation id the undo
    // needs — so look it up. This is the step the doc comment above warns about.
    let history = mem.list_mutations().in_namespace(ns.clone()).await?;
    println!("\nmutation log:");
    for m in &history {
        println!("  #{} {:?} undone={} — {}", m.mutation_id, m.kind, m.undone, m.summary);
    }

    let del = history
        .iter()
        .find(|m| !m.undone)
        .ok_or_else(|| anyhow::anyhow!("expected the delete to appear in the mutation log"))?;

    mem.undo_delete_fact(del.mutation_id).execute().await?;
    println!("\nundid #{}", del.mutation_id);

    let restored = current(&mem, &ns).await?;
    println!("after undo    : {restored:?}");
    assert!(
        restored.iter().any(|(p, _)| p == "home_address"),
        "the undo should restore the deleted fact; got {restored:?}"
    );
    assert!(
        restored.iter().any(|(p, _)| p == "role"),
        "and must still not disturb its neighbour; got {restored:?}"
    );

    println!("\nThree different tools for three different situations:");
    println!("  forget()      — erase everything a SOURCE contributed (a data-subject request)");
    println!("  supersede()   — it USED to be true, close it on the world clock");
    println!("  delete_fact() — it should never have been recorded, and this is reversible");

    mem.close().await?;
    Ok(())
}
