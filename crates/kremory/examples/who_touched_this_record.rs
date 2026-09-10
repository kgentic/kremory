//! **Runs offline.** Prove what happened to one record — including the changes
//! that were later reversed.
//!
//! ```text
//! cargo run --example who_touched_this_record
//! ```
//!
//! ## The problem this solves
//!
//! "Show me everything that ever happened to this record" is an audit question,
//! and it is not the same question as "what can I still undo". A change that was
//! made and then reversed is **still an audit fact** — arguably the most
//! interesting one, because someone made it and someone took it back.
//!
//! kremory has two views over the mutation log, and picking the wrong one for an
//! audit means quietly under-reporting.
//!
//! ## The two views, and their defaults
//!
//! ```text
//!   list_mutations()                    namespace-wide · LIVE ONLY by default
//!   list_mutations().include_undone(true)   ...and the reversed ones
//!   mutation_history(entity_id)         one entity · ALWAYS includes reversed
//! ```
//!
//! `list_mutations()` is undo-oriented: it answers "what can I still reverse",
//! so hiding already-reversed entries is correct for that job. Reach for it to
//! build an undo menu and you get the right answer.
//!
//! **Reach for it to answer an auditor and you get an incomplete one.** That is
//! what this example demonstrates: the same history, seen through both views,
//! with a change deliberately made and then reversed so the difference is
//! visible rather than described.

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("claims");

    let mem = Memory::open(dir.path().join("claims.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    mem.remember("Claim 88-C filed by Solveig Naess.")
        .in_namespace(ns.clone())
        .with_facts(vec![
            fact("claim-88c", "filed_by", "Solveig Naess"),
            fact("claim-88c", "status", "under review"),
            fact("claim-88c", "assessor_note", "possible duplicate"),
        ])
        .skip_extraction()
        .await?;

    println!("claim recorded with 3 facts");

    // ── Someone deletes a note, then thinks better of it ────────────────────
    //
    // This is the sequence the two views disagree about. Both steps are real
    // actions taken by a real person, and an audit must see both.
    let note = mem
        .recall("claim-88c")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .find(|f| f.predicate == "assessor_note")
        .and_then(|f| f.fact_id)
        .ok_or_else(|| anyhow::anyhow!("expected the assessor note to be recallable"))?;

    mem.delete_fact(note).execute().await?;
    println!("  → assessor note deleted");

    let del = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .await?
        .into_iter()
        .find(|m| !m.undone)
        .ok_or_else(|| anyhow::anyhow!("expected the delete to be listed"))?;

    mem.undo_delete_fact(del.mutation_id).execute().await?;
    println!("  → and restored a moment later");

    // ── View one: what can I still undo? ────────────────────────────────────
    let live = mem.list_mutations().in_namespace(ns.clone()).await?;
    println!("\nlist_mutations()                  → {} entry(ies)", live.len());
    for m in &live {
        println!("    #{} {:?} undone={}", m.mutation_id, m.kind, m.undone);
    }

    // ── View two: the same log, nothing hidden ──────────────────────────────
    let all = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .include_undone(true)
        .await?;
    println!("list_mutations(include_undone)     → {} entry(ies)", all.len());

    // ── View three: everything that touched THIS record ─────────────────────
    //
    // No flag needed — an entity's history always includes reversals, because
    // that is what "history" means.
    let history = mem
        .mutation_history("claim-88c")
        .in_namespace(ns.clone())
        .await?;
    println!("mutation_history(\"claim-88c\")      → {} entry(ies)", history.len());
    for m in &history {
        println!("    #{} {:?} undone={} — {}", m.mutation_id, m.kind, m.undone, m.summary);
    }

    // ── The point ───────────────────────────────────────────────────────────
    assert!(
        all.len() > live.len(),
        "the reversed mutation must be hidden by the default view and visible \
         with include_undone — otherwise the two views are the same and there is \
         nothing to choose between them. live={} all={}",
        live.len(),
        all.len()
    );
    assert!(
        history.iter().any(|m| m.undone),
        "an entity's history must include reversed mutations without being \
         asked — a change that was made and taken back is still an audit fact"
    );

    println!("\nThe default view hid the reversed delete, correctly: you cannot undo");
    println!("something already undone, so an undo menu should not offer it.");
    println!("An AUDIT needs the opposite default — use mutation_history for a record,");
    println!("or include_undone(true) for a namespace. Reaching for the undo view to");
    println!("answer an auditor under-reports, and nothing warns you.");

    mem.close().await?;
    Ok(())
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
