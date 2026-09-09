//! **Runs offline.** Something automated changed your graph and it was wrong.
//! Look at what happened, and put it back.
//!
//! ```text
//! cargo run --example undoing_a_bad_change
//! ```
//!
//! ## The problem this solves
//!
//! Memory systems mutate themselves. Consolidation merges two entities it
//! believes are the same person; a cleanup job renames something; a bulk edit
//! runs against the wrong namespace. Occasionally one of those is wrong, and the
//! usual answer — restore last night's backup — throws away everything correct
//! that happened since.
//!
//! kremory logs every graph mutation and can reverse them individually. That is
//! the difference between "we have backups" and "we can fix this."
//!
//! ## The shape worth copying
//!
//! You rarely know the id of the change you want to undo — you know something
//! looks wrong. So the flow is **inspect, then reverse**:
//!
//!   1. `list_mutations()` — what happened, in human-readable summaries
//!   2. pick the one that was wrong
//!   3. `undo_entity_edit(mutation_id)` — put it back
//!
//! That is also exactly how you would build an undo UI, so the API shape and the
//! product shape agree.

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

/// The ENTITY IDS currently in the graph.
///
/// ⚠️ Read `RetrievedContext::entity_id`, NOT `RetrievedFact::subject`. The
/// fact's `subject` is a DISPLAY NAME, which a rename does not change — the
/// first draft of this example probed it and both assertions passed
/// vacuously, "proving" an undo that had not been observed at all.
async fn entity_ids(mem: &Memory, ns: &Namespace) -> anyhow::Result<Vec<String>> {
    let mut v: Vec<String> = mem
        .recall("engineer")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .map(|c| c.entity_id)
        .collect();
    v.sort();
    v.dedup();
    Ok(v)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("directory");

    let mem = Memory::open(dir.path().join("dir.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    mem.remember("Priya is an engineer at Northwind.")
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
        .await?;

    println!("before      : {:?}", entity_ids(&mem, &ns).await?);

    // ── The bad change ──────────────────────────────────────────────────────
    //
    // Pretend a cleanup job decided "priya" should be keyed "p.sharma". It is
    // not wrong syntactically; it is just wrong.
    let edit = mem
        .edit_entity("priya")
        .in_namespace(ns.clone())
        .rename("p.sharma")
        .execute()
        .await?;

    // Assert the edit ACTUALLY happened before asserting the undo reverses it.
    // Without this the whole example can pass while nothing occurred.
    assert!(edit.rekeyed, "the rename should have re-keyed the entity");
    assert_eq!(edit.entity_id, "p.sharma", "the new id should be the renamed one");
    println!("edit        : rekeyed={} facts_repointed={}", edit.rekeyed, edit.facts_repointed);

    println!("after edit  : {:?}", entity_ids(&mem, &ns).await?);

    // ── Inspect, then reverse ───────────────────────────────────────────────
    //
    // NOTE: `EditEntityOutcome` does NOT carry the mutation id, so you cannot
    // undo directly from the result of the edit you just made — you look it up.
    // For an undo UI that is the natural flow anyway (you list, the user picks),
    // and unlike some surfaces the id IS obtainable, which is what matters.
    let history = mem.list_mutations().in_namespace(ns.clone()).await?;

    println!("\nmutation log:");
    for m in &history {
        println!(
            "  #{} {:?} undone={} — {}",
            m.mutation_id, m.kind, m.undone, m.summary
        );
    }

    let bad = history
        .iter()
        .find(|m| !m.undone)
        .ok_or_else(|| anyhow::anyhow!("expected the rename to appear in the mutation log"))?;

    println!("\nundoing #{} …", bad.mutation_id);
    // No `.in_namespace(..)` here: the mutation id already identifies which
    // namespace the change belonged to, so re-stating it would be a second
    // source of truth that could disagree.
    mem.undo_entity_edit(bad.mutation_id)
        .execute()
        .await?;

    let restored = entity_ids(&mem, &ns).await?;
    println!("after undo  : {restored:?}");

    assert!(
        restored.iter().any(|s| s == "priya"),
        "undo should restore the original entity id; got {restored:?}"
    );
    assert!(
        !restored.iter().any(|s| s == "p.sharma"),
        "the renamed id should be gone after the undo; got {restored:?}"
    );

    // The log records the reversal too — an undo is itself auditable, not a
    // quiet rewrite of history.
    let after = mem
        .list_mutations()
        .in_namespace(ns.clone())
        .include_undone(true)
        .await?;
    let marked = after.iter().find(|m| m.mutation_id == bad.mutation_id);
    assert!(
        marked.map(|m| m.undone).unwrap_or(false),
        "the reversed mutation should be marked undone in the log"
    );

    println!("\nThe change is reversed AND the reversal is on the record. You did not");
    println!("restore a backup, so everything else that happened since is untouched.");

    mem.close().await?;
    Ok(())
}
