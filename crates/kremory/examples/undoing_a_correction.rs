//! **Runs offline.** You corrected a record. The correction was wrong. Put it
//! back.
//!
//! ```text
//! cargo run --example undoing_a_correction
//! ```
//!
//! ## The problem this solves
//!
//! `correcting_the_record` closes a fact on the world clock — "she worked there
//! until March". Then someone checks and it turns out she never left. The
//! correction was the error.
//!
//! `unsupersede()` clears the bound and the fact is current again. It takes a
//! `fact_id`, NOT a mutation id, so it pairs directly with what recall already
//! hands you.
//!
//! ## An outcome type worth copying
//!
//! `UnsupersedeOutcome` is an enum, not a boolean:
//!
//! ```text
//!   Cleared { fact_id, cleared_valid_to, cleared_expired_at }
//!   NotSuperseded { fact_id }
//! ```
//!
//! Calling it on a fact that was never superseded returns `NotSuperseded` rather
//! than `Ok(())` — an honest no-op instead of a success that means nothing. Both
//! branches are exercised below, because the second is the one that tells you
//! your assumption was wrong.

use std::future::Future;
use std::sync::Arc;

use chrono::Utc;
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

async fn current(mem: &Memory, ns: &Namespace) -> anyhow::Result<Vec<String>> {
    Ok(mem
        .recall("wren")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .filter(|f| f.predicate == "works_at")
        .map(|f| f.object)
        .collect())
}

async fn works_at_fact_id(mem: &Memory, ns: &Namespace) -> anyhow::Result<i64> {
    mem.recall("wren")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .find(|f| f.predicate == "works_at")
        .and_then(|f| f.fact_id)
        .ok_or_else(|| anyhow::anyhow!("expected a recallable works_at fact with an id"))
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

    mem.remember("Wren works at Halden Institute.")
        .in_namespace(ns.clone())
        .with_facts(vec![StructuredFact {
            subject: "wren".into(),
            predicate: "works_at".into(),
            object: "Halden Institute".into(),
            valid_from: None,
            valid_to: None,
            memory_type: None,
        }])
        .skip_extraction()
        .await?;

    let fact_id = works_at_fact_id(&mem, &ns).await?;
    println!("recorded            : {:?}", current(&mem, &ns).await?);

    // ── The correction, which will turn out to be wrong ─────────────────────
    //
    // `.close_now()` alongside `.at(..)` — see `correcting_the_record` for why
    // `.at(..)` alone leaves the fact still reading as current.
    mem.supersede(fact_id)
        .in_namespace(ns.clone())
        .at(Utc::now())
        .with_reason("heard she left")
        .close_now()
        .execute()
        .await?;

    let after = current(&mem, &ns).await?;
    println!("after 'correction'  : {after:?}");
    assert!(
        after.is_empty(),
        "the supersession should have closed the fact; got {after:?}"
    );

    // ── It was wrong. She never left. ───────────────────────────────────────
    let outcome = mem.unsupersede(fact_id).execute().await?;
    println!("unsupersede         : {outcome:?}");

    let restored = current(&mem, &ns).await?;
    println!("after undo          : {restored:?}");
    assert!(
        restored.iter().any(|o| o == "Halden Institute"),
        "clearing the bound should make the fact current again; got {restored:?}"
    );

    // ── And the honest no-op ────────────────────────────────────────────────
    //
    // Calling it again: the bound is already cleared, so there is nothing to do
    // and the outcome SAYS SO rather than returning a meaningless success.
    let again = mem.unsupersede(fact_id).execute().await?;
    println!("unsupersede again   : {again:?}");
    assert!(
        matches!(
            again,
            kremory::UnsupersedeOutcome::NotSuperseded { .. }
        ),
        "a second call should report NotSuperseded, not a success that means \
         nothing; got {again:?}"
    );

    println!("\nFour tools, four situations — and this is the one for 'the correction");
    println!("itself was the mistake'. Nothing was ever deleted, so it was always");
    println!("recoverable: a closed fact is closed, not gone.");

    mem.close().await?;
    Ok(())
}
