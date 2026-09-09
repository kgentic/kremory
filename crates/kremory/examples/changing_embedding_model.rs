//! **Runs offline.** You switched embedding model. Your existing corpus was
//! embedded with the old one. Fix it without re-ingesting anything.
//!
//! ```text
//! cargo run --example changing_embedding_model
//! ```
//!
//! ## The problem this solves
//!
//! Embeddings are only comparable to other embeddings from the SAME model.
//! Change model and every stored vector becomes noise relative to new queries —
//! silently. Nothing errors; results just quietly get worse, which is the worst
//! way for a system to break.
//!
//! Re-ingesting is the obvious fix and usually the wrong one: it re-runs
//! extraction, costs another pass through a language model, and rewrites
//! provenance you wanted to keep. Re-embedding touches only the vectors.
//!
//! ## Two operations, and picking the wrong one is expensive
//!
//! - `backfill_*` — embeds rows that have NO vector yet. Use it when you added
//!   an embedder to a corpus ingested without one.
//! - `reembed_all_*` — recomputes EVERY vector. Use it when the model changed,
//!   because the existing vectors are not missing, they are wrong.
//!
//! Reaching for `backfill` after a model change is the trap: it finds nothing to
//! do, reports success, and leaves the whole corpus stale. This example shows
//! both so the difference is concrete.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use kremory::{
    CoreResult, EmbeddingProvider, EntityExtractor, ExtractionContext, ExtractionResult, Memory,
    Namespace,
};

const DIM: usize = 16;

/// Two "models" that produce different vectors for the same text. Real models
/// differ far more; what matters here is only that they DISAGREE, which is
/// exactly the condition that makes stored vectors stale.
struct FakeModel {
    /// Changes the output. Stands in for "a different model entirely".
    salt: u8,
}

impl EmbeddingProvider for FakeModel {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a {
        let salt = self.salt;
        async move {
            let mut v = vec![0f32; DIM];
            for (i, b) in text.bytes().enumerate() {
                v[i % DIM] += f32::from(b.wrapping_add(salt)) / 255.0;
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

async fn open_with(path: &Path, ns: &Namespace, salt: u8) -> anyhow::Result<Memory> {
    Ok(Memory::open(path)
        .embedding_dim(DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(FakeModel { salt }))
        .with_extractor(Arc::new(NoExtraction))
        .await?)
}

const CORPUS: &[&str] = &[
    "The northern line closes for maintenance every second Sunday.",
    "Platform staff must log every delay longer than four minutes.",
    "Freight services have priority between midnight and five.",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("corpus.db");
    let ns = Namespace::new("rail-ops");

    // ── Yesterday: ingested with model A ────────────────────────────────────
    {
        let mem = open_with(&db, &ns, 0).await?;
        for line in CORPUS {
            mem.remember(*line)
                .in_namespace(ns.clone())
                .skip_extraction()
                .await?;
        }
        println!("ingested {} episodes with model A", CORPUS.len());
        mem.close().await?;
    }

    // ── Today: same database, different model ───────────────────────────────
    let mem = open_with(&db, &ns, 97).await?;
    println!("reopened with model B — every stored vector is now stale");

    // The trap. Nothing is MISSING a vector, so backfill correctly finds no
    // work — and reports success while the corpus stays wrong.
    let backfilled = mem.backfill_episode_embeddings(64).await?;
    println!(
        "\nbackfill_episode_embeddings : embedded={} failed={}",
        backfilled.embedded, backfilled.failed
    );
    assert_eq!(
        backfilled.embedded, 0,
        "backfill should find nothing to do — the vectors exist, they are just \
         from the wrong model. Reporting work here would mean it had \
         misunderstood the situation."
    );
    println!("  → 0. Correct, and NOT what you needed. This is the trap.");

    // The right operation: recompute everything.
    let redone = mem.reembed_all_episode_embeddings(64).await?;
    println!(
        "\nreembed_all_episode_embeddings: embedded={} failed={}",
        redone.embedded, redone.failed
    );
    assert_eq!(
        redone.embedded as usize,
        CORPUS.len(),
        "every episode should be re-embedded after a model change; got {}",
        redone.embedded
    );
    assert_eq!(redone.failed, 0, "no episode should fail to re-embed");
    println!("  → all {} re-embedded under model B", redone.embedded);

    // Idempotent: safe to re-run, which matters because it is how you retry
    // partial failures without tracking what already succeeded.
    let again = mem.reembed_all_episode_embeddings(64).await?;
    assert_eq!(
        again.embedded as usize,
        CORPUS.len(),
        "re-running should be safe and complete, not skip or double-count"
    );
    println!("  → re-running is safe (idempotent), so retries need no bookkeeping");

    // And the corpus is still searchable afterwards.
    let hits = mem
        .recall("freight priority overnight")
        .in_namespace(ns.clone())
        .content()
        .await?;
    assert!(
        !hits.is_empty(),
        "the corpus must remain searchable after re-embedding"
    );
    println!("\nsearch after migration: {} passage(s)", hits.len());

    println!("\nNo re-ingest, no second pass through a language model, provenance intact.");
    println!("Pick `reembed_all_*` when the MODEL changed and `backfill_*` when the");
    println!("VECTOR is missing — they are not interchangeable, and the wrong one");
    println!("succeeds quietly.");

    mem.close().await?;
    Ok(())
}
