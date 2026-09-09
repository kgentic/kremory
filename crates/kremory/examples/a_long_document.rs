//! **Runs offline.** Ingest a document far longer than an embedding model's
//! context window, and find something buried at the end of it.
//!
//! ```text
//! cargo run --example a_long_document
//! ```
//!
//! ## The problem this solves
//!
//! Real documents are not tweets. A policy PDF, a long meeting transcript, an
//! RFC — these run to thousands of words, and every embedding model has a hard
//! input limit. Something has to decide what happens to the text past that
//! limit.
//!
//! The failure mode is quiet: the document ingests "successfully", but only its
//! opening survives into the index. Everything after the cut is stored and
//! unfindable. You discover it when someone asks about the part at the end and
//! is told there is nothing.
//!
//! kremory splits long content for embedding rather than truncating it. This
//! example proves that by hiding a distinctive term in the LAST paragraph of a
//! deliberately oversized document and then asking for it.
//!
//! ## The assertion that matters
//!
//! Not "the ingest returned Ok" — a truncating implementation also returns Ok.
//! The assertion is that a term **past the cut-off** comes back.

use std::future::Future;
use std::sync::Arc;

use kremory::{
    CoreResult, EmbeddingProvider, EntityExtractor, ExtractionContext, ExtractionResult, Memory,
    Namespace,
};

const DEMO_DIM: usize = 16;

/// Deterministic stand-in embedder — see `offline_remember_recall.rs`.
///
/// Note it accepts any length: the point here is what KREMORY does with a long
/// document before it ever reaches an embedder, not what a particular model's
/// limit happens to be.
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

/// A distinctive term that appears exactly ONCE, in the final paragraph.
/// Nonsense on purpose — it cannot be matched by luck.
const BURIED_TERM: &str = "quillfeather";

/// Build a document long enough to exceed any realistic embedding window,
/// with the marker at the very end.
fn long_document() -> String {
    let filler = "The committee reviewed the quarterly figures and noted that the \
                  regional variance remained within the agreed tolerance band. \
                  Follow-up actions were assigned to the operations group. ";

    let mut doc = String::with_capacity(120_000);
    doc.push_str("MINUTES OF THE ANNUAL REVIEW\n\n");
    for i in 1..=400 {
        doc.push_str(&format!("Paragraph {i}. {filler}"));
    }
    // The needle, in the last paragraph — after everything a truncating
    // implementation would have thrown away.
    doc.push_str(&format!(
        "\n\nFinal item. The board approved the {BURIED_TERM} initiative and asked \
         for a progress report at the next meeting."
    ));
    doc
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("minutes");

    let mem = Memory::open(dir.path().join("minutes.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    let doc = long_document();
    println!(
        "ingesting one document: {} chars, ~{} words, marker in the LAST paragraph",
        doc.len(),
        doc.split_whitespace().count()
    );

    mem.remember(&doc)
        .in_namespace(ns.clone())
        .from_document("annual-review-2026")
        .skip_extraction()
        .await?;

    // ── Ask for the thing at the end ────────────────────────────────────────
    //
    // A truncating implementation ingests this happily and returns nothing here.
    let hits = mem
        .recall(BURIED_TERM)
        .in_namespace(ns.clone())
        .content()
        .await?;

    println!("searching for \"{BURIED_TERM}\" — found {} passage(s)", hits.len());
    if let Some(h) = hits.first() {
        println!(
            "  snippet is {} chars (document was {} chars) — contains marker: {}",
            h.snippet.len(),
            doc.len(),
            h.snippet.to_lowercase().contains(BURIED_TERM)
        );
    }

    assert!(
        !hits.is_empty(),
        "a term in the LAST paragraph of a {}-char document must still be findable. \
         Nothing came back, which means the tail of the document was dropped at \
         ingest — the document is stored but not searchable.",
        doc.len()
    );

    assert!(
        hits.iter()
            .any(|h| h.snippet.to_lowercase().contains(BURIED_TERM)),
        "the returned passage should contain the buried term itself, not merely \
         some other part of the document"
    );

    println!("\nThe end of the document is as findable as the beginning. Long content");
    println!("is split for embedding rather than truncated — the difference between");
    println!("\"it ingested fine\" and \"you can actually get it back\".");
    println!("\n⚠️  But look at the snippet size above: retrieval returned the ENTIRE");
    println!("    document, because a passage is a whole EPISODE and this was ingested");
    println!("    as one. Splitting happens for EMBEDDING, not for what you get back.");
    println!("\n    So if you are feeding a model, chunk at INGEST — one episode per");
    println!("    section — or you will put 78,000 characters into a prompt that");
    println!("    needed one paragraph.");

    mem.close().await?;
    Ok(())
}
