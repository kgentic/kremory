//! **Start here.** Save something, get it back — offline, in about thirty seconds.
//!
//! ```text
//! cargo run --example offline_remember_recall
//! ```
//!
//! No Ollama. No API keys. No environment variables. No network. Nothing to install
//! beyond the crate itself, and nothing left on disk when it exits.
//!
//! This is the smallest useful kremory program. Once it makes sense, read
//! `remembers_across_sessions.rs`, which shows the part no other memory store does:
//! answering "where does Alice live?" and "where did she live in March?" correctly
//! at the same time.
//!
//! ## Why an "extractor" appears in a program that extracts nothing
//!
//! kremory's normal job is to read prose and work out the facts itself, which needs a
//! language model. Here we supply the facts directly and call `.skip_extraction()`,
//! so no model is needed — but the builder still requires you to say HOW facts would
//! be extracted. Hence the do-nothing extractor below. It is never called.

use std::future::Future;
use std::sync::Arc;

use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace, StructuredFact};

const DEMO_DIM: usize = 16;

/// A deterministic stand-in so this example needs no embedding service.
/// **Not for production** — it hashes bytes, it does not understand meaning.
/// A real consumer calls their embedding backend here (OpenAI, Ollama
/// `nomic-embed-text`, a local GGUF, sentence-transformers over HTTP).
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

/// Never invoked — see the module doc. Wiring a real one is the "bring your own
/// extractor" path: return the entities and facts you found, and kremory stores them.
struct NoExtraction;

impl EntityExtractor for NoExtraction {
    fn name(&self) -> &'static str {
        "no-extraction"
    }

    // `async fn` rather than `-> impl Future`: the trait is AFIT-shaped, so this is
    // the simpler spelling and the one clippy's `manual_async_fn` asks for.
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
    let ns = Namespace::new("demo");

    let mem = Memory::open(dir.path().join("memory.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    // Save two things we know about a user.
    mem.remember("Notes about Ada.")
        .in_namespace(ns.clone())
        .with_facts(vec![
            StructuredFact {
                subject: "jim".into(),
                predicate: "writes".into(),
                object: "Rust".into(),
                valid_from: None, // None = "true as of now"
                valid_to: None,   // None = "still true"
                memory_type: None,
            },
            StructuredFact {
                subject: "jim".into(),
                predicate: "prefers".into(),
                object: "concise replies".into(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            },
        ])
        .skip_extraction()
        .await?;

    // Get them back. `.raw()` gives you the structured facts; a bare `.recall(..)`
    // gives you a rendered String ready to drop into a prompt instead.
    let facts: Vec<_> = mem
        .recall("what do we know about jim")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .collect();

    for f in &facts {
        println!("{} {} {}", f.subject, f.predicate, f.object);
    }

    assert!(
        !facts.is_empty(),
        "expected to recall the facts we just stored. If this fails with KREMORY_* \
         environment variables set, unset them — they override recall scoring."
    );

    // And the prompt-ready rendering, which is what most agents actually want.
    let rendered = mem
        .recall("what do we know about jim")
        .in_namespace(ns)
        .await?;
    println!("\n--- prompt-ready ---\n{rendered}");

    mem.close().await?;
    Ok(())
}
