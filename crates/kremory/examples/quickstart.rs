//! Verified-compiling quickstart: BYOE (custom embedder) + BYOM (Ollama LLM).
//!
//! This example exists primarily as a **drift guard**: `cargo build --example
//! quickstart` (run by `cargo test`) compiles it in CI, so the snippets in
//! `README.md` cannot silently fall out of sync with the real public API.
//!
//! To run it for real you need a live Ollama at `http://localhost:11434` with
//! `gemma4-e2b:latest` pulled:
//!
//! ```text
//! cargo run --example quickstart
//! ```

use std::future::Future;
use std::sync::Arc;

use autoagents_llm::{backends::ollama::Ollama, builder::LLMBuilder};
use kremory::{CoreResult, EmbeddingProvider, Memory, Namespace};

/// Dimension of our demo embeddings. kremory does NOT read the dimension from
/// the trait — it defaults to 384, so a custom embedder of any other size must
/// declare it via `.embedding_dim(N)` (see `main` below).
const DEMO_DIM: usize = 16;

/// BYOE — a genuine custom embedder.
///
/// This one is a deterministic, non-semantic bag-of-bytes hash so the example
/// needs no network embedding model. A real consumer calls their embedding
/// backend (OpenAI, Ollama `nomic-embed-text`, a local GGUF, sentence-transformers
/// over HTTP, …) inside `embed` and returns the vector.
///
/// Note the trait shape: a single `embed(&str) -> Vec<f32>` using RPITIT
/// (`impl Future`), so **no `#[async_trait]` is required**.
struct DemoEmbedder {
    dim: usize,
}

impl EmbeddingProvider for DemoEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Future<Output = CoreResult<Vec<f32>>> + Send + 'a {
        // Capture Copy fields before the `async move` block: this matches
        // kremory's own embedder impls and keeps clippy's manual_async_fn happy.
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // BYOM — bring your own chat provider. Here: a real Ollama backend built via
    // autoagents-llm (which a consumer must add as a direct dependency to use
    // `with_llm`). 120s timeout — interactive extraction models take ~37-54s.
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url("http://localhost:11434")
        .model("gemma4-e2b:latest")
        .timeout_seconds(120)
        .build()?;

    // Tier 2 builder — type-state guarded: `.await` won't compile until both
    // `.with_llm()` and `.with_embedder()` are set.
    let mem = Memory::open("./quickstart.db")
        .embedding_dim(DEMO_DIM) // required: our embedder is 16-dim, not the 384 default
        .default_namespace(Namespace::new("quickstart"))
        .with_llm(llm)
        // Tell kremory which model you wired so it picks the right structured-output
        // strategy (Ollama → FormatSchema). Without this the raw `with_llm` path
        // falls back to prompt-only extraction. Must match the LLMBuilder `.model()`.
        .with_model_id("gemma4-e2b:latest")
        .with_embedder(DemoEmbedder { dim: DEMO_DIM }.into_dyn())
        .await?;

    // Ingest — runs real LLM extraction, then recall returns prompt-ready text.
    mem.remember("Jim prefers concise replies and writes Rust.")
        .await?;
    let ctx = mem.recall("what language does Jim use?").await?;
    println!("recall => {ctx}");

    mem.close().await?;
    Ok(())
}
