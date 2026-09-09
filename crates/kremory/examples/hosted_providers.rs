//! **Costs money to run.** One line to a working memory on a hosted model — and
//! the choices that one line makes for you.
//!
//! ```text
//! export OPENAI_API_KEY=sk-...
//! cargo run --example hosted_providers
//! ```
//!
//! ⚠️ This is the ONLY example that calls a paid API. It is deliberately tiny —
//! one write and one read — but it is not free, and it is excluded from
//! `scripts/check-examples.sh` for that reason. Every other example runs offline.
//!
//! ## The problem this solves
//!
//! `Memory::open(..).with_llm(..).with_embedder(..)` is the honest, explicit
//! path, and it is a lot of ceremony when you just want to try the thing. The
//! provider constructors collapse it:
//!
//! ```text
//!   Memory::with_openai(path)            gpt-4o-mini + text-embedding-3-small
//!   Memory::with_anthropic(path)         claude-haiku + NO EMBEDDING MODEL  ← read on
//!   Memory::with_ollama_at_model(..)     local, free  (agent_memory_with_ollama.rs)
//! ```
//!
//! ## The one that will surprise you
//!
//! **Anthropic has no embedding API.** `with_anthropic` therefore wires a
//! deterministic FNV-1a hash as the "embedder", and the crate says so in its own
//! warning: *recall is structural, NOT semantic*.
//!
//! That means semantically similar text will NOT be found — "how do I get paid"
//! will not match a passage about invoicing. Everything still works; it just
//! quietly stops being smart, which is the worst way for a system to degrade.
//!
//! If you want Claude for reasoning AND real semantic recall, pair it with an
//! embedder yourself:
//!
//! ```ignore
//! Memory::open(path)
//!     .with_llm(anthropic_provider)      // Claude for extraction
//!     .with_embedder(openai_or_local)    // something that actually embeds
//!     .await?
//! ```
//!
//! The convenience constructor cannot make that choice for you, so it makes the
//! safe-but-dumb one and tells you.

use kremory::{Memory, Namespace};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fail before spending anything, and name the fix.
    if std::env::var("OPENAI_API_KEY").is_err() {
        anyhow::bail!(
            "OPENAI_API_KEY is not set.\n\
             This example calls a paid API — export the key and re-run, or try\n\
             `cargo run --example agent_memory_with_ollama` for a free local one."
        );
    }

    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("hosted");

    // One line. Chat and embeddings both wired, both hosted.
    let mem = Memory::with_openai(dir.path().join("hosted.db")).await?;
    println!("opened with OpenAI (gpt-4o-mini + text-embedding-3-small)");

    // Deliberately ONE short episode. This example demonstrates wiring, not
    // capability — every additional sentence is real money for no extra lesson.
    mem.remember("Ines runs the harbour pilot service at Lysfjord.")
        .in_namespace(ns.clone())
        .await?;
    println!("  ingested 1 episode (extraction ran on the hosted model)");

    let facts: Vec<String> = mem
        .recall("who runs the pilot service")
        .in_namespace(ns.clone())
        .raw()
        .await?
        .into_iter()
        .flat_map(|c| c.facts)
        .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
        .collect();

    println!("\n  recalled {} fact(s):", facts.len());
    for f in &facts {
        println!("    {f}");
    }

    // Worth noticing in the output above: a hosted model produces the SAME
    // case-variant duplicates a small local one does — "Ines works_at lysfjord"
    // alongside "ines works_at Lysfjord". Paying more per token buys better
    // triples, not canonical casing. `dream()` is what reconciles them.
    assert!(
        !facts.is_empty(),
        "a hosted-model ingest should produce at least one extracted fact — if \
         this is empty, extraction returned nothing and the graph is empty"
    );

    println!("\nOne line got you a working memory on a hosted model. The trade is that");
    println!("the constructor picked the models — fine for trying it, and worth");
    println!("replacing with explicit .with_llm()/.with_embedder() once you care");
    println!("which model, which region, or what it costs per call.");

    mem.close().await?;
    Ok(())
}
