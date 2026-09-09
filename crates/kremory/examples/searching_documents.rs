//! **Runs offline.** You have documents. Make them searchable, and get back the
//! passage rather than the whole file.
//!
//! ```text
//! cargo run --example searching_documents
//! ```
//!
//! ## The problem this solves
//!
//! Most "give the model some context" problems are not graph problems. You have
//! a pile of notes, tickets or pages, and you want the two paragraphs that
//! actually bear on the question — not the top-ranked whole document, and not
//! all of them.
//!
//! `recall(..).content()` answers that directly: it returns `ContentPassage`
//! values — an `episode_id`, the matching text, and a `score` — instead of the
//! entity-and-fact shape the graph surface returns.
//!
//! ⚠️ **`snippet` is the whole episode body, not an extract.** The crate
//! documents this and a rename is tracked. It matters for how you INGEST: this
//! example stores one handbook entry per episode, so a result is
//! paragraph-sized. Store a whole PDF as one episode and a "passage" is the
//! whole PDF. Chunk at ingest if you want passage-sized results.
//!
//! ## This is the DEFAULT path, which is exactly why it has an example
//!
//! `content-search` is on by default (`default = ["content-search"]`). A previous
//! release changed behaviour on this path and moved measured recall by 32
//! percentage points while every one of ~1,800 tests stayed green — nothing
//! exercised it as a consumer would. This example does.
//!
//! ## About the ranking in this example
//!
//! The stand-in embedder below hashes bytes; it has no idea what words mean. So
//! the assertion here deliberately relies on the LEXICAL half of the search
//! (full-text matching on a distinctive term), which is deterministic and true
//! regardless of embedding quality. With a real embedder you additionally get
//! semantic matches — "how do I get paid" finding a passage about invoicing —
//! and `agent_memory_with_ollama.rs` shows that setup.

use std::future::Future;
use std::sync::Arc;

use kremory::{
    CoreResult, EmbeddingProvider, EntityExtractor, ExtractionContext, ExtractionResult, Memory,
    Namespace,
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

/// Never invoked — these writes call `.skip_extraction()`. A document search
/// does not need a graph built from the text; that is a separate capability.
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

/// A small handbook. Each entry is one episode.
const HANDBOOK: &[(&str, &str)] = &[
    (
        "expenses",
        "Expenses are reimbursed monthly. Submit receipts through the finance portal \
         before the last working day of the month. Anything over five hundred pounds \
         needs written approval from your line manager first.",
    ),
    (
        "onboarding",
        "New starters get a laptop on day one and access to the shared drive within \
         48 hours. Your buddy walks you through the deployment runbook in week one.",
    ),
    (
        "incidents",
        "During an incident the on-call engineer owns communication. Post updates in \
         the status channel every thirty minutes, even when there is nothing new to \
         report. Silence reads as an outage getting worse.",
    ),
    (
        "leave",
        "Annual leave is booked in the HR system at least two weeks ahead. Carry-over \
         is capped at five days and expires at the end of March.",
    ),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ns = Namespace::new("handbook");

    let mem = Memory::open(dir.path().join("handbook.db"))
        .embedding_dim(DEMO_DIM)
        .default_namespace(ns.clone())
        .with_embedder(Arc::new(DemoEmbedder { dim: DEMO_DIM }))
        .with_extractor(Arc::new(NoExtraction))
        .await?;

    for (name, body) in HANDBOOK {
        mem.remember(*body)
            .in_namespace(ns.clone())
            .from_document(*name)
            .skip_extraction()
            .await?;
    }
    println!("indexed {} handbook entries\n", HANDBOOK.len());

    // ── The search ──────────────────────────────────────────────────────────
    //
    // `.content()` changes the RETURN SHAPE: passages, not entities and facts.
    // Same query surface, different answer type, chosen by the caller.
    let hits = mem
        .recall("what do I do during an incident")
        .in_namespace(ns.clone())
        .content()
        .await?;

    // ⚠️ The score is BM25 from SQLite's FTS5 `rank` column, where LOWER is MORE
    // relevant — so it is negative, and the best match is the most negative.
    // That is documented and consistent with the rest of the crate's FTS
    // scores, but it inverts the usual convention: sorting DESCENDING by
    // `score` gives you the WORST match first. Results arrive already ordered;
    // re-sort only if you mean to, and sort ASCENDING.
    println!("query: \"what do I do during an incident\"  (lower score = better match)");
    for h in &hits {
        let preview: String = h.snippet.chars().take(72).collect();
        println!("  [score {:>6.3}] episode {} — {}…", h.score, h.episode_id, preview);
    }

    assert!(
        !hits.is_empty(),
        "content search returned nothing — with content-search enabled (the default) \
         an indexed corpus must be searchable"
    );

    // The distinctive term "incident" appears in exactly one entry, so the top
    // hit is deterministic under lexical matching and does not depend on the
    // stand-in embedder understanding anything.
    let top = &hits[0];
    assert!(
        top.snippet.to_lowercase().contains("incident"),
        "the top passage should be the incident entry; got {:?}",
        top.snippet.chars().take(60).collect::<String>()
    );

    println!("\n⚠️  `snippet` is the WHOLE EPISODE, not an extract. The field name is");
    println!("    misleading and the crate says so — a rename is a tracked follow-up.");
    println!("    Here each entry is a paragraph, so a passage IS paragraph-sized. Ingest");
    println!("    one 80-page PDF as ONE episode and you get 80 pages back.");
    println!("\n    So for retrieval-augmented prompting, CHUNK AT INGEST: one episode per");
    println!("    section, not per document. The producer decides passage size, not the");
    println!("    query. See `a_long_document.rs` for what happens when you do not.");
    println!("No graph was built here: `.skip_extraction()` means these are documents,");
    println!("not entities. Searching text and building a knowledge graph are separate");
    println!("capabilities, and you can use either on its own.");

    mem.close().await?;
    Ok(())
}
