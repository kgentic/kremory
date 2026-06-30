//! TD-080 #2 root-cause spike — embedder anisotropy check.
//!
//! Tests the hypothesis that the e2e's nomic-embed-text embedder (used RAW, without
//! nomic's required `search_document:` / `search_query:` task prefix) produces highly
//! anisotropic embeddings where EVERY pair of unrelated proper nouns lands at ~0.9+
//! cosine. If so, kremory's L4 entity-disambiguation merge threshold (0.95) is cleared
//! by unrelated entities → all entities collapse to one canonical id → fact subjects
//! corrupt (the observed `amazon robotics` subject on every fact).
//!
//! Prints the pairwise cosine matrix for the mock_interview entity names so we can see
//! the baseline cosine floor. Also prints the same with nomic's `search_document:` prefix
//! to show the fix direction.
//!
//!   OLLAMA_BASE_URL=http://localhost:11434 cargo test -p kremory --features llm-integration \
//!     --test spike_td080_embedder_cosine -- --ignored --nocapture

#![cfg(feature = "llm-integration")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::embedding::{EmbeddingBuilder, EmbeddingProvider};

fn base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

async fn embed_all(emb: &Arc<Ollama>, texts: &[String]) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for t in texts {
        let mut v = EmbeddingProvider::embed(&**emb, vec![t.clone()])
            .await
            .expect("embed");
        out.push(v.pop().expect("one vec"));
    }
    out
}

fn print_matrix(label: &str, names: &[&str], vecs: &[Vec<f32>]) {
    eprintln!("\n[embed-spike] === {label} ===");
    let mut max_off_diag = 0.0f32;
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let c = cosine(&vecs[i], &vecs[j]);
            if c > max_off_diag {
                max_off_diag = c;
            }
            eprintln!(
                "[embed-spike]   cos({:>22}, {:>22}) = {:.4}",
                names[i], names[j], c
            );
        }
    }
    eprintln!(
        "[embed-spike] MAX off-diagonal cosine = {max_off_diag:.4}  (L4_MERGE_THRESHOLD=0.95 → {} unrelated entities merge)",
        if max_off_diag >= 0.95 { "YES, BUG:" } else { "no:" }
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn nomic_raw_vs_prefixed_anisotropy() {
    let emb: Arc<Ollama> = EmbeddingBuilder::<Ollama>::new()
        .base_url(base_url())
        .model("nomic-embed-text")
        .build()
        .expect("ollama embedder");

    let names = [
        "Ria",
        "Morocco",
        "Boston",
        "Northeastern University",
        "Amazon Robotics",
        "Boston Consulting Group",
    ];

    // RAW (what kremory's embedder does today — bare entity name).
    let raw: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    let raw_vecs = embed_all(&emb, &raw).await;
    print_matrix(
        "RAW (bare name — current kremory behaviour)",
        &names,
        &raw_vecs,
    );

    // PREFIXED (nomic's documented `search_document:` task prefix).
    let prefixed: Vec<String> = names
        .iter()
        .map(|s| format!("search_document: {s}"))
        .collect();
    let pre_vecs = embed_all(&emb, &prefixed).await;
    print_matrix(
        "PREFIXED (search_document: — fix direction)",
        &names,
        &pre_vecs,
    );

    // NAME+TYPE (minimal context — what `name (Label)` would give).
    let nametype = [
        "Ria (Person)",
        "Morocco (Location)",
        "Boston (Location)",
        "Northeastern University (Organisation)",
        "Amazon Robotics (Organisation)",
        "Boston Consulting Group (Organisation)",
    ];
    let nt: Vec<String> = nametype.iter().map(|s| s.to_string()).collect();
    let nt_vecs = embed_all(&emb, &nt).await;
    print_matrix("NAME+TYPE (name (Label))", &names, &nt_vecs);

    // CONTEXT SENTENCE (what embedding the ±100-char context snippet would give —
    // candidate B: embed CONTEXT not bare name).
    let context = [
        "Ria grew up in Morocco and studied at Northeastern University.",
        "Morocco is a country in North Africa where Ria grew up.",
        "Boston is a city in Massachusetts.",
        "Northeastern University is a university located in Boston.",
        "Amazon Robotics is a robotics company where Ria interned.",
        "Boston Consulting Group is a management consulting firm.",
    ];
    let ctx: Vec<String> = context.iter().map(|s| s.to_string()).collect();
    let ctx_vecs = embed_all(&emb, &ctx).await;
    print_matrix(
        "CONTEXT SENTENCE (candidate B — embed context not name)",
        &names,
        &ctx_vecs,
    );
}
