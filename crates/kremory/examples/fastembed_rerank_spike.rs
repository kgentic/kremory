//! TD-062 mandatory Phase-0 spike (spec §5.2,
//! `.ai-docs/specs/td-066-recall-scoring-foundation-spec-2026-07-21.md`).
//!
//! Throwaway compile+runtime spike — NOT part of the `rerank` feature's
//! production wiring. Run BEFORE writing any `Reranker` trait / wiring code:
//!
//! ```sh
//! cargo run -p kremory --features rerank --example fastembed_rerank_spike
//! ```
//!
//! First run downloads the BGE reranker's ONNX weights from HF Hub (network +
//! a few hundred MB) — may take minutes. Subsequent runs hit the local HF Hub
//! cache and are fast (warm-call timing printed below).

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

fn main() -> anyhow::Result<()> {
    let cold_start = std::time::Instant::now();
    let mut model = TextRerank::try_new(RerankInitOptions::new(RerankerModel::BGERerankerBase))?;
    let cold_start_secs = cold_start.elapsed().as_secs_f64();

    let query = "What is the capital of France?";
    let docs = vec![
        "Paris is the capital of France.",
        "Berlin is a city in Germany.",
    ];

    let warm_call = std::time::Instant::now();
    // `return_documents = true` — the default `false` (tried first) omits
    // `RerankResult.document` (always `None`), so results must be identified
    // by `index` into the original `docs` slice, not by document text.
    let results = model.rerank(query, docs, true, None)?;
    let warm_call_secs = warm_call.elapsed().as_secs_f64();

    println!("cold_start_secs={cold_start_secs:.3}");
    println!("warm_call_secs={warm_call_secs:.3}");
    println!("{results:?}");

    let france_doc_score = results
        .iter()
        .find(|r| r.index == 0)
        .map(|r| r.score)
        .ok_or_else(|| anyhow::anyhow!("France doc (index 0) missing from rerank results"))?;
    let germany_doc_score = results
        .iter()
        .find(|r| r.index == 1)
        .map(|r| r.score)
        .ok_or_else(|| anyhow::anyhow!("Germany doc (index 1) missing from rerank results"))?;

    anyhow::ensure!(
        france_doc_score > germany_doc_score,
        "SPIKE FAIL: France doc ({france_doc_score}) did not outscore Germany doc ({germany_doc_score})"
    );

    println!("SPIKE PASS: France doc ({france_doc_score}) > Germany doc ({germany_doc_score})");
    Ok(())
}
