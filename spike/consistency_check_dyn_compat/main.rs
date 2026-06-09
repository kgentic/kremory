// Mechanical compile-spike for ADR-047 Phase A1 — verify the proposed
// run_consistency_check signature compiles with dyn-trait-object receivers
// on Rust MSRV 1.86 (kremory's declared MSRV).
//
// Replicates kremory's ChatProvider + EmbeddingProvider shapes (which use
// the async_trait macro to return Pin<Box<dyn Future + Send>> — making
// trait objects dyn-compatible by construction).
//
// SPIKE VERDICT (run via `rustc --edition 2021 spike/consistency_check_dyn_compat/main.rs`):
// - PASS = compiles → DENT-001 risk MITIGATED
// - FAIL = signature must change before Phase C impl

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ----- Shape #1: Async traits via boxed-future returns (async_trait macro
// expansion form). This is what kremory already uses for ChatProvider per
// core/provider.rs.

pub trait ChatProvider: Send + Sync {
    fn chat_with_tools<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>>;

    fn model(&self) -> &str;
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, String>> + Send + 'a>>;
}

// ----- Shape #2: ADR-047 proposed function signature for Pass 4.

pub struct ConsistencyCheckOpts {
    pub embed_prefilter_threshold: f32,
    pub max_candidates_per_run: Option<usize>,
    pub verify_model_override: Option<String>,
}

impl Default for ConsistencyCheckOpts {
    fn default() -> Self {
        Self {
            embed_prefilter_threshold: 0.6,
            max_candidates_per_run: Some(50),
            verify_model_override: None,
        }
    }
}

pub struct ConsistencyCheckSummary {
    pub scanned: usize,
    pub flagged: usize,
    pub confirmed: usize,
    pub corrected: usize,
    pub uncertain: usize,
    pub cap_overflow_dropped: usize,
}

// ----- The load-bearing signature from ADR-047 §Module Signature.

pub async fn run_consistency_check(
    embedder: &dyn EmbeddingProvider,
    llm: &dyn ChatProvider,
    opts: ConsistencyCheckOpts,
) -> Result<ConsistencyCheckSummary, String> {
    let _emb = embedder.embed("test entity").await?;
    let _resp = llm.chat_with_tools("verify this entity").await?;
    let _model = llm.model();
    let _ = opts.embed_prefilter_threshold;
    Ok(ConsistencyCheckSummary {
        scanned: 0,
        flagged: 0,
        confirmed: 0,
        corrected: 0,
        uncertain: 0,
        cap_overflow_dropped: 0,
    })
}

// ----- Verify Arc<dyn> wrapping also compiles (matches kremory ArcChatProvider
// pattern). Pass 4 may want this for sharing the LLM across spawned tasks.

pub async fn run_with_arcs(
    embedder: Arc<dyn EmbeddingProvider>,
    llm: Arc<dyn ChatProvider>,
) -> Result<(), String> {
    run_consistency_check(&*embedder, &*llm, ConsistencyCheckOpts::default()).await?;
    Ok(())
}

fn main() {
    // No runtime — pure compile-test. Silence warnings.
    let _ = run_consistency_check;
    let _ = run_with_arcs;
    println!("compile-spike: PASS (dyn-compat for ChatProvider + EmbeddingProvider on Rust 1.86 MSRV)");
}
