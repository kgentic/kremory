//! TD-080 #2 root-cause spike — Stage-3 triplet self-loop rate, by model.
//!
//! PROVES the diagnosis in `.ai-docs/research/td080-self-loop-root-cause-investigate-2026-06-29.md`:
//! `build_triplet_prompt` (graphiti.rs:312) never forbids reflexive triples, so weak models
//! (gemma4:e4b) emit `X --pred--> X`. These collapse to `subject_id == object_id` at resolution
//! (deferred.rs:193-207, via `normalize_name`) and the P3 guard (facts.rs:279) rejects every one
//! → 0 persisted facts.
//!
//! This spike calls `IntegerIdLlmExtractor::extract()` DIRECTLY (all 3 stages) and observes the
//! RAW Stage-3 triples BEFORE the P3 guard, counting self-loops where
//! `normalize_name(subject) == normalize_name(object)` (the exact H1 signature).
//!
//! Run per model to answer "prompt gap vs small-model weakness?":
//!   OLLAMA_CHAT_MODEL=qwen2.5:14b  cargo test -p kremory --features llm-integration \
//!     --test it spike_td080_self_loop:: -- --ignored --nocapture
//!   OLLAMA_CHAT_MODEL=gemma4:e4b   cargo test -p kremory --features llm-integration \
//!     --test it spike_td080_self_loop:: -- --ignored --nocapture
//!
//! ORACLE: `self_loops == 0`. FAILS on a model that emits reflexive triples under the CURRENT
//! prompt (proving H1 is reachable); should PASS for that same model after the anti-reflexive
//! prompt rule lands.

#![cfg(feature = "llm-integration")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::builder::LLMBuilder;

use kremory::core::config::ContentType;
use kremory::core::extraction::IntegerIdLlmExtractor;
use kremory::core::intelligence::{EntityExtractor, ExtractionContext};

/// Local replica of `core::resolver::normalize_name` (that fn is `pub(crate)` —
/// invisible to integration tests). MUST stay in sync; the self-loop guard at
/// facts.rs:279 compares ids derived from this exact normalization (deferred.rs:194).
fn normalize_name(s: &str) -> String {
    s.to_lowercase()
        .trim()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}
fn chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "qwen2.5:14b".to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn stage3_triplets_have_no_self_loops() {
    let model = chat_model();
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url(base_url())
        .model(model.clone())
        .reasoning(false)
        .keep_alive("1h")
        .timeout_seconds(300)
        .build()
        .expect("ollama chat provider");

    let extractor = IntegerIdLlmExtractor::new(llm);

    let corpus = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kremory-eval/fixtures/mock_interview.txt"),
    )
    .expect("read mock_interview.txt");

    // Production-faithful context: model set (selects the Ollama FormatSchema arm — without it
    // capability detection picks PromptOnly and stages 2-3 degrade) + generous arm budget for
    // slow local models. Empty registry is fine: it never touches `build_triplet_prompt`.
    let ctx = ExtractionContext {
        content_type: ContentType::Message,
        arm_budget_ms: 300_000,
        model: Some(model.as_str()),
        ..Default::default()
    };

    let result = extractor
        .extract(&corpus, &ctx)
        .await
        .expect("extract must not error");

    let mut self_loops = 0usize;
    eprintln!(
        "[td080-spike] model={model} entities={} raw_facts={}",
        result.entities.len(),
        result.facts.len()
    );
    for f in &result.facts {
        let is_self_loop =
            f.is_entity_ref && normalize_name(&f.subject) == normalize_name(&f.object);
        if is_self_loop {
            self_loops += 1;
        }
        eprintln!(
            "[td080-spike]   {}{} --{}--> {} (entity_ref={})",
            if is_self_loop { "SELF-LOOP " } else { "" },
            f.subject,
            f.predicate,
            f.object,
            f.is_entity_ref
        );
    }
    let total = result.facts.len();
    let rate = if total == 0 {
        0.0
    } else {
        self_loops as f64 / total as f64
    };
    eprintln!(
        "[td080-spike] RESULT model={model} self_loops={self_loops}/{total} ({:.0}%)",
        rate * 100.0
    );

    assert_eq!(
        self_loops, 0,
        "model {model} emitted {self_loops}/{total} reflexive Stage-3 triples (subject==object \
         after normalize_name). These collapse to subject_id==object_id and are rejected by the \
         P3 guard, yielding 0 persisted facts. Root cause: build_triplet_prompt has no \
         anti-reflexive rule (graphiti.rs:312)."
    );
}
