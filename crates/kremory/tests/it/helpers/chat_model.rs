//! Per-test Ollama chat-model selection for the `llm-integration` tier.
//!
//! ## The defect this exists to close
//!
//! Before this helper, ~28 call sites across ~25 test files each did:
//!
//! ```ignore
//! // (spelled out rather than quoted verbatim: the sweep that migrated the 29
//! // call sites to this helper also rewrote this illustration, which is a small
//! // reminder that a mechanical rename does not read doc comments for intent)
//! std::env::var(OLLAMA_CHAT_MODEL).unwrap_or_else(|_| "qwen2.5:14b".to_string())
//! ```
//!
//! — the SAME global env var, with **three different** hardcoded defaults
//! (`gemma4:e4b` ×14, `gemma4-e2b:latest` ×7, `qwen2.5:14b` ×7). Those defaults
//! are not arbitrary: `facade_fact_extraction_e2e` documents `qwen2.5:14b` as
//! "the proven fact extractor", while the dream suite is tuned for `gemma4:e4b`.
//!
//! The consequence is that **there was no way to set the model for one test
//! without silently changing it for every other test**, and nothing anywhere
//! said which model a given test had actually run with. Setting the var to
//! satisfy one file quietly mis-configures ~20 others, and the resulting
//! failures look exactly like product defects.
//!
//! That is not hypothetical — it produced **two wrong diagnoses in one session**
//! (2026-08-11), and is recorded in `.ai-docs/plans/paid-bench-readiness-target-
//! 2026-08-11.md` as the structural reason the `llm-integration` tier was never
//! wired into the standard gate: *"No single `OLLAMA_CHAT_MODEL` satisfies all
//! 26."*
//!
//! ## How to run the tier correctly
//!
//! **Leave `OLLAMA_CHAT_MODEL` UNSET.** Each test then gets its own documented
//! default, which is the configuration its assertions were written against.
//! The env var is a *sweep* knob — "force every test onto model X to compare
//! models" — not a *fix* knob. Using it to make one test pass breaks the rest.
//!
//! ## What this helper adds
//!
//! It cannot stop someone exporting the var, so instead it makes the
//! consequence **impossible to miss**, per [[observability-first-class]]: the
//! information needed to diagnose the failure was simply never emitted.
//!
//! 1. An override that DIFFERS from the call site's documented default prints a
//!    loud, once-per-process banner naming both models and the file that
//!    disagreed — so a red test reports *why* it is red.
//! 2. `KREMORY_TEST_MODEL_STRICT=1` upgrades that warning to a panic, for gate
//!    runs where a silently mis-modelled pass is worse than a hard stop.
//!
//! Deliberately NOT done: reading a per-file env var like
//! `OLLAMA_CHAT_MODEL_LLM_INTEGRATION`. That is a new public surface for a
//! problem the existing primitives already solve — the documented default IS
//! the per-test setting, and it is already correct at all 28 sites
//! ([[contract-first-before-new-public-surface]]: prefer derive/observe over
//! declare).

use std::sync::atomic::{AtomicBool, Ordering};

/// Set once the mismatch banner has been printed, so a 26-test run emits it
/// once rather than 26 times. `Relaxed` is right: this is a print-once latch
/// with no other memory being published through it, and a duplicated banner
/// under a race would be harmless anyway.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Resolve the Ollama chat model for a test whose documented, assertion-bearing
/// default is `documented_default`.
///
/// Precedence: `OLLAMA_CHAT_MODEL` (if set) > `documented_default`. Identical
/// to the ~28 hand-rolled call sites this replaces — the behaviour is unchanged
/// on purpose, so migrating a call site cannot alter which model it runs.
/// What changes is that a divergence is now *reported* instead of silent.
///
/// # Panics
///
/// When `KREMORY_TEST_MODEL_STRICT=1` and the env override differs from
/// `documented_default`. That is the intended failure mode for a gate run: a
/// test passing under a model its assertions were not written for is a false
/// green, which is worse than a stop.
pub fn chat_model_or(documented_default: &str) -> String {
    let Ok(override_model) = std::env::var("OLLAMA_CHAT_MODEL") else {
        return documented_default.to_string();
    };

    if override_model == documented_default {
        return override_model;
    }

    let strict = std::env::var("KREMORY_TEST_MODEL_STRICT")
        .map(|v| v == "1")
        .unwrap_or(false);

    if strict {
        panic!(
            "KREMORY_TEST_MODEL_STRICT=1 and OLLAMA_CHAT_MODEL={override_model} overrides this \
             test's documented default {documented_default}. The default is the model this \
             test's assertions were written against; overriding it produces failures that look \
             like product defects. Unset OLLAMA_CHAT_MODEL to run the tier as designed."
        );
    }

    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "\n\
             ┌─ llm-integration MODEL OVERRIDE ────────────────────────────────\n\
             │ OLLAMA_CHAT_MODEL = {override_model}\n\
             │ this call site's documented default = {documented_default}\n\
             │\n\
             │ The tier's ~25 files have THREE different documented defaults,\n\
             │ each chosen for the assertions in that file. One global override\n\
             │ cannot satisfy all of them — any failure below may be a\n\
             │ CONFIGURATION artefact, not a defect. Bisect before believing it.\n\
             │\n\
             │ To run the tier as designed: unset OLLAMA_CHAT_MODEL.\n\
             │ To make this a hard error:   KREMORY_TEST_MODEL_STRICT=1\n\
             └─────────────────────────────────────────────────────────────────\n"
        );
    }

    override_model
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Env-var mutation is safe per-test because nextest isolates every test in
    /// its own process (TD-109) — the same precedent the TD-141 builder-seam
    /// tests rely on.
    #[test]
    fn unset_env_yields_the_documented_default() {
        std::env::remove_var("OLLAMA_CHAT_MODEL");
        assert_eq!(chat_model_or("qwen2.5:14b"), "qwen2.5:14b");
    }

    #[test]
    fn env_override_wins_preserving_pre_migration_behaviour() {
        std::env::set_var("OLLAMA_CHAT_MODEL", "gemma4:e4b");
        assert_eq!(chat_model_or("qwen2.5:14b"), "gemma4:e4b");
        std::env::remove_var("OLLAMA_CHAT_MODEL");
    }

    /// The non-vacuity partner of the test above: an override EQUAL to the
    /// documented default must be silent and must not trip strict mode. Without
    /// this, a `chat_model_or` that always panicked in strict mode would still
    /// pass the panic test below.
    #[test]
    fn matching_override_is_not_a_mismatch_even_in_strict_mode() {
        std::env::set_var("OLLAMA_CHAT_MODEL", "qwen2.5:14b");
        std::env::set_var("KREMORY_TEST_MODEL_STRICT", "1");
        assert_eq!(chat_model_or("qwen2.5:14b"), "qwen2.5:14b");
        std::env::remove_var("OLLAMA_CHAT_MODEL");
        std::env::remove_var("KREMORY_TEST_MODEL_STRICT");
    }

    #[test]
    #[should_panic(expected = "KREMORY_TEST_MODEL_STRICT=1")]
    fn strict_mode_rejects_a_mismatched_override() {
        std::env::set_var("OLLAMA_CHAT_MODEL", "gemma4:e4b");
        std::env::set_var("KREMORY_TEST_MODEL_STRICT", "1");
        let _ = chat_model_or("qwen2.5:14b");
    }
}
