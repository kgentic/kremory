#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-143 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-143) — end-to-end
//! tests for the `MemoryBuilder::with_embed_task_prefix_enabled` seam.
//!
//! Mirrors `crates/kremory/tests/td139_fact_dense_builder_seam.rs`'s shape
//! for the sibling `fact_dense_enabled` knob: the setter reaching the live
//! `SearchConfig` (via the real public accessor `Memory::search_config()`,
//! driven through the real `MemoryBuilder::into_future` construction path —
//! not a hand-shaped model of the config), precedence over the
//! `KREMORY_EMBED_TASK_PREFIX` env override, and that leaving the knob unset
//! reproduces today's byte-identical default.
//!
//! Env-var tests mutate process env directly: safe because nextest isolates
//! every test in its own process (TD-109).

use kremory::{DynEmbeddingProvider, Memory};
use std::sync::Arc;

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td143_embed_prefix_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Clears the TD-143 env knob so a test doesn't inherit stray state from the
/// calling shell/CI environment. Safe per-test because nextest runs each
/// test in its own process (TD-109) — no cross-test leakage.
fn clear_embed_prefix_env() {
    std::env::remove_var("KREMORY_EMBED_TASK_PREFIX");
}

#[tokio::test]
async fn with_embed_task_prefix_enabled_reaches_live_search_config() {
    clear_embed_prefix_env();
    let mem = Memory::open(unique_db("setter"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_embed_task_prefix_enabled(true)
        .await
        .expect("build with with_embed_task_prefix_enabled");

    assert!(
        mem.search_config().embed_task_prefix_enabled,
        "programmatic with_embed_task_prefix_enabled(true) must reach the live SearchConfig"
    );
}

#[tokio::test]
async fn embed_task_prefix_enabled_unset_matches_documented_default() {
    clear_embed_prefix_env();
    let mem = Memory::open(unique_db("default"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("build with no search overrides");

    assert!(
        !mem.search_config().embed_task_prefix_enabled,
        "default embed_task_prefix_enabled must stay false (bare-text embedding) — \
         byte-identical to pre-TD-143"
    );
}

#[tokio::test]
async fn explicit_embed_task_prefix_enabled_wins_over_env_override() {
    // Env sets the OPPOSITE value — if precedence were wrong (env winning),
    // the assertion below catches it.
    std::env::set_var("KREMORY_EMBED_TASK_PREFIX", "true");

    let mem = Memory::open(unique_db("precedence_explicit_wins"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_embed_task_prefix_enabled(false)
        .await
        .expect("build with explicit override + conflicting env");

    assert!(
        !mem.search_config().embed_task_prefix_enabled,
        "explicit with_embed_task_prefix_enabled(false) must win over \
         KREMORY_EMBED_TASK_PREFIX=true"
    );

    clear_embed_prefix_env();
}

/// The sparse-overlay design (see `PipelineConfigOverrides` rustdoc in
/// `core/config.rs`): leaving `embed_task_prefix_enabled` UNSET while setting
/// a SIBLING knob programmatically must not clobber its own env override.
#[tokio::test]
async fn unset_embed_task_prefix_enabled_still_honours_its_own_env_override() {
    std::env::set_var("KREMORY_EMBED_TASK_PREFIX", "true");

    let mem = Memory::open(unique_db("precedence_sparse_overlay"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        // A sibling TD-141/TD-143 knob is set programmatically;
        // embed_task_prefix_enabled is left untouched — it must still pick up
        // the env override.
        .with_rrf_k(13)
        .await
        .expect("build with one explicit sibling knob + one env knob");

    let cfg = mem.search_config();
    assert_eq!(cfg.rrf_k, 13, "the explicitly-set sibling knob must apply");
    assert!(
        cfg.embed_task_prefix_enabled,
        "the UNSET embed_task_prefix_enabled knob must still honour its own env \
         override (KREMORY_EMBED_TASK_PREFIX=true) — a sibling knob being set \
         programmatically must not clobber it"
    );

    clear_embed_prefix_env();
}

/// Env-var boolean parsing accepts the documented truthy/falsy forms and
/// fails loudly (WARN + default-OFF retained) on garbage — mirrors
/// `KREMORY_EPISODE_DENSE` / `KREMORY_FACT_DENSE`'s discipline
/// (`facade::providers::search_env_overrides`).
#[tokio::test]
async fn embed_task_prefix_env_garbage_value_is_ignored_not_silently_enabled() {
    std::env::set_var("KREMORY_EMBED_TASK_PREFIX", "ture"); // typo, not "true"

    let mem = Memory::open(unique_db("garbage_env"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("build with garbage env value");

    assert!(
        !mem.search_config().embed_task_prefix_enabled,
        "a malformed KREMORY_EMBED_TASK_PREFIX value must be ignored (default OFF \
         retained), never silently enable prefixing"
    );

    clear_embed_prefix_env();
}
