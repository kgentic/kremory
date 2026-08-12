#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-139 (`.ai-docs/tech-debt/tech-debt-register.md` §TD-139 DoD item 2) —
//! end-to-end tests for the `MemoryBuilder::with_fact_dense_enabled` seam.
//!
//! Mirrors `crates/kremory/tests/td141_search_config_builder_seam.rs`'s shape
//! for the sibling `content_stream_weight`/`rrf_k`/`episode_dense_enabled`
//! knobs: the setter reaching the live `SearchConfig` (via the real public
//! accessor `Memory::search_config()`, driven through the real
//! `MemoryBuilder::into_future` construction path — not a hand-shaped model
//! of the config), precedence over the `KREMORY_FACT_DENSE` env override, and
//! that leaving the knob unset reproduces today's byte-identical default.
//!
//! Env-var tests mutate process env directly: safe because nextest isolates
//! every test in its own process (TD-109).

use kremory::{DynEmbeddingProvider, Memory};
use std::sync::Arc;

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_td139_fact_dense_{}_{}_{}.db",
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

/// Clears the TD-139 env knob so a test doesn't inherit stray state from the
/// calling shell/CI environment. Safe per-test because nextest runs each
/// test in its own process (TD-109) — no cross-test leakage.
fn clear_fact_dense_env() {
    std::env::remove_var("KREMORY_FACT_DENSE");
}

#[tokio::test]
async fn with_fact_dense_enabled_reaches_live_search_config() {
    clear_fact_dense_env();
    let mem = Memory::open(unique_db("setter"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_fact_dense_enabled(true)
        .await
        .expect("build with with_fact_dense_enabled");

    assert!(
        mem.search_config().fact_dense_enabled,
        "programmatic with_fact_dense_enabled(true) must reach the live SearchConfig"
    );
}

#[tokio::test]
async fn fact_dense_enabled_unset_matches_documented_default() {
    clear_fact_dense_env();
    let mem = Memory::open(unique_db("default"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("build with no search overrides");

    assert!(
        !mem.search_config().fact_dense_enabled,
        "default fact_dense_enabled must stay false (1-hop-only) — byte-identical to \
         pre-TD-139"
    );
}

#[tokio::test]
async fn explicit_fact_dense_enabled_wins_over_env_override() {
    // Env sets the OPPOSITE value — if precedence were wrong (env winning),
    // the assertion below catches it.
    std::env::set_var("KREMORY_FACT_DENSE", "true");

    let mem = Memory::open(unique_db("precedence_explicit_wins"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .with_fact_dense_enabled(false)
        .await
        .expect("build with explicit override + conflicting env");

    assert!(
        !mem.search_config().fact_dense_enabled,
        "explicit with_fact_dense_enabled(false) must win over KREMORY_FACT_DENSE=true"
    );

    clear_fact_dense_env();
}

/// The sparse-overlay design (see `PipelineConfigOverrides` rustdoc in
/// `core/config.rs`): leaving `fact_dense_enabled` UNSET while setting a
/// SIBLING knob programmatically must not clobber `fact_dense_enabled`'s own
/// env override.
#[tokio::test]
async fn unset_fact_dense_still_honours_its_own_env_override() {
    std::env::set_var("KREMORY_FACT_DENSE", "true");

    let mem = Memory::open(unique_db("precedence_sparse_overlay"))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        // A sibling TD-141 knob is set programmatically; fact_dense_enabled
        // is left untouched — it must still pick up the env override.
        .with_rrf_k(13)
        .await
        .expect("build with one explicit sibling knob + one env knob");

    let cfg = mem.search_config();
    assert_eq!(cfg.rrf_k, 13, "the explicitly-set sibling knob must apply");
    assert!(
        cfg.fact_dense_enabled,
        "the UNSET fact_dense_enabled knob must still honour its own env override \
         (KREMORY_FACT_DENSE=true) — a sibling knob being set programmatically must not \
         clobber it"
    );

    clear_fact_dense_env();
}
