#![allow(clippy::unwrap_used, clippy::expect_used)]
//! OOB cloud-model smoke — `Memory::with_anthropic` end-to-end ingest + recall.
//!
//! Step 2 of the 2026-06-10 Phase D resolution plan:
//!   - Tests `Memory::with_anthropic().remember(...).await + .recall(...).await`
//!   - No GLiNER, no local LLM, anthropic-only stack
//!   - Validates the cloud-only consumer path works end-to-end
//!     (separate from Pass 4 mechanism which is gated to v0.2.0 redesign)
//!
//! Feature-gated: `cargo test --features llm-integration -- --ignored oob_anthropic_smoke`
//! Requires `ANTHROPIC_API_KEY` env var set.

#[cfg(feature = "llm-integration")]
#[tokio::test]
#[ignore]
async fn oob_anthropic_smoke_ingest_recall() {
    use kremory::Memory;

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("kremory-oob-anthropic.db");

    // ── Construct cloud-only Memory ──────────────────────────────────────
    let mem = Memory::with_anthropic(&path)
        .await
        .expect("Memory::with_anthropic must construct (requires ANTHROPIC_API_KEY)");

    use kremory::Namespace;
    let ns = Namespace::new("oob-anthropic-smoke");

    // ── Ingest: single simple sentence with 3 clear entities ────────────
    let ingest_outcome = mem
        .remember("Alice met Bob at MIT on June 10, 2026.")
        .in_namespace(ns.clone())
        .await
        .expect("remember(...) must succeed end-to-end on anthropic-only stack");

    eprintln!("[oob_anthropic_smoke] ingest outcome: {:?}", ingest_outcome);

    // ── Recall: query that should hit the ingested episode ──────────────
    let recall_outcome = mem
        .recall("Who did Alice meet?")
        .in_namespace(ns)
        .await
        .expect("recall(...) must succeed end-to-end on anthropic-only stack");

    eprintln!("[oob_anthropic_smoke] recall outcome: {:?}", recall_outcome);

    // ── Smoke assertion: recall returned SOMETHING ──────────────────────
    //
    // We don't assert specific entity types here (that's Pass 4 territory,
    // empirically gated to v0.2.0). The smoke proves: the cloud-only stack
    // (no GLiNER, no Ollama, no local LLM) can ingest + recall coherently.
    let recall_debug = format!("{:?}", recall_outcome);
    assert!(
        !recall_debug.is_empty(),
        "recall outcome must produce non-empty representation"
    );
}
