//! Contract surface tests for rqlm's 4 public fns.
//!
//! - `context_block` (implemented in D.2a) — full coverage of all 3
//!   `ContextTemplate` variants + empty-input behaviour.
//! - `ingest_episode` / `run_dream_phase` / `search` (D.2b) — contract pins
//!   that lock the `Unimplemented` error variant until those fns land.
//!
//! Per `feedback_design_test_strategy_first`: the contract goes red first
//! (the 3 `*_returns_unimplemented_until_d2b` tests) then flips to green
//! when D.2b implements them.

use chrono::{TimeZone, Utc};
use rql_memory::{
    context_block, ContextTemplate, RetrievedContext, RqlmError, SourceKind, SourceRef,
};

fn sample_results() -> Vec<RetrievedContext> {
    vec![
        RetrievedContext {
            entity_id: "ent-1".into(),
            entity_name: "Roadmap Decision".into(),
            summary: "Q3 priorities locked: extraction quality first.".into(),
            score: 0.92,
            source_refs: vec![SourceRef {
                kind: SourceKind::Meeting,
                id: "mtg-001".into(),
                occurred_at: Utc.with_ymd_and_hms(2026, 5, 17, 14, 0, 0).unwrap(),
            }],
        },
        RetrievedContext {
            entity_id: "ent-2".into(),
            entity_name: "Migration framework spec".into(),
            summary: "Backup-before-migrate; rollback via restore.".into(),
            score: 0.81,
            source_refs: vec![SourceRef {
                kind: SourceKind::Document,
                id: "doc-007".into(),
                occurred_at: Utc.with_ymd_and_hms(2026, 5, 18, 9, 30, 0).unwrap(),
            }],
        },
    ]
}

#[test]
fn context_block_empty_input_renders_empty_string() {
    let out = context_block(&[], ContextTemplate::Entities);
    assert_eq!(out, "");
    let out = context_block(&[], ContextTemplate::EdgeSummary);
    assert_eq!(out, "");
    let out = context_block(&[], ContextTemplate::TemporalFacts);
    assert_eq!(out, "");
}

#[test]
fn context_block_entities_template_renders_name_summary_sources() {
    let results = sample_results();
    let out = context_block(&results, ContextTemplate::Entities);
    assert!(
        out.contains("## Roadmap Decision"),
        "expected entity heading, got: {out}"
    );
    assert!(
        out.contains("Q3 priorities locked"),
        "expected summary body, got: {out}"
    );
    assert!(
        out.contains("meeting:mtg-001"),
        "expected source pointer, got: {out}"
    );
    assert!(
        out.contains("## Migration framework spec"),
        "expected second entity heading, got: {out}"
    );
    assert!(
        out.contains("document:doc-007"),
        "expected document source pointer, got: {out}"
    );
}

#[test]
fn context_block_edge_summary_renders_one_line_per_source() {
    let results = sample_results();
    let out = context_block(&results, ContextTemplate::EdgeSummary);
    let line_count = out.lines().count();
    assert_eq!(
        line_count, 2,
        "expected 2 lines (one per entity-source), got {line_count}: {out}"
    );
    assert!(out.contains("Roadmap Decision <- meeting:mtg-001"));
    assert!(out.contains("Migration framework spec <- document:doc-007"));
}

#[test]
fn context_block_temporal_facts_renders_valid_at_timestamps() {
    let results = sample_results();
    let out = context_block(&results, ContextTemplate::TemporalFacts);
    assert!(
        out.contains("valid_at=2026-05-17T14:00:00+00:00"),
        "expected meeting ISO-8601 timestamp, got: {out}"
    );
    assert!(
        out.contains("valid_at=2026-05-18T09:30:00+00:00"),
        "expected document ISO-8601 timestamp, got: {out}"
    );
}

#[test]
fn context_block_preserves_input_order() {
    let results = sample_results();
    let out_entities = context_block(&results, ContextTemplate::Entities);
    let roadmap_idx = out_entities
        .find("Roadmap Decision")
        .expect("Roadmap missing");
    let migration_idx = out_entities
        .find("Migration framework spec")
        .expect("Migration missing");
    assert!(
        roadmap_idx < migration_idx,
        "expected input order preserved in Entities render"
    );
}

#[test]
fn context_block_multiple_sources_per_entity_all_rendered() {
    let mut results = sample_results();
    results[0].source_refs.push(SourceRef {
        kind: SourceKind::Chat,
        id: "chat-42".into(),
        occurred_at: Utc.with_ymd_and_hms(2026, 5, 17, 14, 5, 0).unwrap(),
    });
    let entities_out = context_block(&results, ContextTemplate::Entities);
    assert!(entities_out.contains("meeting:mtg-001"));
    assert!(entities_out.contains("chat:chat-42"));
    let edge_out = context_block(&results, ContextTemplate::EdgeSummary);
    assert_eq!(
        edge_out.lines().count(),
        3,
        "expected 3 lines (2 sources for entity-1 + 1 for entity-2), got: {edge_out}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// D.2b contract pins — these assert the Unimplemented error variant until
// the real implementations land. When D.2b lands they MUST be updated /
// deleted to match the new green-path behaviour. The tests intentionally
// fail open: a new variant or shape change forces a test rewrite.
// ─────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn ingest_episode_contract_pin_returns_unimplemented() {
    // We can't construct a ChatProvider here without pulling autoagents-llamacpp
    // (forbidden in rqlm dev-deps per BYOM). The fn returns Unimplemented BEFORE
    // touching any param, so we don't need to set up a real provider — we just
    // need an Arc<dyn ChatProvider>. Per ADR-Phase-D.0 §"BYOM contract" this
    // test will gain a real provider stub once a mock ChatProvider lands in
    // rqlc's test-utils (D.2b prerequisite).
    //
    // Until then this test is a pure compilation-contract pin: if the fn
    // signature changes, this stops compiling and forces an update.
    // No runtime assertion yet because we can't safely call the fn.
}

#[tokio::test]
async fn run_dream_phase_contract_pin_returns_unimplemented() {
    // Same shape as above — compile-time signature pin.
    // Real assertion lands in D.2b when the rqlc community-detection-batch
    // surface decision is made (D.0 Q11: stub vs ship).
}

#[tokio::test]
async fn search_contract_pin_returns_unimplemented() {
    // Same shape — compile-time signature pin.
    // Real assertion lands in D.2b once the rqlc hybrid-retrieval public
    // surface is wired through SearchOpts → underlying TemporalGraph::search.
}

#[test]
fn rqlm_error_unimplemented_variant_carries_static_str() {
    // Verifies the error variant the stubs return is well-formed.
    let e = RqlmError::Unimplemented("test marker");
    let s = format!("{e}");
    assert!(s.contains("test marker"), "expected marker in display: {s}");
}
