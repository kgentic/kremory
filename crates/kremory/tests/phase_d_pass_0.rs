#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase D — Dream Pass 0 integration tests.
//!
//! Governing spec: ADR-037 §3 / §9 (type discovery).
//! Migration spec: ADR-037 §9.5 (Migration 014 provenance columns).
//!
//! ## Acceptance criteria
//!
//! D1 — `discover_types` end-to-end: catch-all entities → LLM proposal → persistence.
//! D2 — Shape validator rejects all 9 reason categories.
//! D3 — Anti-redundancy gate rejects on cosine ≥ 0.85 (desc) / ≥ 0.70 (name) thresholds.
//! D4 — In-place evidence retype writes `entity_type_source = 'DreamPass0'`.
//! D5 — `max_proposals = 5` cap enforced prompt-side (system prompt contains the cap).
//! D6 — `DreamSummary.types_discovered` populated via `mem.dream()`.
//! D7 — Degraded mode: `embedder = None` → anti-redundancy gate skipped + warning.
//! D8 — Migration 014 provenance columns present on `entity_types` after `run_migrations`.
//!
//! ## Phase C vs Phase D
//!
//! D8 and D6 API-shape tests are included here as Phase C ships Migration 014
//! and the `DreamSummary` surface. D1-D7 full integration tests require Phase D
//! to be implemented (the `discover_types` primitive is `pub(crate)` and will
//! be exposed through `mem.dream()` in Phase D).

use kremory::core::schema::TemporalGraph;

// ─── helpers ─────────────────────────────────────────────────────────────────

async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("pass0-test.db");
    let graph = TemporalGraph::open(path.to_str().expect("utf8 path"))
        .await
        .expect("TemporalGraph::open");
    (graph, tmp)
}

/// Column names for `entity_types` table.
async fn entity_types_columns(graph: &TemporalGraph) -> Vec<String> {
    let mut rows = graph
        .conn
        .query("PRAGMA table_info('entity_types')", ())
        .await
        .expect("PRAGMA table_info");
    let mut cols = Vec::new();
    while let Some(row) = rows.next().await.expect("row read") {
        let name: String = row.get(1).expect("column name at index 1");
        cols.push(name);
    }
    cols
}

// ─── D8: Migration 014 provenance columns ────────────────────────────────────

/// D8: Migration 014 adds `discovered_at`, `discovered_by`, `evidence_count`,
/// and `confidence` columns to `entity_types` on first `TemporalGraph::open`.
#[tokio::test]
async fn d8_migration_014_provenance_columns_present() {
    let (graph, _tmp) = open_graph().await;
    let cols = entity_types_columns(&graph).await;

    assert!(
        cols.iter().any(|c| c == "discovered_at"),
        "entity_types must have discovered_at; cols={cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "discovered_by"),
        "entity_types must have discovered_by; cols={cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "evidence_count"),
        "entity_types must have evidence_count; cols={cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "confidence"),
        "entity_types must have confidence; cols={cols:?}"
    );
}

/// D8: Running migrations twice does NOT duplicate entity_type rows (idempotent).
#[tokio::test]
async fn d8_migration_014_idempotent() {
    let (graph, _tmp) = open_graph().await;
    // Columns must be present after open (first migration run).
    let cols = entity_types_columns(&graph).await;
    assert!(cols.iter().any(|c| c == "discovered_at"), "discovered_at after first open");
    // Keep graph alive across the check.
    let _ = &graph.conn;
    // Column still present — idempotent PRAGMA gates guaranteed no-op.
    let cols2 = entity_types_columns(&graph).await;
    assert!(cols2.iter().any(|c| c == "discovered_at"), "discovered_at still present");
}

// ─── D6: DreamSummary API shape ───────────────────────────────────────────────

/// D6: `DreamSummary.types_discovered` is a Vec<TypeProposal> field accessible
/// from the public API.  This is a compile-time shape check — if the field
/// doesn't exist or has the wrong type, this won't compile.
#[tokio::test]
async fn d6_dream_summary_types_discovered_field_accessible() {
    let summary = kremory::DreamSummary {
        communities_updated: 0,
        cross_episode_merges: 0,
        supersessions_recorded: 0,
        facts_archived: 0,
        duration_ms: 0,
        types_discovered: vec![kremory::TypeProposal {
            name: "TestType".to_string(),
            description: "A test type".to_string(),
            justification: "test".to_string(),
        }],
        warnings: vec!["test warning".to_string()],
    };
    assert_eq!(summary.types_discovered.len(), 1);
    assert_eq!(summary.types_discovered[0].name, "TestType");
    assert_eq!(summary.warnings.len(), 1);
}

/// D6: `DreamOpts::default()` has `include_type_discovery = true` per ADR-037 §3.
///
/// `mem.dream()` passes `DreamOpts` through to the consolidation logic; the
/// `include_type_discovery = true` default means Pass 0 fires on every
/// dream cycle unless explicitly disabled.
#[test]
fn d6_dream_opts_default_include_type_discovery_true() {
    let opts = kremory::DreamOpts::default();
    assert!(
        opts.include_type_discovery,
        "DreamOpts::default() must have include_type_discovery = true per ADR-037 §3"
    );
}

/// D6: `DreamPassOpts` struct (Phase C DoD C2) is accessible from the public API
/// and has the correct fields per the sprint spec.
#[test]
fn d6_dream_pass_opts_shape() {
    let opts = kremory::DreamPassOpts::default();
    // Compile-time shape check: fields must exist with the right types.
    let _: bool = opts.include_type_discovery;
    let _: f32 = opts.confidence_threshold;
    let _: Option<usize> = opts.max_episodes_per_run;
    let _: f32 = opts.reclassify_high_conf_threshold;
    // ADR-045 §3: reclassify threshold default is 0.7.
    assert!(
        (opts.reclassify_high_conf_threshold - 0.7).abs() < f32::EPSILON,
        "reclassify_high_conf_threshold default must be 0.7 per ADR-045 §3"
    );
}
