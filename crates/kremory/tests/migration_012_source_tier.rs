#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 012 source-tier column tests — DoD A1–A9 + Q-04 symmetric protection.
//!
//! Governing spec: migration-010-detail-spec-closing-vera-cycle-3-highs-2026-06-09.md §1.1–1.4
//! ADR-045 §2 (Phase1Ner), §3 (ConsumerPinned), §6 (GLiNER confidence)
//!
//! ## Acceptance criteria
//!
//! A1 — After run_migrations, entities table has `entity_type_source`, `entity_type_assigned_at`,
//!      and `ner_confidence` columns.
//! A2 — CHECK constraint accepts all 5 valid values; rejects 'Garbage'.
//! A3 — Idempotency: run_migrations twice is a no-op (PRAGMA gate).
//! A4 — Legacy backfill: rows inserted with NULL entity_type_source before migration
//!      get backfilled to 'Phase1Ner' with entity_type_assigned_at populated.
//! A5 — v_entity_drift_candidates view: detects entities with same (name, group_id)
//!      but different entity_type_id; ConsumerPinned excluded from view.
//! A6 — confidence field round-trip: RawEntityIntegerId deserialises with conf=0.9,
//!      missing conf (None), and explicit null conf (None).
//! A7 — Per-extractor source-tier: new entities inserted via insert_entity_with_group
//!      have entity_type_source = 'Phase1Ner'.
//! A8 — ConsumerPinned write: update_entity_source_tier stamps 'ConsumerPinned' correctly.
//! A9 — Implied by A1-A8 + separate `cargo test` invocation.

use std::sync::Arc;

use chrono::Utc;
use kremory::core::config::PipelineConfig;
use kremory::core::extraction::LlmExtractor;
use kremory::core::ingest::{Engine, PrePinnedFact, SourceParams};
use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;

// ─── helpers ─────────────────────────────────────────────────────────────────

async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir must succeed");
    let path = tmp.path().join("migration-012-test.db");
    let path_str = path.to_str().expect("path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed");
    (graph, tmp)
}

async fn entity_columns(graph: &TemporalGraph) -> Vec<String> {
    let mut rows = graph
        .conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .expect("PRAGMA table_info(entities) must succeed");
    let mut cols = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let name: String = row.get(1).expect("column name at index 1");
        cols.push(name);
    }
    cols
}

// ─── A1: schema smoke — three new columns present ────────────────────────────

#[tokio::test]
async fn a1_migration_012_columns_present() {
    let (graph, _tmp) = open_graph().await;
    let cols = entity_columns(&graph).await;

    assert!(
        cols.iter().any(|c| c == "entity_type_source"),
        "entities must have entity_type_source after migration 012; cols: {cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "entity_type_assigned_at"),
        "entities must have entity_type_assigned_at after migration 012; cols: {cols:?}"
    );
    assert!(
        cols.iter().any(|c| c == "ner_confidence"),
        "entities must have ner_confidence after migration 012; cols: {cols:?}"
    );
}

// ─── A2: CHECK constraint — accepts 5 valid values, rejects 'Garbage' ────────

#[tokio::test]
async fn a2_check_constraint_accepts_valid_values() {
    let (graph, _tmp) = open_graph().await;

    let valid_tiers = [
        "Phase1Ner",
        "Phase2Llm",
        "DreamPass0",
        "DreamPass1",
        "ConsumerPinned",
    ];
    for (i, tier) in valid_tiers.iter().enumerate() {
        let id = format!("ent-check-{i}");
        let sql = format!(
            "INSERT INTO entities (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('{id}', 0, '{{}}', datetime('now'), 'default', '{tier}')"
        );
        graph
            .conn
            .execute(&sql, ())
            .await
            .unwrap_or_else(|e| panic!("INSERT with valid tier '{tier}' must succeed: {e}"));
    }
}

#[tokio::test]
async fn a2_check_constraint_rejects_garbage() {
    let (graph, _tmp) = open_graph().await;

    let result = graph
        .conn
        .execute(
            "INSERT INTO entities (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('ent-garbage', 0, '{}', datetime('now'), 'default', 'Garbage')",
            (),
        )
        .await;

    assert!(
        result.is_err(),
        "INSERT with entity_type_source='Garbage' must fail the CHECK constraint"
    );
}

// ─── A3: idempotency — double-apply is a no-op ───────────────────────────────

#[tokio::test]
async fn a3_migration_012_idempotent_double_apply() {
    let (graph, _tmp) = open_graph().await;

    let cols_first = entity_columns(&graph).await;
    assert!(
        cols_first.iter().any(|c| c == "entity_type_source"),
        "pre-condition: entity_type_source must be present after first open()"
    );

    // Second migration run via the test hook — must not error.
    graph
        .run_migrations_again_for_test()
        .await
        .expect("second run_migrations must be idempotent — PRAGMA gate skips existing columns");

    let cols_second = entity_columns(&graph).await;
    assert!(
        cols_second.iter().any(|c| c == "entity_type_source"),
        "entity_type_source must still be present after second migration run"
    );
    assert!(
        cols_second.iter().any(|c| c == "ner_confidence"),
        "ner_confidence must still be present after second migration run"
    );
}

// ─── A4: legacy backfill — NULL rows get Phase1Ner ───────────────────────────

#[tokio::test]
async fn a4_legacy_backfill_stamps_phase1ner() {
    let tmp = tempfile::TempDir::new().expect("tempdir must succeed");
    let path = tmp.path().join("migration-012-backfill.db");
    let path_str = path.to_str().expect("path must be valid UTF-8");

    // Open once to run migrations through 011 — can't easily seed NULL rows
    // before 012 runs in a single open(). Instead: verify that after open(),
    // the backfill UPDATE correctly sets any existing NULL rows. We insert
    // with explicit NULL then re-run migrations to simulate the backfill scenario.
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("first open must succeed");

    // Force a NULL entity_type_source via raw SQL (bypassing the INSERT that
    // now stamps Phase1Ner, to simulate a pre-migration legacy row).
    graph
        .conn
        .execute(
            "UPDATE entities SET entity_type_source = NULL, entity_type_assigned_at = NULL \
             WHERE id IN (SELECT id FROM entities LIMIT 1)",
            (),
        )
        .await
        .expect("raw NULL update must succeed");

    // Insert a new row with explicit NULL to confirm backfill covers it.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('legacy-null-ent', 0, '{}', datetime('now'), 'default', NULL)",
            (),
        )
        .await
        .expect("legacy NULL insert must succeed");

    // Re-run migrations (idempotent — PRAGMA skips ADD COLUMN, but backfill UPDATE fires).
    graph
        .run_migrations_again_for_test()
        .await
        .expect("run_migrations_again must succeed");

    // Verify backfill: the legacy-null-ent must now have Phase1Ner.
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source, entity_type_assigned_at FROM entities \
             WHERE id = 'legacy-null-ent'",
            (),
        )
        .await
        .expect("query must succeed");

    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("legacy-null-ent row must exist");

    let source: Option<String> = row.get(0).expect("entity_type_source at index 0");
    let assigned_at: Option<String> = row.get(1).expect("entity_type_assigned_at at index 1");

    assert_eq!(
        source.as_deref(),
        Some("Phase1Ner"),
        "legacy NULL row must be backfilled to Phase1Ner; got: {source:?}"
    );
    assert!(
        assigned_at.is_some(),
        "entity_type_assigned_at must be set by backfill; got: {assigned_at:?}"
    );
}

// ─── A5: v_entity_drift_candidates view correctness ──────────────────────────

#[tokio::test]
async fn a5_drift_view_detects_type_mismatch() {
    let (graph, _tmp) = open_graph().await;

    // Two entity types must exist: id=0 "Entity" (catch-all) is always present.
    // Insert a second type for this test.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types (id, name, group_id) \
             VALUES (42, 'Person', 'default')",
            (),
        )
        .await
        .expect("entity_types insert must succeed");

    // Insert two entities: same name, same group, different entity_type_id.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('drift-ent-a', 0, '{\"name\":\"Alice\"}', datetime('now'), 'default', 'Phase1Ner')",
            (),
        )
        .await
        .expect("drift-ent-a insert must succeed");

    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('drift-ent-b', 42, '{\"name\":\"Alice\"}', datetime('now'), 'default', 'Phase1Ner')",
            (),
        )
        .await
        .expect("drift-ent-b insert must succeed");

    // Both entities have name='Alice' but different entity_type_id — both should appear.
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id FROM v_entity_drift_candidates ORDER BY entity_id",
            (),
        )
        .await
        .expect("v_entity_drift_candidates query must succeed");

    let mut found_ids: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let id: String = row.get(0).expect("entity_id at index 0");
        found_ids.push(id);
    }

    assert!(
        found_ids.iter().any(|id| id == "drift-ent-a"),
        "drift-ent-a must appear in v_entity_drift_candidates; found: {found_ids:?}"
    );
    assert!(
        found_ids.iter().any(|id| id == "drift-ent-b"),
        "drift-ent-b must appear in v_entity_drift_candidates; found: {found_ids:?}"
    );
}

#[tokio::test]
async fn a5_drift_view_excludes_consumer_pinned() {
    let (graph, _tmp) = open_graph().await;

    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types (id, name, group_id) \
             VALUES (43, 'Org', 'default')",
            (),
        )
        .await
        .expect("entity_types insert must succeed");

    // Entity with ConsumerPinned source — must NOT appear in drift view.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('pinned-ent', 0, '{\"name\":\"Acme\"}', datetime('now'), 'default', 'ConsumerPinned')",
            (),
        )
        .await
        .expect("pinned-ent insert must succeed");

    // Counterpart with different type, Phase1Ner — this one may appear.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('ner-ent', 43, '{\"name\":\"Acme\"}', datetime('now'), 'default', 'Phase1Ner')",
            (),
        )
        .await
        .expect("ner-ent insert must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id FROM v_entity_drift_candidates",
            (),
        )
        .await
        .expect("v_entity_drift_candidates query must succeed");

    let mut found_ids: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let id: String = row.get(0).expect("entity_id at index 0");
        found_ids.push(id);
    }

    assert!(
        !found_ids.iter().any(|id| id == "pinned-ent"),
        "ConsumerPinned entity must NOT appear in v_entity_drift_candidates; found: {found_ids:?}"
    );
}

// ─── A5 addendum: cross-namespace same-name entities must NOT appear in view ──
//
// The view WHERE clause includes `e2.group_id = e1.group_id`, so two entities
// with the same name but different group_ids (namespaces) are NOT drift candidates
// relative to each other. This test seeds one same-name/different-type pair in
// ns_a (should appear) and one entity with the same name in ns_b (different namespace
// — must NOT cause ns_a entities to be excluded, and ns_b entity itself must not appear
// as a drift candidate against ns_a entities).

#[tokio::test]
async fn a5_drift_view_excludes_cross_namespace() {
    let (graph, _tmp) = open_graph().await;

    // Seed entity types needed for the test.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entity_types (id, name, group_id) \
             VALUES (50, 'Person', 'default')",
            (),
        )
        .await
        .expect("entity_types insert must succeed");

    // ns_a: two entities with same name, different type — both should appear in view.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('cross-ns-a1', 0, '{\"name\":\"Shared\"}', datetime('now'), 'ns_a', 'Phase1Ner')",
            (),
        )
        .await
        .expect("cross-ns-a1 insert must succeed");

    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('cross-ns-a2', 50, '{\"name\":\"Shared\"}', datetime('now'), 'ns_a', 'Phase1Ner')",
            (),
        )
        .await
        .expect("cross-ns-a2 insert must succeed");

    // ns_b: one entity with the same name — different namespace, must NOT appear.
    graph
        .conn
        .execute(
            "INSERT OR IGNORE INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('cross-ns-b1', 0, '{\"name\":\"Shared\"}', datetime('now'), 'ns_b', 'Phase1Ner')",
            (),
        )
        .await
        .expect("cross-ns-b1 insert must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT entity_id FROM v_entity_drift_candidates ORDER BY entity_id",
            (),
        )
        .await
        .expect("v_entity_drift_candidates query must succeed");

    let mut found_ids: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        let id: String = row.get(0).expect("entity_id at index 0");
        found_ids.push(id);
    }

    // ns_a pair must appear — they share group_id and have different entity_type_id.
    assert!(
        found_ids.iter().any(|id| id == "cross-ns-a1"),
        "cross-ns-a1 must appear in v_entity_drift_candidates (same-ns drift pair); found: {found_ids:?}"
    );
    assert!(
        found_ids.iter().any(|id| id == "cross-ns-a2"),
        "cross-ns-a2 must appear in v_entity_drift_candidates (same-ns drift pair); found: {found_ids:?}"
    );

    // ns_b entity must NOT appear — no same-namespace counterpart with different type.
    assert!(
        !found_ids.iter().any(|id| id == "cross-ns-b1"),
        "cross-ns-b1 must NOT appear in v_entity_drift_candidates (cross-namespace, view filters e2.group_id = e1.group_id); found: {found_ids:?}"
    );
}

// ─── A6: confidence field round-trip ─────────────────────────────────────────
// RawEntityIntegerId is pub(crate) — round-trip tests live in
// crates/kremory/src/core/extraction/models.rs #[cfg(test)] block
// (see `mod tests_a6_confidence_round_trip`).
// This integration test file covers the DB-layer concerns only (A1-A5, A7-A8).

// ─── A7: per-extractor source-tier — insert_entity_with_group stamps Phase1Ner ─

#[tokio::test]
async fn a7_insert_entity_with_group_stamps_phase1ner() {
    let (graph, _tmp) = open_graph().await;

    graph
        .insert_entity_with_group(
            "a7-ent-001",
            0,
            serde_json::json!({"name": "Alice"}),
            Some("default"),
        )
        .await
        .expect("insert_entity_with_group must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source, entity_type_assigned_at FROM entities WHERE id = 'a7-ent-001'",
            (),
        )
        .await
        .expect("SELECT must succeed");

    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("a7-ent-001 row must exist");

    let source: Option<String> = row.get(0).expect("entity_type_source at index 0");
    let assigned_at: Option<String> = row.get(1).expect("entity_type_assigned_at at index 1");

    assert_eq!(
        source.as_deref(),
        Some("Phase1Ner"),
        "insert_entity_with_group must stamp Phase1Ner; got: {source:?}"
    );
    assert!(
        assigned_at.is_some(),
        "entity_type_assigned_at must be set; got: {assigned_at:?}"
    );
}

#[tokio::test]
async fn a7_insert_entity_no_group_stamps_phase1ner() {
    let (graph, _tmp) = open_graph().await;

    graph
        .insert_entity("a7-nogroup-001", 0, serde_json::json!({"name": "Bob"}))
        .await
        .expect("insert_entity must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source FROM entities WHERE id = 'a7-nogroup-001'",
            (),
        )
        .await
        .expect("SELECT must succeed");

    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("a7-nogroup-001 row must exist");

    let source: Option<String> = row.get(0).expect("entity_type_source at index 0");
    assert_eq!(
        source.as_deref(),
        Some("Phase1Ner"),
        "insert_entity (no-group) must stamp Phase1Ner; got: {source:?}"
    );
}

// ─── A8: ConsumerPinned write — update_entity_source_tier ────────────────────

#[tokio::test]
async fn a8_update_entity_source_tier_consumer_pinned() {
    let (graph, _tmp) = open_graph().await;

    // Insert an entity (stamps Phase1Ner).
    graph
        .insert_entity_with_group(
            "a8-ent-001",
            0,
            serde_json::json!({"name": "Acme Corp"}),
            Some("default"),
        )
        .await
        .expect("insert must succeed");

    // Verify initial source is Phase1Ner.
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source FROM entities WHERE id = 'a8-ent-001'",
            (),
        )
        .await
        .expect("SELECT must succeed");
    let row = rows.next().await.unwrap().unwrap();
    let initial_source: Option<String> = row.get(0).unwrap();
    assert_eq!(initial_source.as_deref(), Some("Phase1Ner"), "initial source must be Phase1Ner");

    // Stamp ConsumerPinned.
    graph
        .update_entity_source_tier("a8-ent-001", Some("default"), "ConsumerPinned")
        .await
        .expect("update_entity_source_tier must succeed");

    // Verify updated source.
    let mut rows2 = graph
        .conn
        .query(
            "SELECT entity_type_source, entity_type_assigned_at FROM entities WHERE id = 'a8-ent-001'",
            (),
        )
        .await
        .expect("SELECT must succeed");
    let row2 = rows2.next().await.unwrap().unwrap();
    let updated_source: Option<String> = row2.get(0).unwrap();
    let updated_at: Option<String> = row2.get(1).unwrap();

    assert_eq!(
        updated_source.as_deref(),
        Some("ConsumerPinned"),
        "update_entity_source_tier must write ConsumerPinned; got: {updated_source:?}"
    );
    assert!(
        updated_at.is_some(),
        "entity_type_assigned_at must be updated; got: {updated_at:?}"
    );
}

#[tokio::test]
async fn a8_set_entity_ner_confidence_persists() {
    let (graph, _tmp) = open_graph().await;

    graph
        .insert_entity_with_group(
            "a8-conf-001",
            0,
            serde_json::json!({"name": "Some Entity"}),
            Some("default"),
        )
        .await
        .expect("insert must succeed");

    graph
        .set_entity_ner_confidence("a8-conf-001", Some("default"), 0.85_f32)
        .await
        .expect("set_entity_ner_confidence must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT ner_confidence FROM entities WHERE id = 'a8-conf-001'",
            (),
        )
        .await
        .expect("SELECT must succeed");

    let row = rows.next().await.unwrap().unwrap();
    let conf: Option<f64> = row.get(0).expect("ner_confidence at index 0");

    assert!(
        conf.is_some(),
        "ner_confidence must be set after set_entity_ner_confidence; got None"
    );
    let conf_val = conf.unwrap();
    assert!(
        (conf_val - 0.85_f64).abs() < 1e-4,
        "ner_confidence must be ~0.85; got {conf_val}"
    );
}

// ─── A8 addendum: upsert preserves existing ConsumerPinned tier ──────────────

#[tokio::test]
async fn a8_upsert_preserves_consumer_pinned_tier() {
    let (graph, _tmp) = open_graph().await;

    // Insert entity, stamp ConsumerPinned.
    graph
        .insert_entity_with_group(
            "a8-upsert-001",
            0,
            serde_json::json!({"name": "Pinned Org"}),
            Some("default"),
        )
        .await
        .expect("insert must succeed");

    graph
        .update_entity_source_tier("a8-upsert-001", Some("default"), "ConsumerPinned")
        .await
        .expect("update must succeed");

    // Upsert (stub-promotion path) — must NOT overwrite ConsumerPinned.
    graph
        .upsert_entity_with_group(
            "a8-upsert-001",
            1,
            serde_json::json!({"name": "Pinned Org", "context": "some context"}),
            Some("default"),
        )
        .await
        .expect("upsert must succeed");

    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source FROM entities WHERE id = 'a8-upsert-001'",
            (),
        )
        .await
        .expect("SELECT must succeed");

    let row = rows.next().await.unwrap().unwrap();
    let source: Option<String> = row.get(0).unwrap();

    assert_eq!(
        source.as_deref(),
        Some("ConsumerPinned"),
        "upsert must preserve ConsumerPinned tier (COALESCE guard); got: {source:?}"
    );
}

// ─── Q-04: symmetric ConsumerPinned — object_id entity stamped ───────────────
//
// Governing spec: ADR-045 §3 amendment 2026-06-09; migration-010-detail-spec §1.2
// (object row). Tests MUST use Engine::ingest_with() + PrePinnedFact directly —
// the Memory::with_facts() facade always sets object_id = None.

/// Q-04 gate: when a PrePinnedFact is inserted with object_id = Some(...),
/// BOTH the subject entity AND the object entity are stamped ConsumerPinned.
#[tokio::test]
async fn q04_with_facts_object_id_stamped_consumer_pinned() {
    let (graph, _tmp) = open_graph().await;
    let graph = Arc::new(graph);

    let llm = Arc::new(MockChatProvider::null());
    let embedder = Arc::new(NullEmbeddingProvider { dim: 384 });
    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig default must succeed");
    let engine = Engine::new(Arc::clone(&graph), Arc::clone(&llm), embedder, config);

    let extractor = LlmExtractor::new(llm);

    // Insert both entities first so the pin_fact branch finds them (stub-insert
    // path is the same; both are inserted Phase1Ner by the pipeline before pin_fact runs).
    // We rely on ingest_with doing the stub-insert internally — all we need is
    // skip_extraction so the LLM-dependent code path is never reached.

    let params = SourceParams {
        pre_pinned_facts: vec![PrePinnedFact {
            subject: "q04-alice".to_string(),
            predicate: "works_at".to_string(),
            object_id: Some("q04-acme".to_string()),
            object_value: None,
            valid_from: Utc::now(),
            confidence: 1.0,
        }],
        skip_extraction: true,
        ..SourceParams::default()
    };

    engine
        .ingest_with(
            &extractor,
            "Q-04 symmetric ConsumerPinned test episode.",
            None,
            Some("default"),
            None,
            params,
        )
        .await
        .expect("ingest_with with object_id fact must succeed");

    // Assert BOTH endpoints are ConsumerPinned.
    for entity_id in &["q04-alice", "q04-acme"] {
        let mut rows = graph
            .conn
            .query(
                "SELECT entity_type_source FROM entities WHERE id = ?1",
                libsql::params![*entity_id],
            )
            .await
            .expect("SELECT must succeed");

        let row = rows
            .next()
            .await
            .expect("row iteration must not error")
            .unwrap_or_else(|| panic!("entity '{entity_id}' must exist after ingest_with"));

        let source: Option<String> = row.get(0).expect("entity_type_source at index 0");
        assert_eq!(
            source.as_deref(),
            Some("ConsumerPinned"),
            "Q-04: entity '{entity_id}' must be ConsumerPinned after pin_fact; got: {source:?}"
        );
    }
}

/// Q-04 negative gate: when a PrePinnedFact has object_id = None (literal
/// object_value), only the subject entity is stamped ConsumerPinned — no object
/// entity is created or stamped.
#[tokio::test]
async fn q04_with_facts_object_value_literal_no_object_stamp() {
    let (graph, _tmp) = open_graph().await;
    let graph = Arc::new(graph);

    let llm = Arc::new(MockChatProvider::null());
    let embedder = Arc::new(NullEmbeddingProvider { dim: 384 });
    let config = PipelineConfig::builder()
        .build()
        .expect("PipelineConfig default must succeed");
    let engine = Engine::new(Arc::clone(&graph), Arc::clone(&llm), embedder, config);

    let extractor = LlmExtractor::new(llm);

    let params = SourceParams {
        pre_pinned_facts: vec![PrePinnedFact {
            subject: "q04-literal-subject".to_string(),
            predicate: "has_value".to_string(),
            object_id: None,
            object_value: Some("some literal string".to_string()),
            valid_from: Utc::now(),
            confidence: 1.0,
        }],
        skip_extraction: true,
        ..SourceParams::default()
    };

    engine
        .ingest_with(
            &extractor,
            "Q-04 literal object_value — no object entity should be created.",
            None,
            Some("default"),
            None,
            params,
        )
        .await
        .expect("ingest_with with literal object_value must succeed");

    // Subject must be ConsumerPinned.
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source FROM entities WHERE id = 'q04-literal-subject'",
            (),
        )
        .await
        .expect("SELECT subject must succeed");

    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("q04-literal-subject must exist after ingest_with");

    let source: Option<String> = row.get(0).expect("entity_type_source at index 0");
    assert_eq!(
        source.as_deref(),
        Some("ConsumerPinned"),
        "Q-04 literal: subject must be ConsumerPinned; got: {source:?}"
    );

    // No entity with id matching the literal value should exist.
    let mut rows2 = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE id = 'some literal string'",
            (),
        )
        .await
        .expect("SELECT literal id must succeed");

    let row2 = rows2
        .next()
        .await
        .expect("row iteration must not error")
        .expect("COUNT(*) must return a row");

    let count: i64 = row2.get(0).expect("count at index 0");
    assert_eq!(
        count, 0,
        "Q-04 literal: no entity must be created for a literal object_value; got count={count}"
    );
}
