#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 013 + 012b Phase B acceptance tests — DoD B1–B6.
//!
//! Governing spec: .ai-docs/specs/migration-013-pass4-source-tier-2026-06-09.md
//! ADR-047 §Integration with v0.1.1 substrate
//!
//! ## Acceptance criteria
//!
//! B1 — Migration 013 extends CHECK constraint: INSERT with 'DreamPass4' succeeds;
//!      INSERT with 'GarbageValue' fails.
//! B2 — Migration 013 creates dream_pass4_audit table with all 8 required columns.
//! B3 — Migration 013 idempotent: running twice = no-op; audit table exists exactly once.
//! B4 — Migration 012b downgrade round-trip: 5 DreamPass4-tagged entities re-stamped to
//!      Phase1Ner; audit table dropped; subsequent DreamPass4 INSERT fails.
//! B5 — Migration 012b idempotent: running twice after downgrade = no-op.
//! B6 — Full chain (run_migrations through 014, then 013 on top) preserves existing
//!      schema invariants: entity_type_source column present, CHECK on pre-012 valid
//!      values still accepted, entity_types table present.
//!
//! ## How these tests fail (Red phase)
//!
//! `crate::core::migrations::migrate_013_pass4_source_tier` and
//! `crate::core::migrations::migrate_012b_revert_pass4_source_tier` do not exist yet.
//! Tests WILL NOT COMPILE until the Green agent adds these functions to migrations.rs
//! and wires migrate_013 into schema.rs::run_migrations.
//!
//! ## Note for Green agent: expected function signatures
//!
//! In `crates/kremory/src/core/migrations.rs`, add:
//!
//! ```rust
//! pub(crate) async fn migrate_013_pass4_source_tier(
//!     conn: &libsql::Connection,
//! ) -> crate::core::error::Result<()> { ... }
//!
//! pub(crate) async fn migrate_012b_revert_pass4_source_tier(
//!     conn: &libsql::Connection,
//! ) -> crate::core::error::Result<()> { ... }
//! ```
//!
//! In `crates/kremory/src/core/schema.rs::run_migrations`, add after migrate_014:
//! ```rust
//! crate::core::migrations::migrate_013_pass4_source_tier(&self.conn).await?;
//! ```

use kremory::core::schema::TemporalGraph;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Open a fresh TemporalGraph using a temp-dir file DB (not in-memory).
/// Runs all migrations in run_migrations() automatically on open.
/// Returns the graph and the TempDir guard (must stay alive to prevent
/// premature deletion of the underlying file).
async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir must succeed");
    let path = tmp.path().join("migration-013-test.db");
    let path_str = path.to_str().expect("path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed (runs all migrations)");
    (graph, tmp)
}

/// Run migrate_013 directly against the graph's connection.
/// Tests call this to apply the forward migration after open().
/// NOTE: migrate_013 is intentionally NOT in run_migrations() at the time
/// these tests are authored (Red phase). The Green agent wires it in.
/// Tests that need the forward migration call this helper explicitly.
async fn run_migrate_013(graph: &TemporalGraph) {
    kremory::core::migrations::migrate_013_pass4_source_tier(&graph.conn)
        .await
        .expect("migrate_013_pass4_source_tier must succeed");
}

/// Run migrate_012b downgrade directly against the graph's connection.
async fn run_migrate_012b(graph: &TemporalGraph) {
    kremory::core::migrations::migrate_012b_revert_pass4_source_tier(&graph.conn)
        .await
        .expect("migrate_012b_revert_pass4_source_tier must succeed");
}

/// Seed a single entity row suitable for testing source-tier writes.
/// Uses the minimal non-null columns: id, entity_type_id, properties, recorded_at.
/// group_id and entity_type_source are provided explicitly.
#[allow(dead_code)]
async fn seed_entity(graph: &TemporalGraph, id: &str, entity_type_source: &str) -> i64 {
    let mut rows = graph
        .conn
        .query(
            "INSERT INTO entities (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES (?1, 1, '{}', datetime('now'), 'default', ?2) RETURNING rowid",
            libsql::params![id, entity_type_source],
        )
        .await
        .expect("seed_entity INSERT must succeed");

    let row = rows
        .next()
        .await
        .expect("RETURNING must yield a row without error")
        .expect("RETURNING must yield a row");
    row.get::<i64>(0).expect("rowid at index 0")
}

// ─── B1: CHECK constraint expanded — DreamPass4 accepted, GarbageValue rejected ──

/// B1: After migrate_013, INSERT with entity_type_source='DreamPass4' must succeed.
/// INSERT with entity_type_source='GarbageValue' must fail the CHECK constraint.
///
/// Verifies: Migration 013 §013.1 — CHECK constraint extended with 'DreamPass4'.
/// Failure mode without fix: entity_type_source CHECK does not include 'DreamPass4';
/// the DreamPass4 insert returns a constraint-violation error.
#[tokio::test]
async fn b1_migration_013_adds_dreampass4_check_value() {
    let (graph, _tmp) = open_graph().await;
    run_migrate_013(&graph).await;

    // DreamPass4 INSERT must succeed after migration 013.
    graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b1-dp4-ent', 1, '{}', datetime('now'), 'default', 'DreamPass4')",
            (),
        )
        .await
        .expect("INSERT with entity_type_source='DreamPass4' must succeed after migration 013");

    // Verify row actually persisted with correct value.
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_source FROM entities WHERE id = 'b1-dp4-ent'",
            (),
        )
        .await
        .expect("SELECT must succeed");
    let row = rows
        .next()
        .await
        .expect("row iteration must not error")
        .expect("b1-dp4-ent must exist");
    let source: Option<String> = row.get(0).expect("entity_type_source at index 0");
    assert_eq!(
        source.as_deref(),
        Some("DreamPass4"),
        "persisted entity_type_source must be 'DreamPass4'; got: {source:?}"
    );

    // GarbageValue INSERT must fail CHECK constraint — even after 013.
    let garbage_result = graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b1-garbage-ent', 1, '{}', datetime('now'), 'default', 'GarbageValue')",
            (),
        )
        .await;
    assert!(
        garbage_result.is_err(),
        "INSERT with entity_type_source='GarbageValue' must fail the CHECK constraint after migration 013"
    );
}

// ─── B2: dream_pass4_audit table schema ──────────────────────────────────────

/// B2: After migrate_013, PRAGMA table_info(dream_pass4_audit) must return all
/// 8 required columns: audit_id, entity_id, pre_type_id, post_type_id,
/// verify_confidence, verify_model, run_id, correction_ts.
///
/// Verifies: Migration 013 §013.2 — dream_pass4_audit table DDL.
/// Failure mode without fix: table does not exist → PRAGMA returns empty → all
/// column assertions fail.
#[tokio::test]
async fn b2_migration_013_creates_audit_table() {
    let (graph, _tmp) = open_graph().await;
    run_migrate_013(&graph).await;

    let mut rows = graph
        .conn
        .query("PRAGMA table_info(dream_pass4_audit)", ())
        .await
        .expect("PRAGMA table_info(dream_pass4_audit) must succeed");

    let mut columns: Vec<String> = Vec::new();
    while let Some(row) = rows.next().await.expect("row iteration must not error") {
        // PRAGMA table_info columns: cid(0), name(1), type(2), notnull(3), dflt_value(4), pk(5)
        let name: String = row.get(1).expect("column name at index 1");
        columns.push(name);
    }

    assert!(
        !columns.is_empty(),
        "dream_pass4_audit table must exist after migration 013; PRAGMA returned no columns"
    );

    let required = [
        "audit_id",
        "entity_id",
        "pre_type_id",
        "post_type_id",
        "verify_confidence",
        "verify_model",
        "run_id",
        "correction_ts",
    ];
    for col in &required {
        assert!(
            columns.contains(&col.to_string()),
            "dream_pass4_audit must have column '{col}'; actual columns: {columns:?}"
        );
    }

    // Verify column count: exactly 8 columns (no extras, no missing).
    assert_eq!(
        columns.len(),
        8,
        "dream_pass4_audit must have exactly 8 columns; got: {columns:?}"
    );
}

// ─── B3: Migration 013 idempotency ───────────────────────────────────────────

/// B3: Running migrate_013 twice must be a no-op on the second call.
/// The audit table must exist exactly once (no duplicate table error).
/// DreamPass4 INSERT must succeed after both runs (CHECK constraint intact).
///
/// Verifies: Migration 013 §013.3 — idempotency guard (sqlite_master LIKE '%DreamPass4%').
/// Failure mode without fix: second run fails with "table already exists" or
/// CHECK constraint conflict.
#[tokio::test]
async fn b3_migration_013_idempotent() {
    let (graph, _tmp) = open_graph().await;

    // First apply.
    kremory::core::migrations::migrate_013_pass4_source_tier(&graph.conn)
        .await
        .expect("first migrate_013 call must succeed");

    // Second apply — must be a no-op, not an error.
    kremory::core::migrations::migrate_013_pass4_source_tier(&graph.conn)
        .await
        .expect("second migrate_013 call must be idempotent (no-op)");

    // Audit table must exist exactly once (sqlite_master entry count = 1).
    let mut rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'table' AND name = 'dream_pass4_audit'",
            (),
        )
        .await
        .expect("sqlite_master query must succeed");
    let row = rows
        .next()
        .await
        .expect("COUNT(*) iteration must not error")
        .expect("COUNT(*) must return a row");
    let table_count: i64 = row.get(0).expect("count at index 0");
    assert_eq!(
        table_count, 1,
        "dream_pass4_audit must appear exactly once in sqlite_master after double apply; got count={table_count}"
    );

    // DreamPass4 CHECK must still function correctly after double apply.
    graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b3-dp4-ent', 1, '{}', datetime('now'), 'default', 'DreamPass4')",
            (),
        )
        .await
        .expect("DreamPass4 INSERT must still succeed after double migration 013 apply");
}

// ─── B4: Migration 012b downgrade round-trip ─────────────────────────────────

/// B4: Full downgrade round-trip.
/// Setup: apply migration 013, seed 5 DreamPass4 entities + matching audit rows.
/// Action: apply migration 012b (downgrade).
/// Assert:
///   - All 5 entities re-stamped to 'Phase1Ner' (dreampass4_count = 0).
///   - dream_pass4_audit table dropped.
///   - Subsequent INSERT with 'DreamPass4' fails (CHECK reverted).
///
/// Verifies: Migration 012b §012b.1 (re-stamp) + §012b.2 (revert CHECK) + §012b.3 (drop audit).
/// Failure mode without fix: DreamPass4 entities persist; audit table not dropped;
/// DreamPass4 INSERT still accepted.
#[tokio::test]
async fn b4_migration_012b_downgrade_round_trip() {
    let (graph, _tmp) = open_graph().await;
    run_migrate_013(&graph).await;

    // Seed 5 entities tagged DreamPass4.
    for i in 0..5_u32 {
        let entity_id = format!("b4-dp4-ent-{i}");

        // Insert entity with DreamPass4 source tier.
        graph
            .conn
            .execute(
                "INSERT INTO entities \
                 (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
                 VALUES (?1, 1, '{}', datetime('now'), 'default', 'DreamPass4')",
                libsql::params![entity_id.as_str()],
            )
            .await
            .unwrap_or_else(|e| panic!("seed entity {entity_id} must succeed: {e}"));

        // Get the integer rowid for the audit FK.
        let mut id_rows = graph
            .conn
            .query(
                "SELECT rowid FROM entities WHERE id = ?1",
                libsql::params![entity_id.as_str()],
            )
            .await
            .expect("SELECT rowid must succeed");
        let id_row = id_rows
            .next()
            .await
            .expect("rowid iteration must not error")
            .expect("entity rowid must exist");
        let rowid: i64 = id_row.get(0).expect("rowid at index 0");

        // Insert matching audit row (pre_type_id = 2 = "Person" placeholder).
        graph
            .conn
            .execute(
                "INSERT INTO dream_pass4_audit \
                 (entity_id, pre_type_id, post_type_id, verify_confidence, verify_model, run_id) \
                 VALUES (?1, 2, 1, 0.92, 'gemma4-e2b:latest', 'b4-test-run')",
                libsql::params![rowid],
            )
            .await
            .unwrap_or_else(|e| panic!("seed audit row for {entity_id} must succeed: {e}"));
    }

    // Verify pre-condition: 5 DreamPass4 entities present.
    let mut pre_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE entity_type_source = 'DreamPass4'",
            (),
        )
        .await
        .expect("pre-condition COUNT must succeed");
    let pre_count: i64 = pre_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        pre_count, 5,
        "pre-condition: 5 DreamPass4 entities must be present before downgrade; got {pre_count}"
    );
    // Drop pre_rows cursor before DDL — open Rows keeps the connection locked.
    drop(pre_rows);

    // Run downgrade.
    run_migrate_012b(&graph).await;

    // Assert 1: zero DreamPass4 entities remain.
    let mut dp4_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE entity_type_source = 'DreamPass4'",
            (),
        )
        .await
        .expect("post-downgrade DreamPass4 COUNT must succeed");
    let dp4_count: i64 = dp4_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        dp4_count, 0,
        "after downgrade, no entities must retain entity_type_source='DreamPass4'; got {dp4_count}"
    );

    // Assert 2: all 5 re-stamped entities now have Phase1Ner.
    let mut ner_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE entity_type_source = 'Phase1Ner' AND id LIKE 'b4-dp4-ent-%'",
            (),
        )
        .await
        .expect("Phase1Ner COUNT must succeed");
    let ner_count: i64 = ner_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        ner_count, 5,
        "after downgrade, all 5 seeded entities must be re-stamped to Phase1Ner; got {ner_count}"
    );

    // Assert 3: dream_pass4_audit table dropped (table must not exist in sqlite_master).
    let mut audit_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'table' AND name = 'dream_pass4_audit'",
            (),
        )
        .await
        .expect("sqlite_master audit table query must succeed");
    let audit_table_count: i64 = audit_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        audit_table_count, 0,
        "dream_pass4_audit must be dropped after migration 012b; sqlite_master still shows {audit_table_count} entries"
    );

    // Assert 4: DreamPass4 INSERT must now fail (CHECK reverted to pre-013 constraint).
    let revert_result = graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b4-post-downgrade-test', 1, '{}', datetime('now'), 'default', 'DreamPass4')",
            (),
        )
        .await;
    assert!(
        revert_result.is_err(),
        "INSERT with entity_type_source='DreamPass4' must fail after migration 012b downgrade (CHECK reverted)"
    );
}

// ─── B5: Migration 012b idempotency ──────────────────────────────────────────

/// B5: Running migrate_012b twice after a downgrade must be a no-op on the second call.
/// No error on the second call; DreamPass4 INSERT still fails both times.
///
/// Verifies: Migration 012b §012b.4 — idempotency guard.
/// Failure mode without fix: second 012b call panics or errors due to missing table.
#[tokio::test]
async fn b5_migration_012b_idempotent() {
    let (graph, _tmp) = open_graph().await;

    // Apply forward migration first.
    kremory::core::migrations::migrate_013_pass4_source_tier(&graph.conn)
        .await
        .expect("migrate_013 must succeed for 012b to have something to downgrade");

    // First downgrade.
    kremory::core::migrations::migrate_012b_revert_pass4_source_tier(&graph.conn)
        .await
        .expect("first migrate_012b call must succeed");

    // Second downgrade — must be a no-op, not an error.
    kremory::core::migrations::migrate_012b_revert_pass4_source_tier(&graph.conn)
        .await
        .expect(
            "second migrate_012b call must be idempotent (DreamPass4 already absent from CHECK)",
        );

    // Post-double-downgrade: DreamPass4 INSERT must still fail.
    let result = graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b5-post-double-downgrade', 1, '{}', datetime('now'), 'default', 'DreamPass4')",
            (),
        )
        .await;
    assert!(
        result.is_err(),
        "DreamPass4 INSERT must fail after double 012b downgrade (CHECK must remain without DreamPass4)"
    );
}

// ─── B6: Full migration chain round-trip ─────────────────────────────────────

/// B6: Full migration chain test.
/// open() runs migrations 001–014 automatically. Then apply 013 on top.
/// Asserts that:
///   - entity_type_source column is present (migration 012 invariant preserved).
///   - All 5 pre-012 valid values still accepted (Phase1Ner, Phase2Llm, DreamPass0,
///     DreamPass1, ConsumerPinned) — migration 013 must NOT break pre-existing values.
///   - DreamPass4 is now additionally accepted (migration 013 adds it).
///   - entity_types table exists (migration 008/010 invariant preserved).
///   - dream_pass4_audit table exists with correct schema (migration 013 invariant).
///
/// Verifies: migration 013 layers cleanly on top of the existing 012 + 014 chain
/// without breaking any established schema invariant.
/// Failure mode without fix: migration 013 table-rebuild drops entity_type_id FK
/// references, or the check constraint drop is incomplete, or pre-existing valid
/// values are rejected after rebuild.
#[tokio::test]
async fn b6_migration_round_trip_full_chain() {
    let (graph, _tmp) = open_graph().await;
    // open() has already applied: 001–009, 010, 011, 012, 014.
    // Now apply 013 on top.
    run_migrate_013(&graph).await;

    // --- Invariant 1: entity_type_source column present ---
    let mut pragma_rows = graph
        .conn
        .query("PRAGMA table_info('entities')", ())
        .await
        .expect("PRAGMA table_info(entities) must succeed");
    let mut entity_cols: Vec<String> = Vec::new();
    while let Some(row) = pragma_rows
        .next()
        .await
        .expect("PRAGMA row iteration must not error")
    {
        let name: String = row.get(1).expect("col name at index 1");
        entity_cols.push(name);
    }
    assert!(
        entity_cols.iter().any(|c| c == "entity_type_source"),
        "entities must have entity_type_source after full chain; got: {entity_cols:?}"
    );
    assert!(
        entity_cols.iter().any(|c| c == "entity_type_id"),
        "entities must have entity_type_id after full chain; got: {entity_cols:?}"
    );

    // --- Invariant 2: pre-012 valid values still accepted after 013 ---
    let pre_013_valid = [
        ("b6-phase1ner", "Phase1Ner"),
        ("b6-phase2llm", "Phase2Llm"),
        ("b6-dreampass0", "DreamPass0"),
        ("b6-dreampass1", "DreamPass1"),
        ("b6-consumerpinned", "ConsumerPinned"),
    ];
    for (id, tier) in &pre_013_valid {
        graph
            .conn
            .execute(
                "INSERT INTO entities \
                 (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
                 VALUES (?1, 1, '{}', datetime('now'), 'default', ?2)",
                libsql::params![*id, *tier],
            )
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "Pre-013 value '{tier}' must still be accepted after migration 013 chain: {e}"
                )
            });
    }

    // --- Invariant 3: DreamPass4 now additionally accepted ---
    graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b6-dreampass4', 1, '{}', datetime('now'), 'default', 'DreamPass4')",
            (),
        )
        .await
        .expect("DreamPass4 must be accepted after migration 013 in full chain");

    // --- Invariant 4: GarbageValue still rejected ---
    let garbage_result = graph
        .conn
        .execute(
            "INSERT INTO entities \
             (id, entity_type_id, properties, recorded_at, group_id, entity_type_source) \
             VALUES ('b6-garbage', 1, '{}', datetime('now'), 'default', 'TotallyInvalid')",
            (),
        )
        .await;
    assert!(
        garbage_result.is_err(),
        "TotallyInvalid entity_type_source must still fail CHECK after full chain"
    );

    // --- Invariant 5: entity_types table exists (migration 008/010 not broken) ---
    let mut et_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='entity_types'",
            (),
        )
        .await
        .expect("sqlite_master query for entity_types must succeed");
    let et_count: i64 = et_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        et_count, 1,
        "entity_types table must exist after full chain; got count={et_count}"
    );

    // --- Invariant 6: dream_pass4_audit table exists ---
    let mut audit_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='dream_pass4_audit'",
            (),
        )
        .await
        .expect("sqlite_master query for dream_pass4_audit must succeed");
    let audit_count: i64 = audit_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        audit_count, 1,
        "dream_pass4_audit must exist in sqlite_master after full chain; got count={audit_count}"
    );

    // --- Invariant 7: audit table INSERT round-trip ---
    // Insert a DreamPass4 entity first, then add an audit row to verify FK integrity.
    let mut entity_rowid_rows = graph
        .conn
        .query("SELECT rowid FROM entities WHERE id = 'b6-dreampass4'", ())
        .await
        .expect("SELECT rowid must succeed");
    let entity_rowid: i64 = entity_rowid_rows
        .next()
        .await
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    graph
        .conn
        .execute(
            "INSERT INTO dream_pass4_audit \
             (entity_id, pre_type_id, post_type_id, verify_confidence, verify_model, run_id) \
             VALUES (?1, 2, 1, 0.88, 'gemma4-e2b:latest', 'b6-integration-run')",
            libsql::params![entity_rowid],
        )
        .await
        .expect("audit row INSERT must succeed in full chain context");

    let mut audit_count_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM dream_pass4_audit WHERE run_id = 'b6-integration-run'",
            (),
        )
        .await
        .expect("audit COUNT must succeed");
    let audit_row_count: i64 = audit_count_rows
        .next()
        .await
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(
        audit_row_count, 1,
        "dream_pass4_audit must contain the inserted audit row; got count={audit_row_count}"
    );
}

// ─── B7: cascade-delete trigger (ADR-047 §IRREV-001 + Quinn REQ-001) ─────────

/// B7: After migrate_013, the AFTER DELETE trigger
/// `trg_dream_pass4_audit_cascade_delete` mirrors `ON DELETE CASCADE` semantic
/// without a FK constraint (the composite PK on `entities` makes single-column
/// FK on rowid structurally impossible per ADR-047 §IRREV-001 amendment).
///
/// Verifies: when an entity is deleted, all rows in `dream_pass4_audit`
/// referencing that entity's rowid are deleted automatically.
///
/// Failure mode without trigger: deleting an entity leaves orphan audit rows
/// (provenance corruption that ADR-047 IRREV-001 amendment was designed to prevent).
#[tokio::test]
async fn b7_dream_pass4_audit_cascade_delete_trigger_fires() {
    let (graph, _tmp) = open_graph().await;
    run_migrate_013(&graph).await;

    // ── Seed: one DreamPass4 entity + two audit rows referencing it ──
    let entity_rowid = seed_entity(&graph, "b7-cascade-ent", "DreamPass4").await;

    for run_id in &["b7-run-1", "b7-run-2"] {
        graph
            .conn
            .execute(
                "INSERT INTO dream_pass4_audit \
                 (entity_id, pre_type_id, post_type_id, verify_confidence, verify_model, run_id) \
                 VALUES (?1, 2, 1, 0.9, 'gemma4-e2b:latest', ?2)",
                libsql::params![entity_rowid, *run_id],
            )
            .await
            .expect("audit row INSERT must succeed");
    }

    // Pre-condition: 2 audit rows exist for this entity.
    let mut pre_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM dream_pass4_audit WHERE entity_id = ?1",
            libsql::params![entity_rowid],
        )
        .await
        .expect("pre-delete COUNT must succeed");
    let pre_count: i64 = pre_rows.next().await.unwrap().unwrap().get(0).unwrap();
    drop(pre_rows);
    assert_eq!(pre_count, 2, "pre-condition: 2 audit rows must exist");

    // ── Delete the entity — trigger should cascade-delete the audit rows ──
    graph
        .conn
        .execute("DELETE FROM entities WHERE id = 'b7-cascade-ent'", ())
        .await
        .expect("DELETE entity must succeed");

    // ── Post-condition: 0 audit rows remain for this entity ──
    let mut post_rows = graph
        .conn
        .query(
            "SELECT COUNT(*) FROM dream_pass4_audit WHERE entity_id = ?1",
            libsql::params![entity_rowid],
        )
        .await
        .expect("post-delete COUNT must succeed");
    let post_count: i64 = post_rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        post_count, 0,
        "trigger trg_dream_pass4_audit_cascade_delete must remove all audit rows for the deleted entity; got post_count={post_count}"
    );
}
