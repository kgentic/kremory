#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Migration 019 (ADR-066 spec §2) — dream CONSOLIDATION substrate tables.
//!
//! Verifies:
//! - the three new tables (`facts_archive`, `entity_communities`,
//!   `community_summaries`) exist after `open_in_memory` runs the migration suite;
//! - a second `run_migrations` pass is a no-op (CREATE TABLE/INDEX IF NOT EXISTS
//!   idempotency, spec §2 / R-13) — no duplicate-table or index error.
//!
//! These are OPTIONAL feature tables and are intentionally NOT in
//! `check_integrity`'s `CRITICAL_TABLES` (spec §2) — their absence on an old db must
//! degrade gracefully, not raise `CorruptStore`. This test only asserts presence +
//! re-run safety, not integrity-gating.

use kremory::core::schema::TemporalGraph;

/// `SELECT 1 FROM sqlite_master WHERE type='table' AND name=?` → bool.
async fn table_exists(graph: &TemporalGraph, name: &str) -> bool {
    let mut rows = graph
        .conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1",
            libsql::params![name],
        )
        .await
        .expect("sqlite_master query must succeed");
    rows.next()
        .await
        .expect("iteration must not error")
        .is_some()
}

#[tokio::test]
async fn migration_019_creates_consolidation_tables() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");
    for table in ["facts_archive", "entity_communities", "community_summaries"] {
        assert!(
            table_exists(&graph, table).await,
            "migration 019 must create table `{table}`"
        );
    }
}

#[tokio::test]
async fn migration_019_idempotent_double_apply() {
    let graph = TemporalGraph::open_in_memory().await.expect("open");

    // Second full-suite run — migration 019's CREATE ... IF NOT EXISTS statements
    // must not error on already-present tables/indexes (R-13 stop condition).
    graph.run_migrations_again_for_test().await.expect(
        "second run_migrations must be idempotent for migration 019 — a \
         duplicate-table/index error means the IF NOT EXISTS gate is missing",
    );

    // Tables still present after the second run.
    for table in ["facts_archive", "entity_communities", "community_summaries"] {
        assert!(
            table_exists(&graph, table).await,
            "table `{table}` must survive a second migration run"
        );
    }
}
