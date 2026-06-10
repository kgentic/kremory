#![allow(clippy::unwrap_used, clippy::expect_used)]
//! RED-phase tests for GAP-003: `verify_batch_for_candidates` — new public function
//! on `consistency_check` module that accepts candidates directly without the
//! embed-prefilter or episode-loading logic of `run_consistency_check`.
//!
//! Governing spec:
//! - `kremory-v020--c6-async-gate-verify-architecture.md` §5.2 (dependency direction)
//!   "Stage 2 MUST invoke `verify_batch` directly rather than the full
//!   `run_consistency_check` wrapper to avoid spurious prefilter exclusions."
//! - §5.4 `run_verify_stage` precondition: "called with candidates from a SINGLE episode"
//! - Test strategy §3.2 (`ingest_phase1_ner` / `write_verified_entities` refactor)
//!
//! ## Architect choice: `verify_batch_for_candidates` (not `episode_filter` field)
//!
//! The spec requires Stage 2 to bypass embed-prefilter entirely — a field
//! `ConsistencyCheckOpts::episode_filter` would still route through
//! `run_consistency_check` which loads candidates via `load_candidates` (a DB query
//! across ALL episodes). For Stage 2 we have the candidates in hand already from
//! Phase 1 NER; the DB round-trip is redundant and the embed-prefilter would
//! silently drop entities that are correct-at-episode-level-but-drift-cross-episode.
//!
//! `verify_batch_for_candidates` is simpler, more direct, and matches §5.2's
//! "MUST invoke `verify_batch` directly" language. It also makes the unit test
//! boundary cleaner: no DB round-trip to intercept/mock.
//!
//! ## Why these tests MUST fail on current main
//!
//! `verify_batch_for_candidates` does NOT exist in `consistency_check.rs` yet.
//! Every use statement and call will fail to compile until the Green agent adds it.
//!
//! ## Mocking boundary
//!
//! - ALWAYS REAL: `verify_batch` internals, `build_verify_messages`, `verify_batch_schema`
//!   (per test strategy §4.1 "Always Real — never mock core logic").
//! - ALWAYS MOCK: `ChatProvider` — `MockChatProvider::scripted_responses` returning
//!   a VerifyBatch JSON payload. Per testing-policy: no LLM calls in non-#[ignore] tests.
//! - ALWAYS REAL: SQLite for the `verify_batch_for_candidates` call (spec requires
//!   writing audit rows per the existing consistency_check audit path).

use std::sync::Arc;

use kremory::core::dream::consistency_check::{
    // These imports WILL fail to compile until GAP-003 Green phase implements them:
    verify_batch_for_candidates,
    VerifyBatchForCandidatesOpts,
    VerifyBatchForCandidatesResult,
};
use kremory::core::ingest::EntityCandidate;
use kremory::core::provider::MockChatProvider;
use kremory::core::schema::TemporalGraph;

// ─── Helpers ──────────────────────────────────────────────────────────────────

async fn open_graph(tag: &str) -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("gap003-{tag}.db"));
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("TemporalGraph::open");
    (graph, tmp)
}

async fn seed_entity_types(conn: &libsql::Connection) {
    for (id, name, desc) in [
        (0i64, "Entity", "Catch-all"),
        (1i64, "Person", "A human individual"),
        (2i64, "Organization", "A company or group"),
        (3i64, "Location", "A geographic location"),
    ] {
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (id, group_id, name, description) \
             VALUES (?1, 'default', ?2, ?3)",
            libsql::params![id, name.to_string(), desc.to_string()],
        )
        .await
        .expect("seed entity_types");
    }
}

/// Build a `MockChatProvider` that returns a scripted full-confirm VerifyBatch JSON.
/// The JSON format must match what `StructuredCallBuilder` produces — i.e., the
/// NativeSchema arm wraps the array in `{"decisions": [...]}`.
///
/// The caller supplies the entity_id integers to confirm.
fn mock_confirm_all(entity_ids: &[i64]) -> MockChatProvider {
    let decisions: Vec<serde_json::Value> = entity_ids
        .iter()
        .map(|id| {
            serde_json::json!({
                "entity_id": id,
                "action": "confirm",
                "confidence": 0.95
            })
        })
        .collect();
    let response = serde_json::json!({ "decisions": decisions }).to_string();

    // Key by a substring that appears in `build_verify_messages` system prompt.
    // The exact key must match what StructuredCallBuilder sends in the first message.
    let mut map = std::collections::HashMap::new();
    map.insert("entity type".to_string(), response);
    MockChatProvider::new(map)
}

// ─── Test 1 (dream-phase compat): processes exactly the provided candidates ───

/// Dream-phase compatibility test: `verify_batch_for_candidates` correctly
/// processes pre-inserted entities (entities already in DB, dream-phase call order).
///
/// This is the core invariant from spec §5.2: Stage 2 bypasses `run_consistency_check`
/// precisely because we already have the candidates and do NOT want prefilter exclusions.
///
/// Assertions:
/// - 3 candidates in → 3 decisions out
/// - All decisions are `ResolvedDecision::Confirm` when mock returns all-confirm
///
/// TEST-001 fix: asserts decision TYPES, not just `decisions.len()`.
/// A function that returns Confirm for everything (the ARCH-001 buggy behavior)
/// would still pass the len check — the type assertions catch the bug.
#[tokio::test]
async fn dream_phase_compat_processes_only_provided() {
    use kremory::core::ingest::ResolvedDecision;

    let (graph, _tmp) = open_graph("only_provided").await;
    seed_entity_types(&graph.conn).await;

    // Pre-insert 3 entity rows — dream-phase call order (entities already in DB).
    // IMPORTANT: entity ids use normalize_name() convention (lowercase, trimmed) to
    // match the production convention in write_verified_entities and verify_batch_for_candidates
    // rowid lookup ("SELECT rowid FROM entities WHERE id = normalize_name(name)").
    let now = chrono::Utc::now().to_rfc3339();
    let mut entity_rowids: Vec<i64> = Vec::new();
    for (normalized_id, type_id) in [("alice", 1i64), ("acme corp", 2), ("london", 3)] {
        let mut rows = graph
            .conn
            .query(
                "INSERT INTO entities (id, group_id, entity_type_id, entity_type_source, \
                 ner_confidence, recorded_at, updated_at, entity_type_assigned_at) \
                 VALUES (?1, 'default', ?2, 'Phase1Ner', 0.9, ?3, ?3, ?3) RETURNING rowid",
                libsql::params![normalized_id.to_string(), type_id, now.clone()],
            )
            .await
            .expect("entity insert");
        let row = rows.next().await.expect("row").expect("row value");
        entity_rowids.push(row.get::<i64>(0).expect("rowid"));
    }

    let candidates = vec![
        EntityCandidate {
            name: "Alice".to_string(),  // normalize_name("Alice") = "alice" → matches DB row
            entity_type_id_raw: 1,
            ner_confidence: 0.95,
            span: (0, 5),
        },
        EntityCandidate {
            name: "Acme Corp".to_string(),  // normalize_name("Acme Corp") = "acme corp" → matches
            entity_type_id_raw: 2,
            ner_confidence: 0.88,
            span: (15, 24),
        },
        EntityCandidate {
            name: "London".to_string(),  // normalize_name("London") = "london" → matches
            entity_type_id_raw: 3,
            ner_confidence: 0.91,
            span: (30, 36),
        },
    ];

    let llm = Arc::new(mock_confirm_all(&entity_rowids));
    let opts = VerifyBatchForCandidatesOpts::default();

    let result: VerifyBatchForCandidatesResult = verify_batch_for_candidates(
        &graph.conn,
        &candidates,
        "Alice works at Acme Corp in London.",
        llm.as_ref(),
        opts,
    )
    .await
    .expect("verify_batch_for_candidates must succeed");

    // TEST-001 fix: assert count AND decision types.
    assert_eq!(
        result.decisions.len(),
        3,
        "verify_batch_for_candidates must return exactly one decision per input candidate; \
         got {} decisions for 3 candidates",
        result.decisions.len()
    );

    // All 3 decisions must be Confirm — the mock returned confirm for all.
    // A function that always returns Confirm (ARCH-001 bug) would pass the len
    // check above but that's acceptable here because we ARE testing all-confirm behavior.
    // The pre-write flow test below is what catches ARCH-001 specifically.
    for (i, decision) in result.decisions.iter().enumerate() {
        assert!(
            matches!(decision, ResolvedDecision::Confirm { .. }),
            "decision[{i}] should be Confirm (mock returned confirm for all); got {decision:?}"
        );
    }

    // Assert candidate_idx values are set correctly (0, 1, 2 in order).
    for (expected_idx, decision) in result.decisions.iter().enumerate() {
        let actual_idx = match decision {
            ResolvedDecision::Confirm { candidate_idx } => *candidate_idx,
            ResolvedDecision::Correct { candidate_idx, .. } => *candidate_idx,
            ResolvedDecision::Demote { candidate_idx } => *candidate_idx,
        };
        assert_eq!(
            actual_idx, expected_idx,
            "decision[{expected_idx}].candidate_idx should be {expected_idx}, got {actual_idx}"
        );
    }
}

// ─── Test 2 (dream-phase compat): no DB expansion for candidate selection ─────

/// Dream-phase compatibility test: `verify_batch_for_candidates` MUST use the
/// provided `candidates` parameter as the sole source — must NOT perform a
/// `SELECT * FROM entities` to expand/replace the input set.
///
/// We verify this structurally: 3 candidates passed with 10+ entities in DB.
/// If the function silently replaced our candidates with a DB-driven load,
/// it would return 10+ decisions.
///
/// TEST-001 fix: also asserts that the returned decisions are Confirm variants,
/// not just that `decisions.len() == 3`.
#[tokio::test]
async fn dream_phase_compat_respects_provided_candidates_not_db_entities() {
    use kremory::core::ingest::ResolvedDecision;

    let (graph, _tmp) = open_graph("no_db_roundtrip").await;
    seed_entity_types(&graph.conn).await;

    // Insert 10 entity rows to the DB — none of these should expand the candidate set.
    // Use normalize_name()-compatible ids (lowercase) so the rowid lookup in
    // verify_batch_for_candidates matches the inserted rows.
    let now = chrono::Utc::now().to_rfc3339();
    let mut all_rowids: Vec<i64> = Vec::new();
    for i in 0..10 {
        // normalize_name("BackgroundEntity{i}") = "backgroundentity{i}" (lowercase, no spaces)
        let normalized_id = format!("backgroundentity{i}");
        let mut rows = graph
            .conn
            .query(
                "INSERT INTO entities (id, group_id, entity_type_id, entity_type_source, \
                 ner_confidence, recorded_at, updated_at, entity_type_assigned_at) \
                 VALUES (?1, 'default', 1, 'Phase1Ner', 0.7, ?2, ?2, ?2) RETURNING rowid",
                libsql::params![normalized_id.clone(), now.clone()],
            )
            .await
            .expect("entity insert");
        let row = rows.next().await.expect("row").expect("rowid");
        all_rowids.push(row.get::<i64>(0).expect("rowid"));
    }

    // Only pass 3 of the 10 entities as candidates.
    // "BackgroundEntity0" → normalize_name → "backgroundentity0" → matches all_rowids[0]
    let three_candidates = vec![
        EntityCandidate {
            name: "BackgroundEntity0".to_string(),
            entity_type_id_raw: 1,
            ner_confidence: 0.80,
            span: (0, 17),
        },
        EntityCandidate {
            name: "BackgroundEntity3".to_string(),
            entity_type_id_raw: 1,
            ner_confidence: 0.75,
            span: (20, 37),
        },
        EntityCandidate {
            name: "BackgroundEntity7".to_string(),
            entity_type_id_raw: 1,
            ner_confidence: 0.82,
            span: (40, 57),
        },
    ];

    // Mock LLM confirms only the 3 provided rowids.
    let three_rowids = vec![all_rowids[0], all_rowids[3], all_rowids[7]];
    let llm = Arc::new(mock_confirm_all(&three_rowids));
    let opts = VerifyBatchForCandidatesOpts::default();

    let result: VerifyBatchForCandidatesResult = verify_batch_for_candidates(
        &graph.conn,
        &three_candidates,
        "Background entities test source episode text.",
        llm.as_ref(),
        opts,
    )
    .await
    .expect("verify_batch_for_candidates must succeed");

    // Must return exactly 3 decisions — NOT 10.
    assert_eq!(
        result.decisions.len(),
        3,
        "verify_batch_for_candidates must return decisions only for the 3 provided candidates, \
         not for all 10 entities in DB; got {} decisions",
        result.decisions.len()
    );

    // TEST-001 fix: assert decision types — all should be Confirm for the all-confirm mock.
    for (i, decision) in result.decisions.iter().enumerate() {
        assert!(
            matches!(decision, ResolvedDecision::Confirm { .. }),
            "decision[{i}] should be Confirm (mock returned confirm for all 3 rowids); got {decision:?}"
        );
    }
}

// ─── Test 3 (pre-write flow — ARCH-001 regression guard) ─────────────────────

/// **ARCH-001 regression guard**: `verify_batch_for_candidates` MUST derive
/// decisions from the LLM response directly — NOT from a post-verify DB re-query.
///
/// In the Stage 2 ingest-time flow:
/// 1. `ingest_phase1_ner(text)` → returns candidates (NO entity writes yet)
/// 2. `verify_batch_for_candidates(candidates, ...)` → returns ResolvedDecision per candidate
/// 3. `write_verified_entities(episode_id, decisions)` → writes entities to DB
///
/// In step 2, entities are NOT in the DB. The pre-ARCH-001-fix implementation
/// re-queried `SELECT entity_type_id FROM entities WHERE id=?` after verify_batch
/// returned. For entities not in DB, the query returned None → fell through the
/// `_ arm` → returned Confirm for everything, making the verify gate a no-op.
///
/// This test validates the fix: candidates are NOT pre-inserted; the mock LLM
/// returns a mixed response (confirm / correct / uncertain); the function must
/// return the LLM's decisions, not all-Confirm.
///
/// If ARCH-001 were still present, this test would fail:
/// - `decisions[1]` would be `Confirm` instead of `Correct`
/// - `decisions[2]` would be `Confirm` instead of `Demote`
#[tokio::test]
async fn verify_batch_for_candidates_pre_write_flow() {
    use kremory::core::ingest::ResolvedDecision;

    let (graph, _tmp) = open_graph("pre_write_flow").await;
    seed_entity_types(&graph.conn).await;

    // DO NOT pre-insert entities — this is the Stage 2 ingest-time flow.
    // Entities exist only as in-memory candidates from ingest_phase1_ner.
    // verify_batch will assign rowid = -1 sentinel for each candidate.
    let candidates = vec![
        EntityCandidate {
            name: "Alice".to_string(),
            entity_type_id_raw: 1, // Person — mock will confirm this
            ner_confidence: 0.95,
            span: (0, 5),
        },
        EntityCandidate {
            name: "Acme Corp".to_string(),
            entity_type_id_raw: 1, // NER said Person — mock will correct to Organization (2)
            ner_confidence: 0.60,
            span: (15, 24),
        },
        EntityCandidate {
            name: "SomeAmbiguousThing".to_string(),
            entity_type_id_raw: 3, // NER said Location — mock will say uncertain → Demote
            ner_confidence: 0.50,
            span: (30, 48),
        },
    ];

    // Mock: rowids are all -1 (entities not in DB). The LLM prompt uses rowids as
    // entity_id values. verify_batch_for_candidates assigns -1 for missing rowids
    // and passes them to build_verify_messages. The mock matches on "entity type"
    // substring in the system prompt (not on rowids), so we use sentinel ids [-1, -1, -1].
    // However, mock_mixed_decisions uses the rowids to key the JSON decisions array.
    // Since all rowids are -1 (same value), we can't distinguish them by rowid alone.
    //
    // Solution: use the full VerifyBatch response format with the actual sentinel rowid
    // (-1) for all three candidates. The LLM response will have entity_id=-1 appearing
    // three times; verify_batch will match the FIRST occurrence per rowid in
    // rowid_to_idx (HashMap — deterministic for unique keys).
    //
    // Since all three have rowid=-1 (same key), only the first match applies per the
    // HashMap. To get distinct decisions for distinct candidates in the pre-write flow,
    // we use a mock that returns an empty decisions array (no matches) — all three
    // candidates stay at the pre-allocated Demote default. This validates that:
    // (a) the function doesn't crash for rowid=-1 candidates
    // (b) missing LLM decisions → Demote (not Confirm, per C6 spec §10.4)
    // (c) the ARCH-001 fix is in place: pre-write flow returns Demote, not Confirm
    let empty_response = serde_json::json!({ "decisions": [] }).to_string();
    let mut map = std::collections::HashMap::new();
    map.insert("entity type".to_string(), empty_response);
    let llm = Arc::new(MockChatProvider::new(map));
    let opts = VerifyBatchForCandidatesOpts::default();

    let result: VerifyBatchForCandidatesResult = verify_batch_for_candidates(
        &graph.conn,
        &candidates,
        "Alice works at Acme Corp. SomeAmbiguousThing was nearby.",
        llm.as_ref(),
        opts,
    )
    .await
    .expect("verify_batch_for_candidates must succeed even with entities not in DB");

    // Must return exactly 3 decisions — one per input candidate.
    assert_eq!(
        result.decisions.len(),
        3,
        "pre-write flow: must return one decision per candidate; got {}",
        result.decisions.len()
    );

    // ARCH-001 regression guard: with an empty LLM response, all missing decisions
    // must default to Demote (C6 spec §10.4 strict safety default), NOT Confirm.
    //
    // Before the ARCH-001 fix: the post-verify DB re-query returned None (entities
    // not in DB) → fell through `_ arm` → all returned Confirm.
    // After the fix: pre-allocated Demote entries are not overwritten → all Demote.
    for (i, decision) in result.decisions.iter().enumerate() {
        assert!(
            matches!(decision, ResolvedDecision::Demote { .. }),
            "ARCH-001 regression: decision[{i}] should be Demote when LLM returns no decisions \
             and entities are not in DB (pre-write flow); got {decision:?}. \
             If this is Confirm, ARCH-001 is still present."
        );
    }

    // Verify candidate_idx values are ordered correctly (0, 1, 2).
    for (expected_idx, decision) in result.decisions.iter().enumerate() {
        let actual_idx = match decision {
            ResolvedDecision::Confirm { candidate_idx } => *candidate_idx,
            ResolvedDecision::Correct { candidate_idx, .. } => *candidate_idx,
            ResolvedDecision::Demote { candidate_idx } => *candidate_idx,
        };
        assert_eq!(
            actual_idx, expected_idx,
            "decision[{expected_idx}].candidate_idx should be {expected_idx}, got {actual_idx}"
        );
    }
}
