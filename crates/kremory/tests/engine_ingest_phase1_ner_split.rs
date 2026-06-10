#![allow(clippy::unwrap_used, clippy::expect_used)]
//! RED-phase tests for GAP-001: Engine::ingest split into two pub(crate) functions.
//!
//! Governing spec:
//! - `kremory-v020--c6-async-gate-verify-architecture.md` §5.5 (worker loop insertion point)
//! - Test strategy `test-strategy-kremory-v020-c6-async-gate-2026-06-10.md` §3.2 + §5
//!
//! ## What these tests verify
//!
//! - `Engine::ingest_phase1_ner()` emits exactly ONE episode row and ZERO entity rows,
//!   and returns `IngestPhase1Result { candidates, episode_id }`.
//! - `Engine::write_verified_entities()` correctly writes entities per `ResolvedDecision`
//!   variant (Confirm → entity_type_id_raw, Correct → new_type_id, Demote → 0).
//! - The existing `Engine::ingest_with()` path still works end-to-end (backwards compat).
//!
//! ## Why these tests MUST fail on current main
//!
//! `ingest_phase1_ner`, `write_verified_entities`, `IngestPhase1Result`, and
//! `ResolvedDecision` do NOT exist in `crates/kremory/src/core/ingest/pipeline.rs` yet.
//! Every test here fails to compile until the Green agent adds them.
//!
//! ## Mocking boundary
//!
//! - ALWAYS REAL: SQLite via `TemporalGraph::open` + `tempfile::TempDir`
//! - ALWAYS MOCK: `ChatProvider` — uses `MockChatProvider::null()` (panics if called,
//!   giving a structural guarantee Stage 1 does NOT invoke LLM for NER).
//! - NER feature: tests that exercise GLiNER path require `--features ner` but the
//!   backwards-compat smoke test can run without it.
//!
//! Per `feedback_no_second_llm_pass_for_entity_extraction`: these tests do NOT invoke
//! any LLM for entity extraction. `ingest_phase1_ner` uses GLiNER (NER feature) or
//! returns empty candidates without it.

use std::sync::Arc;

use kremory::core::config::PipelineConfig;
use kremory::core::extraction::LlmExtractor;
use kremory::core::ingest::{Engine, IngestionResult, SourceParams};
// These imports WILL fail to compile until GAP-001 Green phase implements them:
use kremory::core::ingest::{EntityCandidate, IngestPhase1Result, ResolvedDecision};
use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;

// ─── Helpers ──────────────────────────────────────────────────────────────────

async fn open_graph(tag: &str) -> (Arc<TemporalGraph>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join(format!("engine-phase1-{tag}.db"));
    let graph = TemporalGraph::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("TemporalGraph::open");
    (Arc::new(graph), tmp)
}

fn null_llm() -> Arc<MockChatProvider> {
    Arc::new(MockChatProvider::null())
}

fn null_embedder() -> Arc<NullEmbeddingProvider> {
    Arc::new(NullEmbeddingProvider { dim: 384 })
}

async fn count_table_rows(conn: &libsql::Connection, table: &str) -> i64 {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let mut rows = conn
        .query(&sql, libsql::params![])
        .await
        .expect("count query");
    let row = rows
        .next()
        .await
        .expect("count row iter")
        .expect("count row");
    row.get::<i64>(0).expect("count col")
}

/// Seed a minimal entity_type registry so GLiNER int-ID lookup has something to resolve to.
async fn seed_entity_types(conn: &libsql::Connection) {
    // id=0 catch-all (required), id=1 Person, id=2 Organization
    for (id, name, desc) in [
        (0i64, "Entity", "Catch-all entity type"),
        (1i64, "Person", "A human individual"),
        (2i64, "Organization", "A company or group"),
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

// ─── Test 1: phase1_ner writes only episode, returns candidates ───────────────

/// `ingest_phase1_ner` MUST write exactly ONE episodes row and ZERO entities rows.
///
/// The returned `IngestPhase1Result.episode_id` must match the row written.
/// The returned `IngestPhase1Result.candidates` may be empty (no GLiNER feature)
/// or non-empty (GLiNER active) — we assert it's non-None (the function returned Ok).
///
/// This test FAILS TO COMPILE on current main because `ingest_phase1_ner` and
/// `IngestPhase1Result` do not exist yet.
#[tokio::test]
async fn phase1_ner_writes_only_episode_returns_candidates() {
    let (graph, _tmp) = open_graph("phase1_only").await;
    seed_entity_types(&graph.conn).await;

    let config = PipelineConfig::builder()
        .build()
        .expect("default PipelineConfig");
    let engine = Engine::new(Arc::clone(&graph), null_llm(), null_embedder(), config);

    let episode_count_before = count_table_rows(&graph.conn, "episodes").await;
    let entity_count_before = count_table_rows(&graph.conn, "entities").await;

    // This call MUST NOT exist on current main — compile error expected.
    let result: IngestPhase1Result = engine
        .ingest_phase1_ner("Alice works at Acme Corp.", SourceParams::default())
        .await
        .expect("ingest_phase1_ner must succeed");

    let episode_count_after = count_table_rows(&graph.conn, "episodes").await;
    let entity_count_after = count_table_rows(&graph.conn, "entities").await;

    // Exactly ONE episode row written.
    assert_eq!(
        episode_count_after - episode_count_before,
        1,
        "ingest_phase1_ner must write exactly 1 episode row"
    );

    // ZERO entity rows written — entities are written ONLY after verify stage.
    assert_eq!(
        entity_count_after,
        entity_count_before,
        "ingest_phase1_ner must NOT write any entity rows; entity writes belong to write_verified_entities"
    );

    // episode_id is a valid non-zero integer.
    assert!(
        result.episode_id > 0,
        "IngestPhase1Result.episode_id must be positive, got {}",
        result.episode_id
    );

    // candidates is a Vec (possibly empty if no NER feature, but the field must exist).
    // We just assert the field is accessible — the type check is the meaningful assertion here.
    let _candidates: Vec<EntityCandidate> = result.candidates;
}

// ─── Test 2: write_verified_entities writes entities per decision ─────────────

/// `write_verified_entities` MUST write entity rows according to `ResolvedDecision` semantics:
/// - `Confirm { candidate_idx }` → entity_type_id = candidate.entity_type_id_raw
/// - `Correct { candidate_idx, new_type_id }` → entity_type_id = new_type_id
/// - `Demote { candidate_idx }` → entity_type_id = 0
///
/// This test FAILS TO COMPILE on current main because `write_verified_entities`,
/// `IngestPhase1Result`, `ResolvedDecision`, and `EntityCandidate` do not exist yet.
#[tokio::test]
async fn write_verified_entities_writes_entities_per_decision() {
    let (graph, _tmp) = open_graph("write_entities").await;
    seed_entity_types(&graph.conn).await;

    let config = PipelineConfig::builder()
        .build()
        .expect("default PipelineConfig");
    let engine = Engine::new(Arc::clone(&graph), null_llm(), null_embedder(), config);

    // Insert an episode manually so write_verified_entities has a valid episode_id FK.
    let now = chrono::Utc::now().to_rfc3339();
    let episode_id: i64 = {
        let mut rows = graph
            .conn
            .query(
                "INSERT INTO episodes (content, content_hash, recorded_at, timestamp, \
                 group_id) VALUES ('test text', 'abc123', ?1, ?1, 'default') RETURNING id",
                libsql::params![now.clone()],
            )
            .await
            .expect("episode insert");
        let row = rows.next().await.expect("row").expect("row value");
        row.get::<i64>(0).expect("id")
    };

    // Build 3 synthetic candidates:
    // - idx 0: Alice (type_id_raw = 1 = Person) → Confirm → should get entity_type_id=1
    // - idx 1: Acme Corp (type_id_raw = 2 = Organization) → Correct(new_type_id=1) → should get entity_type_id=1
    // - idx 2: Unknown Thing (type_id_raw = 1) → Demote → should get entity_type_id=0
    let candidates = vec![
        kremory::core::ingest::EntityCandidate {
            name: "Alice".to_string(),
            entity_type_id_raw: 1,
            ner_confidence: 0.95,
            span: (0, 5),
        },
        kremory::core::ingest::EntityCandidate {
            name: "Acme Corp".to_string(),
            entity_type_id_raw: 2,
            ner_confidence: 0.88,
            span: (15, 24),
        },
        kremory::core::ingest::EntityCandidate {
            name: "Unknown Thing".to_string(),
            entity_type_id_raw: 1,
            ner_confidence: 0.60,
            span: (30, 43),
        },
    ];

    let decisions = vec![
        ResolvedDecision::Confirm { candidate_idx: 0 },
        ResolvedDecision::Correct {
            candidate_idx: 1,
            new_type_id: 1,
        },
        ResolvedDecision::Demote { candidate_idx: 2 },
    ];

    // This call MUST NOT exist on current main — compile error expected.
    engine
        .write_verified_entities(episode_id, &candidates, &decisions)
        .await
        .expect("write_verified_entities must succeed");

    // Assert 3 entity rows written.
    let entity_count = count_table_rows(&graph.conn, "entities").await;
    assert_eq!(
        entity_count, 3,
        "write_verified_entities must write exactly 3 entity rows (one per decision)"
    );

    // Assert Confirm row (Alice) has entity_type_id = 1 (= candidate.entity_type_id_raw).
    let rows = graph
        .conn
        .query(
            "SELECT entity_type_id FROM entities WHERE id = 'alice'",
            libsql::params![],
        )
        .await
        .expect("alice entity_type_id query");
    // Allow for name normalization — query by normalized form if needed.
    // If Alice's id key differs, query by name match instead.
    drop(rows);
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_id FROM entities WHERE LOWER(id) LIKE '%alice%'",
            libsql::params![],
        )
        .await
        .expect("alice entity_type_id query by name");
    let alice_row = rows
        .next()
        .await
        .expect("alice row iter")
        .expect("alice row must exist after Confirm decision");
    let alice_type_id: i64 = alice_row.get(0).expect("entity_type_id col");
    assert_eq!(
        alice_type_id, 1,
        "Confirm decision: Alice must have entity_type_id=1 (candidate.entity_type_id_raw=1)"
    );

    // Assert Correct row (Acme Corp) has entity_type_id = 1 (= new_type_id=1, not raw=2).
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_id FROM entities WHERE LOWER(id) LIKE '%acme%'",
            libsql::params![],
        )
        .await
        .expect("acme entity_type_id query");
    let acme_row = rows
        .next()
        .await
        .expect("acme row iter")
        .expect("acme row must exist after Correct decision");
    let acme_type_id: i64 = acme_row.get(0).expect("entity_type_id col");
    assert_eq!(
        acme_type_id, 1,
        "Correct decision: Acme Corp must have entity_type_id=1 (new_type_id), not 2 (entity_type_id_raw)"
    );

    // Assert Demote row (Unknown Thing) has entity_type_id = 0 (catch-all).
    let mut rows = graph
        .conn
        .query(
            "SELECT entity_type_id FROM entities WHERE LOWER(id) LIKE '%unknown%'",
            libsql::params![],
        )
        .await
        .expect("unknown entity_type_id query");
    let unknown_row = rows
        .next()
        .await
        .expect("unknown row iter")
        .expect("unknown thing row must exist after Demote decision");
    let unknown_type_id: i64 = unknown_row.get(0).expect("entity_type_id col");
    assert_eq!(
        unknown_type_id, 0,
        "Demote decision: Unknown Thing must have entity_type_id=0 (catch-all)"
    );
}

// ─── Test 3: legacy Engine::ingest_with still works ──────────────────────────

/// Backwards-compat regression: the existing `Engine::ingest_with()` path still
/// executes end-to-end after the phase1/write split is introduced.
///
/// This test validates that the split does NOT break the public backwards-compat
/// entrypoint. It uses a scripted `MockChatProvider` so no live LLM calls fire.
///
/// This test exercises existing API — it does NOT fail to compile. But it provides
/// the regression guard that prevents the Green phase from accidentally removing
/// `ingest_with`.
#[tokio::test]
async fn legacy_engine_ingest_still_works() {
    let (graph, _tmp) = open_graph("legacy_ingest").await;
    seed_entity_types(&graph.conn).await;

    // Use a MockChatProvider that returns empty entity + fact arrays — sufficient
    // for the backwards-compat path which uses LlmExtractor internally.
    let mut responses = std::collections::HashMap::new();
    responses.insert(
        "Never include type information in the name field.".to_string(),
        "[]".to_string(),
    );
    responses.insert(
        "Output a JSON array of index numbers".to_string(),
        "[]".to_string(),
    );
    responses.insert("For each relationship".to_string(), "[]".to_string());
    responses.insert("different".to_string(), "[]".to_string());
    responses.insert(
        "contradicts".to_string(),
        r#"{"verdict":"no_contradiction","invalidated_ids":[]}"#.to_string(),
    );
    let llm = Arc::new(MockChatProvider::new(responses));
    let embedder = null_embedder();

    let config = PipelineConfig::builder()
        .build()
        .expect("default PipelineConfig");
    // LlmExtractor is `pub` — accessible from integration tests.
    // Clone the llm Arc before moving it into Engine::new.
    let extractor = LlmExtractor::new(Arc::clone(&llm));
    let engine = Engine::new(Arc::clone(&graph), llm, embedder, config);

    // This call uses the EXISTING ingest_with API — must not break.
    let result: IngestionResult = engine
        .ingest_with(
            &extractor,
            "Alice works at Acme Corp on the project.",
            None,
            None,
            None,
            SourceParams::default(),
        )
        .await
        .expect("ingest_with must still succeed after GAP-001 split");

    // Legacy ingest must still produce an episode ID.
    assert!(
        result.episode_id > 0,
        "legacy Engine::ingest_with must produce a positive episode_id, got {}",
        result.episode_id
    );

    // Legacy ingest must still write the episode row.
    let episode_count = count_table_rows(&graph.conn, "episodes").await;
    assert_eq!(
        episode_count, 1,
        "legacy Engine::ingest_with must write exactly 1 episode row"
    );
}
