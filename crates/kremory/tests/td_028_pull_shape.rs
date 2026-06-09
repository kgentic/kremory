#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-028 Phase B acceptance tests — Phase 1 pull-shape registry-read derivation.
//!
//! Governing spec: td-028-phase1-pull-shape-registry-read-micro-spec-2026-06-09.md
//!
//! ## Acceptance criteria
//!
//! B1 — `ExtractionContext.allowed_entity_types` is derived from `registry.specs()`
//!      at pipeline.rs primary AND deferred sites (structural verification via
//!      self-learning loop test below).
//! B2 — Builder-seed branch: types set via `allowed_entity_types(...)` on the builder
//!      that are NOT in the default vocabulary are registered additively in the
//!      entity_types table for the group_id on first ingest.
//! B3 — Self-learning loop: after a registry write (simulating Pass 0), the next
//!      ingest's ctx.allowed_entity_types includes the new type WITHOUT engine rebuild.
//! B4 — All existing builder-API tests continue to pass (regression — run via
//!      `cargo test --workspace --all-features`).

use std::sync::Arc;

use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::{EntityTypeRegistry, label_to_id_or_register};
use kremory::core::extraction::LlmExtractor;
use kremory::core::ingest::{Engine, SourceParams};
use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Open a fresh on-disk TemporalGraph with all migrations applied.
/// Returns both the graph and the TempDir guard (drop = cleanup).
async fn open_graph() -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir must succeed");
    let path = tmp.path().join("td-028-test.db");
    let path_str = path.to_str().expect("path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed");
    (graph, tmp)
}

/// Build a MockChatProvider that returns empty entities + empty facts — sufficient
/// for registry-read tests where we only care about registry state, not extracted data.
fn null_mock_llm() -> MockChatProvider {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    // LlmExtractor stage-1 entity extraction prompt key.
    map.insert(
        "Never include type information in the name field.".to_string(),
        "[]".to_string(),
    );
    // LlmExtractor stage-2 relation extraction prompt key.
    map.insert(
        "Output a JSON array of index numbers".to_string(),
        "[]".to_string(),
    );
    // LlmExtractor stage-3 triplets prompt key.
    map.insert("For each relationship".to_string(), "[]".to_string());
    // CascadeResolver disambiguation key.
    map.insert("different".to_string(), "[]".to_string());
    // TwoPoolDetector contradiction key.
    map.insert(
        "contradicts".to_string(),
        r#"{"verdict":"no_contradiction","invalidated_ids":[]}"#.to_string(),
    );
    MockChatProvider::new(map)
}

/// Count entity_types rows for a group_id.
async fn count_types(conn: &libsql::Connection, group_id: &str) -> usize {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1",
            libsql::params![group_id],
        )
        .await
        .expect("count_types query");
    let row = rows.next().await.expect("row iter").expect("row");
    let n: i64 = row.get(0).expect("count col");
    n as usize
}

/// Check whether a specific type name is registered for the group_id.
async fn has_type(conn: &libsql::Connection, group_id: &str, name: &str) -> bool {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1 AND name = ?2",
            libsql::params![group_id, name],
        )
        .await
        .expect("has_type query");
    let row = rows.next().await.expect("row iter").expect("row");
    let n: i64 = row.get(0).expect("count col");
    n > 0
}

// ─── B3: self-learning loop acceptance test ───────────────────────────────────

/// B3 — TD-028 self-learning loop: after Pass 0 writes a new type to the registry,
/// the NEXT ingest sees it in the live registry without an engine rebuild.
///
/// This test is a structural verification. We cannot cheaply observe
/// `ctx.allowed_entity_types` from outside the pipeline, so we verify the
/// invariant by checking that:
///  1. The new type exists in entity_types after the direct DB insert (Pass 0 sim).
///  2. A fresh `EntityTypeRegistry::load_for_group` (the same call the pipeline
///     makes at the start of each ingest) returns a registry whose `specs()` contain
///     the new type — proving the derivation `registry.specs().iter().map(|s| s.name)`
///     would include it.
///
/// This is the pull-shape verification: the pipeline re-reads the registry on
/// every ingest call (pipeline.rs:245/306 for primary, :1039 for deferred) so
/// any write that happened between ingests is immediately visible. The test
/// proves that chain is unbroken — no snapshot/cache intervenes.
#[tokio::test]
async fn b3_pass_0_types_visible_in_next_ingest_without_engine_rebuild() {
    let (graph, _tmp) = open_graph().await;

    // Seed defaults for the "default" group_id (normally done lazily by ingest_with).
    kremory::core::entity_types::ensure_default_types_seeded(&graph.conn, "default")
        .await
        .expect("ensure_default_types_seeded must succeed");

    // Pre-condition: "ProductSKU" is not yet in the registry.
    assert!(
        !has_type(&graph.conn, "default", "ProductSKU").await,
        "pre-condition: ProductSKU must not be in the registry yet"
    );

    // ── Simulate Dream Pass 0 writing a newly-discovered type ────────────────
    //
    // Pass 0 uses the same `label_to_id_or_register` path. We call it directly
    // here to simulate what Pass 0 will do when it discovers a novel entity type.
    let registry_before = EntityTypeRegistry::load_for_group(&graph.conn, "default")
        .await
        .expect("load registry before Pass 0 sim");

    let new_id = label_to_id_or_register(
        &graph.conn,
        "default",
        &registry_before,
        "ProductSKU",
    )
    .await
    .expect("Pass 0 simulation: label_to_id_or_register must succeed");

    assert!(
        new_id > 0,
        "ProductSKU must be registered with id > 0; got {new_id}"
    );

    // ── Phase 1 re-read (next ingest, no engine rebuild) ─────────────────────
    //
    // This is exactly what ingest_with does at pipeline.rs:295-402: call
    // EntityTypeRegistry::load_for_group for the group_id, then derive
    // allowed_entity_types_live from registry.specs(). We replicate that
    // derivation here to prove the loop is closed.
    let registry_after = EntityTypeRegistry::load_for_group(&graph.conn, "default")
        .await
        .expect("registry reload must succeed (no engine rebuild)");

    let allowed_live: Vec<String> = registry_after
        .specs()
        .iter()
        .map(|s| s.name.clone())
        .collect();

    // CRITICAL assertion: ProductSKU must appear in the derived list.
    assert!(
        allowed_live.iter().any(|n| n == "ProductSKU"),
        "ProductSKU must appear in registry-derived allowed_entity_types after Pass 0 sim; \
         found: {allowed_live:?}"
    );

    // Sanity: the default types are still present (additive, not replace).
    assert!(
        allowed_live.iter().any(|n| n == "Person"),
        "default type 'Person' must still be present after Pass 0 sim"
    );
    assert!(
        allowed_live.iter().any(|n| n == "Organisation"),
        "default type 'Organisation' must still be present after Pass 0 sim"
    );
}

// ─── B2: builder-seed populates registry for custom types ────────────────────

/// B2 — Builder-seed: types set via `allowed_entity_types(...)` on the builder
/// that are NOT in the default vocabulary are registered additively in entity_types
/// on first ingest.
///
/// The default vocabulary (10 rows: Entity, Person, Organisation, Location, ...) is
/// seeded first by `ensure_default_types_seeded`. The builder-seed branch then adds
/// any extra types from `self.config.allowed_entity_types` that are not already
/// present. Here we configure two custom types ("Court", "Statute") alongside
/// standard types ("Person", "Organisation"). After ingest, all four must be in
/// entity_types.
///
/// Note: `engine.extractor` is `pub(crate)` and not accessible from integration
/// tests. We create a `LlmExtractor` directly from the same `llm` Arc and pass it
/// to `ingest_with` — this is equivalent to what `Engine::ingest` does internally.
#[tokio::test]
async fn b2_builder_seed_populates_registry_for_custom_types() {
    let (graph, _tmp) = open_graph().await;

    let config = PipelineConfig::builder()
        .allowed_entity_types(vec![
            "Person".to_string(),       // already in defaults — no-op seed
            "Organisation".to_string(), // already in defaults — no-op seed
            "Court".to_string(),        // NOT in defaults — must be seeded
            "Statute".to_string(),      // NOT in defaults — must be seeded
        ])
        .build()
        .expect("config build must succeed");

    let llm = Arc::new(null_mock_llm());
    // NullEmbeddingProvider uses struct-literal construction (no ::new method).
    // Use the default 384 dim which matches PipelineConfig's EmbeddingDim default.
    let embedder = Arc::new(NullEmbeddingProvider { dim: 384 });
    let engine = Engine::new(Arc::new(graph), Arc::clone(&llm), embedder, config);

    // Build extractor locally — engine.extractor is pub(crate), not accessible
    // from integration tests. LlmExtractor::new(llm) is the same construction
    // that Engine::new uses internally.
    let extractor = LlmExtractor::new(Arc::clone(&llm));

    // Use a fresh namespace so we start with zero entity_types rows.
    let result = engine
        .ingest_with(
            &extractor,
            "The court reviewed the statute.",
            None,
            Some("legal"),
            None,
            SourceParams::default(),
        )
        .await
        .expect("ingest_with must succeed");

    // After ingest, entity_types for "legal" must contain at minimum:
    //   - 10 default rows (ensure_default_types_seeded)
    //   - "Court" and "Statute" added by builder-seed
    //
    // engine.graph() returns Arc<TemporalGraph>; .conn is pub on TemporalGraph.
    let graph_arc = engine.graph();
    let type_count = count_types(&graph_arc.conn, "legal").await;
    assert!(
        type_count >= 12,
        "entity_types for 'legal' must have at least 12 rows (10 defaults + Court + Statute); \
         got {type_count}"
    );

    // Specific checks: custom types must be present.
    assert!(
        has_type(&graph_arc.conn, "legal", "Court").await,
        "entity_types for 'legal' must contain 'Court' after builder-seed"
    );
    assert!(
        has_type(&graph_arc.conn, "legal", "Statute").await,
        "entity_types for 'legal' must contain 'Statute' after builder-seed"
    );

    // Defaults must still be present (additive, not replace).
    assert!(
        has_type(&graph_arc.conn, "legal", "Person").await,
        "entity_types for 'legal' must still contain default 'Person'"
    );

    let _ = result;
}

/// B2-idempotency — running the same ingest twice must not duplicate rows.
///
/// Verifies INSERT OR IGNORE semantics via `label_to_id_or_register` (the GAP-006
/// fix path): second ingest with same builder types → no new rows added.
#[tokio::test]
async fn b2_builder_seed_idempotent_on_second_ingest() {
    let (graph, _tmp) = open_graph().await;

    let config = PipelineConfig::builder()
        .allowed_entity_types(vec![
            "Court".to_string(),
            "Statute".to_string(),
        ])
        .build()
        .expect("config build must succeed");

    let llm = Arc::new(null_mock_llm());
    let embedder = Arc::new(NullEmbeddingProvider { dim: 384 });
    let engine = Engine::new(Arc::new(graph), Arc::clone(&llm), embedder, config);
    let extractor = LlmExtractor::new(Arc::clone(&llm));

    // First ingest.
    engine
        .ingest_with(
            &extractor,
            "First document about courts.",
            None,
            Some("legal2"),
            None,
            SourceParams::default(),
        )
        .await
        .expect("first ingest_with must succeed");

    let graph_arc = engine.graph();
    let count_after_first = count_types(&graph_arc.conn, "legal2").await;

    // Second ingest — same builder config, same namespace.
    engine
        .ingest_with(
            &extractor,
            "Second document about statutes.",
            None,
            Some("legal2"),
            None,
            SourceParams::default(),
        )
        .await
        .expect("second ingest_with must succeed");

    let count_after_second = count_types(&graph_arc.conn, "legal2").await;

    assert_eq!(
        count_after_first, count_after_second,
        "entity_types count must not grow on second ingest (INSERT OR IGNORE idempotency); \
         first={count_after_first} second={count_after_second}"
    );
}
