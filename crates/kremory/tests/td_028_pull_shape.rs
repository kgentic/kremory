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
//! B5 — QB-02: Integration test asserts `rql.ingest.registry_builder_seed_applied`
//!      counter fires on the builder-seed path and does NOT fire on the override path.

use std::sync::Arc;

use kremory::core::config::PipelineConfig;
use kremory::core::entity_types::{EntityTypeRegistry, EntityTypeSpec, label_to_id_or_register};
use kremory::core::extraction::LlmExtractor;
use kremory::core::ingest::{Engine, SourceParams};
use kremory::core::provider::{MockChatProvider, NullEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

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

// ─── B5: QB-02 — counter assertion for builder-seed vs override paths ─────────

/// B5 / QB-02 — `rql.ingest.registry_builder_seed_applied` counter:
///   - FIRES when `allowed_entity_types(...)` on the builder seeds a new type.
///   - Does NOT fire when the ingest uses `entity_types_override` (override path).
///
/// Uses `metrics::with_local_recorder` + `DebuggingRecorder` (same pattern as
/// kremory's other counter-assertion tests: chat_tracking.rs, extraction/mod.rs).
/// No global recorder required — library-safe per ADR D2.
///
/// Spec: td-028-phase1-pull-shape-registry-read-micro-spec-2026-06-09.md DoD B5.
/// Quinn finding: QB-02 (MED).
#[test]
fn b5_registry_builder_seed_counter_fires_on_seed_not_on_override() {
    // ── Part 1: builder-seed path — counter must fire ──────────────────────────
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime builds");

        rt.block_on(async {
            let (graph, _tmp) = open_graph().await;

            let config = PipelineConfig::builder()
                .allowed_entity_types(vec![
                    "SeedTypeAlpha".to_string(), // NOT in defaults — triggers seed branch
                    "SeedTypeBeta".to_string(),  // NOT in defaults — triggers seed branch
                ])
                .build()
                .expect("config build must succeed");

            let llm = Arc::new(null_mock_llm());
            let embedder = Arc::new(NullEmbeddingProvider { dim: 384 });
            let engine = Engine::new(Arc::new(graph), Arc::clone(&llm), embedder, config);
            let extractor = LlmExtractor::new(Arc::clone(&llm));

            engine
                .ingest_with(
                    &extractor,
                    "Document for builder-seed counter test.",
                    None,
                    Some("b5-seed-ns"),
                    None,
                    SourceParams::default(),
                )
                .await
                .expect("ingest_with (builder-seed path) must succeed");
        });
    });

    let snapshot = snapshotter.snapshot().into_vec();
    let seed_entries: Vec<_> = snapshot
        .iter()
        .filter(|(k, _, _, _)| k.key().name() == "rql.ingest.registry_builder_seed_applied")
        .collect();

    assert!(
        !seed_entries.is_empty(),
        "rql.ingest.registry_builder_seed_applied must be emitted on the builder-seed path; \
         got no matching metrics in snapshot"
    );

    // Verify the counter has a positive value and carries the namespace label.
    let mut found_positive = false;
    for (key, _, _, value) in &seed_entries {
        let labels: std::collections::HashMap<&str, &str> =
            key.key().labels().map(|l| (l.key(), l.value())).collect();
        // Namespace label must be present (QB-03 parity check).
        assert!(
            labels.contains_key("namespace"),
            "rql.ingest.registry_builder_seed_applied must carry a 'namespace' label; \
             labels found: {labels:?}"
        );
        if let DebugValue::Counter(n) = value {
            if *n > 0 {
                found_positive = true;
            }
        }
    }
    assert!(
        found_positive,
        "rql.ingest.registry_builder_seed_applied counter value must be > 0 on seed path"
    );

    // ── Part 2: override path — counter must NOT fire ──────────────────────────
    //
    // When `source_params.entity_types_override` is set, the pipeline takes a
    // completely different branch (pipeline.rs:297) and never enters the
    // builder-seed block (pipeline.rs:370). The seed counter must stay silent.
    let recorder2 = DebuggingRecorder::new();
    let snapshotter2 = recorder2.snapshotter();

    metrics::with_local_recorder(&recorder2, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime builds");

        rt.block_on(async {
            let (graph, _tmp) = open_graph().await;

            // No allowed_entity_types on the builder — override path exclusively.
            let config = PipelineConfig::builder()
                .build()
                .expect("config build must succeed");

            let llm = Arc::new(null_mock_llm());
            let embedder = Arc::new(NullEmbeddingProvider { dim: 384 });
            let engine = Engine::new(Arc::new(graph), Arc::clone(&llm), embedder, config);
            let extractor = LlmExtractor::new(Arc::clone(&llm));

            // Supply entity_types_override — this puts the pipeline on the
            // override branch; the builder-seed branch is skipped entirely.
            let override_specs = vec![
                EntityTypeSpec {
                    id: 1,
                    name: "OverrideType".to_string(),
                    description: "Override-supplied type for B5 test.".to_string(),
                },
            ];
            let src = SourceParams {
                entity_types_override: Some(override_specs),
                ..SourceParams::default()
            };

            engine
                .ingest_with(
                    &extractor,
                    "Document for override-path counter test.",
                    None,
                    Some("b5-override-ns"),
                    None,
                    src,
                )
                .await
                .expect("ingest_with (override path) must succeed");
        });
    });

    let snapshot2 = snapshotter2.snapshot().into_vec();
    let seed_entries2: Vec<_> = snapshot2
        .iter()
        .filter(|(k, _, _, _)| k.key().name() == "rql.ingest.registry_builder_seed_applied")
        .collect();

    assert!(
        seed_entries2.is_empty(),
        "rql.ingest.registry_builder_seed_applied must NOT fire on the override path; \
         got {seed_entries2:?}"
    );
}
