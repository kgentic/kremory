#![allow(clippy::unwrap_used, clippy::expect_used)]
//! L4/L5 real-embedding integration tests.
//!
//! Sprint plan reference: v0.2.0 Phase B-prep — T1.5 (prior-art audit finding #10).
//! Prior-art audit finding: L4/L5 real-embedding paths were untested because
//! `NullEmbeddingProvider` returns zero-cosine vectors, which cause all
//! `vector_search_with_index` hits to be skipped (NULL cosine distance in libsql).
//! These tests seed synthetic non-zero embedding vectors via `set_entity_embedding`
//! to exercise the actual threshold-ladder paths.
//!
//! ## Tests
//!
//! - `l4_disambiguate_merge_on_high_similarity_real_embedding`
//! - `l4_disambiguate_potential_alias_on_moderate_similarity_real_embedding`
//! - `l4_disambiguate_new_on_low_similarity_real_embedding`
//! - `l5_canonicalize_merges_when_two_entities_have_near_identical_embeddings`
//!
//! ## Runtime pattern
//!
//! Each test uses a synchronous `#[test]` with an explicit single-threaded Tokio
//! runtime (matching `b1_observability.rs`). This is required because
//! `metrics::with_local_recorder` is synchronous and cannot hold a thread-local
//! recorder across `.await` points — the `block_on` call keeps the thread pinned
//! so the thread-local remains valid for the full async call chain.

use kremory::core::canonicalization::{canonicalize_surface_forms, L5_CANONICALIZATION_THRESHOLD};
use kremory::core::disambiguation::{
    disambiguate, DisambiguationOutcome, L4_MERGE_THRESHOLD, L4_POTENTIAL_ALIAS_THRESHOLD,
};
use kremory::core::error::Result as KResult;
use kremory::core::provider::EmbeddingProvider;
use kremory::core::schema::TemporalGraph;
use metrics_util::debugging::DebuggingRecorder;

// ─── Embedding dimension ──────────────────────────────────────────────────────

/// Match `TemporalGraph::open_in_memory()` default.
const DIM: usize = 384;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Build a unit-norm vector where component at `axis` is 1 and all others are 0.
/// Cosine similarity of two such vectors: 1.0 if same axis, 0.0 if different axis.
fn axis_unit(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[axis] = 1.0;
    v
}

/// Build a vector [cos(θ), sin(θ), 0, 0, …] — unit-normalised by construction.
/// Cosine similarity with `axis_unit(0)` = cos(θ).
fn rotated_unit(theta_rad: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[0] = theta_rad.cos();
    v[1] = theta_rad.sin();
    v
}

/// Build a single-threaded Tokio runtime for use in sync `#[test]` bodies.
fn make_rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds")
}

/// Insert an entity into `group_id` with the given embedding vector.
async fn seed_entity(graph: &TemporalGraph, id: &str, group_id: &str, embedding: &[f32]) {
    let props = serde_json::json!({ "name": id, "description": id });
    graph
        .insert_entity_with_group(id, 0u32, props, Some(group_id))
        .await
        .expect("insert entity");
    graph
        .set_entity_embedding(id, embedding)
        .await
        .expect("set entity embedding");
}

// ─── Fixed-vector embedder ────────────────────────────────────────────────────

/// Embedder that always returns a stored vector regardless of input text.
/// Used to inject a known query embedding into `disambiguate`.
struct FixedVectorEmbedder {
    vector: Vec<f32>,
}

impl FixedVectorEmbedder {
    fn new(vector: Vec<f32>) -> Self {
        Self { vector }
    }
}

impl EmbeddingProvider for FixedVectorEmbedder {
    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> impl std::future::Future<Output = KResult<Vec<f32>>> + Send + 'a {
        let v = self.vector.clone();
        async move { Ok(v) }
    }
}

// ─── Test 1: Merge on high similarity ────────────────────────────────────────

/// Seed entity A on axis 0 ([1, 0, 0, …]).
/// Query with a vector rotated θ = 1.4° from A.
/// cos(1.4°) ≈ 0.9997 which is well above L4_MERGE_THRESHOLD (0.95).
/// Expected: `DisambiguationOutcome::Merge { existing_id = "entity-a", similarity ≥ 0.95 }`.
///
/// Counter: `kremory.l4.merge_total` must increment.
#[test]
fn l4_disambiguate_merge_on_high_similarity_real_embedding() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let outcome = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            seed_entity(&graph, "entity-a", group, &axis_unit(0)).await;

            // θ = 1.4° → cos ≈ 0.9997, well above the 0.95 merge threshold.
            let theta: f32 = 1.4_f32.to_radians();
            let embedder = FixedVectorEmbedder::new(rotated_unit(theta));
            disambiguate("similar entity", Some(group), &graph, &embedder)
                .await
                .expect("disambiguate must succeed")
        })
    });

    match outcome {
        DisambiguationOutcome::Merge {
            ref existing_id,
            similarity,
        } => {
            assert_eq!(
                existing_id, "entity-a",
                "Merge must reference the seeded entity"
            );
            assert!(
                similarity >= L4_MERGE_THRESHOLD,
                "similarity {similarity:.4} must be ≥ L4_MERGE_THRESHOLD ({L4_MERGE_THRESHOLD})"
            );
        }
        other => panic!("Expected Merge, got {other:?}"),
    }

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names.iter().any(|n| n == "kremory.l4.merge_total"),
        "kremory.l4.merge_total counter must be emitted; got: {names:?}"
    );
}

// ─── Test 2: PotentialAlias on moderate similarity ────────────────────────────

/// Seed entity A on axis 0 ([1, 0, 0, …]).
/// Query with [0.8, 0.6, 0, …] — unit-normalised (0.8² + 0.6² = 1.0).
/// Cosine similarity with A = 0.8, which sits in [0.70, 0.95) → `PotentialAlias`.
///
/// Counter: `kremory.l4.potential_alias_total` must increment.
#[test]
fn l4_disambiguate_potential_alias_on_moderate_similarity_real_embedding() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let outcome = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            seed_entity(&graph, "entity-a", group, &axis_unit(0)).await;

            // [0.8, 0.6, 0, …]: cos with axis_unit(0) = 0.8, inside alias band.
            let mut query_vec = vec![0.0_f32; DIM];
            query_vec[0] = 0.8;
            query_vec[1] = 0.6;

            let embedder = FixedVectorEmbedder::new(query_vec);
            disambiguate("alias entity", Some(group), &graph, &embedder)
                .await
                .expect("disambiguate must succeed")
        })
    });

    match outcome {
        DisambiguationOutcome::PotentialAlias {
            ref existing_id,
            similarity,
        } => {
            assert_eq!(existing_id, "entity-a");
            assert!(
                similarity >= L4_POTENTIAL_ALIAS_THRESHOLD,
                "similarity {similarity:.4} must be ≥ L4_POTENTIAL_ALIAS_THRESHOLD \
                 ({L4_POTENTIAL_ALIAS_THRESHOLD})"
            );
            assert!(
                similarity < L4_MERGE_THRESHOLD,
                "similarity {similarity:.4} must be < L4_MERGE_THRESHOLD ({L4_MERGE_THRESHOLD})"
            );
        }
        other => panic!("Expected PotentialAlias, got {other:?}"),
    }

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names
            .iter()
            .any(|n| n == "kremory.l4.potential_alias_total"),
        "kremory.l4.potential_alias_total must be emitted; got: {names:?}"
    );
}

// ─── Test 3: New on low similarity ────────────────────────────────────────────

/// Seed entity A on axis 0 ([1, 0, 0, …]).
/// Query with axis 1 ([0, 1, 0, …]) — cosine similarity = 0.0 < 0.70.
/// Expected: `DisambiguationOutcome::New`.
///
/// Counter: `kremory.l4.new_entity_total` must increment.
#[test]
fn l4_disambiguate_new_on_low_similarity_real_embedding() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let outcome = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            seed_entity(&graph, "entity-a", group, &axis_unit(0)).await;

            // axis 1 is orthogonal to axis 0 → cosine similarity = 0.0
            let embedder = FixedVectorEmbedder::new(axis_unit(1));
            disambiguate("brand new entity", Some(group), &graph, &embedder)
                .await
                .expect("disambiguate must succeed")
        })
    });

    assert_eq!(
        outcome,
        DisambiguationOutcome::New,
        "Orthogonal embedding must produce New outcome"
    );

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names.iter().any(|n| n == "kremory.l4.new_entity_total"),
        "kremory.l4.new_entity_total must be emitted; got: {names:?}"
    );
}

// ─── Test 4: L5 merges near-identical embeddings ──────────────────────────────

/// Seed entities A and B with nearly identical embeddings:
///   A = axis_unit(0)            = [1, 0, 0, …]
///   B = rotated_unit(θ = 0.5°)  ≈ [0.99996, 0.00873, 0, …]
/// cosine(A, B) = cos(0.5°) ≈ 0.99996 which is well above L5_CANONICALIZATION_THRESHOLD (0.8).
///
/// Asserts:
///   - `report.merges_applied > 0`
///   - `kremory.l5.merges_applied_total` counter incremented.
#[test]
fn l5_canonicalize_merges_when_two_entities_have_near_identical_embeddings() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let report = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            // A = exact unit on x-axis
            seed_entity(&graph, "entity-a", group, &axis_unit(0)).await;

            // B = 0.5° rotation from A → cos ≈ 0.99996, far above the 0.8 threshold.
            let theta: f32 = 0.5_f32.to_radians();
            seed_entity(&graph, "entity-b", group, &rotated_unit(theta)).await;

            canonicalize_surface_forms(&graph, group, L5_CANONICALIZATION_THRESHOLD)
                .await
                .expect("canonicalize must succeed")
        })
    });

    assert!(
        report.merges_applied > 0,
        "Expected at least one merge, got merges_applied = {}",
        report.merges_applied
    );
    assert_eq!(report.group_id, "test-group");
    assert!(
        report.pairs_examined >= 1,
        "At least one pair must be examined; got pairs_examined = {}",
        report.pairs_examined
    );

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names.iter().any(|n| n == "kremory.l5.merges_applied_total"),
        "kremory.l5.merges_applied_total must be emitted; got: {names:?}"
    );
}
