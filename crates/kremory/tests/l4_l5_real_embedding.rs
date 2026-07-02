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
    disambiguate, insert_potential_alias_fact, resolve_pending_aliases, AliasProvenance,
    DisambiguateParams, DisambiguationOutcome, InsertPotentialAliasFactParams, L4_MERGE_THRESHOLD,
    L4_POTENTIAL_ALIAS_THRESHOLD,
};
use kremory::core::error::Result as KResult;
use kremory::core::graph::InsertEntityWithGroupParams;
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
// Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a documented
// too_many_arguments allow); TD-042 args-as-object targets `src/` production fns.
#[allow(clippy::too_many_arguments)]
async fn seed_entity(graph: &TemporalGraph, id: &str, group_id: &str, embedding: &[f32]) {
    let props = serde_json::json!({ "name": id, "description": id });
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0u32,
            properties: props,
            group_id: Some(group_id),
        })
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

            // ADR-057: a destructive Merge now requires high cosine AND a deterministic
            // name-compatibility check. Seed + query are SURFACE VARIANTS of one name
            // ("alice johnson" / "alice johnson jr", Jaccard 2/3 ≥ 0.5) — exactly the
            // variant case L4 exists to merge. Lexically-unrelated names with a high
            // (anisotropic) cosine are deliberately NOT merged (see the L4 unit tests).
            seed_entity(&graph, "alice johnson", group, &axis_unit(0)).await;

            // θ = 1.4° → cos ≈ 0.9997, well above the 0.95 merge threshold.
            let theta: f32 = 1.4_f32.to_radians();
            let embedder = FixedVectorEmbedder::new(rotated_unit(theta));
            disambiguate(
                DisambiguateParams {
                    entity_name: "alice johnson jr",
                    group_id: Some(group),
                    graph: &graph,
                },
                &embedder,
            )
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
                existing_id, "alice johnson",
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

/// Seed entity "alice johnson" on axis 0 ([1, 0, 0, …]).
/// Query "alice marie johnson" with [0.8, 0.6, 0, …] — unit-normalised (0.8² + 0.6² = 1.0).
/// Cosine similarity with the seed = 0.8, which sits in [0.70, 0.95) → `PotentialAlias`.
/// Post-TD-098 the alias arm also requires lexical compatibility, so the names are a
/// compatible pair (Jaccard 2/3 ≥ 0.5) — this exercises the alias-band threshold, not
/// the lexical-block path (covered by `l4_high_cosine_but_lexically_incompatible_...`).
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

            seed_entity(&graph, "alice johnson", group, &axis_unit(0)).await;

            // [0.8, 0.6, 0, …]: cos with axis_unit(0) = 0.8, inside alias band.
            let mut query_vec = vec![0.0_f32; DIM];
            query_vec[0] = 0.8;
            query_vec[1] = 0.6;

            let embedder = FixedVectorEmbedder::new(query_vec);
            disambiguate(
                DisambiguateParams {
                    entity_name: "alice marie johnson",
                    group_id: Some(group),
                    graph: &graph,
                },
                &embedder,
            )
            .await
            .expect("disambiguate must succeed")
        })
    });

    match outcome {
        DisambiguationOutcome::PotentialAlias {
            ref existing_id,
            similarity,
        } => {
            assert_eq!(existing_id, "alice johnson");
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
            disambiguate(
                DisambiguateParams {
                    entity_name: "brand new entity",
                    group_id: Some(group),
                    graph: &graph,
                },
                &embedder,
            )
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

            // ADR-057: L5 merge now requires high cosine AND lexical name compatibility.
            // A and B are SURFACE VARIANTS of one name ("alice johnson" / "alice johnson
            // jr", Jaccard 2/3 ≥ 0.5) — the legitimate L5 merge case. Unrelated names with
            // an anisotropic high cosine are deliberately not merged (L5 unit tests cover that).
            // A = exact unit on x-axis
            seed_entity(&graph, "alice johnson", group, &axis_unit(0)).await;

            // B = 0.5° rotation from A → cos ≈ 0.99996, far above the 0.8 threshold.
            let theta: f32 = 0.5_f32.to_radians();
            seed_entity(&graph, "alice johnson jr", group, &rotated_unit(theta)).await;

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

// ─── ADR-057 + TD-098 regression: lexical gate blocks anisotropic over-merge ────

/// THE bug (TD-080 #2): a weak/anisotropic embedder returns a HIGH cosine between
/// two UNRELATED entity names. Pre-fix, L4 merged them (cosine-only) and corrupted
/// every fact's subject. ADR-057 refused the destructive merge; TD-098 (Site #4 of
/// ADR-063) goes further — a high-cosine but lexically-incompatible pair is NOT even
/// recorded as a persistent `PotentialAlias` (which L7 could never revoke, since the
/// degenerate cosine stays ≈ 1.0 forever) — it is classified `New`.
///
/// Counter: `kremory.l4.merge_blocked_lexical_total` must still increment (the blocked
/// case is attributed in the `New` arm now); `merge_total` must NOT.
#[test]
fn l4_high_cosine_but_lexically_incompatible_becomes_new_not_alias() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let outcome = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            // Seed "morocco". Query "ria" with a near-identical embedding (cos ≈ 0.9997)
            // — this is exactly the verified anisotropy (cos(Ria,Morocco)=1.0000).
            seed_entity(&graph, "morocco", group, &axis_unit(0)).await;

            let theta: f32 = 1.4_f32.to_radians();
            let embedder = FixedVectorEmbedder::new(rotated_unit(theta));
            disambiguate(
                DisambiguateParams {
                    entity_name: "ria",
                    group_id: Some(group),
                    graph: &graph,
                },
                &embedder,
            )
            .await
            .expect("disambiguate must succeed")
        })
    });

    match outcome {
        DisambiguationOutcome::New => {
            // Correct: the lexical gate rejected the pair, and TD-098 discards it as
            // New rather than a persistent false alias. The `merge_blocked_lexical`
            // counter asserted below proves the cosine WAS in merge range — that counter
            // fires in the `New` arm only when similarity ≥ L4_MERGE_THRESHOLD.
        }
        other => panic!(
            "Expected New (TD-098: a high-cosine lexically-incompatible pair is anisotropy \
             noise, not a persistent alias), got {other:?}"
        ),
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
            .any(|n| n == "kremory.l4.merge_blocked_lexical_total"),
        "kremory.l4.merge_blocked_lexical_total must be emitted; got: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "kremory.l4.merge_total"),
        "a lexically-blocked pair MUST NOT increment merge_total; got: {names:?}"
    );
}

/// TD-098 alias-band sibling of the above: a pair in the [alias, merge) band
/// (cosine ≈ 0.80) with lexically-INCOMPATIBLE names must NOT be recorded as a
/// persistent `potential_alias` fact — it is `New`, and the NEW in-band blocked
/// counter `kremory.l4.alias_blocked_lexical_total` must fire (the poisoning fix).
#[test]
fn l4_alias_band_but_lexically_incompatible_becomes_new_not_alias() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let outcome = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            // Seed "morocco". Query "ria" at cosine ≈ 0.80 (alias band) — the same
            // verified anisotropy shape as the merge-band test, just at a lower score
            // so it exercises the ALIAS-arm block, not the merge-arm block.
            seed_entity(&graph, "morocco", group, &axis_unit(0)).await;

            // theta = acos(0.80) ≈ 36.87° → cos ≈ 0.80, inside [0.70, 0.95).
            let theta: f32 = 36.87_f32.to_radians();
            let embedder = FixedVectorEmbedder::new(rotated_unit(theta));
            disambiguate(
                DisambiguateParams {
                    entity_name: "ria",
                    group_id: Some(group),
                    graph: &graph,
                },
                &embedder,
            )
            .await
            .expect("disambiguate must succeed")
        })
    });

    assert!(
        matches!(outcome, DisambiguationOutcome::New),
        "an alias-band lexically-incompatible pair must be New (TD-098), got {outcome:?}"
    );

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names
            .iter()
            .any(|n| n == "kremory.l4.alias_blocked_lexical_total"),
        "kremory.l4.alias_blocked_lexical_total must be emitted; got: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "kremory.l4.potential_alias_total"),
        "a lexically-blocked alias-band pair MUST NOT increment potential_alias_total; got: {names:?}"
    );
}

/// L5 sibling of the above: a batch run with two UNRELATED names at near-identical
/// (anisotropic) embeddings must NOT merge — `merges_applied == 0` — and must emit
/// `kremory.l5.merge_blocked_lexical_total`.
#[test]
fn l5_high_cosine_but_lexically_incompatible_does_not_merge() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let report = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            seed_entity(&graph, "morocco", group, &axis_unit(0)).await;
            let theta: f32 = 0.5_f32.to_radians();
            seed_entity(&graph, "amazon robotics", group, &rotated_unit(theta)).await;

            canonicalize_surface_forms(&graph, group, L5_CANONICALIZATION_THRESHOLD)
                .await
                .expect("canonicalize must succeed")
        })
    });

    assert_eq!(
        report.merges_applied, 0,
        "unrelated names at high (anisotropic) cosine MUST NOT merge under ADR-057; \
         got merges_applied = {}",
        report.merges_applied
    );

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names
            .iter()
            .any(|n| n == "kremory.l5.merge_blocked_lexical_total"),
        "kremory.l5.merge_blocked_lexical_total must be emitted; got: {names:?}"
    );
}

/// L7 (dream alias-confirmation) is the THIRD destructive-merge path: it promotes a
/// `potential_alias` to a structural merge on recomputed cosine ≥ 0.95. ADR-057 gates
/// it with the same lexical check — a high-cosine but lexically-incompatible alias is
/// REVOKED (anisotropy noise), NOT confirmed. Without this gate, one dream cycle would
/// re-merge unrelated entities and re-introduce the TD-080 #2 corruption.
///
/// Counter: `kremory.l7.merge_blocked_lexical_total` must increment; the alias must
/// NOT be confirmed/merged.
#[test]
fn l7_high_cosine_but_lexically_incompatible_alias_is_revoked_not_merged() {
    let rt = make_rt();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let resolved = metrics::with_local_recorder(&recorder, || {
        rt.block_on(async {
            let graph = TemporalGraph::open_in_memory().await.expect("open graph");
            let group = "test-group";

            // Two UNRELATED names with near-identical (anisotropic) embeddings:
            // cos ≈ 0.99996 ≥ 0.95, but "morocco" / "amazon robotics" share no tokens.
            seed_entity(&graph, "morocco", group, &axis_unit(0)).await;
            let theta: f32 = 0.5_f32.to_radians();
            seed_entity(&graph, "amazon robotics", group, &rotated_unit(theta)).await;

            // Record a potential_alias fact morocco → amazon robotics (as L4 would).
            insert_potential_alias_fact(InsertPotentialAliasFactParams {
                graph: &graph,
                new_entity_id: "morocco",
                existing_id: "amazon robotics",
                similarity: 0.99,
                provenance: AliasProvenance {
                    source_episode_id: None,
                    group_id: Some(group),
                },
            })
            .await
            .expect("insert alias fact");

            resolve_pending_aliases(&graph, group)
                .await
                .expect("resolve must succeed")
        })
    });

    assert_eq!(
        resolved, 1,
        "the alias must be resolved (revoked), got resolved = {resolved}"
    );

    let names: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(k, _, _, _)| k.key().name().to_string())
        .collect();
    assert!(
        names
            .iter()
            .any(|n| n == "kremory.l7.merge_blocked_lexical_total"),
        "kremory.l7.merge_blocked_lexical_total must be emitted (the gate fired); got: {names:?}"
    );
}
