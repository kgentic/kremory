#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase 2 — deterministic dream passes wired into the facade orchestrator.
//!
//! Governing spec: `.ai-docs/specs/dream-phase-reconciliation-v2-2026-06-30.md`
//! §D3 (canonical 5-pass ordering: type_discovery → aliases → reclassify →
//! consistency_check → canonicalize), Phase 2 of the readiness gate
//! (`.ai-docs/plans/dream-phase-reconciliation-readiness-gate-2026-06-30.md`).
//!
//! ## What this proves (Phase 2 DoD)
//!
//! The `aliases` and `canonicalize` passes are DISPATCHED and do REAL work when
//! `mem.dream()` runs — not merely that they compile. Both pass BODIES are
//! already exhaustively unit-tested (`disambiguation`/`canonicalization`
//! modules); this proves the FACADE ORCHESTRATOR invokes them (previously they
//! were orphaned in the dead `run_dream_phase_passes`, per TD-089).
//!
//! Deterministic (no real LLM): identical unit-vector embeddings force cosine
//! ≈ 1.0, above both `L4_MERGE_THRESHOLD` (0.95, alias confirm) and
//! `L5_CANONICALIZATION_THRESHOLD` (0.8, near-dup merge).

use std::sync::Arc;

use kremory::core::disambiguation::{
    insert_potential_alias_fact, AliasProvenance, InsertPotentialAliasFactParams,
};
use kremory::core::graph::InsertEntityWithGroupParams;
use kremory::core::provider::{MockChatProvider, MockEmbeddingProvider};
use kremory::core::schema::TemporalGraph;
use kremory::memory::ChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

const DIM: usize = 384;

/// Unit-normalised embedding with all components equal → any two are cosine ≈ 1.0.
fn unit_vec(dim: usize) -> Vec<f32> {
    let v = 1.0_f32 / (dim as f32).sqrt();
    vec![v; dim]
}

/// Plant a `entity_type_id = 0` entity with a fixed embedding (mirrors the
/// `canonicalization` module's own test helper, via public graph APIs).
// Test helper: clippy.toml Rule-5 exempt (test helpers may carry a documented
// too_many_arguments allow, per feedback_no_clippy_allow_in_src_args_as_object;
// TD-042 args-as-object targets `src/` production fns only). Same convention as
// `tests/phase_e_reclassify.rs::insert_entity`.
#[allow(clippy::too_many_arguments)]
async fn plant_entity(graph: &TemporalGraph, id: &str, group_id: &str, description: &str) {
    let props = serde_json::json!({ "name": id, "description": description });
    graph
        .insert_entity_with_group(InsertEntityWithGroupParams {
            id,
            entity_type_id: 0u32,
            properties: props,
            group_id: Some(group_id),
        })
        .await
        .expect("insert entity");
    // Embedding planted directly on the row — MockEmbeddingProvider::embed is NOT
    // invoked for these entities; the vector is fixed so cosine ≈ 1.0 is exact.
    graph
        .set_entity_embedding(id, &unit_vec(DIM))
        .await
        .expect("set embedding");
}

async fn open_mem(dir: &std::path::Path, ns: &Namespace) -> Memory {
    let llm: Arc<dyn ChatProvider> = Arc::new(MockChatProvider::null());
    let emb: Arc<dyn DynEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(DIM));
    Memory::open(dir.join("phase2.db"))
        .with_llm(llm)
        .with_embedder(emb)
        .embedding_dim(DIM)
        .default_namespace(ns.clone())
        .await
        .expect("Memory::open must succeed")
}

/// Sum a counter's recorded value from a local metrics snapshot.
fn counter_total(
    snapshot: &[(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
) -> u64 {
    snapshot
        .iter()
        .filter(|(k, _, _, _)| k.key().name() == name)
        .filter_map(|(_, _, _, v)| match v {
            DebugValue::Counter(c) => Some(*c),
            _ => None,
        })
        .sum()
}

// ── aliases pass (ordered BEFORE reclassify per §D3) ────────────────────────

#[tokio::test]
async fn dream_resolves_planted_potential_alias() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("phase2-alias");
    let mem = open_mem(dir.path(), &ns).await;
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    // Two surface-variant entities with identical embeddings (cosine 1.0 ≥ 0.95).
    plant_entity(
        &graph,
        "acme corporation",
        &gid,
        "Acme Corporation, a company.",
    )
    .await;
    plant_entity(&graph, "acme corp", &gid, "Acme Corp.").await;

    // Plant a pending `potential_alias` fact: new "acme corp" → existing "acme corporation".
    insert_potential_alias_fact(InsertPotentialAliasFactParams {
        graph: &graph,
        new_entity_id: "acme corp",
        existing_id: "acme corporation",
        similarity: 0.99,
        provenance: AliasProvenance {
            source_episode_id: None,
            group_id: Some(&gid),
        },
    })
    .await
    .expect("plant potential_alias fact");

    // Run the full dream pass chain.
    let summary = mem.dream().await.expect("dream must succeed");

    // Phase 4: the DreamSummary surfaces the resolved count (accumulator → fold).
    assert!(
        summary.aliases_resolved >= 1,
        "DreamSummary.aliases_resolved must be populated; got {}",
        summary.aliases_resolved,
    );

    // The aliases pass must have run AND resolved the confirmed alias (≥ 1).
    let snapshot = snapshotter.snapshot().into_vec();
    let resolved = counter_total(&snapshot, "kremory.dream.aliases_resolved_total");
    assert!(
        resolved >= 1,
        "aliases pass must resolve the planted potential_alias via mem.dream() \
         (§D3); resolved = {resolved}",
    );

    // Graph-effect proof: the aliases pass INVALIDATES the pending potential_alias
    // fact. Per `resolve_pending_aliases`, both outcomes at cosine ≈ 1.0 invalidate
    // the fact — MERGE (names compatible → L5 does the structural merge later) or
    // REVOKE (names fail the lexical gate → false-alarm). The entity itself is NOT
    // removed here (that is L5 canonicalize's job, gated on name similarity which
    // "acme corp"/"acme corporation" do not pass). The correct post-condition is
    // therefore "no pending alias fact remains" — which proves the counter is not
    // a silent no-op.
    let pending = graph
        .get_alias_facts_in_group(&gid)
        .await
        .expect("get_alias_facts_in_group");
    assert!(
        pending.is_empty(),
        "aliases pass must invalidate the pending potential_alias fact via \
         mem.dream(); {} still pending",
        pending.len(),
    );
}

// ── canonicalize pass (ordered LAST per §D3) ────────────────────────────────

#[tokio::test]
async fn dream_merges_near_duplicate_entities() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let dir = tempfile::tempdir().expect("tempdir");
    let ns = Namespace::new("phase2-canon");
    let mem = open_mem(dir.path(), &ns).await;
    let gid = mem.group_id_for_test(&ns);
    let graph = mem
        .temporal_graph_for_test()
        .expect("temporal_graph_for_test (test-utils)")
        .clone();

    // Surface variants of one entity (Jaccard-gated pair from canonicalization T4),
    // identical embeddings → cosine 1.0 ≥ L5 threshold. No alias fact planted.
    plant_entity(
        &graph,
        "alice johnson",
        &gid,
        "A detailed description of Alice Johnson, software engineer at Acme Corp.",
    )
    .await;
    plant_entity(&graph, "alice j", &gid, "Alice.").await;

    let summary = mem.dream().await.expect("dream must succeed");

    // Phase 4: the DreamSummary surfaces the merge count (accumulator → fold).
    assert!(
        summary.canonicalization_merges >= 1,
        "DreamSummary.canonicalization_merges must be populated; got {}",
        summary.canonicalization_merges,
    );

    // Counter proof: canonicalize ran and applied ≥ 1 merge.
    let snapshot = snapshotter.snapshot().into_vec();
    let merges = counter_total(&snapshot, "kremory.dream.canonicalization_merges_total");
    assert!(
        merges >= 1,
        "canonicalize pass must merge the near-dup pair via mem.dream() (§D3); \
         merges = {merges}",
    );

    // Graph-effect proof: keeper (longer description) survives, loser is gone.
    let ids: Vec<String> = graph
        .list_entities_in_group(&gid)
        .await
        .expect("list entities")
        .iter()
        .map(|e| e.id.clone())
        .collect();
    assert!(
        ids.iter().any(|i| i == "alice johnson") && !ids.iter().any(|i| i == "alice j"),
        "canonicalize must merge 'alice j' into keeper 'alice johnson'; present: {ids:?}",
    );
}
