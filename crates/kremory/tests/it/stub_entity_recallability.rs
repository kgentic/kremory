//! Forward-reference STUB entities must be findable by BOTH retrieval arms.
//!
//! # Why this exists
//!
//! Found 2026-08-05 while running the LoCoMo bench on local Ollama. The server
//! log was full of TD-114 shortfall warnings (`search.rs:353-383`):
//!
//! ```text
//! namespace_recall_shortfall  requested=10 delivered=7 namespace_rows=12 fetch_k=10
//! ```
//!
//! `fetch_k == requested` means the TD-114 over-fetch planner short-circuited, which
//! it only does when `total == ns` — i.e. the group filter was dropping NOTHING. So
//! the missing rows were not a namespace-filter artifact. A direct count found the
//! cause:
//!
//! ```text
//! total entities 67 | with embedding 42 | NULL embedding 25   (37%)
//! ```
//!
//! `ingest_with.rs`'s forward-reference stub path created entities with **neither**
//! of the two things that make an entity retrievable:
//!
//! | arm | requires | stub had it? |
//! |---|---|---|
//! | dense / ANN | a row in `entities.embedding` | **NO** |
//! | FTS seed | `properties["name"]` (`entities_fts` indexes `properties` ONLY — `label` is empty post-Migration-009) | **NO** |
//!
//! TD-113 had already established the FTS half for the `with_facts` PINNED path
//! (`ingest_with.rs:458-466`: *"A bare `{"stub": false}` stub carried no name token
//! → recall returned 0"*), and `make_pinned_entity_recallable` covers its embedding.
//! The EXTRACTION forward-reference path never received either fix.
//!
//! # What makes this test non-vacuous
//!
//! The extractor below returns ONE entity and a fact whose OBJECT is a different,
//! undeclared name — the only deterministic way to force the forward-reference
//! branch. The test asserts the stub EXISTS before asserting anything about it: a
//! run where no stub was created must FAIL, not silently pass with zero rows
//! examined.
//!
//! RED-proof: revert either half of the fix and the corresponding assertion fails —
//! drop `"name"` from `stub_props` and the FTS assertion reports the raw props;
//! drop the post-commit embed loop and `embedding` is NULL.

// Test-module convention, matching 156 of the 160 modules in this target: a test
// SHOULD panic on a failed precondition, so `expect`/`panic` are the correct
// failure mode here. The crate-level deny in Cargo.toml:277-279 targets `src/`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use kremory::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use kremory::core::provider::MockChatProvider;
use kremory::{DynEmbeddingProvider, Memory, Namespace};

/// Declares ONE entity, then references a SECOND by name in a fact. `Analytical
/// Engine` is never declared, so the fact's object is a forward reference and
/// `ingest_with.rs`'s pre-scan must insert it as a stub.
struct ForwardReferenceExtractor;

impl EntityExtractor for ForwardReferenceExtractor {
    fn name(&self) -> &'static str {
        "stub-recallability-forward-reference-extractor"
    }

    async fn extract<'a>(
        &'a self,
        _text: &'a str,
        _ctx: &'a ExtractionContext<'a>,
    ) -> kremory::CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Ada Lovelace".to_string(),
                label: "Person".to_string(),
                properties: serde_json::json!({"name": "Ada Lovelace"}),
            }],
            // `Analytical Engine` is deliberately ABSENT from `entities` above.
            facts: vec![ExtractedFact {
                subject: "Ada Lovelace".to_string(),
                predicate: "wrote algorithms for".to_string(),
                object: "Analytical Engine".to_string(),
                is_entity_ref: true,
                confidence: 1.0,
                valid_at: None,
            }],
        })
    }
}

fn unique_db(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_stub_recall_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forward_reference_stub_is_findable_by_both_fts_and_dense() {
    let embedder: Arc<dyn DynEmbeddingProvider> =
        Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 });
    let mem = Memory::open(unique_db("both_arms"))
        .with_llm(Arc::new(MockChatProvider::null()))
        .with_embedder(embedder)
        .with_extractor(Arc::new(ForwardReferenceExtractor))
        .default_namespace(Namespace::new("stubrecall"))
        .await
        .expect("Memory::open must succeed");

    let commit = mem
        .remember("Ada Lovelace wrote algorithms for the Analytical Engine.")
        .from_chat("stub-recall-session")
        .await
        .expect("remember must succeed");
    // ── PRECONDITION 1 (anti-vacuity), straight off the public surface ──────
    // `EpisodeCommit.stub_entities_inserted` is the producer's own count. If the
    // fixture stops producing a forward reference, this fails HERE with a clear
    // cause rather than further down as a confusing NULL-embedding failure.
    assert_eq!(
        commit.stub_entities_inserted, 1,
        "PRECONDITION FAILED: expected exactly ONE forward-reference stub. The \
         extractor declares only 'Ada Lovelace' while emitting a fact whose object is \
         'Analytical Engine'. Got {}. Fix the fixture — do NOT weaken the assertions \
         below, which would then be proving nothing.",
        commit.stub_entities_inserted
    );

    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");
    let conn = tg.conn_for_test();

    // ── PRECONDITION (anti-vacuity) ──────────────────────────────────────────
    // Without this, a run that created NO stub would sail through every
    // assertion below having examined zero rows.
    let mut rows = conn
        .query(
            "SELECT properties, embedding IS NOT NULL FROM entities WHERE id = ?1",
            libsql::params!["analytical engine"],
        )
        .await
        .expect("stub query must execute");
    let row = rows
        .next()
        .await
        .expect("stub query must return")
        .expect(
            "PRECONDITION FAILED: the commit reported a stub was inserted, but no row \
             exists at id 'analytical engine' — the id normalisation the test assumes \
             has changed.",
        );

    let props: String = row.get(0).expect("properties column");
    let has_embedding: i64 = row.get(1).expect("embedding-not-null column");

    // ── TD-113: FTS findability ──────────────────────────────────────────────
    // `entities_fts` indexes `properties` ONLY (label is empty post-Migration-009),
    // so a stub with no name token cannot be reached by the FTS seed arm at all.
    let parsed: serde_json::Value =
        serde_json::from_str(&props).expect("stub properties must be valid JSON");
    assert_eq!(
        parsed.get("name").and_then(|v| v.as_str()),
        Some("analytical engine"),
        "TD-113: a forward-reference stub MUST carry properties[\"name\"] or it is \
         invisible to the FTS seed arm (entities_fts indexes properties only). \
         Actual properties: {props}"
    );

    // ── TD-114: dense findability ────────────────────────────────────────────
    // A NULL embedding keeps the row out of the ANN index while `plan_index_fetch`
    // still counts it in `namespace_rows` — the exact shortfall reported at
    // search.rs:353-383 (requested=10 delivered=7 namespace_rows=12).
    assert_eq!(
        has_embedding, 1,
        "TD-114: a forward-reference stub MUST carry an embedding or it is invisible to \
         the dense/ANN arm while still inflating `namespace_rows`, which is what produced \
         the filtered-ANN under-fill on the LoCoMo bench (37% of entities affected)."
    );
}
