//! Characterization test: ingest 2-phase boundary documented for spec
//! `aidocs-kremory-idempotent-cutover-2026-06-02` S00 AC #6.
//!
//! The kremory-napi `ingest_episode(draft)` is structurally:
//!   Phase 1 — `remember().from_source(sid, kind).await` → episode row atomic
//!   Phase 2 — `update_source_uri()` + `update_episode_metadata()` → separate
//!             transactions; lib.rs:127-128 documents these as non-atomic with
//!             the Phase 1 commit, with non-fatal warnings on failure.
//!
//! Spec §5.2 backfill idempotency invariant requires a reliable "doc was
//! ingested" signal. The Phase 2 split means `getBySourceId(slug).length > 0`
//! alone is INSUFFICIENT — an episode row can exist with empty metadata if
//! Phase 2 fails. Spec verdict: option α (consumer-level
//! `complete_ingest=true` marker after both phases succeed).
//!
//! This test characterises the boundary that α compensates for. If kremory
//! later makes ingest fully atomic at substrate level (β path), the
//! `metadata.is_none()` assertion in `phase_1_remember_alone_yields_episode_with_empty_metadata`
//! will fail — that is the signal to remove α and switch idempotency back to
//! existence-only.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, SourceKind};
use serde_json::json;

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_phase_boundary_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

async fn make_memory(tag: &str) -> Memory {
    Memory::open(unique_db_path(tag))
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open must succeed")
}

/// Load-bearing for spec §5.2 α-path: Phase 1 alone leaves metadata empty.
///
/// If kremory later makes `remember()` atomically commit metadata (β path),
/// the `metadata.is_none()` assertion below will fail. That failure is the
/// signal to remove α from the backfill script.
#[tokio::test]
async fn phase_1_remember_alone_yields_episode_with_empty_metadata() {
    let ns = Namespace::new("test-phase-boundary");
    let mem = make_memory("phase_1_only").await;

    let slug = "phase-boundary-doc-001";

    let commit = mem
        .remember("Phase 1 content only — no metadata supplied.")
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction() // ner feature: test is not about extraction
        .await
        .expect("Phase 1 remember must succeed");

    assert!(
        !commit.episode_entity_id.is_empty(),
        "Phase 1 commit must yield a non-empty episode_entity_id"
    );

    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");

    assert_eq!(
        episodes.len(),
        1,
        "Phase 1 alone must produce exactly one episode"
    );

    assert!(
        episodes[0].metadata.is_none(),
        "Phase 1 remember alone must NOT populate metadata — got {:?}",
        episodes[0].metadata
    );
}

/// Phase 2 update_episode_metadata persists after Phase 1 commit.
///
/// Documents that the consumer-level α mechanism is plausible: after Phase 1,
/// the consumer can independently call Phase 2 + a marker write. The substrate
/// supports this sequence.
#[tokio::test]
async fn phase_2_metadata_update_persists_after_phase_1() {
    let ns = Namespace::new("test-phase-boundary-2");
    let mem = make_memory("phase_2_seq").await;

    let slug = "phase-boundary-doc-002";

    mem.remember("Phase 1 then Phase 2 sequenced.")
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction() // ner feature: test is not about extraction
        .await
        .expect("Phase 1 must succeed");

    // TD-235 (FIXED): this NOTE used to read "`update_episode_metadata` is
    // source_id-global, NOT namespace-scoped ... if a slug ever has episodes in
    // multiple namespaces, ALL get the metadata patch — flag." That flag was
    // correct and went unactioned; both writers are now namespace-scoped on the
    // same rule as their sibling reader `recall_by_source_id`
    // (`facade/update.rs`, NOT `facade/mod.rs:2433` — the old line reference had
    // also rotted; the code moved out of `mod.rs`). Cross-namespace behaviour is
    // pinned by `it::td235_metadata_namespace_scope`.
    //
    // `.in_namespace(ns)` below is therefore load-bearing, not decoration: it
    // scopes Phase 2 to the same namespace Phase 1 wrote into. Without it this
    // `Memory` has no builder default, so the patch would take the documented
    // span-all-namespaces path.
    mem.update_episode_metadata(slug)
        .in_namespace(ns.clone())
        .patch(json!({ "docType": "spec", "complete_ingest": true }))
        .await
        .expect("Phase 2 metadata update must succeed");

    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");

    assert_eq!(episodes.len(), 1);
    let meta = episodes[0]
        .metadata
        .as_ref()
        .expect("metadata must be Some after Phase 2 patch");
    assert_eq!(meta["docType"], "spec");
    assert_eq!(meta["complete_ingest"], true);
}
