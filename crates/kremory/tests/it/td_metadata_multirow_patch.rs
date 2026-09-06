//! `Memory::update_episode_metadata` must merge PER ROW, not merge once and
//! broadcast.
//!
//! Nothing enforces uniqueness on `episodes.source_id` — Migration 007 creates
//! a plain `CREATE INDEX IF NOT EXISTS idx_episodes_source_id`, not a UNIQUE
//! one (`core/migrations/defs_b.rs:123`). And `docs/api.md` §5.1 teaches
//! reusing ONE source id for a whole chat session, while the Node binding calls
//! this method after every `remember({metadata, source_id})`
//! (`kremory-napi/src/lib.rs:277-291`). So N > 1 rows per `source_id` is the
//! documented normal case, not an edge case.
//!
//! The defect these tests pin: the merge read `SELECT metadata FROM episodes
//! WHERE source_id = ?1` and took `.next()` — the FIRST row, no `ORDER BY`, no
//! `LIMIT` — then wrote the result to EVERY matching row. Episodes 2..N had
//! their own distinct metadata silently replaced by a value derived from an
//! arbitrary sibling. That is data loss, not a merge.
//!
//! `two_rows_with_distinct_metadata_each_keep_their_own_keys` was observed RED
//! against the unfixed code before the fix existed — a test nobody has seen
//! fail is an unvalidated instrument, not evidence.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use kremory::core::graph::EpisodeInsert;
use kremory::{DynEmbeddingProvider, Memory, Namespace};
use serde_json::json;

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// One in-memory `Memory` plus the single namespace + `source_id` a test
/// operates on. Bundling them keeps every helper at or under the project's
/// `too-many-arguments` threshold without an `#[allow]`, and mirrors the real
/// shape: one `source_id` IS one conversation, however many episodes it holds.
struct Conversation {
    mem: Memory,
    ns: Namespace,
    source_id: String,
}

impl Conversation {
    async fn open(ns: &str, source_id: &str) -> Self {
        let mem = Memory::open(":memory:")
            .with_llm(null_llm())
            .with_embedder(null_embedder())
            .await
            .expect("Memory::open(\":memory:\") must succeed");
        Self {
            mem,
            ns: Namespace::new(ns),
            source_id: source_id.to_string(),
        }
    }

    /// Plant an episode row directly on the graph, so each row can start with
    /// its OWN distinct metadata. Going through `remember()` cannot set up this
    /// fixture: `remember()` alone leaves `metadata` NULL (pinned by
    /// `ingest_phase_boundary::phase_1_remember_alone_yields_episode_with_empty_metadata`),
    /// and the only other writer is the very method under test.
    async fn seed(&self, content: &str, metadata: Option<serde_json::Value>) {
        let graph = self
            .mem
            .temporal_graph_for_test()
            .expect("Memory built via the builder path must carry a TemporalGraph");
        let group_id = self.mem.group_id_for_test(&self.ns);

        let mut ep = EpisodeInsert::new(content, chrono::Utc::now())
            .source_type("test")
            .source_id(&self.source_id);
        if let Some(m) = metadata {
            ep = ep.metadata(m);
        }

        graph
            .insert_episode_with_group(ep, Some(&group_id))
            .await
            .expect("insert_episode_with_group must succeed");
    }

    /// Apply the patch under test, returning the reported row count.
    async fn patch(&self, patch: serde_json::Value) -> usize {
        self.mem
            .update_episode_metadata(&self.source_id)
            .patch(patch)
            .await
            .expect("update_episode_metadata must succeed")
    }

    /// Read every episode back as `content -> metadata`, so assertions never
    /// depend on row order (`recall_by_source_id` orders by `recorded_at DESC`,
    /// which ties for rows inserted within the same second).
    async fn metadata_by_content(&self) -> BTreeMap<String, Option<serde_json::Value>> {
        self.mem
            .recall_by_source_id(&self.source_id, Some(self.ns.clone()))
            .await
            .expect("recall_by_source_id must not error")
            .into_iter()
            .map(|e| (e.content, e.metadata))
            .collect()
    }
}

/// THE defect test. Two episodes share a `source_id` and start with DIFFERENT
/// metadata; a patch adding a new key must leave each row's own pre-existing
/// keys intact.
///
/// Against the unfixed code this failed on the second row: it was overwritten
/// with the first row's metadata plus the patch, so `turn-2` lost `"turn": 2` /
/// `"only_on_b"` and gained `turn-1`'s keys.
#[tokio::test]
async fn two_rows_with_distinct_metadata_each_keep_their_own_keys() {
    let conv = Conversation::open("td-metadata-multirow", "conv-multirow-001").await;

    conv.seed("turn-1", Some(json!({ "turn": 1, "only_on_a": "alpha" })))
        .await;
    conv.seed("turn-2", Some(json!({ "turn": 2, "only_on_b": "beta" })))
        .await;

    let updated = conv.patch(json!({ "shared": "patched" })).await;
    assert_eq!(
        updated, 2,
        "the patch must still touch BOTH rows — same rows as before, only the \
         base value changes"
    );

    let got = conv.metadata_by_content().await;
    assert_eq!(got.len(), 2, "expected 2 episodes, got {got:?}");

    let a = got["turn-1"].as_ref().expect("turn-1 metadata");
    assert_eq!(a["turn"], json!(1), "turn-1 lost its own `turn` key: {a}");
    assert_eq!(
        a["only_on_a"],
        json!("alpha"),
        "turn-1 lost its own `only_on_a` key: {a}"
    );
    assert_eq!(
        a["shared"],
        json!("patched"),
        "turn-1 missing the patch: {a}"
    );
    assert!(
        a.get("only_on_b").is_none(),
        "turn-1 was contaminated with turn-2's key: {a}"
    );

    let b = got["turn-2"].as_ref().expect("turn-2 metadata");
    assert_eq!(
        b["turn"],
        json!(2),
        "turn-2's own `turn` was clobbered by a sibling's value: {b}"
    );
    assert_eq!(
        b["only_on_b"],
        json!("beta"),
        "turn-2 lost its own `only_on_b` key — data loss: {b}"
    );
    assert_eq!(
        b["shared"],
        json!("patched"),
        "turn-2 missing the patch: {b}"
    );
    assert!(
        b.get("only_on_a").is_none(),
        "turn-2 was contaminated with turn-1's key: {b}"
    );
}

/// A NULL-metadata row and a populated row under one `source_id`. NULL must
/// still mean "empty object" for ITS OWN row — it must not inherit the sibling's
/// metadata, and the sibling must not be flattened to just the patch.
#[tokio::test]
async fn null_metadata_row_and_populated_row_are_both_preserved() {
    let conv = Conversation::open("td-metadata-multirow-null", "conv-multirow-002").await;

    conv.seed("has-none", None).await;
    conv.seed("has-some", Some(json!({ "kept": true, "n": 7 })))
        .await;

    let updated = conv.patch(json!({ "added": "x" })).await;
    assert_eq!(updated, 2);

    let got = conv.metadata_by_content().await;

    let none_row = got["has-none"].as_ref().expect("has-none metadata");
    assert_eq!(
        none_row,
        &json!({ "added": "x" }),
        "a NULL-metadata row must become exactly the patch, not the sibling's \
         metadata: {none_row}"
    );

    let some_row = got["has-some"].as_ref().expect("has-some metadata");
    assert_eq!(some_row["kept"], json!(true), "populated row lost `kept`");
    assert_eq!(some_row["n"], json!(7), "populated row lost `n`");
    assert_eq!(some_row["added"], json!("x"), "populated row missing patch");
}

/// The single-row case is unchanged: shallow merge, top-level overwrite,
/// arrays REPLACED not concatenated, nested objects replaced wholesale.
#[tokio::test]
async fn single_row_shallow_merge_semantics_unchanged() {
    let conv = Conversation::open("td-metadata-multirow-single", "conv-multirow-003").await;

    conv.seed(
        "only",
        Some(json!({
            "keep": "me",
            "overwrite": "old",
            "refs": ["a"],
            "nested": { "inner": 1 }
        })),
    )
    .await;

    let updated = conv
        .patch(json!({
            "overwrite": "new",
            "refs": ["b"],
            "nested": { "other": 2 },
            "fresh": true
        }))
        .await;
    assert_eq!(updated, 1);

    let got = conv.metadata_by_content().await;
    let m = got["only"].as_ref().expect("metadata");

    assert_eq!(m["keep"], json!("me"), "untouched key must survive");
    assert_eq!(m["overwrite"], json!("new"), "top-level key must overwrite");
    assert_eq!(
        m["refs"],
        json!(["b"]),
        "arrays REPLACE, they do not concatenate"
    );
    assert_eq!(
        m["nested"],
        json!({ "other": 2 }),
        "merge is SHALLOW — nested objects are replaced wholesale, not merged"
    );
    assert_eq!(m["fresh"], json!(true), "new key must be inserted");
}

/// Single row starting from NULL metadata still yields exactly the patch.
#[tokio::test]
async fn single_null_metadata_row_becomes_the_patch() {
    let conv = Conversation::open("td-metadata-multirow-single-null", "conv-multirow-004").await;

    conv.seed("only", None).await;

    let updated = conv.patch(json!({ "a": 1 })).await;
    assert_eq!(updated, 1);

    let got = conv.metadata_by_content().await;
    assert_eq!(got["only"], Some(json!({ "a": 1 })));
}

/// count == 0 must still error, with the SAME wording. Consumers may match on
/// it; the per-row fix must not touch this path.
#[tokio::test]
async fn no_matching_source_id_errors_with_unchanged_message() {
    let conv = Conversation::open("td-metadata-multirow-missing", "does-not-exist-anywhere").await;

    let err = conv
        .mem
        .update_episode_metadata(&conv.source_id)
        .patch(json!({ "a": 1 }))
        .await
        .expect_err("a source_id with zero episodes must error");

    // Exact, including `MemoryError::Other`'s own `other: ` Display prefix —
    // the whole rendered string is what a consumer sees.
    assert_eq!(
        err.to_string(),
        "other: update_episode_metadata: no episode found with \
         source_id=does-not-exist-anywhere",
        "the count==0 error wording is part of the surface and must not drift"
    );
}

/// A non-object patch is still rejected at `.await`, unchanged.
#[tokio::test]
async fn non_object_patch_still_rejected() {
    let conv = Conversation::open("td-metadata-multirow-bad-patch", "irrelevant").await;

    let err = conv
        .mem
        .update_episode_metadata(&conv.source_id)
        .patch(json!(42))
        .await
        .expect_err("a non-object patch must error");

    assert!(
        err.to_string()
            .contains("update_episode_metadata: patch must be a JSON object"),
        "unexpected error: {err}"
    );
}
