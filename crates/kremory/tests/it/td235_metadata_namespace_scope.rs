//! TD-235 — the episode metadata/uri WRITERS must be namespace-scoped, exactly
//! as their sibling READER `Memory::recall_by_source_id` already is.
//!
//! Nothing enforces uniqueness on `episodes.source_id` — Migration 007 creates
//! a plain `CREATE INDEX IF NOT EXISTS idx_episodes_source_id`, not a UNIQUE
//! one (`core/migrations/defs_b.rs:123`) — and `source_id` is caller-chosen.
//! So two tenants sharing one database can collide on one source id, and
//! before this fix EVERY statement in both writers was source-id-global:
//!
//! - `update_episode_metadata`: `SELECT COUNT(*) FROM episodes WHERE
//!   source_id = ?1` and `SELECT id, metadata FROM episodes WHERE
//!   source_id = ?1` — no `group_id` predicate, so the per-row `UPDATE ...
//!   WHERE id = ?2` faithfully patched the OTHER tenant's rows too.
//! - `update_source_uri`: same COUNT, then `UPDATE episodes SET source_uri = ?1
//!   WHERE source_id = ?2` — a blind cross-tenant overwrite.
//!
//! Meanwhile `recall_by_source_id` (same file, ~40 lines below) applied
//! `AND (?2 IS NULL OR group_id = ?2)`. Reader scoped, writers not.
//!
//! ALL SIX scope-sensitive tests here were observed RED before the fix existed
//! — a test nobody has seen fail is an unvalidated instrument, not evidence.
//! The RED was taken against a build carrying the `.in_namespace()` SETTER but
//! NOT the SQL predicate, so every failure is a real assertion failure about
//! behaviour, never a compile error about a missing method. All six reported
//! the same shape (`left: 2, right: 1` — the write hit both tenants) or, for
//! the two not-found cases, an unexpected `Ok(1)`.
//!
//! The two `*_spans_all_namespaces` tests passed in BOTH states by design: they
//! pin the documented no-default leak path, which this change does not alter.

#![allow(clippy::unwrap_used, clippy::expect_used)]

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

/// The source id both tenants collide on. Caller-chosen, not unique.
const SHARED: &str = "shared-doc-001";

fn ns_a() -> Namespace {
    Namespace::new("tenant-a")
}

fn ns_b() -> Namespace {
    Namespace::new("tenant-b")
}

/// The distinguishing payload of one seeded episode.
///
/// Bundled rather than passed positionally so `seed` stays at or under the
/// project's `too-many-arguments` threshold (3) WITHOUT an `#[allow]` — the
/// same shape, and the same stated reason, as
/// `td_metadata_multirow_patch::Conversation`.
struct SeedRow<'a> {
    content: &'a str,
    metadata: serde_json::Value,
    uri: &'a str,
}

/// One in-memory `Memory` serving two tenants that collide on `SHARED`.
///
/// Bundling the handle with its seeding + reading helpers keeps every call site
/// at or under the project's `too-many-arguments` threshold without an
/// `#[allow]`, and mirrors the real shape: one database, many namespaces.
struct TwoTenants {
    mem: Memory,
}

impl TwoTenants {
    /// `default_ns == None` reproduces a `Memory` with no builder default —
    /// the documented span-all-namespaces path.
    async fn open(default_ns: Option<Namespace>) -> Self {
        let mut b = Memory::open(":memory:")
            .with_llm(null_llm())
            .with_embedder(null_embedder());
        if let Some(ns) = default_ns {
            b = b.default_namespace(ns);
        }
        let mem = b.await.expect("Memory::open(\":memory:\") must succeed");
        Self { mem }
    }

    /// Plant an episode row directly on the graph so each namespace can start
    /// with its OWN distinct metadata / source_uri. Going through `remember()`
    /// cannot set up this fixture: `remember()` alone leaves `metadata` NULL
    /// (pinned by `ingest_phase_boundary::phase_1_remember_alone_yields_
    /// episode_with_empty_metadata`), and the only other writers are the two
    /// methods under test.
    async fn seed(&self, ns: &Namespace, row: SeedRow<'_>) {
        let graph = self
            .mem
            .temporal_graph_for_test()
            .expect("Memory built via the builder path must carry a TemporalGraph");
        let group_id = self.mem.group_id_for_test(ns);

        let ep = EpisodeInsert::new(row.content, chrono::Utc::now())
            .source_type("test")
            .source_id(SHARED)
            .source_uri(row.uri)
            .metadata(row.metadata);

        graph
            .insert_episode_with_group(ep, Some(&group_id))
            .await
            .expect("insert_episode_with_group must succeed");
    }

    /// Seed BOTH tenants with distinct metadata + uri under the same source id.
    async fn seed_both(&self) {
        self.seed(
            &ns_a(),
            SeedRow {
                content: "tenant a content",
                metadata: json!({ "owner": "a" }),
                uri: "a/v1",
            },
        )
        .await;
        self.seed(
            &ns_b(),
            SeedRow {
                content: "tenant b content",
                metadata: json!({ "owner": "b" }),
                uri: "b/v1",
            },
        )
        .await;
    }

    /// The single episode `ns` holds for `SHARED`. Read through the ALREADY
    /// namespace-scoped reader, so the assertion cannot be satisfied by the
    /// same scoping bug it is testing for.
    async fn episode(&self, ns: &Namespace) -> kremory::core::schema::Episode {
        let mut eps = self
            .mem
            .recall_by_source_id(SHARED, Some(ns.clone()))
            .await
            .expect("recall_by_source_id must not error");
        assert_eq!(
            eps.len(),
            1,
            "fixture invariant: exactly one episode per namespace for {SHARED}"
        );
        eps.remove(0)
    }

    async fn metadata(&self, ns: &Namespace) -> serde_json::Value {
        self.episode(ns)
            .await
            .metadata
            .expect("seeded episode must carry metadata")
    }

    async fn uri(&self, ns: &Namespace) -> String {
        self.episode(ns)
            .await
            .source_uri
            .expect("seeded episode must carry a source_uri")
    }
}

// ── update_episode_metadata ───────────────────────────────────────────────────

/// RED WITNESS (default-namespace path — expressible without the new setter).
///
/// Observed against the unfixed query, verbatim:
/// `assertion \`left == right\` failed: only tenant-a's single row may be
///  written / left: 2 / right: 1` — the patch landed on BOTH tenants.
#[tokio::test]
async fn metadata_patch_under_default_namespace_leaves_other_namespace_untouched() {
    let t = TwoTenants::open(Some(ns_a())).await;
    t.seed_both().await;

    let n = t
        .mem
        .update_episode_metadata(SHARED)
        .patch(json!({ "docType": "spec" }))
        .await
        .expect("update_episode_metadata must succeed");

    assert_eq!(n, 1, "only tenant-a's single row may be written");
    assert_eq!(
        t.metadata(&ns_a()).await,
        json!({ "owner": "a", "docType": "spec" }),
        "tenant-a is the default namespace and must receive the patch"
    );
    assert_eq!(
        t.metadata(&ns_b()).await,
        json!({ "owner": "b" }),
        "tenant-b metadata must be untouched"
    );
}

/// Explicit `.in_namespace(ns)` overrides the builder default and scopes the
/// write to that namespace only.
#[tokio::test]
async fn metadata_patch_with_explicit_namespace_scopes_to_that_namespace() {
    // Default is tenant-a; the explicit selector picks tenant-b instead, so a
    // pass cannot be an accident of the default also being correct.
    let t = TwoTenants::open(Some(ns_a())).await;
    t.seed_both().await;

    let n = t
        .mem
        .update_episode_metadata(SHARED)
        .in_namespace(ns_b())
        .patch(json!({ "docType": "spec" }))
        .await
        .expect("update_episode_metadata must succeed");

    assert_eq!(n, 1, "only tenant-b's single row may be written");
    assert_eq!(
        t.metadata(&ns_b()).await,
        json!({ "owner": "b", "docType": "spec" }),
        "the explicitly selected namespace must receive the patch"
    );
    assert_eq!(
        t.metadata(&ns_a()).await,
        json!({ "owner": "a" }),
        "the builder-default namespace must NOT be touched when overridden"
    );
}

/// A namespace holding no row for this source id must report "no episode
/// found" even though a SIBLING namespace does hold one — i.e. the existence
/// COUNT is scoped too, not just the write.
#[tokio::test]
async fn metadata_patch_in_namespace_without_the_source_id_errors() {
    let t = TwoTenants::open(None).await;
    t.seed(
        &ns_a(),
        SeedRow {
            content: "tenant a content",
            metadata: json!({ "owner": "a" }),
            uri: "a/v1",
        },
    )
    .await;

    let err = t
        .mem
        .update_episode_metadata(SHARED)
        .in_namespace(ns_b())
        .patch(json!({ "docType": "spec" }))
        .await
        .expect_err("tenant-b holds no row for this source id");

    assert!(
        err.to_string()
            .contains("update_episode_metadata: no episode found with source_id="),
        "expected the scoped not-found error, got: {err}"
    );
    assert_eq!(
        t.metadata(&ns_a()).await,
        json!({ "owner": "a" }),
        "the failed call must not have written tenant-a's row"
    );
}

/// The documented leak path: no explicit namespace AND no builder default →
/// the write spans every namespace. Pinned deliberately so a future change to
/// this behaviour is a test failure, not a silent semantics shift.
#[tokio::test]
async fn metadata_patch_with_no_default_namespace_spans_all_namespaces() {
    let t = TwoTenants::open(None).await;
    t.seed_both().await;

    let n = t
        .mem
        .update_episode_metadata(SHARED)
        .patch(json!({ "docType": "spec" }))
        .await
        .expect("update_episode_metadata must succeed");

    assert_eq!(n, 2, "both namespaces' rows are in scope");
    assert_eq!(
        t.metadata(&ns_a()).await,
        json!({ "owner": "a", "docType": "spec" })
    );
    assert_eq!(
        t.metadata(&ns_b()).await,
        json!({ "owner": "b", "docType": "spec" })
    );
}

// ── threaded namespaces (group_id identity) ──────────────────────────────────

/// A thread is PART of the namespace identity: `Namespace::new("ws")
/// .with_thread("t")` is persisted as `group_id = "ws:t"`
/// (`memory::engine_handle::namespace_to_group_id`). Every other
/// namespace-scoped facade op resolves scope through that function
/// (`facade/dream.rs` x10, `facade/reverse.rs` x6, and this file's own test
/// helpers) — only the `source_id` reader/writers used the raw `ns.namespace`,
/// which silently DROPS the thread.
///
/// Consequence before the fix: the resolved filter was `"ws"` while every row
/// carried `"ws:t"`, so the predicate matched NOTHING. It fails closed rather
/// than leaking, but a threaded consumer could not read or write at all.
#[tokio::test]
async fn threaded_namespace_scopes_writers_to_the_full_group_id() {
    let t = TwoTenants::open(None).await;
    let t1 = Namespace::new("ws").with_thread("t1");
    let t2 = Namespace::new("ws").with_thread("t2");

    t.seed(
        &t1,
        SeedRow {
            content: "thread one",
            metadata: json!({ "owner": "t1" }),
            uri: "t1/v1",
        },
    )
    .await;
    t.seed(
        &t2,
        SeedRow {
            content: "thread two",
            metadata: json!({ "owner": "t2" }),
            uri: "t2/v1",
        },
    )
    .await;

    let n = t
        .mem
        .update_episode_metadata(SHARED)
        .in_namespace(t1.clone())
        .patch(json!({ "docType": "spec" }))
        .await
        .expect("a threaded namespace must resolve to its own group_id");

    assert_eq!(n, 1, "only thread t1's row may be written");
    assert_eq!(
        t.metadata(&t1).await,
        json!({ "owner": "t1", "docType": "spec" })
    );
    assert_eq!(
        t.metadata(&t2).await,
        json!({ "owner": "t2" }),
        "a sibling THREAD of the same workspace is a distinct scope"
    );

    let n = t
        .mem
        .update_source_uri(SHARED)
        .in_namespace(t1.clone())
        .to("t1/v2")
        .await
        .expect("a threaded namespace must resolve to its own group_id");

    assert_eq!(n, 1, "only thread t1's row may be written");
    assert_eq!(t.uri(&t1).await, "t1/v2");
    assert_eq!(t.uri(&t2).await, "t2/v1", "sibling thread untouched");
}

/// The reader half of the same defect: `recall_by_source_id` dropped the thread
/// too, so a threaded namespace returned ZERO episodes for a source id it
/// demonstrably owns. Asserted directly rather than through the `episode()`
/// helper, so the failure names the reader rather than a fixture invariant.
#[tokio::test]
async fn threaded_namespace_is_visible_to_recall_by_source_id() {
    let t = TwoTenants::open(None).await;
    let t1 = Namespace::new("ws").with_thread("t1");
    t.seed(
        &t1,
        SeedRow {
            content: "thread one",
            metadata: json!({ "owner": "t1" }),
            uri: "t1/v1",
        },
    )
    .await;

    let eps = t
        .mem
        .recall_by_source_id(SHARED, Some(t1))
        .await
        .expect("recall_by_source_id must not error");

    assert_eq!(
        eps.len(),
        1,
        "a threaded namespace must see the row it owns"
    );
}

// ── update_source_uri ─────────────────────────────────────────────────────────

/// RED WITNESS (default-namespace path — expressible without the new setter).
///
/// Observed against the unfixed query, verbatim:
/// `assertion \`left == right\` failed: only tenant-a's single row may be
///  written / left: 2 / right: 1` — the uri overwrite hit BOTH tenants.
#[tokio::test]
async fn source_uri_update_under_default_namespace_leaves_other_namespace_untouched() {
    let t = TwoTenants::open(Some(ns_a())).await;
    t.seed_both().await;

    let n = t
        .mem
        .update_source_uri(SHARED)
        .to("patched/v2")
        .await
        .expect("update_source_uri must succeed");

    assert_eq!(n, 1, "only tenant-a's single row may be written");
    assert_eq!(t.uri(&ns_a()).await, "patched/v2");
    assert_eq!(
        t.uri(&ns_b()).await,
        "b/v1",
        "tenant-b source_uri must be untouched"
    );
}

/// Explicit `.in_namespace(ns)` overrides the builder default.
#[tokio::test]
async fn source_uri_update_with_explicit_namespace_scopes_to_that_namespace() {
    let t = TwoTenants::open(Some(ns_a())).await;
    t.seed_both().await;

    let n = t
        .mem
        .update_source_uri(SHARED)
        .in_namespace(ns_b())
        .to("patched/v2")
        .await
        .expect("update_source_uri must succeed");

    assert_eq!(n, 1, "only tenant-b's single row may be written");
    assert_eq!(t.uri(&ns_b()).await, "patched/v2");
    assert_eq!(
        t.uri(&ns_a()).await,
        "a/v1",
        "the builder-default namespace must NOT be touched when overridden"
    );
}

/// Scoped existence check, mirroring the metadata case.
#[tokio::test]
async fn source_uri_update_in_namespace_without_the_source_id_errors() {
    let t = TwoTenants::open(None).await;
    t.seed(
        &ns_a(),
        SeedRow {
            content: "tenant a content",
            metadata: json!({ "owner": "a" }),
            uri: "a/v1",
        },
    )
    .await;

    let err = t
        .mem
        .update_source_uri(SHARED)
        .in_namespace(ns_b())
        .to("patched/v2")
        .await
        .expect_err("tenant-b holds no row for this source id");

    assert!(
        err.to_string()
            .contains("update_source_uri: no episode found with source_id="),
        "expected the scoped not-found error, got: {err}"
    );
    assert_eq!(
        t.uri(&ns_a()).await,
        "a/v1",
        "the failed call must not have written tenant-a's row"
    );
}

/// The documented leak path, mirroring the metadata case.
#[tokio::test]
async fn source_uri_update_with_no_default_namespace_spans_all_namespaces() {
    let t = TwoTenants::open(None).await;
    t.seed_both().await;

    let n = t
        .mem
        .update_source_uri(SHARED)
        .to("patched/v2")
        .await
        .expect("update_source_uri must succeed");

    assert_eq!(n, 2, "both namespaces' rows are in scope");
    assert_eq!(t.uri(&ns_a()).await, "patched/v2");
    assert_eq!(t.uri(&ns_b()).await, "patched/v2");
}
