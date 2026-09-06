#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ADR-080 — prior-turn replay: the graph-layer read half.
//!
//! These tests drive `TemporalGraph::prior_episodes_for_source` directly, which
//! is the query that gives the extractor the preceding turns of a conversation
//! so references in the current turn resolve.
//!
//! Every test here was verified to go RED before the implementation existed —
//! a green test nobody has seen fail is an unvalidated instrument, not evidence
//! (see `.ai-docs/adrs/adr-080-prior-turn-replay-into-extraction-2026-09-06.md`).
//!
//! The thread key is `episodes.source_id`, which is what `.from_chat(id)` sets.
//! No new public API was needed to thread a conversation — see ADR-080 §D1.

use std::sync::Arc;

use chrono::{TimeZone, Utc};
use kremory::core::graph::{EpisodeInsert, PriorEpisodesParams};
use kremory::core::schema::TemporalGraph;

/// Insert `n` episodes under `source_id` / `group_id`, contents `"turn-0"`…
/// Returns the inserted ids in insertion order.
struct Seed<'a> {
    source_id: &'a str,
    group_id: Option<&'a str>,
    contents: &'a [&'a str],
}

async fn seed(graph: &TemporalGraph, p: Seed<'_>) -> Vec<i64> {
    let Seed {
        source_id,
        group_id,
        contents,
    } = p;
    let mut ids = Vec::new();
    for (i, content) in contents.iter().enumerate() {
        // Deliberately NON-monotonic world time: the first episode is dated
        // LATEST. `prior_episodes_for_source` must order by `id` (true insertion
        // order), not by `timestamp` (the caller-supplied world clock), so this
        // fixture would break an ORDER BY timestamp implementation. kremory is
        // bi-temporal — ingesting an older document after a newer one is legal.
        let ts = Utc
            .with_ymd_and_hms(2020, 1, 1, 0, 0, 0)
            .single()
            .expect("valid ts")
            + chrono::Duration::days((contents.len() - i) as i64);
        let ep = EpisodeInsert::new(content, ts)
            .source_type("test")
            .source_id(source_id);
        ids.push(
            graph
                .insert_episode_with_group(ep, group_id)
                .await
                .expect("insert episode"),
        );
    }
    ids
}

async fn open() -> Arc<TemporalGraph> {
    Arc::new(
        TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory"),
    )
}

#[tokio::test]
async fn returns_latest_n_turns_oldest_first() {
    let graph = open().await;
    let contents: Vec<String> = (0..12).map(|i| format!("turn-{i}")).collect();
    let refs: Vec<&str> = contents.iter().map(String::as_str).collect();
    let ids = seed(
        &graph,
        Seed {
            source_id: "conv-a",
            group_id: Some("ns"),
            contents: &refs,
        },
    )
    .await;

    // Pretend we are ingesting a 13th turn.
    let next_id = ids.last().copied().expect("ids") + 1;
    let got = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "conv-a",
            group_id: Some("ns"),
            before_id: next_id,
            limit: 10,
        })
        .await
        .expect("query");

    // The LATEST 10 of 12 — turn-2 .. turn-11 — and in CHRONOLOGICAL order.
    // Both halves matter: taking the latest is what makes replay relevant, and
    // re-sorting ascending is what lets the model read the conversation forwards
    // (mem0 does the same: `created_at DESC` then re-sort ASC, storage.py:298).
    assert_eq!(got.len(), 10, "expected the latest 10 of 12, got {got:?}");
    assert_eq!(got.first().map(String::as_str), Some("turn-2"));
    assert_eq!(got.last().map(String::as_str), Some("turn-11"));
    let expected: Vec<String> = (2..12).map(|i| format!("turn-{i}")).collect();
    assert_eq!(got, expected, "must be oldest-first, not newest-first");
}

#[tokio::test]
async fn excludes_the_episode_being_ingested() {
    let graph = open().await;
    let ids = seed(
        &graph,
        Seed {
            source_id: "conv-a",
            group_id: Some("ns"),
            contents: &["a", "b", "c"],
        },
    )
    .await;

    // `before_id` = the LAST inserted id, i.e. the episode we are extracting.
    // Without the `id < ?` bound the episode replays ITSELF, which would feed
    // the extractor its own text as "earlier context".
    let current = ids.last().copied().expect("ids");
    let got = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "conv-a",
            group_id: Some("ns"),
            before_id: current,
            limit: 10,
        })
        .await
        .expect("query");

    assert_eq!(got, vec!["a".to_string(), "b".to_string()]);
    assert!(!got.contains(&"c".to_string()), "episode replayed itself");
}

#[tokio::test]
async fn isolated_by_source_id() {
    let graph = open().await;
    seed(
        &graph,
        Seed {
            source_id: "conv-a",
            group_id: Some("ns"),
            contents: &["a1", "a2"],
        },
    )
    .await;
    let b_ids = seed(
        &graph,
        Seed {
            source_id: "conv-b",
            group_id: Some("ns"),
            contents: &["b1", "b2"],
        },
    )
    .await;

    let got = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "conv-b",
            group_id: Some("ns"),
            before_id: b_ids.last().copied().expect("ids"),
            limit: 10,
        })
        .await
        .expect("query");

    assert_eq!(got, vec!["b1".to_string()]);
    assert!(
        !got.iter().any(|c| c.starts_with('a')),
        "leaked another conversation's turns: {got:?}"
    );
}

#[tokio::test]
async fn isolated_by_namespace_including_the_null_group() {
    let graph = open().await;
    // Same source_id in two different namespaces — a realistic multi-tenant
    // collision, since source ids are caller-chosen and not globally unique.
    seed(
        &graph,
        Seed {
            source_id: "shared-id",
            group_id: Some("tenant-a"),
            contents: &["a1", "a2"],
        },
    )
    .await;
    let null_ids = seed(
        &graph,
        Seed {
            source_id: "shared-id",
            group_id: None,
            contents: &["n1", "n2"],
        },
    )
    .await;

    // NULL group. This is the case a naive `group_id = ?` gets WRONG: in SQL
    // `NULL = NULL` is NULL, not true, so `=` would return zero rows here while
    // appearing to work in every namespaced test.
    let got_null = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "shared-id",
            group_id: None,
            before_id: null_ids.last().copied().expect("ids"),
            limit: 10,
        })
        .await
        .expect("query");
    assert_eq!(
        got_null,
        vec!["n1".to_string()],
        "NULL-group scoping broken — `IS` vs `=`"
    );

    // And the namespaced side must not see the NULL-group rows.
    let got_a = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "shared-id",
            group_id: Some("tenant-a"),
            before_id: 10_000,
            limit: 10,
        })
        .await
        .expect("query");
    assert_eq!(got_a, vec!["a1".to_string(), "a2".to_string()]);
    assert!(
        !got_a.iter().any(|c| c.starts_with('n')),
        "cross-namespace leak: {got_a:?}"
    );
}

#[tokio::test]
async fn depth_zero_is_the_off_switch() {
    let graph = open().await;
    let ids = seed(
        &graph,
        Seed {
            source_id: "conv-a",
            group_id: Some("ns"),
            contents: &["a", "b", "c"],
        },
    )
    .await;

    let got = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "conv-a",
            group_id: Some("ns"),
            before_id: ids.last().copied().expect("ids") + 1,
            limit: 0,
        })
        .await
        .expect("query");

    // `PipelineConfig::prior_turn_replay_depth = 0` must be a true off switch —
    // this is the control arm for any A/B of the feature.
    assert!(got.is_empty(), "depth 0 must disable replay entirely");
}

#[tokio::test]
async fn empty_source_id_returns_nothing() {
    let graph = open().await;
    seed(
        &graph,
        Seed {
            source_id: "",
            group_id: Some("ns"),
            contents: &["a", "b"],
        },
    )
    .await;

    let got = graph
        .prior_episodes_for_source(PriorEpisodesParams {
            source_id: "",
            group_id: Some("ns"),
            before_id: 10_000,
            limit: 10,
        })
        .await
        .expect("query");

    assert!(got.is_empty(), "empty source id must not match anything");
}

#[tokio::test]
async fn source_id_for_episode_resolves_and_handles_absence() {
    let graph = open().await;
    let ids = seed(
        &graph,
        Seed {
            source_id: "conv-a",
            group_id: Some("ns"),
            contents: &["only"],
        },
    )
    .await;
    let id = ids.first().copied().expect("ids");

    assert_eq!(
        graph
            .source_id_for_episode(id)
            .await
            .expect("resolve")
            .as_deref(),
        Some("conv-a")
    );

    // Missing episode -> None, not an error. The deferred ingest path calls this
    // and must degrade to "no replay", never fail the whole ingest.
    assert_eq!(
        graph.source_id_for_episode(999_999).await.expect("resolve"),
        None
    );

    // Episode with no source_id at all -> None.
    let ep = EpisodeInsert::new(
        "untagged",
        Utc.with_ymd_and_hms(2021, 5, 5, 0, 0, 0)
            .single()
            .expect("ts"),
    );
    let untagged = graph
        .insert_episode_with_group(ep, Some("ns"))
        .await
        .expect("insert");
    assert_eq!(
        graph
            .source_id_for_episode(untagged)
            .await
            .expect("resolve"),
        None
    );
}
