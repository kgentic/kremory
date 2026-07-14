#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-114 e2e — filtered-ANN over-fetch improves per-namespace vector recall.
//!
//! Per `.ai-docs/tech-debt/tech-debt-register.md`: TD-114's `plan_index_fetch`
//! over-fetch code path lives inside `vector_search_*_with_index`, which was
//! DORMANT until TD-115 fixed the vector index so it actually engages instead
//! of always falling back to brute-force (`vector_search_brute_force`, which
//! has no `vector_top_k` fetch-size concept and so cannot exhibit the
//! shortfall TD-114 defends against). This test could not be written until
//! TD-115 landed — it exercises the real DiskANN index path end-to-end.
//!
//! ## Scenario
//!
//! A shared multi-namespace DB: 97 "bulk" entities in namespace `bulk-ns`
//! whose embeddings are near-but-not-identical to the query vector (a small,
//! strictly-increasing perturbation per entity — deliberately NOT exact
//! duplicates; a calibration spike showed libSQL's DiskANN index returns a
//! degenerate, far-below-`k` result count when many rows share an IDENTICAL
//! vector, which is an ANN-index artifact unrelated to TD-114 and would make
//! this fixture flaky), plus 3 "target" entities in namespace `small-ns`
//! whose embeddings are noticeably farther from the query (still far closer
//! than an orthogonal vector, but strictly farther than every bulk entity).
//! `vector_top_k`'s global ranking (no namespace awareness) therefore always
//! ranks all 97 bulk entities ahead of the 3 target entities.
//!
//! Without over-fetch, requesting the global top-`limit` (limit=5) and then
//! post-filtering by `small-ns` yields **zero** rows — the classic
//! filtered-ANN post-filter shortfall. `plan_index_fetch` scales the fetch by
//! the estimated namespace selectivity (`small-ns` is 3/100 of the DB), so
//! the real over-fetched query recovers all 3 target entities.

use kremory::core::schema::TemporalGraph;
use kremory::core::search::{SearchFilters, VectorSearchEntitiesParams};

const DIM: usize = 8;
const BULK_N: usize = 97;
const SMALL_N: usize = 3;

/// Args-as-object per rust-conventions §too_many_arguments (clippy.toml
/// threshold 3).
struct NewTestEntity<'a> {
    id: &'a str,
    group_id: &'a str,
    embedding: &'a [f32],
}

async fn insert_entity(graph: &TemporalGraph, entity: NewTestEntity<'_>) {
    let NewTestEntity {
        id,
        group_id,
        embedding,
    } = entity;
    let now = chrono::Utc::now().to_rfc3339();
    graph
        .conn
        .execute(
            "INSERT INTO entities (id, group_id, entity_type_id, properties, recorded_at) \
             VALUES (?1, ?2, 0, '{}', ?3)",
            libsql::params![id, group_id, now],
        )
        .await
        .expect("insert entity");
    graph
        .set_entity_embedding(id, embedding)
        .await
        .expect("set embedding");
}

#[tokio::test]
async fn overfetch_recovers_small_namespace_results_that_naive_topk_would_miss() {
    let tmp = std::env::temp_dir().join(format!("td114_overfetch_e2e_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let path = tmp.to_str().unwrap().to_string();

    let graph = TemporalGraph::open_with_dim(&path, DIM)
        .await
        .expect("open_with_dim (runs migrate_023 — TD-115 prerequisite)");

    let mut query = vec![0.0_f32; DIM];
    query[0] = 1.0;

    // 97 bulk entities in `bulk-ns`: small, strictly-increasing perturbation
    // per entity (cosine distance from query ranges ~0.0000005 .. ~0.0047) —
    // close to the query but NOT identical to it or each other.
    for i in 0..BULK_N {
        let mut emb = vec![0.0_f32; DIM];
        emb[0] = 1.0;
        emb[1] = 0.001 * (i as f32 + 1.0);
        insert_entity(
            &graph,
            NewTestEntity {
                id: &format!("bulk-{i}"),
                group_id: "bulk-ns",
                embedding: &emb,
            },
        )
        .await;
    }

    // 3 target entities in `small-ns`: noticeably farther from the query
    // (perturbation ~0.30-0.32, cosine distance ~0.045) — strictly farther
    // than every bulk entity (max bulk perturbation 0.097), so bulk always
    // dominates the unfiltered global ranking.
    let mut target_ids = Vec::new();
    for i in 0..SMALL_N {
        let mut emb = vec![0.0_f32; DIM];
        emb[0] = 1.0;
        emb[1] = 0.3 + 0.01 * (i as f32);
        let id = format!("small-{i}");
        insert_entity(
            &graph,
            NewTestEntity {
                id: &id,
                group_id: "small-ns",
                embedding: &emb,
            },
        )
        .await;
        target_ids.push(id);
    }

    let total = BULK_N + SMALL_N;
    assert_eq!(
        total, 100,
        "sanity: fixture sizing assumption for the selectivity comment above"
    );

    // ── Baseline: naive top-k (no over-fetch), mirroring what
    //    `vector_search_with_index` would do WITHOUT TD-114's
    //    `plan_index_fetch` scaling — global top-`limit` then post-filter.
    let limit = 5usize;
    let vec_str = format!(
        "[{}]",
        query
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut naive_rows = graph
        .conn
        .query(
            &format!(
                "SELECT e.id FROM vector_top_k('entities_vec_idx', vector(?1), {limit}) AS v \
                 JOIN entities AS e ON e.rowid = v.id \
                 WHERE e.group_id = 'small-ns'"
            ),
            libsql::params![vec_str],
        )
        .await
        .expect("naive vector_top_k query");
    let mut naive_ids = Vec::new();
    while let Some(row) = naive_rows.next().await.expect("row") {
        naive_ids.push(row.get::<String>(0).expect("id"));
    }
    assert!(
        naive_ids.is_empty(),
        "baseline sanity check failed: naive top-{limit} (no over-fetch) should return \
         ZERO small-ns rows given {BULK_N} closer bulk entities rank ahead of them \
         — got {naive_ids:?}. If this fails, the fixture no longer demonstrates the \
         shortfall TD-114 defends against."
    );

    // ── Real path: `vector_search_entities` (public API), which internally
    //    calls `plan_index_fetch` to scale the fetch by namespace
    //    selectivity before the real ANN query + post-filter + LIMIT.
    let filters = SearchFilters::for_group("small-ns");
    let hits = graph
        .vector_search_entities(VectorSearchEntitiesParams {
            query_embedding: &query,
            limit,
            filters: &filters,
        })
        .await
        .expect("vector_search_entities");

    let mut returned_ids: Vec<String> = hits.iter().map(|h| h.item.id.clone()).collect();
    returned_ids.sort();
    let mut expected_ids = target_ids.clone();
    expected_ids.sort();

    assert_eq!(
        returned_ids, expected_ids,
        "TD-114 over-fetch must recover all {SMALL_N} small-ns entities that the naive \
         top-{limit} query missed entirely; got {returned_ids:?}, expected {expected_ids:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}
