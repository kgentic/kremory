use crate::fusion::{fused_cap, rrf_merge};
use crate::wire::{SearchResultKindWire, SearchResultWire};

/// Test-only `SearchResultWire` builder for the `rrf_merge` tests below,
/// which exercise fusion arithmetic and are indifferent to `kind`/
/// `source_episode_id` — both new fields default to the values an
/// `Entity`-arm result would carry (`rrf_merge`'s `..r` spread passes
/// them through unchanged regardless).
#[cfg(feature = "content-search")]
fn sr(id: &str, content: &str, score: f32) -> SearchResultWire {
    SearchResultWire {
        id: id.into(),
        content: content.into(),
        score,
        kind: SearchResultKindWire::Entity,
        source_episode_id: None,
    }
}

/// Regression guard — with NO explicit `k`, hybrid must return the full
/// deduped union of both arms, never `max(recall, content)`.
///
/// This drives the same composition [`hybrid_mode_results`] performs — the
/// real [`rrf_merge`] followed by the real [`fused_cap`], in that order —
/// rather than asserting the cap arithmetic in isolation. Isolated
/// arithmetic hides exactly this class of bug: a test over
/// hand-shaped inputs to one pure function passed while the knob it
/// claimed to verify reached only one of two call sites.
///
/// The arms are DISJOINT here on purpose. That is the condition under
/// which the old `.max()` bound was wrong, and this module's own
/// [`rrf_merge`] doc records it as the measured normal case (entity-ids vs
/// episode-ids). Under `.max()` this fusion returned 2 of 4 results.
#[cfg(feature = "content-search")]
#[test]
fn hybrid_no_k_returns_full_deduped_union_not_max_arm() {
    let recall = vec![sr("e1", "entity-1", 0.9), sr("e2", "entity-2", 0.8)];
    let content = vec![sr("ep7", "episode-7", 0.7), sr("ep9", "episode-9", 0.6)];
    let (recall_len, content_len) = (recall.len(), content.len());

    let mut merged = rrf_merge(recall, content, 60);
    merged.truncate(fused_cap(None, recall_len, content_len));

    let mut ids: Vec<&str> = merged.iter().map(|r| r.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec!["e1", "e2", "ep7", "ep9"],
        "disjoint arms must all survive when the caller sets no limit; \
         the old `max(2, 2) = 2` bound silently dropped two of these"
    );
}

/// An explicit `k` still bounds the fused set — the fix must not turn the
/// context-budget flood back on.
#[cfg(feature = "content-search")]
#[test]
fn fused_cap_honours_explicit_k_over_union_size() {
    assert_eq!(fused_cap(Some(3), 2, 2), 3, "explicit k below union size");
    assert_eq!(fused_cap(Some(15), 2, 2), 15, "explicit k above union size");
    assert_eq!(fused_cap(Some(0), 9, 9), 0, "k=0 is honoured, not ignored");
}

/// Equivalence with the library fusion's no-limit rule
/// (`core::search::rrf_fuse_with_content`: `limit.unwrap_or(entity_count +
/// content_count)`).
///
/// **What this does and does not prove.** `rrf_fuse_with_content` is
/// `pub(crate)` in `kremory`, so it cannot be called from this bin and the
/// two implementations cannot be executed against one input here. This
/// therefore pins the REST side to the library's *stated* rule rather than
/// its *behaviour* — it catches a future edit to this bin, not a future
/// edit to the library. Closing that gap means either making the library fn
/// reachable for test, or collapsing this bin into a thin caller of it;
/// both remain open.
#[cfg(feature = "content-search")]
#[test]
fn fused_cap_matches_library_no_limit_rule() {
    for (r, c) in [(0, 0), (1, 0), (0, 1), (1, 1), (73, 50), (5, 5)] {
        assert_eq!(
            fused_cap(None, r, c),
            r + c,
            "no-limit bound must be the full union for arms ({r}, {c})"
        );
    }
}

/// `rrf_merge` — Reciprocal Rank Fusion. A result in BOTH streams accrues
/// both `1/(k+rank)` contributions and outranks single-stream hits; ties
/// break by id asc; the first-inserted (recall stream, processed first)
/// copy wins the content dedup while the content-stream duplicate only adds
/// to the fused score.
#[cfg(feature = "content-search")]
#[test]
fn rrf_merge_fuses_by_reciprocal_rank() {
    let recall = vec![sr("a", "recall-a", 0.9), sr("b", "recall-b", 0.8)];
    let content = vec![sr("b", "content-b", 0.7), sr("c", "content-c", 0.6)];
    let merged = rrf_merge(recall, content, 60);
    let ids: Vec<&str> = merged.iter().map(|r| r.id.as_str()).collect();
    // b ∈ both → 1/(60+2)+1/(60+1) ≈ 0.0325 (top). a (recall rank0) 1/61 ≈
    // 0.01639 edges c (content rank1) 1/62 ≈ 0.01613; id-asc tiebreak is
    // moot here since the scores differ.
    assert_eq!(ids, vec!["b", "a", "c"], "RRF fused order: {ids:?}");
    // recall's copy is inserted first (recall list processed first); the
    // content duplicate only adds to the score via `and_modify`.
    assert_eq!(
        merged.iter().find(|r| r.id == "b").unwrap().content,
        "recall-b",
        "first-inserted (recall) copy wins the dedup; content dup only adds score"
    );
    // the dual-stream hit must strictly outrank both single-stream hits.
    assert!(
        merged[0].id == "b" && merged[0].score > merged[1].score,
        "dual-stream result must outrank single-stream: {merged:?}"
    );
}

/// The bin-local
/// `rrf_merge` (fusion site 3 of 3) reads its RRF `k` from the argument
/// (boot `KREMORY_RRF_K` → `AppState.rrf_k`), NOT a hardcoded const. A
/// different `k` must produce different fused scores on identical input,
/// proving the value flows through rather than being ignored.
#[cfg(feature = "content-search")]
#[test]
fn rrf_merge_reads_k_argument_not_const() {
    let input = || (vec![sr("x", "x", 0.9)], vec![sr("y", "y", 0.8)]);
    let (ra, rb) = input();
    let k60 = rrf_merge(ra, rb, 60);
    let (ra, rb) = input();
    let k1 = rrf_merge(ra, rb, 1);
    // rank-0 contribution is 1/(k+0+1): k=60 → 1/61 ≈ 0.0164; k=1 → 1/2 = 0.5.
    let score_x_k60 = k60.iter().find(|r| r.id == "x").unwrap().score;
    let score_x_k1 = k1.iter().find(|r| r.id == "x").unwrap().score;
    assert!(
        (score_x_k60 - (1.0 / 61.0)).abs() < f32::EPSILON,
        "k=60 must yield 1/61 for a rank-0 hit, got {score_x_k60}"
    );
    assert!(
        (score_x_k1 - 0.5).abs() < f32::EPSILON,
        "k=1 must yield 1/2 for a rank-0 hit, got {score_x_k1}"
    );
    assert!(
        score_x_k1 > score_x_k60,
        "a smaller k must raise the fused score — proves rrf_merge reads its k arg"
    );
}
