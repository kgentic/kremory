use kremory_mcp::params::RetrievedContextWire;

use crate::wire::SearchResultWire;

/// NAIVE BASELINE FUSION — real fusion/fairness decision deferred to Arch-1a
/// post-diagnostic per benchmark-completion-roadmap. Union by `id`
/// (first-seen wins across the two ranked lists), rank-interleaved
/// (`recall[0], content[0], recall[1], content[1], ...`) — NOT an RRF or any
/// score-aware fusion.
#[cfg(feature = "content-search")]
/// RRF (Reciprocal Rank Fusion) of two ranked result streams — replaces the v0
/// `naive_merge` rank-interleave (which was explicitly a placeholder: "not
/// kremory's answer to hybrid ranking").
///
/// Rationale (`.ai-docs/research/v011-recall-redesign/W2-hybrid-scoring.md` +
/// a LoCoMo diagnostic run, memory
/// `project_kremory_locomo_recall_root_cause_retrieval_surface`): RRF is the
/// tune-free dominant fusion across Elastic/Weaviate/Graphiti; kremory already
/// uses RRF_K=60 in `search.rs`. The naive 1:1 interleave DILUTED the content
/// stream — judged LoCoMo recall 70.4% for naive-hybrid vs 71.4% content-only;
/// RRF recovers to 71.4% (matches content-only) without letting the
/// LoCoMo-net-negative entity-graph stream dominate. `score(d) = Σ_list
/// 1/(RRF_K + rank_list(d))`, rank 1-based; dedup by id (a result present in
/// both streams accrues both contributions). Deterministic: fused-score desc,
/// then id asc on ties.
/// Truncation bound for the fused hybrid result set.
///
/// Extracted from [`hybrid_mode_results`] so the invariant it encodes is
/// nameable and testable. The invariant is an EQUIVALENCE, not an arithmetic
/// fact: this bin's fusion is a deliberate parallel implementation of the
/// library's `core::search::rrf_fuse_with_content`, and the two must agree on
/// what "no explicit limit" means.
///
/// - **explicit `k`** → honour it, exactly as the library honours `limit`.
/// - **no `k`** → the FULL deduped union, `recall_len + content_len`. Both
///   fusions dedupe by id into a `HashMap`, so the union is at most that many
///   and this bound therefore never truncates.
///
/// It previously read `recall_len.max(content_len)` — an expression that is
/// only correct if the two arms OVERLAP. They do not: see [`rrf_merge`]'s doc
/// on the disjoint entity-id/episode-id spaces. For disjoint arms `.max()`
/// silently discards `min(recall_len, content_len)` distinct results — the same
/// defect that was fixed on the library side but not here.
pub(crate) fn fused_cap(k: Option<usize>, recall_len: usize, content_len: usize) -> usize {
    k.unwrap_or(recall_len + content_len)
}

/// Minimal shape [`rrf_merge`] needs to fuse two ranked streams — id +
/// mutable score. Implemented for both [`SearchResultWire`] (the
/// `format=structured` shape `hybrid_mode_results` fuses) and
/// [`RetrievedContextWire`] (the full-fidelity shape
/// [`hybrid_items_for_render`] fuses for `format=text`) so ONE
/// fusion function serves both — the alternative (a second, hand-rolled
/// merge over the rich shape) would be exactly the "duplicated merge logic"
/// this trait exists to avoid.
pub(crate) trait RrfItem {
    fn rrf_id(&self) -> &str;
    fn rrf_score(&self) -> f32;
    fn set_rrf_score(&mut self, score: f32);
}

impl RrfItem for SearchResultWire {
    fn rrf_id(&self) -> &str {
        &self.id
    }
    fn rrf_score(&self) -> f32 {
        self.score
    }
    fn set_rrf_score(&mut self, score: f32) {
        self.score = score;
    }
}

impl RrfItem for RetrievedContextWire {
    fn rrf_id(&self) -> &str {
        &self.entity_id
    }
    fn rrf_score(&self) -> f32 {
        self.score
    }
    fn set_rrf_score(&mut self, score: f32) {
        self.score = score;
    }
}

pub(crate) fn rrf_merge<T: RrfItem>(a: Vec<T>, b: Vec<T>, rrf_k: usize) -> Vec<T> {
    // `k` is now the
    // boot-read `KREMORY_RRF_K` value (AppState.rrf_k), NOT a hardcoded
    // `const RRF_K = 60.0`, so this bin-local hybrid-fusion sweep site tracks
    // the same k as the library fusion sites.
    let rrf_k = rrf_k as f32;
    let mut fused: std::collections::HashMap<String, T> =
        std::collections::HashMap::with_capacity(a.len() + b.len());
    for list in [a, b] {
        for (rank, r) in list.into_iter().enumerate() {
            let contrib = 1.0 / (rrf_k + (rank as f32) + 1.0);
            let key = r.rrf_id().to_string();
            fused
                .entry(key)
                .and_modify(|e| {
                    let updated = e.rrf_score() + contrib;
                    e.set_rrf_score(updated);
                })
                .or_insert_with(|| {
                    let mut item = r;
                    item.set_rrf_score(contrib);
                    item
                });
        }
    }
    let mut merged: Vec<T> = fused.into_values().collect();
    merged.sort_by(|x, y| {
        y.rrf_score()
            .partial_cmp(&x.rrf_score())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.rrf_id().cmp(y.rrf_id()))
    });
    merged
}

