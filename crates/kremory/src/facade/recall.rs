use super::*;

// ── RecallRequest ─────────────────────────────────────────────────────────────

/// Recall (search) request builder. Obtain via `mem.recall("…")`.
pub struct RecallRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) query: String,
    /// Single-namespace selector. Mutually exclusive with `namespaces`.
    pub(super) namespace: Option<Namespace>,
    /// Multi-namespace selector (ADR-029c Decision 6). Mutually exclusive with
    /// `namespace`. When set, fan-out via `tokio::join_all` executes one
    /// sub-query per namespace (each internally RRF-fused) and merges the
    /// per-namespace results by concatenating them and sorting by score —
    /// NOT a second cross-namespace RRF pass over the combined list.
    pub(super) namespaces: Option<Vec<Namespace>>,
    /// Per-namespace top-K cap before the cross-namespace score-sort merge
    /// (ADR-029c Decision 4). Default: effective `k`. Raising this value
    /// improves recall diversity for low-coverage namespaces at the cost of
    /// extra sub-query work.
    pub(super) per_namespace_top_k: Option<usize>,
    /// When `true`, sub-query errors emit `tracing::warn!` and the failing
    /// namespace is skipped rather than propagating `Err` to the caller
    /// (ADR-029c Decision 4 sub-decision M1). Default `false` (fail-all).
    pub(super) best_effort: bool,
    /// Stable id for correlating tracing spans across multi-namespace fan-out.
    /// Auto-generated at `RecallRequest` construction; override via
    /// `with_recall_id`. (ADR-029c Decision 5).
    pub(super) recall_id: Uuid,
    pub(super) k: Option<usize>,
    pub(super) as_of: Option<DateTime<Utc>>,
    /// TD-062 (spec §3 Increment 3): rerank the top-`n` post-fusion
    /// candidates via a cross-encoder. `None` (default) = no rerank.
    pub(super) rerank_k: Option<usize>,
    pub(super) template: Option<RecallTemplate>,
    pub(super) raw_mode: bool,
    pub(super) opts: Option<SearchOpts>,
    /// G6 — single-value AND filters: `metadata[key] == value` (Vera F6).
    /// Multiple calls AND together. Keys validated via [`validate_metadata_key`].
    pub(super) metadata_filters: Vec<(String, serde_json::Value)>,
    /// G6.b — multi-value OR-within-key filters: `metadata[key] IN values`
    /// (Vera F6). Multiple calls AND across keys. Empty `values` slice causes
    /// .await to return `Err` via [`Self::pending_error`].
    pub(super) metadata_filters_in: Vec<(String, Vec<serde_json::Value>)>,
    /// G6 — deferred-error slot. Set by validating builders (filter_metadata,
    /// filter_metadata_in). Surfaces at `.await` time via `Err`. Matches the
    /// `ConflictingNamespaceSelectors` pattern at facade/mod.rs:1373.
    pub(super) pending_error: Option<MemoryError>,
}

/// G6 — Apply metadata filters as a post-filter to recall results.
///
/// Looks up each result's episode metadata via the
/// `entities → episodic_edges → episodes` JOIN (entity-scoped within the
/// recall namespace), then applies AND-across-filters / OR-within-`filter_in`
/// semantics. An entity is retained iff AT LEAST ONE of its episodes within
/// the namespace satisfies all filters.
///
/// # Why post-filter at v0.1.6
///
/// The substrate-level `GraphHandle::graph_search` predates metadata filters;
/// promoting them to a SQL pre-filter via `json_extract(...)` requires
/// extending the substrate trait + an index opportunity is scheduled for
/// v0.1.7. Post-filter is correct + traceable here.
/// Bundled parameters for [`apply_metadata_post_filter`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
struct MetadataPostFilterParams<'a> {
    memory: &'a Memory,
    namespace: &'a Namespace,
    results: Vec<RetrievedContext>,
    metadata_filters: &'a [(String, serde_json::Value)],
    metadata_filters_in: &'a [(String, Vec<serde_json::Value>)],
}

async fn apply_metadata_post_filter(
    params: MetadataPostFilterParams<'_>,
) -> Result<Vec<RetrievedContext>> {
    let MetadataPostFilterParams {
        memory,
        namespace,
        results,
        metadata_filters,
        metadata_filters_in,
    } = params;
    if metadata_filters.is_empty() && metadata_filters_in.is_empty() {
        return Ok(results);
    }
    let tg = memory.temporal_graph.as_ref().ok_or_else(|| {
        MemoryError::Other(
            "filter_metadata post-filter requires a Memory constructed via the \
             builder/providers path (no Arc<TemporalGraph> attached)"
                .to_string(),
        )
    })?;
    let conn = &tg.conn;
    let mut filtered: Vec<RetrievedContext> = Vec::with_capacity(results.len());
    for r in results {
        let mut rows = conn
            .query(
                "SELECT DISTINCT e.metadata FROM episodes e \
                 JOIN episodic_edges ee ON ee.episode_id = e.id \
                 WHERE ee.entity_id = ?1 AND e.group_id = ?2",
                libsql::params![r.entity_id.clone(), namespace.namespace.clone()],
            )
            .await
            .map_err(CoreError::Database)?;
        let mut any_pass = false;
        while let Some(row) = rows.next().await.map_err(CoreError::Database)? {
            let meta_text = row.get::<Option<String>>(0).map_err(CoreError::Database)?;
            let meta: Option<serde_json::Value> = match meta_text {
                Some(t) => match serde_json::from_str(&t) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        // Quinn C2 — surface parse failures via tracing rather
                        // than silently dropping the episode. Schema corruption
                        // or encoding bugs would otherwise be invisible.
                        tracing::warn!(
                            entity_id = %r.entity_id,
                            namespace = %namespace.namespace,
                            error = %e,
                            "filter_metadata post-filter: episode.metadata JSON parse failed, treating as absent"
                        );
                        None
                    }
                },
                None => None,
            };
            if metadata_matches(meta.as_ref(), metadata_filters, metadata_filters_in) {
                any_pass = true;
                break;
            }
        }
        if any_pass {
            filtered.push(r);
        }
    }
    Ok(filtered)
}

/// AND-across-filters / OR-within-`filter_in`. A metadata object that lacks
/// any required key fails the filter (missing-key counts as no-match).
fn metadata_matches(
    meta: Option<&serde_json::Value>,
    filters: &[(String, serde_json::Value)],
    filters_in: &[(String, Vec<serde_json::Value>)],
) -> bool {
    let Some(m) = meta else {
        return false;
    };
    for (k, v) in filters {
        if m.get(k) != Some(v) {
            return false;
        }
    }
    for (k, vs) in filters_in {
        let Some(actual) = m.get(k) else {
            return false;
        };
        if !vs.iter().any(|v| actual == v) {
            return false;
        }
    }
    true
}

/// G6 — Vera F6 validation for metadata filter keys. Rejects path-injection
/// metachars + length + empty + digit-prefix. Returns `Err` describing the
/// reject reason (caller stores in `pending_error` for deferred surface).
fn validate_metadata_key(key: &str) -> Result<()> {
    if key.is_empty() {
        return Err(MemoryError::Other(
            "filter_metadata: key must not be empty".to_string(),
        ));
    }
    if key.len() > 128 {
        return Err(MemoryError::Other(format!(
            "filter_metadata: key length must be <= 128 (got {})",
            key.len()
        )));
    }
    if let Some(first) = key.chars().next() {
        if first.is_ascii_digit() {
            return Err(MemoryError::Other(format!(
                "filter_metadata: key must not start with a digit (got {key:?})"
            )));
        }
    }
    // Vera F6 path-injection guard. Bracket / quote / dollar / star / backslash
    // / dot are all interpreted by SQLite's json_extract path grammar; rejecting
    // here prevents consumer-controlled keys from escaping `'$.{key}'`.
    for ch in key.chars() {
        // Quinn C3: `{` and `}` added for defence-in-depth. They have no valid
        // top-level metadata key use, and could enable template-string escape
        // in any future code path that wraps the path in `{...}`.
        if matches!(
            ch,
            '.' | '[' | ']' | '\'' | '"' | '\\' | '$' | '*' | '{' | '}'
        ) {
            return Err(MemoryError::Other(format!(
                "filter_metadata: key must not contain JSON-path metachars \
                 (. [ ] ' \" \\ $ * {{ }}); got {key:?}"
            )));
        }
    }
    Ok(())
}

// ── TD-066 Increment 1: content-fusion parity fix ────────────────────────────
//
// `.ai-docs/specs/td-066-recall-scoring-foundation-spec-2026-07-21.md` §3.
// Wires the `core::search::rrf_fuse_with_content` fusion fn (ported from the
// REST layer's proven `kremory-mcp/src/bin/kremory-http.rs::hybrid_mode_results`)
// into the canonical single-namespace recall terminals so `Memory::recall()`
// and `.raw()` (and, transitively, the MCP `kremory_recall` tool, which calls
// `.raw()` for its `Structured` format — `kremory-mcp/src/handlers.rs::do_recall`)
// reach the same fused surface the REST `/search?mode=hybrid` endpoint
// already proves.

/// Bundled parameters for [`fuse_content_stream`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
#[cfg(feature = "content-search")]
struct FuseContentStreamParams<'a> {
    memory: &'a Memory,
    namespace: &'a Namespace,
    query: &'a str,
    limit: Option<usize>,
    entity_results: Vec<RetrievedContext>,
}

/// RRF-fuses `entity_results` (the entity/fact recall stream, already
/// score-sorted by the caller) with the ADR-072 `content_search` BM25 stream
/// for the SAME namespace + query. Returns `entity_results` unchanged (no
/// error) when `Memory` has no attached `TemporalGraph` — the stub-graph
/// test-construction path (`.content()` errors loudly on this absence, but a
/// DEFAULT-behaviour change like this one must degrade silently rather than
/// break every caller that bypasses the builder path; `temporal_graph` is
/// `None` ONLY for that bypass per its own doc comment, `facade/mod.rs`).
#[cfg(feature = "content-search")]
async fn fuse_content_stream(params: FuseContentStreamParams<'_>) -> Result<Vec<RetrievedContext>> {
    let FuseContentStreamParams {
        memory,
        namespace,
        query,
        limit,
        entity_results,
    } = params;

    let Some(tg) = memory.temporal_graph.as_ref() else {
        return Ok(entity_results);
    };

    let group_id = namespace_to_group_id(namespace);
    let filters = crate::core::search::SearchFilters {
        group_ids: vec![group_id],
        ..Default::default()
    };
    // Mirrors `.content()`'s own limit convention (`inner.k.unwrap_or(10)`) —
    // parity with the REST layer's `content_mode_results`, which reaches
    // this same default through `RecallParams.k`.
    let content_limit = limit.unwrap_or(10);

    let fusion_start = std::time::Instant::now();
    let content_results = tg
        .content_search(crate::core::search::ContentSearchParams {
            query,
            limit: content_limit,
            filters: &filters,
        })
        .await
        .map_err(MemoryError::Core)?;

    // TD-066 Increment 2 (spec §3 Increment 2): `SearchConfig::default()` is
    // the only value available here today — `Memory` holds no live
    // `PipelineConfig`/`SearchConfig` instance reachable through the
    // `Arc<dyn GraphHandle>` trait object (unlike `graph_degree_weight`,
    // which the `Engine` reads from its OWN `self.config.search` inside
    // `contextualize`). Wiring a consumer-settable override through
    // `GraphHandle` is out of scope for this increment (spec Files list:
    // `core/config.rs` + the fusion fn only) — today `SearchConfig::default()`
    // and any hypothetical live instance are identical (no builder setter
    // exists for this field, mirroring `graph_degree_weight`/`temporal_weight`'s
    // own current state), so this is a faithful read of "the configured
    // value", not a hardcoded bypass of it.
    let content_stream_weight = crate::core::config::SearchConfig::default().content_stream_weight;
    let fused =
        crate::core::search::rrf_fuse_with_content(crate::core::search::RrfFuseWithContentParams {
            entity_stream: entity_results,
            content_stream: content_results,
            namespace: Some(namespace),
            limit,
            content_stream_weight,
        });
    // Incremental cost THIS Increment adds (the content_search query + the
    // fusion pass) — not the whole recall (the entity-graph stream was
    // already computed by the caller before this fn runs); still surfaces a
    // regression from adding the content stream, per spec §6.
    let fusion_secs = fusion_start.elapsed().as_secs_f64();
    metrics::histogram!("kremory.recall.canonical_fusion_duration_seconds").record(fusion_secs);
    Ok(fused)
}

/// Feature-off degrade — mirrors `kremory-http.rs`'s
/// `#[cfg(not(feature = "content-search"))]` arms: no BM25 stream exists to
/// fuse, so the entity/fact recall stream passes through unchanged. The
/// default build (`content-search` off) stays byte-identical to
/// pre-Increment-1 behaviour.
#[cfg(not(feature = "content-search"))]
async fn fuse_content_stream(
    entity_results: Vec<RetrievedContext>,
) -> Result<Vec<RetrievedContext>> {
    metrics::counter!("kremory.recall.canonical_fusion_feature_off_total").increment(1);
    Ok(entity_results)
}

// ── TD-066 Increment 3 (TD-062 reranker) ──────────────────────────────────────
//
// `.ai-docs/specs/td-066-recall-scoring-foundation-spec-2026-07-21.md` §3
// Increment 3. Optional FINAL stage after Increment 1/2's fusion
// (`fuse_content_stream`, above) — reranks the top-`rerank_k` fused
// candidates with a cross-encoder for precision. `SearchOpts.rerank_k: None`
// (the default) is a total no-op, with or without the `rerank` feature
// compiled in.

/// Bundled parameters for [`apply_rerank`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct ApplyRerankParams<'a> {
    query: &'a str,
    rerank_k: Option<usize>,
    results: Vec<RetrievedContext>,
}

/// Reranks the top-`rerank_k` of `results` (already fused by Increment 1/2)
/// against `query` via the process-wide [`crate::core::rerank::Reranker`]
/// singleton. `None` short-circuits with zero cost. Candidates beyond
/// `rerank_k` pass through unreranked, appended after the reranked head in
/// their original fused order — this is a PRECISION pass over the top
/// slice, not a re-fusion of the whole result set.
///
/// Fails OPEN: a reranker error (model load failure, inference error) logs
/// via `tracing::warn!` + the `error` outcome counter and returns the
/// original fused order unchanged — reranking is a precision enhancement,
/// never a correctness-critical path (spec Risk register #11's opt-in-only
/// framing extends naturally to "never breaks recall on failure").
#[cfg(feature = "rerank")]
async fn apply_rerank(params: ApplyRerankParams<'_>) -> Result<Vec<RetrievedContext>> {
    let ApplyRerankParams {
        query,
        rerank_k,
        results,
    } = params;
    let reranker = crate::core::rerank::default_reranker();
    apply_rerank_with(ApplyRerankWithParams {
        reranker: reranker.as_ref(),
        query,
        rerank_k,
        results,
    })
    .await
}

/// Bundled parameters for [`apply_rerank_with`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
#[cfg(feature = "rerank")]
struct ApplyRerankWithParams<'a> {
    reranker: &'a dyn crate::core::rerank::Reranker,
    query: &'a str,
    rerank_k: Option<usize>,
    results: Vec<RetrievedContext>,
}

/// The actual reorder logic, parameterised over `&dyn Reranker` so the fast
/// tier can exercise it with a deterministic mock (spec §3 Increment 3 test
/// pyramid: "reranker trait mock proving the wiring reorders correctly") —
/// `apply_rerank` (above) is the thin production wrapper that supplies the
/// real process-wide singleton.
#[cfg(feature = "rerank")]
async fn apply_rerank_with(params: ApplyRerankWithParams<'_>) -> Result<Vec<RetrievedContext>> {
    use std::collections::HashMap;

    let ApplyRerankWithParams {
        reranker,
        query,
        rerank_k,
        results,
    } = params;

    let Some(k) = rerank_k else {
        return Ok(results);
    };
    metrics::histogram!("kremory.rerank.rerank_k_requested").record(k as f64);

    if results.len() <= 1 {
        // Nothing meaningful to reorder — 0 or 1 candidates has only one
        // possible order.
        metrics::counter!("kremory.rerank.invoked_total", "outcome" => "skipped_below_rerank_k")
            .increment(1);
        return Ok(results);
    }

    let take = k.min(results.len());
    let mut head = results;
    let tail = if take < head.len() {
        head.split_off(take)
    } else {
        Vec::new()
    };

    // Candidate text: `entity_name` + `summary` — for content-fused entries
    // (Increment 1) `summary` already carries the full episode text
    // (`content_passage_into_retrieved_context`); for entity-only entries
    // it's the recall-time entity summary. Good-enough v1 candidate text;
    // not spec-mandated to be more elaborate.
    let candidates: Vec<(String, String)> = head
        .iter()
        .map(|ctx| {
            (
                ctx.entity_id.clone(),
                format!("{} {}", ctx.entity_name, ctx.summary),
            )
        })
        .collect();
    let original_rank: HashMap<String, usize> = head
        .iter()
        .enumerate()
        .map(|(i, ctx)| (ctx.entity_id.clone(), i))
        .collect();

    match reranker.rerank(query, &candidates).await {
        Ok(scored) => {
            let mut by_id: HashMap<String, RetrievedContext> = head
                .into_iter()
                .map(|ctx| (ctx.entity_id.clone(), ctx))
                .collect();
            let mut reranked_head = Vec::with_capacity(scored.len());
            let mut rank_deltas: Vec<f64> = Vec::with_capacity(scored.len());
            let mut any_reordered = false;
            for (new_rank, (id, score)) in scored.into_iter().enumerate() {
                let Some(mut ctx) = by_id.remove(&id) else {
                    // Reranker returned an id it wasn't handed — defensive
                    // branch, skip rather than fabricate a result.
                    continue;
                };
                ctx.score = score;
                if let Some(&old_rank) = original_rank.get(&id) {
                    rank_deltas.push((new_rank as i64 - old_rank as i64).unsigned_abs() as f64);
                    if new_rank != old_rank {
                        any_reordered = true;
                    }
                }
                reranked_head.push(ctx);
            }
            // Any candidate the reranker silently omitted (not expected per
            // fastembed's own contract — one result per input document —
            // but defends against a future/custom Reranker impl that
            // filters) still survives, appended after in original order.
            reranked_head.extend(by_id.into_values());

            if !rank_deltas.is_empty() {
                let mean_abs_delta = rank_deltas.iter().sum::<f64>() / rank_deltas.len() as f64;
                metrics::histogram!("kremory.rerank.score_delta").record(mean_abs_delta);
            }
            metrics::counter!(
                "kremory.rerank.invoked_total",
                "outcome" => if any_reordered { "reordered" } else { "no_reorder" },
            )
            .increment(1);

            reranked_head.extend(tail);
            Ok(reranked_head)
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "kremory.rerank failed; falling back to un-reranked fusion order"
            );
            metrics::counter!("kremory.rerank.invoked_total", "outcome" => "error").increment(1);
            let mut fallback = head;
            fallback.extend(tail);
            Ok(fallback)
        }
    }
}

/// Feature-off degrade: `rerank` Cargo feature not compiled in. A consumer-
/// requested `rerank_k` becomes a no-op (rather than silently succeeding
/// with zero observability) — `kremory.rerank.feature_off_total` only fires
/// when a caller actually asked for reranking a binary without the feature
/// can't perform (Rule 19: never silently no-op a consumer-requested knob).
#[cfg(not(feature = "rerank"))]
async fn apply_rerank(params: ApplyRerankParams<'_>) -> Result<Vec<RetrievedContext>> {
    let ApplyRerankParams {
        query,
        rerank_k,
        results,
    } = params;
    if rerank_k.is_some() {
        metrics::counter!("kremory.rerank.feature_off_total").increment(1);
        // `query` genuinely used here (not just discarded) — gives a caller
        // wondering why `rerank_k` had no effect the query context to
        // correlate against, rather than an unused-field `#[allow(dead_code)]`
        // band-aid on a field the `rerank`-on build DOES read.
        tracing::debug!(
            query,
            "kremory.rerank requested via rerank_k but the 'rerank' Cargo \
             feature is not compiled in — no-op"
        );
    }
    Ok(results)
}

/// SQLite parameter cap (libsql ≥ 3.32.0). G6.b values must stay under this
/// to avoid `SQLITE_TOOBIG` at the SQL pre-filter promotion in v0.1.7.
const SQLITE_MAX_VARIABLE_NUMBER: usize = 32766;

impl<'a> RecallRequest<'a> {
    /// Set the namespace for this operation (overrides Memory default).
    /// Mutually exclusive with `in_namespaces` — setting both returns
    /// `Err(Error::ConflictingNamespaceSelectors)` at `.await` time.
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Recall across multiple namespaces concurrently. Each namespace runs
    /// its own local RRF-fused search independently; the per-namespace
    /// result sets are then concatenated and sorted by score (not re-fused
    /// via a second cross-namespace RRF pass). Each result in the returned
    /// `Vec` carries `namespace: Some(ns)` identifying its source namespace.
    ///
    /// # Empty slice
    ///
    /// `in_namespaces(&[])` returns `Err(MemoryError::MissingNamespace { ... })`.
    ///
    /// # Single-element equivalence
    ///
    /// `in_namespaces(&[ns])` is equivalent to `in_namespace(ns)` — same query
    /// plan, same result semantics, same `namespace: Some(ns)` attribution.
    ///
    /// # Mutual exclusion with `in_namespace`
    ///
    /// Calling both `in_namespace` and `in_namespaces` on the same request is a
    /// programming error and returns `Err(MemoryError::Core(Error::ConflictingNamespaceSelectors))`
    /// at `.await` time (checked by `check_selectors`).
    ///
    /// # Cold-cache startup latency
    ///
    /// Each namespace may trigger a DB policy lookup on first call if the
    /// `NamespacePolicyCache` is cold. For N > 4 namespaces at server startup,
    /// consider pre-warming via `register_namespace` on each namespace before
    /// the first `in_namespaces` call to avoid the cold-cache thundering-herd
    /// (see ADR-029c Decision 4 for details).
    pub fn in_namespaces(mut self, namespaces: &[Namespace]) -> Self {
        self.namespaces = Some(namespaces.to_vec());
        self
    }

    /// Cap the number of results fetched from each individual namespace before
    /// the cross-namespace score-sort merge. Default: `k` (or `Memory::default_k`
    /// if `k` is unset). Raising this value improves recall diversity for
    /// low-coverage namespaces at the cost of extra per-namespace sub-query work.
    pub fn per_namespace_top_k(mut self, n: usize) -> Self {
        self.per_namespace_top_k = Some(n);
        self
    }

    /// When `true`, sub-query errors emit `tracing::warn!` and the failing
    /// namespace is skipped rather than returning `Err` to the caller. The
    /// returned `Vec<RetrievedContext>` contains results from all namespaces
    /// that succeeded. Default: `false` (fail-all — appropriate for audit
    /// consumers where partial results are worse than no results).
    ///
    /// If ALL sub-queries fail, `best_effort(true)` still returns `Err`
    /// (returning an empty result set silently is worse than surfacing the error).
    pub fn best_effort(mut self, enabled: bool) -> Self {
        self.best_effort = enabled;
        self
    }

    /// Override the auto-generated `recall_id`. Use when correlating kremory
    /// tracing spans with an application-level request id.
    pub fn with_recall_id(mut self, id: Uuid) -> Self {
        self.recall_id = id;
        self
    }

    /// Top-k results to return (clamped per Story #166).
    pub fn k(mut self, n: usize) -> Self {
        self.k = Some(n);
        self
    }

    /// Point-in-time (valid-time) filter (ADR-068) — "what was TRUE in the
    /// world at `ts`," not "what did kremory KNOW at `ts`." Filters which
    /// FACTS the 1-hop expansion surfaces (`TemporalGraph::get_neighbours_at`);
    /// entity search itself is unaffected (entities carry no temporal
    /// columns, so the same query still finds the same seed entities
    /// regardless of `as_of`). A fact later flagged `invalid_at` by the
    /// reconciliation resolver still appears for `as_of(ts)` if `ts` falls
    /// inside its `[valid_from, valid_to)` window — proving what the record
    /// showed as true at `ts`, even after correction, is the point of
    /// bi-temporal audit. `None` (the default) returns present-day results,
    /// unaffected.
    pub fn as_of(mut self, ts: DateTime<Utc>) -> Self {
        self.as_of = Some(ts);
        self
    }

    /// TD-062 (spec §3 Increment 3): rerank the top-`n` post-fusion
    /// candidates with a cross-encoder before returning. A no-op unless the
    /// `rerank` Cargo feature is compiled in (see `apply_rerank`'s two
    /// cfg-gated bodies).
    pub fn rerank_k(mut self, n: usize) -> Self {
        self.rerank_k = Some(n);
        self
    }

    /// Return results as a prompt-ready string using `TemporalFacts` template.
    /// Sets the terminal return type to `String`.
    pub fn as_prompt_text(mut self) -> Self {
        self.template = Some(RecallTemplate::TemporalFacts);
        self.raw_mode = false;
        self
    }

    /// Return results as a string using a specific template.
    pub fn as_template(mut self, t: RecallTemplate) -> Self {
        self.template = Some(t);
        self.raw_mode = false;
        self
    }

    /// Return raw `Vec<RetrievedContext>` — no template rendering.
    pub fn raw(mut self) -> RecallRawRequest<'a> {
        self.raw_mode = true;
        RecallRawRequest { inner: self }
    }

    /// Return BM25-ranked content passages over raw `episodes.content`
    /// (ADR-072 seq1) — sibling of `.raw()`. Does **NOT** extend the
    /// entity-shaped `RetrievedContext` contract; `ContentPassage` is a
    /// distinct projection (ADR-072 §6b) returned as a separate, unfused
    /// BM25-only stream (kremory's RRF is pairwise per-type, not a generic
    /// N-list fuser — `core::search::rrf_fuse_entities`/`rrf_fuse_facts`).
    ///
    /// Requires a `Memory` built via the builder/providers path (needs
    /// `Arc<TemporalGraph>` — same requirement as `.forget()` / the
    /// `filter_metadata` post-filter). Multi-namespace fan-out
    /// (`.in_namespaces()`) is not yet supported for this terminal — use
    /// `.in_namespace()`.
    #[cfg(feature = "content-search")]
    pub fn content(self) -> RecallContentRequest<'a> {
        RecallContentRequest { inner: self }
    }

    /// Escape hatch: set raw `SearchOpts` directly.
    pub fn opts(mut self, opts: SearchOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    /// G6 — single-value equality filter on a top-level episode metadata key.
    /// Multiple calls AND together. Fluent — invalid keys surface as `Err`
    /// at `.await` time via the deferred-error pattern (matches
    /// `ConflictingNamespaceSelectors` at facade/mod.rs:1373).
    ///
    /// # Key validation (Vera F6 — JSON-path injection)
    ///
    /// `key` MUST be a top-level metadata field name only. Rejected at
    /// `.await` time when the key:
    /// - Contains any of `. [ ] ' " \ $ *` (path-traversal / escape chars)
    /// - Length > 128
    /// - Empty
    /// - Starts with an ASCII digit
    ///
    /// The key is later interpolated into SQLite's `json_extract` path
    /// (`'$.{key}'`); SQLite has no parameterized path equivalent so this
    /// validation is the only guard against path injection from
    /// consumer-controlled keys. Values are ALWAYS bound via libsql `?`.
    ///
    /// # v0.1.6 perf characteristics
    ///
    /// At v0.1.6 the filter is applied as a post-filter at facade level
    /// (after `memory::search` returns). Index promotion to a SQL pre-filter
    /// is scheduled for v0.1.7 (per kremory-v016-api-gaps spec G6 note
    /// "Index opportunity deferred to v0.1.7").
    pub fn filter_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        if let Err(e) = validate_metadata_key(key) {
            // Keep the first error; subsequent invalid keys won't overwrite.
            if self.pending_error.is_none() {
                self.pending_error = Some(e);
            }
            return self;
        }
        self.metadata_filters.push((key.to_string(), value));
        self
    }

    /// G6.b — multi-value OR-within-key equality filter:
    /// `metadata[key] ∈ values`. Same `key` validation as
    /// [`Self::filter_metadata`]. Multiple `filter_metadata_in` calls
    /// combine ACROSS keys with AND; values WITHIN a single call combine
    /// with OR.
    ///
    /// # Empty values slice
    ///
    /// Calling `filter_metadata_in("k", &[])` would never match anything
    /// and is almost always a caller bug — rejected at `.await` time as
    /// `Err(MemoryError::Other(...))`.
    ///
    /// # Limits
    ///
    /// libsql ≥ 3.32.0 enforces `SQLITE_MAX_VARIABLE_NUMBER = 32766` total
    /// bound parameters across a single statement (Vera cycle-2 RISK-004).
    /// Caller-side cap: keep `values.len()` under a few thousand per call
    /// to leave room for the query's other parameters.
    pub fn filter_metadata_in(mut self, key: &str, values: &[serde_json::Value]) -> Self {
        if let Err(e) = validate_metadata_key(key) {
            if self.pending_error.is_none() {
                self.pending_error = Some(e);
            }
            return self;
        }
        if values.is_empty() {
            if self.pending_error.is_none() {
                self.pending_error = Some(MemoryError::Other(format!(
                    "filter_metadata_in: values slice must not be empty (key {key:?})"
                )));
            }
            return self;
        }
        // Quinn C7 / Vera RISK-004 — cap values per call to leave room for
        // other bound params in the v0.1.7 SQL pre-filter promotion. The cap
        // already lands at v0.1.6 (post-filter) so behaviour stays stable when
        // the SQL plumbing arrives.
        if values.len() > SQLITE_MAX_VARIABLE_NUMBER {
            if self.pending_error.is_none() {
                self.pending_error = Some(MemoryError::Other(format!(
                    "filter_metadata_in: values length {} exceeds SQLite parameter cap {} (key {key:?})",
                    values.len(),
                    SQLITE_MAX_VARIABLE_NUMBER
                )));
            }
            return self;
        }
        self.metadata_filters_in
            .push((key.to_string(), values.to_vec()));
        self
    }

    /// Check that `in_namespace` and `in_namespaces` were not both set on this
    /// request. Called from both `execute()` and `RecallRawRequest::into_future`
    /// (ADR-029c Decision 6, closes M3).
    fn check_selectors(&self) -> Result<()> {
        if self.namespace.is_some() && self.namespaces.is_some() {
            return Err(MemoryError::Core(
                crate::core::error::Error::ConflictingNamespaceSelectors {
                    request: "in_namespace and in_namespaces both set on the same RecallRequest; \
                              use one or the other"
                        .to_string(),
                },
            ));
        }
        Ok(())
    }

    async fn execute(mut self) -> Result<String> {
        self.check_selectors()?;
        if let Some(err) = self.pending_error.take() {
            return Err(err);
        }

        let template = self.template.unwrap_or(RecallTemplate::TemporalFacts);

        // Multi-namespace fan-out path (ADR-029c Decision 4 + 6).
        if self.namespaces.is_some() {
            if self.namespaces.as_ref().is_none_or(|v| v.is_empty()) {
                return Err(MemoryError::MissingNamespace {
                    request:
                        "in_namespaces called with empty slice; provide at least one namespace",
                });
            }
            let results = self.execute_multi_namespace().await?;
            return Ok(memory::context_block(&results, template.into()));
        }

        // Single-namespace path (original behaviour).
        let ns = self.memory.resolve_namespace(self.namespace.clone())?;
        let opts = self.opts.clone().unwrap_or(SearchOpts {
            limit: self.k,
            as_of: self.as_of,
            source_kind: None,
            rerank_k: self.rerank_k,
        });
        // TD-066 Increment 3: captured before `opts` moves into `memory::search`
        // below — `apply_rerank` runs as the LAST pipeline stage, after fusion.
        let rerank_k = opts.rerank_k;
        // ADR-029a lazy population.
        self.memory.ensure_namespace_policy(&ns).await?;
        let recall_id = self.recall_id;
        let span = tracing::info_span!(
            "kremory.recall.single_ns",
            recall_id = %recall_id,
            namespace = %ns.namespace,
        );
        let _enter = span.enter();
        let mut results = memory::search(memory::SearchParams {
            graph: self.memory.graph.as_ref(),
            query: &self.query,
            namespace: ns.clone(),
            opts,
        })
        .await?
        .into_iter()
        .map(|r| r.with_namespace(ns.clone()))
        .collect::<Vec<_>>();
        // G6 — apply metadata post-filter before template rendering.
        results = apply_metadata_post_filter(MetadataPostFilterParams {
            memory: self.memory,
            namespace: &ns,
            results,
            metadata_filters: &self.metadata_filters,
            metadata_filters_in: &self.metadata_filters_in,
        })
        .await?;
        // Phase 0 (recall-v2-architecture-2026-07-03 Decision 8): sort by score
        // DESC before render. `context_block` renders in input order, and
        // `contextualize` hands entities back in `HashSet`-insert order (seeds in
        // score order, then 1-hop neighbours interleaved out of order) — NOT
        // score order. Without this sort EVERY post-RRF scoring signal (the
        // RRF-normalised seed score, TD-066's neighbour-decay + graph-degree
        // bonus) was dormant at the output: the boost mutated `scores` but never
        // re-ordered what the LLM consuming `context_block` actually saw. The
        // multi-namespace path already sorted (below); this closes the
        // single-namespace gap so downstream axes are measurable.
        sort_by_score_desc(&mut results);
        // TD-066 Increment 1 (content-fusion parity fix, spec §3) — fuse in
        // the ADR-072 `content_search` BM25 stream so `mem.recall(q).await`
        // reaches the same surface `.raw()` does (below) and the REST
        // `/search?mode=hybrid` endpoint already proves. Feature-gated;
        // feature-off passes `results` through unchanged (see
        // `fuse_content_stream`'s two cfg-gated bodies).
        #[cfg(feature = "content-search")]
        let results = fuse_content_stream(FuseContentStreamParams {
            memory: self.memory,
            namespace: &ns,
            query: &self.query,
            limit: self.k,
            entity_results: results,
        })
        .await?;
        #[cfg(not(feature = "content-search"))]
        let results = fuse_content_stream(results).await?;
        // TD-066 Increment 3 (TD-062 reranker, spec §3) — optional final
        // stage, after fusion. `rerank_k: None` (default) is a no-op.
        let results = apply_rerank(ApplyRerankParams {
            query: &self.query,
            rerank_k,
            results,
        })
        .await?;
        Ok(memory::context_block(&results, template.into()))
    }

    /// Fan-out recall across all namespaces in `self.namespaces` (each namespace
    /// internally RRF-fused), merge the per-namespace results by concatenation +
    /// score-descending sort, trim to `self.k`, and return results with
    /// `namespace: Some(ns)` attribution.
    ///
    /// Precondition: `self.namespaces` is `Some` and non-empty (caller checks).
    async fn execute_multi_namespace(self) -> Result<Vec<RetrievedContext>> {
        // Extract all fields upfront before consuming `self`.
        let memory = self.memory;
        let query = self.query;
        let namespaces = self
            .namespaces
            .ok_or_else(|| MemoryError::MissingNamespace {
                request:
                    "execute_multi_namespace called without namespaces (internal precondition)",
            })?;
        let opts_template = self.opts;
        let per_ns_k = self.per_namespace_top_k.or(self.k);
        let final_k = self.k;
        let as_of = self.as_of;
        let best_effort = self.best_effort;
        let recall_id = self.recall_id;
        // G6 — Quinn C1 fix: filters MUST apply per-namespace inside the sub-
        // future, otherwise multi-namespace recall silently drops them. Wrap in
        // Arc so the closures (one per namespace) can share without cloning the
        // Vecs N times.
        let metadata_filters = std::sync::Arc::new(self.metadata_filters);
        let metadata_filters_in = std::sync::Arc::new(self.metadata_filters_in);

        let outer_span = tracing::info_span!(
            "kremory.recall.multi_ns",
            recall_id = %recall_id,
            namespace_count = %namespaces.len(),
        );
        let _outer = outer_span.enter();

        // Collect sub-query futures — one per namespace.
        let mut sub_futures = Vec::with_capacity(namespaces.len());
        for ns in namespaces {
            let query = query.clone();
            // TD-066 Increment 3: multi-namespace fan-out does not wire
            // `apply_rerank` (each sub-query is independent; a cross-
            // namespace top-k rerank is a follow-up, not this increment's
            // scope) — `rerank_k` still threads through if the caller set it
            // via `opts_template`, it's simply unused by this path today.
            let opts = opts_template.clone().unwrap_or(SearchOpts {
                limit: per_ns_k,
                as_of,
                source_kind: None,
                rerank_k: None,
            });
            let metadata_filters = std::sync::Arc::clone(&metadata_filters);
            let metadata_filters_in = std::sync::Arc::clone(&metadata_filters_in);
            sub_futures.push(async move {
                let span = tracing::info_span!(
                    "kremory.recall.sub_query",
                    recall_id = %recall_id,
                    namespace = %ns.namespace,
                    per_namespace_top_k = per_ns_k,
                );
                let _enter = span.enter();
                memory.ensure_namespace_policy(&ns).await?;
                let hits = memory::search(memory::SearchParams {
                    graph: memory.graph.as_ref(),
                    query: &query,
                    namespace: ns.clone(),
                    opts,
                })
                .await?;
                let attributed: Vec<RetrievedContext> = hits
                    .into_iter()
                    .map(|r| r.with_namespace(ns.clone()))
                    .collect();
                // G6 — Quinn C1: filters must apply per-namespace (entity_id
                // scoping requires the namespace's own conn lookup).
                let filtered = apply_metadata_post_filter(MetadataPostFilterParams {
                    memory,
                    namespace: &ns,
                    results: attributed,
                    metadata_filters: &metadata_filters,
                    metadata_filters_in: &metadata_filters_in,
                })
                .await?;
                Ok::<Vec<RetrievedContext>, MemoryError>(filtered)
            });
        }

        let sub_results = futures::future::join_all(sub_futures).await;

        // Collect results, honouring best_effort semantics.
        let mut all_results: Vec<RetrievedContext> = Vec::new();
        let mut last_err: Option<MemoryError> = None;
        for outcome in sub_results {
            match outcome {
                Ok(hits) => all_results.extend(hits),
                Err(e) => {
                    if best_effort {
                        tracing::warn!(
                            target: "kremory.recall",
                            recall_id = %recall_id,
                            error = %e,
                            "best_effort: namespace sub-query failed, skipping"
                        );
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        // If best_effort and ALL sub-queries failed, surface the last error.
        if best_effort && all_results.is_empty() {
            if let Some(e) = last_err {
                return Err(e);
            }
        }

        // Merge the per-namespace result sets: sort the concatenated list by
        // score descending (each score is the per-namespace RRF score from its
        // own sub-query — this is a score-sort merge, NOT a second cross-
        // namespace RRF pass over the combined list). Shares the single
        // `sort_by_score_desc` comparator with the single-namespace path so the
        // two never drift (recall-v2-architecture-2026-07-03 Decision 8 / R2).
        sort_by_score_desc(&mut all_results);

        // Trim to final_k if set.
        if let Some(k) = final_k {
            all_results.truncate(k);
        }

        Ok(all_results)
    }
}

/// Sort recall results by score DESCENDING (highest first) — the order
/// [`memory::context_block`] renders in (its doc contract: *"Order is preserved
/// from the input — callers are expected to pass results already sorted by
/// score."*).
///
/// Extracted as ONE free fn used by BOTH the single-namespace ([`RecallRequest::
/// execute`]) and multi-namespace ([`RecallRequest::execute_multi_namespace`])
/// paths so the two comparators never drift (recall-v2-architecture-2026-07-03
/// Decision 8 / risk R2). `sort_by` is stable, so score ties preserve the input
/// order (for single-namespace that's `contextualize`'s already-score-ordered
/// seed sequence; for multi-namespace it's the per-namespace concat order).
fn sort_by_score_desc(results: &mut [RetrievedContext]) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

impl<'a> IntoFuture for RecallRequest<'a> {
    type Output = Result<String>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Raw recall variant — returns `Vec<RetrievedContext>` without template rendering.
pub struct RecallRawRequest<'a> {
    inner: RecallRequest<'a>,
}

impl<'a> IntoFuture for RecallRawRequest<'a> {
    type Output = Result<Vec<RetrievedContext>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        let mut inner = self.inner;
        Box::pin(async move {
            // ADR-029c Decision 6 / M3: check_selectors fires on the raw path too.
            inner.check_selectors()?;
            // G6 — surface deferred validation error from filter_metadata /
            // filter_metadata_in setters.
            if let Some(err) = inner.pending_error.take() {
                return Err(err);
            }

            // Multi-namespace fan-out path.
            if inner.namespaces.is_some() {
                if inner.namespaces.as_ref().is_none_or(|v| v.is_empty()) {
                    return Err(MemoryError::MissingNamespace {
                        request:
                            "in_namespaces called with empty slice; provide at least one namespace",
                    });
                }
                return inner.execute_multi_namespace().await;
            }

            let ns = inner.memory.resolve_namespace(inner.namespace.clone())?;
            let rerank_k_from_field = inner.rerank_k;
            let opts = inner.opts.unwrap_or(SearchOpts {
                limit: inner.k,
                as_of: inner.as_of,
                source_kind: None,
                rerank_k: rerank_k_from_field,
            });
            // TD-066 Increment 3: captured before `opts` moves into
            // `memory::search` below.
            let rerank_k = opts.rerank_k;
            // ADR-029a lazy population.
            inner.memory.ensure_namespace_policy(&ns).await?;
            let recall_id = inner.recall_id;
            let span = tracing::info_span!(
                "kremory.recall.single_ns",
                recall_id = %recall_id,
                namespace = %ns.namespace,
            );
            let _enter = span.enter();
            let results: Vec<RetrievedContext> = memory::search(memory::SearchParams {
                graph: inner.memory.graph.as_ref(),
                query: &inner.query,
                namespace: ns.clone(),
                opts,
            })
            .await?
            .into_iter()
            .map(|r| r.with_namespace(ns.clone()))
            .collect();
            // G6 — apply metadata post-filter before returning.
            let mut filtered = apply_metadata_post_filter(MetadataPostFilterParams {
                memory: inner.memory,
                namespace: &ns,
                results,
                metadata_filters: &inner.metadata_filters,
                metadata_filters_in: &inner.metadata_filters_in,
            })
            .await?;
            // Phase 0 (recall-v2-architecture-2026-07-03 Decision 8): sort by
            // score DESC. `raw()` is the terminal the HTTP `/search` benchmark
            // path uses (`do_recall` → `req.raw()`), and its consumers rank
            // by result order — but `memory::search` returns entities in
            // `contextualize`'s `HashSet`-insert order (seeds score-ordered,
            // 1-hop neighbours interleaved out of order), NOT score order. The
            // sibling `execute()` (String) terminal sorts identically; both
            // single-namespace terminals must, so every post-RRF scoring signal
            // (RRF-normalised score, TD-066 neighbour-decay + graph-degree bonus)
            // actually re-orders output rather than staying dormant.
            sort_by_score_desc(&mut filtered);
            // TD-066 Increment 1 (content-fusion parity fix, spec §3) — fuse
            // in the ADR-072 `content_search` BM25 stream so `.raw()` (and,
            // transitively, `kremory_recall`'s `Structured` format, which
            // calls `.raw()` — `kremory-mcp/src/handlers.rs::do_recall`)
            // reaches the same fused surface the REST `/search?mode=hybrid`
            // endpoint already proves. Feature-gated; feature-off passes
            // `filtered` through unchanged.
            #[cfg(feature = "content-search")]
            let filtered = fuse_content_stream(FuseContentStreamParams {
                memory: inner.memory,
                namespace: &ns,
                query: &inner.query,
                limit: inner.k,
                entity_results: filtered,
            })
            .await?;
            #[cfg(not(feature = "content-search"))]
            let filtered = fuse_content_stream(filtered).await?;
            // TD-066 Increment 3 (TD-062 reranker, spec §3) — optional final
            // stage, after fusion. `rerank_k: None` (default) is a no-op.
            // This is the terminal `kremory_recall`'s `Structured` format
            // uses, so wiring here reaches the MCP tool surface too.
            let filtered = apply_rerank(ApplyRerankParams {
                query: &inner.query,
                rerank_k,
                results: filtered,
            })
            .await?;
            Ok(filtered)
        })
    }
}

// ── RecallContentRequest (ADR-072 seq1) ───────────────────────────────────────

/// Content-search recall variant — returns `Vec<ContentPassage>` via
/// BM25-only full-text search over `episodes.content` (ADR-072 seq1). Obtain
/// via `mem.recall(query).content()` — sibling of `.raw()`. Feature-gated
/// behind `content-search`.
#[cfg(feature = "content-search")]
pub struct RecallContentRequest<'a> {
    inner: RecallRequest<'a>,
}

#[cfg(feature = "content-search")]
impl<'a> IntoFuture for RecallContentRequest<'a> {
    type Output = Result<Vec<crate::memory::types::ContentPassage>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        let mut inner = self.inner;
        Box::pin(async move {
            inner.check_selectors()?;
            if let Some(err) = inner.pending_error.take() {
                return Err(err);
            }

            // ADR-072 seq1: content-search is single-namespace-only for now.
            // Multi-namespace fan-out (`.in_namespaces()`) would mirror the
            // entity/fact path's `execute_multi_namespace` merge (concatenate
            // per-namespace results + sort by score) — content passages have
            // no such merge yet (BM25-only, no fusion); wiring fan-out is a
            // later increment, not part of seq1's scope.
            if inner.namespaces.is_some() {
                return Err(MemoryError::Other(
                    "`.content()` recall does not yet support multi-namespace fan-out \
                     (in_namespaces); use `.in_namespace()` for content-search (ADR-072 seq1)."
                        .to_string(),
                ));
            }

            let ns = inner.memory.resolve_namespace(inner.namespace.clone())?;
            // ADR-029a lazy population — same precondition every other recall
            // terminal enforces before touching the namespace's rows.
            inner.memory.ensure_namespace_policy(&ns).await?;

            // ADR-072 seq1 bypasses `GraphHandle::graph_search` entirely (that
            // trait has NO content-search method — adding one would be a
            // required, breaking addition across every implementor per D.6.4
            // "no defaults"). Mirrors the existing `apply_metadata_post_filter`
            // precedent (`facade/recall.rs` above) and `ForgetRequest::execute`
            // (`facade/forget.rs`): both already reach past the trait straight
            // to `Memory::temporal_graph` for substrate-level SQL that the
            // trait's opinionated surface doesn't (yet) cover.
            let tg = inner.memory.temporal_graph.as_ref().ok_or_else(|| {
                MemoryError::Other(
                    "Memory::recall(...).content() requires a Memory constructed via the \
                     builder/providers path (no Arc<TemporalGraph> attached)"
                        .to_string(),
                )
            })?;

            let group_id = namespace_to_group_id(&ns);
            let filters = crate::core::search::SearchFilters {
                group_ids: vec![group_id],
                ..Default::default()
            };
            let limit = inner.k.unwrap_or(10);

            tg.content_search(crate::core::search::ContentSearchParams {
                query: &inner.query,
                limit,
                filters: &filters,
            })
            .await
            .map_err(MemoryError::Core)
        })
    }
}

// ── G5 tests — recall_by_source_id ───────────────────────────────────────────
//
// Spec AC.6 — direct lookup of episodes matching source_id, ordered
// recorded_at DESC, optional namespace scope (Vera F11 fold-in).

#[cfg(test)]
mod recall_by_source_id_tests {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    /// Direct seeding helper; bypasses the facade ingest path so the test
    /// only exercises recall_by_source_id. `recorded_at` is set to control
    /// ORDER BY ordering directly.
    // Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a
    // documented too_many_arguments allow); TD-042 args-as-object targets `src/`
    // production fns, not `#[cfg(test)]` seeders.
    #[allow(clippy::too_many_arguments)]
    async fn seed_episode(
        mem: &Memory,
        source_id: &str,
        ns: &Namespace,
        recorded_at: &str,
        metadata_json: Option<&str>,
    ) {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        conn.execute(
            "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, source_id, source_uri, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            libsql::params![
                "test content",
                "2026-05-29T12:00:00Z",
                "Document",
                metadata_json,
                ns.namespace.as_str(),
                source_id,
                "uri/x",
                recorded_at
            ],
        )
        .await
        .expect("seed insert must succeed");
    }

    /// AC.6 — happy path: returns matching episodes ordered recorded_at DESC.
    #[tokio::test]
    async fn returns_matching_episodes_ordered_recorded_at_desc() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g5-order");

        seed_episode(&mem, "doc-1", &ns, "2026-05-27T10:00:00Z", None).await;
        seed_episode(&mem, "doc-1", &ns, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "doc-1", &ns, "2026-05-28T10:00:00Z", None).await;

        let eps = mem
            .recall_by_source_id("doc-1", Some(ns))
            .await
            .expect("recall must succeed");

        assert_eq!(eps.len(), 3, "must return all 3 episodes for doc-1");
        // recorded_at DESC = newest first.
        assert_eq!(eps[0].recorded_at.as_deref(), Some("2026-05-29T10:00:00Z"));
        assert_eq!(eps[1].recorded_at.as_deref(), Some("2026-05-28T10:00:00Z"));
        assert_eq!(eps[2].recorded_at.as_deref(), Some("2026-05-27T10:00:00Z"));
        // Quinn C6 — proves column-ordinal mapping is live, not a silent zero.
        assert!(
            eps[0].id > 0,
            "Episode.id must be populated (column-ordinal mapping check)"
        );
    }

    /// AC.6 / Quinn C4 — placeholder anchor (do not remove).
    #[allow(dead_code)]
    fn _g5_anchor() {}
}

// ── G6 + G6.b tests — filter_metadata + filter_metadata_in ───────────────────
//
// Spec G6 (key validation + AND-across-filters) + G6.b (OR-within-key +
// multi-value). Covers the Vera F6 path-injection guard and the deferred-
// error pattern surfacing at `.await` time.

#[cfg(test)]
mod filter_metadata_tests {
    use std::sync::Arc;

    use serde_json::json;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::{metadata_matches, sort_by_score_desc, validate_metadata_key, Memory};
    use crate::memory::types::{RetrievedContext, RetrievedContextNewParams};

    /// Build a minimal [`RetrievedContext`] carrying only the `score` the
    /// Phase-0 sort cares about (other fields are irrelevant to ordering).
    fn ctx(id: &str, score: f32) -> RetrievedContext {
        RetrievedContext::new(RetrievedContextNewParams {
            entity_id: id.to_owned(),
            entity_name: id.to_owned(),
            summary: String::new(),
            score,
            source_refs: vec![],
        })
    }

    // ── Phase 0 (recall-v2-architecture-2026-07-03 Decision 8) ───────────────
    //
    // `context_block` renders in input order, so recall output must be
    // score-DESCENDING before render or every post-RRF scoring signal is
    // dormant at the output (the exact "effect-never-reaches-output" gap the
    // spec Decision 8 closes). These gate the extracted comparator both paths
    // now share.

    #[test]
    fn sort_by_score_desc_orders_highest_first() {
        let mut v = vec![ctx("low", 0.2), ctx("high", 0.9), ctx("mid", 0.5)];
        sort_by_score_desc(&mut v);
        let ids: Vec<&str> = v.iter().map(|c| c.entity_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["high", "mid", "low"],
            "results must render score-descending (Phase 0 sort-order fix)"
        );
    }

    #[test]
    fn sort_by_score_desc_is_stable_on_ties() {
        // Stable sort: equal scores preserve input order (single-namespace relies
        // on this to keep `contextualize`'s already-score-ordered seed sequence).
        let mut v = vec![ctx("a", 0.5), ctx("b", 0.5), ctx("c", 0.5)];
        sort_by_score_desc(&mut v);
        let ids: Vec<&str> = v.iter().map(|c| c.entity_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["a", "b", "c"],
            "tied scores must preserve input order"
        );
    }

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    // ── validate_metadata_key — Vera F6 path-injection guard ─────────────────

    #[test]
    fn validate_key_accepts_simple_alphanumeric() {
        assert!(validate_metadata_key("status").is_ok());
        assert!(validate_metadata_key("doc_type").is_ok());
        assert!(validate_metadata_key("key2").is_ok());
        assert!(validate_metadata_key("a").is_ok());
    }

    #[test]
    fn validate_key_rejects_empty() {
        assert!(validate_metadata_key("").is_err());
    }

    #[test]
    fn validate_key_rejects_over_128_chars() {
        let long = "a".repeat(129);
        assert!(validate_metadata_key(&long).is_err());
    }

    #[test]
    fn validate_key_rejects_digit_prefix() {
        assert!(validate_metadata_key("1foo").is_err());
        assert!(validate_metadata_key("0").is_err());
    }

    #[test]
    fn validate_key_rejects_path_metachars() {
        // Vera F6 reject list + Quinn C3 defence-in-depth (`{` and `}`).
        for ch in ['.', '[', ']', '\'', '"', '\\', '$', '*', '{', '}'] {
            let bad = format!("foo{ch}bar");
            assert!(
                validate_metadata_key(&bad).is_err(),
                "key with {ch:?} must be rejected (path-injection guard)"
            );
        }
    }

    // ── metadata_matches — semantics ─────────────────────────────────────────

    #[test]
    fn matches_empty_filters_passes_when_meta_is_some() {
        // No filters at all → caller of helper shouldn't invoke; defensive
        // check: even with Some(meta), empty filter set must short-circuit to
        // true via the for-loops being empty.
        let meta = json!({"k": "v"});
        assert!(metadata_matches(Some(&meta), &[], &[]));
    }

    #[test]
    fn matches_returns_false_when_meta_is_none_and_any_filter() {
        let filters = vec![("k".to_string(), json!("v"))];
        assert!(!metadata_matches(None, &filters, &[]));
    }

    #[test]
    fn matches_and_across_filters_all_must_pass() {
        let meta = json!({"a": 1, "b": 2});
        assert!(metadata_matches(
            Some(&meta),
            &[("a".to_string(), json!(1)), ("b".to_string(), json!(2)),],
            &[]
        ));
        // Mismatch on one → false.
        assert!(!metadata_matches(
            Some(&meta),
            &[("a".to_string(), json!(1)), ("b".to_string(), json!(99)),],
            &[]
        ));
    }

    #[test]
    fn matches_in_or_within_key() {
        let meta = json!({"status": "accepted"});
        // status ∈ {"accepted", "approved"} → match.
        assert!(metadata_matches(
            Some(&meta),
            &[],
            &[(
                "status".to_string(),
                vec![json!("accepted"), json!("approved")],
            )]
        ));
        // status ∈ {"draft"} → no match.
        assert!(!metadata_matches(
            Some(&meta),
            &[],
            &[("status".to_string(), vec![json!("draft")])]
        ));
    }

    #[test]
    fn matches_missing_required_key_fails() {
        let meta = json!({"a": 1});
        assert!(!metadata_matches(
            Some(&meta),
            &[("b".to_string(), json!(2))],
            &[]
        ));
    }

    // ── End-to-end deferred-error path via .await ────────────────────────────

    #[tokio::test]
    async fn filter_metadata_invalid_key_errs_at_await() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g6-bad-key");
        let result = mem
            .recall("q")
            .in_namespace(ns)
            .filter_metadata("bad.key", json!("v"))
            .raw()
            .await;
        assert!(
            result.is_err(),
            "filter_metadata with path-metachar key must err at .await, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn filter_metadata_in_empty_values_errs_at_await() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g6-empty-vals");
        let result = mem
            .recall("q")
            .in_namespace(ns)
            .filter_metadata_in("status", &[])
            .raw()
            .await;
        assert!(
            result.is_err(),
            "filter_metadata_in with empty values slice must err at .await, got: {result:?}"
        );
    }

    /// G8 cap test runner moved to separate `forget_by_source_id_tests` mod
    /// below; this anchor keeps the filter_metadata module focused on G6/G6.b.
    #[allow(dead_code)]
    fn _g6_anchor() {}

    #[tokio::test]
    async fn filter_metadata_first_error_wins() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g6-first-err");
        // First call: invalid (digit prefix). Second call: empty key.
        let result = mem
            .recall("q")
            .in_namespace(ns)
            .filter_metadata("1bad", json!("a"))
            .filter_metadata("", json!("b"))
            .raw()
            .await;
        let err = result.expect_err("must err");
        assert!(
            err.to_string().contains("1bad"),
            "first error must surface (the digit-prefix one); got: {err}"
        );
    }
}

#[cfg(test)]
mod recall_by_source_id_tests_part2 {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    // Test helper: Rule-5 exempt per clippy.toml (test helpers may carry a
    // documented too_many_arguments allow); TD-042 args-as-object targets `src/`
    // production fns, not `#[cfg(test)]` seeders.
    #[allow(clippy::too_many_arguments)]
    async fn seed_episode(
        mem: &Memory,
        source_id: &str,
        ns: &Namespace,
        recorded_at: &str,
        metadata_json: Option<&str>,
    ) {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        conn.execute(
            "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, source_id, source_uri, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            libsql::params![
                "test content",
                "2026-05-29T12:00:00Z",
                "Document",
                metadata_json,
                ns.namespace.as_str(),
                source_id,
                "uri/x",
                recorded_at
            ],
        )
        .await
        .expect("seed insert must succeed");
    }

    /// AC.6 / Quinn C4 — when no `namespace` is passed and Memory has a default
    /// namespace set, results must be scoped to that default (not span all NS).
    #[tokio::test]
    async fn no_namespace_with_memory_default_scopes_to_default() {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        let ns_default = Namespace::new("test-g5-default-scope");
        let ns_other = Namespace::new("test-g5-other-scope");

        let mem = Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .default_namespace(ns_default.clone())
            .await
            .expect("Memory with default namespace must build");

        seed_episode(&mem, "doc-x", &ns_default, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "doc-x", &ns_other, "2026-05-29T11:00:00Z", None).await;

        let eps = mem
            .recall_by_source_id("doc-x", None)
            .await
            .expect("recall with no explicit namespace must succeed");

        assert_eq!(
            eps.len(),
            1,
            "namespace=None + Memory default set must scope to the default ns"
        );
        assert_eq!(
            eps[0].group_id.as_deref(),
            Some("test-g5-default-scope"),
            "returned episode must come from the default namespace"
        );
    }

    /// AC.6 — missing source_id returns empty Vec, not Err.
    #[tokio::test]
    async fn missing_source_id_returns_empty_vec() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g5-empty");

        let eps = mem
            .recall_by_source_id("does-not-exist-xyz", Some(ns))
            .await
            .expect("recall must succeed even with no matches");

        assert!(eps.is_empty(), "no matches must return empty Vec, not Err");
    }

    /// AC.6 — Vera F11: namespace filter restricts results when Some.
    #[tokio::test]
    async fn namespace_filter_restricts_results() {
        let mem = make_memory().await;
        let ns_a = Namespace::new("test-g5-ns-a");
        let ns_b = Namespace::new("test-g5-ns-b");

        seed_episode(&mem, "shared-doc", &ns_a, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "shared-doc", &ns_b, "2026-05-29T11:00:00Z", None).await;

        // Filter to ns_a only.
        let eps_a = mem
            .recall_by_source_id("shared-doc", Some(ns_a.clone()))
            .await
            .expect("recall ns_a must succeed");
        assert_eq!(eps_a.len(), 1, "ns_a filter must return only the ns_a row");
        assert_eq!(eps_a[0].group_id.as_deref(), Some("test-g5-ns-a"));

        // Filter to ns_b only.
        let eps_b = mem
            .recall_by_source_id("shared-doc", Some(ns_b))
            .await
            .expect("recall ns_b must succeed");
        assert_eq!(eps_b.len(), 1, "ns_b filter must return only the ns_b row");
        assert_eq!(eps_b[0].group_id.as_deref(), Some("test-g5-ns-b"));
    }

    /// AC.6 — Vera F11: when namespace=None and no Memory default, returns
    /// episodes across ALL namespaces (the only opt-in cross-namespace leak path).
    #[tokio::test]
    async fn no_namespace_and_no_default_spans_all_namespaces() {
        let mem = make_memory().await;
        let ns_a = Namespace::new("test-g5-span-a");
        let ns_b = Namespace::new("test-g5-span-b");

        seed_episode(&mem, "doc-span", &ns_a, "2026-05-29T10:00:00Z", None).await;
        seed_episode(&mem, "doc-span", &ns_b, "2026-05-29T11:00:00Z", None).await;

        let eps = mem
            .recall_by_source_id("doc-span", None)
            .await
            .expect("recall with no namespace must succeed");

        assert_eq!(
            eps.len(),
            2,
            "namespace=None + no Memory default must span all namespaces"
        );
    }

    /// AC.6 — metadata JSON is parsed into serde_json::Value, NULL → None.
    #[tokio::test]
    async fn metadata_parsing_round_trips() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g5-meta");

        seed_episode(
            &mem,
            "doc-meta",
            &ns,
            "2026-05-29T10:00:00Z",
            Some(r#"{"k":"v","n":42}"#),
        )
        .await;
        seed_episode(&mem, "doc-meta-null", &ns, "2026-05-29T11:00:00Z", None).await;

        let with_meta = mem
            .recall_by_source_id("doc-meta", Some(ns.clone()))
            .await
            .expect("recall must succeed");
        assert_eq!(with_meta.len(), 1);
        let meta = with_meta[0].metadata.as_ref().expect("metadata Some");
        assert_eq!(meta.get("k").and_then(|v| v.as_str()), Some("v"));
        assert_eq!(meta.get("n").and_then(|v| v.as_i64()), Some(42));

        let null_meta = mem
            .recall_by_source_id("doc-meta-null", Some(ns))
            .await
            .expect("recall must succeed");
        assert_eq!(null_meta.len(), 1);
        assert!(null_meta[0].metadata.is_none(), "NULL metadata → None");
    }
}

/// TD-066 Increment 1 — `fuse_content_stream`'s "no attached `TemporalGraph`"
/// degrade path (spec §3 Increment 1 step 6: a DEFAULT-behaviour change must
/// degrade silently for any caller bypassing the builder path, never error).
/// Feature-gated because `fuse_content_stream`'s `content-search` arm (the
/// one with the guard this test exercises) only exists under that feature.
#[cfg(all(test, feature = "content-search"))]
mod fuse_content_stream_tests {
    use super::*;

    async fn make_memory() -> Memory {
        use crate::core::provider::{
            DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider,
        };
        use std::sync::Arc;
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    /// `Memory` constructed via a stub `GraphHandle` has `temporal_graph:
    /// None`. `fuse_content_stream` must return the entity stream unchanged
    /// (not `Err`) in that case — mirrors `.content()`'s OWN precondition
    /// check existing loudly, but this is a default-path fusion, not an
    /// opt-in terminal, so silent passthrough is the correct degrade here.
    ///
    /// Builds a REAL Memory via the normal builder path (so every field
    /// besides `temporal_graph` matches production construction, robust to
    /// future field additions) then overrides just `temporal_graph: None`
    /// via functional-update syntax — the field is `pub(crate)`, readable
    /// from this in-crate test module.
    #[tokio::test]
    async fn fuse_content_stream_passes_through_unchanged_without_temporal_graph() {
        let base = make_memory().await;
        let memory = Memory {
            temporal_graph: None,
            ..base
        };

        let ns = Namespace::new("fuse-no-tg");
        let seed = vec![RetrievedContext::new(
            crate::memory::types::RetrievedContextNewParams {
                entity_id: "seed".to_owned(),
                entity_name: "seed".to_owned(),
                summary: "unchanged".to_owned(),
                score: 0.5,
                source_refs: Vec::new(),
            },
        )];

        let fused = fuse_content_stream(FuseContentStreamParams {
            memory: &memory,
            namespace: &ns,
            query: "anything",
            limit: None,
            entity_results: seed.clone(),
        })
        .await
        .expect("must degrade to Ok, not Err, when temporal_graph is None");

        assert_eq!(
            fused.len(),
            seed.len(),
            "entity stream must pass through unchanged when there's no TemporalGraph \
             to fuse a content stream from: {fused:?}"
        );
        assert_eq!(fused[0].entity_id, "seed");
    }
}

/// TD-066 Increment 3 (TD-062 reranker) — fast tier: `apply_rerank_with`'s
/// wiring logic (head/tail split, id→context reassembly, score overwrite,
/// rank-delta bookkeeping) exercised via a deterministic mock `Reranker` —
/// zero model load, zero I/O (spec §3 Increment 3 test pyramid).
#[cfg(all(test, feature = "rerank"))]
mod apply_rerank_tests {
    use super::*;
    use crate::core::rerank::Reranker;

    fn ctx(id: &str, score: f32) -> RetrievedContext {
        RetrievedContext::new(crate::memory::types::RetrievedContextNewParams {
            entity_id: id.to_owned(),
            entity_name: id.to_owned(),
            summary: format!("summary for {id}"),
            score,
            source_refs: Vec::new(),
        })
    }

    /// Deterministic mock scoring every candidate by a caller-supplied
    /// `HashMap<id, score>` — proves `apply_rerank_with`'s wiring reorders
    /// by the RERANKER's score, not the fused-input order.
    struct MockReranker {
        scores: std::collections::HashMap<String, f32>,
    }

    #[async_trait::async_trait]
    impl Reranker for MockReranker {
        async fn rerank(
            &self,
            _query: &str,
            candidates: &[(String, String)],
        ) -> crate::core::error::Result<Vec<(String, f32)>> {
            let mut out: Vec<(String, f32)> = candidates
                .iter()
                .map(|(id, _)| (id.clone(), *self.scores.get(id).unwrap_or(&0.0)))
                .collect();
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(out)
        }
    }

    /// A reranker whose `rerank()` always errors — proves the fail-OPEN
    /// contract (a reranker failure must never break recall).
    struct FailingReranker;

    #[async_trait::async_trait]
    impl Reranker for FailingReranker {
        async fn rerank(
            &self,
            _query: &str,
            _candidates: &[(String, String)],
        ) -> crate::core::error::Result<Vec<(String, f32)>> {
            Err(crate::core::error::Error::Search("mock failure".to_owned()))
        }
    }

    #[tokio::test]
    async fn rerank_k_none_is_a_no_op() {
        let results = vec![ctx("a", 0.9), ctx("b", 0.5)];
        let reranker = MockReranker {
            scores: std::collections::HashMap::new(),
        };
        let out = apply_rerank_with(ApplyRerankWithParams {
            reranker: &reranker,
            query: "query",
            rerank_k: None,
            results: results.clone(),
        })
        .await
        .unwrap();
        assert_eq!(
            out.iter().map(|c| c.entity_id.clone()).collect::<Vec<_>>(),
            results
                .iter()
                .map(|c| c.entity_id.clone())
                .collect::<Vec<_>>(),
            "rerank_k: None must leave fused order completely unchanged"
        );
    }

    #[tokio::test]
    async fn reranker_reorders_the_top_k_head() {
        // Fused order: a, b, c (by RRF/fusion score). Mock reranker scores
        // "c" highest — proves the wiring adopts the RERANKER's order for
        // the reranked head, not the original fused order.
        let results = vec![ctx("a", 0.9), ctx("b", 0.5), ctx("c", 0.1)];
        let reranker = MockReranker {
            scores: [
                ("a".to_string(), 0.1),
                ("b".to_string(), 0.2),
                ("c".to_string(), 0.9),
            ]
            .into_iter()
            .collect(),
        };
        let out = apply_rerank_with(ApplyRerankWithParams {
            reranker: &reranker,
            query: "query",
            rerank_k: Some(3),
            results,
        })
        .await
        .unwrap();
        assert_eq!(
            out.iter().map(|c| c.entity_id.as_str()).collect::<Vec<_>>(),
            vec!["c", "b", "a"],
            "reranker's score order must win for the reranked head"
        );
        // Score field must be OVERWRITTEN with the reranker's score, not the
        // stale fusion score.
        assert!((out[0].score - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn candidates_beyond_rerank_k_pass_through_unreranked_after_head() {
        // rerank_k=1 reranks only "a"; "b" and "c" must survive, appended
        // after in their ORIGINAL fused order (untouched by the mock, which
        // would otherwise put "c" first).
        let results = vec![ctx("a", 0.9), ctx("b", 0.5), ctx("c", 0.1)];
        let reranker = MockReranker {
            scores: [("a".to_string(), 0.5), ("c".to_string(), 0.9)]
                .into_iter()
                .collect(),
        };
        let out = apply_rerank_with(ApplyRerankWithParams {
            reranker: &reranker,
            query: "query",
            rerank_k: Some(1),
            results,
        })
        .await
        .unwrap();
        assert_eq!(
            out.iter().map(|c| c.entity_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"],
            "only the top-1 head is reranked ('a' alone can't reorder); \
             the tail ('b','c') must pass through in original fused order: {out:?}"
        );
    }

    #[tokio::test]
    async fn zero_or_one_candidate_skips_reranker_entirely() {
        let results = vec![ctx("a", 0.9)];
        let reranker = FailingReranker;
        // If the wiring actually called the (failing) reranker for a
        // single-candidate set, this would hit the Err arm and log a
        // warning rather than short-circuit — asserting Ok here proves the
        // `results.len() <= 1` guard fires BEFORE any reranker call.
        let out = apply_rerank_with(ApplyRerankWithParams {
            reranker: &reranker,
            query: "query",
            rerank_k: Some(5),
            results,
        })
        .await
        .unwrap();
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn reranker_error_fails_open_to_original_fused_order() {
        let results = vec![ctx("a", 0.9), ctx("b", 0.5)];
        let reranker = FailingReranker;
        let out = apply_rerank_with(ApplyRerankWithParams {
            reranker: &reranker,
            query: "query",
            rerank_k: Some(2),
            results,
        })
        .await
        .expect("a reranker error must fail OPEN, not propagate Err");
        assert_eq!(
            out.iter().map(|c| c.entity_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"],
            "on reranker error, original fused order must survive unchanged"
        );
    }
}
