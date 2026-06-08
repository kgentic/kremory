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
    /// sub-query per namespace and blends results via cross-namespace RRF.
    pub(super) namespaces: Option<Vec<Namespace>>,
    /// Per-namespace top-K cap before cross-namespace RRF blend (ADR-029c
    /// Decision 4). Default: effective `k`. Raising this value improves recall
    /// diversity for low-coverage namespaces at the cost of extra sub-query work.
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
async fn apply_metadata_post_filter(
    memory: &Memory,
    namespace: &Namespace,
    results: Vec<RetrievedContext>,
    metadata_filters: &[(String, serde_json::Value)],
    metadata_filters_in: &[(String, Vec<serde_json::Value>)],
) -> Result<Vec<RetrievedContext>> {
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

    /// Recall across multiple namespaces concurrently, blending results via
    /// per-namespace top-K RRF. Each result in the returned `Vec` carries
    /// `namespace: Some(ns)` identifying its source namespace.
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
    /// cross-namespace RRF blending. Default: `k` (or `Memory::default_k` if
    /// `k` is unset). Raising this value improves recall diversity for
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

    /// Point-in-time filter (v0.1.0: emits `tracing::warn!`, not yet implemented in SQL).
    /// v0.1.1 will apply the filter. Setting this now ensures forward-compatible caller code.
    pub fn as_of(mut self, ts: DateTime<Utc>) -> Self {
        self.as_of = Some(ts);
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
        });
        // ADR-029a lazy population.
        self.memory.ensure_namespace_policy(&ns).await?;
        let recall_id = self.recall_id;
        let span = tracing::info_span!(
            "kremory.recall.single_ns",
            recall_id = %recall_id,
            namespace = %ns.namespace,
        );
        let _enter = span.enter();
        let mut results = memory::search(self.memory.graph.as_ref(), &self.query, ns.clone(), opts)
            .await?
            .into_iter()
            .map(|r| r.with_namespace(ns.clone()))
            .collect::<Vec<_>>();
        // G6 — apply metadata post-filter before template rendering.
        results = apply_metadata_post_filter(
            self.memory,
            &ns,
            results,
            &self.metadata_filters,
            &self.metadata_filters_in,
        )
        .await?;
        Ok(memory::context_block(&results, template.into()))
    }

    /// Fan-out recall across all namespaces in `self.namespaces`, blend via RRF,
    /// trim to `self.k`, and return results with `namespace: Some(ns)` attribution.
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
            let opts = opts_template.clone().unwrap_or(SearchOpts {
                limit: per_ns_k,
                as_of,
                source_kind: None,
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
                let hits = memory::search(memory.graph.as_ref(), &query, ns.clone(), opts).await?;
                let attributed: Vec<RetrievedContext> = hits
                    .into_iter()
                    .map(|r| r.with_namespace(ns.clone()))
                    .collect();
                // G6 — Quinn C1: filters must apply per-namespace (entity_id
                // scoping requires the namespace's own conn lookup).
                let filtered = apply_metadata_post_filter(
                    memory,
                    &ns,
                    attributed,
                    &metadata_filters,
                    &metadata_filters_in,
                )
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

        // Sort blended results by score descending (RRF scores from sub-queries).
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to final_k if set.
        if let Some(k) = final_k {
            all_results.truncate(k);
        }

        Ok(all_results)
    }
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
            let opts = inner.opts.unwrap_or(SearchOpts {
                limit: inner.k,
                as_of: inner.as_of,
                source_kind: None,
            });
            // ADR-029a lazy population.
            inner.memory.ensure_namespace_policy(&ns).await?;
            let recall_id = inner.recall_id;
            let span = tracing::info_span!(
                "kremory.recall.single_ns",
                recall_id = %recall_id,
                namespace = %ns.namespace,
            );
            let _enter = span.enter();
            let results: Vec<RetrievedContext> =
                memory::search(inner.memory.graph.as_ref(), &inner.query, ns.clone(), opts)
                    .await?
                    .into_iter()
                    .map(|r| r.with_namespace(ns.clone()))
                    .collect();
            // G6 — apply metadata post-filter before returning.
            let filtered = apply_metadata_post_filter(
                inner.memory,
                &ns,
                results,
                &inner.metadata_filters,
                &inner.metadata_filters_in,
            )
            .await?;
            Ok(filtered)
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

    use super::{metadata_matches, validate_metadata_key, Memory};

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
