use super::*;

// ── RememberRequest ───────────────────────────────────────────────────────────

/// Ingest request builder. Obtain via `mem.remember("…")`.
pub struct RememberRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) content: String,
    pub(super) source_ref: Option<SourceRef>,
    pub(super) namespace: Option<Namespace>,
    pub(super) published_at: Option<DateTime<Utc>>,
    pub(super) facts: Vec<StructuredFact>,
    pub(super) sink: Option<Arc<dyn EnrichmentEventSink>>,
    pub(super) no_wait: bool,
    pub(super) opts: Option<SubmitOpts>,
    /// When `true`, Phase 2 LLM extraction is skipped at engine
    /// level. Episode + embedding + `with_facts` triples are still persisted.
    /// Set via [`RememberRequest::skip_extraction`] builder method.
    pub(super) skip_extraction: bool,
}

impl<'a> RememberRequest<'a> {
    /// Tag this episode as originating from a chat session.
    pub fn from_chat(mut self, id: impl Into<String>) -> Self {
        self.source_ref = Some(SourceRef {
            kind: SourceKind::Chat,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Tag this episode as originating from a note.
    pub fn from_note(mut self, id: impl Into<String>) -> Self {
        self.source_ref = Some(SourceRef {
            kind: SourceKind::Document,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Tag this episode as originating from a document.
    pub fn from_document(mut self, id: impl Into<String>) -> Self {
        self.source_ref = Some(SourceRef {
            kind: SourceKind::Document,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Full escape hatch: specify source ID and kind directly.
    pub fn from_source(mut self, id: impl Into<String>, kind: SourceKind) -> Self {
        self.source_ref = Some(SourceRef {
            kind,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });
        self
    }

    /// Set the namespace for this operation (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Set the bi-temporal anchor for the source document.
    ///
    /// When set, `valid_from` for structured facts without their own anchor
    /// falls back to this timestamp.
    pub fn published_at(mut self, ts: DateTime<Utc>) -> Self {
        self.published_at = Some(ts);
        // Propagate into source_ref if already set
        if let Some(ref mut sr) = self.source_ref {
            sr.published_at = Some(ts);
        }
        self
    }

    /// Attach pre-extracted structured facts.
    ///
    /// Caller's facts are pinned into the graph BEFORE Phase 2 LLM extraction
    /// runs. LLM-extracted duplicates of the same
    /// `(subject, predicate, object)` triple are silently swallowed via the
    /// substrate `try_insert_fact` helper — caller wins by virtue of being
    /// there first. Phase 2 LLM extraction still runs on `content`
    /// (additive default).
    ///
    /// ⚠️ `StructuredFact.predicate` values recognised as reserved (e.g.
    /// `"potential_alias"` — see `crate::core::disambiguation::
    /// is_reserved_predicate`) are stored but never returned by any
    /// `recall()` call. This is a read-side exclusion, not a rejection: the
    /// call succeeds, the fact is written, and it simply never comes back to
    /// a consumer. Callers should avoid these predicate strings for their
    /// own facts.
    ///
    /// To skip Phase 2 LLM extraction entirely (caller is the sole source of
    /// truth for facts), chain [`RememberRequest::skip_extraction`].
    pub fn with_facts(mut self, facts: Vec<StructuredFact>) -> Self {
        self.facts = facts;
        self
    }

    /// Skip Phase 2 LLM extraction for this episode.
    ///
    /// When set, the engine persists the episode + embedding + any
    /// [`with_facts`](Self::with_facts)-supplied triples, then bails before
    /// invoking the entity/edge extractor. Suitable for bulk-import workloads
    /// where the caller already has high-confidence structured data and LLM
    /// cycles would be wasted.
    ///
    /// Method name is implementation-agnostic — if a future kremory version
    /// swaps the Phase 2 extractor (GLiNER, regex, hybrid), the semantics
    /// remain "skip the discovery step". Added in v0.1.8.
    pub fn skip_extraction(mut self) -> Self {
        self.skip_extraction = true;
        self
    }

    /// Per-call event sink override. Overrides the Memory-level default for
    /// this operation only.
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Return after Phase 1 commit (store + embed) without blocking on Phase 2.
    ///
    /// The returned `EpisodeCommit.run_id` will be `Some(...)` — use
    /// `mem.status_of(&commit)` or `mem.await_enrichment(&commit, timeout)` to
    /// track Phase 2 completion.
    pub fn no_wait(mut self) -> Self {
        self.no_wait = true;
        self
    }

    /// Explicit form of the default: block until Phase 2 enrichment is done.
    ///
    /// Equivalent to the default `.await` terminal; provided for clarity in code
    /// that toggles between `.no_wait()` and blocking paths.
    pub fn await_enrichment(mut self) -> Self {
        self.no_wait = false;
        self
    }

    /// Escape hatch: set raw substrate `SubmitOpts` directly.
    pub fn opts(mut self, opts: SubmitOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    async fn execute(self) -> Result<EpisodeCommit> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let sink = self.memory.resolve_sink(self.sink);

        // Soft warn at content threshold. Never enforced; observability only.
        // This threshold does not protect the dense/embedding arm — see
        // `MemoryBuilder::episode_content_warn_threshold`'s doc comment. Content
        // past the embedder's own (smaller, provider-specific) context window
        // silently loses dense-arm coverage; check
        // `EpisodeCommit::dense_embedded` to detect it, not this warning.
        if let Some(threshold) = self.memory.episode_content_warn_threshold {
            let chars = self.content.chars().count();
            if chars > threshold {
                tracing::warn!(
                    episode_chars = chars,
                    threshold = threshold,
                    "episode content exceeds soft threshold — extraction quality may degrade AND the embedder's own (separate, provider-specific) context window may silently drop the dense-arm; pre-chunk with kremory::split_for_embedding before remember() if this matters, and check EpisodeCommit::dense_embedded"
                );
                metrics::counter!("kremory_episode_oversize_total").increment(1);
            }
        }

        let source_ref = self.source_ref.unwrap_or_else(|| SourceRef {
            kind: SourceKind::Chat,
            id: uuid::Uuid::new_v4().to_string(),
            occurred_at: Utc::now(),
            published_at: self.published_at,
        });

        // Build SubmitOpts; if caller invoked `.skip_extraction()`, force
        // `enrich_per_episode = false` so the gate at engine_handle propagates
        // into `SourceParams.skip_extraction` and the engine bails after the
        // caller-pin step. Caller's explicit `.opts(...)` overrides this.
        let mut opts = self.opts.unwrap_or(SubmitOpts {
            enrich_per_episode: true,
            run_in_background: self.no_wait,
        });
        if self.skip_extraction {
            opts.enrich_per_episode = false;
        }

        // Lazy population: ensures a default-policy row exists for
        // the namespace before the first write.
        self.memory.ensure_namespace_policy(&ns).await?;

        let commit = memory::submit_episode(memory::SubmitEpisodeParams {
            graph: self.memory.graph.as_ref(),
            content: &self.content,
            source_ref,
            structured_facts: self.facts,
            provider: self.memory.llm_or_stub(),
            namespace: ns,
            batch_id: None,
            opts,
            sink,
        })
        .await?;

        // Opt-in synchronous-extraction ergonomics.
        // When `await_extraction = true`, block until the background worker
        // transitions the episode to `Verified` (or return Err on
        // `Failed` / timeout).
        //
        // The episode rowid is stored in `episode_entity_id` as a decimal
        // string (see engine_handle.rs — `ingest_result.episode_id.to_string()`).
        // We parse it back to i64 here.
        //
        // When parsing fails, the path was `run_in_background=true` (`.no_wait()`),
        // which sets `episode_entity_id` to a UUID string (see engine_handle.rs
        // background path — `run_id.to_string()`). A UUID is not a parseable i64
        // rowid, so `wait_for_processing` cannot be called (it requires a rowid to
        // poll). Behavior: skip the wait (fire-and-forget semantics are correct for
        // the background path), but emit a tracing::warn + counter so the skip is
        // observable. Callers combining `with_await_extraction(true)` + `.no_wait()`
        // receive the commit immediately — `await_extraction` semantics are degraded
        // to fire-and-forget for this combination, and the log makes that visible.
        // Silent behavior changes MUST emit a signal.
        if self.memory.await_extraction {
            match commit.episode_entity_id.parse::<i64>() {
                Ok(episode_id) => {
                    tracing::debug!(
                        target: "kremory.remember",
                        episode_id,
                        timeout_secs = self.memory.await_extraction_timeout.as_secs(),
                        "await_extraction=true — waiting for background processing"
                    );
                    self.memory
                        .wait_for_processing(episode_id, self.memory.await_extraction_timeout)
                        .await?;
                }
                Err(parse_err) => {
                    // episode_entity_id is a UUID (background/no_wait path) — cannot
                    // call wait_for_processing without an i64 rowid. Skip wait, emit
                    // an observability signal.
                    tracing::warn!(
                        target: "kremory.remember",
                        episode_entity_id = %commit.episode_entity_id,
                        parse_error = %parse_err,
                        "await_extraction=true but episode_entity_id is not a parseable rowid \
                         (UUID path — run_in_background=true?); wait skipped. \
                         Combine with_await_extraction(true) with the default blocking path \
                         (not .no_wait()) to enable wait semantics."
                    );
                    metrics::counter!("kremory.remember.episode_id_parse_fail_total").increment(1);
                }
            }
        }

        Ok(commit)
    }
}

impl<'a> IntoFuture for RememberRequest<'a> {
    type Output = Result<EpisodeCommit>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

// ── RememberBatchBuilder ──────────────────────────────────────────────────────

/// Bulk ingest request builder. Obtain via `mem.remember_batch()`.
pub struct RememberBatchBuilder<'a> {
    pub(super) memory: &'a Memory,
    pub(super) episodes: Vec<PendingEpisode>,
    pub(super) batch_id: Option<String>,
    pub(super) sink: Option<Arc<dyn EnrichmentEventSink>>,
}

pub(super) struct PendingEpisode {
    content: String,
    source_ref: Option<SourceRef>,
    namespace: Option<Namespace>,
    published_at: Option<DateTime<Utc>>,
    facts: Vec<StructuredFact>,
}

impl<'a> RememberBatchBuilder<'a> {
    /// Add an episode to the batch. Returns an `EpisodeEntryBuilder` for chaining.
    pub fn entry(self, content: impl Into<String>) -> EpisodeEntryBuilder<'a> {
        EpisodeEntryBuilder {
            batch: self,
            pending: PendingEpisode {
                content: content.into(),
                source_ref: None,
                namespace: None,
                published_at: None,
                facts: vec![],
            },
        }
    }

    /// Set the batch ID for idempotent batch tracking.
    pub fn with_batch_id(mut self, id: impl Into<String>) -> Self {
        self.batch_id = Some(id.into());
        self
    }

    /// Per-batch event sink (applies to all episodes in this batch).
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    async fn execute(self) -> Result<Vec<EpisodeCommit>> {
        let mut commits = Vec::with_capacity(self.episodes.len());
        for ep in self.episodes {
            let ns = self.memory.resolve_namespace(ep.namespace)?;
            let sink = self.memory.resolve_sink(self.sink.clone());
            let source_ref = ep.source_ref.unwrap_or_else(|| SourceRef {
                kind: SourceKind::Chat,
                id: uuid::Uuid::new_v4().to_string(),
                occurred_at: Utc::now(),
                published_at: ep.published_at,
            });
            // Lazy population: ensures a namespace-policy row exists.
            self.memory.ensure_namespace_policy(&ns).await?;
            let commit = memory::submit_episode(memory::SubmitEpisodeParams {
                graph: self.memory.graph.as_ref(),
                content: &ep.content,
                source_ref,
                structured_facts: ep.facts,
                provider: self.memory.llm_or_stub(),
                namespace: ns,
                batch_id: self.batch_id.clone(),
                opts: SubmitOpts {
                    enrich_per_episode: true,
                    run_in_background: false,
                },
                sink,
            })
            .await?;
            commits.push(commit);
        }
        Ok(commits)
    }
}

impl<'a> IntoFuture for RememberBatchBuilder<'a> {
    type Output = Result<Vec<EpisodeCommit>>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Per-episode entry builder within a `RememberBatchBuilder`.
pub struct EpisodeEntryBuilder<'a> {
    batch: RememberBatchBuilder<'a>,
    pending: PendingEpisode,
}

impl<'a> EpisodeEntryBuilder<'a> {
    /// Tag as chat source.
    pub fn from_chat(mut self, id: impl Into<String>) -> Self {
        self.pending.source_ref = Some(SourceRef {
            kind: SourceKind::Chat,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.pending.published_at,
        });
        self
    }

    /// Tag as document source.
    pub fn from_document(mut self, id: impl Into<String>) -> Self {
        self.pending.source_ref = Some(SourceRef {
            kind: SourceKind::Document,
            id: id.into(),
            occurred_at: Utc::now(),
            published_at: self.pending.published_at,
        });
        self
    }

    /// Set namespace for this episode.
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.pending.namespace = Some(ns);
        self
    }

    /// Set bi-temporal anchor for this episode.
    pub fn published_at(mut self, ts: DateTime<Utc>) -> Self {
        self.pending.published_at = Some(ts);
        if let Some(ref mut sr) = self.pending.source_ref {
            sr.published_at = Some(ts);
        }
        self
    }

    /// Attach pre-extracted facts.
    pub fn with_facts(mut self, facts: Vec<StructuredFact>) -> Self {
        self.pending.facts = facts;
        self
    }

    /// Finish this entry and return to the batch builder for chaining or terminal.
    pub fn done(mut self) -> RememberBatchBuilder<'a> {
        self.batch.episodes.push(self.pending);
        self.batch
    }
}

//
// Spec AC.5 — soft warn at configurable content threshold. Asserts the warn
// fires above threshold, does not fire below, can be disabled via `None`, and
// is never enforced (never returns Err on its own).

#[cfg(test)]
mod episode_content_warn_threshold_tests {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory_with_threshold(threshold: Option<usize>) -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .episode_content_warn_threshold(threshold)
            .await
            .expect("in-memory Memory construction must not fail")
    }

    /// AC.5 — warn fires when content.chars().count() exceeds threshold.
    /// Threshold set deliberately small (5 chars) so test content trivially
    /// triggers; production default of 10_000 is irrelevant here.
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn warn_fires_above_threshold() {
        let mem = make_memory_with_threshold(Some(5)).await;
        let ns = Namespace::new("test-g4-warn-fires");

        // 11 chars > threshold 5 → warn must fire.
        // .await may fail downstream (null providers + enrichment), but the
        // warn emits BEFORE submit_episode so logs_contain captures it.
        let _ = mem.remember("hello world").in_namespace(ns).no_wait().await;

        assert!(
            logs_contain("episode content exceeds soft threshold"),
            "warn must fire when content (11 chars) exceeds threshold (5)"
        );
    }

    /// AC.5 — warn does NOT fire when content is at or below threshold.
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn warn_does_not_fire_below_threshold() {
        let mem = make_memory_with_threshold(Some(100)).await;
        let ns = Namespace::new("test-g4-no-warn");

        // 11 chars <= threshold 100 → warn must NOT fire.
        let _ = mem.remember("hello world").in_namespace(ns).no_wait().await;

        assert!(
            !logs_contain("episode content exceeds soft threshold"),
            "warn must NOT fire when content (11 chars) is below threshold (100)"
        );
    }

    /// AC.5 — `None` disables the warning entirely (even for very large content).
    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn warn_disabled_when_threshold_is_none() {
        let mem = make_memory_with_threshold(None).await;
        let ns = Namespace::new("test-g4-disabled");

        // 50_000-char content; without threshold the warn must NOT fire.
        let huge = "x".repeat(50_000);
        let _ = mem.remember(huge).in_namespace(ns).no_wait().await;

        assert!(
            !logs_contain("episode content exceeds soft threshold"),
            "warn must NOT fire when threshold is None, regardless of content size"
        );
    }

    /// AC.5 — default threshold via `Memory::open` is `Some(10_000)`.
    /// Validates the documented default + builder default propagation.
    #[tokio::test(flavor = "multi_thread")]
    async fn default_threshold_is_some_10_000() {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });

        let mem = Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("default-config Memory must build");

        assert_eq!(
            mem.episode_content_warn_threshold,
            Some(10_000),
            "default threshold must be Some(10_000) per spec"
        );
    }
}
