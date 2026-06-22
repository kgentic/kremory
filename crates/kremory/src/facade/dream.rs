use super::*;

// ── DreamRequest ──────────────────────────────────────────────────────────────

/// Dream phase (batch consolidation) request builder. Obtain via `mem.dream()`.
pub struct DreamRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) namespace: Option<Namespace>,
    pub(super) batch_id: Option<String>,
    pub(super) batch_size: Option<usize>,
    pub(super) sink: Option<Arc<dyn EnrichmentEventSink>>,
    pub(super) fire_and_forget: bool,
    pub(super) opts: Option<DreamOpts>,
}

impl<'a> DreamRequest<'a> {
    /// Set the namespace scope for this consolidation.
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Idempotent batch key. Multiple calls with same `(namespace, batch_id)` return
    /// the existing handle without starting a new run.
    pub fn for_batch(mut self, id: impl Into<String>) -> Self {
        self.batch_id = Some(id.into());
        self
    }

    /// Tunable: set the episode batch size for this consolidation pass.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = Some(n);
        self
    }

    /// Per-call event sink override.
    pub fn with_event_sink(mut self, sink: Arc<dyn EnrichmentEventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Return `DreamHandle` immediately without blocking on completion.
    /// Use `mem.await_dream(&handle, timeout)` to wait.
    pub fn fire_and_forget(mut self) -> DreamFireAndForget<'a> {
        self.fire_and_forget = true;
        DreamFireAndForget { inner: self }
    }

    /// Explicit form of the default: block until the dream phase completes.
    pub fn await_completion(mut self) -> Self {
        self.fire_and_forget = false;
        self
    }

    /// Escape hatch: set raw `DreamOpts` directly.
    pub fn opts(mut self, opts: DreamOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    async fn execute_blocking(self) -> Result<DreamSummary> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        let sink = self.memory.resolve_sink(self.sink);
        let opts = self.opts.unwrap_or_default();
        // ADR-029a lazy population.
        self.memory.ensure_namespace_policy(&ns).await?;

        // ADR-029b §3.1 enforcement — v0.1.5 closure of the v0.1.4
        // declare-but-don't-enforce contract. DreamRequest mutates the
        // graph (consolidation rewrites facts) and is prohibited on
        // AppendOnly namespaces. Returns the canonical CoreError variant
        // so callers can pattern-match on the policy violation.
        if let Some(tg) = self.memory.temporal_graph.as_ref() {
            let group_id = namespace_to_group_id(&ns);
            let policy = tg
                .get_namespace_policy_cached(&group_id)
                .await
                .map_err(MemoryError::Core)?;
            if let Some(p) = &policy {
                if p.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                    return Err(MemoryError::Core(
                        crate::core::error::Error::NamespacePolicyViolation {
                            namespace: group_id.clone(),
                            operation: "dream".to_string(),
                            policy: p.clone(),
                        },
                    ));
                }
            }
        }

        // Use legacy synchronous path: run_dream_phase → DreamPhaseResult → DreamSummary
        // This is the correct substrate call for blocking dream at v0.1.0.
        // dream() is a Category B method (ADR-041) — requires LLM; returns LlmRequired on NoLlm.
        let llm = self.memory.dream_llm_or_main(
            "dream",
            "wire an LLM via Memory::open(…).with_llm(…) to enable the dream consolidation phase",
        )?;
        #[allow(deprecated)]
        let mut result =
            memory::run_dream_phase(self.memory.graph.as_ref(), ns.clone(), llm.clone()).await?;
        // Sink is accepted but dream events are fired by the graph impl internally.
        // The sink parameter is stored for future use when non-blocking dream fires events.
        let _ = sink;

        // ADR-037 §3 D6 — Dream Pass 0: type discovery.
        // Run after core dream phase so Pass 0 can observe freshly-consolidated graph state.
        if opts.include_type_discovery {
            if let Some(tg) = self.memory.temporal_graph.as_ref() {
                let group_id = namespace_to_group_id(&ns);
                let embedder_ref: Option<&dyn crate::core::provider::DynEmbeddingProvider> =
                    Some(self.memory.embedder.as_ref());
                // ADR-037 §3 D6: wrap `Arc<dyn ChatProvider>` in `ArcChatProvider`
                // newtype so `discover_types<L: ChatProvider>` (Sized bound) can
                // accept it monomorphized. Orphan rules prevent
                // `impl ChatProvider for Arc<dyn ChatProvider>` directly.
                let arc_llm = crate::core::provider::ArcChatProvider::new(llm.clone());
                // F3 — apply max_episodes_per_run cap to Pass 0 cluster budget.
                // `max_proposals` bounds the entity clusters surfaced to the LLM;
                // capping it is the observable knob the ops team can tune.
                let pass0_max = opts
                    .max_episodes_per_run
                    .map(|cap| {
                        let default_max = crate::core::dream::discover_types::MAX_PROPOSALS;
                        if cap < default_max {
                            metrics::counter!(
                                "kremory.dream.batch_cap_hit_total",
                                "phase" => "pass0"
                            )
                            .increment(1);
                            tracing::debug!(
                                target: "kremory::dream::pass0",
                                cap,
                                default_max,
                                "kremory.dream.pass0 batch capped by max_episodes_per_run"
                            );
                            cap
                        } else {
                            default_max
                        }
                    })
                    .unwrap_or(crate::core::dream::discover_types::MAX_PROPOSALS);
                match crate::core::dream::discover_types::discover_types(
                    &arc_llm,
                    crate::core::dream::discover_types::DiscoverTypesParams {
                        conn: &tg.conn,
                        group_id: &group_id,
                        embedder: embedder_ref,
                        max_proposals: pass0_max,
                    },
                )
                .await
                {
                    Ok(discovery) => {
                        result.types_discovered = discovery.types_accepted;
                        result.dream_warnings.extend(discovery.warnings);
                    }
                    Err(e) => {
                        // Pass 0 failure is non-fatal — surface as warning, don't abort dream.
                        tracing::warn!(
                            target: "kremory::dream::pass0",
                            error = %e,
                            "Dream Pass 0 type discovery failed — skipping; dream phase result unaffected"
                        );
                        result
                            .dream_warnings
                            .push(format!("Dream Pass 0 type discovery failed: {e}"));
                    }
                }
            }
        }

        // ADR-046 Option E — Dream Pass 2: reclassify.
        // Runs AFTER Pass 0 so newly discovered types (from Pass 0) are available in
        // the entity type registry for the reclassify LLM prompt.
        // Pass ordering per DoD E7: Pass 0 commits → Pass 2 (reclassify) → Pass 3 (canonicalize).
        // Pass 3 (canonicalize) is handled by `run_dream_phase` above (legacy path).
        {
            if let Some(tg) = self.memory.temporal_graph.as_ref() {
                let group_id = namespace_to_group_id(&ns);
                let arc_llm = crate::core::provider::ArcChatProvider::new(llm.clone());
                // F3 — apply max_episodes_per_run cap to Pass 2 entity-candidate batch.
                // `ReclassifyOpts::max_batch_size` bounds candidates processed per run.
                let pass2_opts = {
                    let mut o = crate::core::dream::reclassify::ReclassifyOpts::default();
                    if let Some(cap) = opts.max_episodes_per_run {
                        if cap < o.max_batch_size {
                            metrics::counter!(
                                "kremory.dream.batch_cap_hit_total",
                                "phase" => "pass2"
                            )
                            .increment(1);
                            tracing::debug!(
                                target: "kremory::dream::pass2",
                                cap,
                                default_batch = o.max_batch_size,
                                "kremory.dream.pass2 batch capped by max_episodes_per_run"
                            );
                            o.max_batch_size = cap;
                        }
                    }
                    o
                };
                // Use reclassify_high_conf_threshold from opts if available via DreamOpts;
                // DreamOpts does not yet carry per-pass thresholds — use canonical defaults.
                // Full per-pass tuning via DreamPassOpts is available on the Engine path;
                // the DreamRequest path uses sensible defaults until DreamOpts is extended.
                match crate::core::dream::reclassify::reclassify(
                    &arc_llm,
                    crate::core::dream::reclassify::ReclassifyParams {
                        conn: &tg.conn,
                        group_id: &group_id,
                        opts: pass2_opts,
                    },
                )
                .await
                {
                    Ok(reclassify_result) => {
                        // Aggregate entities_reclassified into DreamSummary (E8).
                        // DreamPhaseResult does not yet carry entities_reclassified;
                        // we accumulate it separately and fold into DreamSummary after From.
                        let reclassified = reclassify_result.entities_reclassified;
                        result.dream_warnings.extend(reclassify_result.warnings);
                        // Convert result → summary, then set reclassified count (E8).
                        let mut summary = DreamSummary::from(result);
                        summary.entities_reclassified = reclassified;
                        return Ok(summary);
                    }
                    Err(e) => {
                        // Pass 2 failure is non-fatal — surface as warning, don't abort dream.
                        tracing::warn!(
                            target: "kremory::dream::pass2",
                            error = %e,
                            "Dream Pass 2 reclassify failed — skipping; dream phase result unaffected"
                        );
                        result
                            .dream_warnings
                            .push(format!("Dream Pass 2 reclassify failed: {e}"));
                    }
                }
            }
        }

        Ok(DreamSummary::from(result))
    }
}

impl<'a> IntoFuture for DreamRequest<'a> {
    type Output = Result<DreamSummary>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute_blocking())
    }
}

/// Fire-and-forget dream variant — returns `DreamHandle` without blocking.
pub struct DreamFireAndForget<'a> {
    inner: DreamRequest<'a>,
}

impl<'a> IntoFuture for DreamFireAndForget<'a> {
    type Output = Result<DreamHandle>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let ns = self.inner.memory.resolve_namespace(self.inner.namespace)?;
            let sink = self.inner.memory.resolve_sink(self.inner.sink);
            let opts = self.inner.opts.unwrap_or_default();
            // ADR-029a lazy population.
            self.inner.memory.ensure_namespace_policy(&ns).await?;
            // dream consolidation is Category B (ADR-041) — requires LLM.
            let llm = self.inner.memory.dream_llm_or_main(
                "dream",
                "wire an LLM via Memory::open(…).with_llm(…) to enable the dream consolidation phase",
            )?;
            memory::submit_dream_phase(memory::SubmitDreamPhaseParams {
                graph: self.inner.memory.graph.as_ref(),
                namespace: ns,
                provider: llm,
                batch_id: self.inner.batch_id,
                opts,
                sink,
            })
            .await
        })
    }
}
