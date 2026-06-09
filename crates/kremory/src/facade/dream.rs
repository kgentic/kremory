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
        let _opts = self.opts.unwrap_or_default();
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
        let llm = self.memory.llm_or_err(
            "dream",
            "wire an LLM via Memory::open(…).with_llm(…) to enable the dream consolidation phase",
        )?;
        #[allow(deprecated)]
        let result =
            memory::run_dream_phase(self.memory.graph.as_ref(), ns, llm)
                .await?;
        // Sink is accepted but dream events are fired by the graph impl internally.
        // The sink parameter is stored for future use when non-blocking dream fires events.
        let _ = sink;
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
            let llm = self.inner.memory.llm_or_err(
                "dream",
                "wire an LLM via Memory::open(…).with_llm(…) to enable the dream consolidation phase",
            )?;
            memory::submit_dream_phase(
                self.inner.memory.graph.as_ref(),
                ns,
                llm,
                self.inner.batch_id,
                opts,
                sink,
            )
            .await
        })
    }
}
