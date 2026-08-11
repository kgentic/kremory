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

    /// Batch key, carried into [`DreamHandle::batch_id`] for correlation.
    ///
    /// ⚠️ **NOT idempotent today, despite the ADR having specified it.** This doc
    /// previously claimed *"multiple calls with the same `(namespace, batch_id)`
    /// return the existing handle without starting a new run"*. That is FALSE and
    /// was never implemented: `batch_id` is read at exactly one place in this file
    /// ([`DreamFireAndForget::into_future`]), and the awaited path
    /// ([`DreamRequest::execute_blocking`]) never reads it at all — so every call
    /// re-runs the whole pass chain from scratch. The `(namespace, batch_id)`
    /// registry the ADR describes does not exist; the only idempotency table in
    /// the tree (`dream_idempotency_keys`, Migration 016) is keyed
    /// `(pass_name, entity_id, content_hash)` and is written solely by the
    /// background *ingest* verify stage, never by dream.
    ///
    /// Corrected 2026-08-06 per `docs-match-implementation-as-dod` (stale-done
    /// severity — downstream reasoning rested on a capability that is absent;
    /// see the sibling corrections at `kremory-mcp/src/bin/kremory-http.rs` and
    /// `kremory-mcp/src/params.rs`). Tracked as TD-188.
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
    ///
    /// Named `with_opts` (not `opts`) so the builder verb is distinct from the
    /// `opts` field it sets (DX1 — a method named `opts` reads like a getter).
    pub fn with_opts(mut self, opts: DreamOpts) -> Self {
        self.opts = Some(opts);
        self
    }

    /// L1 opinionated control for the cross-episode entity-merge op (consumer-API
    /// hardening D3). Maps the honest tri-state [`CrossEpisodeMode`] onto the two
    /// coupled `DreamOpts` bools so a consumer never has to reason about which bool
    /// combination means what:
    ///
    /// - [`CrossEpisodeMode::Off`] → `include_cross_episode_merges = false`
    /// - [`CrossEpisodeMode::Shadow`] → `include = true`, `dry_run = true`
    /// - [`CrossEpisodeMode::Apply`] → `include = true`, `dry_run = false`
    ///
    /// Composes with any `DreamOpts` already set via [`Self::with_opts`] — only the
    /// two cross_episode fields are overwritten; every other knob is preserved.
    /// Flows through both the awaited and `fire_and_forget` paths.
    pub fn cross_episode(mut self, mode: CrossEpisodeMode) -> Self {
        let mut opts = self.opts.take().unwrap_or_default();
        match mode {
            CrossEpisodeMode::Off => opts.include_cross_episode_merges = false,
            CrossEpisodeMode::Shadow => {
                opts.include_cross_episode_merges = true;
                opts.cross_episode_dry_run = true;
            }
            CrossEpisodeMode::Apply => {
                opts.include_cross_episode_merges = true;
                opts.cross_episode_dry_run = false;
            }
        }
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

        // dream() is a Category B method (ADR-041) — requires LLM; returns LlmRequired on NoLlm.
        // Consolidation (communities/merges/supersessions/archival) IS implemented —
        // `run_consolidation` below wires all four ops when
        // `opts.any_consolidation_enabled()` (default true; cross-episode merge
        // defaults to Shadow — see `cross_episode()` above). DreamSummary's
        // consolidation fields are honest zeros only when an op is explicitly
        // disabled or a namespace has nothing to consolidate.
        let llm = self.memory.dream_llm_or_main(
            "dream",
            "wire an LLM via Memory::open(…).with_llm(…) to enable the dream consolidation phase",
        )?;
        // TD-094: resolve the model id the dream LLM passes use for capability
        // detection — the dedicated `with_dream_model_id` string when set, else
        // the main `with_model_id` string, else "" (→ PromptOnly degrade). This
        // is the missing half of the Option-1 (2026-06-23) refactor: the passes
        // were made to take a consumer-supplied model id, but the facade never
        // threaded one, silently degrading every LLM pass to zero output.
        let dream_model_id = self.memory.dream_model_id_or_main().unwrap_or_default();
        let dream_start = std::time::Instant::now();
        let mut result = memory::DreamPhaseResult::default();
        // SCOPE-001 (dream-phase-reconciliation-v2 Phase 1): accumulate per-pass
        // counts in locals so the DreamSummary is built ONCE at the end of the
        // pass chain. The reclassify pass previously early-returned on success,
        // making every pass ordered after it unreachable — the root cause of
        // "mem.dream() runs only 2 of the 5 designed passes".
        let mut entities_reclassified: usize = 0;
        // Phase 2 (§D3) deterministic-pass accumulators — consumed by per-pass
        // counters below. Phase 4 folds these into DreamSummary as pub fields;
        // that schema change MUST re-run the napi parity gate (readiness R-05)
        // and also close the pre-existing JsDreamSummary gap for
        // `types_discovered` + `entities_reclassified`.
        let mut aliases_resolved: usize = 0;
        let mut canonicalization_merges: usize = 0;
        // Phase 3 (§D3, ADR-047) consistency_check accumulator — Full-mode LLM pass.
        let mut consistency_check_corrected: usize = 0;
        // The resolved sink is threaded into `run_consolidation` below (ADR-070 Fork 5),
        // where the orchestrator fires `on_merge_proposed` for each cross_episode merge
        // decision. Other dream events are still fired by the graph impl internally.

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
                        // TD-094: thread the resolved dream model id for capability detection.
                        model_id: dream_model_id,
                        // Site #2 (ADR-063 "The six sites" #2) — spike-gated, default
                        // `false` (spec §8). Threaded from `DreamOpts::include_type_
                        // novelty_llm_verify` so the DEFAULT build's Pass-0 outcome is
                        // unchanged from before Site #2 landed.
                        llm_verify_band: opts.include_type_novelty_llm_verify,
                        // TD-123 — quarantined default `false` (unspiked cosine-alone
                        // degeneracy risk). Threaded from `DreamOpts::include_
                        // evidence_retype_by_similarity`.
                        evidence_retype_by_similarity: opts.include_evidence_retype_by_similarity,
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

        // Dream Pass — aliases (dream-phase-reconciliation-v2 §D3): resolve
        // pending `potential_alias` facts (merge or revoke). Deterministic (no
        // LLM). Ordered BEFORE reclassify so an entity about to be merged away
        // is not reclassified first. Non-fatal: failure warns + continues.
        if let Some(tg) = self.memory.temporal_graph.as_ref() {
            let group_id = namespace_to_group_id(&ns);
            match crate::core::disambiguation::resolve_pending_aliases(tg, &group_id).await {
                Ok(n) => aliases_resolved = n,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::aliases",
                        error = %e,
                        "Dream aliases pass failed — skipping; dream phase result unaffected"
                    );
                    result
                        .dream_warnings
                        .push(format!("Dream aliases pass failed: {e}"));
                }
            }
            // Counter emitted inside the graph-present block so it reflects an
            // actual pass run (not the degenerate no-temporal-graph path).
            //
            // Quinn MNT-002: carries `sweep="pre"` so it shares a label KEY with
            // the post-consolidation sweep's emission below. Emitting the same
            // metric name with two different label SETS splits it into series a
            // `sweep`-filtered query silently drops half of — an observability
            // trap inside the fix for an observability trap.
            metrics::counter!("kremory.dream.aliases_resolved_total", "sweep" => "pre")
                .increment(aliases_resolved as u64);
        }

        // Dream Pass — acronym_nickname_recall (ADR-063 spec §3, "Site #5"):
        // nominate entity-instance pairs via a deterministic structural
        // pre-filter (initialism test OR graph co-occurrence, spec §3.1) and
        // adjudicate nominated pairs via batched LLM verdicts + the shared
        // write_gate (spec §3.2/§3.3). Gated by
        // `include_acronym_nickname_recall` (default `true` — VALIDATED
        // 2026-07-03, `memory/types.rs`). Ordered
        // immediately AFTER aliases (`resolve_pending_aliases`, above) and
        // BEFORE reclassify (spec §3.0): merges land before type-correctness
        // is re-verified, avoiding a wasted reclassify pass on an entity
        // about to be merged away, and can reuse the now-resolved alias
        // state as one of its co-occurrence signals without racing a
        // concurrent alias mutation. Non-fatal: failure warns + continues.
        let mut acronym_recall_merges: usize = 0;
        if opts.include_acronym_nickname_recall {
            if let Some(tg) = self.memory.temporal_graph.as_ref() {
                let group_id = namespace_to_group_id(&ns);
                let arc_llm = crate::core::provider::ArcChatProvider::new(llm.clone());
                match crate::core::dream::acronym_nickname_recall::acronym_nickname_recall(
                    &arc_llm,
                    crate::core::dream::acronym_nickname_recall::AcronymNicknameRecallParams {
                        graph: tg,
                        group_id: &group_id,
                        // TD-094-style threading: reuse the resolved dream model id.
                        model_id: dream_model_id,
                        // TD-112: Site #5 merges live (no dry-run gate, ON by
                        // default) — thread the embedder so the surviving keeper's
                        // stored embedding is refreshed post-merge (Quinn M1).
                        embedder: Some(self.memory.embedder.as_ref()),
                    },
                )
                .await
                {
                    Ok(recall_report) => {
                        acronym_recall_merges = recall_report.merges_applied;
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "kremory::dream::acronym_recall",
                            error = %e,
                            "Dream acronym_nickname_recall pass failed — skipping; dream phase result unaffected"
                        );
                        result
                            .dream_warnings
                            .push(format!("Dream acronym_nickname_recall pass failed: {e}"));
                    }
                }
                // Counter emitted inside the graph-present block so it reflects an
                // actual pass run (not the degenerate no-temporal-graph path).
                metrics::counter!("kremory.dream.acronym_recall.merges_applied_total")
                    .increment(acronym_recall_merges as u64);
            }
        }
        // Folded into DreamSummary.acronym_nickname_merges at the end of the chain
        // (ADR-063 §3 observability — surfaced to consumers, not just a counter).

        // ADR-046 Option E — Dream Pass 2: reclassify.
        // Runs AFTER Pass 0 so newly discovered types (from Pass 0) are available in
        // the entity type registry for the reclassify LLM prompt.
        // Pass ordering per DoD E7: Pass 0 commits → Pass 2 (reclassify) → Pass 3 (consolidation).
        // Pass 3 (consolidation) IS implemented — see `run_consolidation` below,
        // dispatched after consistency_check + canonicalize.
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
                        // TD-094: thread the resolved dream model id for capability detection.
                        model_id: dream_model_id,
                    },
                )
                .await
                {
                    Ok(reclassify_result) => {
                        // Aggregate entities_reclassified into DreamSummary (E8).
                        // SCOPE-001 (dream-phase-reconciliation-v2 §D1/§D3): accumulate
                        // into the local and FALL THROUGH — do NOT early-return. Passes
                        // ordered after reclassify (consistency_check, canonicalize per
                        // the D3 canonical ordering) dispatch below and must be reachable.
                        entities_reclassified = reclassify_result.entities_reclassified;
                        result.dream_warnings.extend(reclassify_result.warnings);
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

        // Dream Pass — consistency_check (dream-phase-reconciliation-v2 §D3,
        // ADR-047): re-verify entity types (embed-prefilter → LLM verify).
        // LLM-cost pass — gated by the per-pass opt-out `include_consistency_check`
        // (default true; mirrors `include_type_discovery`). NOTE: this is the
        // interim per-pass cost lever; the coarser Full/Light `DreamMode` gating
        // (what `Light` should mean at the enum level) is the separate SCOPE-002
        // concern deferred to Phase 5 — do NOT conflate the two. Ordered AFTER
        // reclassify (correct the types reclassify just assigned) and BEFORE
        // canonicalize (merges benefit from corrected types). Non-fatal.
        if opts.include_consistency_check {
            if let Some(tg) = self.memory.temporal_graph.as_ref() {
                match crate::core::dream::consistency_check::run_consistency_check(
                    &tg.conn,
                    crate::core::dream::consistency_check::RunConsistencyCheckParams {
                        embedder: self.memory.embedder.as_ref(),
                        llm: llm.as_ref(),
                        opts: crate::core::dream::consistency_check::ConsistencyCheckOpts {
                            // TD-094: thread the resolved dream model id as the
                            // verify model. `None` when unknown → verify.rs
                            // degrades to PromptOnly (unchanged from before).
                            verify_model_override: (!dream_model_id.is_empty())
                                .then(|| dream_model_id.to_string()),
                            ..Default::default()
                        },
                    },
                )
                .await
                {
                    Ok(cc_summary) => consistency_check_corrected = cc_summary.corrected,
                    Err(e) => {
                        tracing::warn!(
                            target: "kremory::dream::consistency_check",
                            error = %e,
                            "Dream consistency_check pass failed — skipping; dream phase result unaffected"
                        );
                        result
                            .dream_warnings
                            .push(format!("Dream consistency_check pass failed: {e}"));
                    }
                }
                // Counter emitted inside the graph-present block so it reflects an
                // actual pass run (not the degenerate no-temporal-graph path).
                metrics::counter!("kremory.dream.consistency_check_corrected_total")
                    .increment(consistency_check_corrected as u64);
            }
        }

        // Dream Pass — canonicalize (dream-phase-reconciliation-v2 §D3): merge
        // near-duplicate surface forms by embedding similarity above
        // L5_CANONICALIZATION_THRESHOLD. Deterministic (no LLM). Ordered LAST so
        // merges benefit from the corrected type distribution. Non-fatal.
        if let Some(tg) = self.memory.temporal_graph.as_ref() {
            let group_id = namespace_to_group_id(&ns);
            // TD-112 (`.ai-docs/tech-debt/tech-debt-register.md:2547`): thread the
            // embedder so the keeper's stored embedding is recomputed + persisted
            // after each merge instead of going stale.
            match crate::core::canonicalization::canonicalize_surface_forms_with_embedder(
                tg,
                crate::core::canonicalization::CanonicalizeSurfaceFormsParams {
                    group_id: &group_id,
                    threshold: crate::core::canonicalization::L5_CANONICALIZATION_THRESHOLD,
                    embedder: Some(self.memory.embedder.as_ref()),
                },
            )
            .await
            {
                Ok(report) => canonicalization_merges = report.merges_applied,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::canonicalize",
                        error = %e,
                        "Dream canonicalize pass failed — skipping; dream phase result unaffected"
                    );
                    result
                        .dream_warnings
                        .push(format!("Dream canonicalize pass failed: {e}"));
                }
            }
            // Counter emitted inside the graph-present block so it reflects an
            // actual pass run (not the degenerate no-temporal-graph path).
            metrics::counter!("kremory.dream.canonicalization_merges_total")
                .increment(canonicalization_merges as u64);
        }

        // Dream Pass — type_registry_collapse (ADR-063 spec §4, "Site #3"):
        // merge near-duplicate `entity_types` rows via description-cosine +
        // lexical pre-filter + LLM-verify band, remapping
        // `entities.entity_type_id` onto the keeper. Spike-gated per spec §8 —
        // gated by `include_type_registry_collapse` (default `false`). Ordered
        // LAST (after canonicalize) per spec §4.0: type collapse benefits from
        // a stable entity population that Pass 0/2/4/L5 have already finished
        // touching this cycle, and downstream queries against `entity_types`
        // see the collapsed registry as early as possible in the NEXT cycle
        // without perturbing the CURRENT cycle's other passes mid-flight.
        // Non-fatal: failure warns + continues.
        let mut type_registry_merges: usize = 0;
        if opts.include_type_registry_collapse {
            if let Some(tg) = self.memory.temporal_graph.as_ref() {
                let group_id = namespace_to_group_id(&ns);
                let arc_llm = crate::core::provider::ArcChatProvider::new(llm.clone());
                match crate::core::dream::type_registry_collapse::type_registry_collapse(
                    &arc_llm,
                    crate::core::dream::type_registry_collapse::TypeRegistryCollapseParams {
                        conn: &tg.conn,
                        group_id: &group_id,
                        embedder: Some(self.memory.embedder.as_ref()),
                        // TD-094-style threading: reuse the resolved dream model id.
                        model_id: dream_model_id,
                    },
                )
                .await
                {
                    Ok(collapse_report) => {
                        type_registry_merges = collapse_report.merges_applied;
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "kremory::dream::type_registry_collapse",
                            error = %e,
                            "Dream type_registry_collapse pass failed — skipping; dream phase result unaffected"
                        );
                        result
                            .dream_warnings
                            .push(format!("Dream type_registry_collapse pass failed: {e}"));
                    }
                }
                // Counter emitted inside the graph-present block so it reflects an
                // actual pass run (not the degenerate no-temporal-graph path).
                metrics::counter!("kremory.dream.type_registry_collapse.merges_applied_total")
                    .increment(type_registry_merges as u64);
            }
        }
        // Folded into DreamSummary.type_registry_merges at the end of the chain
        // (ADR-063 §4 observability — surfaced to consumers, not just a counter).

        // Dream CONSOLIDATION sub-phase (ADR-066) — graph-global cleanup ops
        // (supersession / archive / cross_episode / communities). Runs AFTER the
        // reconciliation chain (all merges/reclassifications settled). Gated by the
        // per-op `DreamOpts.include_*` flags. Since ADR-071 Item 1, cross_episode
        // defaults ON (in SHADOW — `cross_episode_dry_run: true`), so
        // `any_consolidation_enabled()` is TRUE by default and this block RUNS (the op
        // computes merge decisions + emits shadow telemetry, but fuses nothing);
        // supersession / archive / communities remain default-off. Non-fatal:
        // `unwrap_or_default()` folds a
        // dispatcher error into an all-zero summary + the counts land on the four
        // (already-existing) DreamSummary consolidation fields. The four ops
        // (P1-P4) are fully implemented (doc-drift fix, ADR-071 impl-spec
        // §"Pre-existing doc drift to fix in passing" — this comment previously said
        // "Ops are STUBS at P0 (return 0); P1-P4 fill them").
        let consolidation = if opts.any_consolidation_enabled() {
            if let Some(tg) = self.memory.temporal_graph.as_ref() {
                let group_id = namespace_to_group_id(&ns);
                crate::core::dream::consolidation::run_consolidation(
                    crate::core::dream::consolidation::RunConsolidationParams {
                        graph: tg,
                        group_id: &group_id,
                        opts: &opts,
                        model_id: dream_model_id,
                        // ADR-070 Fork 5: the orchestrator fires on_merge_proposed from
                        // this sink for each cross_episode merge decision.
                        sink: sink.as_ref(),
                    },
                )
                .await
                .unwrap_or_default()
            } else {
                crate::core::dream::consolidation::ConsolidationSummary::default()
            }
        } else {
            crate::core::dream::consolidation::ConsolidationSummary::default()
        };

        // Dream Pass — aliases, SECOND SWEEP (TD-203 D2). Deterministic, no LLM.
        //
        // The first sweep runs at position 2, BEFORE `acronym_nickname_recall`
        // (which CREATES `potential_alias` facts), `canonicalize` and
        // `run_consolidation` (whose merges re-point existing alias facts onto
        // the keeper). Everything those passes produce therefore missed the only
        // consumer and waited for a NEXT dream — and a caller that runs one
        // dream per namespace, as the LoCoMo harness does, never provides one.
        //
        // MEASURED on the shipped corpus: 42 alias facts, 0 resolved, across TEN
        // completed dreams. `melanie -> mel` was created by pass 3 at 19:11:44,
        // 84 seconds AFTER the pass that resolves aliases had already run
        // (`.context/kremory-http.log`); `graph_mutation_log` rows 2-3 show
        // canonicalize re-pointing alias facts at 19:12:00, likewise after.
        // Re-running the pass on a copy resolved 28 of the 42 in under a second.
        //
        // A SECOND sweep rather than MOVING the first: the first sweep's
        // position is load-bearing for its own stated reason (spec §D3 — an
        // entity about to be merged away must not be reclassified first), so
        // moving it would trade this defect for that one. The pass is
        // idempotent (it only ever invalidates) and cheap (0.72-17 ms measured
        // per namespace on a 1197-entity corpus), so running it twice is
        // strictly safer than running it once in the wrong place.
        if let Some(tg) = self.memory.temporal_graph.as_ref() {
            let group_id = namespace_to_group_id(&ns);
            match crate::core::disambiguation::resolve_pending_aliases(tg, &group_id).await {
                Ok(n) => {
                    aliases_resolved += n;
                    metrics::counter!("kremory.dream.aliases_resolved_total", "sweep" => "post")
                        .increment(n as u64);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::aliases",
                        error = %e,
                        "Dream aliases second sweep failed — skipping; dream phase result unaffected"
                    );
                    result
                        .dream_warnings
                        .push(format!("Dream aliases second sweep failed: {e}"));
                }
            }
        }

        // SCOPE-001 restructure gate (dream-phase-reconciliation-v2 Phase 1):
        // reaching this point proves control flowed PAST the reclassify pass
        // instead of early-returning inside its success arm. Passes wired in
        // Phase 2-3 (consistency_check, canonicalize per §D3) dispatch between
        // the reclassify block above and this line. This counter is the
        // mechanical regression guard for the early-return trap
        // (tests/dream_scope001_restructure.rs).
        metrics::counter!("kremory.dream.passes_continued_past_reclassify_total").increment(1);

        // Build the DreamSummary ONCE, at the end of the pass chain. Per-pass
        // counts accumulated in locals above are folded in here (§SCOPE-001).
        let mut summary = DreamSummary::from(result);
        summary.entities_reclassified = entities_reclassified;
        summary.aliases_resolved = aliases_resolved;
        summary.canonicalization_merges = canonicalization_merges;
        summary.acronym_nickname_merges = acronym_recall_merges;
        summary.type_registry_merges = type_registry_merges;
        summary.consistency_check_corrected = consistency_check_corrected;
        // ADR-066 CONSOLIDATION: fold the four op counts into the (already-existing)
        // DreamSummary consolidation fields, replacing their honest-zeros. Inert
        // (all zero) unless a consolidation op was enabled + fired.
        summary.communities_updated = consolidation.communities_updated;
        // D5: would-merge (decisions) vs merged (actual fusions) — in-band split.
        summary.cross_episode_would_merge = consolidation.cross_episode_merges;
        summary.cross_episode_merged = consolidation.cross_episode_merged;
        summary.supersessions_recorded = consolidation.supersessions_recorded;
        summary.facts_archived = consolidation.facts_archived;
        // D1b: per-op actually-ran signal (disambiguates disabled vs ran-empty).
        summary.consolidation_ops_ran = crate::facade::ConsolidationOpsRan {
            community: consolidation.ran_community_detection,
            cross_episode: consolidation.ran_cross_episode,
            archival: consolidation.ran_fact_archival,
            supersession_sweep: consolidation.ran_supersession_sweep,
        };
        summary.warnings.extend(consolidation.warnings);
        // TD-060 (ADR-071 §Item 4a step 6, Vera HIGH-1): propagate the budget flag
        // past the internal ConsolidationSummary — without this hop the flag is
        // set on a struct discarded 4 lines earlier and no consumer can read it.
        summary.budget_exhausted = consolidation.budget_exhausted;
        summary.duration_ms = dream_start.elapsed().as_millis() as u64;
        Ok(summary)
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

#[cfg(test)]
mod cross_episode_mode_tests {
    //! D3 (consumer-API hardening): `DreamRequest::cross_episode(mode)` maps the
    //! honest tri-state onto the two coupled `DreamOpts` bools.
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::CrossEpisodeMode;

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

    #[tokio::test]
    async fn cross_episode_mode_maps_to_opts_bools() {
        let mem = make_memory().await;

        let off = mem
            .dream()
            .cross_episode(CrossEpisodeMode::Off)
            .opts
            .expect("opts set");
        assert!(!off.include_cross_episode_merges, "Off → include=false");

        let shadow = mem
            .dream()
            .cross_episode(CrossEpisodeMode::Shadow)
            .opts
            .expect("opts set");
        assert!(shadow.include_cross_episode_merges, "Shadow → include=true");
        assert!(shadow.cross_episode_dry_run, "Shadow → dry_run=true");

        let apply = mem
            .dream()
            .cross_episode(CrossEpisodeMode::Apply)
            .opts
            .expect("opts set");
        assert!(apply.include_cross_episode_merges, "Apply → include=true");
        assert!(!apply.cross_episode_dry_run, "Apply → dry_run=false");
    }

    #[tokio::test]
    async fn cross_episode_preserves_other_opts_set_via_opts() {
        // Composing with a prior `.with_opts(...)` must overwrite ONLY the two
        // cross_episode fields, preserving every other knob.
        let mem = make_memory().await;
        let base = crate::memory::types::DreamOpts {
            include_community_detection: false,
            include_fact_archival: false,
            ..Default::default()
        };
        let composed = mem
            .dream()
            .with_opts(base)
            .cross_episode(CrossEpisodeMode::Apply)
            .opts
            .expect("opts set");
        assert!(composed.include_cross_episode_merges);
        assert!(!composed.cross_episode_dry_run);
        // Untouched knobs preserved from the base opts.
        assert!(!composed.include_community_detection, "community preserved");
        assert!(!composed.include_fact_archival, "archival preserved");
    }
}
