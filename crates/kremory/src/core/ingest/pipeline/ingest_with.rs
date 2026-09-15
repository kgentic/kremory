use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use metrics::{counter, histogram};

use crate::core::config::{ContentType, ResolutionStrategy};
use crate::core::contradiction::{DetectParams, TwoPoolDetector};
use crate::core::embed_prefix::{document_embed_text, query_embed_text};
use crate::core::extraction_window::ExtractionWindowSplitter;
use crate::core::graph::{
    EpisodeInsert, FactInsert, InsertEntityWithGroupParams, InsertEpisodicEdgeParams,
    InvalidateFactWithReasonParams, PriorEpisodesParams,
};
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
    ResolutionResult,
};
use crate::core::provider::{ChatProvider, EmbeddingProvider, TokenUsage};
use crate::core::resolver::{entity_name, normalize_name, CascadeResolver, UnionFind};
use crate::core::search::{FtsSearchFactsParams, SearchFilters, VectorSearchEntitiesNoCountParams};

use super::deferred_emissions::{flush_deferred_emissions, DeferredEmission};
use super::entity_rules;
use super::entity_upsert::{EntityUpsertParams, Phase1EntityState};
use super::fact_rules;
use super::fact_rules::{
    contradiction_labels, is_within_episode_duplicate, is_within_episode_multivalue,
    pool_b_sharing_an_entity, resolve_fact_endpoints, EpisodeTriple, FactEndpoints,
};
use super::forward_refs::forward_reference_names;
use super::pre_pinned::PrePinnedWriteParams;
use crate::core::ingest::{Engine, IngestionResult, SourceParams};

/// Bundled call-context parameters for [`Engine::ingest_with`] — args-as-object
/// to keep the function under clippy's too_many_arguments threshold. The generic
/// `extractor: &E` stays a lead positional param; these are the non-generic args.
pub struct IngestWithParams<'a> {
    pub text: &'a str,
    pub reference_time: Option<DateTime<Utc>>,
    /// The caller-DECLARED document anchor (`SourceRef::published_at`)
    /// ONLY — never `reference_time` / `occurred_at` / wall-clock. Forwarded
    /// unchanged into [`crate::core::intelligence::ExtractionContext::reference_time`]
    /// at every extraction call site below. See that field's doc comment for
    /// the VCR-fingerprint hazard this separation exists to avoid.
    pub declared_reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<&'a str>,
    pub content_type: Option<ContentType>,
    pub source_params: SourceParams,
}

/// Args-as-object for [`Engine::make_pinned_entity_recallable`] to keep the
/// function under clippy's too_many_arguments threshold (clippy.toml threshold 3).
pub(super) struct PinnedEntityRecall<'a> {
    /// Entity id (== the literal pinned subject/object text).
    pub(super) id: &'a str,
    /// Namespace the entity + its episodic edge live in (composite-FK scope).
    pub(super) group_id: Option<&'a str>,
    /// Source episode to attribute the entity to.
    pub(super) episode_id: i64,
}

/// Bundled parameters for [`Engine::block_resolution_candidates`] — args-as-object
/// to keep the function under clippy's too_many_arguments threshold. All fields
/// share the `'a` borrow of the pre-batch existing-entity slice so the returned candidate refs
/// tie back to it.
pub(super) struct BlockCandidatesParams<'a> {
    pub(super) extracted: &'a ExtractedEntity,
    pub(super) existing_entities: &'a [crate::core::schema::Entity],
    pub(super) group_id: Option<&'a str>,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Select the bounded set of existing entities that a
    /// newly-extracted entity should be resolved against, instead of comparing
    /// it to EVERY existing entity. Without this, each new entity fanned out to
    /// O(existing) LLM `ResolutionVerdict` calls (92% of ingest LLM calls on a
    /// growing graph).
    ///
    /// Candidates = **exact normalized-name matches** (cheap, no LLM — REQUIRED
    /// because it catches stub→real promotion; stubs carry `embedding=None` so
    /// the ANN arm can never surface them) **UNION the embedding-ANN top-`k`**
    /// nearest existing entities. Anything else is semantically far and defaults
    /// to `Different` with zero LLM. Dream-phase L5 canonicalization is the
    /// exhaustive completeness backstop for any true match this misses.
    ///
    /// Back-compat guards (both return the full exhaustive list, i.e. the exact
    /// earlier behaviour):
    /// - the group has ≤`k` existing entities (blocking only pays off past `k`;
    ///   keeps the whole small-graph test suite behaviour-identical), or
    /// - no usable embedding is available (null/failing embedder, or the ANN
    ///   query errors) — blocking needs embeddings; without them, behave as before.
    ///
    /// Returned refs borrow `existing_entities`, keeping the candidate universe
    /// == the pre-batch existing set so the downstream stub-check + union-find
    /// (which reference `existing_entities` by id) stay valid.
    pub(super) async fn block_resolution_candidates<'a>(
        &self,
        params: BlockCandidatesParams<'a>,
    ) -> Vec<&'a crate::core::schema::Entity> {
        let BlockCandidatesParams {
            extracted,
            existing_entities,
            group_id,
        } = params;
        let k = self.config.resolution_block_k;

        // Guard 1: no-op on small graphs — exhaustive comparison at/below k.
        if existing_entities.len() <= k {
            return existing_entities.iter().collect();
        }

        let norm = normalize_name(&extracted.name);
        let mut seen: HashSet<&str> = HashSet::new();
        let mut candidates: Vec<&crate::core::schema::Entity> = Vec::new();

        // Arm 1 — exact normalized-name matches (all N, cheap, no LLM).
        for e in existing_entities {
            if normalize_name(entity_name(e)) == norm && seen.insert(e.id.as_str()) {
                candidates.push(e);
            }
        }

        // Arm 2 — embedding-ANN top-k. Guard 2: null/failing embedder → full list.
        // This is a QUERY against the SAME `entities` vector index that
        // stores document-prefixed writes (see `set_entity_embedding` call sites
        // below) — must use `query_embed_text`, not the raw name, or a flipped
        // knob would compare an unprefixed probe against a document-prefixed
        // corpus (the exact mixed-index footgun the embed-prefix correctness note
        // warns against).
        let embedding = match self
            .embedder
            .embed(&query_embed_text(
                &extracted.name,
                self.config.search.embed_task_prefix_enabled,
            ))
            .await
        {
            Ok(v) if !v.is_empty() && v.iter().any(|x| *x != 0.0) => v,
            _ => return existing_entities.iter().collect(),
        };
        let filters = SearchFilters {
            group_ids: group_id.map(|g| vec![g.to_string()]).unwrap_or_default(),
            // This `false` currently has no effect — entity search never
            // reads `exclude_expired`. Kept (not deleted) as the documented
            // expression of real intent (include expired-entity merge
            // candidates during resolution) for whoever wires this up to honour.
            exclude_expired: false,
        };
        // `_no_count` variant: avoid the access_count bump the public search applies.
        let min_cosine = self.config.resolution_min_cosine;
        let hit_ids: HashSet<String> = match self
            .graph
            .vector_search_entities_no_count(VectorSearchEntitiesNoCountParams {
                query_embedding: &embedding,
                limit: k,
                filters: &filters,
            })
            .await
        {
            Ok(hits) => {
                // Auto-different cosine floor. `vector_search` returns
                // score = -cosine_distance = cosine_sim - 1, so cosine_sim = score + 1.
                // Drop candidates below the floor WITHOUT an LLM verdict (the model
                // would say "different" for embedding-far pairs anyway). min_cosine=0.0
                // keeps everything → pure P0. Can only reduce merges, never create a
                // false one; dream L5 canonicalization is the completeness backstop.
                let total = hits.len();
                let kept: HashSet<String> = hits
                    .into_iter()
                    .filter(|h| (h.score as f32) + 1.0 >= min_cosine)
                    .map(|h| h.item.id)
                    .collect();
                metrics::counter!("kremory.resolution.cosine_floor_skipped_total")
                    .increment((total - kept.len()) as u64);
                kept
            }
            // ANN failed even with a real embedding → conservative exhaustive list.
            Err(_) => return existing_entities.iter().collect(),
        };
        // Intersect ANN hits with the pre-batch existing set (universe invariant).
        for e in existing_entities {
            if hit_ids.contains(&e.id) && seen.insert(e.id.as_str()) {
                candidates.push(e);
            }
        }

        candidates
    }

    /// Full pipeline with a caller-supplied extractor.
    /// Any type implementing `EntityExtractor` can be used (a built-in extractor
    /// or a custom BYOE impl).
    // Substrate primitive; the consumer-facing surface is the kremory::Memory facade.
    // Generic `extractor: &E` stays a lead positional param;
    // the remaining call-context args are bundled into `IngestWithParams`.
    #[tracing::instrument(
        name = "kremory.ingest",
        skip(self, extractor, params),
        fields(
            kremory.operation = "ingest",
        )
    )]
    pub async fn ingest_with<E: EntityExtractor>(
        &self,
        extractor: &E,
        params: IngestWithParams<'_>,
    ) -> crate::core::error::Result<IngestionResult> {
        let IngestWithParams {
            text,
            reference_time,
            declared_reference_time,
            group_id,
            content_type,
            source_params,
        } = params;
        let ingest_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);
        let token_usage = TokenUsage::default();

        // 1. Store episode (namespace-scoped via group_id).
        //    source_id / source_uri / recorded_at from SourceParams are written to the
        //    Migration 007 columns so that recall_by_source_id can find this episode.
        let mut episode = EpisodeInsert::new(text, ref_time).source_type("ingest");
        if let Some(source_id) = source_params.source_id.as_deref() {
            episode = episode.source_id(source_id);
        }
        if let Some(source_uri) = source_params.source_uri.as_deref() {
            episode = episode.source_uri(source_uri);
        }
        if let Some(recorded_at) = source_params.recorded_at {
            episode = episode.recorded_at(recorded_at);
        }
        let episode_id = self
            .graph
            .insert_episode_with_group(episode, group_id)
            .await?;

        // Prior-turn replay. Fetch the preceding turns of THIS
        // conversation so the extractor can resolve references in `text`.
        //
        // Ordering matters and is not incidental: the episode row was inserted
        // immediately above, so the query MUST bound on `id < episode_id` or
        // the episode replays itself.
        //
        // Inert unless the caller threaded a conversation. An un-tagged
        // `remember()` is handed a random uuid source id
        // (`facade/remember.rs`), so no prior row can match and this returns
        // empty — which renders nothing and leaves the prompt (and therefore
        // every committed VCR cassette) byte-identical.
        let prior_turns: Vec<String> = match source_params.source_id.as_deref() {
            Some(source_id) => {
                self.graph
                    .prior_episodes_for_source(PriorEpisodesParams {
                        source_id,
                        group_id,
                        before_id: episode_id,
                        limit: self.config.prior_turn_replay_depth,
                    })
                    .await?
            }
            None => Vec::new(),
        };
        if !prior_turns.is_empty() {
            counter!("kremory.replay.ingest_with_replayed_total").increment(1);
            tracing::debug!(
                episode_id,
                prior_turns = prior_turns.len(),
                "kremory.replay.prior_turns_attached"
            );
        }

        // Dense episode arm — embed + store the episode's embedding when
        // the dense arm is enabled (no-op / byte-identical when off).
        // Capture the outcome — see `IngestionResult::dense_embedded`.
        #[cfg(feature = "content-search")]
        let dense_embedded = self.maybe_embed_episode(episode_id, text).await;
        #[cfg(not(feature = "content-search"))]
        let dense_embedded = true;

        // 1b. Pin caller-pre-extracted facts BEFORE Phase 2 LLM.
        //
        // Caller-supplied triples enter the graph first; the LLM Phase 2 extraction
        // path uses `try_insert_fact` which silently swallows the resulting
        // `Err(Duplicate)` on the same `content_hash`, so caller wins via
        // pre-write ordering. Intra-set duplicates (caller passes the same
        // triple twice in their own set) also dedup cleanly via
        // `try_insert_fact_with_group`.
        let (pinned_fact_ids, mut embedding_failures): (Vec<i64>, Vec<String>) = self
            .write_pre_pinned_facts(PrePinnedWriteParams {
                pre_pinned_facts: &source_params.pre_pinned_facts,
                episode_id,
                group_id,
            })
            .await;

        // 1c. `skip_extraction` early return.
        //
        // When the caller has set `RememberRequest::skip_extraction()` (or the
        // legacy `SubmitOpts.enrich_per_episode = false` once the engine_handle
        // gate is wired), bail after caller-pin completion. Episode + embedding
        // (if pre-pinned) + caller-pre-extracted facts are persisted; Phase 2
        // LLM extraction is skipped entirely. Useful for bulk-import workloads
        // where caller already has high-confidence data.
        if source_params.skip_extraction {
            metrics::counter!("kremory.skip_extraction.invoked_total").increment(1);
            tracing::info!(
                episode_id,
                pinned_fact_count = pinned_fact_ids.len(),
                "kremory.ingest.skip_extraction_early_return"
            );
            histogram!("rql.ingest.skip_extraction_ms")
                .record(ingest_start.elapsed().as_secs_f64() * 1000.0);
            return Ok(IngestionResult {
                episode_id,
                upserted_entities: Vec::new(),
                inserted_fact_ids: pinned_fact_ids,
                invalidated_fact_ids: Vec::new(),
                merged_entities: Vec::new(),
                token_usage,
                stub_entities_inserted: 0,
                dense_embedded,
                embedding_failures,
            });
        }

        // ── Fire-site: on_stage_change(Extracting) ──────────────────────────────
        // Re-establishes the 737e152 verify_stage Fire-site 1 on the UNIFIED
        // inline/background extraction routine (`ingest_with`) that the public
        // `Memory::remember(...).with_event_sink(...)` consumer journey actually
        // drives (engine_handle::graph_ingest_episode → engine.ingest →
        // ingest_with). The sink is carried on `source_params.sink` as the
        // core-layer `IngestEventSink` (see SourceParams::sink doc-comment) and
        // is fired synchronously per the sync-inline contract. Triple-emit:
        // sink + counter + tracing. episode_id is a
        // tracing field only, NEVER a metric label.
        let sink = source_params.sink.as_deref();
        if let Some(s) = sink {
            s.on_stage_change(crate::core::error::IngestStatus::Extracting);
        }
        metrics::counter!(
            "kremory.sink.stage_transition_total",
            "from" => "Pending",
            "to" => "Extracting"
        )
        .increment(1);
        tracing::info!(episode_id, "kremory.sink.stage_change.extracting");

        // ── State-machine invariant: Extracting → Failed on ANY error ───────────
        // Canonical pattern: `core/background/verify_stage.rs` fires `Failed`
        // synchronously BEFORE every `Err` propagates (the function's own
        // doc-comment invariant). The previous terminal `match
        // phase_result` only fired `Failed` for errors that broke the inner
        // `'phases` block. Every fallible `?`-step AFTER the `Extracting` fire but
        // OUTSIDE `'phases` — registry load/seed, `ExtractionWindowSplitter` work,
        // `extractor.extract(..).await?`, the in-`'phases` `llm_for_detector?`,
        // `begin_immediate_if_needed().await?`, and `outer_guard.commit().await?` —
        // returned early WITHOUT firing `Failed`; the consumer saw `Extracting`
        // then silence.
        //
        // Cause-fix: capture the WHOLE post-`Extracting`
        // body in one `async` block that returns `Result<IngestionResult>` (every
        // `?` inside resolves to the block's `Err`, not a function return), then
        // fire `Failed` exactly once at the single exit before re-propagating. No
        // error is swallowed — `Failed` fires THEN the `Err` is returned via `?`.
        let fire_failed = |e: &crate::core::error::Error| {
            if let Some(s) = sink {
                s.on_stage_change(crate::core::error::IngestStatus::Failed(e.to_string()));
            }
            metrics::counter!(
                "kremory.sink.stage_transition_total",
                "from" => "Extracting",
                "to" => "Failed",
                "arm" => "post_extracting"
            )
            .increment(1);
            tracing::error!(episode_id, error = %e, "kremory.sink.stage_change.failed");
        };

        let ingest_outcome: crate::core::error::Result<IngestionResult> = async {
        // 2. Slice into LLM-extraction-prompt windows (no-op for normally-sized episodes;
        //    see core/extraction_window.rs module docstring for kind-2 semantics).
        let splitter = ExtractionWindowSplitter::new(self.config.extraction_window.clone());
        let chunks = splitter.split(text, &content_type);

        let chunk_count = chunks.len();
        histogram!("rql.ingest.chunk_count").record(chunk_count as f64);
        tracing::info!(chunk_count, "kremory.ingest.chunked");

        // 2b. L2: resolve EntityTypeRegistry for this group BEFORE extraction so
        //     registry specs can be injected into extraction prompts (Phase 3).
        //     This is a cheap DB read (a handful of rows); loading early does not
        //     duplicate the work at step 4 — step 4 re-uses the same `registry`
        //     binding for L3 validation.
        //
        //     Hybrid registry: if caller supplied a per-call override, use it.
        //     - First-call persistence: if DB is empty for this group_id, persist
        //       the override so subsequent calls without override see the vocabulary.
        //     - Ephemeral: if DB already has rows, use override for this call only.
        let effective_gid = group_id.unwrap_or("default");

        // 2a. Migration 010 lazy-seed: ensure the default vocabulary is
        //     present for `effective_gid` before any registry-dependent work
        //     (L2 prompt, L3 validation). Idempotent: no-ops once seeded.
        //     This catches namespaces created AFTER Migration 010 ran at boot.
        crate::core::entity_types::ensure_default_types_seeded(&self.graph.conn, effective_gid)
            .await?;

        let registry = self
            .resolve_entity_type_registry(source_params.entity_types_override.as_deref(), effective_gid)
            .await?;

        // 2c. L4': fetch top-N existing entities for prompt-time injection.
        //
        //     The extraction prompt lists the canonical entities already in the
        //     knowledge graph so the LLM reuses exact names rather than inventing
        //     variant spellings ("Alice Johnson" vs "Alice J.").  We fetch the
        //     top-50 by access_count (most-recently-used) as context.
        //
        //     This is a separate, read-only fetch from step 4 (which fetches all
        //     entities for resolver dedup).  The two fetches serve different
        //     purposes and are not merged — step 4 deduplication needs all
        //     entities; L4' injection needs only the top-N most relevant.
        let existing_entities_for_prompt: Vec<(String, String)> = {
            const L4_PRIME_INJECT_LIMIT: usize = 50;
            let raw = match group_id {
                Some(gid) => self.graph.list_entities_in_group(gid).await?,
                None => self.graph.list_entities().await?,
            };
            // Ranking, truncation and the display-name-over-id choice are a pure
            // rule with its own tests — see `entity_rules::top_n_display_names`.
            entity_rules::top_n_display_names(raw, L4_PRIME_INJECT_LIMIT)
        };

        // 3. Extract from all chunks, merge results.
        // known_entities grows with each iteration so subsequent chunks receive
        // the entities already found in earlier chunks as context.
        //
        // Derive allowed_entity_types from the live registry
        // (loaded above) rather than the config snapshot taken at engine construction.
        // This closes the self-learning loop: Pass 0 writes new types to the registry;
        // the next ingest sees them immediately without an engine rebuild.
        let allowed_entity_types_live: Vec<String> =
            registry.specs().iter().map(|s| s.name.clone()).collect();

        let mut all_entities: Vec<ExtractedEntity> = Vec::new();
        let mut all_facts: Vec<ExtractedFact> = Vec::new();

        // TD-NEW-A Lane C: optionally overlap the per-chunk extraction LLM
        // calls. `extraction_concurrency == 1` keeps the byte-identical
        // pre-change sequential path (each chunk's prompt sees earlier chunks'
        // entities via a growing `known_entities`) — the clean rollback + the
        // pure-W1 baseline for benchmark attribution. `> 1` uses `buffered(N)`
        // (order-PRESERVING, NOT `buffer_unordered`, per the TD-NEW-B
        // determinism rule); extraction is read-only with no cross-chunk graph
        // dependency, so overlap is safe, but `known_entities` becomes
        // window-local (empty) — the within-ingest cross-chunk hint is dropped,
        // which dream L5 canonicalization backstops (recall no-regression MUST
        // be measured, per the W2 plan).
        {
            use futures::stream::StreamExt as _;
            let concurrency = self.config.extraction_concurrency.max(1);
            if concurrency == 1 {
                for chunk in &chunks {
                    let ctx = ExtractionContext {
                        allowed_entity_types: &allowed_entity_types_live,
                        allowed_edge_types: &self.config.allowed_edge_types,
                        known_entities: &all_entities,
                        excluded_entity_types: &self.config.excluded_entity_types,
                        content_type: content_type.clone(),
                        registry_specs: registry.specs(),
                        existing_graph_entities: &existing_entities_for_prompt,
                        arm_budget_ms: self.config.extraction_arm_budget_ms,
                        model: self.model.as_deref(),
                        reference_time: declared_reference_time,
                        prior_turns: &prior_turns,
                    };
                    let result = extractor.extract(chunk.as_str(), &ctx).await?;
                    all_entities.extend(result.entities);
                    all_facts.extend(result.facts);
                    if let Some(ref auditor) = self.oov_auditor {
                        let audit_adds = auditor.audit(chunk, &all_entities);
                        all_entities.extend(audit_adds);
                    }
                }
            } else {
                // One shared ctx serves every chunk (identical once
                // known_entities is window-local).
                let shared_ctx = ExtractionContext {
                    allowed_entity_types: &allowed_entity_types_live,
                    allowed_edge_types: &self.config.allowed_edge_types,
                    known_entities: &[],
                    excluded_entity_types: &self.config.excluded_entity_types,
                    content_type: content_type.clone(),
                    registry_specs: registry.specs(),
                    existing_graph_entities: &existing_entities_for_prompt,
                    arm_budget_ms: self.config.extraction_arm_budget_ms,
                    model: self.model.as_deref(),
                    reference_time: declared_reference_time,
                    prior_turns: &prior_turns,
                };
                // Build the futures eagerly via `Iterator::map` (monomorphised
                // at the concrete chunk lifetime) rather than `StreamExt::map`
                // (which would require the borrowing future to be general over
                // ANY item lifetime — an HRTB the `extract<'a>` signature can't
                // satisfy inside the enclosing `tokio::spawn`).
                let extract_futures: Vec<_> = chunks
                    .iter()
                    .map(|chunk| extractor.extract(chunk.as_str(), &shared_ctx))
                    .collect();
                let extraction_results: Vec<ExtractionResult> =
                    futures::stream::iter(extract_futures)
                        .buffered(concurrency)
                        .collect::<Vec<crate::core::error::Result<ExtractionResult>>>()
                        .await
                        .into_iter()
                        .collect::<crate::core::error::Result<Vec<_>>>()?;

                // Fold results back in chunk order (deterministic). The OOV
                // audit runs sequentially per chunk so its "terms this chunk
                // missed given everything found so far" semantics are preserved.
                for (chunk, result) in chunks.iter().zip(extraction_results) {
                    all_entities.extend(result.entities);
                    all_facts.extend(result.facts);
                    if let Some(ref auditor) = self.oov_auditor {
                        let audit_adds = auditor.audit(chunk, &all_entities);
                        all_entities.extend(audit_adds);
                    }
                }
            }
        }

        // scan_proper_nouns DISABLED pending
        // scanner rework — scanner hardcodes label="Entity" instead of routing
        // candidates through LLM classification (the original two-call design
        // per spike_phase3_scanner_vs_pipeline.rs, never finished). Disabling
        // gives us pure-LLM extraction matching Graphiti/Cognee/LightRAG.
        // Re-enable after wiring the second LLM call OR delete entirely.
        //
        // let proper_noun_candidates = text_utils::scan_proper_nouns(text, &all_entities);
        // all_entities.extend(proper_noun_candidates);

        // 3b. Pre-mutation intra-batch duplicate scan.
        //
        // Original behaviour: FATAL `Err(IntraBatchDuplicate)` on duplicate names.
        // The intent was to surface malformed batches submitted by human callers
        // rather than silently dropping rows.
        //
        // Why we softened it in v0.1.4: kremory's pipeline ingests LLM-extracted
        // entities, and noisy extractors (smaller local models — llama3.2:3b,
        // gemma4-e2b — observed emitting `'car'` / `'VerbatimString'`
        // multiple times in a single batch) cannot be expected to dedupe their
        // own output. Treating that as fatal forced operators to pick a
        // hardened-extractor model (qwen2.5:14b+) rather than letting the
        // substrate accept noisy upstream input gracefully.
        //
        // New behaviour: emit a `tracing::warn!` with the duplicated names and
        // continue — the dedup below removes the offenders, exactly one row
        // per normalized name lands in the DB. Human callers who want strict
        // dedup-rejection can layer that contract on top of `remember_batch`
        // at the application layer.
        {
            let dup_names = entity_rules::duplicate_extracted_names(&all_entities);
            if !dup_names.is_empty() {
                tracing::warn!(
                    target: "kremory.ingest",
                    dup_count = dup_names.len(),
                    dup_names = ?dup_names,
                    "extractor emitted duplicate entity names — deduplicating silently \
                     (post-v0.1.4 behaviour; previously FATAL IntraBatchDuplicate)"
                );
            }
        }

        // Deduplicate extracted entities by normalized name.
        // sort + dedup_by because dedup_by only removes consecutive duplicates.
        all_entities.sort_by_key(|e| normalize_name(&e.name));
        all_entities.dedup_by(|a, b| normalize_name(&a.name) == normalize_name(&b.name));

        // 4. Resolve entities against existing graph (namespace-scoped dedup)
        let existing_entities = match group_id {
            Some(gid) => self.graph.list_entities_in_group(gid).await?,
            None => self.graph.list_entities().await?,
        };

        // registry was loaded above at step 2b — reused here for L3 validation
        // (label → integer entity_type_id at insert time).

        let llm_for_resolver =
            self.llm
                .as_ref()
                .ok_or_else(|| crate::core::error::Error::LlmRequired {
                    method: "ingest_with",
                    hint: "wire an LLM via Memory::open(…).with_llm(…) to enable entity \
                       resolution; or use .with_facts(…) to pin triples without LLM",
                })?;
        let resolver = CascadeResolver::new(
            Arc::clone(llm_for_resolver),
            self.config.minhash.clone(),
            self.config.entropy.clone(),
        )
        .with_model(self.model.clone());

        // ── Pass 1 + Pass 2: batched entity resolution ─────────────────────────
        // Runs BEFORE the write transaction — both passes
        // only read the frozen `existing_entities` snapshot loaded above and
        // do zero DB writes, so they execute in the pre-transaction window
        // (shrinking lock-hold vs the pairwise path, which resolves inline
        // inside the transaction's entity loop below).
        //
        // Pass 1 (deterministic, no LLM): for each extracted entity, run the
        // existing candidate block + the cheap Tier-1/Tier-2 tiers.
        // A hit pre-resolves the entity; a miss adds it to the ambiguous
        // worklist. Pass 2 batches the ambiguous remainder into windowed
        // structured-output calls (`resolver_batched::resolve_batched`).
        //
        // `batched_resolved` maps `normalize_name(entity.name) -> existing
        // entity id` for every CONFIDENT resolution (deterministic or
        // batched-LLM); entities absent from this map are NEW. Only built
        // when the strategy is `Batched` — the `Pairwise` arm below resolves
        // inline exactly as before this batched-resolution mechanism.
        let mut batched_resolved: HashMap<String, String> = HashMap::new();
        if self.config.resolution_strategy == ResolutionStrategy::Batched {
            let mut ambiguous: Vec<crate::core::resolver_batched::AmbiguousEntity<'_>> =
                Vec::new();

            for (idx, extracted) in all_entities.iter().enumerate() {
                let candidates = self
                    .block_resolution_candidates(BlockCandidatesParams {
                        extracted,
                        existing_entities: &existing_entities,
                        group_id,
                    })
                    .await;
                metrics::counter!("kremory.resolution.candidates_considered_total")
                    .increment(candidates.len() as u64);
                metrics::counter!("kremory.resolution.blocked_out_total")
                    .increment(existing_entities.len().saturating_sub(candidates.len()) as u64);

                let mut deterministic_hit: Option<String> = None;
                for existing in candidates.iter().copied() {
                    if resolver.resolve_deterministic(extracted, existing)
                        == Some(ResolutionResult::Same)
                    {
                        deterministic_hit = Some(existing.id.clone());
                        break;
                    }
                }

                match deterministic_hit {
                    Some(existing_id) => {
                        batched_resolved.insert(normalize_name(&extracted.name), existing_id);
                    }
                    // An entity whose OWN block is empty
                    // (e.g. cold-store first ingest — `existing_entities` empty)
                    // can never merge (the map-back own-block guard would reject
                    // any assignment anyway), so it must NOT enter the batch —
                    // otherwise we'd fire an LLM call against an empty pool and
                    // REGRESS call-count vs Pairwise (which makes 0 calls here).
                    // Zero candidates → NEW for free, matching Pairwise exactly.
                    None if !candidates.is_empty() => {
                        ambiguous.push((idx, extracted, candidates))
                    }
                    None => {}
                }
            }

            let batched_from_llm = crate::core::resolver_batched::resolve_batched(
                crate::core::resolver_batched::ResolveBatchedParams {
                    llm: llm_for_resolver.as_ref(),
                    model: self.model.as_deref(),
                    ambiguous: &ambiguous,
                    max_window: self.config.resolution_batch_max_entities,
                },
            )
            .await;
            for (idx, existing_id) in batched_from_llm {
                batched_resolved.insert(normalize_name(&all_entities[idx].name), existing_id);
            }
        }

        // ── Bug E: open a single outer transaction wrapping Phase 1 (entities +
        // stubs) and Phase 2 (facts).  All inner graph methods call
        // begin_immediate_if_needed() which is a no-op when a transaction is
        // already open, so nesting is safe.
        let outer_guard = self.graph.begin_immediate_if_needed().await?;

        // All mutable state that accumulates across the two phases lives here.
        // These are declared before the match so they can be moved into the Ok
        // branch cleanly — no partial-move issues.
        let mut union_find = UnionFind::new();
        let mut upserted_entities: Vec<String> = Vec::new();
        let mut merged_entities: Vec<(String, String)> = Vec::new();
        let mut name_to_id: HashMap<String, String> = HashMap::new();
        let mut inserted_fact_ids: Vec<i64> = Vec::new();
        let mut invalidated_fact_ids: Vec<i64> = Vec::new();
        let mut stub_entities_inserted: usize = 0;
        // Sink callbacks and durable-write counters captured
        // during the transaction and replayed only after it commits. Declared alongside
        // the other cross-phase accumulators so it survives the `'phases` block.
        let mut deferred: Vec<DeferredEmission> = Vec::new();

        // Forward-reference stub names awaiting an embedding, flushed POST-COMMIT
        // beside `flush_deferred_emissions` for the same reason: embedding is a
        // network round-trip, and holding the write transaction open across it would
        // serialise every other writer behind an LLM call. Collected here so it
        // survives the `'phases` block.
        let mut stub_names_to_embed: Vec<String> = Vec::new();

        // Helper closure-like block that returns Result<()> so we can commit or
        // rollback in one place.  We use a labelled block instead of an async
        // closure to avoid capture/lifetime complexity.
        let phase_result: crate::core::error::Result<()> = 'phases: {
            // ── Bug E §1: pre-scan all_facts for forward-reference entity names ─────
            // Collect every entity name appearing as a subject or object in facts.
            // Any name NOT already mapped from the extraction list is a forward
            // reference — insert it as an UNKNOWN stub so the fact loop can resolve
            // it without producing a dangling subject_id.
            // Names the facts reference that no extracted entity accounts for.
            // The RULE is now a pure function with its own tests (`forward_refs.rs`);
            // what stays here is the INSERT that acts on it.
            for norm_name in forward_reference_names(&all_entities, &all_facts) {
                // entities.id is a sole TEXT PK pre-migration-004.
                // Post-migration-004: composite PK (id, group_id) closes the bypass surface.
                // Stub INSERT uses INSERT OR IGNORE — cross-namespace name collision silently
                // skips stub creation. Single-namespace use only for v0.1.1.
                //
                // Strategy: attempt insert_entity_with_group; if the entity already exists
                // (UNIQUE constraint error), that is fine — a real row is present.
                // `properties["name"]` is REQUIRED for FTS findability.
                // `entities_fts` indexes `properties` ONLY — `entities_fts.label` is
                // empty post-Migration-009 — so a stub without a name token is
                // invisible to the FTS seed arm. The `with_facts` pinned path
                // established exactly this (`:458-466`: *"A bare `{"stub": false}`
                // stub carried no name token → recall returned 0"*), but the
                // EXTRACTION forward-reference path never received the same fix.
                //
                // So these stubs were unreachable by BOTH retrieval arms: no
                // `properties["name"]` (no FTS) and no embedding (no dense — see the
                // post-commit embed below). It also feeds `graph_search`'s
                // original-case `entity_name` render.
                let stub_props = serde_json::json!({
                    "name": norm_name,
                    "stub": true,
                    "source": "forward_reference",
                });
                match self
                    .graph
                    .insert_entity_with_group(InsertEntityWithGroupParams {
                        id: &norm_name,
                        entity_type_id: 0,
                        properties: stub_props,
                        group_id,
                    })
                    .await
                {
                    Ok(()) => {
                        // Newly inserted stub.
                        name_to_id.insert(norm_name.clone(), norm_name.clone());
                        stub_entities_inserted += 1;
                        // Both counters claim a persisted stub row, so they
                        // are replayed after the outer commit, never here.
                        deferred.push(DeferredEmission::StubInserted);
                        // A stub written here carries NO embedding, so it is absent
                        // from the ANN index while still being counted by
                        // `plan_index_fetch`'s `namespace_rows` — which is exactly
                        // the shortfall `search.rs:353-383` reports. Measured
                        // on LoCoMo conv-26: 25 of 67 entities (37%) had a NULL
                        // embedding and were unreachable by dense retrieval, and the
                        // filtered-ANN arm under-filled (requested=10 delivered=7)
                        // even with the group filter matching every row (see
                        // `search.rs:353-383`'s namespace-shortfall reporting).
                        //
                        // Queued, not embedded inline: see `stub_names_to_embed`.
                        stub_names_to_embed.push(norm_name.clone());
                        tracing::warn!(
                            target: "kremory.ingest.stub",
                            name = %norm_name,
                            "inserted UNKNOWN stub for forward reference"
                        );
                    }
                    Err(_) => {
                        // Entity already exists (real or from a previous batch) — use it.
                        name_to_id.insert(norm_name.clone(), norm_name.clone());
                    }
                }
            }

            // Resolved entity ids that received a "mention" presence edge in the
            // entity loop below. Populated at EACH of the three mention-write
            // sites (merge/promote-stub, L4-merge, new-insert) immediately after a
            // successful insert — NOT at a single consolidated point, because the
            // L4-merge branch `continue`s and would skip a consolidated insert
            // The fact loop consults this to keep presence
            // single-owned: an extracted entity already linked here MUST NOT also
            // receive an "object" edge (the duplicate `(episode_id, entity_id)`
            // bug). Keyed on the RESOLVED id (post merge/alias) so it is robust to
            // disambiguation, unlike a surface-name set.
            let mut entity_loop_ids: HashSet<String> = HashSet::new();

            // ── Phase 1: entity loop (Bug B snippet + Bug A episodic_edge) ──────────
            // The per-entity upsert/merge pass. 448 lines of it used to sit inline
            // here, threading six pieces of mutable state through a single loop body
            // (TD-045). `break 'phases Err(e)` becomes an early `Err` return inside
            // the extracted method and the same labelled break at THIS call site, so
            // an error still abandons the whole phase block exactly as before.
            if let Err(e) = self
                .upsert_extracted_entities(EntityUpsertParams {
                    all_entities: &all_entities,
                    existing_entities: &existing_entities,
                    batched_resolved: &batched_resolved,
                    resolver: &resolver,
                    registry: &registry,
                    group_id,
                    episode_id,
                    text,
                    state: Phase1EntityState {
                        union_find: &mut union_find,
                        upserted_entities: &mut upserted_entities,
                        merged_entities: &mut merged_entities,
                        name_to_id: &mut name_to_id,
                        entity_loop_ids: &mut entity_loop_ids,
                        deferred: &mut deferred,
                    },
                })
                .await
            {
                break 'phases Err(e);
            }

            // ── Phase 2: detect contradictions and store facts ───────────────────────
            //
            // Contradiction detection is DEFAULT-ON again.
            // It was default-OFF for ~4h while it treated every
            // predicate as functional and SUPERSEDED SET-VALUED FACTS (81 of 1,021
            // facts invalidated over 8 LongMemEval sessions, ≥31% provably
            // multi-valued — a festival's 2nd..6th performer superseded by the
            // last). The cause was the PROMPT, and the coexistence rewrite measured
            // 7/8 destroyed -> 0/8. Disable with KREMORY_CONTRADICTION_DETECTION=0.
            //
            // When disabled, facts are still STORED by this loop and only the
            // invalidation is skipped — a loss of capability, never of data.
            //
            // Building the detector conditionally also removes a spurious
            // `LlmRequired` error: with detection off there is nothing here that
            // needs an LLM, so a caller pinning triples via `.with_facts(…)`
            // should not be forced to wire one for a pass that will not run.
            let detector = if self.config.contradiction_detection_enabled {
                let llm_for_detector =
                    self.llm
                        .as_ref()
                        .ok_or_else(|| crate::core::error::Error::LlmRequired {
                            method: "ingest_with",
                            hint: "wire an LLM via Memory::open(…).with_llm(…) to enable \
                               contradiction detection; or use .with_facts(…) to pin \
                               triples without LLM",
                        })?;
                Some(
                    TwoPoolDetector::new(Arc::clone(llm_for_detector))
                        .with_model(self.model.clone()),
                )
            } else {
                None
            };

            for fact in &all_facts {
                // Resolve subject and object IDs through the merge map. The RULE
                // is a pure function with its own tests (`fact_rules.rs`).
                let FactEndpoints {
                    subject_id,
                    object_id,
                    object_value,
                } = resolve_fact_endpoints(fact, &name_to_id);
                let object_value = object_value.as_deref();

                // Get candidate pools for contradiction detection
                let pool_a = match self
                    .graph
                    .get_facts_by_subject_predicate(&subject_id, &fact.predicate)
                    .await
                {
                    Ok(p) => p,
                    Err(e) => break 'phases Err(e),
                };

                // For pool_b, use FTS search on the predicate to find semantically related facts
                let pool_b_hits = match self
                    .graph
                    .fts_search_facts(FtsSearchFactsParams {
                        query: &fact.predicate,
                        limit: 10,
                        filters: &SearchFilters::new(),
                    })
                    .await
                {
                    Ok(hits) => hits,
                    Err(e) => break 'phases Err(e),
                };
                // Keep only candidates that SHARE AN ENTITY with the new fact.
                // The RULE — and the measured cost argument behind it — is a pure
                // function with its own tests (`fact_rules.rs`).
                let pool_b = pool_b_sharing_an_entity(
                    pool_b_hits.into_iter().map(|h| h.item).collect(),
                    &subject_id,
                    object_id.as_deref(),
                );

                // Run contradiction detection (skipped when disabled —
                // no LLM call, no invalidation; the fact below is still stored)
                let contradiction_result = match &detector {
                    Some(d) => match d
                        .detect(DetectParams {
                            new_fact: fact,
                            pool_a: &pool_a,
                            pool_b: &pool_b,
                            reference_time: &ref_time,
                        })
                        .await
                    {
                        Ok(r) => r,
                        Err(e) => break 'phases Err(e),
                    },
                    None => crate::core::contradiction::ContradictionResult::no_conflicts(),
                };

                // ── Did this check EARN its LLM call? ───────────────────────────
                //
                // Contradiction detection is 38.4% of all ingest LLM calls
                // (361/941 measured over 20 real sessions) and had
                // NO success metric of any kind — the only related counter in
                // the crate was `kremory.graph.unsupersede_total`, which meters
                // the REVERSE operation. So the single largest consumer of the
                // ingest budget could not be shown to find anything, and a 48x
                // token amplification went unnoticed for months.
                //
                // `source` is the load-bearing label. `pool_a` is scoped to
                // (subject, predicate); `pool_b` is an FTS hit on the PREDICATE
                // ALONE (see the search above), so it can return other
                // subjects' facts entirely. Splitting outcomes by which pool
                // supplied the candidates answers, with data, whether the
                // predicate-only arm earns its cost — the question that
                // otherwise has to be settled by opinion.
                //
                // Cardinality is bounded: 3 outcomes x 4 sources = 12 series.
                // The labelling RULE is a pure function with its own tests
                // (`fact_rules.rs`).
                {
                    let (outcome, source) = contradiction_labels(
                        &pool_a,
                        &contradiction_result.contradictions,
                        &contradiction_result.duplicates,
                    );
                    metrics::counter!(
                        "kremory.ingest.contradiction_outcome_total",
                        "outcome" => outcome,
                        "source" => source,
                    )
                    .increment(1);
                    // Was the LLM actually consulted? `detect` short-circuits on
                    // empty/all-duplicate pools (contradiction.rs:365-376), so a
                    // call here is NOT guaranteed. Label by whether the
                    // predicate-only arm contributed candidates, which is what
                    // makes the call reachable in the first place.
                    metrics::counter!(
                        "kremory.ingest.contradiction_pool_total",
                        "pool_a_nonempty" => if pool_a.is_empty() { "false" } else { "true" },
                        "pool_b_nonempty" => if pool_b.is_empty() { "false" } else { "true" },
                    )
                    .increment(1);
                }

                // Invalidate contradicted facts
                //
                // `invalid_at: ref_time` is CORRECT — do not "fix" it to the
                // superseding fact's `valid_at`. That change was proposed during
                // adversarial review ("an as_of()-visible
                // window overlap") and REFUTED on classification:
                //
                //   * `Fact.invalid_at` is the CONTRADICTION-RESOLVER'S invalidation
                //     timestamp — "when did the resolver rule this out" — an
                //     explicitly DISTINCT concept from `valid_to`, per the
                //     `RetrievedFact` mapping at `memory/engine_handle.rs:831-837`.
                //     The wire-facing world-clock `invalid_at`
                //     the consumer sees IS `Fact.valid_to`, not this column.
                //   * NOTHING filters on `invalid_at`. `as_of()` gates on
                //     `valid_from`/`valid_to`, and every read path
                //     (`facts_at`, `entity_facts_at`, `get_facts_by_subject_predicate`)
                //     additionally requires `expired_at IS NULL` — which this same
                //     call sets. A superseded fact is therefore excluded by
                //     `expired_at`, so no overlap window can exist.
                //
                // The resolver ran during THIS ingest, whose reference time is
                // `ref_time`; stamping it is coherent and stays correct now that a
                // fact can carry its own earlier `valid_at`.
                for fact_id in &contradiction_result.contradictions {
                    if let Err(e) = self
                        .graph
                        .invalidate_fact_with_reason(InvalidateFactWithReasonParams {
                            fact_id: *fact_id,
                            expired_at: Utc::now(),
                            invalid_at: ref_time,
                        })
                        .await
                    {
                        break 'phases Err(e);
                    }
                    invalidated_fact_ids.push(*fact_id);

                    // ── Fire-site: on_contradiction ─────────────────────────────────
                    // BUILT here because it needs `pool_a`/`pool_b`, which are scoped to
                    // this loop iteration, but DELIVERED after the commit. The payload
                    // rule itself — including why a prior fact missing from both pools
                    // yields `None` rather than a default-filled event — is documented and
                    // tested on `fact_rules::contradiction_event`.
                    let contradiction_event = if sink.is_some() {
                        fact_rules::contradiction_event(fact_rules::ContradictionEventInput {
                            pool_a: &pool_a,
                            pool_b: &pool_b,
                            prior_fact_id: *fact_id,
                            new_fact: fact,
                            subject_id: &subject_id,
                            ref_time,
                            detected_at: Utc::now(),
                        })
                    } else {
                        None
                    };
                    deferred.push(DeferredEmission::Contradiction {
                        event: contradiction_event,
                        prior_fact_id: *fact_id,
                    });
                }

                // Within-episode DUPLICATE pre-check (SQL-only).
                //
                // This compared subject+predicate only, and
                // `pool_a` comes from `get_facts_by_subject_predicate` — i.e. it is
                // object-agnostic. So one episode asserting "Alice speaks English",
                // "Alice speaks French", "Alice speaks Spanish" stored ONLY THE FIRST:
                // facts 2 and 3 matched an existing same-episode row on the pair and
                // were dropped. Set-valued predicates lost every value but one, with no
                // count surfaced to the caller.
                //
                // The check must be on the FULL TRIPLE. Within a single episode there is
                // no temporal ordering that could make one assertion supersede another —
                // they are co-asserted, so differing objects are multiple values, not a
                // contradiction. Cross-episode supersession remains the separate,
                // temporally-ordered mechanism and is unaffected.
                //
                // Purely a client-side filter on data already fetched; no extra DB
                // round-trip is needed. The RULE — full triple, not the pair — is
                // a pure function with its own tests (`fact_rules.rs`).
                if is_within_episode_duplicate(
                    &pool_a,
                    EpisodeTriple {
                        episode_id,
                        object_id: object_id.as_deref(),
                        object_value,
                    },
                ) {
                    let ns = group_id.unwrap_or("default");
                    // Renamed from `within_episode_contradiction`: the check
                    // now fires only on an exact repeated triple, which is a duplicate,
                    // not a contradiction. The old name described what the code was
                    // wrongly doing.
                    metrics::counter!(
                        "rql.ingest.within_episode_duplicate_triple",
                        "namespace" => ns.to_string()
                    )
                    .increment(1);
                    tracing::debug!(
                        subject = %subject_id,
                        predicate = %fact.predicate,
                        episode_id,
                        namespace = %ns,
                        "kremory.ingest.within_episode_duplicate_triple: skipping exact repeat"
                    );
                    continue;
                }

                // DETECTION intent, preserved additively.
                //
                // An earlier design specified a *contradiction* check — same subject+predicate
                // with a DIFFERENT object — that kept the higher-confidence fact and surfaced
                // the loser via `IngestError`. That mechanism never shipped; what shipped
                // silently dropped, object-agnostically. The fix stores every
                // value, because within one episode there is no temporal ordering that
                // could make one assertion supersede another, and a set-valued predicate is
                // indistinguishable from a contradiction without predicate-cardinality
                // knowledge the engine does not have.
                //
                // Rather than discard the signal, emit it: this fires when ONE episode
                // asserts multiple DISTINCT objects for the same subject+predicate. Data
                // loss is irreversible; detection is additive — so measure the phenomenon
                // first, then design policy against real counts instead of assumptions.
                if is_within_episode_multivalue(&pool_a, episode_id) {
                    let ns = group_id.unwrap_or("default");
                    metrics::counter!(
                        "rql.ingest.within_episode_multivalue",
                        "namespace" => ns.to_string()
                    )
                    .increment(1);
                    tracing::debug!(
                        subject = %subject_id,
                        predicate = %fact.predicate,
                        episode_id,
                        namespace = %ns,
                        "kremory.ingest.within_episode_multivalue: same subject+predicate, \
                         distinct object, same episode — stored, not dropped"
                    );
                }

                // Insert the new fact — skip gracefully if FK constraint fails
                // (e.g., fact references an entity not in the extraction results).
                //
                // MUST be `try_insert_fact_with_group(.., group_id)`: entities are
                // upserted into the episode's `group_id` namespace, so a namespace-less
                // `try_insert_fact` writes the fact in the "default" namespace and its
                // FK to the namespaced subject/object entities FAILS → the fact is
                // silently skipped. That dropped EVERY LLM-extracted fact whenever a
                // group_id was set — i.e. on the entire facade path
                // (`Memory::remember` always passes `group_id = Some(namespace)`),
                // even though `Engine::ingest_with(group_id = None)` worked. Mirrors the
                // pre-pinned `with_facts` path above. (Deterministic repro: facts=1 at
                // group_id=None vs facts=0 at group_id=Some.)
                //
                // The `_with_group` variant still dedups
                // caller-pinned `with_facts` triples (caller wins by being there first).
                match self
                    .graph
                    .try_insert_fact_with_group(
                        FactInsert {
                            subject_id: &subject_id,
                            predicate: &fact.predicate,
                            object_id: object_id.as_deref(),
                            object_value,
                            // Per-fact world time, episode time as fallback.
                            // `None` reproduces the earlier behaviour EXACTLY (every fact
                            // from an episode shared `ref_time`), so this is strictly additive:
                            // only facts whose source text stated a resolvable date change.
                            // Measured on a 30-turn real-corpus sample, that is ~10% of facts.
                            // `as_of()` filters on this column.
                            valid_from: fact.valid_at.unwrap_or(ref_time),
                            confidence: fact.confidence,
                            source_episode_id: Some(episode_id),
                            embedding: None,
                        },
                        // Use `effective_gid` (= group_id.unwrap_or("default")), the SAME
                        // namespace the subject/object entities were upserted into above
                        // (l.398/498). Passing the raw `group_id` Option would write a NULL
                        // group for the `None` case while entities live in "default" — the
                        // same FK mismatch, just inverted. `try_insert_fact` (non-group)
                        // hardcodes "default", which is why it only worked at group_id=None.
                        Some(effective_gid),
                    )
                    .await
                {
                    Ok(Some(fact_id)) => {
                        // Embed the fact triple as a single string (subject predicate object)
                        // and store it so vector_search_facts can find it semantically.
                        // WRITE into `facts.embedding` — document-prefix it.
                        let fact_text =
                            format!("{} {} {}", fact.subject, fact.predicate, fact.object);
                        let prefixed_fact_text = document_embed_text(
                            &fact_text,
                            self.config.search.embed_task_prefix_enabled,
                        );
                        match self.embedder.embed(&prefixed_fact_text).await {
                            Ok(embedding) => {
                                // TD-253: a successful embed can still leave
                                // `facts.embedding` NULL if the STORE write
                                // fails — same caller-visible gap as an embed
                                // failure, so it gets the same accumulator push
                                // (mirrors the stub-entity loop's
                                // `store_failed` arm below).
                                if self
                                    .graph
                                    .set_fact_embedding(fact_id, &embedding)
                                    .await
                                    .is_err()
                                {
                                    embedding_failures.push(fact_text.clone());
                                }
                            }
                            // TD-253: previously silent (no log, no metric) —
                            // the fact still inserted with `embedding = NULL`,
                            // invisible until a later `recall()` found nothing.
                            Err(_) => {
                                embedding_failures.push(fact_text.clone());
                            }
                        }
                        inserted_fact_ids.push(fact_id);
                    }
                    Ok(None) => {
                        // Caller pre-pinned this triple via with_facts;
                        // LLM duplicate silently deduped. Counter already incremented in
                        // try_insert_fact. This is the expected silent-dedup path.
                        tracing::debug!(
                            subject = %fact.subject,
                            predicate = %fact.predicate,
                            object = %fact.object,
                            "kremory.ingest.phase2_fact_dedup_against_prior_pin"
                        );
                    }
                    Err(e) => {
                        // A swallowed fact-insert failure (e.g. FK constraint) was
                        // previously only a tracing::warn — invisible in aggregate, which hid
                        // the composite-FK namespace bug (facade facts silently dropped). Emit
                        // a counter so dropped facts are observable, not silent.
                        //
                        // Observability hardening: this bare counter gave the
                        // foreground path no `reason`/`namespace` attribution, unlike its
                        // deferred-path sibling (`deferred.rs`). Bring it to parity: label via
                        // the shared `Error::fact_insert_failure_reason` taxonomy and
                        // `effective_gid` (the SAME namespace the subject/object entities were
                        // upserted into above), plus the matching KREMORY_DEBUG
                        // namespace-mismatch dump.
                        let reason = e.fact_insert_failure_reason();
                        // Bounded-cardinality label only (`reason`): `namespace`
                        // (group_id) is consumer-supplied + unbounded — it stays
                        // in the warn log + KREMORY_DEBUG dump below, never a
                        // metric label (observability-first: no unbounded label
                        // cardinality).
                        metrics::counter!(
                            "kremory.ingest.phase2_fact_insert_failed_total",
                            "reason" => reason,
                        )
                        .increment(1);
                        tracing::warn!(
                            subject = %fact.subject,
                            predicate = %fact.predicate,
                            object = %fact.object,
                            namespace = %effective_gid,
                            reason,
                            error = %e,
                            "kremory.ingest.phase2_fact_insert_failed"
                        );
                        if reason == "fk_mismatch" && std::env::var("KREMORY_DEBUG").is_ok() {
                            tracing::error!(
                                target: "kremory.ingest.namespace_mismatch",
                                fact_namespace = %effective_gid,
                                subject = %fact.subject,
                                object = %fact.object,
                                "[KREMORY_DEBUG] foreground fact dropped: composite-FK mismatch — \
                                 subject/object entity not found in namespace (see fact_namespace). \
                                 Compare entity-write namespace via \
                                 rql.ingest.entity_persisted_total{{namespace}}."
                            );
                        }
                    }
                }

                // Subject-side episodic edge is written in the entity loop
                // (role="mention"), guaranteeing presence coverage for all extracted
                // entities regardless of whether they appear in facts.
                // The object-side episodic edge is the SOLE presence source for fact
                // objects that are stubs / forward references (the pre-scan inserts
                // the stub entity but writes no edge). It must therefore fire ONLY
                // when the object did NOT already receive a "mention" edge above —
                // otherwise an extracted-entity-that-is-also-a-fact-object gets a
                // duplicate (episode_id, entity_id) presence edge (the recall
                // double-render bug). Keyed on the resolved obj_id so merge/alias
                // resolution is honoured.
                if let Some(ref obj_id) = object_id {
                    if !entity_loop_ids.contains(obj_id.as_str()) {
                        // Fire-site: on_edge_added("object"). Fires on Ok
                        // only (insert error is soft `.ok()` precedent). ids in
                        // tracing fields only, NOT metric labels. Migration 006: thread
                        // `group_id` — this branch is gated on the object being in
                        // `name_to_id` (a this-batch entity stored under `group_id`),
                        // so the edge must reference the same namespace.
                        let object_ok = self
                            .graph
                            .insert_episodic_edge(InsertEpisodicEdgeParams {
                                episode_id,
                                entity_id: obj_id,
                                entity_group_id: group_id,
                                role: "object",
                            })
                            .await
                            .is_ok();
                        if object_ok {
                            // Captured, not fired — replayed after the outer commit.
                            deferred.push(DeferredEmission::ObjectEdge {
                                to_entity_id: obj_id.clone(),
                            });
                        }
                    }
                }
            }

            Ok(())
        }; // end 'phases block

        // Commit or rollback the outer transaction based on phase result.
        //
        // ── Fire-sites: terminal stage transitions ──────────────────────────────
        // Re-establishes 737e152 verify_stage Fire-sites 4 (EntitiesReady) + 5
        // (Failed) on the unified inline routine, plus Complete. Successful
        // sequence per IngestStatus doc: Extracting → EntitiesReady → Complete.
        // On error: Failed. Triple-emit at each (sink + counter + tracing).
        // episode_id is a tracing field only, NEVER a metric label. Failed-arm
        // tracing is `tracing::error!`.
        match phase_result {
            Ok(()) => {
                // ── The flush point ───────────────────────────────────────────────
                // `commit()` FIRST, then replay. Every emission buffered during the
                // transaction describes a row that only becomes real on the line above,
                // and neither a sink callback nor a counter can be rolled back.
                //
                // ⚠️ RE-ENTRANCY. `begin_immediate_if_needed()` hands back a NESTED
                // guard when this writer already owns a transaction, and that guard's
                // `commit()` is a NO-OP (schema.rs:293-303) — the durable commit belongs
                // to the outer caller. So for a nested `ingest_with` the flush below
                // still runs one level too early, and the outer transaction could yet
                // roll back. That residual is narrower than the defect being fixed (the
                // rollback of THIS phase no longer emits at all) and is currently
                // unreachable: every production caller reaches `ingest_with` through the
                // facade's `ingest()`, which holds no open transaction. It is recorded
                // rather than silently accepted — a future caller that DOES nest must
                // thread the flush to the outermost commit, because the bug would then
                // return for nested callers only, which is the hardest variant to see.
                outer_guard.commit().await?;
                flush_deferred_emissions(deferred, sink, episode_id);

                // ── Embed forward-reference stubs (shortfall root cause) ────────
                //
                // POST-COMMIT for the reason above plus one of its own: each
                // iteration is a network round-trip to the embedder, and running it
                // inside the write transaction would hold the lock across an LLM call.
                //
                // BEST-EFFORT, deliberately. A stub is already the degraded path — a
                // forward reference to an entity the extractor never described. Failing
                // the whole ingest because its name vector could not be written would
                // trade a retrieval gap for data loss. The per-reason counter is what
                // makes the failure visible instead of silent.
                //
                // This is a WRITE into `entities.embedding`, so it MUST go
                // through `document_embed_text` with the same `embed_task_prefix_enabled`
                // config as the main entity path (`:1699-1714`). Embedding the bare name
                // here would place the vector in a DIFFERENT space from every other
                // entity and silently degrade dense recall rather than fix it.
                for stub_name in stub_names_to_embed {
                    let text = document_embed_text(
                        &stub_name,
                        self.config.search.embed_task_prefix_enabled,
                    );
                    match self.embedder.embed(&text).await {
                        // Scoped by `effective_gid` — the SAME
                        // value the stub's own `insert_entity_with_group` used
                        // above. The unscoped form wrote this vector into every
                        // namespace holding an entity of the same name.
                        Ok(v) => match self
                            .graph
                            .set_entity_embedding_in_group(
                                crate::core::graph::SetEntityEmbeddingParams {
                                    id: &stub_name,
                                    group_id: effective_gid,
                                    embedding: &v,
                                },
                            )
                            .await
                        {
                            Ok(()) => {
                                metrics::counter!(
                                    "kremory.ingest.stub_embedded_total",
                                    "outcome" => "ok",
                                )
                                .increment(1);
                            }
                            Err(e) => {
                                metrics::counter!(
                                    "kremory.ingest.stub_embedded_total",
                                    "outcome" => "store_failed",
                                )
                                .increment(1);
                                tracing::warn!(
                                    target: "kremory.ingest.stub",
                                    name = %stub_name,
                                    error = %e,
                                    "stub embedding computed but NOT stored — this stub \
                                     stays invisible to dense retrieval"
                                );
                                // TD-253: computed-but-unstored still leaves
                                // `entities.embedding` NULL — same caller-visible
                                // gap as an embed failure.
                                embedding_failures.push(stub_name.clone());
                            }
                        },
                        Err(e) => {
                            metrics::counter!(
                                "kremory.ingest.stub_embedded_total",
                                "outcome" => "embed_failed",
                            )
                            .increment(1);
                            tracing::warn!(
                                target: "kremory.ingest.stub",
                                name = %stub_name,
                                error = %e,
                                "stub embedding FAILED — this stub stays invisible to \
                                 dense retrieval (ingest continues)"
                            );
                            // TD-253: this loop already logged + counted the
                            // failure — now also surfaced on EpisodeCommit so a
                            // caller can act on it instead of only an operator
                            // watching metrics.
                            embedding_failures.push(stub_name.clone());
                        }
                    }
                }
                if let Some(s) = sink {
                    s.on_stage_change(crate::core::error::IngestStatus::EntitiesReady);
                    s.on_stage_change(crate::core::error::IngestStatus::Complete);
                }
                metrics::counter!(
                    "kremory.sink.stage_transition_total",
                    "from" => "Extracting",
                    "to" => "EntitiesReady"
                )
                .increment(1);
                metrics::counter!(
                    "kremory.sink.stage_transition_total",
                    "from" => "EntitiesReady",
                    "to" => "Complete"
                )
                .increment(1);
                tracing::info!(
                    episode_id,
                    entities_written = upserted_entities.len(),
                    "kremory.sink.stage_change.entities_ready_then_complete"
                );
            }
            Err(e) => {
                // Roll back the outer transaction, then surface the error as the
                // async block's `Err`. The `Failed` sink/counter/tracing fire is
                // performed ONCE at the single exit point below (the outer
                // `match ingest_outcome`), covering this phase-rollback path AND
                // every other post-`Extracting` `?` failure uniformly. Do NOT
                // fire `Failed` here too, or it would double-emit.
                let _ = outer_guard.rollback().await;
                return Err(e);
            }
        }

        let total_ms = ingest_start.elapsed().as_secs_f64() * 1000.0;
        let entity_count = upserted_entities.len();
        let fact_count = inserted_fact_ids.len();
        let merge_count = merged_entities.len();
        let contradiction_count = invalidated_fact_ids.len();
        histogram!("rql.ingest.total_ms").record(total_ms);
        // phase1_ms exposes the sync ingest-with-extraction phase
        // separately from total_ms so consumers can distinguish user-facing
        // latency (this) from deferred relationship work (phase2_ms).
        histogram!("rql.ingest.phase1_ms").record(total_ms);
        histogram!("rql.ingest.entity_count").record(entity_count as f64);
        histogram!("rql.ingest.fact_count").record(fact_count as f64);
        histogram!("rql.ingest.merge_count").record(merge_count as f64);
        histogram!("rql.ingest.contradiction_count").record(contradiction_count as f64);
        tracing::info!(
            total_ms,
            entity_count,
            fact_count,
            merge_count,
            contradiction_count,
            stub_entities_inserted,
            "kremory.ingest.completed"
        );

        Ok(IngestionResult {
            episode_id,
            upserted_entities,
            inserted_fact_ids,
            invalidated_fact_ids,
            merged_entities,
            token_usage,
            stub_entities_inserted,
            dense_embedded,
            embedding_failures,
        })
        }
        .await;

        // ── Single Failed-fire exit ─────────────────────────────────────────────
        // The async block above captured every fallible step after the `Extracting`
        // fire. On ANY `Err`, fire `Failed` exactly once (sink + counter + tracing)
        // BEFORE re-propagating — the error is NOT swallowed.
        match ingest_outcome {
            Ok(result) => Ok(result),
            Err(e) => {
                fire_failed(&e);
                // Second defect: persist Failed. Previously the inline
                // path left the row at 'Pending', while BackgroundIngestor
                // writes 'Failed' — so a FAILED inline ingest was
                // indistinguishable from an IN-FLIGHT one. That is the
                // absence-read-as-measurement shape: a crashed episode looked
                // exactly like a slow one, forever.
                //
                // engine_handle.rs:392 claimed the inline path "cannot without
                // episode_id" — true THERE, but `episode_id` is in scope HERE
                // (it is already in the fire_failed tracing above), and
                // engine_handle.rs:432 shows the same UPDATE on the success
                // path. So this was a fixable defect, not a real limit.
                //
                // Best-effort by necessity: the ingest has already failed, so a
                // status-write failure must not mask the original error — but
                // it IS counted, never silent.
                if let Err(status_err) = self
                    .graph
                    .conn
                    .execute(
                        "UPDATE episodes SET episode_processing_status = 'Failed' WHERE id = ?1",
                        libsql::params![episode_id],
                    )
                    .await
                {
                    metrics::counter!("kremory.ingest.failed_status_write_failed_total")
                        .increment(1);
                    tracing::warn!(
                        episode_id,
                        error = %status_err,
                        "kremory.ingest.failed_status_write_failed — episode remains \
                         Pending and is indistinguishable from in-flight"
                    );
                }
                Err(e)
            }
        }
    }

    /// Make a caller-pinned entity recall-findable AND attributable
    /// WITHOUT a second LLM. Recall seeds on entities (`Engine::contextualize`
    /// — NOT `hybrid_search_entities`, which has zero production
    /// callers) then renders via the DEFAULT `TemporalFacts` template — so a
    /// pin must satisfy THREE channels to actually surface, all mirroring
    /// mode-(a):
    ///
    /// 1. **FTS** — the literal name is stamped into `properties["name"]` at the
    ///    pin call site (`insert_entity_with_group`); the FTS seed arm indexes
    ///    `entities_fts.properties` (label is empty post-Migration-009). This is
    ///    the channel that works even under a null embedder.
    /// 2. **Vector** — embed the literal name into `entities.embedding`. The
    ///    embedder is NOT the chat LLM, so "no second LLM" holds.
    /// 3. **Attribution** — link the entity to its source episode via an episodic
    ///    edge (`role="mention"`). Without it the entity has zero `source_refs`,
    ///    and the default `TemporalFacts` renderer (which emits output ONLY per
    ///    source_ref) renders a found entity to `""` — invisible despite the FTS
    ///    hit. This is the piece the skip-extraction early-return skipped (all
    ///    other `insert_episodic_edge` calls run AFTER it).
    ///
    /// All best-effort (success + failure both
    /// emit a labeled signal): a null/failing embedder still leaves the entity
    /// FTS-findable + attributed. `set_entity_embedding` + `insert_episodic_edge`
    /// are UPDATE/INSERT-OR-IGNORE, so they also cover entities that pre-existed
    /// as bare stubs (the FTS-name INSERT, by contrast, is skipped on Duplicate).
    /// Returns `true` iff the entity's embedding was computed AND stored — `false`
    /// on either failure (TD-253: both leave `entities.embedding` NULL for this id,
    /// so the caller cannot distinguish them and does not need to). The channel-3
    /// attribution edge below is independent and does not affect this return value.
    pub(super) async fn make_pinned_entity_recallable(&self, p: PinnedEntityRecall<'_>) -> bool {
        let PinnedEntityRecall {
            id,
            group_id,
            episode_id,
        } = p;
        // Channel 2 — vector. WRITE into `entities.embedding` — document-prefix it.
        let embedded = match self
            .embedder
            .embed(&document_embed_text(
                id,
                self.config.search.embed_task_prefix_enabled,
            ))
            .await
        {
            Ok(embedding) => {
                // Scoped by the pin's own `group_id`, matching
                // the sibling `insert_entity_with_group` pin call site.
                if let Err(e) = self
                    .graph
                    .set_entity_embedding_in_group(crate::core::graph::SetEntityEmbeddingParams {
                        id,
                        group_id: group_id.unwrap_or("default"),
                        embedding: &embedding,
                    })
                    .await
                {
                    metrics::counter!("kremory.with_facts.pinned_embedding_stamp_failed")
                        .increment(1);
                    tracing::warn!(
                        entity_id = %id,
                        error = %e,
                        "kremory.with_facts.pinned_embedding_stamp_failed"
                    );
                    false
                } else {
                    metrics::counter!("kremory.with_facts.pinned_entity_embedded_total")
                        .increment(1);
                    true
                }
            }
            Err(e) => {
                metrics::counter!("kremory.with_facts.pinned_embedding_failed").increment(1);
                tracing::warn!(
                    entity_id = %id,
                    error = %e,
                    "kremory.with_facts.pinned_embedding_failed"
                );
                false
            }
        };

        // Channel 3 — attribution. entity_group_id MUST match the entity's
        // namespace or the Migration-006 composite FK (entity_id, entity_group_id)
        // silently FK-fails and the edge is dropped.
        if let Err(e) = self
            .graph
            .insert_episodic_edge(InsertEpisodicEdgeParams {
                episode_id,
                entity_id: id,
                entity_group_id: group_id,
                role: "mention",
            })
            .await
        {
            metrics::counter!("kremory.with_facts.pinned_episodic_edge_failed").increment(1);
            tracing::warn!(
                entity_id = %id,
                error = %e,
                "kremory.with_facts.pinned_episodic_edge_failed"
            );
        } else {
            metrics::counter!("kremory.with_facts.pinned_episodic_edge_total").increment(1);
        }

        embedded
    }
}
