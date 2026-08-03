use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use metrics::histogram;

use crate::core::config::{ContentType, ResolutionStrategy};
use crate::core::contradiction::{DetectParams, TwoPoolDetector};
use crate::core::embed_prefix::{document_embed_text, query_embed_text};
use crate::core::entity_types::EntityTypeRegistry;
use crate::core::extraction::normalize_label;
use crate::core::extraction_window::ExtractionWindowSplitter;
use crate::core::graph::{
    EpisodeInsert, FactInsert, InsertEntityWithGroupParams, InsertEpisodicEdgeParams,
    InvalidateFactWithReasonParams, SetEntityNerConfidenceParams, UpdateEntitySourceTierParams,
    UpsertEntityWithGroupParams,
};
use crate::core::intelligence::{
    EntityExtractor, EntityResolver, ExtractedEntity, ExtractedFact, ExtractionContext,
    ExtractionResult, ResolutionResult,
};
use crate::core::provider::{ChatProvider, EmbeddingProvider, TokenUsage};
use crate::core::resolver::{entity_name, normalize_name, CascadeResolver, UnionFind};
use crate::core::search::{FtsSearchFactsParams, SearchFilters, VectorSearchEntitiesNoCountParams};

use crate::core::ingest::helpers::extract_context_snippet;
use crate::core::ingest::{Engine, IngestionResult, SourceParams};

// ─── ADR-052 Gap 1 sink fire-site metrics helper ────────────────────────────

/// Triple-emit metrics companion for the per-entity / per-mention-edge sink
/// fire-sites in `ingest_with` (ADR-052 Gap 1; canonical pattern from 737e152
/// verify_stage Fire-sites 2 + 3). Kept as a free fn so the three `mention`
/// call sites (merged / L4-merge / new-entity) emit identical metric labels
/// without copy-paste drift. The sink callback itself fires inline at each call
/// site (it needs the per-site ids); only the label-stable counters live here.
///
/// D7 cardinality: NO entity_id / episode_id labels — `predicate_kind` is a
/// bounded enum, `mention_persisted` a bool string.
fn fire_entity_edge_metrics(mention_ok: bool) {
    metrics::counter!("kremory.sink.entity_extracted_total", "arm" => "inline").increment(1);
    if mention_ok {
        metrics::counter!(
            "kremory.sink.edge_added_total",
            "predicate_kind" => "episodic"
        )
        .increment(1);
    }
}

/// Bundled call-context parameters for [`Engine::ingest_with`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). The generic `extractor: &E`
/// stays a lead positional param (brief rule 4); these are the non-generic args.
pub struct IngestWithParams<'a> {
    pub text: &'a str,
    pub reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<&'a str>,
    pub content_type: Option<ContentType>,
    pub source_params: SourceParams,
}

/// TD-113: args-as-object for [`Engine::make_pinned_entity_recallable`]
/// (rust-conventions §too_many_arguments; clippy.toml threshold 3).
struct PinnedEntityRecall<'a> {
    /// Entity id (== the literal pinned subject/object text).
    id: &'a str,
    /// Namespace the entity + its episodic edge live in (composite-FK scope).
    group_id: Option<&'a str>,
    /// Source episode to attribute the entity to.
    episode_id: i64,
}

/// Bundled parameters for [`Engine::block_resolution_candidates`] — args-as-object
/// per TD-042 (rust-conventions §too_many_arguments). All fields share the `'a`
/// borrow of the pre-batch existing-entity slice so the returned candidate refs
/// tie back to it.
struct BlockCandidatesParams<'a> {
    extracted: &'a ExtractedEntity,
    existing_entities: &'a [crate::core::schema::Entity],
    group_id: Option<&'a str>,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// ADR-075 (TD-124): select the bounded set of existing entities that a
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
    /// pre-ADR-075 behaviour):
    /// - the group has ≤`k` existing entities (blocking only pays off past `k`;
    ///   keeps the whole small-graph test suite behaviour-identical), or
    /// - no usable embedding is available (null/failing embedder, or the ANN
    ///   query errors) — blocking needs embeddings; without them, behave as before.
    ///
    /// Returned refs borrow `existing_entities`, keeping the candidate universe
    /// == the pre-batch existing set so the downstream stub-check + union-find
    /// (which reference `existing_entities` by id) stay valid.
    async fn block_resolution_candidates<'a>(
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
        // TD-143: this is a QUERY against the SAME `entities` vector index that
        // stores document-prefixed writes (see `set_entity_embedding` call sites
        // below) — must use `query_embed_text`, not the raw name, or a flipped
        // knob would compare an unprefixed probe against a document-prefixed
        // corpus (the exact mixed-index footgun TD-143's correctness note warns
        // against).
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
            valid_after: None,
            valid_before: None,
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
                // ADR-075 P1: auto-different cosine floor. `vector_search` returns
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
    // Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
    // Generic `extractor: &E` stays a lead positional param (TD-042 brief rule 4);
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

        // TD-136: dense episode arm — embed + store the episode's embedding when
        // the dense arm is enabled (no-op / byte-identical when off).
        #[cfg(feature = "content-search")]
        self.maybe_embed_episode(episode_id, text).await;

        // 1b. ADR-035 §5 Option A — Pin caller-pre-extracted facts BEFORE Phase 2 LLM.
        //
        // Caller-supplied triples enter the graph first; the LLM Phase 2 extraction
        // path uses `try_insert_fact` which silently swallows the resulting
        // `Err(Duplicate)` on the same `content_hash`, so caller wins via
        // pre-write ordering. Intra-set duplicates (caller passes the same
        // triple twice in their own set) also dedup cleanly via
        // `try_insert_fact_with_group`.
        let mut pinned_fact_ids: Vec<i64> = Vec::new();
        if !source_params.pre_pinned_facts.is_empty() {
            let mut pinned_count: u64 = 0;
            for pf in &source_params.pre_pinned_facts {
                // Auto-stub the subject entity (entity_type_id=0 "Entity") so the
                // fact insert FK constraint resolves. Mirrors how Phase 2 LLM
                // extraction auto-creates UNKNOWN entities for forward references.
                // Duplicate-entity is the expected case (entity already exists);
                // explicit swallow. Per [[treat-cause-not-symptom]] + Quinn M-02,
                // NON-Duplicate errors (connection failure, schema gap, etc) are
                // surfaced via tracing::warn — masking them silently would hide
                // real production failures.
                // Stub entities MUST be created in the fact's `group_id` namespace, not the
                // default one: the facts composite FK is (subject_id, subject_group_id) →
                // entities(id, group_id) (schema.rs:1450). A namespace-less `insert_entity`
                // puts the stub in "default" while the pinned fact below stamps
                // `subject_group_id = group_id`, so the FK fails and the fact is dropped.
                // TD-113: stamp the entity's literal name into `properties` so
                // the FTS seed arm (`entities_fts.properties`) can find a
                // caller-pinned entity WITHOUT a second LLM. Mode-(a) LLM
                // extraction is recall-findable precisely because its
                // `properties["name"]` carries the name text (entities_fts.label
                // is empty post-Migration-009 — the FTS index is over
                // `properties` only). A bare `{"stub": false}` stub carried no
                // name token → recall returned 0. `properties["name"]` also
                // feeds `graph_search`'s original-case `entity_name` render.
                if let Err(e) = self
                    .graph
                    .insert_entity_with_group(InsertEntityWithGroupParams {
                        id: &pf.subject,
                        entity_type_id: 0,
                        properties: serde_json::json!({"name": pf.subject, "stub": false}),
                        group_id,
                    })
                    .await
                {
                    if !matches!(e, crate::core::error::Error::Duplicate { .. }) {
                        tracing::warn!(
                            subject = %pf.subject,
                            error = %e,
                            "kremory.with_facts.stub_entity_insert_failed"
                        );
                    }
                }
                // TD-113: stamp the vector channel too — embed the literal
                // subject text into `entities.embedding` so the vector seed arm
                // finds the pin under a real embedder. Best-effort + always-run
                // (not gated on `skip_extraction`): Phase 2 is not guaranteed to
                // re-cover a pinned subject that never appears in the episode
                // text, so pins must be findable independent of enrichment. The
                // embedder is NOT the chat LLM — "no second LLM" (spec §3 F1)
                // still holds. Also links the entity to its episode (attribution
                // channel) so it renders under the default TemporalFacts template.
                self.make_pinned_entity_recallable(PinnedEntityRecall {
                    id: &pf.subject,
                    group_id,
                    episode_id,
                })
                .await;
                if let Some(ref obj_id) = pf.object_id {
                    if let Err(e) = self
                        .graph
                        .insert_entity_with_group(InsertEntityWithGroupParams {
                            id: obj_id,
                            entity_type_id: 0,
                            properties: serde_json::json!({"name": obj_id, "stub": false}),
                            group_id,
                        })
                        .await
                    {
                        if !matches!(e, crate::core::error::Error::Duplicate { .. }) {
                            tracing::warn!(
                                object_id = %obj_id,
                                error = %e,
                                "kremory.with_facts.stub_entity_insert_failed"
                            );
                        }
                    }
                    self.make_pinned_entity_recallable(PinnedEntityRecall {
                        id: obj_id,
                        group_id,
                        episode_id,
                    })
                    .await;
                }

                // MED-1 time-inversion guard — mirror `facade/supersede.rs`'s
                // `if valid_to < fact.valid_from` reject (§3b step 3). A
                // caller-asserted `valid_to` predating the fact's own resolved
                // `valid_from` is a nonsensical `[valid_from, valid_to)` window
                // that, once bound, makes the fact permanently invisible to
                // every `as_of(t)` query. The default `valid_from` is ingest
                // `now()` (`sf.valid_from.or(published_at).unwrap_or(occurred_at)`
                // in `engine_handle.rs`), so a caller who supplies `valid_to`
                // but NOT `valid_from` trivially inverts the window.
                //
                // DECISION: reject the WHOLE pin (skip the insert entirely) —
                // do NOT silently drop the `valid_to` and persist an open-ended
                // window the caller never asked for. Caller-asserted temporal
                // data is never silently altered; a nonsensical assertion is
                // refused outright. Observable via counter + warn, never silent.
                if let Some(valid_to) = pf.valid_to {
                    if valid_to < pf.valid_from {
                        metrics::counter!(
                            "kremory.with_facts.pin_rejected_total",
                            "outcome" => "rejected_time_inversion"
                        )
                        .increment(1);
                        tracing::warn!(
                            subject = %pf.subject,
                            predicate = %pf.predicate,
                            valid_from = %pf.valid_from,
                            valid_to = %valid_to,
                            "kremory.with_facts.pin_rejected_time_inversion"
                        );
                        continue;
                    }
                }

                match self
                    .graph
                    .try_insert_fact_with_group(
                        FactInsert {
                            subject_id: &pf.subject,
                            predicate: &pf.predicate,
                            object_id: pf.object_id.as_deref(),
                            object_value: pf.object_value.as_deref(),
                            valid_from: pf.valid_from,
                            confidence: pf.confidence,
                            source_episode_id: Some(episode_id),
                            embedding: None,
                        },
                        group_id,
                    )
                    .await
                {
                    Ok(Some(fact_id)) => {
                        pinned_count = pinned_count.saturating_add(1);
                        pinned_fact_ids.push(fact_id);
                        // ADR-068 boy-scout: bind the caller-asserted bounded
                        // window (StructuredFact.valid_to, "None = open-ended")
                        // if given — this used to be silently dropped
                        // (`PrePinnedFact` carried `valid_from` only). Mirrors
                        // `SupersedeRequest`'s own `bound_valid_to` call
                        // (Amendment C) — NOT `invalidate_fact`/
                        // `invalidate_fact_with_reason`, which write
                        // `expired_at`/`invalid_at` (a separate, later,
                        // system-time retirement act).
                        if let Some(valid_to) = pf.valid_to {
                            if let Err(e) = self.graph.bound_valid_to(fact_id, valid_to).await {
                                // MED-2 (Rule 19): a dropped caller-asserted
                                // `valid_to` (DB-level bind failure — distinct
                                // from the MED-1 time-inversion reject above)
                                // must be observable, not just warn-logged. The
                                // success path already has
                                // `valid_to_bound_total`; pair it with a failure
                                // counter so the drop rate is a metric.
                                metrics::counter!("kremory.with_facts.valid_to_bind_failed_total")
                                    .increment(1);
                                tracing::warn!(
                                    subject = %pf.subject,
                                    fact_id,
                                    error = %e,
                                    "kremory.with_facts.valid_to_bind_failed"
                                );
                            } else {
                                metrics::counter!("kremory.with_facts.valid_to_bound_total")
                                    .increment(1);
                            }
                        }
                        // ADR-045 §3 / spec §1.2: stamp ConsumerPinned on the subject entity
                        // so the dream reclassify pass skips it. Best-effort — a warn on
                        // failure is sufficient; the fact is already pinned.
                        if let Err(e) = self
                            .graph
                            .update_entity_source_tier(UpdateEntitySourceTierParams {
                                id: &pf.subject,
                                group_id,
                                source_tier: "ConsumerPinned",
                            })
                            .await
                        {
                            tracing::warn!(
                                subject = %pf.subject,
                                error = %e,
                                "kremory.with_facts.consumer_pinned_stamp_failed"
                            );
                        } else {
                            metrics::counter!(
                                "kremory.with_facts.consumer_pinned_tier_stamped_total"
                            )
                            .increment(1);
                        }
                        // Q-04 (Option A — symmetric protection per ADR-045 §3 amendment 2026-06-09):
                        // When a pinned fact references an object entity (object_id = Some), stamp it
                        // ConsumerPinned too — the consumer asserted a typed relationship, so dream
                        // re-typing of either endpoint silently invalidates their assertion. When
                        // object is a literal (object_id = None), no object entity exists to stamp —
                        // subject-only is correct.
                        //
                        // Peer convergence: Graphiti add_triplet + Mem0 infer=False both treat
                        // endpoints symmetrically. Zero precedent for subject-only protection across
                        // surveyed peers (Graphiti / Letta / Mem0 / LightRAG / Cognee / LangChain).
                        if let Some(ref object_id) = pf.object_id {
                            match self
                                .graph
                                .update_entity_source_tier(UpdateEntitySourceTierParams {
                                    id: object_id,
                                    group_id,
                                    source_tier: "ConsumerPinned",
                                })
                                .await
                            {
                                Ok(()) => {
                                    metrics::counter!(
                                        "kremory.with_facts.consumer_pinned_object_tier_stamped_total"
                                    )
                                    .increment(1);
                                }
                                Err(e) => {
                                    metrics::counter!(
                                        "kremory.with_facts.consumer_pinned_object_stamp_failed"
                                    )
                                    .increment(1);
                                    tracing::warn!(
                                        object_id = %object_id,
                                        error = %e,
                                        "kremory.with_facts.consumer_pinned_object_stamp_failed"
                                    );
                                }
                            }
                        }
                    }
                    Ok(None) => {
                        // Intra-caller-set duplicate (already counted as
                        // axis=caller_vs_llm inside the helper; relabel here
                        // for diagnostic clarity via a second counter increment).
                        metrics::counter!(
                            "kremory.with_facts.deduped_total",
                            "axis" => "intra_caller_set"
                        )
                        .increment(1);
                    }
                    Err(e) => {
                        // DIAG: surface failures via counter so tests can detect.
                        metrics::counter!("kremory.with_facts.pin_failed_total").increment(1);
                        tracing::warn!(
                            subject = %pf.subject,
                            predicate = %pf.predicate,
                            error = %e,
                            "kremory.with_facts.pin_failed"
                        );
                    }
                }
            }
            if pinned_count > 0 {
                metrics::counter!(
                    "kremory.with_facts.pinned_total",
                    "source" => "caller"
                )
                .increment(pinned_count);
                tracing::info!(
                    pinned = pinned_count,
                    requested = source_params.pre_pinned_facts.len(),
                    episode_id,
                    "kremory.with_facts.pinned"
                );
            } else {
                // Per Quinn L-01: avoid INFO-level noise for all-dedup caller sets;
                // bulk-import workloads can hit this thousands of times per batch.
                tracing::debug!(
                    pinned = 0_u64,
                    requested = source_params.pre_pinned_facts.len(),
                    episode_id,
                    "kremory.with_facts.all_deduped_or_failed"
                );
            }
        }

        // 1c. ADR-035 §2/§6 — `skip_extraction` early return.
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
            });
        }

        // ── ADR-052 Gap 1 fire-site: on_stage_change(Extracting) ────────────────
        // Re-establishes the 737e152 verify_stage Fire-site 1 on the UNIFIED
        // inline/background extraction routine (`ingest_with`) that the public
        // `Memory::remember(...).with_event_sink(...)` consumer journey actually
        // drives (engine_handle::graph_ingest_episode → engine.ingest →
        // ingest_with). The sink is carried on `source_params.sink` as the
        // core-layer `IngestEventSink` (see SourceParams::sink doc-comment) and
        // is fired synchronously per the §4.8 sync-inline contract. Triple-emit:
        // sink + counter + tracing (ADR-2026-05-20 D1). D7: episode_id is a
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

        // ── ADR-051 §4 state-machine invariant: Extracting → Failed on ANY error ──
        // Canonical pattern: `core/background/verify_stage.rs` fires `Failed`
        // synchronously BEFORE every `Err` propagates (the function's own
        // doc-comment invariant). Quinn MED-01: the previous terminal `match
        // phase_result` only fired `Failed` for errors that broke the inner
        // `'phases` block. Every fallible `?`-step AFTER the `Extracting` fire but
        // OUTSIDE `'phases` — registry load/seed, `ExtractionWindowSplitter` work,
        // `extractor.extract(..).await?`, the in-`'phases` `llm_for_detector?`,
        // `begin_immediate_if_needed().await?`, and `outer_guard.commit().await?` —
        // returned early WITHOUT firing `Failed`; the consumer saw `Extracting`
        // then silence.
        //
        // Cause-fix per Rule 8 (treat-cause): capture the WHOLE post-`Extracting`
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
        //     TD-013 §1 hybrid: if caller supplied a per-call override, use it.
        //     - First-call persistence: if DB is empty for this group_id, persist
        //       the override so subsequent calls without override see the vocabulary.
        //     - Ephemeral: if DB already has rows, use override for this call only.
        let effective_gid = group_id.unwrap_or("default");

        // 2a. TD-013 Migration 010 lazy-seed: ensure the default vocabulary is
        //     present for `effective_gid` before any registry-dependent work
        //     (L2 prompt, L3 validation). Idempotent: no-ops once seeded.
        //     This catches namespaces created AFTER Migration 010 ran at boot.
        crate::core::entity_types::ensure_default_types_seeded(&self.graph.conn, effective_gid)
            .await?;

        let registry = if let Some(ref override_specs) = source_params.entity_types_override {
            let db_registry =
                EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?;
            if db_registry.is_empty() {
                // First-call persistence: seed DB from override.
                crate::core::entity_types::upsert_entity_types(
                    &self.graph.conn,
                    effective_gid,
                    override_specs,
                )
                .await?;
                metrics::counter!(
                    "rql.ingest.registry_override_applied",
                    "persisted" => "true",
                )
                .increment(1);
            } else {
                // Additive merge: persist any override types missing from DB.
                //
                // Was previously "ephemeral, do not touch DB" (TD-013) but that
                // created an entity-type/JOIN hole: an entity row stored with
                // entity_type_id = override-only id had no entity_types row →
                // SQL COALESCE(et.name, 'Entity') resolved label='Entity' at
                // read time regardless of the stored integer id. Per
                // [[audit-what-guards-mask-before-deleting]] the right fix is
                // additive merge — INSERT OR IGNORE each missing override
                // spec, preserving previously-stored rows.
                let mut newly_persisted: usize = 0;
                for spec in override_specs.iter() {
                    if db_registry.name_to_id(&spec.name).is_none() {
                        self.graph
                            .conn
                            .execute(
                                "INSERT OR IGNORE INTO entity_types \
                                 (group_id, id, name, description) \
                                 VALUES (?1, ?2, ?3, ?4)",
                                libsql::params![
                                    effective_gid,
                                    spec.id as i64,
                                    spec.name.clone(),
                                    spec.description.clone()
                                ],
                            )
                            .await
                            .map_err(|e| {
                                crate::core::error::Error::Other(anyhow::anyhow!(
                                    "additive override persist failed for spec '{}': {e}",
                                    spec.name
                                ))
                            })?;
                        newly_persisted += 1;
                    }
                }
                metrics::counter!(
                    "rql.ingest.registry_override_applied",
                    "persisted" => if newly_persisted > 0 { "additive" } else { "false" },
                )
                .increment(1);
                metrics::histogram!("rql.ingest.registry_override_additive_count")
                    .record(newly_persisted as f64);
            }
            EntityTypeRegistry::from_specs(override_specs.clone())
        } else {
            let db_registry =
                EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?;
            // TD-028 Phase B — builder-seed: if the builder set `allowed_entity_types`,
            // any type not already in the registry for this group_id is registered
            // additively via `label_to_id_or_register`. This runs AFTER
            // `ensure_default_types_seeded` so the registry is never empty here;
            // the merge is additive-only (INSERT OR IGNORE via the same race-safe
            // MAX(id)+1 path that Pass 0 uses). No caller-specified id → SQLite
            // assigns the next free id, avoiding position-based collisions with
            // future Pass 0 writes (GAP-006 fix).
            if !self.config.allowed_entity_types.is_empty() {
                let mut new_types_seeded: usize = 0;
                for name in &self.config.allowed_entity_types {
                    // QB-01: use case-insensitive match here so we don't fire the
                    // seed branch when the DB already has the canonical form under a
                    // different case (e.g. builder has "court", DB has "Court").
                    // `label_to_id_or_register` does a case-insensitive DB lookup
                    // (COLLATE NOCASE path), so a case-sensitive `name_to_id` guard
                    // would count a "skip that never inserts" as a seed — Rule 19
                    // cardinal failure mode #9 (lying counter).
                    let already_registered = db_registry
                        .specs()
                        .iter()
                        .any(|s| s.name.eq_ignore_ascii_case(name));
                    if !already_registered {
                        crate::core::entity_types::label_to_id_or_register(
                            crate::core::entity_types::LabelToIdOrRegisterParams {
                                conn: &self.graph.conn,
                                group_id: effective_gid,
                                registry: &db_registry,
                                label: name,
                            },
                        )
                        .await?;
                        new_types_seeded += 1;
                    }
                }
                if new_types_seeded > 0 {
                    metrics::counter!(
                        "rql.ingest.registry_builder_seed_applied",
                        "namespace" => effective_gid.to_string(),
                    )
                    .increment(1);
                    // QB-03: histogram gains namespace label matching the paired counter
                    // so operators can disaggregate by namespace (Rule 19 observability).
                    metrics::histogram!(
                        "rql.ingest.registry_builder_seed_count",
                        "namespace" => effective_gid.to_string(),
                    )
                    .record(new_types_seeded as f64);
                    // Reload the registry so the derived allowed_entity_types_live
                    // below includes the newly-seeded builder types.
                    EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?
                } else {
                    db_registry
                }
            } else {
                db_registry
            }
        };

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
            let mut raw = match group_id {
                Some(gid) => self.graph.list_entities_in_group(gid).await?,
                None => self.graph.list_entities().await?,
            };
            // Sort descending by access_count (most-recently-used first).
            raw.sort_by(|a, b| b.access_count.cmp(&a.access_count));
            raw.truncate(L4_PRIME_INJECT_LIMIT);
            raw.into_iter()
                .map(|e| {
                    // Prefer properties["name"] (original casing) over the
                    // normalized id so the LLM sees the exact display name.
                    let display_name = e
                        .properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_owned())
                        .unwrap_or(e.id);
                    (display_name, e.label)
                })
                .collect()
        };

        // 3. Extract from all chunks, merge results.
        // known_entities grows with each iteration so subsequent chunks receive
        // the entities already found in earlier chunks as context.
        //
        // TD-028 Phase B — B1: derive allowed_entity_types from the live registry
        // (loaded above) rather than the config snapshot taken at engine construction.
        // This closes the self-learning loop: Pass 0 writes new types to the registry;
        // the next ingest sees them immediately without an engine rebuild.
        // Spec: td-028-phase1-pull-shape-registry-read-micro-spec-2026-06-09.md §1.
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

        // TD-013 Phase 8 v3 (2026-06-04): scan_proper_nouns DISABLED pending
        // task #15 — scanner hardcodes label="Entity" instead of routing
        // candidates through LLM classification (the original two-call design
        // per spike_phase3_scanner_vs_pipeline.rs, never finished). Disabling
        // gives us pure-LLM extraction matching Graphiti/Cognee/LightRAG.
        // Re-enable after wiring the second LLM call OR delete entirely.
        //
        // let proper_noun_candidates = text_utils::scan_proper_nouns(text, &all_entities);
        // all_entities.extend(proper_noun_candidates);

        // 3b. Pre-mutation intra-batch duplicate scan (Story #150 history).
        //
        // Original behaviour: FATAL `Err(IntraBatchDuplicate)` on duplicate names.
        // The intent was to surface malformed batches submitted by human callers
        // rather than silently dropping rows.
        //
        // Why we softened it in v0.1.4: kremory's pipeline ingests LLM-extracted
        // entities, and noisy extractors (smaller local models — llama3.2:3b,
        // gemma4-e2b — observed 2026-05-28 emitting `'car'` / `'VerbatimString'`
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
            let mut seen_this_call: HashSet<String> = HashSet::new();
            let mut dup_names: Vec<String> = Vec::new();
            for extracted in &all_entities {
                let id = normalize_name(&extracted.name);
                if !seen_this_call.insert(id.clone()) {
                    dup_names.push(id);
                }
            }
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

        // ── ADR-076 (TD-127) Pass 1 + Pass 2: batched entity resolution ────────
        // Runs BEFORE the write transaction (ADR-076 SCOPE-001) — both passes
        // only read the frozen `existing_entities` snapshot loaded above and
        // do zero DB writes, so they execute in the pre-transaction window
        // (shrinking lock-hold vs the pairwise path, which resolves inline
        // inside the transaction's entity loop below).
        //
        // Pass 1 (deterministic, no LLM): for each extracted entity, run the
        // existing ADR-075 candidate block + the cheap Tier-1/Tier-2 tiers.
        // A hit pre-resolves the entity; a miss adds it to the ambiguous
        // worklist. Pass 2 batches the ambiguous remainder into windowed
        // structured-output calls (`resolver_batched::resolve_batched`).
        //
        // `batched_resolved` maps `normalize_name(entity.name) -> existing
        // entity id` for every CONFIDENT resolution (deterministic or
        // batched-LLM); entities absent from this map are NEW. Only built
        // when the strategy is `Batched` — the `Pairwise` arm below resolves
        // inline exactly as before ADR-076.
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
                    // Quinn HIGH (ADR-076): an entity whose OWN block is empty
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

        // Helper closure-like block that returns Result<()> so we can commit or
        // rollback in one place.  We use a labelled block instead of an async
        // closure to avoid capture/lifetime complexity.
        let phase_result: crate::core::error::Result<()> = 'phases: {
            // ── Bug E §1: pre-scan all_facts for forward-reference entity names ─────
            // Collect every entity name appearing as a subject or object in facts.
            // Any name NOT already mapped from the extraction list is a forward
            // reference — insert it as an UNKNOWN stub so the fact loop can resolve
            // it without producing a dangling subject_id.
            let extracted_names: HashSet<String> = all_entities
                .iter()
                .map(|e| normalize_name(&e.name))
                .collect();

            for fact in &all_facts {
                let mut forward_refs: Vec<String> = vec![normalize_name(&fact.subject)];
                if fact.is_entity_ref {
                    forward_refs.push(normalize_name(&fact.object));
                }
                for norm_name in forward_refs {
                    if extracted_names.contains(&norm_name) {
                        // Will be handled in the entity loop — skip.
                        continue;
                    }
                    if name_to_id.contains_key(&norm_name) {
                        // Already inserted as a stub in a previous fact iteration.
                        continue;
                    }
                    // RISK-003: entities.id is a sole TEXT PK pre-migration-004.
                    // Post-migration-004: composite PK (id, group_id) closes the bypass surface.
                    // Stub INSERT uses INSERT OR IGNORE — cross-namespace name collision silently
                    // skips stub creation. Single-namespace use only for v0.1.1.
                    //
                    // Strategy: attempt insert_entity_with_group; if the entity already exists
                    // (UNIQUE constraint error), that is fine — a real row is present.
                    let stub_props =
                        serde_json::json!({ "stub": true, "source": "forward_reference" });
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
                            metrics::counter!("kremory.ingest.stub_inserted").increment(1);
                            metrics::counter!(
                                "rql.ingest.entity_persisted_total",
                                "source" => "stub",
                            )
                            .increment(1);
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
            }

            // Resolved entity ids that received a "mention" presence edge in the
            // entity loop below. Populated at EACH of the three mention-write
            // sites (merge/promote-stub, L4-merge, new-insert) immediately after a
            // successful insert — NOT at a single consolidated point, because the
            // L4-merge branch `continue`s and would skip a consolidated insert
            // (Quinn F-01). The fact loop consults this to keep presence
            // single-owned: an extracted entity already linked here MUST NOT also
            // receive an "object" edge (the duplicate `(episode_id, entity_id)`
            // bug). Keyed on the RESOLVED id (post merge/alias) so it is robust to
            // disambiguation, unlike a surface-name set. ADR-052 §3.1 fire-sites.
            let mut entity_loop_ids: HashSet<String> = HashSet::new();

            // ── Phase 1: entity loop (Bug B snippet + Bug A episodic_edge) ──────────
            for extracted in &all_entities {
                // TD-013 Phase 8 finalizes Vera M1: the TD-012 over-rejection guard
                // (`is_canonical_entity_type` + `allowed_entity_types` policy) is
                // DELETED under the L1 integer-ID design. Live diagnostic
                // 2026-06-03 showed it rejecting 14 "Entity" emissions per
                // mock_interview ingest — id=0 "Entity" is the legitimate
                // Graphiti-pattern catch-all, NOT a placeholder to drop.
                //
                // Replacement guardrails (all already wired):
                //   - L3 validate_or_fallback bounds-checks emitted entity_type_id
                //   - L4 disambiguation handles surface-form duplicates
                //   - L5 vector canonicalization merges variant spellings
                //   - L7 dream-phase reclassification upgrades id=0 entities once
                //     corpus accumulates 3+ episodes
                //
                // We still normalise the label string so downstream resolver +
                // FTS get the canonical case ("ORG" → "Organisation").
                let label = normalize_label(&extracted.label);

                // ADR-076 (TD-127): which existing id (if any) `extracted`
                // resolves to, computed differently per strategy.
                //
                // - `Batched` (default): Pass 1 (deterministic tiers) + Pass 2
                //   (batched LLM call) already ran BEFORE this transaction —
                //   see `batched_resolved` above. This arm is a pure lookup;
                //   it must NOT call `block_resolution_candidates` again
                //   (that already ran once per entity in Pass 1).
                // - `Pairwise`: the pre-ADR-076 behaviour, unchanged. ADR-075
                //   (TD-124) resolves `extracted` only against a bounded
                //   candidate block (exact-name ∪ embedding-ANN top-k), not
                //   every existing entity — collapses the LLM
                //   `ResolutionVerdict` fan-out from O(new × existing) to
                //   O(k). No-op on ≤k-entity groups.
                let resolved_to: Option<String> = match self.config.resolution_strategy {
                    ResolutionStrategy::Batched => batched_resolved
                        .get(&normalize_name(&extracted.name))
                        .cloned(),
                    ResolutionStrategy::Pairwise => {
                        let candidates = self
                            .block_resolution_candidates(BlockCandidatesParams {
                                extracted,
                                existing_entities: &existing_entities,
                                group_id,
                            })
                            .await;
                        metrics::counter!("kremory.resolution.candidates_considered_total")
                            .increment(candidates.len() as u64);
                        metrics::counter!("kremory.resolution.blocked_out_total").increment(
                            existing_entities.len().saturating_sub(candidates.len()) as u64,
                        );

                        let mut found: Option<String> = None;
                        for existing in candidates.iter().copied() {
                            let result = match resolver.resolve(extracted, existing).await {
                                Ok(r) => r,
                                Err(e) => break 'phases Err(e),
                            };
                            if result == ResolutionResult::Same {
                                found = Some(existing.id.clone());
                                break;
                            }
                        }
                        found
                    }
                };

                let entity_id = if let Some(existing_id) = resolved_to {
                    // Merged with existing entity
                    union_find.make_set(&existing_id);
                    let norm = normalize_name(&extracted.name);
                    union_find.make_set(&norm);
                    union_find.union(&norm, &existing_id);
                    merged_entities.push((existing_id.clone(), extracted.name.clone()));

                    // Bug E (F-4): stub promotion — if the existing entity is a stub
                    // (entity_type_id=0, properties.stub=true), overwrite it with the
                    // real entity_type_id and a fresh context snippet. The upsert removes
                    // the stub flag because the new properties map does not carry it.
                    // First-mention-wins policy still applies for non-stubs.
                    let existing_is_stub = existing_entities
                        .iter()
                        .find(|e| e.id == existing_id)
                        .map(|e| {
                            e.entity_type_id == 0
                                && e.properties
                                    .get("stub")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false)
                        })
                        .unwrap_or(false);

                    if existing_is_stub {
                        let context_snippet = extract_context_snippet(text, &extracted.name, 200);
                        let promoted_props = serde_json::json!({
                            "context": context_snippet,
                            "name": extracted.name.clone()
                        });
                        // TD-021 open vocabulary: register novel type label on the fly.
                        // If registration itself fails (DB error), fall back to id=0
                        // and emit a swallow counter (Vera Finding 1) — the stub
                        // promotion is best-effort and must not abort the transaction.
                        let entity_type_id =
                            match crate::core::entity_types::label_to_id_or_register(
                                crate::core::entity_types::LabelToIdOrRegisterParams {
                                    conn: &self.graph.conn,
                                    group_id: group_id.unwrap_or("default"),
                                    registry: &registry,
                                    label: &label,
                                },
                            )
                            .await
                            {
                                Ok(id) => id,
                                Err(_) => {
                                    metrics::counter!(
                                        "rql.entity_types.registration_swallowed_total",
                                        "site" => "stub_promotion",
                                    )
                                    .increment(1);
                                    0
                                }
                            };
                        // `.ok()` — promotion is best-effort; failure to promote
                        // leaves the stub row but does not abort the transaction.
                        self.graph
                            .upsert_entity_with_group(UpsertEntityWithGroupParams {
                                id: &existing_id,
                                entity_type_id,
                                properties: promoted_props,
                                group_id,
                            })
                            .await
                            .ok();
                        // Stub-promotion path: a row that was previously source=stub
                        // is now being filled with LLM-extracted content. Count it as
                        // a NET-NEW llm-source write (the stub row is no longer a
                        // stub after this upsert).
                        metrics::counter!(
                            "rql.ingest.entity_persisted_total",
                            "source" => "llm",
                            "via" => "stub_promotion",
                        )
                        .increment(1);
                        tracing::debug!(
                            target: "kremory.ingest.stub",
                            id = %existing_id,
                            label = %label,
                            entity_type_id,
                            "promoted stub entity to real entity"
                        );
                    }

                    // Bug A: episodic edge for merged entity (this episode now references it).
                    // ADR-052 Gap 1 fire-sites: on_entity_extracted + on_edge_added("mention").
                    // Fires on Ok only (insert error is soft `.ok()` precedent). D7: ids in
                    // tracing fields only, NOT metric labels.
                    // ADR-052 / Migration 006: thread the ingest namespace so the
                    // composite FK (entity_id, entity_group_id) resolves — the
                    // entity was just merged/promoted under `group_id`, so the edge
                    // must reference the same namespace or it silently FK-fails.
                    let mention_ok = self
                        .graph
                        .insert_episodic_edge(InsertEpisodicEdgeParams {
                            episode_id,
                            entity_id: &existing_id,
                            entity_group_id: group_id,
                            role: "mention",
                        })
                        .await
                        .is_ok();
                    if let Some(s) = sink {
                        s.on_entity_extracted(&existing_id, &extracted.name);
                        if mention_ok {
                            s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                                from_entity_id: &episode_id.to_string(),
                                to_entity_id: &existing_id,
                                predicate: "mention",
                            });
                        }
                    }
                    fire_entity_edge_metrics(mention_ok);
                    tracing::debug!(entity_id = %existing_id, name = %extracted.name, episode_id, via = "merged", "kremory.sink.entity_extracted");
                    if mention_ok {
                        entity_loop_ids.insert(existing_id.clone());
                    }

                    existing_id
                } else {
                    // ── L4: graph-time disambiguation (Cognee pattern) ──────────────────
                    // Before inserting a new entity row, probe cosine similarity against
                    // existing entities in the same group_id.  This catches "Alice Johnson"
                    // vs "Alice J." style variants that the string-match resolver (step 4)
                    // misses because they are not lexically close enough.
                    //
                    // Outcomes:
                    //   Merge         → reuse the existing entity_id (same real-world entity)
                    //   PotentialAlias → insert new row AND a `potential_alias` fact edge
                    //   New           → proceed as normal (no existing match)
                    //
                    let l4_outcome = match crate::core::disambiguation::disambiguate(
                        crate::core::disambiguation::DisambiguateParams {
                            entity_name: &extracted.name,
                            group_id,
                            graph: &self.graph,
                        },
                        &*self.embedder,
                        self.config.search.embed_task_prefix_enabled,
                    )
                    .await
                    {
                        Ok(o) => o,
                        Err(e) => break 'phases Err(e),
                    };

                    // ── L4 Merge path ───────────────────────────────────────────────────
                    if let crate::core::disambiguation::DisambiguationOutcome::Merge {
                        existing_id: ref l4_existing_id,
                        ..
                    } = l4_outcome
                    {
                        let norm = normalize_name(&extracted.name);
                        union_find.make_set(l4_existing_id);
                        union_find.make_set(&norm);
                        union_find.union(&norm, l4_existing_id);
                        merged_entities.push((l4_existing_id.clone(), extracted.name.clone()));
                        // Episodic edge: this episode now references the existing entity.
                        // ADR-052 Gap 1 fire-sites: on_entity_extracted + on_edge_added("mention").
                        // ADR-052 / Migration 006: thread the ingest namespace so
                        // the composite FK resolves for the L4-merged entity (stored
                        // under `group_id`).
                        let mention_ok = self
                            .graph
                            .insert_episodic_edge(InsertEpisodicEdgeParams {
                                episode_id,
                                entity_id: l4_existing_id,
                                entity_group_id: group_id,
                                role: "mention",
                            })
                            .await
                            .is_ok();
                        if let Some(s) = sink {
                            s.on_entity_extracted(l4_existing_id, &extracted.name);
                            if mention_ok {
                                s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                                    from_entity_id: &episode_id.to_string(),
                                    to_entity_id: l4_existing_id,
                                    predicate: "mention",
                                });
                            }
                        }
                        fire_entity_edge_metrics(mention_ok);
                        tracing::debug!(entity_id = %l4_existing_id, name = %extracted.name, episode_id, via = "l4_merge", "kremory.sink.entity_extracted");
                        if mention_ok {
                            entity_loop_ids.insert(l4_existing_id.clone());
                        }
                        name_to_id.insert(norm, l4_existing_id.clone());
                        // Skip the rest of the else block — entity_id is the existing one.
                        // SAFETY: the outer `let entity_id = if ... { ... } else { ... };`
                        // expression needs a value; we push to name_to_id above and
                        // `continue` to the next extracted entity below.
                        continue;
                    }

                    // ── New or PotentialAlias — insert new entity row ────────────────────
                    let entity_id = normalize_name(&extracted.name);

                    // Bug B: capture verbatim first-mention snippet (±100 chars around the
                    // entity name in the source text).  First-mention wins: this branch only
                    // runs for genuinely new entity rows.
                    let entity_context =
                        extract_context_snippet(text, &extracted.name, 100);
                    let props_with_context = serde_json::json!({
                        "context": entity_context,
                        "name": extracted.name.clone(),
                    });

                    // TD-021 open vocabulary: register novel type label on the fly.
                    // Unlike the stub-promotion site, the insert-new path is NOT
                    // best-effort — registration failure propagates as a hard
                    // ingest failure (matches the existing insert_entity_with_group
                    // error path that breaks 'phases below).
                    let entity_type_id = match crate::core::entity_types::label_to_id_or_register(
                        crate::core::entity_types::LabelToIdOrRegisterParams {
                            conn: &self.graph.conn,
                            group_id: group_id.unwrap_or("default"),
                            registry: &registry,
                            label: &label,
                        },
                    )
                    .await
                    {
                        Ok(id) => id,
                        Err(e) => break 'phases Err(e),
                    };
                    if let Err(e) = self
                        .graph
                        .insert_entity_with_group(InsertEntityWithGroupParams {
                            id: &entity_id,
                            entity_type_id,
                            properties: props_with_context,
                            group_id,
                        })
                        .await
                    {
                        // TD-168: a UNIQUE violation here means the row ALREADY
                        // EXISTS — benign, and exactly what the sibling stub
                        // path at :1114 already concluded ("that is fine — a
                        // real row is present"). This path previously failed the
                        // ENTIRE ingest on it, because the raw libsql error was
                        // indistinguishable from a genuine DB failure. Observed
                        // 2026-07-29: 1 of 8 LongMemEval sessions HTTP 500'd on
                        // `UNIQUE constraint failed: entities.id, entities.group_id`.
                        //
                        // The reachable cause is the extractor emitting the same
                        // NORMALISED name twice within one episode — this path
                        // believes the entity is new because it just decided so.
                        //
                        // NOT a blanket swallow (Rule 8): only the unique case
                        // is tolerated, and it is COUNTED so the rate stays
                        // visible. Every other error still fails the ingest.
                        if e.is_unique_violation() {
                            metrics::counter!(
                                "kremory.ingest.entity_insert_duplicate_tolerated_total",
                                "via" => "insert_new",
                            )
                            .increment(1);
                            tracing::warn!(
                                entity_id = %entity_id,
                                "kremory.ingest.entity_insert_duplicate — row already \
                                 exists; continuing (TD-168). Extractor likely emitted \
                                 the same normalised name twice in one episode."
                            );
                        } else {
                            break 'phases Err(e);
                        }
                    }
                    metrics::counter!(
                        "rql.ingest.entity_persisted_total",
                        "source" => "llm",
                        "via" => "insert_new",
                    )
                    .increment(1);

                    // ADR-045 §6 / spec §1.1: persist GLiNER span confidence when present.
                    // `properties["confidence"]` is written by ner.rs for Phase 1 GLiNER
                    // extractions; absent on LLM-only paths. Best-effort — warn only.
                    if let Some(conf_val) = extracted.properties.get("confidence") {
                        if let Some(conf_f64) = conf_val.as_f64() {
                            let conf_f32 = conf_f64 as f32;
                            if let Err(e) = self
                                .graph
                                .set_entity_ner_confidence(SetEntityNerConfidenceParams {
                                    id: &entity_id,
                                    group_id,
                                    confidence: conf_f32,
                                })
                                .await
                            {
                                tracing::warn!(
                                    entity_id = %entity_id,
                                    error = %e,
                                    "kremory.ingest.ner_confidence_write_failed"
                                );
                            }
                        }
                    }

                    // Embed and store the entity name vector for L4 disambiguation probing.
                    // TD-143: this is a WRITE into `entities.embedding` — document-prefix it.
                    let embedding = match self
                        .embedder
                        .embed(&document_embed_text(
                            &extracted.name,
                            self.config.search.embed_task_prefix_enabled,
                        ))
                        .await
                    {
                        Ok(v) => v,
                        Err(e) => break 'phases Err(e),
                    };
                    if let Err(e) = self
                        .graph
                        .set_entity_embedding(&entity_id, &embedding)
                        .await
                    {
                        break 'phases Err(e);
                    }

                    // ── L4 PotentialAlias: record the meta-edge fact ─────────────────────
                    // Best-effort — alias recording failure does not abort ingest.
                    if let crate::core::disambiguation::DisambiguationOutcome::PotentialAlias {
                        existing_id: ref alias_target_id,
                        similarity: alias_sim,
                    } = l4_outcome
                    {
                        // `.ok()` — best-effort; duplicate alias entries are swallowed
                        // inside `insert_potential_alias_fact`.
                        crate::core::disambiguation::insert_potential_alias_fact(
                            crate::core::disambiguation::InsertPotentialAliasFactParams {
                                graph: &self.graph,
                                new_entity_id: &entity_id,
                                existing_id: alias_target_id,
                                similarity: alias_sim,
                                provenance: crate::core::disambiguation::AliasProvenance {
                                    source_episode_id: Some(episode_id),
                                    group_id,
                                },
                            },
                        )
                        .await
                        .ok();
                    }

                    // Bug A: episodic edge for newly inserted entity.
                    // ADR-052 Gap 1 fire-sites: on_entity_extracted + on_edge_added("mention").
                    // Migration 006: thread `group_id` — the entity was just inserted
                    // via `insert_entity_with_group(.., group_id)` above, so the edge
                    // must reference the same namespace for the composite FK to resolve.
                    let mention_ok = self
                        .graph
                        .insert_episodic_edge(InsertEpisodicEdgeParams {
                            episode_id,
                            entity_id: &entity_id,
                            entity_group_id: group_id,
                            role: "mention",
                        })
                        .await
                        .is_ok();
                    if let Some(s) = sink {
                        s.on_entity_extracted(&entity_id, &extracted.name);
                        if mention_ok {
                            s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                                from_entity_id: &episode_id.to_string(),
                                to_entity_id: &entity_id,
                                predicate: "mention",
                            });
                        }
                    }
                    fire_entity_edge_metrics(mention_ok);
                    tracing::debug!(entity_id = %entity_id, name = %extracted.name, episode_id, via = "insert_new", "kremory.sink.entity_extracted");
                    if mention_ok {
                        entity_loop_ids.insert(entity_id.clone());
                    }

                    upserted_entities.push(entity_id.clone());
                    entity_id
                };

                name_to_id.insert(normalize_name(&extracted.name), entity_id);
            }

            // ── Phase 2: detect contradictions and store facts ───────────────────────
            //
            // TD-167 / ADR-079 rev.2: contradiction detection is DEFAULT-ON again.
            // It was default-OFF for ~4h on 2026-07-29 while it treated every
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
                // Resolve subject and object IDs through the merge map
                let subject_id = name_to_id
                    .get(&normalize_name(&fact.subject))
                    .cloned()
                    .unwrap_or_else(|| normalize_name(&fact.subject));

                let object_id = if fact.is_entity_ref {
                    Some(
                        name_to_id
                            .get(&normalize_name(&fact.object))
                            .cloned()
                            .unwrap_or_else(|| normalize_name(&fact.object)),
                    )
                } else {
                    None
                };

                let object_value = if !fact.is_entity_ref {
                    Some(fact.object.as_str())
                } else {
                    None
                };

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
                // TD-165: keep only candidates that SHARE AN ENTITY with the new
                // fact. The FTS above matches on the PREDICATE alone, so without
                // this it returns other subjects' facts entirely — "Alice likes
                // tea" pulls in "Bob likes coffee" and buys a full LLM
                // round-trip to ask whether they contradict. They cannot.
                //
                // Measured before adding this filter (Groq, 8 sessions,
                // 2026-07-29, counters below):
                //   pool_a contributed to 85 LLM-reachable checks -> 33 contradictions + 4 duplicates
                //   pool_b contributed to 63 LLM-reachable checks ->  0 contradictions,  0 duplicates
                // 0/63 gives a 95% CI upper bound of 4.8% on pool_b's hit rate,
                // against ~43% for pool_a. It was pure cost.
                //
                // This is a COST fix, not a capability removal: same-subject
                // contradiction is unaffected (that is pool_a's job, and pool_b
                // candidates sharing the subject survive the filter). What it
                // drops is the CROSS-ENTITY case, which was never designed —
                // the pool_b search is documented above as a "semantically
                // related facts" recall heuristic, and genuine cross-entity
                // contradiction ("X is CEO of Acme" vs "Y is CEO of Acme")
                // needs predicate CARDINALITY, which kremory does not model.
                // Build that deliberately if wanted; do not leave it as an
                // accident of a text search. See TD-165.
                //
                // The `contradiction_outcome_total{source=...}` counter added
                // alongside this keeps the decision falsifiable: if pool_b ever
                // starts earning its keep, `source="pool_b"` will show it.
                let pool_b: Vec<crate::core::schema::Fact> = pool_b_hits
                    .into_iter()
                    .map(|h| h.item)
                    .filter(|f| {
                        let shares_subject = f.subject_id == subject_id;
                        let shares_object = match (&f.object_id, &object_id) {
                            (Some(a), Some(b)) => a == b,
                            _ => false,
                        };
                        // A candidate whose OBJECT is our SUBJECT (or vice
                        // versa) is still about the same entity — keep it.
                        let cross_ref = f.object_id.as_deref() == Some(subject_id.as_str())
                            || object_id.as_deref() == Some(f.subject_id.as_str());
                        shares_subject || shares_object || cross_ref
                    })
                    .collect();

                // Run contradiction detection (TD-167: skipped when disabled —
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

                // ── TD-165 o11y: did this check EARN its LLM call? ──────────────
                //
                // Contradiction detection is 38.4% of all ingest LLM calls
                // (361/941 measured over 20 real sessions, 2026-07-29) and had
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
                // Cardinality is bounded: 3 outcomes x 3 sources = 9 series.
                {
                    let a_ids: std::collections::HashSet<i64> =
                        pool_a.iter().map(|f| f.id).collect();
                    let (mut from_a, mut from_b) = (false, false);
                    for id in contradiction_result
                        .contradictions
                        .iter()
                        .chain(contradiction_result.duplicates.iter())
                    {
                        if a_ids.contains(id) {
                            from_a = true;
                        } else {
                            from_b = true;
                        }
                    }
                    let source = match (from_a, from_b) {
                        (true, true) => "mixed",
                        (true, false) => "pool_a",
                        (false, true) => "pool_b",
                        (false, false) => "none",
                    };
                    let outcome = if !contradiction_result.contradictions.is_empty() {
                        "contradiction"
                    } else if !contradiction_result.duplicates.is_empty() {
                        "duplicate"
                    } else {
                        "no_conflict"
                    };
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

                    // ── ADR-052 Gap 1 fire-site: on_contradiction ───────────────────
                    // The new `fact` superseded a prior fact (`*fact_id`) that lives in
                    // pool_a/pool_b (both already fetched above). Build a faithful
                    // ContradictionDetected from the prior fact snapshot + the new triple.
                    // Resolution is always `Superseded` on this path (the prior fact was
                    // invalidated in favour of the new one — there is no Retained/Merged
                    // branch in `ingest_with`). Best-effort: if the prior fact cannot be
                    // located in the pools, skip the sink call rather than fabricate a
                    // payload (parse-loudly: no silent defaults for load-bearing fields).
                    if let Some(s) = sink {
                        if let Some(prior) = pool_a
                            .iter()
                            .chain(pool_b.iter())
                            .find(|f| f.id == *fact_id)
                        {
                            use crate::core::sink::{ContradictionDetected, EntityId, SinkFact};
                            // An empty object_id is LEGITIMATE here, not a silent
                            // default / parse-loudly violation (Rule 21): a fact may
                            // be a unary predicate / objectless triple where BOTH
                            // `object_id` and `object_value` are absent. `unwrap_or_default()`
                            // yields "" for that case, which is the correct faithful
                            // representation of an objectless prior fact in the sink
                            // payload — there is no missing-required-field to surface.
                            let prior_object = prior
                                .object_id
                                .clone()
                                .or_else(|| prior.object_value.clone())
                                .unwrap_or_default();
                            s.on_contradiction(ContradictionDetected {
                                entity_id: EntityId(subject_id.clone()),
                                prior_fact: SinkFact {
                                    subject: prior.subject_id.clone(),
                                    predicate: prior.predicate.clone(),
                                    object: prior_object,
                                    valid_at: Some(prior.valid_from),
                                },
                                new_fact: SinkFact {
                                    subject: fact.subject.clone(),
                                    predicate: fact.predicate.clone(),
                                    object: fact.object.clone(),
                                    valid_at: Some(ref_time),
                                },
                                resolution: crate::core::error::ContradictionResolution::Superseded,
                                detected_at: Utc::now(),
                            });
                        }
                    }
                    metrics::counter!(
                        "kremory.sink.contradiction_total",
                        "resolution" => "Superseded"
                    )
                    .increment(1);
                    tracing::debug!(prior_fact_id = *fact_id, episode_id, "kremory.sink.contradiction");
                }

                // F5 — within-episode contradiction pre-check (SQL-only).
                // If pool_a already contains a non-expired fact for this subject+predicate
                // that was created BY THIS episode (`source_episode_id == episode_id`), a
                // same-episode ingest round is about to emit a contradicting triple.
                // Emit the observability counter and skip insertion — the first fact wins.
                // This is purely a client-side filter on data already fetched; no extra
                // DB round-trip is needed.
                let within_episode_conflict = pool_a
                    .iter()
                    .any(|f| f.source_episode_id == Some(episode_id));
                if within_episode_conflict {
                    let ns = group_id.unwrap_or("default");
                    metrics::counter!(
                        "rql.ingest.within_episode_contradiction",
                        "namespace" => ns.to_string()
                    )
                    .increment(1);
                    tracing::debug!(
                        subject = %subject_id,
                        predicate = %fact.predicate,
                        episode_id,
                        namespace = %ns,
                        "kremory.ingest.within_episode_contradiction: skipping duplicate triple"
                    );
                    continue;
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
                // group_id=None vs facts=0 at group_id=Some, 2026-06-29.)
                //
                // ADR-035 §5 Option A: the `_with_group` variant still dedups
                // caller-pinned `with_facts` triples (caller wins by being there first).
                match self
                    .graph
                    .try_insert_fact_with_group(
                        FactInsert {
                            subject_id: &subject_id,
                            predicate: &fact.predicate,
                            object_id: object_id.as_deref(),
                            object_value,
                            valid_from: ref_time,
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
                        // TD-143: WRITE into `facts.embedding` — document-prefix it.
                        let fact_text =
                            format!("{} {} {}", fact.subject, fact.predicate, fact.object);
                        let prefixed_fact_text = document_embed_text(
                            &fact_text,
                            self.config.search.embed_task_prefix_enabled,
                        );
                        if let Ok(embedding) = self.embedder.embed(&prefixed_fact_text).await {
                            self.graph
                                .set_fact_embedding(fact_id, &embedding)
                                .await
                                .ok();
                        }
                        inserted_fact_ids.push(fact_id);
                    }
                    Ok(None) => {
                        // ADR-035 Path X: caller pre-pinned this triple via with_facts;
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
                        // Rule 19: a swallowed fact-insert failure (e.g. FK constraint) was
                        // previously only a tracing::warn — invisible in aggregate, which hid
                        // the composite-FK namespace bug (facade facts silently dropped). Emit
                        // a counter so dropped facts are observable, not silent.
                        //
                        // B2 observability hardening (2026-07-21): this bare counter gave the
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
                        // cardinality). TD-133 B3 follow-up.
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

                // Bug A: subject-side episodic edge is written in the entity loop
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
                        // ADR-052 Gap 1 fire-site: on_edge_added("object"). Fires on Ok
                        // only (insert error is soft `.ok()` precedent). D7: ids in
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
                            if let Some(s) = sink {
                                s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                                    from_entity_id: &episode_id.to_string(),
                                    to_entity_id: obj_id,
                                    predicate: "object",
                                });
                            }
                            metrics::counter!(
                                "kremory.sink.edge_added_total",
                                "predicate_kind" => "object"
                            )
                            .increment(1);
                            tracing::debug!(entity_id = %obj_id, episode_id, predicate = "object", "kremory.sink.edge_added");
                        }
                    }
                }
            }

            Ok(())
        }; // end 'phases block

        // Commit or rollback the outer transaction based on phase result.
        //
        // ── ADR-052 Gap 1 fire-sites: terminal stage transitions ─────────────────
        // Re-establishes 737e152 verify_stage Fire-sites 4 (EntitiesReady) + 5
        // (Failed) on the unified inline routine, plus Complete. Successful
        // sequence per IngestStatus doc: Extracting → EntitiesReady → Complete.
        // On error: Failed. Triple-emit at each (sink + counter + tracing). D7:
        // episode_id is a tracing field only, NEVER a metric label. Failed-arm
        // tracing is `tracing::error!` (ADR-052 §3.1 row 12 / MED-01 precedent).
        match phase_result {
            Ok(()) => {
                outer_guard.commit().await?;
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
                // every other post-`Extracting` `?` failure uniformly — Quinn
                // MED-01. Do NOT fire `Failed` here too, or it would double-emit.
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
        // TD-019 Gap 2: phase1_ms exposes the sync ingest-with-extraction phase
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
        })
        }
        .await;

        // ── Single Failed-fire exit (ADR-051 §4 / Quinn MED-01) ─────────────────
        // The async block above captured every fallible step after the `Extracting`
        // fire. On ANY `Err`, fire `Failed` exactly once (sink + counter + tracing)
        // BEFORE re-propagating — the error is NOT swallowed.
        match ingest_outcome {
            Ok(result) => Ok(result),
            Err(e) => {
                fire_failed(&e);
                // TD-168 (second defect): persist Failed. Previously the inline
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
                         Pending and is indistinguishable from in-flight (TD-168)"
                    );
                }
                Err(e)
            }
        }
    }

    /// TD-113: make a caller-pinned entity recall-findable AND attributable
    /// WITHOUT a second LLM. Recall seeds on entities (`hybrid_search_entities`)
    /// then renders via the DEFAULT `TemporalFacts` template — so a pin must
    /// satisfy THREE channels to actually surface, all mirroring mode-(a):
    ///
    /// 1. **FTS** — the literal name is stamped into `properties["name"]` at the
    ///    pin call site (`insert_entity_with_group`); the FTS seed arm indexes
    ///    `entities_fts.properties` (label is empty post-Migration-009). This is
    ///    the channel that works even under a null embedder.
    /// 2. **Vector** — embed the literal name into `entities.embedding`. The
    ///    embedder is NOT the chat LLM, so spec §3 F1 "no second LLM" holds.
    /// 3. **Attribution** — link the entity to its source episode via an episodic
    ///    edge (`role="mention"`). Without it the entity has zero `source_refs`,
    ///    and the default `TemporalFacts` renderer (which emits output ONLY per
    ///    source_ref) renders a found entity to `""` — invisible despite the FTS
    ///    hit. This is the piece the skip-extraction early-return skipped (all
    ///    other `insert_episodic_edge` calls run AFTER it).
    ///
    /// All best-effort per [[observability-first-class]] (success + failure both
    /// emit a labeled signal): a null/failing embedder still leaves the entity
    /// FTS-findable + attributed. `set_entity_embedding` + `insert_episodic_edge`
    /// are UPDATE/INSERT-OR-IGNORE, so they also cover entities that pre-existed
    /// as bare stubs (the FTS-name INSERT, by contrast, is skipped on Duplicate).
    async fn make_pinned_entity_recallable(&self, p: PinnedEntityRecall<'_>) {
        let PinnedEntityRecall {
            id,
            group_id,
            episode_id,
        } = p;
        // Channel 2 — vector. TD-143: WRITE into `entities.embedding` — document-prefix it.
        match self
            .embedder
            .embed(&document_embed_text(
                id,
                self.config.search.embed_task_prefix_enabled,
            ))
            .await
        {
            Ok(embedding) => {
                if let Err(e) = self.graph.set_entity_embedding(id, &embedding).await {
                    metrics::counter!("kremory.with_facts.pinned_embedding_stamp_failed")
                        .increment(1);
                    tracing::warn!(
                        entity_id = %id,
                        error = %e,
                        "kremory.with_facts.pinned_embedding_stamp_failed"
                    );
                } else {
                    metrics::counter!("kremory.with_facts.pinned_entity_embedded_total")
                        .increment(1);
                }
            }
            Err(e) => {
                metrics::counter!("kremory.with_facts.pinned_embedding_failed").increment(1);
                tracing::warn!(
                    entity_id = %id,
                    error = %e,
                    "kremory.with_facts.pinned_embedding_failed"
                );
            }
        }

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
    }
}
