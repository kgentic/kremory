use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use metrics::histogram;

use crate::core::config::ContentType;
use crate::core::contradiction::{DetectParams, TwoPoolDetector};
use crate::core::embed_prefix::document_embed_text;
use crate::core::entity_types::EntityTypeRegistry;
use crate::core::error::{ContradictionResolution, IngestStatus};
use crate::core::extraction_window::ExtractionWindowSplitter;
use crate::core::graph::{
    FactInsert, InsertEpisodicEdgeParams, InvalidateFactWithReasonParams, PriorEpisodesParams,
};
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext,
};
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::core::resolver::normalize_name;
use crate::core::search::{FtsSearchFactsParams, SearchFilters};
use crate::core::sink::{ContradictionDetected, EntityId, IngestEventSink, SinkFact};

use crate::core::ingest::Engine;

/// Bundled parameters for [`Engine::ingest_deferred`] — args-as-object to
/// keep the function under clippy's too_many_arguments threshold.
pub struct IngestDeferredParams<'a> {
    pub text: &'a str,
    pub reference_time: Option<DateTime<Utc>>,
    /// The caller-DECLARED document anchor (`SourceRef::published_at`)
    /// ONLY — never `reference_time` / `occurred_at` / wall-clock. Forwarded
    /// into [`crate::core::intelligence::ExtractionContext::reference_time`]
    /// below. Currently always `None` on this background-worker path: the
    /// `BackgroundIngestor` plumbing (`IngestRequest` / `DeferredRequest`)
    /// does not carry a caller-declared publish time distinct from
    /// `reference_time` today — a pre-existing gap in the background-ingestor
    /// surface; the inline `EngineGraphHandle::graph_ingest_episode` path
    /// (`memory/engine_handle.rs`) wires it. See that field's doc comment for
    /// the VCR-fingerprint hazard this separation exists to avoid.
    pub declared_reference_time: Option<DateTime<Utc>>,
    pub group_id: Option<&'a str>,
    pub content_type: Option<ContentType>,
    pub episode_id: i64,
    pub ner_entity_names: &'a [String],
    pub sink: Option<&'a dyn IngestEventSink>,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Phase 2 deferred LLM fact extraction.
    ///
    /// Called by the background worker when the NER channel is idle.  Takes the
    /// same text that was processed in Phase 1 along with the entity names that
    /// NER already inserted, runs the LLM extractor to discover relationship
    /// triplets, and stores them linked to the existing episode.
    ///
    /// Returns the number of facts successfully inserted.
    ///
    /// Entity insertion is skipped — Phase 1 (NER) already owns that path.
    /// Only facts (relationship triplets) are added in this phase.
    ///
    /// `sink` receives per-event callbacks during deferred fact extraction.
    /// Fires: `on_stage_change(Deduplicating)` once on entry (when ≥1 fact
    /// extracted), `on_stage_change(Invalidating)` once on first Superseded
    /// contradiction, `on_contradiction` per resolved contradiction,
    /// `on_edge_added` after each episodic edge insert (subject + object sites).
    // Substrate primitive; the consumer-facing surface is the kremory::Memory facade.
    pub async fn ingest_deferred(
        &self,
        params: IngestDeferredParams<'_>,
    ) -> crate::core::error::Result<usize> {
        let IngestDeferredParams {
            text,
            reference_time,
            declared_reference_time,
            group_id,
            content_type,
            episode_id,
            ner_entity_names,
            sink,
        } = params;
        // phase2_ms — deferred relationship extraction latency,
        // distinct from phase1_ms (user-facing sync work).
        let phase2_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);

        // Build known_entities hint list from the NER entity names.
        let known_entities: Vec<ExtractedEntity> = ner_entity_names
            .iter()
            .map(|name| ExtractedEntity {
                label: String::new(),
                name: name.clone(),
                properties: serde_json::Value::Null,
            })
            .collect();

        // Run LLM extractor to obtain relationship triplets.
        let extractor = Arc::clone(&self.extractor);
        let splitter = ExtractionWindowSplitter::new(self.config.extraction_window.clone());
        let chunks = splitter.split(text, &content_type);

        // L2: load registry so deferred-fact extraction can inject registry specs
        // into prompts. Uses the same effective_gid as the primary ingest path.
        let deferred_effective_gid = group_id.unwrap_or("default");
        let deferred_registry =
            EntityTypeRegistry::load_for_group(&self.graph.conn, deferred_effective_gid).await?;

        // Deferred path: derive allowed_entity_types from the live
        // deferred_registry, mirroring the primary ingest path derivation above.
        let deferred_allowed_entity_types_live: Vec<String> = deferred_registry
            .specs()
            .iter()
            .map(|s| s.name.clone())
            .collect();

        // Prior-turn replay on the DEFERRED path. `IngestDeferredParams`
        // does not carry the source id (same plumbing gap its
        // `declared_reference_time` doc records), so resolve it from the episode
        // row rather than letting background ingests silently skip replay —
        // extraction quality must not depend on which path the caller took.
        let prior_turns: Vec<String> = match self.graph.source_id_for_episode(episode_id).await? {
            Some(source_id) if !source_id.is_empty() => {
                self.graph
                    .prior_episodes_for_source(PriorEpisodesParams {
                        source_id: &source_id,
                        group_id,
                        before_id: episode_id,
                        limit: self.config.prior_turn_replay_depth,
                    })
                    .await?
            }
            _ => Vec::new(),
        };

        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &deferred_allowed_entity_types_live,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &known_entities,
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: content_type.clone(),
                registry_specs: deferred_registry.specs(),
                // Deferred-fact extraction is relationship-only (entities are
                // pre-supplied as `ner_entity_names`); existing-entity injection
                // is not needed here — the NER names already serve as the anchor.
                existing_graph_entities: &[],
                arm_budget_ms: self.config.extraction_arm_budget_ms,
                model: self.model.as_deref(),
                reference_time: declared_reference_time,
                prior_turns: &prior_turns,
            };
            let result = extractor.extract(chunk, &ctx).await?;
            all_facts.extend(result.facts);
        }

        if all_facts.is_empty() {
            histogram!("rql.ingest.deferred_fact_count").record(0.0);
            tracing::info!(
                deferred_fact_count = 0,
                "kremory.ingest.deferred_facts empty"
            );
            return Ok(0);
        }

        // Build a name → entity_id map by resolving against the existing graph (namespace-scoped).
        let existing_entities = match group_id {
            Some(gid) => self.graph.list_entities_in_group(gid).await?,
            None => self.graph.list_entities().await?,
        };
        let mut name_to_id: HashMap<String, String> = HashMap::new();
        for entity in &existing_entities {
            // entity.label is the type name ("Person", "Entity") — not the entity name.
            // Use entity.id (normalized name) + properties["name"] (original case) as
            // lookup keys so fact subject/object resolution finds existing entities.
            if let Some(name_val) = entity.properties.get("name").and_then(|v| v.as_str()) {
                name_to_id.insert(normalize_name(name_val), entity.id.clone());
            }
            name_to_id.insert(normalize_name(&entity.id), entity.id.clone());
        }
        // Also seed the map with the NER entity names directly so facts that
        // reference them by their original surface form are resolved correctly.
        for name in ner_entity_names {
            let norm = normalize_name(name);
            name_to_id
                .entry(norm)
                .or_insert_with(|| normalize_name(name));
        }

        // Store facts (contradiction detection + insert), same logic as ingest_with.
        //
        // Gated identically to `ingest_with`. This is the SECOND call
        // site — gating only the first would leave the deferred/background
        // ingest path still destroying set-valued facts, which is exactly the
        // fork-shaped bug this codebase keeps hitting. Facts are still stored;
        // only the invalidation and its LLM call are skipped.
        let detector = if self.config.contradiction_detection_enabled {
            let llm_for_batch_detector =
                self.llm
                    .as_ref()
                    .ok_or_else(|| crate::core::error::Error::LlmRequired {
                        method: "ingest_batch",
                        hint: "wire an LLM via Memory::open(…).with_llm(…) to enable \
                           contradiction detection; or use .with_facts(…) to pin \
                           triples without LLM",
                    })?;
            Some(
                TwoPoolDetector::new(Arc::clone(llm_for_batch_detector))
                    .with_model(self.model.clone()),
            )
        } else {
            None
        };
        let mut inserted_count: usize = 0;

        // Fire-once flags:
        //   fired_deduplicating — on_stage_change(Deduplicating) fires ONCE per call,
        //     on entry to the contradiction-detection loop (when ≥1 fact extracted).
        //   fired_invalidating  — on_stage_change(Invalidating) fires ONCE per call,
        //     on the FIRST Superseded contradiction resolution.
        // D7: entity_id / episode_id / fact_id NEVER used as metric labels.
        let mut fired_deduplicating = false;
        let mut fired_invalidating = false;

        // Inline helper: bounded ContradictionResolution discriminant string.
        // DO NOT use format!("{:?}", r) — would expose struct interior + bust cardinality.
        // Within-crate matching is exhaustive for #[non_exhaustive] enums; no wildcard needed.
        let resolution_as_str = |r: &ContradictionResolution| -> &'static str {
            match r {
                ContradictionResolution::Superseded => "Superseded",
                ContradictionResolution::Retained => "Retained",
                ContradictionResolution::Merged => "Merged",
            }
        };

        for fact in &all_facts {
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

            // TD-254: scope both candidate pools to this episode's own namespace
            // — see the sibling call site in ingest_with.rs for the full
            // rationale (unscoped pools can leak facts across namespaces).
            let pool_a = self
                .graph
                .get_facts_by_subject_predicate(
                    crate::core::graph::GetFactsBySubjectPredicateParams {
                        subject_id: &subject_id,
                        predicate: &fact.predicate,
                        group_id: deferred_effective_gid,
                    },
                )
                .await?;

            let pool_b_hits = self
                .graph
                .fts_search_facts(FtsSearchFactsParams {
                    query: &fact.predicate,
                    limit: 10,
                    filters: &SearchFilters::for_group(deferred_effective_gid),
                })
                .await?;
            let pool_b: Vec<crate::core::schema::Fact> =
                pool_b_hits.into_iter().map(|h| h.item).collect();

            // ── Fire-site 1: on_stage_change(Deduplicating) ──────────────────────
            // Fires ONCE per ingest_deferred call, on entry to the
            // contradiction-detection loop when ≥1 fact is extracted. Fires regardless
            // of whether contradictions are actually found.
            // callback_duration_ms wraps on_stage_change (slow-consumer detection).
            if !fired_deduplicating {
                let cb_start = Instant::now();
                if let Some(s) = sink {
                    s.on_stage_change(IngestStatus::Deduplicating);
                }
                metrics::counter!(
                    "kremory.sink.stage_transition_total",
                    "from" => "EntitiesReady",
                    "to" => "Deduplicating"
                )
                .increment(1);
                metrics::histogram!(
                    "kremory.sink.callback_duration_ms",
                    "callback" => "on_stage_change",
                    "stage" => "Deduplicating"
                )
                .record(cb_start.elapsed().as_secs_f64() * 1000.0);
                tracing::info!(episode_id, "kremory.background.stage_change.deduplicating");
                fired_deduplicating = true;
            }

            // Skipped when disabled — no LLM call, no invalidation.
            let contradiction_result = match &detector {
                Some(d) => {
                    d.detect(DetectParams {
                        new_fact: fact,
                        pool_a: &pool_a,
                        pool_b: &pool_b,
                        reference_time: &ref_time,
                    })
                    .await?
                }
                None => crate::core::contradiction::ContradictionResult::no_conflicts(),
            };

            // Build combined pool for prior-fact lookup (used by on_contradiction below).
            let all_pool: Vec<&crate::core::schema::Fact> =
                pool_a.iter().chain(pool_b.iter()).collect();

            for fact_id in &contradiction_result.contradictions {
                // ── Fire-site 3: on_stage_change(Invalidating) ───────────────────
                // Fires ONCE per call on the FIRST Superseded contradiction.
                // Resolution is always Superseded here — invalidate_fact_with_reason
                // marks the prior fact invalid_at (bi-temporal supersession path).
                // callback_duration_ms wraps on_stage_change (slow-consumer detection).
                if !fired_invalidating {
                    let cb_start = Instant::now();
                    if let Some(s) = sink {
                        s.on_stage_change(IngestStatus::Invalidating);
                    }
                    metrics::counter!(
                        "kremory.sink.stage_transition_total",
                        "from" => "Deduplicating",
                        "to" => "Invalidating"
                    )
                    .increment(1);
                    metrics::histogram!(
                        "kremory.sink.callback_duration_ms",
                        "callback" => "on_stage_change",
                        "stage" => "Invalidating"
                    )
                    .record(cb_start.elapsed().as_secs_f64() * 1000.0);
                    tracing::info!(episode_id, "kremory.background.stage_change.invalidating");
                    fired_invalidating = true;
                }

                // `invalid_at: ref_time` is CORRECT — see the full classification at
                // the `ingest_with.rs` site. `Fact.invalid_at` is the resolver's
                // invalidation timestamp, NOT world time, and nothing filters on it.
                self.graph
                    .invalidate_fact_with_reason(InvalidateFactWithReasonParams {
                        fact_id: *fact_id,
                        expired_at: Utc::now(),
                        invalid_at: ref_time,
                    })
                    .await?;

                // ── Fire-site 4: on_contradiction ────────────────────────────────
                // Per-contradiction basis (loop). Fires for each resolved contradiction
                // regardless of resolution variant. Resolution = Superseded: the prior
                // fact was marked invalid_at above.
                let prior_sink_fact =
                    all_pool
                        .iter()
                        .find(|f| f.id == *fact_id)
                        .map(|f| SinkFact {
                            subject: f.subject_id.clone(),
                            predicate: f.predicate.clone(),
                            object: f
                                .object_value
                                .as_deref()
                                .or(f.object_id.as_deref())
                                .unwrap_or("")
                                .to_string(),
                            valid_at: Some(f.valid_from),
                        });
                let new_sink_fact = SinkFact {
                    subject: subject_id.clone(),
                    predicate: fact.predicate.clone(),
                    object: fact.object.clone(),
                    valid_at: Some(ref_time),
                };
                let resolution = ContradictionResolution::Superseded;
                if let Some(s) = sink {
                    s.on_contradiction(ContradictionDetected {
                        entity_id: EntityId(subject_id.clone()),
                        prior_fact: prior_sink_fact.unwrap_or_else(|| SinkFact {
                            subject: subject_id.clone(),
                            predicate: fact.predicate.clone(),
                            object: String::new(),
                            valid_at: None,
                        }),
                        new_fact: new_sink_fact,
                        resolution: resolution.clone(),
                        detected_at: Utc::now(),
                    });
                }
                metrics::counter!(
                    "kremory.sink.contradiction_total",
                    "resolution" => resolution_as_str(&resolution)
                )
                .increment(1);
                tracing::info!(
                    resolution = resolution_as_str(&resolution),
                    episode_id,
                    "kremory.background.contradiction_resolved"
                );
            }

            // Within-episode DUPLICATE pre-check (deferred path).
            //
            // Second copy of the object-agnostic check: the inline path
            // (`ingest_with.rs`) has an identical check; this copy lives here too,
            // and this is the path `Memory::remember()` actually drives (Path β),
            // so this copy is the one most consumers hit.
            //
            // Must match on the FULL TRIPLE: `pool_a` is object-agnostic, so comparing
            // subject+predicate alone dropped every value but the first for a
            // set-valued predicate asserted once in a single episode.
            let deferred_within_duplicate = pool_a.iter().any(|f| {
                f.source_episode_id == Some(episode_id)
                    && f.object_id.as_deref() == object_id.as_deref()
                    && f.object_value.as_deref() == object_value
            });
            if deferred_within_duplicate {
                let ns = group_id.unwrap_or("default");
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
                    "kremory.ingest.within_episode_duplicate_triple: deferred path skipping exact repeat"
                );
                continue;
            }

            // DETECTION intent, preserved additively — deferred path.
            // Mirrors the inline path exactly (see `ingest_with.rs` for the full
            // reasoning): fires when ONE episode asserts multiple DISTINCT objects for
            // the same subject+predicate. The value is now STORED rather than dropped;
            // this counter is what makes the phenomenon measurable so contradiction
            // policy can later be designed against real counts. Parity matters here in
            // particular because this is the path `Memory::remember()` drives.
            let deferred_within_multivalue = pool_a
                .iter()
                .any(|f| f.source_episode_id == Some(episode_id));
            if deferred_within_multivalue {
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
                    "kremory.ingest.within_episode_multivalue: deferred path — stored, not dropped"
                );
            }

            // Caller-pin dedup parity with the inline path.
            // MUST be `_with_group(.., Some(deferred_effective_gid))`: entities are
            // upserted into this namespace (entity_group_id: group_id below), and the
            // facts composite FK (subject_id, subject_group_id) → entities(id, group_id)
            // (schema.rs:1450) fails — silently dropping the fact — if the group columns
            // aren't stamped with the entities' namespace. This is the same fix as the
            // inline path (ingest_with.rs); the deferred/background path had the identical
            // bug.
            match self
                .graph
                .try_insert_fact_with_group(
                    FactInsert {
                        subject_id: &subject_id,
                        predicate: &fact.predicate,
                        object_id: object_id.as_deref(),
                        object_value,
                        // Mirrors the inline path in `ingest_with.rs`.
                        // Both fact-insert sites MUST apply the same fallback; a divergence
                        // here would make a fact's world time depend on whether extraction
                        // ran inline or deferred, which the consumer cannot observe or control.
                        valid_from: fact.valid_at.unwrap_or(ref_time),
                        confidence: fact.confidence,
                        source_episode_id: Some(episode_id),
                        embedding: None,
                    },
                    Some(deferred_effective_gid),
                )
                .await
            {
                Ok(Some(fact_id)) => {
                    // WRITE into `facts.embedding` — document-prefix it.
                    let fact_text = format!("{} {} {}", fact.subject, fact.predicate, fact.object);
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
                    // Migration 006: thread `group_id` — deferred-phase entities were
                    // persisted under this namespace by Phase 1, so the episodic edge
                    // must reference the same namespace for the composite FK to resolve.
                    if name_to_id.contains_key(&normalize_name(&fact.subject)) {
                        // ── Fire-site 5a: on_edge_added (subject) ────────────────────
                        // Guard sink on is_ok() — mirrors verify_stage.rs stage3_write
                        // pattern. Sink MUST NOT fire for edges that were never written.
                        // predicate_kind = "fact" distinguishes from Phase 3a mention edges.
                        // D7: episode_id not used as label; carried in tracing field only.
                        // Migration 006: thread `group_id` — deferred-phase entities were
                        // persisted under this namespace by Phase 1, so the episodic edge
                        // must reference the same namespace for the composite FK to resolve.
                        if self
                            .graph
                            .insert_episodic_edge(InsertEpisodicEdgeParams {
                                episode_id,
                                entity_id: &subject_id,
                                entity_group_id: group_id,
                                role: "subject",
                            })
                            .await
                            .is_ok()
                        {
                            if let Some(s) = sink {
                                s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                                    from_entity_id: &episode_id.to_string(),
                                    to_entity_id: &subject_id,
                                    predicate: "subject",
                                });
                            }
                            metrics::counter!(
                                "kremory.sink.edge_added_total",
                                "predicate_kind" => "fact"
                            )
                            .increment(1);
                            tracing::info!(
                                episode_id,
                                predicate = "subject",
                                "kremory.background.edge_added"
                            );
                        }
                    }
                    if let Some(ref obj_id) = object_id {
                        if name_to_id.contains_key(&normalize_name(&fact.object)) {
                            // ── Fire-site 5b: on_edge_added (object) ─────────────────
                            // Guard sink on is_ok() — same pattern as subject site.
                            // predicate_kind = "fact" (same bounded label as subject site).
                            // Migration 006: thread `group_id` for composite-FK parity.
                            if self
                                .graph
                                .insert_episodic_edge(InsertEpisodicEdgeParams {
                                    episode_id,
                                    entity_id: obj_id,
                                    entity_group_id: group_id,
                                    role: "object",
                                })
                                .await
                                .is_ok()
                            {
                                if let Some(s) = sink {
                                    s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                                        from_entity_id: &episode_id.to_string(),
                                        to_entity_id: obj_id,
                                        predicate: "object",
                                    });
                                }
                                metrics::counter!(
                                    "kremory.sink.edge_added_total",
                                    "predicate_kind" => "fact"
                                )
                                .increment(1);
                                tracing::info!(
                                    episode_id,
                                    predicate = "object",
                                    "kremory.background.edge_added"
                                );
                            }
                        }
                    }
                    inserted_count += 1;
                }
                Ok(None) => {
                    tracing::debug!(
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        object = %fact.object,
                        "kremory.ingest.deferred_phase2_fact_dedup_against_prior_pin"
                    );
                }
                Err(e) => {
                    // Label by namespace + reason so the silent
                    // composite-FK drop class is distinguishable from generic failures.
                    // `fk_mismatch` ⇒ the subject/object entity isn't resolvable in this
                    // fact's namespace. Routed through the shared
                    // `Error::fact_insert_failure_reason` so this taxonomy matches
                    // the foreground ingest + graph-layer insert paths — this also
                    // upgrades this path to distinguish `unique_violation` /
                    // `db_error`, which the prior ad-hoc string match collapsed into
                    // `other`.
                    let reason = e.fact_insert_failure_reason();
                    // Bounded-cardinality label only (`reason`): `namespace`
                    // (group_id) is consumer-supplied + unbounded — it stays in
                    // the warn log + KREMORY_DEBUG dump below, never a metric
                    // label (observability-first: no unbounded label cardinality).
                    metrics::counter!(
                        "kremory.ingest.deferred_phase2_fact_insert_failed_total",
                        "reason" => reason,
                    )
                    .increment(1);
                    tracing::warn!(
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        object = %fact.object,
                        namespace = %deferred_effective_gid,
                        reason,
                        error = %e,
                        "kremory.ingest.deferred_phase2_fact_insert_failed"
                    );
                    // KREMORY_DEBUG mismatch detector (matches parsers.rs convention):
                    // surfaces the exact signal needed to diagnose a namespace mismatch in
                    // seconds — the fact's namespace vs its subject/object entity being
                    // absent there. Zero cost when the switch is off.
                    if reason == "fk_mismatch" && std::env::var("KREMORY_DEBUG").is_ok() {
                        tracing::error!(
                            target: "kremory.ingest.namespace_mismatch",
                            fact_namespace = %deferred_effective_gid,
                            subject = %fact.subject,
                            object = %fact.object,
                            "[KREMORY_DEBUG] deferred fact dropped: composite-FK mismatch — \
                             subject/object entity not found in namespace (see fact_namespace). \
                             Compare entity-write namespace via \
                             rql.ingest.entity_persisted_total{{namespace}}."
                        );
                    }
                }
            }
        }

        histogram!("rql.ingest.deferred_fact_count").record(inserted_count as f64);
        histogram!("rql.ingest.phase2_ms").record(phase2_start.elapsed().as_secs_f64() * 1000.0);
        tracing::info!(inserted_count, "kremory.ingest.deferred_facts inserted");
        Ok(inserted_count)
    }
}
