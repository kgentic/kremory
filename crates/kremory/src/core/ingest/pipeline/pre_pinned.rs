//! Writing the caller's PRE-PINNED facts — the `with_facts(..)` path (TD-045 split
//! out of `ingest_with.rs`).
//!
//! These facts arrive already extracted, so nothing here calls a model. The work is
//! all FK plumbing: stub the subject (and object, when it is an entity reference)
//! into the fact's own namespace so the composite foreign key resolves, insert the
//! fact, then make the pinned subject reachable from the episode.
//!
//! Deliberately INFALLIBLE. Every failure is counted and logged, never propagated:
//! a caller-pinned fact that cannot be written must not abort an ingest whose
//! episode is already durable. That is why this returns a plain `Vec<i64>` rather
//! than a `Result` — extracted verbatim, and the shape is worth naming because it
//! was invisible while the code sat inline.

use crate::core::graph::{FactInsert, InsertEntityWithGroupParams, UpdateEntitySourceTierParams};
use crate::core::ingest::{Engine, PrePinnedFact};
use crate::core::provider::{ChatProvider, EmbeddingProvider};

use super::ingest_with::PinnedEntityRecall;

/// Args-as-object for [`Engine::write_pre_pinned_facts`] (clippy.toml
/// `too-many-arguments-threshold = 3`, and `self` counts).
pub(super) struct PrePinnedWriteParams<'a> {
    /// The caller's already-extracted facts, in caller order.
    pub(super) pre_pinned_facts: &'a [PrePinnedFact],
    /// The episode these facts were pinned against — their provenance, and the
    /// anchor `make_pinned_entity_recallable` attaches the subject to.
    pub(super) episode_id: i64,
    /// Namespace for BOTH the stub entities and the facts. Load-bearing: the facts
    /// composite FK is `(subject_id, subject_group_id) -> entities(id, group_id)`,
    /// so a stub written to the default namespace cannot satisfy a fact written to
    /// a named one.
    ///
    /// NB: the episode TEXT is deliberately absent. These facts arrive already
    /// extracted, so nothing on this path reads the body — a fact that had to be
    /// re-derived from the prose would not be pre-pinned.
    pub(super) group_id: Option<&'a str>,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Write the caller's pre-pinned facts and return the ids that landed, plus
    /// (TD-253) the identifiers of any pinned entity whose embedding FAILED — the
    /// entity is still pinned/recallable via BM25 + graph traversal, just not via
    /// dense search. `.1` feeds `IngestionResult::embedding_failures`.
    ///
    /// Infallible by design — see the module doc. The embedding-failure list is an
    /// OBSERVABILITY addition, not a change to that contract: nothing here starts
    /// returning `Err`.
    pub(super) async fn write_pre_pinned_facts(
        &self,
        p: PrePinnedWriteParams<'_>,
    ) -> (Vec<i64>, Vec<String>) {
        let PrePinnedWriteParams {
            pre_pinned_facts,
            episode_id,
            group_id,
        } = p;
        let mut pinned_fact_ids: Vec<i64> = Vec::new();
        let mut embedding_failures: Vec<String> = Vec::new();
        let mut pinned_count: u64 = 0;
        for pf in pre_pinned_facts {
            // Auto-stub the subject entity (entity_type_id=0 "Entity") so the
            // fact insert FK constraint resolves. Mirrors how Phase 2 LLM
            // extraction auto-creates UNKNOWN entities for forward references.
            // Duplicate-entity is the expected case (entity already exists);
            // explicit swallow. Treat the cause, not the symptom:
            // NON-Duplicate errors (connection failure, schema gap, etc) are
            // surfaced via tracing::warn — masking them silently would hide
            // real production failures.
            // Stub entities MUST be created in the fact's `group_id` namespace, not the
            // default one: the facts composite FK is (subject_id, subject_group_id) →
            // entities(id, group_id) (schema.rs:1450). A namespace-less `insert_entity`
            // puts the stub in "default" while the pinned fact below stamps
            // `subject_group_id = group_id`, so the FK fails and the fact is dropped.
            // Stamp the entity's literal name into `properties` so
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
            // Stamp the vector channel too — embed the literal
            // subject text into `entities.embedding` so the vector seed arm
            // finds the pin under a real embedder. Best-effort + always-run
            // (not gated on `skip_extraction`): Phase 2 is not guaranteed to
            // re-cover a pinned subject that never appears in the episode
            // text, so pins must be findable independent of enrichment. The
            // embedder is NOT the chat LLM — "no second LLM" still holds.
            // Also links the entity to its episode (attribution
            // channel) so it renders under the default TemporalFacts template.
            if !self
                .make_pinned_entity_recallable(PinnedEntityRecall {
                    id: &pf.subject,
                    group_id,
                    episode_id,
                })
                .await
            {
                embedding_failures.push(pf.subject.clone());
            }
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
                if !self
                    .make_pinned_entity_recallable(PinnedEntityRecall {
                        id: obj_id,
                        group_id,
                        episode_id,
                    })
                    .await
                {
                    embedding_failures.push(obj_id.clone());
                }
            }

            // Time-inversion guard — mirror `facade/supersede.rs`'s
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
                    // Bind the caller-asserted bounded
                    // window (StructuredFact.valid_to, "None = open-ended")
                    // if given — this used to be silently dropped
                    // (`PrePinnedFact` carried `valid_from` only). Mirrors
                    // `SupersedeRequest`'s own `bound_valid_to` call — NOT
                    // `invalidate_fact`/
                    // `invalidate_fact_with_reason`, which write
                    // `expired_at`/`invalid_at` (a separate, later,
                    // system-time retirement act).
                    if let Some(valid_to) = pf.valid_to {
                        if let Err(e) = self.graph.bound_valid_to(fact_id, valid_to).await {
                            // A dropped caller-asserted
                            // `valid_to` (DB-level bind failure — distinct
                            // from the time-inversion reject above)
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
                    // Stamp ConsumerPinned on the subject entity
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
                        metrics::counter!("kremory.with_facts.consumer_pinned_tier_stamped_total")
                            .increment(1);
                    }
                    // Symmetric protection: when a pinned fact references an object
                    // entity (object_id = Some), stamp it
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
                requested = pre_pinned_facts.len(),
                episode_id = episode_id,
                "kremory.with_facts.pinned"
            );
        } else {
            // Avoid INFO-level noise for all-dedup caller sets;
            // bulk-import workloads can hit this thousands of times per batch.
            tracing::debug!(
                pinned = 0_u64,
                requested = pre_pinned_facts.len(),
                episode_id = episode_id,
                "kremory.with_facts.all_deduped_or_failed"
            );
        }
        (pinned_fact_ids, embedding_failures)
    }
}
