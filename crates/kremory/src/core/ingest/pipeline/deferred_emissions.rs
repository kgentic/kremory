//! Emissions held back until `ingest_with`'s outer transaction durably commits
//! (TD-045 split out of `ingest_with.rs`).
//!
//! A sink callback and a metric increment are both IRREVERSIBLE: `ROLLBACK` undoes
//! the row, not the notification. Everything here exists so a write that never
//! landed cannot tell anyone it did — the buffer, its replay after `commit()`, and
//! the label-stable counter helper the three per-mention fire-sites share.

// ─── Sink fire-site metrics helper ──────────────────────────────────────────

/// Triple-emit metrics companion for the per-entity / per-mention-edge sink
/// fire-sites in `ingest_with` (canonical pattern from 737e152
/// verify_stage Fire-sites 2 + 3). Kept as a free fn so the three `mention`
/// call sites (merged / L4-merge / new-entity) emit identical metric labels
/// without copy-paste drift. The sink callback itself fires inline at each call
/// site (it needs the per-site ids); only the label-stable counters live here.
///
/// No entity_id / episode_id labels — `predicate_kind` is a
/// bounded enum, `mention_persisted` a bool string.
pub(super) fn fire_entity_edge_metrics(mention_ok: bool) {
    metrics::counter!("kremory.sink.entity_extracted_total", "arm" => "inline").increment(1);
    if mention_ok {
        metrics::counter!(
            "kremory.sink.edge_added_total",
            "predicate_kind" => "episodic"
        )
        .increment(1);
    }
}

// ─── Emissions deferred until the outer transaction durably commits ────────

/// One emission captured DURING `ingest_with`'s outer transaction and replayed only
/// once that transaction has committed.
///
/// # Why this exists
///
/// A sink callback and a metric increment are both **irreversible**: `ROLLBACK` undoes
/// the row, not the notification. Every one of these sites previously fired inline, at
/// the moment the row was written, inside a transaction spanning ~1,086 lines. So a
/// rolled-back ingest told its consumer about entities and edges that do not exist —
/// and never corrected itself. For the napi / MCP event surfaces, which build state
/// from the sink, that is a **correctness** defect, not a metrics-accuracy one.
///
/// This is the same collect-during-txn / flush-post-commit shape already used by
/// `forget.rs` and `supersession::window_closeout`, and the placement rule is
/// the one stated on [`BeginGuard::opened`]: an emission summarising a durable write
/// must fire only once that write is actually committed.
///
/// # What is deliberately NOT deferred
///
/// Only emissions that **claim a durable row** belong here. Counters describing work
/// that genuinely happened regardless of the outcome stay inline, because deferring
/// them would UNDER-count real work on the rollback path — the opposite error:
///
/// | counter | why it stays inline |
/// |---|---|
/// | `kremory.resolution.{candidates_considered,blocked_out}_total` | comparisons that were performed |
/// | `kremory.ingest.contradiction_{outcome,pool}_total` | the detector's verdict + its real LLM spend |
/// | `rql.ingest.within_episode_{duplicate_triple,multivalue}` | filter decisions that write nothing |
/// | `rql.entity_types.registration_swallowed_total` | a failure that was observed |
/// | `kremory.ingest.entity_insert_duplicate_tolerated_total` | a tolerated collision that was observed |
/// | `kremory.ingest.phase2_fact_insert_failed_total` | a fact that was NOT written |
///
/// The durable-write claim for a contradiction is carried by
/// `kremory.sink.contradiction_total`, which IS deferred, so the split is clean:
/// `contradiction_outcome_total` measures the detection pass, `sink.contradiction_total`
/// measures the invalidation that survived.
pub(super) enum DeferredEmission {
    /// The three entity fire-sites (merged / L4-merge / insert-new) are byte-identical
    /// in shape, so they share one variant — the same anti-drift rationale as
    /// [`fire_entity_edge_metrics`], which this variant's replay calls.
    EntityMention {
        entity_id: String,
        name: String,
        /// Whether the `role="mention"` episodic edge actually inserted. Gates both
        /// the `on_edge_added` callback and the episodic-edge counter.
        mention_ok: bool,
        /// Which arm produced this entity — a tracing field only, never a metric label.
        via: &'static str,
    },
    /// `rql.ingest.entity_persisted_total` — one persisted entity row.
    EntityPersisted {
        source: &'static str,
        via: Option<&'static str>,
    },
    /// `kremory.ingest.stub_inserted` + its `entity_persisted_total{source=stub}` pair.
    StubInserted,
    /// `on_contradiction` + `kremory.sink.contradiction_total`.
    ///
    /// `event` is `Option` because the two halves have DIFFERENT firing conditions and
    /// the inline code they replace did too: the counter fires for every invalidation,
    /// while the callback fires only when a sink is wired AND the prior fact was
    /// locatable in `pool_a`/`pool_b` (best-effort — the payload is never fabricated,
    /// per the parse-loudly discipline). Collapsing them would silently change one.
    ///
    /// Boxed because the payload is by far the largest variant and would otherwise set
    /// the size of every element in the buffer (clippy `large_enum_variant`).
    Contradiction {
        event: Option<Box<crate::core::sink::ContradictionDetected>>,
        prior_fact_id: i64,
    },
    /// `on_edge_added("object")` + `kremory.sink.edge_added_total{object}`.
    ObjectEdge { to_entity_id: String },
}

/// Replay every deferred emission, in capture order. Called ONLY from the `Ok` arm of
/// the outer transaction's commit — see the call site for why the placement matters
/// for re-entrant (nested) callers.
pub(super) fn flush_deferred_emissions(
    deferred: Vec<DeferredEmission>,
    sink: Option<&dyn crate::core::sink::IngestEventSink>,
    episode_id: i64,
) {
    // Computed once rather than per-edge: the previous inline sites each rebuilt this
    // with `episode_id.to_string()` inside the loop body.
    let episode_key = episode_id.to_string();

    for emission in deferred {
        match emission {
            DeferredEmission::EntityMention {
                entity_id,
                name,
                mention_ok,
                via,
            } => {
                if let Some(s) = sink {
                    s.on_entity_extracted(&entity_id, &name);
                    if mention_ok {
                        s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                            from_entity_id: &episode_key,
                            to_entity_id: &entity_id,
                            predicate: "mention",
                        });
                    }
                }
                fire_entity_edge_metrics(mention_ok);
                tracing::debug!(
                    entity_id = %entity_id,
                    name = %name,
                    episode_id,
                    via,
                    "kremory.sink.entity_extracted"
                );
            }
            DeferredEmission::EntityPersisted { source, via } => match via {
                Some(v) => {
                    metrics::counter!(
                        "rql.ingest.entity_persisted_total",
                        "source" => source,
                        "via" => v,
                    )
                    .increment(1);
                }
                None => {
                    metrics::counter!(
                        "rql.ingest.entity_persisted_total",
                        "source" => source,
                    )
                    .increment(1);
                }
            },
            DeferredEmission::StubInserted => {
                metrics::counter!("kremory.ingest.stub_inserted").increment(1);
                metrics::counter!(
                    "rql.ingest.entity_persisted_total",
                    "source" => "stub",
                )
                .increment(1);
            }
            DeferredEmission::Contradiction {
                event,
                prior_fact_id,
            } => {
                // Both conditions must still hold, exactly as inline: a sink is wired
                // AND the prior fact was locatable when the payload was captured.
                if let (Some(s), Some(ev)) = (sink, event) {
                    s.on_contradiction(*ev);
                }
                metrics::counter!(
                    "kremory.sink.contradiction_total",
                    "resolution" => "Superseded"
                )
                .increment(1);
                tracing::debug!(prior_fact_id, episode_id, "kremory.sink.contradiction");
            }
            DeferredEmission::ObjectEdge { to_entity_id } => {
                if let Some(s) = sink {
                    s.on_edge_added(crate::core::sink::OnEdgeAddedParams {
                        from_entity_id: &episode_key,
                        to_entity_id: &to_entity_id,
                        predicate: "object",
                    });
                }
                metrics::counter!(
                    "kremory.sink.edge_added_total",
                    "predicate_kind" => "object"
                )
                .increment(1);
                tracing::debug!(
                    entity_id = %to_entity_id,
                    episode_id,
                    predicate = "object",
                    "kremory.sink.edge_added"
                );
            }
        }
    }
}
