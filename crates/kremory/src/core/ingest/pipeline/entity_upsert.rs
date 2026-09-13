//! Turning each extracted entity into a graph row (TD-045 split out of
//! `ingest_with.rs`).
//!
//! One pass over the extraction's entities: decide whether this mention resolves to
//! an entity that already exists, upsert or merge accordingly, record the merge in
//! the union-find, and give the entity a presence edge on the episode.
//!
//! ## The state, finally named
//!
//! Six pieces of mutable state used to be six loose `let mut` bindings threaded
//! through a 448-line loop body inside a 1,100-line block. They are
//! [`Phase1EntityState`] now, which is not cosmetic: it is the list of everything
//! this pass can change, and it was previously something you could only establish by
//! reading every line.
//!
//! ## Errors abandon the whole phase, as before
//!
//! Inline, a failure was `break 'phases Err(e)` — abandon the labelled block, skip
//! the fact loop, land in the terminal match. Here it is an early `Err` return, and
//! the CALL SITE re-raises the same labelled break. Identical control flow; the only
//! difference is that the abandonment is now visible in a signature.

use std::collections::{HashMap, HashSet};

use crate::core::config::ResolutionStrategy;
use crate::core::embed_prefix::document_embed_text;
use crate::core::entity_types::EntityTypeRegistry;
use crate::core::error::Result;
use crate::core::extraction::normalize_label;
use crate::core::graph::{
    InsertEntityWithGroupParams, InsertEpisodicEdgeParams, SetEntityNerConfidenceParams,
    UpsertEntityWithGroupParams,
};
use crate::core::ingest::Engine;
use crate::core::intelligence::ExtractedEntity;
use crate::core::intelligence::{EntityResolver, ResolutionResult};
use crate::core::provider::{ChatProvider, EmbeddingProvider};
use crate::core::resolver::{normalize_name, CascadeResolver, UnionFind};
use crate::core::schema::Entity;

use super::deferred_emissions::DeferredEmission;
use super::ingest_with::BlockCandidatesParams;
use crate::core::ingest::helpers::extract_context_snippet;

/// Everything the entity pass MUTATES. One struct so the blast radius is a list you
/// can read rather than a loop you have to.
pub(super) struct Phase1EntityState<'a> {
    /// Merge equivalence classes — `union(new_name, existing_id)` on every resolve.
    pub(super) union_find: &'a mut UnionFind,
    /// Ids written this ingest, in encounter order.
    pub(super) upserted_entities: &'a mut Vec<String>,
    /// `(loser, keeper)` for every mention that resolved onto an existing entity.
    pub(super) merged_entities: &'a mut Vec<(String, String)>,
    /// Normalized name -> id, consumed by the fact loop to resolve endpoints.
    pub(super) name_to_id: &'a mut HashMap<String, String>,
    /// Ids that received a presence edge here, so the fact loop does not write a
    /// second one for the same episode.
    pub(super) entity_loop_ids: &'a mut HashSet<String>,
    /// Sink/metric emissions held until the outer transaction commits.
    pub(super) deferred: &'a mut Vec<DeferredEmission>,
}

/// Args-as-object for [`Engine::upsert_extracted_entities`].
pub(super) struct EntityUpsertParams<'a, L: ChatProvider> {
    pub(super) all_entities: &'a [ExtractedEntity],
    pub(super) existing_entities: &'a [Entity],
    /// Pass 1 + Pass 2 results, computed BEFORE the transaction opened. Under the
    /// `Batched` strategy this pass is a pure lookup into it.
    pub(super) batched_resolved: &'a HashMap<String, String>,
    pub(super) resolver: &'a CascadeResolver<L>,
    pub(super) registry: &'a EntityTypeRegistry,
    pub(super) group_id: Option<&'a str>,
    pub(super) episode_id: i64,
    /// The episode body — read ONLY to cut context snippets around a mention.
    pub(super) text: &'a str,
    pub(super) state: Phase1EntityState<'a>,
}

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Upsert every extracted entity, resolving each against what already exists.
    ///
    /// Returns `Err` on the first failure, which the caller turns back into the
    /// labelled break that abandons the phase block.
    pub(super) async fn upsert_extracted_entities(
        &self,
        p: EntityUpsertParams<'_, L>,
    ) -> Result<()> {
        let EntityUpsertParams {
            all_entities,
            existing_entities,
            batched_resolved,
            resolver,
            registry,
            group_id,
            episode_id,
            text,
            state:
                Phase1EntityState {
                    union_find,
                    upserted_entities,
                    merged_entities,
                    name_to_id,
                    entity_loop_ids,
                    deferred,
                },
        } = p;

        for extracted in all_entities {
            // The over-rejection guard
            // (`is_canonical_entity_type` + `allowed_entity_types` policy) is
            // DELETED under the L1 integer-ID design. Live diagnostic
            // showed it rejecting 14 "Entity" emissions per
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

            // Which existing id (if any) `extracted`
            // resolves to, computed differently per strategy.
            //
            // - `Batched` (default): Pass 1 (deterministic tiers) + Pass 2
            //   (batched LLM call) already ran BEFORE this transaction —
            //   see `batched_resolved` above. This arm is a pure lookup;
            //   it must NOT call `block_resolution_candidates` again
            //   (that already ran once per entity in Pass 1).
            // - `Pairwise`: the earlier behaviour, unchanged. It
            //   resolves `extracted` only against a bounded
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
                            existing_entities,
                            group_id,
                        })
                        .await;
                    metrics::counter!("kremory.resolution.candidates_considered_total")
                        .increment(candidates.len() as u64);
                    metrics::counter!("kremory.resolution.blocked_out_total")
                        .increment(existing_entities.len().saturating_sub(candidates.len()) as u64);

                    let mut found: Option<String> = None;
                    for existing in candidates.iter().copied() {
                        let result = match resolver.resolve(extracted, existing).await {
                            Ok(r) => r,
                            Err(e) => return Err(e),
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
                    // Open vocabulary: register novel type label on the fly.
                    // If registration itself fails (DB error), fall back to id=0
                    // and emit a swallow counter — the stub
                    // promotion is best-effort and must not abort the transaction.
                    let entity_type_id = match crate::core::entity_types::label_to_id_or_register(
                        crate::core::entity_types::LabelToIdOrRegisterParams {
                            conn: &self.graph.conn,
                            group_id: group_id.unwrap_or("default"),
                            registry,
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
                    // Claims a persisted row — replayed after the outer commit.
                    deferred.push(DeferredEmission::EntityPersisted {
                        source: "llm",
                        via: Some("stub_promotion"),
                    });
                    tracing::debug!(
                        target: "kremory.ingest.stub",
                        id = %existing_id,
                        label = %label,
                        entity_type_id,
                        "promoted stub entity to real entity"
                    );
                }

                // Episodic edge for merged entity (this episode now references it).
                // Fires on Ok only (insert error is soft `.ok()` precedent). ids
                // in tracing fields only, NOT metric labels.
                // Migration 006: thread the ingest namespace so the
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
                // Captured, not fired — replayed after the outer commit.
                deferred.push(DeferredEmission::EntityMention {
                    entity_id: existing_id.clone(),
                    name: extracted.name.clone(),
                    mention_ok,
                    via: "merged",
                });
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
                    Err(e) => return Err(e),
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
                    // Migration 006: thread the ingest namespace so
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
                    // Captured, not fired — replayed after the outer commit.
                    deferred.push(DeferredEmission::EntityMention {
                        entity_id: l4_existing_id.clone(),
                        name: extracted.name.clone(),
                        mention_ok,
                        via: "l4_merge",
                    });
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
                let entity_context = extract_context_snippet(text, &extracted.name, 100);
                let props_with_context = serde_json::json!({
                    "context": entity_context,
                    "name": extracted.name.clone(),
                });

                // Open vocabulary: register novel type label on the fly.
                // Unlike the stub-promotion site, the insert-new path is NOT
                // best-effort — registration failure propagates as a hard
                // ingest failure (matches the existing insert_entity_with_group
                // error path that breaks 'phases below).
                let entity_type_id = match crate::core::entity_types::label_to_id_or_register(
                    crate::core::entity_types::LabelToIdOrRegisterParams {
                        conn: &self.graph.conn,
                        group_id: group_id.unwrap_or("default"),
                        registry,
                        label: &label,
                    },
                )
                .await
                {
                    Ok(id) => id,
                    Err(e) => return Err(e),
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
                    // A UNIQUE violation here means the row ALREADY
                    // EXISTS — benign, and exactly what the sibling stub
                    // path at :1114 already concluded ("that is fine — a
                    // real row is present"). This path previously failed the
                    // ENTIRE ingest on it, because the raw libsql error was
                    // indistinguishable from a genuine DB failure. Observed:
                    // 1 of 8 LongMemEval sessions HTTP 500'd on
                    // `UNIQUE constraint failed: entities.id, entities.group_id`.
                    //
                    // The reachable cause is the extractor emitting the same
                    // NORMALISED name twice within one episode — this path
                    // believes the entity is new because it just decided so.
                    //
                    // NOT a blanket swallow: only the unique case
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
                             exists; continuing. Extractor likely emitted \
                             the same normalised name twice in one episode."
                        );
                    } else {
                        return Err(e);
                    }
                }
                // Claims a persisted row — replayed after the outer commit.
                deferred.push(DeferredEmission::EntityPersisted {
                    source: "llm",
                    via: Some("insert_new"),
                });

                // Persist GLiNER span confidence when present.
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
                // This is a WRITE into `entities.embedding` — document-prefix it.
                let embedding = match self
                    .embedder
                    .embed(&document_embed_text(
                        &extracted.name,
                        self.config.search.embed_task_prefix_enabled,
                    ))
                    .await
                {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
                // SCOPED write. This previously used the
                // namespace-unscoped `set_entity_embedding`, whose SQL matches
                // `WHERE id = ?` alone — so ingesting an entity named X into
                // namespace A silently overwrote X's embedding in EVERY other
                // namespace holding that name. 90 entity names exist in more
                // than one namespace on the shipped LoCoMo corpus, and
                // `entities.embedding` is a live retrieval signal, so the
                // corruption was silent and cross-tenant.
                //
                // `group_id.unwrap_or("default")` mirrors the sibling
                // `insert_entity_with_group` call above exactly, so the write
                // lands on the row this ingest just created and on no other.
                self.graph
                    .set_entity_embedding_in_group(crate::core::graph::SetEntityEmbeddingParams {
                        id: &entity_id,
                        group_id: group_id.unwrap_or("default"),
                        embedding: &embedding,
                    })
                    .await?;

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

                // Episodic edge for newly inserted entity.
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
                // Captured, not fired — replayed after the outer commit.
                deferred.push(DeferredEmission::EntityMention {
                    entity_id: entity_id.clone(),
                    name: extracted.name.clone(),
                    mention_ok,
                    via: "insert_new",
                });
                if mention_ok {
                    entity_loop_ids.insert(entity_id.clone());
                }

                upserted_entities.push(entity_id.clone());
                entity_id
            };

            name_to_id.insert(normalize_name(&extracted.name), entity_id);
        }
        Ok(())
    }
}
