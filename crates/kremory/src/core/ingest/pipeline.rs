//! Ingest pipeline — `ingest_with` and `ingest_deferred` impl blocks.
//!
//! Split from `ingest.rs` as part of TD-001 (E0-C).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use metrics::histogram;

use crate::core::config::ContentType;
use crate::core::contradiction::TwoPoolDetector;
use crate::core::entity_types::EntityTypeRegistry;
use crate::core::extraction::normalize_label;
use crate::core::extraction_window::ExtractionWindowSplitter;
use crate::core::intelligence::{
    EntityExtractor, EntityResolver, ExtractedEntity, ExtractedFact, ExtractionContext,
    ResolutionResult,
};
use crate::core::provider::{ChatProvider, EmbeddingProvider, TokenUsage};
use crate::core::resolver::{normalize_name, CascadeResolver, UnionFind};
use crate::core::search::SearchFilters;

use super::helpers::extract_context_snippet;
use super::{Engine, IngestionResult, SourceParams};

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Full pipeline with a caller-supplied extractor.
    /// Any type implementing `EntityExtractor` can be used (DefaultExtractor, NuExtractExtractor, etc.).
    // Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        name = "kremory.ingest",
        skip(self, extractor, text),
        fields(
            kremory.operation = "ingest",
        )
    )]
    pub async fn ingest_with<E: EntityExtractor>(
        &self,
        extractor: &E,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<&str>,
        content_type: Option<ContentType>,
        source_params: SourceParams,
    ) -> crate::core::error::Result<IngestionResult> {
        let ingest_start = Instant::now();
        let ref_time = reference_time.unwrap_or_else(Utc::now);
        let content_type = content_type.unwrap_or(ContentType::Text);
        let token_usage = TokenUsage::default();

        // 1. Store episode (namespace-scoped via group_id).
        //    source_id / source_uri / recorded_at from SourceParams are written to the
        //    Migration 007 columns so that recall_by_source_id can find this episode.
        let episode_id = self
            .graph
            .insert_episode_with_group(
                text,
                ref_time,
                Some("ingest"),
                None,
                group_id,
                None,
                None,
                source_params.source_id.as_deref(),
                source_params.source_uri.as_deref(),
                source_params.recorded_at,
            )
            .await?;

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
                if let Err(e) = self
                    .graph
                    .insert_entity(&pf.subject, 0, serde_json::json!({"stub": false}))
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
                if let Some(ref obj_id) = pf.object_id {
                    if let Err(e) = self
                        .graph
                        .insert_entity(obj_id, 0, serde_json::json!({"stub": false}))
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
                }

                match self
                    .graph
                    .try_insert_fact_with_group(
                        &pf.subject,
                        &pf.predicate,
                        pf.object_id.as_deref(),
                        pf.object_value.as_deref(),
                        pf.valid_from,
                        pf.confidence,
                        Some(episode_id),
                        group_id,
                        None,
                    )
                    .await
                {
                    Ok(Some(fact_id)) => {
                        pinned_count = pinned_count.saturating_add(1);
                        pinned_fact_ids.push(fact_id);
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
            EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?
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
        let mut all_entities: Vec<ExtractedEntity> = Vec::new();
        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &self.config.allowed_entity_types,
                allowed_edge_types: &self.config.allowed_edge_types,
                known_entities: &all_entities,
                excluded_entity_types: &self.config.excluded_entity_types,
                content_type: content_type.clone(),
                registry_specs: registry.specs(),
                existing_graph_entities: &existing_entities_for_prompt,
                arm_budget_ms: self.config.extraction_arm_budget_ms,
            };
            let result = extractor.extract(chunk, &ctx).await?;
            all_entities.extend(result.entities);
            all_facts.extend(result.facts);

            // OOV audit: catch domain terms the LLM missed (language-agnostic safety net)
            if let Some(ref auditor) = self.oov_auditor {
                let audit_adds = auditor.audit(chunk, &all_entities);
                all_entities.extend(audit_adds);
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

        let llm_for_resolver = self.llm.as_ref().ok_or_else(|| {
            crate::core::error::Error::LlmRequired {
                method: "ingest_with",
                hint: "wire an LLM via Memory::open(…).with_llm(…) to enable entity \
                       resolution; or use .with_facts(…) to pin triples without LLM",
            }
        })?;
        let resolver = CascadeResolver::new(
            Arc::clone(llm_for_resolver),
            self.config.minhash.clone(),
            self.config.entropy.clone(),
        );

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
                        .insert_entity_with_group(&norm_name, 0, stub_props, group_id)
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

                let mut resolved_to: Option<String> = None;

                for existing in &existing_entities {
                    let result = match resolver.resolve(extracted, existing).await {
                        Ok(r) => r,
                        Err(e) => break 'phases Err(e),
                    };
                    if result == ResolutionResult::Same {
                        resolved_to = Some(existing.id.clone());
                        break;
                    }
                }

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
                                &self.graph.conn,
                                group_id.unwrap_or("default"),
                                &registry,
                                &label,
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
                            .upsert_entity_with_group(
                                &existing_id,
                                entity_type_id,
                                promoted_props,
                                group_id,
                            )
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
                    self.graph
                        .insert_episodic_edge(episode_id, &existing_id, "mention")
                        .await
                        .ok();

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
                    let l4_outcome = match crate::core::disambiguation::disambiguate(
                        &extracted.name,
                        group_id,
                        &self.graph,
                        &*self.embedder,
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
                        self.graph
                            .insert_episodic_edge(episode_id, l4_existing_id, "mention")
                            .await
                            .ok();
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
                    let snippet = extract_context_snippet(text, &extracted.name, 100);
                    let props_with_context = serde_json::json!({
                        "context": snippet,
                        "name": extracted.name.clone(),
                    });

                    // TD-021 open vocabulary: register novel type label on the fly.
                    // Unlike the stub-promotion site, the insert-new path is NOT
                    // best-effort — registration failure propagates as a hard
                    // ingest failure (matches the existing insert_entity_with_group
                    // error path that breaks 'phases below).
                    let entity_type_id = match crate::core::entity_types::label_to_id_or_register(
                        &self.graph.conn,
                        group_id.unwrap_or("default"),
                        &registry,
                        &label,
                    )
                    .await
                    {
                        Ok(id) => id,
                        Err(e) => break 'phases Err(e),
                    };
                    if let Err(e) = self
                        .graph
                        .insert_entity_with_group(
                            &entity_id,
                            entity_type_id,
                            props_with_context,
                            group_id,
                        )
                        .await
                    {
                        break 'phases Err(e);
                    }
                    metrics::counter!(
                        "rql.ingest.entity_persisted_total",
                        "source" => "llm",
                        "via" => "insert_new",
                    )
                    .increment(1);

                    // Embed and store embedding
                    let embedding = match self.embedder.embed(&extracted.name).await {
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
                            &self.graph,
                            &entity_id,
                            alias_target_id,
                            alias_sim,
                            crate::core::disambiguation::AliasProvenance {
                                source_episode_id: Some(episode_id),
                                group_id,
                            },
                        )
                        .await
                        .ok();
                    }

                    // Bug A: episodic edge for newly inserted entity.
                    self.graph
                        .insert_episodic_edge(episode_id, &entity_id, "mention")
                        .await
                        .ok();

                    upserted_entities.push(entity_id.clone());
                    entity_id
                };

                name_to_id.insert(normalize_name(&extracted.name), entity_id);
            }

            // ── Phase 2: detect contradictions and store facts ───────────────────────
            let llm_for_detector = self.llm.as_ref().ok_or_else(|| {
                crate::core::error::Error::LlmRequired {
                    method: "ingest_with",
                    hint: "wire an LLM via Memory::open(…).with_llm(…) to enable \
                           contradiction detection; or use .with_facts(…) to pin \
                           triples without LLM",
                }
            })?;
            let detector = TwoPoolDetector::new(Arc::clone(llm_for_detector));

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
                    .fts_search_facts(&fact.predicate, 10, &SearchFilters::new())
                    .await
                {
                    Ok(hits) => hits,
                    Err(e) => break 'phases Err(e),
                };
                let pool_b: Vec<crate::core::schema::Fact> =
                    pool_b_hits.into_iter().map(|h| h.item).collect();

                // Run contradiction detection
                let contradiction_result =
                    match detector.detect(fact, &pool_a, &pool_b, &ref_time).await {
                        Ok(r) => r,
                        Err(e) => break 'phases Err(e),
                    };

                // Invalidate contradicted facts
                for fact_id in &contradiction_result.contradictions {
                    if let Err(e) = self
                        .graph
                        .invalidate_fact_with_reason(*fact_id, Utc::now(), ref_time)
                        .await
                    {
                        break 'phases Err(e);
                    }
                    invalidated_fact_ids.push(*fact_id);
                }

                // Insert the new fact — skip gracefully if FK constraint fails
                // (e.g., fact references an entity not in the extraction results).
                //
                // ADR-035 §5 Option A: use try_insert_fact so caller-pinned facts
                // (via mem.remember(...).with_facts(...)) silently dedup at LLM
                // Phase 2 — the caller wins by virtue of being there first.
                match self
                    .graph
                    .try_insert_fact(
                        &subject_id,
                        &fact.predicate,
                        object_id.as_deref(),
                        object_value,
                        ref_time,
                        fact.confidence,
                        Some(episode_id),
                        None,
                    )
                    .await
                {
                    Ok(Some(fact_id)) => {
                        // Embed the fact triple as a single string (subject predicate object)
                        // and store it so vector_search_facts can find it semantically.
                        let fact_text =
                            format!("{} {} {}", fact.subject, fact.predicate, fact.object);
                        if let Ok(embedding) = self.embedder.embed(&fact_text).await {
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
                        tracing::warn!(
                            subject = %fact.subject,
                            predicate = %fact.predicate,
                            object = %fact.object,
                            error = %e,
                            "kremory.ingest.phase2_fact_insert_failed"
                        );
                    }
                }

                // Bug A: subject-side episodic edge is now written in the entity loop
                // (role="mention"), guaranteeing coverage for all extracted entities
                // regardless of whether they appear in facts.
                // The object-side episodic edge is retained here to handle fact objects
                // that are stubs or entities not present in the current extraction batch.
                if let Some(ref obj_id) = object_id {
                    if name_to_id.contains_key(&normalize_name(&fact.object)) {
                        self.graph
                            .insert_episodic_edge(episode_id, obj_id, "object")
                            .await
                            .ok();
                    }
                }
            }

            Ok(())
        }; // end 'phases block

        // Commit or rollback the outer transaction based on phase result.
        match phase_result {
            Ok(()) => outer_guard.commit().await?,
            Err(e) => {
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
    // Substrate primitive; consumer-facing surface is kremory::Memory facade per ADR-027.
    #[allow(clippy::too_many_arguments)]
    pub async fn ingest_deferred(
        &self,
        text: &str,
        reference_time: Option<DateTime<Utc>>,
        group_id: Option<&str>,
        content_type: Option<ContentType>,
        episode_id: i64,
        ner_entity_names: &[String],
    ) -> crate::core::error::Result<usize> {
        // TD-019 Gap 2: phase2_ms — deferred relationship extraction latency,
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

        let mut all_facts: Vec<ExtractedFact> = Vec::new();
        for chunk in &chunks {
            let ctx = ExtractionContext {
                allowed_entity_types: &self.config.allowed_entity_types,
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
        let llm_for_batch_detector = self.llm.as_ref().ok_or_else(|| {
            crate::core::error::Error::LlmRequired {
                method: "ingest_batch",
                hint: "wire an LLM via Memory::open(…).with_llm(…) to enable \
                       contradiction detection; or use .with_facts(…) to pin \
                       triples without LLM",
            }
        })?;
        let detector = TwoPoolDetector::new(Arc::clone(llm_for_batch_detector));
        let mut inserted_count: usize = 0;

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

            let pool_a = self
                .graph
                .get_facts_by_subject_predicate(&subject_id, &fact.predicate)
                .await?;

            let pool_b_hits = self
                .graph
                .fts_search_facts(&fact.predicate, 10, &SearchFilters::new())
                .await?;
            let pool_b: Vec<crate::core::schema::Fact> =
                pool_b_hits.into_iter().map(|h| h.item).collect();

            let contradiction_result = detector.detect(fact, &pool_a, &pool_b, &ref_time).await?;

            for fact_id in &contradiction_result.contradictions {
                self.graph
                    .invalidate_fact_with_reason(*fact_id, Utc::now(), ref_time)
                    .await?;
            }

            // ADR-035 §5 Option A: use try_insert_fact for caller-pin dedup parity
            // with the inline path. Deferred Phase 2 writes also silently skip
            // triples already pre-pinned by the caller via with_facts.
            match self
                .graph
                .try_insert_fact(
                    &subject_id,
                    &fact.predicate,
                    object_id.as_deref(),
                    object_value,
                    ref_time,
                    fact.confidence,
                    Some(episode_id),
                    None,
                )
                .await
            {
                Ok(Some(fact_id)) => {
                    let fact_text = format!("{} {} {}", fact.subject, fact.predicate, fact.object);
                    if let Ok(embedding) = self.embedder.embed(&fact_text).await {
                        self.graph
                            .set_fact_embedding(fact_id, &embedding)
                            .await
                            .ok();
                    }
                    if name_to_id.contains_key(&normalize_name(&fact.subject)) {
                        self.graph
                            .insert_episodic_edge(episode_id, &subject_id, "subject")
                            .await
                            .ok();
                    }
                    if let Some(ref obj_id) = object_id {
                        if name_to_id.contains_key(&normalize_name(&fact.object)) {
                            self.graph
                                .insert_episodic_edge(episode_id, obj_id, "object")
                                .await
                                .ok();
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
                    tracing::warn!(
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        object = %fact.object,
                        error = %e,
                        "kremory.ingest.deferred_phase2_fact_insert_failed"
                    );
                }
            }
        }

        histogram!("rql.ingest.deferred_fact_count").record(inserted_count as f64);
        histogram!("rql.ingest.phase2_ms").record(phase2_start.elapsed().as_secs_f64() * 1000.0);
        tracing::info!(inserted_count, "kremory.ingest.deferred_facts inserted");
        Ok(inserted_count)
    }
}
