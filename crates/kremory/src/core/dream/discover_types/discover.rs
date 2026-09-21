use std::time::Instant;

use metrics::{counter, histogram};

use crate::core::{
    dream::{
        anti_redundancy::{self, GateOutcome},
        proposed_type::{validate_proposed_name, DiscoveryProposalBatch},
    },
    entity_types::EntityTypeRegistry,
    error::Result,
    extraction::structured::StructuredCallBuilder,
    provider::{ChatProvider, DynEmbeddingProvider},
};

use super::{
    adjudicate::{
        adjudicate_type_novelty, record_gate_decision, record_type_novelty_decision,
        type_novelty_is_redundant, AdjudicateTypeNoveltyParams, RecordGateDecisionParams,
    },
    cluster::{build_discovery_messages, load_catch_all_entities, reason_to_string, top_k_clusters},
    helpers::{accept_proposal, AcceptProposalParams},
    types::{DiscoveryResult, TypeProposal},
};

// ─── Main primitive ───────────────────────────────────────────────────────────

/// Discover new entity types from catch-all entities in `group_id`.
///
/// Called by `mem.dream()` when `include_type_discovery = true` (D6).
/// Can also be called standalone via the escape-hatch `mem.discover_types()`.
///
/// Bundled non-generic parameters for [`discover_types`] — args-as-object
/// (rust-conventions §too_many_arguments). The generic `llm: &L` stays a
/// lead positional param (brief rule 4); the remaining args bundle here.
///
/// Fields:
/// - `conn` — open SQLite connection
/// - `group_id` — the namespace (equals `namespace_to_group_id(&ns)`)
/// - `embedder` — `None` = degraded mode (anti-redundancy gate skipped)
/// - `max_proposals` — caller can override; defaults to [`MAX_PROPOSALS`]
pub(crate) struct DiscoverTypesParams<'a> {
    pub(crate) conn: &'a libsql::Connection,
    pub(crate) group_id: &'a str,
    pub(crate) embedder: Option<&'a dyn DynEmbeddingProvider>,
    pub(crate) max_proposals: usize,
    /// Concrete model id for capability detection. Threaded from the
    /// facade dream path (`dream_model_id_or_main`). Empty (`""`) → `PromptOnly`
    /// degrade — the correct behaviour when the model is unknown. Before this
    /// field existed the model id was hardcoded to `String::new()`, silently
    /// degrading every call.
    pub(crate) model_id: &'a str,
    /// Site #2 — `DreamOpts::include_type_novelty_
    /// llm_verify`, threaded from `facade/dream.rs`. `false` (default): a
    /// `GateOutcome::NeedsLlmVerify` classification falls back to the pre-Site-#2
    /// behaviour (≥0.85 → reject, else accept) — the DEFAULT build's outcome is
    /// UNCHANGED. `true`: `NeedsLlmVerify` proposals are adjudicated via the
    /// shared `write_gate` (spec §2.2).
    pub(crate) llm_verify_band: bool,
    /// `DreamOpts::include_evidence_retype_by_similarity`, threaded
    /// from `facade/dream.rs`. `false` (default): the in-place evidence-retype
    /// step (D4) skips the cosine-only bare-name-vs-type-description comparison
    /// entirely (unspiked degeneracy risk, see module docs) and leaves evidence
    /// entities as catch-all for Pass 2 `reclassify` to pick up safely. `true`:
    /// opt-in to the pre-existing cosine-alone retype behaviour.
    pub(crate) evidence_retype_by_similarity: bool,
}

/// Discover new entity types from catch-all entities in `group_id`.
///
/// `llm` — the chat provider (generic lead positional param).
pub(crate) async fn discover_types<L: ChatProvider>(
    llm: &L,
    params: DiscoverTypesParams<'_>,
) -> Result<DiscoveryResult> {
    let DiscoverTypesParams {
        conn,
        group_id,
        embedder,
        max_proposals,
        model_id,
        llm_verify_band,
        evidence_retype_by_similarity,
    } = params;
    let mut result = DiscoveryResult::default();

    // ── Step 1: Load catch-all entities ──────────────────────────────────────

    let catch_alls = load_catch_all_entities(conn, group_id).await?;
    if catch_alls.is_empty() {
        tracing::debug!(
            target: "kremory::dream::discover_types",
            group_id = %group_id,
            "discover_types: no catch-all entities found — skipping"
        );
        return Ok(result);
    }

    tracing::debug!(
        target: "kremory::dream::discover_types",
        group_id = %group_id,
        catch_all_count = catch_alls.len(),
        "discover_types: found catch-all entities"
    );

    // ── Step 2: Top-K clusters by frequency (name-grouping) ──────────────────

    let clusters = top_k_clusters(&catch_alls, max_proposals);
    if clusters.is_empty() {
        return Ok(result);
    }

    // ── Step 3: Load existing registry for anti-redundancy gate ──────────────

    let registry = EntityTypeRegistry::load_for_group(conn, group_id).await?;
    // The consumer-supplied model id now reaches
    // this pass via `DiscoverTypesParams.model_id`, threaded from the facade
    // dream path (`with_dream_model_id` / `with_model_id` → `dream_model_id_or_main`).
    // Empty → `PromptOnly` degrade (correct when the model is unknown); a
    // populated id drives provider-native / FormatSchema capability detection.
    let model_str = model_id.to_string();

    // ── Step 4: Anti-redundancy embeddings (degraded mode check) ─────────────

    let existing_embeddings: Vec<_> = if let Some(emb) = embedder {
        let mut embs = Vec::new();
        for spec in registry.specs() {
            // Skip catch-all (id=0) — it's not a valid comparison target
            if spec.id == 0 {
                continue;
            }
            // Site #1: only the DESCRIPTION is embedded now. The name signal
            // is a deterministic normalized exact-match in `check_proposal`, not a
            // (degenerate) bare-name cosine, so no name embedding is computed.
            let desc_emb = emb.embed_dyn(&spec.description).await?;
            embs.push((spec.clone(), desc_emb));
        }
        embs
    } else {
        result
            .warnings
            .push("no embedder configured — anti-redundancy gate skipped".to_string());
        anti_redundancy::emit_gate_skipped();
        Vec::new()
    };

    // ── Step 5: Build LLM prompt ──────────────────────────────────────────────

    let messages = build_discovery_messages(&clusters, max_proposals);
    let schema = crate::core::dream::proposed_type::discovery_proposal_schema()
        .map_err(|e| crate::core::error::Error::Other(anyhow::anyhow!(e)))?;

    // ── Step 6: LLM call with observability ──────────────────────────────────

    let call_start = Instant::now();
    let raw_value = StructuredCallBuilder::new(llm, &schema, "DiscoveryProposalBatch")
        .model(&model_str)
        .messages(messages)
        .call()
        .await;
    let elapsed_ms = call_start.elapsed().as_millis() as f64;

    histogram!(
        "kremory.dream.proposal_call_duration_ms",
        "model" => model_str.clone()
    )
    .record(elapsed_ms);

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            counter!(
                "kremory.dream.proposal_call_outcome_total",
                "outcome" => "llm_err",
                "model" => model_str.clone(),
                "namespace" => group_id.to_string()
            )
            .increment(1);
            tracing::warn!(
                target: "kremory::dream::discover_types",
                error = %e,
                group_id = %group_id,
                "discover_types: LLM call failed — returning empty result"
            );
            return Ok(result);
        }
    };

    // Deserialise — use separate counters for direct vs repair path (ADR §6 / rule llm-output-parse-loudly)
    let batch: DiscoveryProposalBatch = match serde_json::from_value(raw_value.clone()) {
        Ok(b) => {
            counter!(
                "kremory.dream.proposal_call_outcome_total",
                "outcome" => "ok",
                "model" => model_str.clone(),
                "namespace" => group_id.to_string()
            )
            .increment(1);
            b
        }
        Err(_) => {
            // Try repair: wrap in {"proposals": ...} if the LLM returned a bare array
            let repaired = if raw_value.is_array() {
                serde_json::json!({ "proposals": raw_value })
            } else {
                raw_value.clone()
            };
            match serde_json::from_value::<DiscoveryProposalBatch>(repaired) {
                Ok(b) => {
                    counter!(
                        "kremory.dream.proposal_call_outcome_total",
                        "outcome" => "parse_repair",
                        "model" => model_str.clone(),
                        "namespace" => group_id.to_string()
                    )
                    .increment(1);
                    b
                }
                Err(e2) => {
                    counter!(
                        "kremory.dream.proposal_call_outcome_total",
                        "outcome" => "parse_fail",
                        "model" => model_str.clone(),
                        "namespace" => group_id.to_string()
                    )
                    .increment(1);
                    tracing::warn!(
                        target: "kremory::dream::discover_types",
                        error = %e2,
                        group_id = %group_id,
                        "discover_types: failed to parse LLM batch — returning empty result"
                    );
                    return Ok(result);
                }
            }
        }
    };

    // ── Step 7: Per-proposal: shape validate → anti-redundancy → persist ──────

    // One run_id per `discover_types` invocation, shared by every
    // `identity_verdict_audit` row this call writes — mirrors
    // `type_registry_collapse.rs`'s `run_id` (generated once per call, not
    // once per row) so audit rows from the same Pass-0 pass are correlatable.
    let run_id = uuid::Uuid::new_v4().to_string();

    for raw_proposal in batch.proposals {
        let proposal = TypeProposal {
            name: raw_proposal.name.clone(),
            description: raw_proposal.description.clone(),
            justification: raw_proposal.justification.clone(),
        };

        // Count proposed
        counter!(
            "kremory.dream.types_proposed_total",
            "model" => model_str.clone(),
            "namespace" => group_id.to_string()
        )
        .increment(1);
        result.types_proposed.push(TypeProposal {
            name: proposal.name.clone(),
            description: proposal.description.clone(),
            justification: proposal.justification.clone(),
        });

        // Shape validate
        if let Err(reason) = validate_proposed_name(&proposal.name, group_id) {
            let reason_str = reason_to_string(reason);
            result.types_rejected.push((proposal, reason_str));
            continue;
        }

        // Anti-redundancy gate
        if let Some(emb) = embedder {
            let desc_emb = match emb.embed_dyn(&proposal.description).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::discover_types",
                        error = %e,
                        name = %proposal.name,
                        "discover_types: failed to embed proposal description — skipping gate"
                    );
                    result.warnings.push(format!(
                        "embed failed for '{}' — anti-redundancy gate skipped for this proposal",
                        proposal.name
                    ));
                    anti_redundancy::emit_gate_skipped();
                    // Fall through to acceptance (can't gate without embedding)
                    accept_proposal(AcceptProposalParams {
                        conn,
                        group_id,
                        model_str: &model_str,
                        proposal: &proposal,
                        catch_alls: &catch_alls,
                        desc_emb_and_embedder: None,
                        evidence_retype_by_similarity,
                        result: &mut result,
                    })
                    .await?;
                    continue;
                }
            };
            // Site #1: no name embedding — `check_proposal` uses a
            // deterministic normalized exact-match on `proposal.name` instead of a
            // (degenerate) bare-name cosine.
            match anti_redundancy::check_proposal(anti_redundancy::CheckProposalParams {
                proposal_name: &proposal.name,
                proposal_desc_emb: &desc_emb,
                existing_type_embeddings: &existing_embeddings,
                namespace: group_id,
                model: &model_str,
            }) {
                GateOutcome::Redundant {
                    existing_name,
                    desc_cosine,
                } => {
                    let reason = format!("redundant_with:{existing_name}");
                    // Leave evidence for this rejection regardless of
                    // `llm_verify_band` — `desc_cosine` is `None` when the
                    // exact-name pre-filter fired (no cosine was ever computed
                    // for this decision) and `Some` when the desc-cosine
                    // threshold decided it; both are honest, not dropped.
                    record_gate_decision(RecordGateDecisionParams {
                        conn,
                        group_id,
                        run_id: &run_id,
                        proposal_name: &proposal.name,
                        existing_name: Some(existing_name.as_str()),
                        desc_cosine,
                        verdict: None,
                        decision: "reject",
                        outcome_kind: if desc_cosine.is_some() {
                            "redundant_desc_cosine"
                        } else {
                            "redundant_exact_name"
                        },
                        model: &model_str,
                    })
                    .await?;
                    result.types_rejected.push((proposal, reason));
                    continue;
                }
                GateOutcome::Pass {
                    existing_name,
                    desc_cosine,
                } => {
                    accept_proposal(AcceptProposalParams {
                        conn,
                        group_id,
                        model_str: &model_str,
                        proposal: &proposal,
                        catch_alls: &catch_alls,
                        desc_emb_and_embedder: Some((&desc_emb, emb)),
                        evidence_retype_by_similarity,
                        result: &mut result,
                    })
                    .await?;
                    // This is the branch that previously left NO trace
                    // at all — an accept via `Pass` never called
                    // `record_type_novelty_decision` (that fn is only reached
                    // from the LLM-verify sub-path below) and never wrote to
                    // `identity_verdict_audit`. `existing_name`/`desc_cosine`
                    // are `Some` when a real (if below-lower-band) comparison
                    // happened, `None` only when the registry had no existing
                    // type to compare against at all.
                    record_gate_decision(RecordGateDecisionParams {
                        conn,
                        group_id,
                        run_id: &run_id,
                        proposal_name: &proposal.name,
                        existing_name: existing_name.as_deref(),
                        desc_cosine,
                        verdict: None,
                        decision: "accept",
                        outcome_kind: if existing_name.is_some() {
                            "pass_below_lower_band"
                        } else {
                            "pass_no_existing_types"
                        },
                        model: &model_str,
                    })
                    .await?;
                }
                GateOutcome::NeedsLlmVerify {
                    existing_name,
                    desc_cosine,
                } => {
                    if !llm_verify_band {
                        // Default behaviour (spike-gated off, spec §8): preserve
                        // the PRE-Site-#2 outcome exactly — ≥0.85 rejects, the
                        // [0.70, 0.85) ambiguous band accepts (there was no
                        // "candidate" concept before Site #2 existed).
                        if desc_cosine >= anti_redundancy::DESC_COSINE_THRESHOLD {
                            let reason = format!("redundant_with:{existing_name}");
                            record_gate_decision(RecordGateDecisionParams {
                                conn,
                                group_id,
                                run_id: &run_id,
                                proposal_name: &proposal.name,
                                existing_name: Some(existing_name.as_str()),
                                desc_cosine: Some(desc_cosine),
                                verdict: None,
                                decision: "reject",
                                outcome_kind: "ambiguous_band_reject_flag_off",
                                model: &model_str,
                            })
                            .await?;
                            result.types_rejected.push((proposal, reason));
                        } else {
                            accept_proposal(AcceptProposalParams {
                                conn,
                                group_id,
                                model_str: &model_str,
                                proposal: &proposal,
                                catch_alls: &catch_alls,
                                desc_emb_and_embedder: Some((&desc_emb, emb)),
                                evidence_retype_by_similarity,
                                result: &mut result,
                            })
                            .await?;
                            record_gate_decision(RecordGateDecisionParams {
                                conn,
                                group_id,
                                run_id: &run_id,
                                proposal_name: &proposal.name,
                                existing_name: Some(existing_name.as_str()),
                                desc_cosine: Some(desc_cosine),
                                verdict: None,
                                decision: "accept",
                                outcome_kind: "ambiguous_band_accept_flag_off",
                                model: &model_str,
                            })
                            .await?;
                        }
                        continue;
                    }

                    // Flag ON: trust the LLM as terminal arbiter — a
                    // Site-#2-LOCAL decision that intentionally does NOT call the
                    // shared `write_gate`. Type synonyms ("Firm"/"Company") are
                    // lexically dissimilar by nature, so write_gate Row 6's
                    // deterministic-corroboration requirement (an ENTITY-
                    // homonymy guard) over-generalized to schema types and
                    // downgraded correct confident `true` verdicts to accept →
                    // duplicate types. See `type_novelty_is_redundant`.
                    let existing_desc = existing_embeddings
                        .iter()
                        .find(|(spec, _)| spec.name == existing_name)
                        .map(|(spec, _)| spec.description.clone())
                        .unwrap_or_default();
                    let verdict = adjudicate_type_novelty(AdjudicateTypeNoveltyParams {
                        llm,
                        model_id: &model_str,
                        proposal_name: &proposal.name,
                        proposal_desc: &proposal.description,
                        existing_name: &existing_name,
                        existing_desc: &existing_desc,
                        group_id,
                    })
                    .await;

                    let redundant = type_novelty_is_redundant(&verdict);
                    record_type_novelty_decision(redundant, &verdict);

                    if redundant {
                        // Confident `is_same_entity == true` → the proposal IS the
                        // same concept as the existing type → redundant → reject.
                        let reason = format!("redundant_with:{existing_name}");
                        record_gate_decision(RecordGateDecisionParams {
                            conn,
                            group_id,
                            run_id: &run_id,
                            proposal_name: &proposal.name,
                            existing_name: Some(existing_name.as_str()),
                            desc_cosine: Some(desc_cosine),
                            verdict: verdict.as_ref(),
                            decision: "reject",
                            outcome_kind: "llm_verify_reject",
                            model: &model_str,
                        })
                        .await?;
                        result.types_rejected.push((proposal, reason));
                    } else {
                        // Novel (`is_same_entity == false`), OR low-confidence / no
                        // verdict → conservative accept. A new type must never be
                        // blocked on weak or absent evidence (types have no
                        // potential-alias edge concept, mirroring Site #3's brief).
                        // The `is_same_entity == false` path is the common EDC
                        // false-reject-prevention case; log only the low-confidence/
                        // absent case (the decision counter records both).
                        let llm_said_distinct = matches!(&verdict, Some(v) if !v.is_same_entity);
                        if !llm_said_distinct {
                            tracing::info!(
                                target: "kremory::dream::discover_types",
                                proposal_name = %proposal.name,
                                existing_name = %existing_name,
                                desc_cosine = %desc_cosine,
                                "discover_types: Site #2 low-confidence/absent type-novelty \
                                 verdict — accepting proposal (a new type must not be blocked \
                                 on weak evidence)"
                            );
                        }
                        accept_proposal(AcceptProposalParams {
                            conn,
                            group_id,
                            model_str: &model_str,
                            proposal: &proposal,
                            catch_alls: &catch_alls,
                            desc_emb_and_embedder: Some((&desc_emb, emb)),
                            evidence_retype_by_similarity,
                            result: &mut result,
                        })
                        .await?;
                        record_gate_decision(RecordGateDecisionParams {
                            conn,
                            group_id,
                            run_id: &run_id,
                            proposal_name: &proposal.name,
                            existing_name: Some(existing_name.as_str()),
                            desc_cosine: Some(desc_cosine),
                            verdict: verdict.as_ref(),
                            decision: "accept",
                            outcome_kind: "llm_verify_accept",
                            model: &model_str,
                        })
                        .await?;
                    }
                }
            }
        } else {
            // Degraded mode: gate skipped, accept directly
            accept_proposal(AcceptProposalParams {
                conn,
                group_id,
                model_str: &model_str,
                proposal: &proposal,
                catch_alls: &catch_alls,
                desc_emb_and_embedder: None,
                evidence_retype_by_similarity,
                result: &mut result,
            })
            .await?;
        }
    }

    // ── Fail-loud on silent zero-discovery ───────────────────────────────────
    //
    // We only reach here with a NON-EMPTY catch-all bucket (empty buckets and
    // empty clusters early-returned above), so `types_accepted.is_empty()` here
    // means discovery *engaged but produced nothing usable* — model too weak
    // (e.g. interactive-tier gemma4-e2b emits placeholder names that the shape
    // validator rejects) or prompt drift. Previously this was SILENT: the
    // DreamSummary looked identical to "nothing to discover". Pass-0 is
    // non-fatal, so we WARN (never abort): a counter + a tracing::warn! + a
    // consumer-visible `DiscoveryResult.warnings` entry.
    if result.types_accepted.is_empty() {
        counter!(
            "kremory.dream.discovery_yielded_zero_total",
            "model" => model_str.clone(),
            "namespace" => group_id.to_string()
        )
        .increment(1);
        let msg = format!(
            "type discovery yielded ZERO accepted types from {} catch-all \
             entit{} ({} proposed, {} rejected) using model '{}' — the model may \
             be under-powered for discovery (the interactive-tier gemma4-e2b emits \
             placeholder names the validator rejects); wire the deferred quality \
             model (e.g. gemma4:e4b) for the dream phase",
            catch_alls.len(),
            if catch_alls.len() == 1 { "y" } else { "ies" },
            result.types_proposed.len(),
            result.types_rejected.len(),
            model_str,
        );
        tracing::warn!(
            target: "kremory::dream::discover_types",
            group_id = %group_id,
            model = %model_str,
            catch_all_count = catch_alls.len(),
            proposed = result.types_proposed.len(),
            rejected = result.types_rejected.len(),
            "{msg}"
        );
        result.warnings.push(msg);
    }

    Ok(result)
}

