//! Pass 0 type discovery primitive (ADR-037 §3 / §9).
//!
//! ## What this does
//!
//! 1. **Signal source** — loads all `entity_type_id = 0` entities for `group_id`.
//! 2. **Top-K clusters** — groups catch-alls by name, ranks by frequency, takes
//!    top `max_proposals` clusters as proposal candidates (one cluster → one LLM prompt entry).
//!    (Embedding-based clustering is deferred to v0.2.0; name-frequency grouping is the
//!    v0.1.1 minimal-first implementation anchored by ADR-037 §9.2.)
//! 3. **LLM proposal call** — single `StructuredCallBuilder` call asking the LLM to
//!    propose entity types for the supplied clusters.  `max_proposals` is enforced
//!    PROMPT-SIDE (cap in the system prompt), not post-emission filter.
//! 4. **Shape validator** — each proposal runs through `validate_proposed_name`
//!    (9 rejection categories per ADR-037 §3.1).
//! 5. **Anti-redundancy gate** — if an embedder is available, each proposal's
//!    description and name are embedded and compared against existing types at
//!    0.85 / 0.70 cosine thresholds.  Without an embedder the gate is skipped
//!    and a warning is recorded (D7 degraded mode).
//! 6. **Persistence** — accepted proposals are inserted into `entity_types` via
//!    `INSERT OR IGNORE` with 4 provenance columns (Migration 014).
//! 7. **In-place evidence retype** — evidence entities whose name is semantically
//!    close to the new type's description (cosine ≥ 0.75) are updated to
//!    `entity_type_id = new_id, entity_type_source = 'DreamPass0'`.
//!    Without an embedder: retype ALL evidence entities for the accepted type (D4).
//!
//!    **TD-123 guard (default OFF):** this cosine-only comparison is a BARE
//!    ENTITY NAME (`entities.id`) embedded against a TYPE DESCRIPTION — the same
//!    degenerate-embedding failure class TD-097/ADR-063 documented (short bare
//!    labels collapse to near-identical vectors under `nomic-embed-text`), just
//!    cross-domain instead of name-vs-name. Unlike ADR-063's six sites, this
//!    comparison was never spike-validated (ADR-063: "every kremory-specific
//!    numeric threshold MUST PASS a `/ship-spike`... BEFORE it is wired into the
//!    production path") and has no deterministic corroboration signal — a
//!    lexical gate on entity-name-vs-type-name would reject legitimate matches
//!    (an instance name like "Nobu Malibu" shares no lemma with its type
//!    "Restaurant"), so the ADR-057/063 lexical-gate pattern does not transplant
//!    here unmodified. Gated behind `DreamOpts::include_evidence_retype_by_
//!    similarity` (default `false`) pending a proper spike (mirrors the
//!    quarantine-until-spiked posture ADR-063 used for every other new
//!    mechanism). When OFF, evidence entities are left as catch-all
//!    (`entity_type_id = 0`) — Pass 2 `reclassify` (LLM + confidence-gated, not
//!    cosine-alone) runs immediately after Pass 0 in the same `mem.dream()` call
//!    and safely picks up promotion instead (ADR-037 §9.6), so disabling this
//!    path does not lose retype coverage, only the risky cosine-alone shortcut.
//!
//! ## Observability (ADR-037 §6)
//!
//! All 6 required metrics are emitted:
//! - `kremory.dream.types_proposed_total{model, namespace}`
//! - `kremory.dream.types_accepted_total{model, namespace}`
//! - `kremory.dream.types_rejected_total{reason, model, namespace}`
//! - `kremory.dream.entities_retyped_total{source, namespace}`
//! - `kremory.dream.proposal_call_duration_ms{model}` (histogram)
//! - `kremory.dream.anti_redundancy_gate_skipped_total{reason="no_embedder"}`

use std::time::Instant;

use chrono::Utc;
use metrics::{counter, histogram};

use crate::core::{
    dream::{
        anti_redundancy::{self, GateOutcome},
        proposed_type::{validate_proposed_name, DiscoveryProposalBatch, RejectionReason},
    },
    entity_types::EntityTypeRegistry,
    error::Result,
    extraction::structured::StructuredCallBuilder,
    identity_verdict::{
        identity_verdict_batch_schema, IdentityVerdictBatch, IdentityVerdictItem,
        LLM_VERIFY_CONFIDENCE_FLOOR,
    },
    provider::{chat_msg_system, chat_msg_user, ChatProvider, DynEmbeddingProvider},
};

// ─── Public result types ──────────────────────────────────────────────────────

/// A single discovered or proposed entity type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypeProposal {
    pub name: String,
    pub description: String,
    pub justification: String,
}

/// Result of a single `discover_types` invocation.
#[derive(Debug, Default)]
pub struct DiscoveryResult {
    /// All proposals emitted by the LLM (before gating).
    pub types_proposed: Vec<TypeProposal>,
    /// Proposals that passed both the shape validator and the anti-redundancy gate.
    pub types_accepted: Vec<TypeProposal>,
    /// Proposals that were rejected, with the reason.
    pub types_rejected: Vec<(TypeProposal, String)>,
    /// Warnings for operator attention (e.g. degraded-mode runs).
    pub warnings: Vec<String>,
    /// Number of evidence entities retyped in-place.
    pub entities_retyped: usize,
}

/// Maximum cluster candidates passed to the LLM in one call.
/// Enforced PROMPT-SIDE — embedded in the system prompt instruction.
pub(crate) const MAX_PROPOSALS: usize = 5;

/// Cosine threshold for in-place evidence retype (ADR-037 §9.6 / §3.5).
const EVIDENCE_RETYPE_COSINE: f32 = 0.75;

/// Site label used on every shared `kremory.identity.*` counter for Site #2
/// (ADR-063 spec §6 site-labeled observability convention, mirrors Site #3's
/// `type_registry_collapse.rs::SITE_LABEL`).
const SITE_LABEL: &str = "site2_type_novelty";

// ─── Main primitive ───────────────────────────────────────────────────────────

/// Discover new entity types from catch-all entities in `group_id`.
///
/// Called by `mem.dream()` when `include_type_discovery = true` (D6).
/// Can also be called standalone via the escape-hatch `mem.discover_types()` (ADR-037 §4.2).
///
/// Bundled non-generic parameters for [`discover_types`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). The generic `llm: &L` stays a
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
    /// Concrete model id for capability detection (TD-094). Threaded from the
    /// facade dream path (`dream_model_id_or_main`). Empty (`""`) → `PromptOnly`
    /// degrade — the correct behaviour when the model is unknown. Before TD-094
    /// this was hardcoded to `String::new()`, silently degrading every call.
    pub(crate) model_id: &'a str,
    /// Site #2 (ADR-063 "The six sites" #2) — `DreamOpts::include_type_novelty_
    /// llm_verify`, threaded from `facade/dream.rs`. `false` (default): a
    /// `GateOutcome::NeedsLlmVerify` classification falls back to the pre-Site-#2
    /// behaviour (≥0.85 → reject, else accept) — the DEFAULT build's outcome is
    /// UNCHANGED. `true`: `NeedsLlmVerify` proposals are adjudicated via the
    /// shared `write_gate` (spec §2.2).
    pub(crate) llm_verify_band: bool,
    /// TD-123 — `DreamOpts::include_evidence_retype_by_similarity`, threaded
    /// from `facade/dream.rs`. `false` (default): the in-place evidence-retype
    /// step (D4) skips the cosine-only bare-name-vs-type-description comparison
    /// entirely (unspiked degeneracy risk, see module docs) and leaves evidence
    /// entities as catch-all for Pass 2 `reclassify` to pick up safely. `true`:
    /// opt-in to the pre-TD-123 cosine-alone retype behaviour.
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
    // TD-094: the consumer-supplied model id (Option-1, 2026-06-23) now reaches
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
            // TD-097 (Site #1): only the DESCRIPTION is embedded now. The name signal
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

    // TD-210: one run_id per `discover_types` invocation, shared by every
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
            // TD-097 (Site #1): no name embedding — `check_proposal` uses a
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
                    // TD-210: leave evidence for this rejection regardless of
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
                    // TD-210: this is the branch that previously left NO trace
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

                    // Flag ON (ADR-065): trust the LLM as terminal arbiter — a
                    // Site-#2-LOCAL decision that intentionally does NOT call the
                    // shared `write_gate`. Type synonyms ("Firm"/"Company") are
                    // lexically dissimilar by nature, so write_gate Row 6's
                    // deterministic-corroboration requirement (an ADR-057 ENTITY-
                    // homonymy guard) over-generalized to schema types and
                    // downgraded correct confident `true` verdicts to accept →
                    // duplicate types. See `type_novelty_is_redundant` + ADR-065.
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
                                 on weak evidence; ADR-065)"
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

    // ── Fail-loud on silent zero-discovery (TD-052 fix) ───────────────────────
    //
    // We only reach here with a NON-EMPTY catch-all bucket (empty buckets and
    // empty clusters early-returned above), so `types_accepted.is_empty()` here
    // means discovery *engaged but produced nothing usable* — model too weak
    // (e.g. interactive-tier gemma4-e2b emits placeholder names that the shape
    // validator rejects) or prompt drift. Previously this was SILENT: the
    // DreamSummary looked identical to "nothing to discover". Pass-0 is
    // non-fatal, so we WARN (never abort): a counter + a tracing::warn! + a
    // consumer-visible `DiscoveryResult.warnings` entry. See ADR-037 §6 +
    // [[project_dream_discovery_needs_deferred_quality_model]].
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Persist an accepted proposal and retype its evidence entities.
///
/// Uses Migration 014's provenance columns (`discovered_at`, `discovered_by`,
/// `evidence_count`, `confidence`).
///
/// Bundled parameters for [`accept_proposal`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments). Same precedent as
/// `extraction/structured.rs:TryArmParams`. A plain field-literal struct (all
/// fields required) — no builder layer needed.
struct AcceptProposalParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    model_str: &'a str,
    proposal: &'a TypeProposal,
    catch_alls: &'a [CatchAllEntity],
    /// (proposal_desc_embedding, embedder) — `None` = degraded mode.
    desc_emb_and_embedder: Option<(&'a [f32], &'a dyn DynEmbeddingProvider)>,
    /// TD-123 — `DreamOpts::include_evidence_retype_by_similarity` (default
    /// `false`). Gates ONLY the `desc_emb_and_embedder = Some(..)` cosine-only
    /// retype path (see module docs); irrelevant when `desc_emb_and_embedder`
    /// is `None` (degraded mode always uses `retype_evidence_all`, unaffected).
    evidence_retype_by_similarity: bool,
    result: &'a mut DiscoveryResult,
}

async fn accept_proposal(params: AcceptProposalParams<'_>) -> Result<()> {
    let AcceptProposalParams {
        conn,
        group_id,
        model_str,
        proposal,
        catch_alls,
        desc_emb_and_embedder,
        evidence_retype_by_similarity,
        result,
    } = params;
    let now = Utc::now().to_rfc3339();
    let discovered_by = format!("dream:llm:{model_str}");

    // Count evidence entities that belong to this proposal
    // (all catch-alls contribute to the proposal's evidence_count in the minimal impl)
    let evidence_count = catch_alls.len() as i64;

    // INSERT OR IGNORE into entity_types with Migration 014 provenance columns.
    //
    // `id` MUST be allocated explicitly: `entity_types` has a COMPOSITE primary
    // key `(group_id, id)` (migrations/defs_b.rs:207), so `id` is NOT a rowid
    // alias and does NOT auto-assign. Omitting it inserts NULL → `NOT NULL
    // constraint failed: entity_types.id`. We mirror `label_to_id_or_register`
    // (entity_types.rs:460/491): allocate `COALESCE(MAX(id),0)+1` per group_id
    // as a subquery inside the INSERT (atomic at statement level). New custom
    // types land above any seeded range; id=0 catch-all is never touched. On a
    // name clash, UNIQUE(group_id, name) triggers OR IGNORE and the SELECT-back
    // below returns the existing id. (Fixes TD-051.)
    conn.execute(
        "INSERT OR IGNORE INTO entity_types \
         (group_id, id, name, description, discovered_at, discovered_by, evidence_count, confidence) \
         VALUES (?1, (SELECT COALESCE(MAX(id), 0) + 1 FROM entity_types WHERE group_id = ?1), \
                 ?2, ?3, ?4, ?5, ?6, ?7)",
        libsql::params![
            group_id.to_string(),
            proposal.name.clone(),
            proposal.description.clone(),
            now.clone(),
            discovered_by.clone(),
            evidence_count,
            1.0f64, // Pass 0 treats all surviving proposals as confidence = 1.0
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: INSERT entity_types failed for '{}': {e}",
            proposal.name
        ))
    })?;

    // Re-load the assigned ID (INSERT OR IGNORE means we must SELECT back)
    let new_id: u32 = {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2 LIMIT 1",
                libsql::params![group_id.to_string(), proposal.name.clone()],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "discover_types: SELECT new id failed for '{}': {e}",
                    proposal.name
                ))
            })?;
        let row = rows.next().await.map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: SELECT row read failed for '{}': {e}",
                proposal.name
            ))
        })?;
        match row {
            Some(r) => r.get::<i64>(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "discover_types: id column read failed: {e}"
                ))
            })? as u32,
            None => {
                tracing::warn!(
                    target: "kremory::dream::discover_types",
                    name = %proposal.name,
                    "discover_types: row vanished after INSERT — skipping retype"
                );
                return Ok(());
            }
        }
    };

    // Count + emit accepted metric
    counter!(
        "kremory.dream.types_accepted_total",
        "model" => model_str.to_string(),
        "namespace" => group_id.to_string()
    )
    .increment(1);
    result.types_accepted.push(TypeProposal {
        name: proposal.name.clone(),
        description: proposal.description.clone(),
        justification: proposal.justification.clone(),
    });

    tracing::info!(
        target: "kremory::dream::discover_types",
        group_id = %group_id,
        type_name = %proposal.name,
        new_id = new_id,
        "discover_types: accepted and persisted new entity type"
    );

    // ── In-place evidence retype (D4) ─────────────────────────────────────────
    //
    // If embedder available AND `evidence_retype_by_similarity` opted in: retype
    // only evidence entities whose name embedding is cosine ≥ 0.75 to the new
    // type's description embedding (ADR-037 §9.6). TD-123 (default OFF): this
    // bare-name-vs-type-description comparison is an unspiked degeneracy risk
    // (module docs) — when the flag is off, evidence entities are left as
    // catch-all here; Pass 2 `reclassify` (LLM + confidence-gated) picks them up
    // safely in the same `mem.dream()` call.
    //
    // If no embedder (degraded): retype ALL evidence entities for this type
    // (accepted type's name is the only signal) — unaffected by the flag.

    let retyped_count = if let Some((desc_emb, emb)) = desc_emb_and_embedder {
        if evidence_retype_by_similarity {
            retype_evidence_by_similarity(RetypeBySimilarityParams {
                conn,
                group_id,
                catch_alls,
                new_type_id: new_id,
                type_desc_emb: desc_emb,
                emb,
                now: &now,
            })
            .await?
        } else {
            counter!(
                "kremory.dream.evidence_retype_by_similarity_skipped_total",
                "namespace" => group_id.to_string()
            )
            .increment(1);
            tracing::debug!(
                target: "kremory::dream::discover_types",
                group_id = %group_id,
                type_name = %proposal.name,
                new_id = new_id,
                "discover_types: TD-123 — skipping cosine-only evidence retype \
                 (DreamOpts::include_evidence_retype_by_similarity is false); \
                 evidence entities remain catch-all for Pass 2 reclassify"
            );
            0
        }
    } else {
        retype_evidence_all(RetypeAllParams {
            conn,
            group_id,
            catch_alls,
            new_type_id: new_id,
            now: &now,
        })
        .await?
    };

    if retyped_count > 0 {
        counter!(
            "kremory.dream.entities_retyped_total",
            "source" => "pass_0_evidence",
            "namespace" => group_id.to_string()
        )
        .increment(retyped_count as u64);
        result.entities_retyped += retyped_count;
    }

    Ok(())
}

/// Bundled parameters for [`retype_evidence_by_similarity`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). Same precedent as
/// `AcceptProposalParams` above.
struct RetypeBySimilarityParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    catch_alls: &'a [CatchAllEntity],
    new_type_id: u32,
    type_desc_emb: &'a [f32],
    emb: &'a dyn DynEmbeddingProvider,
    now: &'a str,
}

/// Retype evidence entities with cosine ≥ 0.75 similarity to the type description.
async fn retype_evidence_by_similarity(params: RetypeBySimilarityParams<'_>) -> Result<usize> {
    let RetypeBySimilarityParams {
        conn,
        group_id,
        catch_alls,
        new_type_id,
        type_desc_emb,
        emb,
        now,
    } = params;
    let mut count = 0usize;
    for entity in catch_alls {
        // Embed the entity's stored ID (which is the normalised name)
        let entity_name_emb = match emb.embed_dyn(&entity.id).await {
            Ok(v) => v,
            Err(_) => continue,
        };
        let sim = anti_redundancy::cosine(&entity_name_emb, type_desc_emb);
        if sim >= EVIDENCE_RETYPE_COSINE {
            retype_entity(RetypeEntityParams {
                conn,
                group_id,
                entity_id: &entity.id,
                new_type_id,
                now,
            })
            .await?;
            count += 1;
        }
    }
    Ok(count)
}

/// Bundled parameters for [`retype_evidence_all`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct RetypeAllParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    catch_alls: &'a [CatchAllEntity],
    new_type_id: u32,
    now: &'a str,
}

/// Retype all evidence catch-all entities (degraded mode: no embedder).
async fn retype_evidence_all(params: RetypeAllParams<'_>) -> Result<usize> {
    let RetypeAllParams {
        conn,
        group_id,
        catch_alls,
        new_type_id,
        now,
    } = params;
    for entity in catch_alls {
        retype_entity(RetypeEntityParams {
            conn,
            group_id,
            entity_id: &entity.id,
            new_type_id,
            now,
        })
        .await?;
    }
    Ok(catch_alls.len())
}

/// Bundled parameters for [`retype_entity`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct RetypeEntityParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    entity_id: &'a str,
    new_type_id: u32,
    now: &'a str,
}

/// Update a single entity's type to `new_type_id` with DreamPass0 provenance (D4).
async fn retype_entity(params: RetypeEntityParams<'_>) -> Result<()> {
    let RetypeEntityParams {
        conn,
        group_id,
        entity_id,
        new_type_id,
        now,
    } = params;
    conn.execute(
        "UPDATE entities \
         SET entity_type_id = ?1, \
             entity_type_source = 'DreamPass0', \
             entity_type_assigned_at = ?2 \
         WHERE id = ?3 AND group_id = ?4 AND entity_type_id = 0",
        libsql::params![
            new_type_id as i64,
            now.to_string(),
            entity_id.to_string(),
            group_id.to_string(),
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: retype entity failed for id='{}': {e}",
            entity_id
        ))
    })?;
    Ok(())
}

// ─── Catch-all entity loader ──────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct CatchAllEntity {
    pub(crate) id: String,
}

async fn load_catch_all_entities(
    conn: &libsql::Connection,
    group_id: &str,
) -> Result<Vec<CatchAllEntity>> {
    let mut rows = conn
        .query(
            // `id` tiebreak makes ordering deterministic when candidates tie on
            // access_count (e.g. freshly-ingested entities all at 0) — the row
            // order feeds the LLM prompt, so an unspecified tie-break makes the
            // whole pass non-reproducible run-to-run (surfaced by dream-loop E1).
            "SELECT id FROM entities \
             WHERE group_id = ?1 AND entity_type_id = 0 \
             ORDER BY access_count DESC, id",
            libsql::params![group_id.to_string()],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: load_catch_all_entities query failed: {e}"
            ))
        })?;

    let mut entities = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: load_catch_all_entities row read failed: {e}"
        ))
    })? {
        let id: String = row.get(0).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: load_catch_all_entities id read failed: {e}"
            ))
        })?;
        entities.push(CatchAllEntity { id });
    }
    Ok(entities)
}

// ─── Name-frequency clustering (v0.1.1 minimal) ──────────────────────────────

/// A cluster of catch-all entities with the same normalised name.
#[derive(Debug)]
pub(crate) struct NameCluster {
    /// Representative name (first occurrence).
    pub(crate) name: String,
    /// Number of catch-all entities with this name.
    pub(crate) count: usize,
}

/// Group catch-alls by normalised name, rank by frequency, take top K.
///
/// ADR-037 §9.2 specifies embedding-cosine clustering at threshold 0.7 with
/// minimum cluster size 3. v0.1.1 minimal-first: name-frequency grouping.
/// Embedding clustering ships in v0.2.0 when the embedder is always available.
fn top_k_clusters(entities: &[CatchAllEntity], k: usize) -> Vec<NameCluster> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for entity in entities {
        let norm = normalise_name(&entity.id);
        *counts.entry(norm).or_insert(0) += 1;
    }
    // Filter: minimum cluster size of 2 (relaxed from spec's 3 at v0.1.1 to avoid
    // suppressing all proposals on small test namespaces).
    let mut clusters: Vec<NameCluster> = counts
        .into_iter()
        .filter(|(_, c)| *c >= 1) // include singletons at v0.1.1
        .map(|(name, count)| NameCluster { name, count })
        .collect();
    // Sort descending by count, then ascending by name as a DETERMINISTIC
    // tiebreak. The HashMap `.into_iter()` above yields randomized per-process
    // order, so a count-only sort leaves tied clusters in arbitrary order and
    // `truncate(k)` then keeps an arbitrary subset run-to-run — making this
    // pass's LLM prompt (and thus dream results) non-reproducible. Surfaced by
    // dream-loop E2; sibling of the ORDER BY tiebreak fix (b8208c9).
    clusters.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
    clusters.truncate(k);
    clusters
}

/// Normalise entity id/name for clustering: lowercase + trim.
fn normalise_name(s: &str) -> String {
    s.trim().to_lowercase()
}

// ─── Prompt builder ───────────────────────────────────────────────────────────

fn build_discovery_messages(
    clusters: &[NameCluster],
    max_proposals: usize,
) -> Vec<crate::core::provider::ChatMessage> {
    let cluster_list = clusters
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. \"{}\" (seen {} time(s))", i + 1, c.name, c.count))
        .collect::<Vec<_>>()
        .join("\n");

    let system = format!(
        "You are a knowledge-graph type analyst. Your task is to propose NEW entity type \
categories for a knowledge graph.\n\
\n\
You will receive a list of entity names that could not be classified into existing types. \
Your job is to propose up to {max_proposals} NEW entity type definitions that would cover \
these entities.\n\
\n\
Rules:\n\
- Propose at most {max_proposals} new types. Fewer is fine.\n\
- Each type must have a short name (3-50 characters), a clear description, and a justification.\n\
- The name must be a concrete noun phrase — no placeholders like 'Unknown', 'Other', 'TBD', 'N/A'.\n\
- The description should explain what kinds of entities belong to this type.\n\
- Be specific: 'MedicalDevice' is better than 'Equipment'. 'LegalCase' is better than 'Item'.\n\
- Respond ONLY with the JSON structure — no extra commentary."
    );

    let user = format!(
        "These entities could not be classified into existing types:\n\n{cluster_list}\n\n\
Propose up to {max_proposals} new entity type categories that would cover them. \
Output JSON with a `proposals` array containing objects with `name`, `description`, and `justification` fields."
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

// ─── Rejection reason → string ────────────────────────────────────────────────

fn reason_to_string(r: RejectionReason) -> String {
    r.as_str().to_string()
}

// ─── Site #2 LLM-verify-band adjudication (ADR-063 spec §2.2/§4.4 sibling) ────

/// `pub` + `#[doc(hidden)]` (MNT-002 pattern, same as the Site #3/#5 test-utils
/// re-exports in `mod.rs`) — this and [`adjudicate_type_novelty`] were
/// previously module-private; promoted so `tests/dream_metrics_harness_site2.rs`
/// (an external integration-test binary) can replicate the Site #2
/// discover_types decision exactly. Not part of the stable public API contract.
#[doc(hidden)]
pub struct AdjudicateTypeNoveltyParams<'a, L: ChatProvider> {
    pub llm: &'a L,
    pub model_id: &'a str,
    pub proposal_name: &'a str,
    pub proposal_desc: &'a str,
    pub existing_name: &'a str,
    pub existing_desc: &'a str,
    pub group_id: &'a str,
}

/// Adjudicate ONE Site #2 candidate pair via a single-item `IdentityVerdictBatch`
/// call, reusing the shared schema (spec §2.1) exactly as
/// `type_registry_collapse.rs::adjudicate_batch` does for Site #3 — this is a
/// one-item batch rather than a genuinely new call shape, since Pass 0 proposals
/// are adjudicated one at a time as they surface in the per-proposal loop (unlike
/// Site #3, which nominates all candidate pairs up front and can batch them
/// together in one dream-pass invocation).
///
/// Returns `None` on any LLM/parse failure — the caller's `write_gate` treats a
/// missing verdict as "no LLM adjudication" (spec §2.3 failure-mode default: never
/// silently promotes to Merge).
///
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern) — promoted from module-private so
/// `tests/dream_metrics_harness_site2.rs` can call this directly, replicating
/// the exact discover_types Site #2 decision flow.
#[doc(hidden)]
pub async fn adjudicate_type_novelty<L: ChatProvider>(
    params: AdjudicateTypeNoveltyParams<'_, L>,
) -> Option<IdentityVerdictItem> {
    let AdjudicateTypeNoveltyParams {
        llm,
        model_id,
        proposal_name,
        proposal_desc,
        existing_name,
        existing_desc,
        group_id,
    } = params;

    let messages = build_type_novelty_adjudication_messages(BuildTypeNoveltyMessagesParams {
        proposal_name,
        proposal_desc,
        existing_name,
        existing_desc,
    });
    let schema = identity_verdict_batch_schema(1);

    let call_start = Instant::now();
    let raw_value = StructuredCallBuilder::new(llm, &schema, "IdentityVerdictBatch")
        .model(model_id)
        .messages(messages)
        .call()
        .await;
    let elapsed_ms = call_start.elapsed().as_millis() as f64;
    histogram!(
        "kremory.identity.llm_call_latency_ms_histogram",
        "site" => SITE_LABEL
    )
    .record(elapsed_ms);

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory::dream::discover_types::raw_payload",
            model_id = %model_id,
            group_id = %group_id,
            response = ?raw_value,
            "discover_types Site #2 adjudication raw response"
        );
    }

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "kremory::dream::discover_types",
                error = %e,
                group_id = %group_id,
                "discover_types: Site #2 adjudication LLM call failed — defaulting to no-verdict"
            );
            return None;
        }
    };

    let batch: IdentityVerdictBatch = match serde_json::from_value(raw_value.clone()) {
        Ok(b) => b,
        Err(_) => {
            let repaired = if raw_value.is_array() {
                serde_json::json!({ "verdicts": raw_value })
            } else {
                raw_value.clone()
            };
            match serde_json::from_value::<IdentityVerdictBatch>(repaired) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::discover_types",
                        error = %e,
                        group_id = %group_id,
                        "discover_types: failed to parse Site #2 IdentityVerdictBatch — defaulting to no-verdict"
                    );
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(1);
                    return None;
                }
            }
        }
    };

    for raw_item in &batch.verdicts {
        match serde_json::from_value::<IdentityVerdictItem>(raw_item.clone()) {
            Ok(item) => return Some(item),
            Err(e) => {
                counter!(
                    "kremory.identity.verdict_parse_fail_total",
                    "site" => SITE_LABEL
                )
                .increment(1);
                tracing::warn!(
                    target: "kremory::dream::discover_types",
                    error = %e,
                    "discover_types: skipping malformed Site #2 verdict item"
                );
            }
        }
    }
    None
}

/// Bundled parameters for [`build_type_novelty_adjudication_messages`] —
/// args-as-object per TD-042 (rust-conventions §too_many_arguments, threshold 3).
struct BuildTypeNoveltyMessagesParams<'a> {
    proposal_name: &'a str,
    proposal_desc: &'a str,
    existing_name: &'a str,
    existing_desc: &'a str,
}

fn build_type_novelty_adjudication_messages(
    params: BuildTypeNoveltyMessagesParams<'_>,
) -> Vec<crate::core::provider::ChatMessage> {
    let BuildTypeNoveltyMessagesParams {
        proposal_name,
        proposal_desc,
        existing_name,
        existing_desc,
    } = params;
    let system = "You are a knowledge-graph type registry analyst. You will be shown a \
PROPOSED new entity-type definition (name + description) alongside an EXISTING \
registered type that a similarity gate has flagged as POSSIBLY the same underlying \
type. Decide whether the two type definitions describe the SAME semantic category of \
entity (e.g. 'Company' and 'Business Organisation' are the same; 'LegalPrecedent' and \
'LegalRuling' are DISTINCT, related-but-different legal concepts). \
Respond ONLY with the JSON structure — no extra commentary."
        .to_string();

    let user = format!(
        "Adjudicate this single candidate pair. Output a `verdicts` array with exactly one \
verdict object: `pair_id` (use 0), `is_same_entity` (true if the two definitions describe \
the same type), `confidence` (0.0-1.0), and `reasoning` (a short free-text justification).\n\n\
Proposed: name=\"{proposal_name}\" description=\"{proposal_desc}\"\n\
Existing: name=\"{existing_name}\" description=\"{existing_desc}\""
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

/// ADR-065: Site #2 type-novelty write decision. UNLIKE the shared
/// [`write_gate`](crate::core::identity_verdict::write_gate) (`identity_verdict.rs`),
/// this does NOT require deterministic (lexical) corroboration — type synonyms
/// ("Firm"/"Company") are lexically dissimilar by nature, and schema-level
/// matching trusts the LLM as terminal arbiter (ADR-065; ontology-alignment
/// prior art). `write_gate`'s Row 6 protects against ENTITY homonymy (same name,
/// different referent), a failure mode that cannot recur here after the
/// exact-match pre-filter. See ADR-065 for the residual-risk analysis. The two
/// decision rules carry a bidirectional doc cross-reference (Vera Finding 3) so
/// a maintainer grepping `write_gate` finds this carve-out and does not re-unify
/// them.
///
/// Returns `true` when the proposal is REDUNDANT with the existing type (the LLM
/// is confident they are the same concept) → the caller rejects it. `false`
/// (novel / low-confidence / no verdict) → the caller conservatively accepts.
///
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern, mirrors `adjudicate_type_novelty`)
/// so `tests/dream_metrics_harness_site2.rs` replicates the discover_types Site
/// #2 decision flow via the REAL fn (single source of truth). Not part of the
/// stable public API contract.
#[doc(hidden)]
pub fn type_novelty_is_redundant(verdict: &Option<IdentityVerdictItem>) -> bool {
    matches!(verdict, Some(v) if v.is_same_entity && v.confidence >= LLM_VERIFY_CONFIDENCE_FLOOR)
}

/// ADR-065 observability (Rule 19): a per-decision counter for the Site #2
/// type-novelty gate. NOT the shared `write_gate_decision_total` — Site #2
/// bypasses `write_gate`, so a distinct, honestly-named metric avoids conflating
/// the two decision rules. `decision` ∈ {`redundant`, `novel`,
/// `accept_low_confidence`}.
fn record_type_novelty_decision(redundant: bool, verdict: &Option<IdentityVerdictItem>) {
    let decision = if redundant {
        "redundant"
    } else if matches!(verdict, Some(v) if !v.is_same_entity) {
        "novel"
    } else {
        "accept_low_confidence"
    };
    counter!(
        "kremory.identity.type_novelty_decision_total",
        "site" => SITE_LABEL,
        "decision" => decision
    )
    .increment(1);
}

/// Bundled parameters for [`record_gate_decision`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments).
struct RecordGateDecisionParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    /// Shared across every row this `discover_types` invocation writes —
    /// generated once per call (see the call site), not once per row.
    run_id: &'a str,
    proposal_name: &'a str,
    /// `None` ONLY when no existing (non-catch-all) type existed to compare
    /// against at all (the `check_proposal` registry-empty early return) —
    /// never because a comparison happened and its result was discarded.
    existing_name: Option<&'a str>,
    /// `None` ONLY when no cosine was ever computed for this decision (the
    /// TD-097 exact-name pre-filter fires before the desc-cosine loop runs).
    desc_cosine: Option<f32>,
    /// `Some` only on the flag-on LLM-verify sub-path; `None` for every
    /// deterministic decision (Pass, Redundant, and the flag-off
    /// `NeedsLlmVerify` fallback, which makes no LLM call).
    verdict: Option<&'a IdentityVerdictItem>,
    /// `"accept"` | `"reject"` — the FINAL decision (did the proposal end up
    /// persisted into `entity_types`).
    decision: &'a str,
    /// Fine-grained provenance for the metric label — which branch of the
    /// gate decided this, e.g. `"pass_no_existing_types"`,
    /// `"redundant_exact_name"`, `"llm_verify_accept"`. See call sites.
    outcome_kind: &'a str,
    model: &'a str,
}

/// TD-210: leave a durable, always-on trace of EVERY Pass-0 anti-redundancy
/// gate decision — Pass, Redundant, and both `NeedsLlmVerify` sub-paths (flag
/// off AND flag on) — independent of `DreamOpts::include_type_novelty_llm_verify`.
///
/// Before this, only the flag-on LLM-verify sub-path left ANY trace at all,
/// and even that was metric-only (`record_type_novelty_decision`, above) —
/// no persisted row. A default run's `Pass` accepts (the overwhelming
/// majority of decisions) left ZERO evidence: `identity_verdict_audit` never
/// received a Site #2 row, so an accepted proposal's `desc_cosine` could not
/// be reconstructed from stored state after the fact (see tech-debt register
/// TD-210, "the flag-on counterfactual CANNOT be read from stored state").
///
/// Emits BOTH:
/// 1. An always-on counter (`kremory.dream.type_novelty_gate_decision_total`)
///    labelled by `outcome` (fine-grained branch) + `decision` (accept/reject),
///    following this module's existing `kremory.dream.*_total{model,namespace}`
///    label convention (see `types_proposed_total` / `types_accepted_total` /
///    `types_rejected_total` above).
/// 2. A persisted `identity_verdict_audit` row (Migration 018,
///    `identity_verdict_prereqs`) — the SAME table Site #3
///    (`type_registry_collapse.rs::write_audit_row`) and Site #5
///    (`acronym_nickname_recall.rs`) already write to, reusing its
///    `site`/`cosine`/`structural_signal`/`llm_*`/`decision`/`run_id` shape.
///    `structural_signal` mirrors what those sites mean by it (a corroborating
///    deterministic/lexical signal) — for Site #2 that is
///    `names_share_lemma_or_exact(proposal_name, existing_name)`, computed
///    fresh here since none of the `Pass`/flag-off/flag-on branches already
///    carry it forward.
///
/// `candidate_b` (`identity_verdict_audit.candidate_b`) is `NOT NULL TEXT` —
/// unlike Site #3/#5, which always compare two NAMED things, Site #2's `Pass`
/// outcome can legitimately have NO existing type to name (an empty registry).
/// That case persists `candidate_b = ""` (an empty string can never collide
/// with a real type name — the shape validator requires 3-50 characters), so
/// the ABSENCE of a comparison is still visible in the row rather than adding
/// a schema migration for one nullable column on a table three sites share.
///
/// Runs OUTSIDE any DB transaction (this function is not wrapped in
/// `BEGIN`/`COMMIT` anywhere in `discover_types`, unlike Site #3's
/// merge-transaction audit rows) — so there is no "emit inside a transaction
/// that might roll back" hazard here; this call always reflects a decision
/// that has already been finalised (the `accept_proposal` INSERT, when this
/// is an accept, has already been awaited and returned `Ok` before this runs).
async fn record_gate_decision(params: RecordGateDecisionParams<'_>) -> Result<()> {
    let RecordGateDecisionParams {
        conn,
        group_id,
        run_id,
        proposal_name,
        existing_name,
        desc_cosine,
        verdict,
        decision,
        outcome_kind,
        model,
    } = params;

    counter!(
        "kremory.dream.type_novelty_gate_decision_total",
        "outcome" => outcome_kind.to_string(),
        "decision" => decision.to_string(),
        "model" => model.to_string(),
        "namespace" => group_id.to_string()
    )
    .increment(1);

    let structural_signal = existing_name
        .map(|n| anti_redundancy::names_share_lemma_or_exact(proposal_name, n))
        .unwrap_or(false);

    conn.execute(
        "INSERT INTO identity_verdict_audit \
         (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
          llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        libsql::params![
            SITE_LABEL,
            group_id,
            proposal_name,
            existing_name.unwrap_or(""),
            desc_cosine.map(f64::from),
            structural_signal,
            verdict.map(|v| v.is_same_entity),
            verdict.map(|v| f64::from(v.confidence)),
            verdict.map(|v| v.reasoning.clone()),
            decision,
            run_id,
        ],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: identity_verdict_audit insert failed for '{proposal_name}': {e}"
        ))
    })?;

    Ok(())
}

// ─── ADR-065 type_novelty_is_redundant unit tests ────────────────────────────
#[cfg(test)]
mod adr065_type_novelty_redundant_tests {
    use super::*;

    fn verdict(is_same: bool, confidence: f32) -> Option<IdentityVerdictItem> {
        Some(IdentityVerdictItem {
            pair_id: 0,
            is_same_entity: is_same,
            confidence,
            reasoning: String::new(),
        })
    }

    /// is_same + confidence at/above the floor → REDUNDANT (the fix's core case:
    /// a correct confident `true` verdict is no longer downgraded by write_gate
    /// Row 6). This is exactly the s2-001 Firm/Company shape.
    #[test]
    fn is_same_high_conf_is_redundant() {
        assert!(type_novelty_is_redundant(&verdict(
            true,
            LLM_VERIFY_CONFIDENCE_FLOOR
        )));
        assert!(type_novelty_is_redundant(&verdict(true, 1.0)));
    }

    /// is_same but confidence BELOW the floor → NOT redundant (conservative
    /// accept; weak agreement must not reject a new type).
    #[test]
    fn is_same_below_floor_not_redundant() {
        assert!(!type_novelty_is_redundant(&verdict(
            true,
            LLM_VERIFY_CONFIDENCE_FLOOR - 0.01
        )));
    }

    /// LLM says distinct → NOT redundant (EDC false-reject-prevention: accept).
    #[test]
    fn not_same_not_redundant() {
        assert!(!type_novelty_is_redundant(&verdict(false, 1.0)));
    }

    /// No verdict (LLM/parse failure) → NOT redundant (conservative accept —
    /// a failed adjudication must not silently reject a proposal).
    #[test]
    fn no_verdict_not_redundant() {
        assert!(!type_novelty_is_redundant(&None));
    }
}

// ─── TD-051 regression tests ──────────────────────────────────────────────────
//
// `accept_proposal` is the ADR-037 Pass-0 persistence step. It was buried below
// the LLM call + clustering + anti-redundancy gate, so the only test exercising
// it was the `#[ignore]`d real-LLM smoke (`tests/phase_d_pass_0.rs`) — which hid
// TD-051 (INSERT omitted `id` on a composite-PK table → runtime NOT NULL crash).
// These tests drive the REAL `accept_proposal` deterministically (no LLM, no
// embedder; empty `catch_alls` makes retype a no-op) so the persistence/id
// allocation is asserted in the default `cargo test` gate.
#[cfg(test)]
mod td051_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::schema::TemporalGraph;

    async fn accepted_type_id(conn: &libsql::Connection, group_id: &str, name: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
                libsql::params![group_id, name],
            )
            .await
            .expect("select discovered type");
        rows.next()
            .await
            .expect("row iter")
            .expect("discovered-type row must exist — INSERT must not have crashed")
            .get::<i64>(0)
            .expect("id column")
    }

    /// First discovered type allocates `id = 10` — above the seeded range (0..=9).
    /// Before the TD-051 fix this panicked with `NOT NULL constraint failed`.
    #[tokio::test]
    async fn accept_proposal_allocates_id_above_seed_range() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "g1")
            .await
            .expect("seed defaults 0..=9");

        let proposal = TypeProposal {
            name: "Vehicle".to_string(),
            description: "A car, truck, or other conveyance.".to_string(),
            justification: "Several catch-all entities were vehicles.".to_string(),
        };
        let mut result = DiscoveryResult::default();

        accept_proposal(AcceptProposalParams {
            conn: &conn,
            group_id: "g1",
            model_str: "test-model",
            proposal: &proposal,
            catch_alls: &[],
            desc_emb_and_embedder: None,
            evidence_retype_by_similarity: false,
            result: &mut result,
        })
        .await
        .expect("accept_proposal must persist the discovered type (TD-051)");

        assert_eq!(
            accepted_type_id(&conn, "g1", "Vehicle").await,
            10,
            "first discovered type allocates id=10 (above seeded 0..=9)"
        );
        assert_eq!(result.types_accepted.len(), 1, "one type accepted");
        assert_eq!(result.types_accepted[0].name, "Vehicle");
    }

    /// Successive discoveries climb `MAX(id)+1` (10, 11) without colliding with
    /// the seeded range or the id=0 catch-all.
    #[tokio::test]
    async fn accept_proposal_increments_id_across_discoveries() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "g1")
            .await
            .expect("seed");

        for (name, expected_id) in [("Vehicle", 10i64), ("Statute", 11i64)] {
            let proposal = TypeProposal {
                name: name.to_string(),
                description: format!("description for {name}"),
                justification: "j".to_string(),
            };
            let mut result = DiscoveryResult::default();
            accept_proposal(AcceptProposalParams {
                conn: &conn,
                group_id: "g1",
                model_str: "m",
                proposal: &proposal,
                catch_alls: &[],
                desc_emb_and_embedder: None,
                evidence_retype_by_similarity: false,
                result: &mut result,
            })
            .await
            .expect("accept_proposal persists");
            assert_eq!(
                accepted_type_id(&conn, "g1", name).await,
                expected_id,
                "{name} must allocate id={expected_id}"
            );
        }

        // id=0 catch-all is untouched by discovery.
        assert_eq!(accepted_type_id(&conn, "g1", "Entity").await, 0);
    }
}

// ─── TD-050 deterministic full-workflow test ──────────────────────────────────
//
// The TD-051 tests above drive `accept_proposal` (the persistence STEP) in
// isolation. The only test exercising the FULL chain (load catch-alls → cluster
// → LLM proposal → shape-validate → accept → retype evidence) was the `#[ignore]`d
// real-LLM smoke (`tests/phase_d_pass_0.rs`), which is stochastic + shape-only
// ("types_discovered is stochastic — type-shape check only"). So the SEMANTIC
// correctness of discovery — does it grow the table by the proposed type AND
// retype the catch-all evidence with `entity_type_source='DreamPass0'`? — was
// unasserted in the default `cargo test` gate (the TD-050 gap).
//
// This test closes it deterministically: a scripted `ChatProvider` returns a
// KNOWN proposal batch and `embedder = None` takes the degraded-mode path
// (anti-redundancy gate skipped → no embedding-similarity nondeterminism), so
// the OUTCOME is fully determined and asserted. No live LLM; runs in CI.
#[cfg(test)]
mod td050_full_workflow_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::provider::{
        ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
    };
    use crate::core::schema::TemporalGraph;

    /// Scripted `ChatProvider` that returns one fixed discovery-proposal batch,
    /// ignoring the prompt entirely (mirrors `tests/helpers/scripted_llm.rs`).
    #[derive(Debug)]
    struct ScriptedProposalProvider {
        json: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedProposalProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: self.json.clone(),
            }))
        }
    }

    /// Full workflow: discover_types proposes a KNOWN type, grows `entity_types`
    /// above the seeded range, and retypes ALL catch-all evidence in-place with
    /// `entity_type_source='DreamPass0'`. Asserts the OUTCOME, not the shape.
    #[tokio::test]
    async fn discover_types_grows_table_and_retypes_evidence_deterministically() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "legal")
            .await
            .expect("seed defaults 0..=9");

        // Three out-of-vocab entities parked as catch-all (entity_type_id = 0) —
        // the residue Pass-0 discovery operates on. Minimal-column INSERT per the
        // precedent at schema.rs (id, entity_type_id, recorded_at, group_id).
        let now = Utc::now().to_rfc3339();
        for id in ["vanguard therapeutics", "acme capital", "nexus ventures"] {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![id.to_string(), now.clone(), "legal".to_string()],
            )
            .await
            .expect("insert catch-all entity");
        }

        let llm = ScriptedProposalProvider {
            json: r#"{"proposals":[{"name":"Company","description":"A business organisation, firm, or investment fund.","justification":"Vanguard Therapeutics, Acme Capital and Nexus Ventures are all companies."}]}"#
                .to_string(),
        };

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "legal",
                embedder: None,
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        // ── Outcome of the discovery result (not shape) ───────────────────────
        assert_eq!(result.types_proposed.len(), 1, "exactly one type proposed");
        assert_eq!(result.types_accepted.len(), 1, "exactly one type accepted");
        assert_eq!(
            result.types_accepted[0].name, "Company",
            "the accepted type is the one the scripted LLM proposed"
        );
        assert!(
            result.types_rejected.is_empty(),
            "'Company' is a valid name — must not be rejected, got {:?}",
            result.types_rejected
        );
        assert_eq!(
            result.entities_retyped, 3,
            "all 3 catch-all entities retyped in degraded mode"
        );

        // ── entity_types table grew by the KNOWN type, id above seeded range ──
        let new_id: i64 = {
            let mut rows = conn
                .query(
                    "SELECT id FROM entity_types WHERE group_id = 'legal' AND name = 'Company'",
                    (),
                )
                .await
                .expect("query discovered type");
            rows.next()
                .await
                .expect("row iter")
                .expect("'Company' row must exist — discovery must have persisted it")
                .get::<i64>(0)
                .expect("id column")
        };
        assert!(
            new_id > 9,
            "discovered type id {new_id} must be above the seeded 0..=9 range"
        );

        // ── every catch-all entity retyped to the new id with DreamPass0 ──────
        let mut rows = conn
            .query(
                "SELECT entity_type_id, entity_type_source FROM entities \
                 WHERE group_id = 'legal'",
                (),
            )
            .await
            .expect("query retyped entities");
        let mut checked = 0usize;
        while let Some(r) = rows.next().await.expect("row iter") {
            let tid: i64 = r.get(0).expect("entity_type_id");
            let src: String = r.get(1).expect("entity_type_source");
            assert_eq!(
                tid, new_id,
                "every catch-all entity must be retyped to the discovered type id"
            );
            assert_eq!(
                src, "DreamPass0",
                "retype provenance must be 'DreamPass0' (ADR-037 D4)"
            );
            checked += 1;
        }
        assert_eq!(
            checked, 3,
            "all 3 entities present and retyped — none left at id=0"
        );
    }

    /// TD-052 fail-loud: when discovery engages on a non-empty catch-all bucket
    /// but accepts ZERO types (here the scripted proposal is a `"..."` placeholder
    /// the shape validator rejects — the exact gemma4-e2b real-world failure), the
    /// result must carry a loud, consumer-visible warning, NOT silently look like
    /// "nothing to discover". Evidence must NOT be retyped.
    #[tokio::test]
    async fn discover_types_warns_loud_when_all_proposals_rejected() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "med")
            .await
            .expect("seed defaults 0..=9");

        let now = Utc::now().to_rfc3339();
        for id in ["aspirin", "ibuprofen", "paracetamol"] {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![id.to_string(), now.clone(), "med".to_string()],
            )
            .await
            .expect("insert catch-all entity");
        }

        // Scripted placeholder name — rejected by the validator (ellipsis_placeholder).
        let llm = ScriptedProposalProvider {
            json: r#"{"proposals":[{"name":"...","description":"x","justification":"y"}]}"#
                .to_string(),
        };

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "med",
                embedder: None,
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed (zero-discovery is non-fatal)");

        assert_eq!(result.types_proposed.len(), 1, "one proposal seen");
        assert!(
            result.types_accepted.is_empty(),
            "the '...' placeholder must be rejected → zero accepted"
        );
        assert_eq!(result.types_rejected.len(), 1, "one rejection");
        assert_eq!(
            result.entities_retyped, 0,
            "nothing accepted → nothing retyped"
        );

        // The load-bearing assertion: zero-discovery is LOUD, not silent.
        assert!(
            result.warnings.iter().any(|w| w.contains("ZERO accepted")),
            "a non-empty catch-all bucket yielding zero accepted types must emit a \
             consumer-visible warning; warnings={:?}",
            result.warnings
        );

        // Evidence untouched — still catch-all.
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE group_id = 'med' AND entity_type_id = 0",
                (),
            )
            .await
            .expect("count");
        let still_catch_all: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("count row")
            .get::<i64>(0)
            .expect("count col");
        assert_eq!(
            still_catch_all, 3,
            "all 3 entities remain catch-all when discovery accepts nothing"
        );
    }
}

// ─── Site #2 (ADR-063 "The six sites" #2) — type-novelty LLM-verify band ──────
//
// Deterministic tests for the `DreamOpts::include_type_novelty_llm_verify` flag:
// (a) flag OFF preserves the pre-Site-#2 outcome exactly (regression guard); (b)
// flag ON + LLM says "same" → proposal rejected as redundant; (c) flag ON + LLM
// says "distinct" → proposal accepted (the EDC false-reject-prevention case this
// site exists to fix). Mirrors `type_registry_collapse.rs`'s
// `ScriptedVerdictProvider` pattern, but `discover_types` makes TWO sequential LLM
// calls when the LLM-verify band engages (1: the discovery proposal call, 2: the
// Site #2 adjudication call) — `ScriptedSequenceProvider` returns one scripted
// response per call, in order.
#[cfg(test)]
mod site2_type_novelty_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::provider::{
        ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
    };
    use crate::core::schema::TemporalGraph;
    use std::sync::Mutex;

    /// Scripted `ChatProvider` returning one fixed JSON response PER CALL, in
    /// order (first call gets `responses[0]`, second gets `responses[1]`, ...).
    /// Panics if called more times than responses are scripted — a test-shape
    /// bug, not a production concern.
    #[derive(Debug)]
    struct ScriptedSequenceProvider {
        responses: Mutex<std::collections::VecDeque<String>>,
    }

    impl ScriptedSequenceProvider {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            }
        }
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedSequenceProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            let mut queue = self.responses.lock().expect("mutex poisoned");
            let text = queue
                .pop_front()
                .expect("ScriptedSequenceProvider: called more times than scripted responses");
            Ok(Box::new(MockChatResponse { text }))
        }
    }

    /// Seed a group with the default 0..=9 types plus ONE custom existing type
    /// (id=11) whose description embeds to `unit_vec4(1,0,0,0)` — used as the
    /// Site #2 candidate's `existing_name`/`existing_desc`.
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn seed_existing_type(conn: &libsql::Connection, group_id: &str, name: &str, desc: &str) {
        ensure_default_types_seeded(conn, group_id)
            .await
            .expect("seed defaults 0..=9");
        conn.execute(
            "INSERT INTO entity_types (group_id, id, name, description) VALUES (?1, 11, ?2, ?3)",
            libsql::params![group_id, name, desc],
        )
        .await
        .expect("insert existing type");
    }

    /// Deterministic embedder: pre-registered vector per input text (exact
    /// match), zero vector otherwise (mirrors `type_registry_collapse.rs`'s
    /// `MockEmbeddingProvider`).
    #[derive(Debug, Clone)]
    struct MockEmbeddingProvider {
        vectors: std::collections::HashMap<String, Vec<f32>>,
    }

    impl DynEmbeddingProvider for MockEmbeddingProvider {
        fn embed_dyn<'a>(
            &'a self,
            text: &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>>
        {
            let v = self
                .vectors
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0_f32; 4]);
            Box::pin(async move { Ok(v) })
        }
        fn last_usage_tokens_dyn(&self) -> Option<u64> {
            None
        }
    }

    /// (a) Flag OFF (default): a proposal whose desc-cosine is ≥0.85 against an
    /// existing type with ZERO lemma overlap (the `NeedsLlmVerify` case) is
    /// REJECTED — the exact pre-Site-#2 outcome (hard cutoff, no LLM adjudication
    /// call made). Only ONE LLM call is scripted (the discovery proposal call) —
    /// if the flag wrongly engaged Site #2 adjudication, `ScriptedSequenceProvider`
    /// would panic on a second call, proving no extra call was made.
    #[tokio::test]
    async fn flag_off_preserves_pre_site2_hard_cutoff_reject() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        // "Individual" (NOT the seeded default "Person", id=1 — avoid a UNIQUE
        // collision) — zero lemma overlap with the proposed "Human".
        seed_existing_type(
            &conn,
            "g1",
            "Individual",
            "A living individual, described by name and biography.",
        )
        .await;

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
            libsql::params!["alice smith", now, "g1"],
        )
        .await
        .expect("insert catch-all entity");

        // Proposal "Human" desc embeds identically to "Individual" desc → cosine 1.0.
        let shared_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(
            "A living individual, described by name and biography.".to_string(),
            shared_vec.clone(),
        );
        vectors.insert(
            "A person, described by name and biography facts.".to_string(),
            shared_vec,
        );
        let embedder = MockEmbeddingProvider { vectors };

        // Only ONE response scripted — the discovery proposal call. No second
        // (adjudication) call should ever be made with the flag off.
        let llm = ScriptedSequenceProvider::new(vec![
            r#"{"proposals":[{"name":"Human","description":"A person, described by name and biography facts.","justification":"catch-all evidence"}]}"#,
        ]);

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g1",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        assert!(
            result.types_accepted.is_empty(),
            "flag OFF: NeedsLlmVerify at >=0.85 must reject (pre-Site-#2 hard cutoff), got accepted={:?}",
            result.types_accepted
        );
        assert_eq!(result.types_rejected.len(), 1, "one rejection");
        assert!(
            result.types_rejected[0].1.starts_with("redundant_with:"),
            "rejection reason must be redundant_with:, got {:?}",
            result.types_rejected[0].1
        );
    }

    /// (a') Flag OFF: a proposal in the [0.70, 0.85) ambiguous band is ACCEPTED —
    /// the pre-Site-#2 "no candidate" outcome (there was no ambiguous-band concept
    /// before Site #2; anything below 0.85 simply passed).
    #[tokio::test]
    async fn flag_off_mid_band_accepts() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_existing_type(
            &conn,
            "g2",
            "Vehicle",
            "A car, truck, or other conveyance used for transport.",
        )
        .await;

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
            libsql::params!["red wagon", now, "g2"],
        )
        .await
        .expect("insert catch-all entity");

        // Cosine 0.80 (mid-band): A=[1,0,0,0], B=[0.8,0.6,0,0].
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(
            "A car, truck, or other conveyance used for transport.".to_string(),
            vec![1.0f32, 0.0, 0.0, 0.0],
        );
        vectors.insert(
            "A wheeled conveyance pulled or pushed by hand.".to_string(),
            vec![0.8f32, 0.6, 0.0, 0.0],
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedSequenceProvider::new(vec![
            r#"{"proposals":[{"name":"Conveyance","description":"A wheeled conveyance pulled or pushed by hand.","justification":"catch-all evidence"}]}"#,
        ]);

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g2",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        assert_eq!(
            result.types_accepted.len(),
            1,
            "flag OFF: mid-band [0.70,0.85) must accept (pre-Site-#2 'no candidate' outcome), rejected={:?}",
            result.types_rejected
        );
    }

    /// (b) Flag ON + scripted LLM `is_same_entity=true` (high confidence) →
    /// `type_novelty_is_redundant` (ADR-065) → the proposal IS redundant →
    /// REJECTED. (The lemma overlap between "Organizations"/"Organization" is now
    /// irrelevant to the decision — it mattered only to the OLD write_gate Row 5;
    /// the confident `true` verdict alone is decisive under ADR-065.) Mid-band
    /// cosine (0.80, in `[0.70, 0.85)`) is used so `check_proposal` nominates
    /// `NeedsLlmVerify` rather than auto-rejecting at the pure classification step
    /// (lemma overlap at cosine ≥0.85 would short-circuit to `Redundant` before
    /// any LLM call).
    #[tokio::test]
    async fn flag_on_llm_says_same_with_lemma_signal_rejects_proposal() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_existing_type(
            &conn,
            "g3",
            "Organization",
            "A group of people with a shared purpose or structure.",
        )
        .await;

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
            libsql::params!["acme co", now, "g3"],
        )
        .await
        .expect("insert catch-all entity");

        // Cosine 0.80 (mid-band): A=[1,0,0,0], B=[0.8,0.6,0,0].
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(
            "A group of people with a shared purpose or structure.".to_string(),
            vec![1.0f32, 0.0, 0.0, 0.0],
        );
        vectors.insert(
            "A collective body of people organised for a common purpose.".to_string(),
            vec![0.8f32, 0.6, 0.0, 0.0],
        );
        let embedder = MockEmbeddingProvider { vectors };

        // Call 1: discovery proposal "Organizations" (trailing-s lemma match with
        // "Organization" → deterministic signal fires). Call 2: Site #2
        // adjudication → LLM says same.
        let llm = ScriptedSequenceProvider::new(vec![
            r#"{"proposals":[{"name":"Organizations","description":"A collective body of people organised for a common purpose.","justification":"catch-all evidence"}]}"#,
            r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.95,"reasoning":"same concept as Organization"}]}"#,
        ]);

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g3",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: true,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        assert!(
            result.types_accepted.is_empty(),
            "flag ON + confident LLM true verdict: proposal must be rejected as \
             redundant (ADR-065 type_novelty_is_redundant), got accepted={:?}",
            result.types_accepted
        );
        assert_eq!(result.types_rejected.len(), 1);
        assert!(result.types_rejected[0].1.starts_with("redundant_with:"));
    }

    /// (b') Flag ON + scripted LLM `is_same_entity=true` (high confidence) but
    /// WITHOUT any deterministic lemma signal — the exact Site #2 bug ADR-065
    /// fixes. Under the OLD shared `write_gate` this hit Row 6 (no deterministic
    /// corroboration → `PotentialAlias` → accept → DUPLICATE type). Under
    /// ADR-065's `type_novelty_is_redundant`, a confident `true` verdict is the
    /// terminal arbiter (type synonyms are lexically dissimilar by nature) → the
    /// proposal IS redundant → REJECTED. This assertion FLIPPED with the fix.
    #[tokio::test]
    async fn flag_on_llm_says_same_without_lemma_signal_now_rejects_redundant() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_existing_type(
            &conn,
            "g5",
            "Individual",
            "A living individual, described by name and biography.",
        )
        .await;

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
            libsql::params!["bob jones", now, "g5"],
        )
        .await
        .expect("insert catch-all entity");

        let shared_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(
            "A living individual, described by name and biography.".to_string(),
            shared_vec.clone(),
        );
        vectors.insert(
            "A person, described by name and biography facts.".to_string(),
            shared_vec,
        );
        let embedder = MockEmbeddingProvider { vectors };

        // "Human"/"Individual" share zero lemma overlap → no deterministic signal.
        let llm = ScriptedSequenceProvider::new(vec![
            r#"{"proposals":[{"name":"Human","description":"A person, described by name and biography facts.","justification":"catch-all evidence"}]}"#,
            r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.95,"reasoning":"same concept as Individual"}]}"#,
        ]);

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g5",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: true,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        assert!(
            result.types_accepted.is_empty(),
            "ADR-065: confident LLM `is_same=true` verdict (no lemma signal needed) → \
             redundant → REJECT (was accept under write_gate Row 6), accepted={:?}",
            result.types_accepted
        );
        assert_eq!(result.types_rejected.len(), 1);
        assert!(result.types_rejected[0].1.starts_with("redundant_with:"));
    }

    /// (c) Flag ON + scripted LLM `is_same_entity=false` → the proposal is
    /// DISTINCT → ACCEPTED. This is the EDC false-reject-prevention case Site #2
    /// exists to fix: a hard 0.85 cutoff alone would have rejected this proposal
    /// (desc-cosine 1.0, zero lemma overlap with "Person"), but the LLM correctly
    /// distinguishes "Human" (a person, generically) from "Person" (an existing
    /// registered type) as intended for this fixture — the LLM verdict is honored.
    #[tokio::test]
    async fn flag_on_llm_says_distinct_accepts_proposal() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_existing_type(
            &conn,
            "g4",
            "LegalRuling",
            "A court's official decision on a case.",
        )
        .await;

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
            libsql::params!["marbury v madison", now, "g4"],
        )
        .await
        .expect("insert catch-all entity");

        let shared_vec = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(
            "A court's official decision on a case.".to_string(),
            shared_vec.clone(),
        );
        vectors.insert(
            "A prior court decision used as precedent for future cases.".to_string(),
            shared_vec,
        );
        let embedder = MockEmbeddingProvider { vectors };

        // Call 1: discovery proposal "LegalPrecedent" (desc-cosine 1.0 vs
        // "LegalRuling", zero lemma overlap → NeedsLlmVerify). Call 2: Site #2
        // adjudication → LLM correctly says DISTINCT.
        let llm = ScriptedSequenceProvider::new(vec![
            r#"{"proposals":[{"name":"LegalPrecedent","description":"A prior court decision used as precedent for future cases.","justification":"catch-all evidence"}]}"#,
            r#"{"verdicts":[{"pair_id":0,"is_same_entity":false,"confidence":0.9,"reasoning":"precedent and ruling are related but distinct legal concepts"}]}"#,
        ]);

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g4",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: true,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        assert_eq!(
            result.types_accepted.len(),
            1,
            "flag ON + LLM false verdict: distinct-but-similar type must be accepted \
             (EDC false-reject-prevention case), rejected={:?}",
            result.types_rejected
        );
        assert_eq!(result.types_accepted[0].name, "LegalPrecedent");
    }

    /// TD-210 — the flag-OFF, no-LLM-call, ACCEPT-via-`Pass` path is exactly
    /// the branch the tech-debt register flags as leaving ZERO evidence: prior
    /// to this fix, `discover_types` never wrote to `identity_verdict_audit`
    /// at all on `GateOutcome::Pass`, so an accepted proposal's `desc_cosine`
    /// (the value that decided it was novel enough to accept) could not be
    /// reconstructed from stored state after the fact — only re-embedding the
    /// stored descriptions live could recover it (register TD-210, "the
    /// flag-on counterfactual CANNOT be read from stored state").
    ///
    /// One existing type is seeded so a REAL comparison happens (best_cosine
    /// is computed, not skipped) — and the proposal's description embeds
    /// ORTHOGONALLY to it, so cosine ~0.0, well below `TYPE_NOVELTY_LOWER_BAND`
    /// (0.70) → `GateOutcome::Pass` with `existing_name = Some`, `desc_cosine
    /// = Some(0.0)`. The assertion is specifically that the persisted cosine
    /// is NON-NULL — proving the audit row records a REAL comparison, not the
    /// registry-empty `None` case (that's covered by
    /// `check_proposal`'s own `gate_passes_with_empty_existing_types` unit
    /// test in `anti_redundancy.rs`).
    #[tokio::test]
    async fn accepted_proposal_with_existing_type_leaves_audit_row_with_cosine() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_existing_type(
            &conn,
            "g_audit",
            "WeatherEvent",
            "A meteorological occurrence such as a storm or heatwave.",
        )
        .await;

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, 0, ?2, ?3)",
            libsql::params!["some catch-all evidence", now, "g_audit"],
        )
        .await
        .expect("insert catch-all entity");

        // Orthogonal vectors: existing type at dim0, proposal at dim1 → cosine
        // 0.0, well below TYPE_NOVELTY_LOWER_BAND (0.70) → GateOutcome::Pass
        // with existing_name=Some + desc_cosine=Some(0.0) — a REAL comparison.
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(
            "A meteorological occurrence such as a storm or heatwave.".to_string(),
            vec![1.0f32, 0.0, 0.0, 0.0],
        );
        vectors.insert(
            "A legal instrument transferring ownership of real property.".to_string(),
            vec![0.0f32, 1.0, 0.0, 0.0],
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedSequenceProvider::new(vec![
            r#"{"proposals":[{"name":"PropertyDeed","description":"A legal instrument transferring ownership of real property.","justification":"catch-all evidence"}]}"#,
        ]);

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g_audit",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must succeed");

        assert_eq!(
            result.types_accepted.len(),
            1,
            "orthogonal, distinct proposal must be accepted, rejected={:?}",
            result.types_rejected
        );

        let mut rows = conn
            .query(
                "SELECT cosine, decision, structural_signal FROM identity_verdict_audit \
                 WHERE site = 'site2_type_novelty' AND group_id = 'g_audit' \
                 AND candidate_a = 'PropertyDeed'",
                (),
            )
            .await
            .expect("query identity_verdict_audit");
        let row = rows
            .next()
            .await
            .expect("row read")
            .expect(
                "an accepted Pass-0 proposal must leave an identity_verdict_audit row (TD-210) \
                 — this is the RED assertion: pre-fix, discover_types never wrote to this table \
                 on the Pass path at all, so this query returns zero rows",
            );
        let cosine: Option<f64> = row.get(0).expect("cosine column");
        let decision: String = row.get(1).expect("decision column");
        let structural_signal: bool = row.get(2).expect("structural_signal column");
        assert!(
            cosine.is_some(),
            "an accepted proposal compared against a real existing type must persist a \
             NON-NULL cosine — the evidence for WHY it was accepted (TD-210)"
        );
        assert_eq!(decision, "accept");
        assert!(
            !structural_signal,
            "\"PropertyDeed\"/\"WeatherEvent\" share no lemma or exact-name overlap"
        );
    }
}

// ─── TD-123 — cosine-alone evidence-retype guard ──────────────────────────────
//
// Vera-surfaced 7th cosine-alone-write site (outside ADR-063's six enumerated
// sites): `retype_evidence_by_similarity` embeds a BARE ENTITY NAME and
// compares it to the newly-accepted TYPE's DESCRIPTION embedding — the same
// degenerate-embedding failure class TD-097 documented (short bare labels
// collapse to near-identical vectors under weak embedders), just cross-domain
// (name vs. description) rather than name-vs-name. This test simulates that
// exact degeneracy: an UNRELATED catch-all entity's name embeds IDENTICALLY
// (cosine = 1.0) to a newly-discovered, semantically-unrelated type's
// description — proving the DEFAULT build does not wrongly retype it.
#[cfg(test)]
mod td123_evidence_retype_guard_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::provider::{
        ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
    };
    use crate::core::schema::TemporalGraph;

    #[derive(Debug)]
    struct ScriptedProposalProvider {
        json: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedProposalProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: self.json.clone(),
            }))
        }
    }

    /// Deterministic embedder: pre-registered vector per input text (exact
    /// match), zero vector otherwise (mirrors `site2_type_novelty_tests`'s
    /// `MockEmbeddingProvider`).
    #[derive(Debug, Clone)]
    struct MockEmbeddingProvider {
        vectors: std::collections::HashMap<String, Vec<f32>>,
    }

    impl DynEmbeddingProvider for MockEmbeddingProvider {
        fn embed_dyn<'a>(
            &'a self,
            text: &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>>
        {
            let v = self
                .vectors
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0_f32; 4]);
            Box::pin(async move { Ok(v) })
        }
        fn last_usage_tokens_dyn(&self) -> Option<u64> {
            None
        }
    }

    async fn seed_catch_all(conn: &libsql::Connection, group_id: &str, entity_id: &str) {
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
             VALUES (?1, 0, ?2, ?3)",
            libsql::params![entity_id.to_string(), now, group_id.to_string()],
        )
        .await
        .expect("insert catch-all entity");
    }

    async fn entity_type_id(conn: &libsql::Connection, group_id: &str, entity_id: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT entity_type_id FROM entities WHERE id = ?1 AND group_id = ?2",
                libsql::params![entity_id.to_string(), group_id.to_string()],
            )
            .await
            .expect("query entity_type_id");
        let row = rows
            .next()
            .await
            .expect("row read")
            .expect("entity must exist");
        row.get::<i64>(0).expect("entity_type_id column")
    }

    const DEGENERATE_PROPOSAL_JSON: &str = r#"{"proposals":[{"name":"Recipe","description":"A step-by-step cooking guide with ingredients and cook time.","justification":"catch-all evidence suggests a recipe type."}]}"#;

    /// RED (pre-fix): with the OLD unguarded code, `cosine("ibm", recipe_desc)
    /// = 1.0 >= EVIDENCE_RETYPE_COSINE (0.75)` retypes "ibm" (an unrelated
    /// catch-all entity) to the newly-discovered "Recipe" type — a false
    /// retype driven by cosine alone, with zero lexical/LLM corroboration.
    ///
    /// GREEN (post-fix): `DreamOpts::include_evidence_retype_by_similarity`
    /// defaults `false`, so the cosine-only retype path is skipped entirely —
    /// "ibm" stays catch-all (`entity_type_id = 0`), left for Pass 2
    /// `reclassify`'s LLM+confidence gate to handle safely.
    #[tokio::test]
    async fn default_flag_off_does_not_retype_on_degenerate_cosine_collision() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "g1")
            .await
            .expect("seed defaults 0..=9");
        seed_catch_all(&conn, "g1", "ibm").await;

        // Degenerate collision: the catch-all entity's bare name embeds
        // IDENTICALLY to the new type's description (cosine = 1.0), simulating
        // an anisotropic embedder that does not discriminate unrelated bare
        // labels (TD-097's `cos(Person, Date) = 1.0000` finding, cross-domain).
        let collision_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mut vectors = std::collections::HashMap::new();
        vectors.insert("ibm".to_string(), collision_vec.clone());
        vectors.insert(
            "A step-by-step cooking guide with ingredients and cook time.".to_string(),
            collision_vec,
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedProposalProvider {
            json: DEGENERATE_PROPOSAL_JSON.to_string(),
        };

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g1",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false, // DEFAULT
            },
        )
        .await
        .expect("discover_types must succeed");

        assert_eq!(
            result.types_accepted.len(),
            1,
            "the Recipe type itself must still be accepted (distinct from \
             existing defaults) — only the RETYPE decision is guarded, got {:?}",
            result.types_rejected
        );
        assert_eq!(
            result.entities_retyped, 0,
            "flag OFF (default): the cosine-only retype path must be entirely \
             skipped — a degenerate collision must not retype 'ibm' into 'Recipe'"
        );
        assert_eq!(
            entity_type_id(&conn, "g1", "ibm").await,
            0,
            "'ibm' must remain catch-all (entity_type_id=0) — a bare-name-vs-\
             type-description cosine collision is not a valid retype signal \
             without corroboration (TD-123)"
        );
    }

    /// Regression: explicitly opting IN to the pre-TD-123 behaviour still
    /// retypes on the same degenerate collision — proves the flag genuinely
    /// gates the code path (not a permanently-dead branch) and preserves the
    /// escape hatch for a caller who has independently validated the threshold.
    #[tokio::test]
    async fn flag_on_preserves_pre_td123_cosine_retype_behaviour() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "g2")
            .await
            .expect("seed defaults 0..=9");
        seed_catch_all(&conn, "g2", "ibm").await;

        let collision_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mut vectors = std::collections::HashMap::new();
        vectors.insert("ibm".to_string(), collision_vec.clone());
        vectors.insert(
            "A step-by-step cooking guide with ingredients and cook time.".to_string(),
            collision_vec,
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedProposalProvider {
            json: DEGENERATE_PROPOSAL_JSON.to_string(),
        };

        let result = discover_types(
            &llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "g2",
                embedder: Some(&embedder),
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: true, // explicit opt-in
            },
        )
        .await
        .expect("discover_types must succeed");

        assert_eq!(
            result.entities_retyped, 1,
            "flag ON: opt-in preserves the pre-TD-123 cosine-only retype \
             behaviour on the same degenerate collision"
        );
        assert_ne!(
            entity_type_id(&conn, "g2", "ibm").await,
            0,
            "flag ON: 'ibm' is retyped away from catch-all, matching pre-TD-123 \
             behaviour exactly"
        );
    }
}

// ─── TD-050 real-LLM discovery-OUTCOME test ───────────────────────────────────
//
// Closes the gaps the existing `#[ignore]`d smoke (`tests/phase_d_pass_0.rs`)
// leaves open: that smoke (a) can pass VACUOUSLY (if stochastic extraction
// produced zero catch-alls, discovery early-returns and every assertion still
// passes), and (b) never asserts that evidence was RETYPED. This test drives a
// REAL model through the discovery path with a DETERMINISTIC trigger:
//
// - The catch-all bucket is seeded directly (5 drugs) — discovery cannot run
//   vacuously; the trigger is asserted to exist before the call.
// - Drugs are a type genuinely ABSENT from the 10 defaults, so a competent
//   model's proposal is NOT (correctly) rejected by anti-redundancy.
// - `embedder = None` skips the gate and takes `retype_evidence_all`, making the
//   RETYPE COUNT deterministic. The only stochastic element is "did the real
//   model propose >=1 valid type for an unambiguous drug cluster" — which is
//   exactly the discovery-quality signal this test exists to surface (and is
//   reliable for gemma4-e2b on a clear cluster).
//
// `#[ignore]` + `feature = "llm-integration"`: needs live Ollama. Run with:
//   OLLAMA_CHAT_MODEL=gemma4:e4b cargo test -p kremory \
//     --features llm-integration --lib discover_types_real_llm -- --ignored --nocapture
//
// MODEL TIER (load-bearing finding, 2026-06-22): defaults to `gemma4:e4b` — the
// DEFERRED-phase QUALITY model (90%, Phase 2 per tests/llm_integration.rs:1-25),
// NOT the interactive `gemma4-e2b`. Discovery is a background/quality task. The
// smoke run that built this test showed `gemma4-e2b` proposing a placeholder
// name `"..."` → rejected (`ellipsis_placeholder`) → ZERO types discovered,
// while `gemma4:e4b` proposes "Over-the-Counter Pain Reliever" → accepted → all
// evidence retyped. A consumer wiring only the fast interactive model for dreams
// gets silent zero-discovery. See [[project_dream_discovery_needs_deferred_quality_model]].
//
// Governed by ADR-037 (Pass-0 discovery, §9.6 D4 provenance). Complements the
// deterministic `td050_full_workflow_tests` (scripted proposal) by proving the
// REAL model end of the chain.
#[cfg(all(test, feature = "llm-integration"))]
mod td050_real_llm_tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::schema::TemporalGraph;
    use autoagents_llm::backends::ollama::Ollama;
    use autoagents_llm::builder::LLMBuilder;
    use std::sync::Arc;

    async fn count_catch_alls(conn: &libsql::Connection, group_id: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE group_id = ?1 AND entity_type_id = 0",
                libsql::params![group_id],
            )
            .await
            .expect("count catch-alls");
        rows.next()
            .await
            .expect("row")
            .expect("count row")
            .get::<i64>(0)
            .expect("count col")
    }

    async fn type_id_by_name(conn: &libsql::Connection, group_id: &str, name: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
                libsql::params![group_id, name],
            )
            .await
            .expect("select type id");
        rows.next()
            .await
            .expect("row")
            .expect("discovered-type row must exist")
            .get::<i64>(0)
            .expect("id col")
    }

    #[tokio::test]
    #[ignore]
    async fn discover_types_real_llm_proposes_accepts_and_retypes() {
        let graph = TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory");
        let conn = graph.conn.clone();
        ensure_default_types_seeded(&conn, "med")
            .await
            .expect("seed defaults 0..=9");

        // Deterministic catch-all trigger: 5 drugs (a type ABSENT from the 10
        // defaults). entity_type_id = 0 = catch-all.
        let now = Utc::now().to_rfc3339();
        let drugs = [
            "aspirin",
            "ibuprofen",
            "paracetamol",
            "metformin",
            "atorvastatin",
        ];
        for d in drugs {
            conn.execute(
                "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) \
                 VALUES (?1, 0, ?2, ?3)",
                libsql::params![d.to_string(), now.clone(), "med".to_string()],
            )
            .await
            .expect("insert catch-all entity");
        }

        // Non-vacuous guarantee: the discovery trigger MUST exist.
        assert_eq!(
            count_catch_alls(&conn, "med").await,
            5,
            "5 catch-all entities must exist before discovery — guards against a vacuous pass"
        );

        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        // Deferred-phase QUALITY model (Phase 2, 90% per tests/llm_integration.rs:1-25).
        // gemma4-e2b (interactive) is too weak for discovery — see module doc.
        let chat_model =
            std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());
        let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
            .base_url(&base_url)
            .model(&chat_model)
            .timeout_seconds(120)
            .keep_alive("1h")
            .build()
            .expect("Ollama LLM builder must succeed");

        // embedder = None → anti-redundancy gate skipped + retype_evidence_all
        // (deterministic retype count). Discovery itself is fully real.
        let result = discover_types(
            &*llm,
            DiscoverTypesParams {
                conn: &conn,
                group_id: "med",
                embedder: None,
                max_proposals: 3,
                model_id: "test-model",
                llm_verify_band: false,
                evidence_retype_by_similarity: false,
            },
        )
        .await
        .expect("discover_types must not error with a live model");

        // Surface what the real model actually discovered (operator observability).
        if std::env::var("KREMORY_DEBUG").is_ok() {
            tracing::debug!(
                target: "kremory.dream.discover_types",
                model = %chat_model,
                proposed = ?result.types_proposed.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
                accepted = ?result.types_accepted.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
                rejected = ?result.types_rejected.iter().map(|(t, r)| format!("{}:{r}", t.name)).collect::<Vec<_>>(),
                entities_retyped = result.entities_retyped,
                "td050-real-llm discover_types result"
            );
        }

        // Gap: the real model actually produced a usable proposal (not vacuous,
        // not scripted). Reliable for an unambiguous drug cluster; if this flaps,
        // that IS the discovery-quality signal this test surfaces.
        assert!(
            !result.types_proposed.is_empty(),
            "real model must propose >=1 type for an unambiguous Drug cluster"
        );
        assert!(
            !result.types_accepted.is_empty(),
            "the proposal must survive the shape validator and be accepted; rejected={:?}",
            result.types_rejected
        );

        // Gap: evidence retyped (deterministic in degraded mode → all 5).
        assert_eq!(
            result.entities_retyped, 5,
            "degraded-mode accept retypes ALL catch-all evidence"
        );

        // Consistency: every accepted type persisted above the seeded range.
        for t in &result.types_accepted {
            let id = type_id_by_name(&conn, "med", &t.name).await;
            assert!(
                id > 9,
                "discovered type '{}' must allocate id>9 (above seeded 0..=9), got {id}",
                t.name
            );
        }

        // Retype provenance: every drug entity now non-catch-all with DreamPass0.
        let mut rows = conn
            .query(
                "SELECT entity_type_id, entity_type_source FROM entities WHERE group_id = 'med'",
                (),
            )
            .await
            .expect("query retyped entities");
        let mut n = 0usize;
        while let Some(r) = rows.next().await.expect("row") {
            let tid: i64 = r.get(0).expect("entity_type_id");
            let src: String = r.get(1).expect("entity_type_source");
            assert!(
                tid > 9,
                "every drug entity must be retyped above the seeded range, got {tid}"
            );
            assert_eq!(src, "DreamPass0", "retype provenance must be 'DreamPass0'");
            n += 1;
        }
        assert_eq!(n, 5, "all 5 drug entities retyped — none left at id=0");
    }
}
