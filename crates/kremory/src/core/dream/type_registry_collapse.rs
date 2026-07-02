//! Dream pass — type-registry post-hoc collapse (ADR-063 spec §4, "Site #3").
//!
//! Periodic dream-phase pass that de-duplicates the `entity_types` registry for
//! a `group_id`: near-duplicate types (e.g. Pass-0-discovered "Company" +
//! "Business Organisation") are merged, remapping every `entities.entity_type_id`
//! that pointed at the loser onto the keeper.
//!
//! ## Pass shape (spec §4.0-§4.5)
//!
//! 1. Load `entity_types` for `group_id` (skip id=0 catch-all — never a merge
//!    candidate).
//! 2. Embed each type's **DESCRIPTION** (never the bare name — spec §4.1; bare-
//!    label cosine is TD-097's own confirmed degeneracy, `cos(Person, Date) =
//!    1.0000`). Degraded mode (no embedder): skip the pass entirely with a
//!    warning, mirroring `discover_types`'s D7 degraded-mode discipline.
//! 3. For every candidate pair, compute description-cosine + a lexical
//!    pre-filter on the type NAME (`normalize_name` exact-match, or a lemma
//!    match that strips a trailing `s` — spec §4.2).
//! 4. Band the pair per spec §4.3's cosine table: auto-merge (row 1
//!    equivalent, no LLM call), LLM-verify band, or no-candidate.
//! 5. LLM-verify band pairs are batched into ONE `IdentityVerdictBatch` call
//!    (shared schema, spec §2.1/§4.4) per dream-pass invocation.
//! 6. Each resolved pair is decided via the shared [`write_gate`] (spec §2.2).
//!    `WriteDecision::Merge` triggers keeper selection (spec §4.5: longest
//!    description wins, `evidence_count` secondary tiebreak) + an atomic
//!    `entity_type_id` remap + `identity_verdict_audit` row, all inside ONE
//!    `BEGIN IMMEDIATE` transaction (spec §5.1 RISK-003).
//!    `WriteDecision::PotentialAlias` writes an audit row only (no merge — types
//!    have no potential-alias edge concept, spec brief). `WriteDecision::Reject`
//!    writes nothing.
//!
//! ## Spike gating (spec §8)
//!
//! The 0.85 primary threshold is ADR-037's own already-calibrated number
//! (reused, not invented). The 0.70 lower band edge and the lemma-heuristic
//! pre-filter are SPIKE-GATED (S3) — this pass therefore ships behind
//! `DreamOpts::include_type_registry_collapse`, DEFAULT `false` (spec §8).
//!
//! ## Observability (spec §6)
//!
//! - `kremory.dream.type_registry_collapse.pairs_examined_total`
//! - `kremory.dream.type_registry_collapse.merges_applied_total`
//! - `kremory.dream.type_registry_collapse.lexical_prefilter_hit_total`
//! - `kremory.identity.candidate_nominated_total{site="site3_type_registry"}`
//! - `kremory.identity.verdict_parse_fail_total{site="site3_type_registry"}`
//! - `kremory.identity.llm_call_latency_ms_histogram{site="site3_type_registry"}`
//! - `kremory.identity.write_gate_decision_total{site="site3_type_registry",decision}`
//! - `kremory.identity.write_gate_llm_authorized_merge_total{site="site3_type_registry"}`
//! - `KREMORY_DEBUG=1` raw-payload trace at target
//!   `kremory::dream::type_registry_collapse::raw_payload`

use std::collections::HashMap;
use std::time::Instant;

use metrics::{counter, histogram};

use crate::core::entity_types::EntityTypeSpec;
use crate::core::error::{Error, Result};
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::identity_verdict::{
    identity_verdict_batch_schema, write_gate, DeterministicSignal, IdentityVerdictBatch,
    IdentityVerdictItem, WriteDecision, WriteGateInputs,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider, DynEmbeddingProvider};

/// Site label used on every shared `kremory.identity.*` counter (spec §6).
const SITE_LABEL: &str = "site3_type_registry";

/// ADR-037's existing primary description-cosine threshold, reused per spec
/// §4.3 (NOT a fresh number — R2's explicit recommendation against
/// threshold-per-site inconsistency).
pub(crate) const TYPE_COLLAPSE_PRIMARY_COSINE: f32 = 0.85;

/// Provisional lower band edge (spec §4.3/§8, spike-gated S3). Carried by
/// analogy from ADR-037's secondary name-gate value; NOT independently
/// derived for the description-gate use here.
pub(crate) const TYPE_COLLAPSE_LOWER_BAND_COSINE: f32 = 0.70;

// ─── Report ────────────────────────────────────────────────────────────────

/// Summary returned by [`type_registry_collapse`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypeRegistryCollapseReport {
    pub group_id: String,
    /// Number of type pairs examined (upper triangle, excluding id=0).
    pub pairs_examined: usize,
    /// Number of pairs where the cheap lexical pre-filter alone answered the
    /// pair without needing cosine/LLM (spec §4.2 — currently folded into the
    /// auto-merge path; tracked separately per §6's counter for visibility).
    pub lexical_prefilter_hits: usize,
    /// Number of pairs nominated for LLM adjudication (spec §4.3 LLM-verify band).
    pub candidates_nominated: usize,
    /// Number of merges applied (loser type removed, entities remapped).
    pub merges_applied: usize,
}

// ─── Internal type slot ──────────────────────────────────────────────────────

/// A loaded `entity_types` row plus its description embedding, used for
/// pairwise comparison. id=0 is never loaded into this slice (spec §4.0).
struct TypeSlot {
    spec: EntityTypeSpec,
    desc_embedding: Vec<f32>,
    /// `entity_types.evidence_count` (Migration 014) — how many entities have
    /// historically been assigned this type. Advisory LLM adjudication context
    /// (spec §4.4) AND the SECONDARY keeper tiebreak for near-equal-length
    /// descriptions (spec §4.5). NOT a `write_gate` input.
    evidence_count: i64,
}

/// A candidate pair nominated for the LLM-verify band (spec §4.3/§4.4),
/// correlated back to its `write_gate` inputs via index into the caller's
/// `nominated` vec (mirrors `IdentityVerdictItem::pair_id`, spec §2.1).
struct NominatedPair {
    a_idx: usize,
    b_idx: usize,
    cosine: f32,
    lexical_compatible: bool,
}

/// A resolved `loser -> keeper` merge with the pair-provenance needed to write
/// a faithful `identity_verdict_audit` row (spec §5.1). Carries the merge's
/// cosine and — for LLM-verify-band merges only — the winning
/// `IdentityVerdictItem`. Auto-merges (clear-case write_gate row 1, no LLM
/// call) carry `verdict = None` and write NO audit row (spec §5.2 — audit only
/// LLM-touched decisions).
struct MergeEdge {
    keeper_id: u32,
    cosine: Option<f32>,
    verdict: Option<IdentityVerdictItem>,
}

// ─── Public entry-point ───────────────────────────────────────────────────────

/// Bundled parameters for [`type_registry_collapse`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments). `llm` stays a lead generic
/// positional param (project convention, mirrors `discover_types<L>`).
///
/// `pub` + `#[doc(hidden)]` (not `pub(crate)`) per the MNT-002 precedent
/// (`dream/mod.rs`'s `consistency_check` re-export block): `pub(crate)` items
/// cannot be re-exported as `pub` (E0365), and the S3 spike's integration test
/// (`tests/type_registry_collapse_s3_spike.rs`) lives outside the crate
/// boundary. Re-exported under `feature = "test-utils"` in `dream/mod.rs` —
/// explicitly NOT part of the stable public API contract.
#[doc(hidden)]
pub struct TypeRegistryCollapseParams<'a> {
    pub conn: &'a libsql::Connection,
    pub group_id: &'a str,
    pub embedder: Option<&'a dyn DynEmbeddingProvider>,
    /// Concrete model id for capability detection (mirrors TD-094 threading
    /// used by `discover_types`/`reclassify`). Empty (`""`) → `PromptOnly` degrade.
    pub model_id: &'a str,
}

/// Run Site #3 type-registry post-hoc collapse over `group_id` (ADR-063 spec §4).
///
/// Called by `mem.dream()` when `DreamOpts::include_type_registry_collapse =
/// true` (spike-gated, default `false` — spec §8). Hooked LAST in the pass
/// chain, after `canonicalize_surface_forms` (spec §4.0 rationale: type
/// collapse benefits from a stable entity population that Pass 0/2/4/L5 have
/// already finished touching this cycle).
///
/// `pub` + `#[doc(hidden)]` — see [`TypeRegistryCollapseParams`]'s doc comment
/// for the MNT-002 re-export rationale (S3 spike integration test).
#[doc(hidden)]
pub async fn type_registry_collapse<L: ChatProvider>(
    llm: &L,
    params: TypeRegistryCollapseParams<'_>,
) -> Result<TypeRegistryCollapseReport> {
    let TypeRegistryCollapseParams {
        conn,
        group_id,
        embedder,
        model_id,
    } = params;

    let mut report = TypeRegistryCollapseReport {
        group_id: group_id.to_string(),
        ..Default::default()
    };

    // ── Step 1/2: load types + embed DESCRIPTIONS (spec §4.0/§4.1) ───────────
    let Some(embedder) = embedder else {
        // Degraded mode (D7 precedent, discover_types.rs): no embedder → the
        // description-cosine signal (spec §4.1's PRIMARY signal) cannot be
        // computed at all. Skip the pass entirely rather than silently
        // operating on a fabricated cosine.
        tracing::warn!(
            target: "kremory::dream::type_registry_collapse",
            group_id = %group_id,
            "type_registry_collapse: no embedder configured — pass skipped"
        );
        counter!(
            "kremory.dream.anti_redundancy_gate_skipped_total",
            "reason" => "no_embedder",
            "site" => SITE_LABEL
        )
        .increment(1);
        return Ok(report);
    };

    let specs = load_entity_types_excluding_catch_all(conn, group_id).await?;
    if specs.len() < 2 {
        return Ok(report);
    }

    let mut slots: Vec<TypeSlot> = Vec::with_capacity(specs.len());
    for (spec, evidence_count) in specs {
        let desc_embedding = embedder.embed_dyn(&spec.description).await?;
        slots.push(TypeSlot {
            spec,
            desc_embedding,
            evidence_count,
        });
    }

    // ── Step 3/4: pairwise cosine + lexical pre-filter → band (spec §4.2/§4.3) ─
    let mut nominated: Vec<NominatedPair> = Vec::new();
    // Clear-case auto-merge pairs (row 1 of write_gate — no LLM call needed).
    let mut auto_merge_pairs: Vec<(usize, usize, f32)> = Vec::new();

    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            report.pairs_examined += 1;
            let cos = crate::core::dream::anti_redundancy::cosine(
                &slots[i].desc_embedding,
                &slots[j].desc_embedding,
            );
            let lexical_compatible =
                names_share_lemma_or_exact(&slots[i].spec.name, &slots[j].spec.name);
            if lexical_compatible {
                report.lexical_prefilter_hits += 1;
            }

            if cos >= TYPE_COLLAPSE_PRIMARY_COSINE && lexical_compatible {
                // Row: cosine ≥ 0.85 AND shared lemma/exact → auto-merge, no LLM.
                auto_merge_pairs.push((i, j, cos));
            } else if cos >= TYPE_COLLAPSE_PRIMARY_COSINE {
                // Row: cosine ≥ 0.85, zero lemma overlap → LLM-verify band.
                nominated.push(NominatedPair {
                    a_idx: i,
                    b_idx: j,
                    cosine: cos,
                    lexical_compatible,
                });
            } else if cos >= TYPE_COLLAPSE_LOWER_BAND_COSINE {
                // Row: 0.70-0.85 (either lexical state) → LLM-verify band.
                nominated.push(NominatedPair {
                    a_idx: i,
                    b_idx: j,
                    cosine: cos,
                    lexical_compatible,
                });
            }
            // cos < 0.70 → no candidate, nothing written (spec §4.3 row 4).
        }
    }

    counter!("kremory.dream.type_registry_collapse.pairs_examined_total")
        .increment(report.pairs_examined as u64);
    counter!("kremory.dream.type_registry_collapse.lexical_prefilter_hit_total")
        .increment(report.lexical_prefilter_hits as u64);

    report.candidates_nominated = nominated.len();
    for _ in &nominated {
        counter!(
            "kremory.identity.candidate_nominated_total",
            "site" => SITE_LABEL
        )
        .increment(1);
    }

    // ── Step 5: LLM-verify band adjudication (spec §4.4) ─────────────────────
    let verdicts_by_pair_id: HashMap<usize, IdentityVerdictItem> = if nominated.is_empty() {
        HashMap::new()
    } else {
        adjudicate_batch(AdjudicateBatchParams {
            llm,
            model_id,
            slots: &slots,
            nominated: &nominated,
            group_id,
        })
        .await?
    };

    // ── Step 6: write_gate decision per pair + atomic writes (spec §2.2/§4.5) ─
    let run_id = uuid::Uuid::new_v4().to_string();

    // loser_id → MergeEdge (keeper + merge provenance), resolved by
    // longest-description-wins (+ evidence_count tiebreak, spec §4.5).
    // Keyed on the type's integer id within this group_id namespace.
    let mut loser_to_keeper: HashMap<u32, MergeEdge> = HashMap::new();
    // id -> slot index, for keeper-selection lookups after transitive resolution.
    let id_to_idx: HashMap<u32, usize> = slots
        .iter()
        .enumerate()
        .map(|(idx, s)| (s.spec.id, idx))
        .collect();

    // Auto-merge pairs (write_gate row 1 — no LLM call → verdict = None → NO
    // audit row per spec §5.2's "audit only LLM-touched decisions").
    for (i, j, cos) in &auto_merge_pairs {
        let decision = write_gate(WriteGateInputs {
            cosine: *cos,
            merge_threshold: TYPE_COLLAPSE_PRIMARY_COSINE,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: None,
            min_confidence_floor: None,
        });
        record_write_gate_decision(decision);
        if decision == WriteDecision::Merge {
            let (keeper_i, loser_i) = resolve_keeper_pair(&slots, *i, *j);
            queue_merge(QueueMergeParams {
                loser_to_keeper: &mut loser_to_keeper,
                slots: &slots,
                keeper_idx: keeper_i,
                loser_idx: loser_i,
                cosine: Some(*cos),
                verdict: None,
            });
        }
    }

    // LLM-verify band pairs.
    for (pair_id, pair) in nominated.iter().enumerate() {
        let verdict = verdicts_by_pair_id.get(&pair_id).cloned();
        let decision = write_gate(WriteGateInputs {
            cosine: pair.cosine,
            merge_threshold: TYPE_COLLAPSE_PRIMARY_COSINE,
            deterministic_signal: DeterministicSignal::from_lexical(pair.lexical_compatible),
            llm_verdict: verdict.clone(),
            min_confidence_floor: None,
        });
        record_write_gate_decision(decision);

        match decision {
            WriteDecision::Merge => {
                // The merge audit row (with real cosine + verdict) is written
                // INSIDE apply_type_merge's transaction (spec §5.1 RISK-003).
                // Carry the pair's cosine + winning verdict through the
                // MergeEdge so the final keeper's audit row reflects the pair
                // that actually authorised the merge.
                let (keeper_i, loser_i) = resolve_keeper_pair(&slots, pair.a_idx, pair.b_idx);
                queue_merge(QueueMergeParams {
                    loser_to_keeper: &mut loser_to_keeper,
                    slots: &slots,
                    keeper_idx: keeper_i,
                    loser_idx: loser_i,
                    cosine: Some(pair.cosine),
                    verdict: verdict.clone(),
                });
            }
            WriteDecision::PotentialAlias => {
                // Types have no potential-alias edge concept (spec brief) — this
                // cycle defers the merge, audit-only, no destructive write.
                write_audit_row(WriteAuditRowParams {
                    conn,
                    group_id,
                    candidate_a: &slots[pair.a_idx].spec.name,
                    candidate_b: &slots[pair.b_idx].spec.name,
                    cosine: Some(pair.cosine),
                    structural_signal: pair.lexical_compatible,
                    verdict: verdict.as_ref(),
                    decision: "potential_alias",
                    run_id: &run_id,
                })
                .await?;
            }
            WriteDecision::Reject => {
                // No write per spec §2.2 — Reject is silent (no audit row for
                // clear-case rejects, mirrors ADR-047's dream_pass4_audit
                // convention of auditing only LLM-touched decisions, spec §5.2).
            }
        }
    }

    // ── Resolve transitive chains + apply atomic remaps (spec §4.5) ──────────
    let losers: Vec<u32> = loser_to_keeper.keys().copied().collect();
    for loser_id in losers {
        let keeper_id = resolve_keeper_transitive(loser_id, &loser_to_keeper);
        if keeper_id == loser_id {
            continue; // degenerate self-merge guard
        }
        let Some(&loser_idx) = id_to_idx.get(&loser_id) else {
            continue;
        };
        let Some(&keeper_idx) = id_to_idx.get(&keeper_id) else {
            continue;
        };
        // The audit provenance travels on the DIRECT edge from this loser
        // (its own cosine + verdict), even when the final keeper is reached
        // transitively — the row documents the adjudication that removed THIS
        // loser type.
        let (edge_cosine, edge_verdict) = loser_to_keeper
            .get(&loser_id)
            .map(|e| (e.cosine, e.verdict.clone()))
            .unwrap_or((None, None));
        apply_type_merge(ApplyTypeMergeParams {
            conn,
            group_id,
            loser_id,
            keeper_id,
            loser_name: &slots[loser_idx].spec.name,
            keeper_name: &slots[keeper_idx].spec.name,
            cosine: edge_cosine,
            verdict: edge_verdict,
            run_id: &run_id,
        })
        .await?;
        report.merges_applied += 1;
    }

    counter!("kremory.dream.type_registry_collapse.merges_applied_total")
        .increment(report.merges_applied as u64);

    Ok(report)
}

fn record_write_gate_decision(decision: WriteDecision) {
    let label = match decision {
        WriteDecision::Merge => "merge",
        WriteDecision::PotentialAlias => "potential_alias",
        WriteDecision::Reject => "reject",
    };
    counter!(
        "kremory.identity.write_gate_decision_total",
        "site" => SITE_LABEL,
        "decision" => label
    )
    .increment(1);
    if decision == WriteDecision::Merge {
        counter!(
            "kremory.identity.write_gate_llm_authorized_merge_total",
            "site" => SITE_LABEL
        )
        .increment(1);
    }
}

/// Resolve which of `(i, j)` is keeper vs loser per spec §4.5: longest
/// description wins (PRIMARY); on near-equal length, the HIGHER
/// `evidence_count` wins (SECONDARY tiebreak); a true evidence_count tie
/// resolves to the first index (`i`) deterministically. Returns
/// `(keeper_idx, loser_idx)`.
fn resolve_keeper_pair(slots: &[TypeSlot], i: usize, j: usize) -> (usize, usize) {
    let len_a = slots[i].spec.description.len();
    let len_b = slots[j].spec.description.len();
    // "Near-equal" epsilon: descriptions within 5 chars of each other are
    // treated as a tie, falling through to the evidence_count tiebreak (spec §4.5).
    const EPSILON: usize = 5;
    if len_a.abs_diff(len_b) <= EPSILON {
        // Secondary tiebreak: the type more entities were assigned to
        // (`evidence_count`) is the more-established canonical type — analogous
        // to LightRAG's frequency-vote pattern, applied here as a tiebreaker
        // only (spec §4.5 / R2 §8.3's ruling-out of Counter-as-primary).
        let ev_a = slots[i].evidence_count;
        let ev_b = slots[j].evidence_count;
        if ev_a >= ev_b {
            (i, j)
        } else {
            (j, i)
        }
    } else if len_a > len_b {
        (i, j)
    } else {
        (j, i)
    }
}

/// Record `(loser -> keeper)` in the merge map as a [`MergeEdge`], keeping the
/// keeper with the longest description seen so far for this loser across all
/// pairs it appears in — mirrors `canonicalization.rs::canonicalize_surface_forms`'s
/// `loser_to_keeper` construction (attribution: duplicated, id-space-adapted
/// copy per spec §4.5's explicit "duplication with attribution is acceptable"
/// ruling — entity ids are normalized name strings, type ids are
/// `(group_id, u32)` composite keys, so a shared generic abstraction is not
/// attempted here).
///
/// When a longer-description keeper displaces the previously-stored one, the
/// merge provenance (`cosine` + `verdict`) is ALSO switched to that winning
/// pair's, so the audit row reflects the pair that actually authorised the
/// surviving keeper (spec §5.1).
/// Bundled parameters for [`queue_merge`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments, threshold 3).
struct QueueMergeParams<'a> {
    loser_to_keeper: &'a mut HashMap<u32, MergeEdge>,
    slots: &'a [TypeSlot],
    keeper_idx: usize,
    loser_idx: usize,
    cosine: Option<f32>,
    verdict: Option<IdentityVerdictItem>,
}

fn queue_merge(params: QueueMergeParams<'_>) {
    let QueueMergeParams {
        loser_to_keeper,
        slots,
        keeper_idx,
        loser_idx,
        cosine,
        verdict,
    } = params;
    let keeper_id = slots[keeper_idx].spec.id;
    let loser_id = slots[loser_idx].spec.id;
    if keeper_id == loser_id {
        return;
    }
    let keeper_len = slots
        .iter()
        .find(|s| s.spec.id == keeper_id)
        .map(|s| s.spec.description.len())
        .unwrap_or(0);
    loser_to_keeper
        .entry(loser_id)
        .and_modify(|existing| {
            let existing_len = slots
                .iter()
                .find(|s| s.spec.id == existing.keeper_id)
                .map(|s| s.spec.description.len())
                .unwrap_or(0);
            if keeper_len > existing_len {
                existing.keeper_id = keeper_id;
                existing.cosine = cosine;
                existing.verdict = verdict.clone();
            }
        })
        .or_insert(MergeEdge {
            keeper_id,
            cosine,
            verdict,
        });
}

/// Bounded-hop transitive-chain resolution — mirrors
/// `canonicalization.rs::resolve_keeper` exactly (spec §4.5's explicit
/// "duplication with attribution" ruling; type ids are `u32` here vs
/// entity ids being `String`s there, so the shapes differ enough that a
/// shared generic helper is not attempted).
fn resolve_keeper_transitive(loser: u32, map: &HashMap<u32, MergeEdge>) -> u32 {
    let mut current = loser;
    for _ in 0..=map.len() {
        if let Some(edge) = map.get(&current) {
            if edge.keeper_id == current {
                break;
            }
            current = edge.keeper_id;
        } else {
            break;
        }
    }
    current
}

// ─── Lexical pre-filter (spec §4.2) ───────────────────────────────────────────

/// Case-insensitive-normalized exact match OR a naive singular/plural lemma
/// match (strip a trailing `s`) between two type names (spec §4.2).
///
/// The lemma heuristic is SPIKE-GATED (S3, spec §8) — if S3's fixture shows
/// false positives, the fallback is exact-normalized-match only. Both checks
/// are cheap and dictionary-free, mirroring `names_lexically_compatible`'s
/// reuse of `normalize_name`.
fn names_share_lemma_or_exact(a: &str, b: &str) -> bool {
    let norm_a = crate::core::resolver::normalize_name(a);
    let norm_b = crate::core::resolver::normalize_name(b);
    if norm_a == norm_b {
        return true;
    }
    strip_trailing_s(&norm_a) == strip_trailing_s(&norm_b)
}

/// Strip a single trailing `s` (naive singular/plural lemma heuristic, spec §4.2).
fn strip_trailing_s(s: &str) -> &str {
    s.strip_suffix('s').unwrap_or(s)
}

// ─── LLM adjudication (spec §4.4) ─────────────────────────────────────────────

struct AdjudicateBatchParams<'a, L: ChatProvider> {
    llm: &'a L,
    model_id: &'a str,
    slots: &'a [TypeSlot],
    nominated: &'a [NominatedPair],
    group_id: &'a str,
}

/// Run ONE batched `IdentityVerdictBatch` LLM call adjudicating every
/// nominated pair (spec §4.4/§3.2 batch shape), returning verdicts keyed by
/// `pair_id` (index into `nominated`). Missing/parse-failed entries are
/// simply absent from the map — callers treat a missing `pair_id` as
/// `llm_verdict = None` (spec §2.3 failure-mode default).
async fn adjudicate_batch<L: ChatProvider>(
    params: AdjudicateBatchParams<'_, L>,
) -> Result<HashMap<usize, IdentityVerdictItem>> {
    let AdjudicateBatchParams {
        llm,
        model_id,
        slots,
        nominated,
        group_id,
    } = params;

    let messages = build_adjudication_messages(slots, nominated);
    let schema = identity_verdict_batch_schema(nominated.len());

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
            target: "kremory::dream::type_registry_collapse::raw_payload",
            model_id = %model_id,
            group_id = %group_id,
            response = ?raw_value,
            "type_registry_collapse adjudication raw response"
        );
    }

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "kremory::dream::type_registry_collapse",
                error = %e,
                group_id = %group_id,
                "type_registry_collapse: adjudication LLM call failed — all nominated pairs default to no-verdict"
            );
            return Ok(HashMap::new());
        }
    };

    // Batch envelope parse — deliberately loose (Vec<serde_json::Value>), then
    // per-item parse below (spec §2.1's "one malformed element ≠ whole batch loss").
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
                        target: "kremory::dream::type_registry_collapse",
                        error = %e,
                        group_id = %group_id,
                        "type_registry_collapse: failed to parse IdentityVerdictBatch — all nominated pairs default to no-verdict"
                    );
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(nominated.len() as u64);
                    return Ok(HashMap::new());
                }
            }
        }
    };

    let mut verdicts_by_pair_id: HashMap<usize, IdentityVerdictItem> = HashMap::new();
    for raw_item in &batch.verdicts {
        match serde_json::from_value::<IdentityVerdictItem>(raw_item.clone()) {
            Ok(item) => {
                if item.pair_id >= nominated.len() {
                    // Out-of-range pair_id — echoed id doesn't correlate to any
                    // nominated pair; drop it loudly (Vera Cycle 2 OBS-01).
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(1);
                    tracing::warn!(
                        target: "kremory::dream::type_registry_collapse",
                        pair_id = item.pair_id,
                        nominated_len = nominated.len(),
                        "type_registry_collapse: verdict pair_id out of range — dropped"
                    );
                    continue;
                }
                verdicts_by_pair_id.entry(item.pair_id).or_insert(item);
            }
            Err(e) => {
                counter!(
                    "kremory.identity.verdict_parse_fail_total",
                    "site" => SITE_LABEL
                )
                .increment(1);
                tracing::warn!(
                    target: "kremory::dream::type_registry_collapse",
                    error = %e,
                    "type_registry_collapse: skipping malformed verdict item"
                );
            }
        }
    }

    Ok(verdicts_by_pair_id)
}

/// Build the LLM adjudication prompt for a batch of nominated type pairs
/// (spec §4.4): both type names, both descriptions, and each type's
/// `evidence_count` as advisory context.
fn build_adjudication_messages(
    slots: &[TypeSlot],
    nominated: &[NominatedPair],
) -> Vec<crate::core::provider::ChatMessage> {
    let system = "You are a knowledge-graph type registry analyst. You will be shown \
pairs of entity-type definitions (name + description) that a similarity gate has \
flagged as POSSIBLY the same underlying type. For each pair, decide whether the two \
type definitions describe the SAME semantic category of entity (e.g. 'Company' and \
'Business Organisation' are the same; 'Person' and 'Organisation' are NOT). \
Respond ONLY with the JSON structure — no extra commentary."
        .to_string();

    let pairs_list = nominated
        .iter()
        .enumerate()
        .map(|(pair_id, pair)| {
            let slot_a = &slots[pair.a_idx];
            let slot_b = &slots[pair.b_idx];
            format!(
                "Pair {pair_id}:\n  A: name=\"{}\" description=\"{}\" evidence_count={}\n  B: name=\"{}\" description=\"{}\" evidence_count={}",
                slot_a.spec.name, slot_a.spec.description, slot_a.evidence_count,
                slot_b.spec.name, slot_b.spec.description, slot_b.evidence_count,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let user = format!(
        "Adjudicate the following type-definition pairs. For each pair, output a \
verdict object with `pair_id` (matching the pair number below), `is_same_entity` \
(true if the two definitions describe the same type), `confidence` (0.0-1.0), and \
`reasoning` (a short free-text justification).\n\n{pairs_list}"
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

// ─── DB helpers ────────────────────────────────────────────────────────────

/// Load `entity_types` for `group_id`, EXCLUDING id=0 (catch-all — never a
/// merge candidate per spec §4.0). Returns each spec paired with its
/// `entity_types.evidence_count` (Migration 014) — needed as advisory LLM
/// context (spec §4.4) and the secondary keeper tiebreak (spec §4.5).
///
/// `evidence_count` is `NOT NULL DEFAULT 0` in Migration 014, but defensively
/// read as nullable and coalesced to 0 to tolerate any pre-provenance rows.
async fn load_entity_types_excluding_catch_all(
    conn: &libsql::Connection,
    group_id: &str,
) -> Result<Vec<(EntityTypeSpec, i64)>> {
    let mut rows = conn
        .query(
            "SELECT id, name, description, COALESCE(evidence_count, 0) FROM entity_types \
             WHERE group_id = ?1 AND id != 0 ORDER BY id ASC",
            libsql::params![group_id],
        )
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "type_registry_collapse: load entity_types failed for group_id={group_id}: {e}"
            ))
        })?;

    let mut specs = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "type_registry_collapse: entity_types row read failed: {e}"
        ))
    })? {
        let id: i64 = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_registry_collapse: id read: {e}")))?;
        if id < 0 {
            continue;
        }
        let name: String = row
            .get(1)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_registry_collapse: name read: {e}")))?;
        let description: String = row.get(2).map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "type_registry_collapse: description read: {e}"
            ))
        })?;
        let evidence_count: i64 = row.get(3).map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "type_registry_collapse: evidence_count read: {e}"
            ))
        })?;
        specs.push((
            EntityTypeSpec {
                id: id as u32,
                name,
                description,
            },
            evidence_count,
        ));
    }
    Ok(specs)
}

struct WriteAuditRowParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    candidate_a: &'a str,
    candidate_b: &'a str,
    cosine: Option<f32>,
    structural_signal: bool,
    verdict: Option<&'a IdentityVerdictItem>,
    decision: &'a str,
    run_id: &'a str,
}

/// Insert one `identity_verdict_audit` row (spec §5.1). Used ONLY for
/// `PotentialAlias` decisions (audit-only, no destructive write) OUTSIDE any
/// merge transaction. Merge audit rows are written INSIDE `apply_type_merge`'s
/// `BEGIN IMMEDIATE` (spec §5.1 RISK-003), never here.
async fn write_audit_row(params: WriteAuditRowParams<'_>) -> Result<()> {
    let WriteAuditRowParams {
        conn,
        group_id,
        candidate_a,
        candidate_b,
        cosine,
        structural_signal,
        verdict,
        decision,
        run_id,
    } = params;

    conn.execute(
        "INSERT INTO identity_verdict_audit \
         (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
          llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        libsql::params![
            SITE_LABEL,
            group_id,
            candidate_a,
            candidate_b,
            cosine.map(f64::from),
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
        Error::Other(anyhow::anyhow!(
            "type_registry_collapse: identity_verdict_audit insert failed: {e}"
        ))
    })?;
    Ok(())
}

struct ApplyTypeMergeParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    loser_id: u32,
    keeper_id: u32,
    loser_name: &'a str,
    keeper_name: &'a str,
    /// Description-cosine of the adjudicating pair (spec §5.1 audit fidelity).
    /// `None` only if provenance was lost (defensive); LLM-band merges always
    /// carry it.
    cosine: Option<f32>,
    /// The winning `IdentityVerdictItem` for LLM-verify-band merges. `None` for
    /// clear-case auto-merges (write_gate row 1, no LLM) — in which case NO
    /// audit row is written (spec §5.2: audit only LLM-touched decisions).
    verdict: Option<IdentityVerdictItem>,
    run_id: &'a str,
}

/// Atomic remap of `entities.entity_type_id` from `loser_id` to `keeper_id`
/// and deletion of the loser `entity_types` row — ALWAYS. PLUS, for
/// LLM-adjudicated merges only (`verdict.is_some()`), an `identity_verdict_audit`
/// INSERT with the REAL cosine + verdict values — ALL inside ONE
/// `BEGIN IMMEDIATE` transaction (spec §4.5/§5.1 RISK-003). Clear-case
/// auto-merges (`verdict.is_none()`) write NO audit row (spec §5.2 — audit
/// only LLM-touched decisions). Mirrors `canonicalization.rs::apply_merge`'s
/// transactional discipline, adapted for the `entity_types` id-space.
async fn apply_type_merge(params: ApplyTypeMergeParams<'_>) -> Result<()> {
    let ApplyTypeMergeParams {
        conn,
        group_id,
        loser_id,
        keeper_id,
        loser_name,
        keeper_name,
        cosine,
        verdict,
        run_id,
    } = params;

    conn.execute("BEGIN IMMEDIATE", ()).await.map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "type_registry_collapse: BEGIN IMMEDIATE failed: {e}"
        ))
    })?;

    let result: Result<()> = async {
        // 1. Remap entities.entity_type_id from loser -> keeper (ALWAYS).
        conn.execute(
            "UPDATE entities SET entity_type_id = ?1 \
             WHERE entity_type_id = ?2 AND group_id = ?3",
            libsql::params![keeper_id as i64, loser_id as i64, group_id],
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("entities remap failed: {e}")))?;

        // 2. Delete the loser row from entity_types (ALWAYS).
        conn.execute(
            "DELETE FROM entity_types WHERE group_id = ?1 AND id = ?2",
            libsql::params![group_id, loser_id as i64],
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("entity_types delete failed: {e}")))?;

        // 3. Audit row — INSIDE this same transaction (spec §5.1 RISK-003),
        //    but ONLY for LLM-touched (verdict-carrying) merges (spec §5.2).
        //    Real cosine + verdict values captured; `structural_signal` = true
        //    (a Merge always required a deterministic signal per write_gate
        //    rows 5/6).
        if let Some(v) = verdict.as_ref() {
            conn.execute(
                "INSERT INTO identity_verdict_audit \
                 (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
                  llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'merge', ?10)",
                libsql::params![
                    SITE_LABEL,
                    group_id,
                    loser_name,
                    keeper_name,
                    cosine.map(f64::from),
                    true,
                    v.is_same_entity,
                    f64::from(v.confidence),
                    v.reasoning.clone(),
                    run_id,
                ],
            )
            .await
            .map_err(|e| {
                Error::Other(anyhow::anyhow!("identity_verdict_audit insert failed: {e}"))
            })?;
        }

        Ok(())
    }
    .await;

    match result {
        Ok(()) => {
            conn.execute("COMMIT", ()).await.map_err(|e| {
                Error::Other(anyhow::anyhow!(
                    "type_registry_collapse: COMMIT failed: {e}"
                ))
            })?;
            tracing::info!(
                target: "kremory::dream::type_registry_collapse",
                group_id = %group_id,
                loser_id,
                keeper_id,
                "type_registry_collapse: merge applied"
            );
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(e)
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
//
// Deterministic — a scripted `ChatProvider` mirrors the
// `discover_types.rs::td050_full_workflow_tests` pattern exactly (no live LLM,
// runs in the default `cargo test` gate). A `MockEmbeddingProvider` supplies
// deterministic description embeddings so cosine values are fully controlled.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entity_types::ensure_default_types_seeded;
    use crate::core::provider::{
        ChatMessage, ChatResponse, DynEmbeddingProvider, LLMError, MockChatResponse,
        StructuredOutputFormat, Tool,
    };
    use crate::core::schema::TemporalGraph;
    use chrono::Utc;
    use std::collections::HashMap as StdHashMap;

    /// Scripted `ChatProvider` returning one fixed `IdentityVerdictBatch` JSON
    /// response, ignoring the prompt entirely (mirrors
    /// `discover_types.rs::ScriptedProposalProvider`).
    #[derive(Debug)]
    struct ScriptedVerdictProvider {
        json: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedVerdictProvider {
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

    /// Deterministic embedder: returns a pre-registered vector per input text
    /// (looked up by exact string match), or a zero vector for unknown text
    /// (cosine 0.0 against anything — safe default for unrelated descriptions).
    #[derive(Debug, Clone)]
    struct MockEmbeddingProvider {
        vectors: StdHashMap<String, Vec<f32>>,
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

    /// Unit-normalised vector helper (dot product == cosine for normalised inputs,
    /// matching `anti_redundancy::cosine`'s documented assumption).
    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    fn unit_vec4(a: f32, b: f32, c: f32, d: f32) -> Vec<f32> {
        let norm = (a * a + b * b + c * c + d * d).sqrt();
        if norm == 0.0 {
            return vec![0.0, 0.0, 0.0, 0.0];
        }
        vec![a / norm, b / norm, c / norm, d / norm]
    }

    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn seed_two_types(
        conn: &libsql::Connection,
        group_id: &str,
        name_a: &str,
        desc_a: &str,
        name_b: &str,
        desc_b: &str,
    ) {
        ensure_default_types_seeded(conn, group_id)
            .await
            .expect("seed defaults 0..=10");
        conn.execute(
            "INSERT INTO entity_types (group_id, id, name, description) VALUES (?1, 11, ?2, ?3)",
            libsql::params![group_id, name_a, desc_a],
        )
        .await
        .expect("insert type A");
        conn.execute(
            "INSERT INTO entity_types (group_id, id, name, description) VALUES (?1, 12, ?2, ?3)",
            libsql::params![group_id, name_b, desc_b],
        )
        .await
        .expect("insert type B");
    }

    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn insert_entity_of_type(
        conn: &libsql::Connection,
        group_id: &str,
        id: &str,
        type_id: i64,
    ) {
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, entity_type_id, recorded_at, group_id) VALUES (?1, ?2, ?3, ?4)",
            libsql::params![id, type_id, now, group_id],
        )
        .await
        .expect("insert entity");
    }

    async fn count_entity_types(conn: &libsql::Connection, group_id: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1",
                libsql::params![group_id],
            )
            .await
            .expect("count query");
        rows.next()
            .await
            .expect("row")
            .expect("row present")
            .get::<i64>(0)
            .expect("count col")
    }

    /// (a) Two type names that normalize-equal (exact match) with high
    /// desc-cosine → AUTO-merge (write_gate row 1, no LLM call). Entities
    /// pointing at the loser are repointed to the keeper; loser row deleted.
    /// Because no LLM adjudicated this clear case, NO `identity_verdict_audit`
    /// row is written (spec §5.2 — audit only LLM-touched decisions).
    #[tokio::test]
    async fn exact_lexical_match_high_cosine_auto_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        // "Company" / "company" normalize-equal (case-insensitive exact match).
        seed_two_types(
            &conn,
            "g1",
            "Company",
            "A business organisation, firm, or investment fund.",
            "company",
            "A business organisation, firm, or investment fund.",
        )
        .await;
        insert_entity_of_type(&conn, "g1", "acme corp", 11).await;

        let v = unit_vec4(1.0, 0.0, 0.0, 0.0);
        let mut vectors = StdHashMap::new();
        vectors.insert(
            "A business organisation, firm, or investment fund.".to_string(),
            v,
        );
        let embedder = MockEmbeddingProvider { vectors };

        // No LLM call expected on this path — scripted provider would error if
        // invoked with an empty batch; use an empty-verdicts response as a safe
        // sentinel (never reached for the auto-merge row).
        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[]}"#.to_string(),
        };

        let report = type_registry_collapse(
            &llm,
            TypeRegistryCollapseParams {
                conn: &conn,
                group_id: "g1",
                embedder: Some(&embedder),
                model_id: "test-model",
            },
        )
        .await
        .expect("type_registry_collapse must succeed");

        assert_eq!(
            report.merges_applied, 1,
            "exact-lexical high-cosine pair auto-merges"
        );
        assert_eq!(
            count_entity_types(&conn, "g1").await,
            12,
            "11 seeded (ids 0..=10) + 1 surviving custom type after the merge"
        );

        // Entity remapped onto whichever id survived (keeper).
        let mut rows = conn
            .query(
                "SELECT entity_type_id FROM entities WHERE id = 'acme corp'",
                (),
            )
            .await
            .expect("query entity");
        let type_id: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("row present")
            .get(0)
            .expect("entity_type_id col");
        assert!(
            type_id == 11 || type_id == 12,
            "entity must be remapped to the surviving keeper id, got {type_id}"
        );

        // NO audit row: this is a clear-case AUTO-merge (write_gate row 1, no
        // LLM adjudication), which is NOT audited per spec §5.2.
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM identity_verdict_audit WHERE group_id = 'g1'",
                (),
            )
            .await
            .expect("audit query");
        let audit_count: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("row present")
            .get(0)
            .expect("count col");
        assert_eq!(
            audit_count, 0,
            "an auto-merge (no LLM) must write NO audit row (spec §5.2)"
        );
    }

    /// (b) LLM-verify band pair with ZERO lexical overlap ("Human"/"Individual"),
    /// LLM says `is_same_entity=true` at high confidence → NOT a merge but a
    /// `PotentialAlias` (write_gate ROW 6, the defense-in-depth invariant: cosine +
    /// one LLM verdict never authorise a destructive type merge WITHOUT a
    /// deterministic lexical signal — spec §2.2 row 6). Both types survive; one
    /// `potential_alias` audit row records the LLM-touched verdict (spec §5.2).
    #[tokio::test]
    async fn llm_verify_zero_lexical_true_verdict_is_potential_alias_row6() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        // "Human" / "Individual" share zero lemma overlap but are near-duplicate
        // in meaning — the exact case spec §4.3 routes to the LLM-verify band.
        seed_two_types(
            &conn,
            "g2",
            "Human",
            "A living person, described by name and biography.",
            "Individual",
            "A single living person, described by name and biography facts.",
        )
        .await;

        // Cosine 1.0 (identical vectors) — routes to the >=0.85 zero-lemma LLM-verify row.
        let v = unit_vec4(0.0, 1.0, 0.0, 0.0);
        let mut vectors = StdHashMap::new();
        vectors.insert(
            "A living person, described by name and biography.".to_string(),
            v.clone(),
        );
        vectors.insert(
            "A single living person, described by name and biography facts.".to_string(),
            v,
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.95,"reasoning":"same concept"}]}"#
                .to_string(),
        };

        let report = type_registry_collapse(
            &llm,
            TypeRegistryCollapseParams {
                conn: &conn,
                group_id: "g2",
                embedder: Some(&embedder),
                model_id: "test-model",
            },
        )
        .await
        .expect("type_registry_collapse must succeed");

        assert_eq!(
            report.candidates_nominated, 1,
            "one pair nominated for LLM adjudication"
        );
        assert_eq!(
            report.merges_applied, 0,
            "zero-lexical + LLM-true is PotentialAlias, never Merge (write_gate row 6 — \
             cosine + LLM alone cannot authorise a destructive type merge)"
        );
        assert_eq!(
            count_entity_types(&conn, "g2").await,
            13,
            "both types survive — row 6 defers to PotentialAlias, no destructive merge"
        );

        // Exactly ONE potential_alias audit row (LLM-touched → carries the real
        // verdict; spec §5.2). A merge would be decision='merge' — this is
        // decision='potential_alias', proving row 6 did NOT authorise a merge.
        let mut rows = conn
            .query(
                "SELECT llm_is_same, llm_confidence, decision \
                 FROM identity_verdict_audit WHERE group_id = 'g2'",
                (),
            )
            .await
            .expect("audit query");
        let row = rows
            .next()
            .await
            .expect("row")
            .expect("exactly one potential_alias audit row");
        let llm_is_same: bool = row.get(0).expect("llm_is_same col");
        let llm_confidence: f64 = row.get(1).expect("llm_confidence col");
        let decision: String = row.get(2).expect("decision col");
        assert!(llm_is_same, "the LLM's true verdict is recorded");
        assert!(
            (llm_confidence - 0.95).abs() < 1e-6,
            "real confidence recorded, got {llm_confidence}"
        );
        assert_eq!(
            decision, "potential_alias",
            "row 6 → potential_alias, not merge"
        );
        assert!(
            rows.next().await.expect("row iter").is_none(),
            "exactly one audit row — no extras"
        );
    }

    /// (b') LLM-verify band MERGE via write_gate ROW 5 — the ONLY path an LLM-band
    /// pair reaches a destructive merge: a pair in the 0.70–0.85 cosine band (below
    /// the auto-merge threshold, so an LLM call IS made) WITH a lexical lemma match
    /// ("Organization"/"Organizations", singular/plural), LLM `is_same_entity=true`
    /// high confidence → Merge (deterministic lexical signal AND the LLM both agree,
    /// spec §2.2 row 5). Writes one `merge` audit row with the real verdict + cosine
    /// inside the merge transaction (spec §5.1).
    #[tokio::test]
    async fn llm_verify_band_lexical_match_true_verdict_merges_row5() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_two_types(
            &conn,
            "g5",
            "Organization",
            "A group of people with a shared purpose or structure.",
            "Organizations",
            "A collective body of people organised for a common purpose.",
        )
        .await;

        // Cosine 0.80 (in [0.70, 0.85) → LLM-verify band, NOT auto-merge):
        // A=[1,0,0,0], B=[0.8,0.6,0,0] (unit-norm) → dot = 0.80. Names
        // "Organization"/"Organizations" strip-trailing-s to the same lemma → the
        // deterministic signal fires (from_lexical(true)), enabling row 5.
        let mut vectors = StdHashMap::new();
        vectors.insert(
            "A group of people with a shared purpose or structure.".to_string(),
            unit_vec4(1.0, 0.0, 0.0, 0.0),
        );
        vectors.insert(
            "A collective body of people organised for a common purpose.".to_string(),
            unit_vec4(0.8, 0.6, 0.0, 0.0),
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.9,"reasoning":"singular/plural of one type"}]}"#
                .to_string(),
        };

        let report = type_registry_collapse(
            &llm,
            TypeRegistryCollapseParams {
                conn: &conn,
                group_id: "g5",
                embedder: Some(&embedder),
                model_id: "test-model",
            },
        )
        .await
        .expect("type_registry_collapse must succeed");

        assert_eq!(
            report.candidates_nominated, 1,
            "mid-band lexical pair is nominated (cosine below the auto-merge threshold)"
        );
        assert_eq!(
            report.merges_applied, 1,
            "LLM-band lexical-match + true verdict merges (write_gate row 5)"
        );
        assert_eq!(
            count_entity_types(&conn, "g5").await,
            12,
            "11 seeded (ids 0..=10) + 1 surviving custom type after the row-5 merge"
        );

        // One merge audit row carrying the REAL cosine + verdict (spec §5.1).
        let mut rows = conn
            .query(
                "SELECT cosine, llm_is_same, llm_confidence, decision \
                 FROM identity_verdict_audit WHERE group_id = 'g5'",
                (),
            )
            .await
            .expect("audit query");
        let row = rows
            .next()
            .await
            .expect("row")
            .expect("exactly one merge audit row");
        let cosine: f64 = row.get(0).expect("cosine col");
        let llm_is_same: bool = row.get(1).expect("llm_is_same col");
        let decision: String = row.get(3).expect("decision col");
        assert!(
            (cosine - 0.80).abs() < 0.02,
            "audit cosine reflects the real mid-band value, got {cosine}"
        );
        assert!(llm_is_same, "the true verdict is recorded on the merge row");
        assert_eq!(decision, "merge");
        assert!(
            rows.next().await.expect("row iter").is_none(),
            "exactly one audit row — no extras"
        );
    }

    /// (c) LLM-verify band pair where the scripted LLM says `is_same_entity=false`
    /// → no merge (write_gate row 3, honored without further checks).
    #[tokio::test]
    async fn llm_verify_band_false_does_not_merge() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_two_types(
            &conn,
            "g3",
            "Human",
            "A living person, described by name and biography.",
            "Vehicle",
            "A car, truck, or other conveyance used for transport.",
        )
        .await;

        let v = unit_vec4(0.0, 0.0, 1.0, 0.0);
        let mut vectors = StdHashMap::new();
        vectors.insert(
            "A living person, described by name and biography.".to_string(),
            v.clone(),
        );
        vectors.insert(
            "A car, truck, or other conveyance used for transport.".to_string(),
            v,
        );
        let embedder = MockEmbeddingProvider { vectors };

        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[{"pair_id":0,"is_same_entity":false,"confidence":0.99,"reasoning":"different concepts"}]}"#
                .to_string(),
        };

        let report = type_registry_collapse(
            &llm,
            TypeRegistryCollapseParams {
                conn: &conn,
                group_id: "g3",
                embedder: Some(&embedder),
                model_id: "test-model",
            },
        )
        .await
        .expect("type_registry_collapse must succeed");

        assert_eq!(report.candidates_nominated, 1);
        assert_eq!(
            report.merges_applied, 0,
            "LLM false verdict must never merge"
        );
        assert_eq!(
            count_entity_types(&conn, "g3").await,
            13,
            "both types survive — 11 seeded (ids 0..=10) + 2 distinct custom types"
        );
    }

    /// (d) Idempotency: a second run over an already-collapsed registry yields
    /// zero merges (no pair remains above threshold with matching signals).
    #[tokio::test]
    async fn second_run_is_idempotent_zero_merges() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        seed_two_types(
            &conn,
            "g4",
            "Company",
            "A business organisation, firm, or investment fund.",
            "company",
            "A business organisation, firm, or investment fund.",
        )
        .await;

        let v = unit_vec4(1.0, 1.0, 0.0, 0.0);
        let mut vectors = StdHashMap::new();
        vectors.insert(
            "A business organisation, firm, or investment fund.".to_string(),
            v,
        );
        let embedder = MockEmbeddingProvider { vectors };
        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[]}"#.to_string(),
        };

        let first = type_registry_collapse(
            &llm,
            TypeRegistryCollapseParams {
                conn: &conn,
                group_id: "g4",
                embedder: Some(&embedder),
                model_id: "test-model",
            },
        )
        .await
        .expect("first run must succeed");
        assert_eq!(
            first.merges_applied, 1,
            "first run merges the duplicate pair"
        );

        let second = type_registry_collapse(
            &llm,
            TypeRegistryCollapseParams {
                conn: &conn,
                group_id: "g4",
                embedder: Some(&embedder),
                model_id: "test-model",
            },
        )
        .await
        .expect("second run must succeed");
        assert_eq!(
            second.merges_applied, 0,
            "second run is a no-op — idempotent"
        );
    }

    // ── Pure-function unit tests ──────────────────────────────────────────────

    #[test]
    fn lexical_prefilter_exact_match() {
        assert!(names_share_lemma_or_exact("Company", "company"));
    }

    #[test]
    fn lexical_prefilter_trailing_s_lemma() {
        assert!(names_share_lemma_or_exact("Organization", "Organizations"));
    }

    #[test]
    fn lexical_prefilter_unrelated_names_no_match() {
        assert!(!names_share_lemma_or_exact("Person", "Vehicle"));
    }
}
