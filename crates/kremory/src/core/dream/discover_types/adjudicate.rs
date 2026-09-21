use std::time::Instant;

use metrics::{counter, histogram};

use crate::core::{
    dream::anti_redundancy,
    error::Result,
    extraction::structured::StructuredCallBuilder,
    identity_verdict::{
        identity_verdict_batch_schema, IdentityVerdictBatch, IdentityVerdictItem,
        LLM_VERIFY_CONFIDENCE_FLOOR,
    },
    provider::{chat_msg_system, chat_msg_user, ChatProvider},
};

/// Site label used on every shared `kremory.identity.*` counter for Site #2
/// (mirrors Site #3's `type_registry_collapse.rs::SITE_LABEL`).
const SITE_LABEL: &str = "site2_type_novelty";

// ─── Site #2 LLM-verify-band adjudication ──────────────────────────────────

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
/// args-as-object (rust-conventions §too_many_arguments, threshold 3).
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

/// Site #2 type-novelty write decision. UNLIKE the shared
/// [`write_gate`](crate::core::identity_verdict::write_gate) (`identity_verdict.rs`),
/// this does NOT require deterministic (lexical) corroboration — type synonyms
/// ("Firm"/"Company") are lexically dissimilar by nature, and schema-level
/// matching trusts the LLM as terminal arbiter (ontology-alignment
/// prior art). `write_gate`'s Row 6 protects against ENTITY homonymy (same name,
/// different referent), a failure mode that cannot recur here after the
/// exact-match pre-filter. The two
/// decision rules carry a bidirectional doc cross-reference so
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

/// Observability: a per-decision counter for the Site #2
/// type-novelty gate. NOT the shared `write_gate_decision_total` — Site #2
/// bypasses `write_gate`, so a distinct, honestly-named metric avoids conflating
/// the two decision rules. `decision` ∈ {`redundant`, `novel`,
/// `accept_low_confidence`}.
pub(super) fn record_type_novelty_decision(redundant: bool, verdict: &Option<IdentityVerdictItem>) {
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

/// Bundled parameters for [`record_gate_decision`] — args-as-object
/// (rust-conventions §too_many_arguments).
pub(super) struct RecordGateDecisionParams<'a> {
    pub(super) conn: &'a libsql::Connection,
    pub(super) group_id: &'a str,
    /// Shared across every row this `discover_types` invocation writes —
    /// generated once per call (see the call site), not once per row.
    pub(super) run_id: &'a str,
    pub(super) proposal_name: &'a str,
    /// `None` ONLY when no existing (non-catch-all) type existed to compare
    /// against at all (the `check_proposal` registry-empty early return) —
    /// never because a comparison happened and its result was discarded.
    pub(super) existing_name: Option<&'a str>,
    /// `None` ONLY when no cosine was ever computed for this decision (the
    /// exact-name pre-filter fires before the desc-cosine loop runs).
    pub(super) desc_cosine: Option<f32>,
    /// `Some` only on the flag-on LLM-verify sub-path; `None` for every
    /// deterministic decision (Pass, Redundant, and the flag-off
    /// `NeedsLlmVerify` fallback, which makes no LLM call).
    pub(super) verdict: Option<&'a IdentityVerdictItem>,
    /// `"accept"` | `"reject"` — the FINAL decision (did the proposal end up
    /// persisted into `entity_types`).
    pub(super) decision: &'a str,
    /// Fine-grained provenance for the metric label — which branch of the
    /// gate decided this, e.g. `"pass_no_existing_types"`,
    /// `"redundant_exact_name"`, `"llm_verify_accept"`. See call sites.
    pub(super) outcome_kind: &'a str,
    pub(super) model: &'a str,
}

/// Leave a durable, always-on trace of EVERY Pass-0 anti-redundancy
/// gate decision — Pass, Redundant, and both `NeedsLlmVerify` sub-paths (flag
/// off AND flag on) — independent of `DreamOpts::include_type_novelty_llm_verify`.
///
/// Before this, only the flag-on LLM-verify sub-path left ANY trace at all,
/// and even that was metric-only (`record_type_novelty_decision`, above) —
/// no persisted row. A default run's `Pass` accepts (the overwhelming
/// majority of decisions) left ZERO evidence: `identity_verdict_audit` never
/// received a Site #2 row, so an accepted proposal's `desc_cosine` could not
/// be reconstructed from stored state after the fact ("the flag-on
/// counterfactual CANNOT be read from stored state").
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
pub(super) async fn record_gate_decision(params: RecordGateDecisionParams<'_>) -> Result<()> {
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

