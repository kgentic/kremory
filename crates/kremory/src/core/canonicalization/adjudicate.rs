//! LLM adjudication for L5 surface-form canonicalization (ADR-063's shared
//! identity machinery, applied to the one identity site that never got it).
//!
//! ## Why this module exists
//!
//! Governing spec: `.ai-docs/specs/adr-063-embedding-identity-impl-spec-2026-07-02.md`
//! (§2.1 batch schema, §2.2 write-gate decision table, §2.3 failure-mode default,
//! §4.4 batched adjudication, §5.1 RISK-003 audit-inside-transaction).
//!
//! ADR-063 gave three identity sites a nominate → adjudicate → `write_gate`
//! pipeline (Site #3 type-registry collapse, Site #5 acronym/nickname recall,
//! Site #2's verify band). L5 was cited throughout that spec as the *precedent*
//! — the destructive-merge site whose dangers motivated the design — and was
//! never itself wired through it. Its `names_lexically_compatible` gate was left
//! as the DECIDER.
//!
//! Measured cost of that omission (2026-08-19, LoCoMo conv0, n=149, k=10,
//! `evidence_eval` instrument-validated by its own shuffle self-test):
//!
//! | arm | nDCG@10 |
//! |---|---|
//! | dream OFF | 64.0 |
//! | dream ON  | 54.1 |
//!
//! **−9.9 nDCG.** All 10 merges in the dream-ON arm were `site=canonicalize`, and
//! every one was a HYPERNYM COLLAPSE that deleted the distinguishing token:
//! `lgbtq community` → `community`, `pottery class` → `pottery`, six more. Eight
//! of the ten sat at token-Jaccard EXACTLY 0.500 — the threshold's own boundary.
//!
//! ## Why raising the threshold is NOT the fix (measured, then reverted)
//!
//! Raising `L4_LEXICAL_JACCARD_MIN` 0.5 → 0.6 recovered the loss (−0.4 nDCG) and
//! was still WRONG, because `alice j` / `alice johnson` and `Ria Patel` / `Ria`
//! are *also* Jaccard 0.500. Initial-abbreviated person names are structurally
//! identical to `pottery class` / `pottery`; no threshold separates them, 8 tests
//! across 5 files encode the abbreviation case, and corpus recall fell
//! 0.568 → 0.263. The discriminator is SEMANTIC (is one a category the other
//! belongs to?), which is exactly what an adjudicator is for — and a fourth
//! per-pair deterministic discriminator is banned by `20e1f4e3`.
//!
//! ## Shape (mirrors `dream::type_registry_collapse`, spec §4.4)
//!
//! 1. The caller nominates: cosine > threshold AND `names_lexically_compatible`.
//!    That gate is DEMOTED from decider to NOMINATOR and stays at 0.5.
//! 2. Nominees are chunked ([`chunk_pair_indices`]) into `IdentityVerdictBatch`
//!    calls — the shared schema, the shared parse-loudly discipline.
//! 3. Every candidate is decided by the shared deterministic [`write_gate`].
//!
//! ## Two deliberate divergences from Site #3, both load-bearing
//!
//! **1. `merge_threshold` is [`f32::INFINITY`], not L5's real cosine threshold.**
//! `write_gate` row 1 merges when there is NO LLM verdict but
//! `cosine >= merge_threshold` and a deterministic signal fired. Site #3 is immune
//! because its LLM-verify band sits BELOW its auto-merge threshold, so a failed
//! call lands in row 2 (`Reject`). L5 is not: every nominated pair is above the
//! cosine threshold BY CONSTRUCTION, so a timed-out or unparseable call would fall
//! straight through row 1 and merge anyway — silently restoring the −9.9 nDCG
//! defect on every transient LLM hiccup. An unreachable threshold makes row 1
//! unreachable, so adjudication failure is `Reject` (fail-closed). Dream is
//! idempotent: the pair is re-nominated next cycle. Pinned by
//! `absent_verdict_rejects_never_merges`.
//!
//! **2. The deterministic signal is RE-DERIVED here, never asserted by the caller.**
//! Per [[contract-first-before-new-public-surface]] (derive > declare): the caller
//! could pass `from_lexical(true)` since its nominees passed the gate, but that
//! makes the fact ATTESTED. Re-deriving it costs one pure string comparison and
//! catches a real hazard — L5's transitive chain resolution can hand this module a
//! `(loser, effective_keeper)` pair that NO raw cosine pair contained and that is
//! NOT lexically compatible (the `melanie` → `caroline` → `loved ones` cascade
//! shape). Re-derivation makes row 6 fire on exactly that pair, downgrading it to
//! `PotentialAlias` instead of merging. Pinned by
//! `transitive_incompatible_pair_downgrades_via_row_6`.
//!
//! ## Audit rows — wider than Site #3, deliberately
//!
//! Site #3 writes an `identity_verdict_audit` row for `Merge` and
//! `PotentialAlias` only. This site additionally writes one for `Reject` whenever
//! a verdict exists, because a rejected hypernym collapse IS the fix working and
//! the `reasoning` text is the diagnostic surface for re-measuring the 2×2. A
//! counter alone records that a reject happened; it cannot say `pottery class is a
//! class ABOUT pottery, not the same entity` (Rule 19 — a routing decision that
//! drops a candidate must be inspectable, not merely counted).

use std::collections::HashMap;
use std::time::Instant;

use metrics::{counter, histogram};

use crate::core::error::{Error, Result};
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::identity_verdict::{
    chunk_pair_indices, identity_verdict_batch_schema, write_gate, DeterministicSignal,
    IdentityVerdictBatch, IdentityVerdictItem, WriteDecision, WriteGateInputs,
    ADJUDICATION_TTFT_BUDGET_MS,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ArcChatProvider};

/// Site label on every shared `kremory.identity.*` counter this module emits.
/// Matches `MergeSite::Canonicalize`'s own `"canonicalize"` wire string rather
/// than inventing a `site#N` number — L5 is ADR-063's precedent site, not one of
/// its numbered three.
pub(super) const SITE_LABEL: &str = "l5_canonicalize";

/// The LLM handle for the adjudicated L5 path. Bundling the provider WITH its
/// model id makes "provider present but model unknown" unrepresentable — the
/// caller cannot supply half of it.
pub struct L5Adjudicator<'a> {
    /// Concrete newtype (never a bare `&dyn ChatProvider`) so
    /// [`CanonicalizeSurfaceFormsParams`](super::CanonicalizeSurfaceFormsParams)
    /// stays non-generic and every existing `adjudicator: None` call site
    /// compiles without a turbofish.
    pub llm: &'a ArcChatProvider,
    /// Resolved dream model id, threaded from the facade (TD-094 threading —
    /// the bug where dream's LLM passes ran with no model and returned empty).
    pub model_id: &'a str,
}

/// One resolved `(loser → keeper)` candidate awaiting adjudication. Carries the
/// descriptions because hypernym discrimination is exactly what the names alone
/// cannot do: `pottery class` vs `pottery` is decidable from context, not tokens.
pub(super) struct Candidate<'a> {
    pub(super) loser_id: &'a str,
    pub(super) keeper_id: &'a str,
    /// Cosine of the originating raw pair. `None` when chain resolution produced
    /// a transitive `(loser, effective_keeper)` pair that no raw pair contained —
    /// audit fidelity, NOT a gate input (see divergence 1: the gate's threshold
    /// is unreachable here regardless).
    pub(super) cosine: Option<f32>,
    pub(super) loser_description: &'a str,
    pub(super) keeper_description: &'a str,
}

/// The decision for one candidate, parallel-indexed to the caller's slice.
pub(super) struct Adjudicated {
    pub(super) decision: WriteDecision,
    /// The verdict that produced `decision`, for the audit row. `None` when the
    /// LLM call failed or that item's parse failed (→ `decision` is `Reject`).
    pub(super) verdict: Option<IdentityVerdictItem>,
}

/// Bundled params — args-as-object per TD-042 (`clippy.toml`
/// `too-many-arguments-threshold = 3`; `#[allow]` banned in `src`).
pub(super) struct AdjudicateParams<'a> {
    pub(super) candidates: &'a [Candidate<'a>],
    pub(super) group_id: &'a str,
}

/// Adjudicate every candidate and return one [`Adjudicated`] per input, in input
/// order. Never fails on LLM trouble: a failed call or unparseable item leaves
/// that candidate with no verdict, which [`write_gate`] turns into `Reject`.
pub(super) async fn adjudicate(
    adjudicator: &L5Adjudicator<'_>,
    params: AdjudicateParams<'_>,
) -> Result<Vec<Adjudicated>> {
    let AdjudicateParams {
        candidates,
        group_id,
    } = params;

    counter!(
        "kremory.identity.candidate_nominated_total",
        "site" => SITE_LABEL
    )
    .increment(candidates.len() as u64);

    let mut verdicts: HashMap<usize, IdentityVerdictItem> = HashMap::new();
    for range in chunk_pair_indices(candidates.len()) {
        let chunk = &candidates[range.clone()];
        let chunk_verdicts = adjudicate_chunk(adjudicator, chunk, group_id).await?;
        for (local_pair_id, verdict) in chunk_verdicts {
            // Chunk-LOCAL pair_id → GLOBAL index into `candidates`.
            verdicts.insert(range.start + local_pair_id, verdict);
        }
    }

    Ok(candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let verdict = verdicts.remove(&idx);
            let decision = decide(candidate, verdict.clone());
            record_write_gate_decision(decision);
            Adjudicated { decision, verdict }
        })
        .collect())
}

/// The [`write_gate`] call for one candidate — extracted so the two divergences
/// documented in this module's header are unit-testable without an LLM.
fn decide(candidate: &Candidate<'_>, verdict: Option<IdentityVerdictItem>) -> WriteDecision {
    write_gate(WriteGateInputs {
        // Audit-fidelity value only; row 1 is unreachable (see below).
        cosine: candidate.cosine.unwrap_or(0.0),
        // Divergence 1 — fail-closed. Row 1 ("no verdict + cosine ≥ threshold +
        // deterministic → Merge") MUST NOT be reachable on this site, because
        // every candidate is above L5's real cosine threshold by construction and
        // would therefore merge on LLM failure.
        merge_threshold: f32::INFINITY,
        // Divergence 2 — DERIVED, never asserted by the caller.
        deterministic_signal: DeterministicSignal::from_lexical(
            crate::core::disambiguation::names_lexically_compatible(
                candidate.loser_id,
                candidate.keeper_id,
            ),
        ),
        llm_verdict: verdict,
        // Site #6 floor not shipped — composes as a no-op (spec §2.2.1).
        min_confidence_floor: None,
        // Row 0 temporal veto: never collapse two points in time.
        names: Some((candidate.loser_id, candidate.keeper_id)),
    })
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

/// One batched `IdentityVerdictBatch` call over `chunk`, returning verdicts keyed
/// by chunk-LOCAL `pair_id`. A missing key means "no verdict" — the caller's
/// `write_gate` turns that into `Reject`.
async fn adjudicate_chunk(
    adjudicator: &L5Adjudicator<'_>,
    chunk: &[Candidate<'_>],
    group_id: &str,
) -> Result<HashMap<usize, IdentityVerdictItem>> {
    let schema = identity_verdict_batch_schema(chunk.len());

    let call_start = Instant::now();
    let raw_value = StructuredCallBuilder::new(adjudicator.llm, &schema, "IdentityVerdictBatch")
        .model(adjudicator.model_id)
        .messages(build_adjudication_messages(chunk))
        // Dream is latency-tolerant by design (ADR-063 spec) — raised for THIS
        // call site only, never the shared default other sites depend on.
        .ttft_budget_ms(ADJUDICATION_TTFT_BUDGET_MS)
        .call()
        .await;
    histogram!(
        "kremory.identity.llm_call_latency_ms_histogram",
        "site" => SITE_LABEL
    )
    .record(call_start.elapsed().as_millis() as f64);

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory::canonicalization::adjudicate::raw_payload",
            model_id = %adjudicator.model_id,
            group_id = %group_id,
            response = ?raw_value,
            "L5 adjudication raw response"
        );
    }

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "kremory::canonicalization::adjudicate",
                error = %e,
                group_id = %group_id,
                chunk_len = chunk.len(),
                "L5 adjudication LLM call failed — chunk's candidates default to no-verdict (fail-closed: write_gate rejects them)"
            );
            return Ok(HashMap::new());
        }
    };

    // Loose envelope first (spec §2.1: one malformed element ≠ whole batch loss),
    // then per-item strict parse below.
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
                        target: "kremory::canonicalization::adjudicate",
                        error = %e,
                        group_id = %group_id,
                        "L5 adjudication: failed to parse IdentityVerdictBatch — chunk's candidates default to no-verdict"
                    );
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(chunk.len() as u64);
                    return Ok(HashMap::new());
                }
            }
        }
    };

    let mut verdicts: HashMap<usize, IdentityVerdictItem> = HashMap::new();
    for raw_item in &batch.verdicts {
        match serde_json::from_value::<IdentityVerdictItem>(raw_item.clone()) {
            Ok(item) => {
                if item.pair_id >= chunk.len() {
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(1);
                    tracing::warn!(
                        target: "kremory::canonicalization::adjudicate",
                        pair_id = item.pair_id,
                        chunk_len = chunk.len(),
                        "L5 adjudication: verdict pair_id out of range — dropped"
                    );
                    continue;
                }
                verdicts.entry(item.pair_id).or_insert(item);
            }
            Err(e) => {
                counter!(
                    "kremory.identity.verdict_parse_fail_total",
                    "site" => SITE_LABEL
                )
                .increment(1);
                tracing::warn!(
                    target: "kremory::canonicalization::adjudicate",
                    error = %e,
                    "L5 adjudication: skipping malformed verdict item"
                );
            }
        }
    }
    Ok(verdicts)
}

/// Build the adjudication prompt. The system message names the HYPERNYM trap
/// explicitly and gives both sides of it, because that is the discrimination the
/// measured failure needed and the one token-Jaccard provably cannot make: the
/// merges that cost 9.9 nDCG (`pottery class` → `pottery`) and the merges that
/// must be KEPT (`alice j` → `alice johnson`) are the same Jaccard 0.500 shape.
///
/// Positive AND negative worked examples per
/// [[prompt-engineering-positive-and-generic-examples]]; the examples are generic
/// (not drawn from any corpus under evaluation) so this prompt is not tuned to the
/// benchmark it will be measured on.
fn build_adjudication_messages(chunk: &[Candidate<'_>]) -> Vec<crate::core::provider::ChatMessage> {
    let system = "You are a knowledge-graph entity-resolution analyst. You will be shown pairs \
of entity names (with descriptions) that a similarity gate flagged as POSSIBLY the same \
real-world entity. Decide, for each pair, whether they denote the SAME individual entity.\n\n\
The critical distinction is IDENTITY versus CATEGORY. Two names can share most of their \
words and still be different entities, because one names a CATEGORY, TOPIC or ACTIVITY that \
the other BELONGS TO:\n\
  - \"pottery class\" vs \"pottery\" -> NOT the same: a class ABOUT a subject is not the subject.\n\
  - \"chess club\" vs \"chess\" -> NOT the same.\n\
  - \"lgbtq community\" vs \"community\" -> NOT the same: dropping the qualifier changes the referent.\n\
  - \"amazon river\" vs \"amazon\" -> NOT the same.\n\n\
Conversely, an ABBREVIATED or PARTIAL form of the same name IS the same entity:\n\
  - \"alice j\" vs \"alice johnson\" -> SAME: an abbreviated personal name.\n\
  - \"j smith\" vs \"john smith\" -> SAME.\n\
  - \"dr patel\" vs \"priya patel\" -> SAME person.\n\n\
When the two names differ by a word that NARROWS or QUALIFIES the meaning, they are \
different entities. When they differ only by an abbreviation, initial, title or spelling \
of the SAME referent, they are the same entity. If genuinely unsure, answer false with low \
confidence — a wrong merge destroys data irreversibly, a missed merge is retried next cycle.\n\n\
Respond ONLY with the JSON structure — no extra commentary."
        .to_string();

    let pairs_list = chunk
        .iter()
        .enumerate()
        .map(|(pair_id, c)| {
            format!(
                "Pair {pair_id}:\n  A: name=\"{}\" description=\"{}\"\n  B: name=\"{}\" description=\"{}\"",
                c.keeper_id,
                truncate_for_prompt(c.keeper_description),
                c.loser_id,
                truncate_for_prompt(c.loser_description),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let user = format!(
        "Adjudicate the following entity pairs. For each pair output a verdict object with \
`pair_id` (matching the pair number below), `is_same_entity` (true only if A and B denote \
the same real-world entity), `confidence` (0.0-1.0), and `reasoning` (one short \
sentence).\n\n{pairs_list}"
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

/// Cap each description's contribution to the prompt. A single pathological
/// description must not consume the context budget the other 9 pairs in the chunk
/// need — the S3 spike's root cause was per-call budget exhaustion, and an
/// unbounded field reintroduces it by a different route. Char-boundary safe.
fn truncate_for_prompt(description: &str) -> String {
    const MAX_CHARS: usize = 400;
    if description.chars().count() <= MAX_CHARS {
        return description.to_string();
    }
    let truncated: String = description.chars().take(MAX_CHARS).collect();
    format!("{truncated}…")
}

/// Insert one `identity_verdict_audit` row for a NON-merge decision, outside any
/// transaction. `Merge` rows are written INSIDE the merge's `BEGIN IMMEDIATE` by
/// `apply_merge_with_audit` (spec §5.1 RISK-003), never here.
pub(super) async fn write_non_merge_audit_row(
    conn: &libsql::Connection,
    row: super::IdentityVerdictAuditRow<'_>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO identity_verdict_audit \
         (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
          llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        libsql::params![
            row.site,
            row.group_id,
            row.candidate_a,
            row.candidate_b,
            row.cosine.map(f64::from),
            row.structural_signal,
            row.verdict.map(|v| v.is_same_entity),
            row.verdict.map(|v| f64::from(v.confidence)),
            row.verdict.map(|v| v.reasoning.clone()),
            row.decision,
            row.run_id,
        ],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "L5 adjudication: identity_verdict_audit insert failed: {e}"
        ))
    })?;
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate<'a>(loser: &'a str, keeper: &'a str, cosine: f32) -> Candidate<'a> {
        Candidate {
            loser_id: loser,
            keeper_id: keeper,
            cosine: Some(cosine),
            loser_description: "",
            keeper_description: "",
        }
    }

    fn verdict(is_same: bool, confidence: f32) -> IdentityVerdictItem {
        IdentityVerdictItem {
            pair_id: 0,
            is_same_entity: is_same,
            confidence,
            reasoning: "test".to_string(),
        }
    }

    /// Divergence 1, the fail-closed guard. A nominated pair is ALWAYS above L5's
    /// real cosine threshold, so if `merge_threshold` were that threshold, a
    /// failed LLM call would fall through `write_gate` row 1 and merge anyway —
    /// silently restoring the −9.9 nDCG defect. Cosine 0.99 here is deliberately
    /// far above it.
    #[test]
    fn absent_verdict_rejects_never_merges() {
        assert_eq!(
            decide(&candidate("pottery class", "pottery", 0.99), None),
            WriteDecision::Reject,
            "an absent verdict on a high-cosine, lexically-compatible pair must \
             fail CLOSED — row 1 must be unreachable on this site"
        );
    }

    /// The measured defect: 8 of 10 damaging merges were Jaccard exactly 0.500,
    /// so they pass the nominator. The adjudicator's `false` is what stops them.
    #[test]
    fn hypernym_collapse_is_rejected_when_llm_says_not_same() {
        assert_eq!(
            decide(
                &candidate("pottery class", "pottery", 0.92),
                Some(verdict(false, 0.9))
            ),
            WriteDecision::Reject
        );
    }

    /// The other half of the same Jaccard-0.500 shape, which the reverted
    /// threshold fix would have destroyed. This is why the fix had to be semantic.
    #[test]
    fn abbreviated_person_name_still_merges_when_llm_says_same() {
        assert_eq!(
            decide(
                &candidate("alice j", "alice johnson", 0.92),
                Some(verdict(true, 0.9))
            ),
            WriteDecision::Merge
        );
    }

    /// Divergence 2. L5's transitive chain resolution can produce a
    /// `(loser, effective_keeper)` pair that no raw cosine pair contained and that
    /// is NOT lexically compatible — the `melanie` → `caroline` → `loved ones`
    /// cascade shape. Because the signal is DERIVED here rather than asserted by
    /// the caller, row 6 fires and the destructive write is downgraded.
    #[test]
    fn transitive_incompatible_pair_downgrades_via_row_6() {
        assert_eq!(
            decide(
                &candidate("melanie", "loved ones", 0.91),
                Some(verdict(true, 0.95))
            ),
            WriteDecision::PotentialAlias,
            "zero shared tokens ⇒ no deterministic corroboration ⇒ the LLM must \
             not authorize a destructive write alone (write_gate row 6)"
        );
    }

    /// Low-confidence agreement is never a destructive write (row 4).
    #[test]
    fn low_confidence_agreement_downgrades_to_potential_alias() {
        assert_eq!(
            decide(
                &candidate("alice j", "alice johnson", 0.92),
                Some(verdict(true, 0.5))
            ),
            WriteDecision::PotentialAlias
        );
    }

    /// Row 0. A temporal discriminator mismatch is a HARD disqualifier — it must
    /// survive even a confident `true` on an otherwise compatible pair.
    #[test]
    fn temporal_conflict_vetoes_even_a_confident_yes() {
        assert_eq!(
            decide(
                &candidate("meeting on 27 june 2023", "meeting on 28 june 2023", 0.99),
                Some(verdict(true, 0.99))
            ),
            WriteDecision::Reject
        );
    }

    #[test]
    fn truncate_for_prompt_is_char_boundary_safe() {
        let long = "é".repeat(500);
        let out = truncate_for_prompt(&long);
        assert_eq!(out.chars().count(), 401, "400 chars + the ellipsis");
        assert!(out.ends_with('…'));
        assert_eq!(truncate_for_prompt("short"), "short");
    }

    /// The prompt must carry BOTH sides of the Jaccard-0.500 ambiguity, or the
    /// model has no basis to separate them. Guards against a future edit that
    /// trims the system message down to the hypernym half only.
    #[test]
    fn prompt_teaches_both_sides_of_the_jaccard_ambiguity() {
        let messages = build_adjudication_messages(&[candidate("alice j", "alice johnson", 0.9)]);
        let system = format!("{messages:?}");
        assert!(
            system.contains("pottery class"),
            "hypernym negative example"
        );
        assert!(system.contains("alice j"), "abbreviation positive example");
        assert!(
            system.contains("Pair 0:"),
            "candidate must be enumerated with its pair_id"
        );
    }
}
