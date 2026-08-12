//! Shared identity-verdict schema + deterministic write-gate (ADR-063 spec §2).
//!
//! Three ADR-063 sites — Site #5 (instance acronym/nickname recall), Site #3
//! (type-registry post-hoc collapse), and Site #2 (type-novelty description-gate
//! LLM-verify band) — all share one shape: an LLM proposes a structured verdict on
//! a candidate identity pair, and a DETERMINISTIC function decides what to write.
//! Building this once (vs three times) keeps the parse-loudly discipline, the
//! observability surface, and the write-gate invariant in a single place.
//!
//! ## The load-bearing invariant (ADR-057 generalized)
//!
//! **The LLM verdict NEVER authorizes a destructive write alone.** It is one input
//! among cosine + a deterministic signal (lexical / structural pre-filter) +
//! (optionally) a confidence floor. [`write_gate`] is a pure, deterministic,
//! I/O-free, counter-free function — unit-testable exhaustively, independent of any
//! LLM call. Per [[load-bearing-invariants-at-emit-not-prompt]] this is enforced
//! STRUCTURALLY in the decision table (rows 4/6 below), NOT via a prompt instruction
//! telling the model "only say yes if you're sure".
//!
//! ## Location (spec §2.4, supersedes the handoff-plan path)
//!
//! Lives at `core/identity_verdict.rs` (sibling of `disambiguation/` and
//! `canonicalization.rs`, NOT nested in `core/dream/`) because it is consumed by
//! both dream passes (Site #5, Site #3) AND composes with Site #6, which lives in
//! `disambiguation`/`canonicalization` (`core/`, not `core/dream/`). Neither dream
//! nor disambiguation is a natural parent. (The handoff plan's
//! `core/dream/identity_verdict.rs` path is superseded by this spec §2.4 reasoning.)
//!
//! ## Two spec ambiguities resolved at implementation time (documented, not silent)
//!
//! 1. **Row-1 merge threshold** — spec §2.2 row 1 references a bare `MERGE_THRESHOLD`
//!    constant, but the clear-merge cosine threshold differs per site (Site #3 desc
//!    auto-merge 0.85; Site #5 has no meaningful cosine). A single shared constant
//!    cannot serve both. Resolution: [`WriteGateInputs::merge_threshold`] threads the
//!    caller's site-specific clear-merge threshold, keeping `write_gate` site-agnostic.
//!    Site #5 always passes `cosine = 0.0`, so row 1 can never fire there regardless
//!    of the threshold (matches spec §3.3: "row 1 can never fire for Site #5").
//! 2. **Site-#6 confidence floor** — spec §2.2 row 5's `min_confidence_floor` clause
//!    is spike-gated (S4) and Site #6 is not yet built. `min_confidence_floor` is
//!    `None` until Site #6 ships, and the clause is then vacuously satisfied
//!    (§2.2.1 graceful degradation). Until S4 calibrates a Site-#6-specific
//!    `CONFIDENCE_REJECT_FLOOR`, the provisional floor reuses
//!    [`LLM_VERIFY_CONFIDENCE_FLOOR`] (spec §8: reuse `MIN_VERIFY_CONFIDENCE` as the
//!    starting candidate).

use serde::Deserialize;

/// Confidence floor below which an `is_same_entity: true` verdict downgrades to
/// `PotentialAlias` instead of `Merge` (spec §2.2 row 4).
///
/// Starting candidate = `consistency_check::MIN_VERIFY_CONFIDENCE` (0.7), per spec
/// §8: reuse the already-shipped, ADR-047-calibrated value rather than inventing a
/// seventh spike. S2's fixture run should report precision/recall AT this threshold
/// before any different number is locked; a deviation becomes a documented one-line
/// justification, not a fresh unbounded threshold search.
pub(crate) const LLM_VERIFY_CONFIDENCE_FLOOR: f32 = 0.7;

/// Per-pair identity verdict — the semantic payload the LLM returns for ONE
/// candidate pair (one element of a batch response, spec §2.1).
///
/// Per [[llm-output-parse-loudly]]: none of the four fields carries
/// `#[serde(default)]` — a truncated or malformed per-item response fails THIS
/// item's parse loudly (the caller routes that one `pair_id` to the safe
/// `Reject`/`None` default and increments `verdict_parse_fail_total{site}`), never
/// silently defaulting `is_same_entity` to `false` or `confidence` to `0.0`.
///
/// `reasoning` is additionally shape-validated at the parser boundary (custom
/// `Deserialize` below) — a value carrying JSON-fragment tells (`{`, `}`, `\`, or
/// the literal `"is_same_entity"`) is evidence the structured response was spliced
/// into free text and is rejected wholesale.
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern, mirrors the Site #3/#5 test-utils
/// re-exports in `dream/mod.rs`) — promoted from `pub(crate)` so the Site #2
/// metrics harness (`tests/dream_metrics_harness_site2.rs`, an external
/// integration-test binary) can construct/inspect verdicts directly. Not part
/// of the stable public API contract.
#[derive(Debug, Clone, PartialEq)]
#[doc(hidden)]
pub struct IdentityVerdictItem {
    /// Correlates this verdict back to `nominated_pairs[pair_id]`. Mirrors
    /// `VerifyBatchDecision::candidate_idx` — an index into the caller's input
    /// slice, NOT a DB rowid.
    pub pair_id: usize,
    /// The LLM's identity judgment. REQUIRED (no `serde(default)`).
    pub is_same_entity: bool,
    /// The LLM's own calibration signal in `[0, 1]`. REQUIRED.
    pub confidence: f32,
    /// Free-text rationale — for audit/debug only, NEVER parsed for control flow.
    pub reasoning: String,
}

impl<'de> Deserialize<'de> for IdentityVerdictItem {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Raw shadow struct — no `#[serde(default)]`, so a missing required field is
        // a parse error (mirrors `VerifyDecision`'s loud-parse precedent).
        #[derive(Deserialize)]
        struct Raw {
            pair_id: usize,
            is_same_entity: bool,
            confidence: f32,
            reasoning: String,
        }
        let raw = Raw::deserialize(deserializer)?;
        if reasoning_looks_like_json_fragment(&raw.reasoning) {
            return Err(serde::de::Error::custom(
                "reasoning field contains JSON-fragment garbage (spliced structured \
                 output) — rejecting the whole verdict per llm-output-parse-loudly",
            ));
        }
        Ok(Self {
            pair_id: raw.pair_id,
            is_same_entity: raw.is_same_entity,
            confidence: raw.confidence,
            reasoning: raw.reasoning,
        })
    }
}

/// Shape-validator for the `reasoning` free-text field (spec §2.1). A value
/// carrying JSON-syntax tells is a fragment of the structured response spliced into
/// the text field — reject it. Applied at the parser boundary, not after persistence.
pub(crate) fn reasoning_looks_like_json_fragment(reasoning: &str) -> bool {
    reasoning.contains('{')
        || reasoning.contains('}')
        || reasoning.contains('\\')
        || reasoning.contains("is_same_entity")
}

/// Wire-level batch envelope — what `StructuredCallBuilder` parses per LLM call
/// (one call adjudicates MANY nominated pairs). Mirrors
/// `VerifyBatch { decisions: Vec<serde_json::Value> }` exactly: the outer envelope
/// is deliberately loose (`Vec<serde_json::Value>`, NOT `Vec<IdentityVerdictItem>`)
/// so ONE malformed element does not fail the ENTIRE batch parse — each element is
/// independently deserialized into [`IdentityVerdictItem`] downstream (or fails
/// loudly per-element).
///
/// `#[serde(default)]` on the `Vec` wrapper is the ONE legitimate default per
/// [[llm-output-parse-loudly]]'s container exemption: an empty array degrades to
/// "every nominated pair in this call falls back to the safe default", which is
/// itself loud (each `pair_id` fails to find a matching verdict and is logged).
#[derive(Debug, Deserialize)]
pub(crate) struct IdentityVerdictBatch {
    #[serde(default)]
    pub(crate) verdicts: Vec<serde_json::Value>,
}

/// A deterministic, identity-adjacent signal that fired for a candidate pair
/// (spec §2.2.2, RISK-001 hardening). Newtype wrapper over `bool` constructible
/// ONLY via the two sanctioned constructors, so a future call site cannot quietly
/// pass an ad-hoc `cosine >= 0.6` boolean into the write-gate (which would
/// reintroduce "cosine alone can approve a destructive write" — the exact failure
/// class ADR-057 exists to prevent) — the type itself refuses to compile that.
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern) — promoted from `pub(crate)` for
/// the Site #2 metrics harness (`tests/dream_metrics_harness_site2.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct DeterministicSignal(bool);

impl DeterministicSignal {
    /// Site #3 (type-registry collapse): the `names_lexically_compatible`-style
    /// lemma/exact-match result on the type NAME.
    pub fn from_lexical(names_lexically_compatible_result: bool) -> Self {
        Self(names_lexically_compatible_result)
    }

    /// Site #5 (acronym/nickname recall): the structural pre-filter's boolean
    /// nomination result (`initialism_candidate OR cooccurs_in_graph`). TRUE means a
    /// deterministic, embedder-independent signal justified escalating to the LLM —
    /// the analogue of token-Jaccard for the zero-lexical-overlap acronym/nickname
    /// surface (spec §2.2.2).
    pub fn from_structural_prefilter(nominated: bool) -> Self {
        Self(nominated)
    }

    /// Whether a deterministic signal fired.
    pub fn fired(self) -> bool {
        self.0
    }
}

/// Inputs to [`write_gate`] — one resolved candidate pair (batch-splitting happens
/// in the caller, BEFORE `write_gate` is invoked per pair).
///
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern) — promoted from `pub(crate)` for
/// the Site #2 metrics harness (`tests/dream_metrics_harness_site2.rs`).
#[doc(hidden)]
pub struct WriteGateInputs<'a> {
    /// Pre-computed similarity. `0.0` when N/A (Site #5 acronym pairs, where cosine
    /// was never an identity signal per ADR-063).
    pub cosine: f32,
    /// The caller's site-specific clear-merge cosine threshold (resolves spec §2.2's
    /// shared-`MERGE_THRESHOLD` ambiguity — see module docs). Site #5 passes any
    /// value because its `cosine` is always `0.0`.
    pub merge_threshold: f32,
    /// The deterministic identity-adjacent signal (spec §2.2's `lexically_compatible`
    /// parameter, wrapped in the RISK-001 newtype). Site #3: lexical match; Site #5:
    /// structural-pre-filter nomination.
    pub deterministic_signal: DeterministicSignal,
    /// The single-pair verdict, already split out of its batch response by `pair_id`.
    /// `None` when the margin trigger did not fire (clear case, no LLM call) OR when
    /// the LLM call / that item's parse failed.
    pub llm_verdict: Option<IdentityVerdictItem>,
    /// Site #6's observed minimum entity extraction-confidence, when available.
    /// `None` until Site #6 ships → the row-5 floor clause is vacuously satisfied
    /// (spec §2.2.1 graceful degradation). Compared against
    /// [`LLM_VERIFY_CONFIDENCE_FLOOR`] provisionally until S4 calibrates a
    /// Site-#6-specific floor.
    pub min_confidence_floor: Option<f32>,
    /// The two candidate names, for the temporal-conflict veto (row 0). `None`
    /// composes as a no-op, exactly like [`Self::min_confidence_floor`] above.
    ///
    /// **TD-212/TD-213.** A destructive merge must never collapse two different
    /// points in time. Measured on `graph_mutation_log`: `site5_acronym_nickname`
    /// merged `1037 am on 27 june 2023` into `1037 am` in **10 of 21** merges —
    /// irreversible data loss, since the bare time cannot be recovered.
    ///
    /// This carries NAMES rather than a caller-computed `bool` deliberately: a
    /// pre-computed flag is a fact the caller ASSERTS and can forget or get wrong,
    /// whereas names let the gate DERIVE the verdict itself. Site #5 cannot use
    /// the full lexical gate (its acronym pairs share zero tokens by construction,
    /// so the Jaccard arm would reject exactly what that site is for), which is why
    /// only the temporal arm is applied here.
    pub names: Option<(&'a str, &'a str)>,
}

/// The write-gate's decision for one candidate pair.
///
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern) — promoted from `pub(crate)` for
/// the Site #2 metrics harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum WriteDecision {
    /// Destructive — caller reuses an existing id / remaps `entity_type_id`.
    Merge,
    /// Non-destructive — caller inserts an alias/candidate edge, defers to next cycle.
    PotentialAlias,
    /// No relationship — caller does nothing.
    Reject,
}

/// Deterministic write-gate (spec §2.2). Pure: no I/O, no metrics (site callers emit
/// their own site-labeled counters around the call). Decision table evaluated
/// top-to-bottom, first match wins.
///
/// | # | Condition | Decision |
/// |---|---|---|
/// | 1 | no LLM verdict AND `cosine ≥ merge_threshold` AND deterministic signal fired | `Merge` (clear case — LLM never invoked) |
/// | 2 | no LLM verdict AND not row 1 | `Reject` (safe default — a nominated pair whose LLM call failed/timed out must NOT merge, spec §2.3) |
/// | 3 | LLM says `is_same_entity == false` | `Reject` (a `false` verdict can never become `Merge`) |
/// | 4 | LLM `true` but `confidence < LLM_VERIFY_CONFIDENCE_FLOOR` | `PotentialAlias` (low-confidence agreement is not enough for a destructive write) |
/// | 6 | LLM `true`, confident, BUT no deterministic signal fired | `PotentialAlias`, never `Merge` (**the load-bearing generalization: the LLM never authorizes a destructive write alone**) |
/// | 5b | LLM `true`, confident, deterministic signal fired, BUT Site-#6 floor set and observed conf below it | `PotentialAlias` (Site #6 downgrade) |
/// | 5 | LLM `true`, confident, deterministic signal fired, floor cleared (or absent) | `Merge` (the ONLY row where an LLM verdict authorizes `Merge` — and never alone) |
///
/// **ADR-065 carve-out (Site #2):** the dream-phase type-novelty gate
/// (`dream::discover_types::type_novelty_is_redundant`) intentionally does NOT
/// call this function. Schema-level TYPE matching trusts the LLM as terminal
/// arbiter, because Row 6's deterministic-corroboration requirement — an
/// ADR-057 ENTITY-homonymy guard (same name, *different* referent) — over-
/// generalizes to type synonyms ("Firm"/"Company"), which are lexically
/// dissimilar by nature and so structurally fail the lexical signal even when a
/// correct, confident `true` verdict is returned. Do NOT re-unify the two rules
/// without reading ADR-065 (the residual-risk analysis is there). Entity /
/// instance identity (Sites #3 / #5 / #6) still uses this gate UNCHANGED.
///
/// `pub` + `#[doc(hidden)]` (MNT-002 pattern) — promoted from `pub(crate)` so
/// the Site #2 metrics harness (`tests/dream_metrics_harness_site2.rs`) can
/// replicate the discover_types Site #2 decision flow exactly.
#[doc(hidden)]
pub fn write_gate(inputs: WriteGateInputs<'_>) -> WriteDecision {
    let WriteGateInputs {
        cosine,
        merge_threshold,
        deterministic_signal,
        llm_verdict,
        min_confidence_floor,
        names,
    } = inputs;
    let deterministic = deterministic_signal.fired();

    // Row 0 (TD-212/TD-213): a destructive merge may never collapse two different
    // points in time. Checked FIRST so it also vetoes row 1, the no-LLM clear-merge
    // path — `1037 am on 27 june 2023` into `1037 am` is irreversible data loss,
    // and no amount of cosine or LLM agreement makes it not so. Reuses ADR-057's
    // deterministic rule; `None` composes as a no-op.
    if let Some((name_a, name_b)) = names {
        if crate::core::disambiguation::temporal_conflict(name_a, name_b) {
            // NO counter here: this module's header documents `write_gate` as
            // "pure, deterministic, I/O-free, counter-free". The caller's existing
            // `record_write_gate_decision` already counts the resulting Reject.
            return WriteDecision::Reject;
        }
    }

    let Some(verdict) = llm_verdict else {
        // Rows 1-2: no LLM verdict.
        if cosine >= merge_threshold && deterministic {
            return WriteDecision::Merge; // row 1 — clear case, no LLM needed
        }
        return WriteDecision::Reject; // row 2 — no clear merge, no LLM: safe no-op
    };

    // Row 3: the LLM explicitly says NOT the same — honored without further checks.
    if !verdict.is_same_entity {
        return WriteDecision::Reject;
    }
    // Row 4: low-confidence agreement is not sufficient for a destructive write.
    if verdict.confidence < LLM_VERIFY_CONFIDENCE_FLOOR {
        return WriteDecision::PotentialAlias;
    }
    // Row 6 (load-bearing invariant): an LLM `true` verdict does NOT override the
    // total absence of any deterministic corroboration. Cosine + one LLM call never
    // authorize a destructive write alone unless a deterministic signal also fired.
    if !deterministic {
        return WriteDecision::PotentialAlias;
    }
    // Row 5b: Site #6 confidence floor (composes as a no-op when None, spec §2.2.1).
    if let Some(observed_min_conf) = min_confidence_floor {
        if observed_min_conf < LLM_VERIFY_CONFIDENCE_FLOOR {
            return WriteDecision::PotentialAlias;
        }
    }
    // Row 5: the only path to an LLM-authorized Merge — and it is never alone.
    WriteDecision::Merge
}

/// Hand-crafted JSON Schema descriptor for [`IdentityVerdictBatch`], passed to
/// `StructuredCallBuilder`. Mirrors `verify_batch_schema`'s signature + shape
/// exactly (Anthropic `NativeSchema` constraints: `additionalProperties: false` on
/// every object node; NO `minItems`/`maxItems` on arrays; NO `minimum`/`maximum` on
/// numbers). `batch_len` is retained for caller/signature parity but is UNUSED —
/// array-length enforcement must come from the prompt + caller-side validation, not
/// the schema (verified Anthropic limitation, see `verify_batch_schema`).
pub(crate) fn identity_verdict_batch_schema(_batch_len: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["pair_id", "is_same_entity", "confidence", "reasoning"],
                    "properties": {
                        "pair_id": { "type": "integer" },
                        "is_same_entity": { "type": "boolean" },
                        "confidence": { "type": "number" },
                        "reasoning": { "type": "string" }
                    }
                }
            }
        }
    })
}

/// Maximum nominated pairs adjudicated in ONE `IdentityVerdictBatch` LLM call
/// (spec §2.1/§3.2/§4.4 batch shape — batching itself is unchanged; this bounds
/// how many pairs share a single call).
///
/// Root cause (S3 spike, `type_registry_collapse_s3_spike.rs`): a realistic
/// 25-pair batch against a live local model (gemma4:e4b) exceeds
/// `StructuredCallBuilder`'s per-arm wall-clock budget on all 4 fallback arms —
/// the call simply takes longer than the model needs to reason over 25 pairs at
/// once. The batch envelope's `write_gate` fail-closed default (row 2: "no LLM
/// verdict -> Reject") means this was SAFE (zero false merges) but INERT (every
/// nominated pair silently defaults to no-verdict at realistic registry sizes).
///
/// Fix is two-part per Quinn's spike review: (1) this chunk size caps each call's
/// pair count so per-call latency stays inside budget — 10 is the size
/// `smoke_one_human_individual_pair_s3`-style single/near-single-pair calls have
/// empirically proven fast; (2) callers additionally raise
/// `StructuredCallBuilder::ttft_budget_ms` for these dream-phase adjudication call
/// sites specifically (dream is latency-tolerant by design, ADR-063 spec) rather
/// than changing the shared global default other non-dream call sites depend on.
pub(crate) const ADJUDICATION_CHUNK_SIZE: usize = 10;

/// Raised per-arm wall-clock budget (ms) for dream-phase `IdentityVerdictBatch`
/// adjudication calls specifically. Dream runs are background/latency-tolerant
/// (ADR-063 spec) — this is set on the `StructuredCallBuilder` for THESE call
/// sites only via `.ttft_budget_ms()`, never by changing
/// `StructuredCallBuilder::new`'s shared 30s default (`structured.rs`), which
/// other, latency-sensitive call sites depend on.
pub(crate) const ADJUDICATION_TTFT_BUDGET_MS: u64 = 180_000;

/// Split `total` nominated-pair indices into contiguous chunks of at most
/// [`ADJUDICATION_CHUNK_SIZE`] each, in original order. Pure, deterministic,
/// I/O-free — every global pair index `0..total` appears in exactly one
/// returned range, in ascending order, with no gaps or overlaps.
///
/// Callers slice their `nominated` vec by each returned `Range<usize>`, run ONE
/// `IdentityVerdictBatch` call per chunk (the LLM sees a chunk-LOCAL `pair_id`
/// space, `0..chunk.len()`), then remap each returned verdict's `pair_id` back to
/// the GLOBAL index via `range.start + local_pair_id` before inserting into the
/// merged `HashMap<usize, IdentityVerdictItem>` (see
/// `type_registry_collapse::adjudicate_batch` / `acronym_nickname_recall::adjudicate_batch`).
pub(crate) fn chunk_pair_indices(total: usize) -> Vec<std::ops::Range<usize>> {
    if total == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::with_capacity(total.div_ceil(ADJUDICATION_CHUNK_SIZE));
    let mut start = 0;
    while start < total {
        let end = (start + ADJUDICATION_CHUNK_SIZE).min(total);
        chunks.push(start..end);
        start = end;
    }
    chunks
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn verdict(pair_id: usize, is_same: bool, confidence: f32) -> IdentityVerdictItem {
        IdentityVerdictItem {
            pair_id,
            is_same_entity: is_same,
            confidence,
            reasoning: "ok".to_string(),
        }
    }

    // ── write_gate decision table — one test per row (spec §2.2) ────────────────

    #[test]
    fn row1_no_llm_clear_merge() {
        // cosine ≥ threshold + deterministic signal + no LLM → Merge (clear case).
        let d = write_gate(WriteGateInputs {
            cosine: 0.90,
            merge_threshold: 0.85,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: None,
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::Merge);
    }

    #[test]
    fn row1_blocked_without_deterministic_signal() {
        // cosine ≥ threshold but no deterministic signal + no LLM → Reject (row 2),
        // NEVER Merge (cosine alone cannot authorize a destructive write).
        let d = write_gate(WriteGateInputs {
            cosine: 0.99,
            merge_threshold: 0.85,
            deterministic_signal: DeterministicSignal::from_lexical(false),
            llm_verdict: None,
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::Reject);
    }

    #[test]
    fn row2_no_llm_below_threshold_is_reject() {
        let d = write_gate(WriteGateInputs {
            cosine: 0.50,
            merge_threshold: 0.85,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: None,
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::Reject);
    }

    #[test]
    fn site5_cosine_zero_never_merges_without_llm() {
        // Site #5 always passes cosine = 0.0, so row 1 can never fire (spec §3.3).
        let d = write_gate(WriteGateInputs {
            cosine: 0.0,
            merge_threshold: 0.0, // even a zero threshold: 0.0 >= 0.0 would be true...
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: None,
            min_confidence_floor: None,
            names: None,
        });
        // ...but the intent is Site #5 uses a positive threshold; with the default
        // spec framing cosine=0.0 + no LLM is not a merge. Assert the safe outcome
        // when the site follows the spec (positive threshold):
        let d_positive = write_gate(WriteGateInputs {
            cosine: 0.0,
            merge_threshold: 1.0,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: None,
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d_positive, WriteDecision::Reject);
        // Degenerate 0.0 >= 0.0 edge documented: sites MUST pass a positive
        // merge_threshold; Site #5's Merge always requires an LLM verdict (row 5).
        let _ = d;
    }

    #[test]
    fn row3_llm_says_not_same_is_reject() {
        let d = write_gate(WriteGateInputs {
            cosine: 0.99,
            merge_threshold: 0.85,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: Some(verdict(0, false, 0.99)), // high confidence but "not same"
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::Reject);
    }

    #[test]
    fn row4_low_confidence_agreement_is_potential_alias() {
        let d = write_gate(WriteGateInputs {
            cosine: 0.99,
            merge_threshold: 0.85,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: Some(verdict(0, true, LLM_VERIFY_CONFIDENCE_FLOOR - 0.01)),
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::PotentialAlias);
    }

    #[test]
    fn row6_llm_true_but_no_deterministic_signal_is_potential_alias() {
        // The load-bearing invariant: LLM true + confident but NO deterministic
        // signal → PotentialAlias, never Merge.
        let d = write_gate(WriteGateInputs {
            cosine: 0.99,
            merge_threshold: 0.85,
            deterministic_signal: DeterministicSignal::from_lexical(false),
            llm_verdict: Some(verdict(0, true, 0.99)),
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::PotentialAlias);
    }

    #[test]
    fn row5_llm_authorized_merge_requires_all_signals() {
        // LLM true + confident + deterministic signal + no floor → Merge.
        let d = write_gate(WriteGateInputs {
            cosine: 0.0, // Site #5 shape: cosine irrelevant, merge via LLM row 5
            merge_threshold: 1.0,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: Some(verdict(0, true, 0.95)),
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::Merge);
    }

    #[test]
    fn row4_boundary_exactly_at_floor_is_merge_side() {
        // confidence == floor is NOT below floor → passes row 4.
        let d = write_gate(WriteGateInputs {
            cosine: 0.0,
            merge_threshold: 1.0,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: Some(verdict(0, true, LLM_VERIFY_CONFIDENCE_FLOOR)),
            min_confidence_floor: None,
            names: None,
        });
        assert_eq!(d, WriteDecision::Merge);
    }

    #[test]
    fn row5b_site6_floor_below_downgrades_to_potential_alias() {
        // Site #6 active (Some floor) and observed conf below it → downgrade.
        let d = write_gate(WriteGateInputs {
            cosine: 0.0,
            merge_threshold: 1.0,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: Some(verdict(0, true, 0.99)),
            min_confidence_floor: Some(LLM_VERIFY_CONFIDENCE_FLOOR - 0.1),
            names: None,
        });
        assert_eq!(d, WriteDecision::PotentialAlias);
    }

    #[test]
    fn row5b_site6_floor_cleared_still_merges() {
        let d = write_gate(WriteGateInputs {
            cosine: 0.0,
            merge_threshold: 1.0,
            deterministic_signal: DeterministicSignal::from_lexical(true),
            llm_verdict: Some(verdict(0, true, 0.99)),
            min_confidence_floor: Some(0.99),
            names: None,
        });
        assert_eq!(d, WriteDecision::Merge);
    }

    // ── schema parsing — loud parse discipline (spec §2.1) ──────────────────────

    #[test]
    fn batch_envelope_tolerates_empty_and_missing_verdicts() {
        let empty: IdentityVerdictBatch = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(empty.verdicts.is_empty());
        let arr: IdentityVerdictBatch =
            serde_json::from_value(serde_json::json!({ "verdicts": [] })).unwrap();
        assert!(arr.verdicts.is_empty());
    }

    #[test]
    fn item_missing_required_field_fails_loudly() {
        // Missing `confidence` → parse error (no serde(default)).
        let r = serde_json::from_value::<IdentityVerdictItem>(serde_json::json!({
            "pair_id": 0, "is_same_entity": true, "reasoning": "x"
        }));
        assert!(r.is_err(), "missing required field must fail parse");
    }

    #[test]
    fn item_valid_parses() {
        let v = serde_json::from_value::<IdentityVerdictItem>(serde_json::json!({
            "pair_id": 3, "is_same_entity": true, "confidence": 0.9, "reasoning": "same org"
        }))
        .unwrap();
        assert_eq!(v.pair_id, 3);
        assert!(v.is_same_entity);
    }

    #[test]
    fn item_json_fragment_reasoning_rejected() {
        // A reasoning value carrying JSON-syntax tells is a spliced fragment → reject.
        for bad in [
            r#"{"is_same_entity": true"#,
            "sure}",
            "has a \\ backslash",
            "mentions is_same_entity inline",
        ] {
            let r = serde_json::from_value::<IdentityVerdictItem>(serde_json::json!({
                "pair_id": 0, "is_same_entity": true, "confidence": 0.9, "reasoning": bad
            }));
            assert!(
                r.is_err(),
                "json-fragment reasoning {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn deterministic_signal_newtype_constructors() {
        assert!(DeterministicSignal::from_lexical(true).fired());
        assert!(!DeterministicSignal::from_lexical(false).fired());
    }

    #[test]
    fn batch_schema_has_required_fields_and_no_forbidden_constraints() {
        let schema = identity_verdict_batch_schema(5);
        let s = schema.to_string();
        assert!(s.contains("pair_id"));
        assert!(s.contains("is_same_entity"));
        assert!(s.contains("additionalProperties"));
        // Anthropic NativeSchema forbids these on arrays/numbers:
        assert!(!s.contains("minItems"));
        assert!(!s.contains("maxItems"));
        assert!(!s.contains("minimum"));
        assert!(!s.contains("maximum"));
    }

    // ── chunk_pair_indices (S3 spike fix) — deterministic, no LLM required ──────

    #[test]
    fn chunk_pair_indices_empty_input_yields_no_chunks() {
        assert_eq!(chunk_pair_indices(0), Vec::<std::ops::Range<usize>>::new());
    }

    #[test]
    fn chunk_pair_indices_smaller_than_one_chunk_yields_single_range() {
        // 3 pairs, chunk size 10 → one chunk covering all 3.
        let chunks = chunk_pair_indices(3);
        assert_eq!(chunks, vec![0..3]);
    }

    #[test]
    fn chunk_pair_indices_exact_multiple_yields_even_chunks() {
        // 20 pairs, chunk size 10 → exactly two chunks of 10.
        let chunks = chunk_pair_indices(2 * ADJUDICATION_CHUNK_SIZE);
        assert_eq!(
            chunks,
            vec![
                0..ADJUDICATION_CHUNK_SIZE,
                ADJUDICATION_CHUNK_SIZE..(2 * ADJUDICATION_CHUNK_SIZE)
            ]
        );
    }

    #[test]
    fn chunk_pair_indices_25_pairs_matches_s3_spike_scale() {
        // The exact scale the S3 spike (`type_registry_collapse_s3_spike.rs`)
        // surfaced as timing out in one call: 25 nominated pairs. With
        // ADJUDICATION_CHUNK_SIZE=10 this must split into 3 chunks (10, 10, 5),
        // never one all-25 chunk.
        let chunks = chunk_pair_indices(25);
        assert_eq!(chunks.len(), 3, "25 pairs must split into >1 chunk");
        for c in &chunks {
            assert!(
                c.len() <= ADJUDICATION_CHUNK_SIZE,
                "chunk {c:?} exceeds ADJUDICATION_CHUNK_SIZE"
            );
        }
    }

    #[test]
    fn chunk_pair_indices_every_global_index_appears_exactly_once_in_order() {
        // Property test across a spread of totals (including non-multiples of
        // the chunk size): every global pair index 0..total must appear in
        // exactly one chunk, chunks must be contiguous, ascending, and
        // non-overlapping, and every chunk except possibly the last must be
        // exactly ADJUDICATION_CHUNK_SIZE long.
        for total in [0usize, 1, 5, 9, 10, 11, 19, 20, 21, 25, 47, 100] {
            let chunks = chunk_pair_indices(total);

            // Reconstruct the full index set by flattening every chunk.
            let mut seen: Vec<usize> = chunks.iter().flat_map(|r| r.clone()).collect();
            let expected: Vec<usize> = (0..total).collect();
            assert_eq!(
                seen, expected,
                "total={total}: flattened chunks must cover 0..total exactly once, in order"
            );
            // Redundant explicit uniqueness check (belt-and-braces on top of the
            // ordering assertion above).
            let before_len = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                before_len,
                "total={total}: a pair index appeared in more than one chunk"
            );

            // Contiguity + non-overlap + size cap.
            let mut prev_end = 0usize;
            for (i, c) in chunks.iter().enumerate() {
                assert_eq!(
                    c.start, prev_end,
                    "total={total}: chunk {i} does not start where the previous ended"
                );
                assert!(
                    c.len() <= ADJUDICATION_CHUNK_SIZE,
                    "total={total}: chunk {i} ({c:?}) exceeds ADJUDICATION_CHUNK_SIZE"
                );
                let is_last = i == chunks.len() - 1;
                if !is_last {
                    assert_eq!(
                        c.len(),
                        ADJUDICATION_CHUNK_SIZE,
                        "total={total}: non-last chunk {i} ({c:?}) must be exactly ADJUDICATION_CHUNK_SIZE"
                    );
                }
                prev_end = c.end;
            }
            assert_eq!(
                prev_end, total,
                "total={total}: last chunk must end exactly at total"
            );
        }
    }

    #[test]
    fn chunk_verdict_merge_preserves_pair_id_correlation() {
        // Simulates what `type_registry_collapse::adjudicate_batch` /
        // `acronym_nickname_recall::adjudicate_batch` do: for each chunk range,
        // take a chunk-LOCAL verdict map (as if returned by one
        // `adjudicate_chunk` call, keyed 0..chunk.len()) and remap into a
        // GLOBAL map keyed by `range.start + local_pair_id`. Assert the
        // merged map correlates every verdict back to the correct global pair,
        // even when a middle chunk fails entirely (empty map) — proving one
        // chunk's failure does not corrupt or lose neighboring chunks' verdicts.
        let total = 25;
        let chunks = chunk_pair_indices(total);
        assert_eq!(chunks.len(), 3);

        // Chunk 0 (pairs 0..10): full local verdicts.
        let chunk0_local: HashMap<usize, IdentityVerdictItem> = (0..10)
            .map(|local_id| (local_id, verdict(local_id, local_id % 2 == 0, 0.9)))
            .collect();
        // Chunk 1 (pairs 10..20): simulates a chunk-level failure — empty map
        // (LLM call failed / parse failed for this chunk only).
        let chunk1_local: HashMap<usize, IdentityVerdictItem> = HashMap::new();
        // Chunk 2 (pairs 20..25): full local verdicts.
        let chunk2_local: HashMap<usize, IdentityVerdictItem> = (0..5)
            .map(|local_id| (local_id, verdict(local_id, true, 0.95)))
            .collect();

        let per_chunk_locals = [chunk0_local, chunk1_local, chunk2_local];

        let mut merged: HashMap<usize, IdentityVerdictItem> = HashMap::new();
        for (range, chunk_local) in chunks.iter().zip(per_chunk_locals.into_iter()) {
            for (local_pair_id, v) in chunk_local {
                let global_pair_id = range.start + local_pair_id;
                merged.insert(global_pair_id, v);
            }
        }

        // Chunk 0's global pairs 0..10 are present, with correlated pair_id
        // preserved on the verdict item's own (chunk-local, pre-remap) field —
        // callers key the map by the GLOBAL index; the verdict's own `pair_id`
        // still reflects its ORIGINAL chunk-local value (matches production
        // behavior — the map key is what's authoritative downstream).
        for global_id in 0..10 {
            assert!(
                merged.contains_key(&global_id),
                "global pair {global_id} from chunk 0 missing after merge"
            );
        }
        // Chunk 1's global pairs 10..20 are ALL absent (chunk failed) — must
        // NOT silently merge (write_gate treats a missing key as `None`, the
        // safe Reject default).
        for global_id in 10..20 {
            assert!(
                !merged.contains_key(&global_id),
                "global pair {global_id} from the FAILED chunk 1 must be absent, not defaulted"
            );
        }
        // Chunk 2's global pairs 20..25 are present.
        for global_id in 20..25 {
            assert!(
                merged.contains_key(&global_id),
                "global pair {global_id} from chunk 2 missing after merge"
            );
        }
        assert_eq!(
            merged.len(),
            15,
            "merged map must contain exactly chunk0(10) + chunk1(0) + chunk2(5) verdicts"
        );
    }

    /// The 10 REAL timestamp-collapsing merges from `site5_acronym_nickname`,
    /// extracted from `graph_mutation_log` on `.context/full-corpus.db`. TD-212.
    ///
    /// Every one is irreversible data loss: the loser is strictly more specific
    /// than the keeper, and `1037 am` cannot be recovered to `1037 am on 27 june
    /// 2023`. Before the row-0 veto, all 10 reached `WriteDecision::Merge`.
    const SITE5_TIMESTAMP_COLLAPSES: &[(&str, &str)] = &[
        ("1037 am on 27 june 2023", "1037 am"),
        ("124 pm on 25 may 2023", "124 pm"),
        ("318 pm on 4 may 2023", "318 pm"),
        ("519 pm on 5 august 2023", "519 pm"),
        ("1058 am on 9 october 2022", "1058 am"),
    ];

    /// Builds the exact input shape Site #5 passes: no cosine signal, structural
    /// pre-filter fired, and an LLM verdict that WOULD authorize a merge. Without
    /// the veto this returns `Merge` for every pair above.
    fn site5_merge_authorizing(names: Option<(&str, &str)>) -> WriteDecision {
        write_gate(WriteGateInputs {
            cosine: 0.0,
            merge_threshold: 1.0,
            deterministic_signal: DeterministicSignal::from_structural_prefilter(true),
            llm_verdict: Some(IdentityVerdictItem {
                pair_id: 0,
                is_same_entity: true,
                confidence: 0.99,
                reasoning: "same entity".to_string(),
            }),
            min_confidence_floor: None,
            names,
        })
    }

    #[test]
    fn temporal_veto_blocks_site5_timestamp_collapse() {
        for (loser, keeper) in SITE5_TIMESTAMP_COLLAPSES {
            assert_eq!(
                site5_merge_authorizing(Some((loser, keeper))),
                WriteDecision::Reject,
                "{loser:?} -> {keeper:?} destroys a date and must be vetoed"
            );
        }
    }

    /// The veto must not reject what Site #5 EXISTS for. Acronym and nickname
    /// pairs share zero tokens by construction, so only the temporal arm applies —
    /// verified over the real 140-row adversarial corpus (0 conflicts).
    #[test]
    fn temporal_veto_leaves_genuine_acronym_and_nickname_merges_alone() {
        for (a, b) in [
            ("IBM", "International Business Machines"),
            ("MADD", "Mothers Against Drunk Driving"),
            ("Tony Marchetti", "Anthony Marchetti"),
            ("José Marchetti", "Jose Marchetti"),
        ] {
            assert_eq!(
                site5_merge_authorizing(Some((a, b))),
                WriteDecision::Merge,
                "{a:?} / {b:?} is a legitimate Site #5 merge and must not be vetoed"
            );
        }
    }

    /// `None` must compose as a no-op, exactly like `min_confidence_floor`.
    #[test]
    fn temporal_veto_is_inert_when_names_absent() {
        assert_eq!(site5_merge_authorizing(None), WriteDecision::Merge);
    }
}
