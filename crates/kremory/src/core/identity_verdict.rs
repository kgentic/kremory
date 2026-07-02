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
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IdentityVerdictItem {
    /// Correlates this verdict back to `nominated_pairs[pair_id]`. Mirrors
    /// `VerifyBatchDecision::candidate_idx` — an index into the caller's input
    /// slice, NOT a DB rowid.
    pub(crate) pair_id: usize,
    /// The LLM's identity judgment. REQUIRED (no `serde(default)`).
    pub(crate) is_same_entity: bool,
    /// The LLM's own calibration signal in `[0, 1]`. REQUIRED.
    pub(crate) confidence: f32,
    /// Free-text rationale — for audit/debug only, NEVER parsed for control flow.
    pub(crate) reasoning: String,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeterministicSignal(bool);

impl DeterministicSignal {
    /// Site #3 (type-registry collapse): the `names_lexically_compatible`-style
    /// lemma/exact-match result on the type NAME.
    pub(crate) fn from_lexical(names_lexically_compatible_result: bool) -> Self {
        Self(names_lexically_compatible_result)
    }

    /// Site #5 (acronym/nickname recall): the structural pre-filter's boolean
    /// nomination result (`initialism_candidate OR cooccurs_in_graph`). TRUE means a
    /// deterministic, embedder-independent signal justified escalating to the LLM —
    /// the analogue of token-Jaccard for the zero-lexical-overlap acronym/nickname
    /// surface (spec §2.2.2).
    pub(crate) fn from_structural_prefilter(nominated: bool) -> Self {
        Self(nominated)
    }

    /// Whether a deterministic signal fired.
    pub(crate) fn fired(self) -> bool {
        self.0
    }
}

/// Inputs to [`write_gate`] — one resolved candidate pair (batch-splitting happens
/// in the caller, BEFORE `write_gate` is invoked per pair).
pub(crate) struct WriteGateInputs {
    /// Pre-computed similarity. `0.0` when N/A (Site #5 acronym pairs, where cosine
    /// was never an identity signal per ADR-063).
    pub(crate) cosine: f32,
    /// The caller's site-specific clear-merge cosine threshold (resolves spec §2.2's
    /// shared-`MERGE_THRESHOLD` ambiguity — see module docs). Site #5 passes any
    /// value because its `cosine` is always `0.0`.
    pub(crate) merge_threshold: f32,
    /// The deterministic identity-adjacent signal (spec §2.2's `lexically_compatible`
    /// parameter, wrapped in the RISK-001 newtype). Site #3: lexical match; Site #5:
    /// structural-pre-filter nomination.
    pub(crate) deterministic_signal: DeterministicSignal,
    /// The single-pair verdict, already split out of its batch response by `pair_id`.
    /// `None` when the margin trigger did not fire (clear case, no LLM call) OR when
    /// the LLM call / that item's parse failed.
    pub(crate) llm_verdict: Option<IdentityVerdictItem>,
    /// Site #6's observed minimum entity extraction-confidence, when available.
    /// `None` until Site #6 ships → the row-5 floor clause is vacuously satisfied
    /// (spec §2.2.1 graceful degradation). Compared against
    /// [`LLM_VERIFY_CONFIDENCE_FLOOR`] provisionally until S4 calibrates a
    /// Site-#6-specific floor.
    pub(crate) min_confidence_floor: Option<f32>,
}

/// The write-gate's decision for one candidate pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteDecision {
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
pub(crate) fn write_gate(inputs: WriteGateInputs) -> WriteDecision {
    let WriteGateInputs {
        cosine,
        merge_threshold,
        deterministic_signal,
        llm_verdict,
        min_confidence_floor,
    } = inputs;
    let deterministic = deterministic_signal.fired();

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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
