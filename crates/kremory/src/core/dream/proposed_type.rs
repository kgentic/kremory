//! `ProposedType` + `DiscoveryProposalBatch` — structured output for Pass 0
//! type discovery, plus the shape validator (9 rejection categories per
//! ADR-037 §3.1 + TD-010).
//!
//! ## Parse discipline
//!
//! Per [[llm-output-parse-loudly]]:
//! - NO `#[serde(default)]` on required fields.  Missing field → parse error.
//! - `Vec<ProposedType>` wrapper uses `#[serde(default)]` (empty = no proposals,
//!   semantically identical to `Vec::new()`).
//!
//! ## Shape validator (9 categories)
//!
//! [`validate_proposed_name`] rejects `name` values that fail any of the
//! 9 categories and emits `kremory.dream.types_rejected_total{reason}` per
//! rejection so operators can pinpoint model drift.

use metrics::counter;
use schemars::JsonSchema;
use serde::Deserialize;

// ─── Structs emitted by the LLM ───────────────────────────────────────────────

/// Single type proposal emitted by the LLM in Pass 0.
///
/// All fields are required — per [[llm-output-parse-loudly]], NO `#[serde(default)]`
/// on any field.  A missing field is a parse error that triggers the fallback
/// ladder (FormatSchema → LlmJsonRepair), not a silent default.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(crate) struct ProposedType {
    /// Candidate entity type name.  The shape validator rejects garbage (see module doc).
    pub(crate) name: String,
    /// Prose description used to prompt the LLM and for anti-redundancy gate
    /// (description-pair cosine ≥ 0.85 rejects near-duplicates).
    pub(crate) description: String,
    /// Why this type was proposed (evidence justification).  Not used for gate
    /// decisions but persisted in `entity_types.discovered_by` string.
    pub(crate) justification: String,
}

/// Wrapper batch struct — the root object the LLM emits.
///
/// `proposals` uses `#[serde(default)]` because an empty array is semantically
/// correct ("no proposals this call") and is not a parse failure.  Per the
/// split in [[llm-output-parse-loudly]]: wrapper containers may use default.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(crate) struct DiscoveryProposalBatch {
    #[serde(default)]
    pub(crate) proposals: Vec<ProposedType>,
}

// ─── Shape validator ──────────────────────────────────────────────────────────

/// Reason a proposed type name was rejected by the shape validator.
///
/// Each variant maps to the `reason` label in
/// `kremory.dream.types_rejected_total{reason}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RejectionReason {
    /// Name contains `"`, `\`, `{`, `}`, `<`, `>` — likely a JSON fragment.
    NameJsonFragment,
    /// Name contains a control character below 0x20 (excluding `\t` / `\n`).
    ControlChar,
    /// Name is a repeated-punctuation placeholder: `"..."`, `"...."`, `"???"`,
    /// `"---"`, `"___"` (and similar).
    EllipsisPlaceholder,
    /// Name is a low-information placeholder word (case-insensitive):
    /// TBD, N/A, None, Unknown, Other, Misc, Various, Unclear.
    LowInformation,
    /// Trimmed name is shorter than 3 characters.
    TooShort,
    /// Trimmed name is longer than 50 characters.
    TooLong,
    /// Every character in the name fails `is_alphanumeric`.
    AllPunctuation,
    /// Name consists entirely of digits (matches `^\d+$`).
    NumericOnly,
    /// Name is blank / whitespace only (matches `^\s*$`).
    WhitespaceOnly,
}

impl RejectionReason {
    /// Static label used in metrics dimension.  Stable — never rename without
    /// also updating dashboards that key on this string.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NameJsonFragment => "name_json_fragment",
            Self::ControlChar => "control_char",
            Self::EllipsisPlaceholder => "ellipsis_placeholder",
            Self::LowInformation => "low_information",
            Self::TooShort => "too_short",
            Self::TooLong => "too_long",
            Self::AllPunctuation => "all_punctuation",
            Self::NumericOnly => "numeric_only",
            Self::WhitespaceOnly => "whitespace_only",
        }
    }
}

/// Low-information placeholder words rejected case-insensitively.
const LOW_INFO_NAMES: &[&str] = &[
    "tbd", "n/a", "none", "unknown", "other", "misc", "various", "unclear",
];

/// Repeated-punctuation placeholder patterns (exact, lower-cased comparison).
const ELLIPSIS_PATTERNS: &[&str] = &["...", "....", "???", "---", "___"];

/// Validate a proposed type `name` against the 9 rejection categories.
///
/// Returns `Ok(())` when the name passes all checks, or
/// `Err(RejectionReason)` with the FIRST failing category.
///
/// Each rejection increments `kremory.dream.types_rejected_total{reason, namespace}`
/// before returning (observability is built-in at the check boundary, not deferred).
///
/// `namespace` is the metrics label value; pass the group_id string.
pub(crate) fn validate_proposed_name(name: &str, namespace: &str) -> Result<(), RejectionReason> {
    let trimmed = name.trim();

    // 1. Whitespace-only (must come before length check)
    if trimmed.is_empty() {
        emit_rejected(RejectionReason::WhitespaceOnly, namespace);
        return Err(RejectionReason::WhitespaceOnly);
    }

    // 2. Control characters (< 0x20 except \t and \n)
    if name
        .chars()
        .any(|c| (c as u32) < 0x20 && c != '\t' && c != '\n')
    {
        emit_rejected(RejectionReason::ControlChar, namespace);
        return Err(RejectionReason::ControlChar);
    }

    // 3. JSON fragment characters
    if name.contains('"')
        || name.contains('\\')
        || name.contains('{')
        || name.contains('}')
        || name.contains('<')
        || name.contains('>')
    {
        emit_rejected(RejectionReason::NameJsonFragment, namespace);
        return Err(RejectionReason::NameJsonFragment);
    }

    // 4. Too short (trimmed < 3 chars)
    if trimmed.len() < 3 {
        emit_rejected(RejectionReason::TooShort, namespace);
        return Err(RejectionReason::TooShort);
    }

    // 5. Too long (trimmed > 50 chars)
    if trimmed.len() > 50 {
        emit_rejected(RejectionReason::TooLong, namespace);
        return Err(RejectionReason::TooLong);
    }

    // 6. Numeric only (all chars are ASCII digits)
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        emit_rejected(RejectionReason::NumericOnly, namespace);
        return Err(RejectionReason::NumericOnly);
    }

    // 7. Ellipsis / repeated-punctuation placeholder (case-insensitive exact match).
    // Must precede AllPunctuation so "---" / "..." / "???" emit the more specific
    // reason rather than falling through to the generic all-punctuation check.
    let lower = trimmed.to_lowercase();
    if ELLIPSIS_PATTERNS.contains(&lower.as_str()) {
        emit_rejected(RejectionReason::EllipsisPlaceholder, namespace);
        return Err(RejectionReason::EllipsisPlaceholder);
    }

    // 8. All punctuation (every char fails is_alphanumeric)
    if trimmed.chars().all(|c| !c.is_alphanumeric()) {
        emit_rejected(RejectionReason::AllPunctuation, namespace);
        return Err(RejectionReason::AllPunctuation);
    }

    // 9. Low-information placeholder word (case-insensitive)
    if LOW_INFO_NAMES.contains(&lower.as_str()) {
        emit_rejected(RejectionReason::LowInformation, namespace);
        return Err(RejectionReason::LowInformation);
    }

    Ok(())
}

/// Emit the per-reason rejection counter.  Called exactly once per rejection,
/// at the point of detection — built-in observability per [[observability-first-class]].
fn emit_rejected(reason: RejectionReason, namespace: &str) {
    counter!(
        "kremory.dream.types_rejected_total",
        "reason" => reason.as_str(),
        "namespace" => namespace.to_string()
    )
    .increment(1);
}

// ─── JSON Schema for StructuredCallBuilder ────────────────────────────────────

/// Compute the JSON Schema for `DiscoveryProposalBatch`.
///
/// Returns `Ok(schema)` on success.  `schemars::schema_for!` is derived and
/// cannot fail in practice, but `serde_json::to_value` is fallible by type
/// signature — propagate via `?` so callers can use the `Result` path cleanly
/// rather than masking the error with `.expect()`.
pub(crate) fn discovery_proposal_schema() -> Result<serde_json::Value, serde_json::Error> {
    let schema = schemars::schema_for!(DiscoveryProposalBatch);
    serde_json::to_value(schema)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(name: &str) {
        validate_proposed_name(name, "test_ns").expect(name);
    }

    fn rejected(name: &str, expected: RejectionReason) {
        let err = validate_proposed_name(name, "test_ns").expect_err(name);
        assert_eq!(err, expected, "name={name:?}");
    }

    #[test]
    fn passes_valid_names() {
        ok("ProductSKU");
        ok("LegalPrecedent");
        ok("SoftwareModule");
        ok("Drug");
        ok("BiomarkerPanel");
    }

    #[test]
    fn rejects_whitespace_only() {
        rejected("", RejectionReason::WhitespaceOnly);
        rejected("   ", RejectionReason::WhitespaceOnly);
    }

    #[test]
    fn rejects_json_fragment_chars() {
        rejected(r#"{"entity":"Foo"}"#, RejectionReason::NameJsonFragment);
        rejected("Type\\n", RejectionReason::NameJsonFragment);
        rejected("<html>", RejectionReason::NameJsonFragment);
    }

    #[test]
    fn rejects_too_short() {
        rejected("AB", RejectionReason::TooShort);
        rejected("A", RejectionReason::TooShort);
    }

    #[test]
    fn rejects_too_long() {
        let long = "A".repeat(51);
        rejected(&long, RejectionReason::TooLong);
    }

    #[test]
    fn rejects_numeric_only() {
        rejected("123", RejectionReason::NumericOnly);
        rejected("007", RejectionReason::NumericOnly);
    }

    #[test]
    fn rejects_all_punctuation() {
        rejected("---", RejectionReason::EllipsisPlaceholder); // caught by ellipsis first
        rejected("!!!", RejectionReason::AllPunctuation);
        rejected("...", RejectionReason::EllipsisPlaceholder); // ellipsis wins
        rejected("@#$", RejectionReason::AllPunctuation);
    }

    #[test]
    fn rejects_ellipsis_patterns() {
        rejected("...", RejectionReason::EllipsisPlaceholder);
        rejected("....", RejectionReason::EllipsisPlaceholder);
        rejected("???", RejectionReason::EllipsisPlaceholder);
        rejected("___", RejectionReason::EllipsisPlaceholder);
    }

    #[test]
    fn rejects_low_information_case_insensitive() {
        rejected("Unknown", RejectionReason::LowInformation);
        rejected("UNKNOWN", RejectionReason::LowInformation);
        rejected("tbd", RejectionReason::LowInformation);
        rejected("N/A", RejectionReason::LowInformation);
        rejected("other", RejectionReason::LowInformation);
        rejected("Misc", RejectionReason::LowInformation);
        rejected("Various", RejectionReason::LowInformation);
        rejected("Unclear", RejectionReason::LowInformation);
        rejected("None", RejectionReason::LowInformation);
    }

    #[test]
    fn rejects_control_chars() {
        rejected("Type\x01Name", RejectionReason::ControlChar);
        rejected("Bad\x00Val", RejectionReason::ControlChar);
    }

    #[test]
    fn schema_is_valid_json() {
        let v = discovery_proposal_schema().expect("schema serialization");
        assert!(v.is_object(), "schema must be a JSON object");
    }

    #[test]
    fn batch_deserializes_from_json() {
        let json = r#"{"proposals":[{"name":"ProductSKU","description":"A product identifier","justification":"Many entities had SKU codes"}]}"#;
        let batch: DiscoveryProposalBatch = serde_json::from_str(json).expect("deserialize batch");
        assert_eq!(batch.proposals.len(), 1);
        assert_eq!(batch.proposals[0].name, "ProductSKU");
    }

    #[test]
    fn batch_missing_required_name_fails_parse() {
        // name is required — missing it must fail, not default
        let json =
            r#"{"proposals":[{"description":"A desc","justification":"justification here"}]}"#;
        let result: Result<DiscoveryProposalBatch, _> = serde_json::from_str(json);
        assert!(result.is_err(), "missing `name` must fail parse");
    }

    #[test]
    fn batch_empty_proposals_is_ok() {
        // proposals is a Vec<> — empty array is correct (no proposals)
        let json = r#"{"proposals":[]}"#;
        let batch: DiscoveryProposalBatch =
            serde_json::from_str(json).expect("empty proposals is valid");
        assert!(batch.proposals.is_empty());
    }

    #[test]
    fn batch_missing_proposals_field_defaults_empty() {
        // proposals has #[serde(default)] — missing = empty vec
        let json = r#"{}"#;
        let batch: DiscoveryProposalBatch =
            serde_json::from_str(json).expect("missing proposals defaults to empty");
        assert!(batch.proposals.is_empty());
    }
}
