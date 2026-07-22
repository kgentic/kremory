//! Serde models for LLM JSON output coercion.
//!
//! All `Raw*` structs, label helpers, and hybrid typing wrappers used across
//! the extraction submodules. Split from `mod.rs` as part of TD-001 (E0-B).

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use serde::Deserialize;

// ─── Serde helper: accept string or array ───────────────────────────────────

/// Deserialize a JSON string or array into a `String`.
///
/// Some LLMs (e.g. `llama3.2:3b`) emit arrays where the schema expects
/// a scalar: `"object": ["Python", "Rust", "Julia"]`.  This helper joins
/// array elements with `", "` so the extracted fact is still useful.
pub(super) fn deser_string_or_array<'de, D>(
    deserializer: D,
) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct StringOrArray;

    impl<'de> Visitor<'de> for StringOrArray {
        type Value = String;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "a string or an array of strings")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<String, E> {
            Ok(v.to_owned())
        }

        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<String, E> {
            Ok(v)
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<String, A::Error> {
            let mut parts: Vec<String> = Vec::new();
            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                match elem {
                    serde_json::Value::String(s) => parts.push(s),
                    other => parts.push(other.to_string()),
                }
            }
            Ok(parts.join(", "))
        }

        fn visit_unit<E: de::Error>(self) -> std::result::Result<String, E> {
            Ok(String::new())
        }

        fn visit_none<E: de::Error>(self) -> std::result::Result<String, E> {
            Ok(String::new())
        }
    }

    deserializer.deserialize_any(StringOrArray)
}

/// Default empty string for `#[serde(default)]` + `deser_string_or_array` fields.
pub(super) fn default_string() -> String {
    String::new()
}

// ─── Serde models for LLM JSON output coercion ──────────────────────────────

/// Top-level LLM extraction output. Both fields are optional — LLMs may omit one.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct LlmExtractionOutput {
    #[serde(default)]
    pub(crate) entities: Vec<RawEntity>,
    #[serde(default)]
    pub(crate) relationships: Vec<RawRelationship>,
}

/// Entity as emitted by the LLM. `label` defaults to "Entity" when missing.
///
/// `label` uses `deser_string_or_array` to tolerate LLMs (e.g. qwen2.5:14b) that
/// emit `"label": ["Person"]` instead of `"label": "Person"`.
/// Excluded from schema: uses `deser_string_or_array` (string|array) — routes to llm_json fallback.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawEntity {
    #[serde(default)]
    pub(crate) name: String,
    /// Excluded from schema: uses `deser_string_or_array` (string|array) — routes to llm_json fallback.
    #[serde(
        default = "default_entity_label",
        deserialize_with = "deser_string_or_array"
    )]
    #[schemars(skip)]
    pub(crate) label: String,
}

pub(super) fn default_entity_label() -> String {
    "Entity".to_string()
}

// ─── L2: Runtime label allowlist ────────────────────────────────────────────

/// Canonical-form examples for common entity types. NOT an exhaustive list —
/// the system accepts any structurally-valid label (see `is_canonical_entity_type`).
/// This array exists only for prompt interpolation + test fixtures + alias-map
/// canonical-form reference.
///
/// TD-013 PR1-corrected (2026-06-03): the ENTITY_TYPE_ALLOWLIST positive
/// hard-reject mechanism was DELETED. It was an ecosystem outlier — no peer
/// agentic-memory system (Graphiti, Mem0, LightRAG, Cognee, LlamaIndex)
/// uses a positive allowlist. Graphiti uses regex-only Cypher-safety
/// validation; Cognee uses unconstrained `type: str`. kremory now follows
/// Graphiti's pattern: structural-validity check + placeholder-reject only.
/// See ADR adr-td-013-graph-quality-remediation-2026-06-03 (PR1-corrected
/// amendment) and `feedback_vera_challenges_contents_user_challenges_mechanism`.
pub const ENTITY_TYPE_CANONICAL_FORMS: &[&str] = &[
    // Core NER types (referenced in extraction prompts as guidance, not constraint)
    "Person",
    "Organisation",
    "Location",
    "Technology",
    "Product",
    "Event",
    "Date",
    "Time",
    "Money",
    "Quantity",
    "Percent",
];

/// Cypher-safe identifier regex (Graphiti pattern from
/// `graphiti_core/helpers.py:35`): `^[A-Za-z_][A-Za-z0-9_]*$`.
///
/// kremory's libsql backend doesn't have Cypher injection risk, but this
/// pattern serves as a structural-validity check: rejects whitespace, leading
/// digits, special chars, all-punctuation strings, and overly-long labels.
pub(crate) fn label_is_structurally_valid(label: &str) -> bool {
    let trimmed = label.trim();
    if trimmed.len() < 2 || trimmed.len() > 64 {
        return false;
    }
    let mut chars = trimmed.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ' ')
        && trimmed.chars().any(|c| c.is_ascii_alphabetic())
}

/// Normalize an LLM-emitted label to canonical form via alias map + case fix.
///
/// Examples: "ORGANIZATION"/"organisation"/"Org"/"Company" → "Organisation".
/// Returns the input trimmed if no alias matches (preserves novel labels).
pub fn normalize_label(raw: &str) -> String {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "person" | "people" | "human" | "individual" | "per" => "Person".to_string(),
        "organisation" | "organization" | "org" | "company" | "corporation" | "corp" => {
            "Organisation".to_string()
        }
        "location" | "place" | "loc" | "gpe" => "Location".to_string(),
        "technology" | "tech" => "Technology".to_string(),
        "product" => "Product".to_string(),
        "event" => "Event".to_string(),
        "date" => "Date".to_string(),
        "time" => "Time".to_string(),
        "money" | "amount" | "currency" => "Money".to_string(),
        "quantity" | "number" => "Quantity".to_string(),
        "percent" | "percentage" => "Percent".to_string(),
        _ => {
            // Title-case any single ASCII-alphabetic word so "court" → "Court",
            // "software" → "Software" without needing an explicit alias entry.
            // Multi-word and non-alphabetic labels pass through trimmed.
            if trimmed.chars().all(|c| c.is_ascii_alphabetic() || c == '_') && !trimmed.is_empty() {
                let mut chars = trimmed.chars();
                let Some(first_char) = chars.next() else {
                    return trimmed.to_string();
                };
                let first = first_char.to_ascii_uppercase();
                let rest: String = chars.map(|c| c.to_ascii_lowercase()).collect();
                format!("{first}{rest}")
            } else {
                trimmed.to_string()
            }
        }
    }
}

/// Returns `true` when `label` passes structural validity + placeholder-reject.
///
/// TD-013 PR1-corrected (2026-06-03): this function's semantics CHANGED from
/// "positive allowlist match" to "structural validity + placeholder reject".
/// Now accepts ANY label that is (a) not empty / "Entity" / "UNKNOWN", and
/// (b) matches a Cypher-safe identifier pattern. Aligns with Graphiti's
/// `validate_node_labels` (graphiti_core/helpers.py:174-186) and ecosystem
/// consensus. The function name preserves API compatibility with existing
/// call sites; the BEHAVIOR is now ecosystem-aligned.
///
/// Placeholder labels ("Entity", "UNKNOWN", empty) ALWAYS return `false` —
/// they indicate LLM classification failure (TD-012 protection preserved).
pub fn is_canonical_entity_type(label: &str) -> bool {
    let trimmed = label.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("entity")
        || trimmed.eq_ignore_ascii_case("unknown")
    {
        return false;
    }
    label_is_structurally_valid(trimmed)
}

/// Relationship triplet as emitted by the LLM.
///
/// `subject`, `predicate`, `object` use `deser_string_or_array` to tolerate
/// LLMs (e.g. llama3.2:3b) that emit arrays instead of scalar strings.
///
/// Schema note: `subject`/`predicate`/`object` are excluded from the derived
/// JSON Schema (`#[schemars(skip)]`) because `deser_string_or_array` accepts
/// both `String` and `array` inputs, which is incompatible with schemars's
/// field-level schema inference. Any structured-output path using this struct
/// must route through `llm_json` fallback parsing rather than schema enforcement.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawRelationship {
    /// Excluded from schema: uses `deser_string_or_array` (string|array) — routes to llm_json fallback.
    #[serde(default = "default_string", deserialize_with = "deser_string_or_array")]
    #[schemars(skip)]
    pub(crate) subject: String,
    /// Excluded from schema: uses `deser_string_or_array` (string|array) — routes to llm_json fallback.
    #[serde(default = "default_string", deserialize_with = "deser_string_or_array")]
    #[schemars(skip)]
    pub(crate) predicate: String,
    /// Excluded from schema: uses `deser_string_or_array` (string|array) — routes to llm_json fallback.
    #[serde(default = "default_string", deserialize_with = "deser_string_or_array")]
    #[schemars(skip)]
    pub(crate) object: String,
    #[serde(default)]
    pub(crate) is_entity_ref: bool,
    #[serde(default = "default_confidence")]
    pub(crate) confidence: f64,
}

pub(crate) fn default_confidence() -> f64 {
    1.0
}

/// Entity array as emitted by the IntegerIdLlmExtractor (stage 1).
///
/// `label` uses `deser_string_or_array` to tolerate LLMs (e.g. qwen2.5:14b) that
/// emit `"label": ["Person"]` instead of `"label": "Person"`.
///
/// Used by schemars reflection in `schemas.rs` (`EntityListWrapper`, `EntityOnlyOutput`)
/// and by `parse_entities` (legacy string-label test fixture).  Rustc dead-code
/// analysis cannot see schemars reflection or `#[cfg(test)]` callers from a
/// production-code vantage point, so the item-level allow is correct here.
#[allow(dead_code)]
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawEntitySimple {
    #[serde(default)]
    pub(crate) name: String,
    /// Excluded from schema: uses `deser_string_or_array` (string|array) — routes to llm_json fallback.
    #[serde(default, deserialize_with = "deser_string_or_array")]
    #[schemars(skip)]
    pub(crate) label: String,
}

// ─── L1: Integer-ID classification structs (TD-013) ─────────────────────────

/// Entity as emitted by the LLM in the integer-ID path (TD-013 L1).
///
/// The LLM emits `entity_type_id` — an integer constrained at decode time via
/// JSON schema `enum` to the registered type IDs for this namespace.  Grammar
/// enforcement (Ollama FormatSchema / Anthropic NativeSchema) prevents any
/// out-of-range value from reaching the application; `validate_or_fallback`
/// provides a second safety net for the PromptOnly fallback arm.
///
/// Spike 1c/1d (2026-06-03) confirmed: qwen2.5:14b emits id=0 (catch-all)
/// even when prompted to bypass — grammar physically prevents out-of-range.
// NOTE: `#[serde(default)]` is intentionally REMOVED from `name` and `entity_type_id`
// (2026-06-04). Previously defaults swallowed truncated LLM output: a fragmented
// post-repair JSON like `[{"name":"Boston\", \"entity_type_id\":3}, {"}]` deserialized
// as a single entity with name=<garbage> and entity_type_id=0 (the default), collapsing
// all extractions to label="Entity". Without defaults, missing fields fail loudly so
// the fallback ladder (LlmJsonRepair → DelimitedTuple → PromptOnly) can retry.
//
// `confidence` is the sole explicit exception per migration-010 spec §1.4:
// missing confidence is semantically valid (LLM-only path does not emit it),
// NOT a parse failure. `Option<f32>` + `#[serde(default)]` models "absent" correctly.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawEntityIntegerId {
    pub(crate) name: String,
    /// Integer entity type id constrained to the active namespace registry.
    /// id=0 = "Entity" catch-all; id ≥ 1 = user-defined types.
    pub(crate) entity_type_id: u32,
    /// GLiNER / NER span confidence score (Phase 1 only). Absent on LLM-only extraction paths.
    /// `#[serde(default)]` is intentional: missing = None, not a parse error (spec §1.4).
    #[serde(default)]
    pub(crate) confidence: Option<f32>,
}

/// Wrapper for the integer-ID entity list (TD-013 L1).
///
/// Root key `entities` matches the spike 1c grammar shape and aligns with
/// Graphiti's extraction output format.  Distinct from `EntityListWrapper`
/// (which uses `items`) so the two paths are independent.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct EntityListIntegerWrapper {
    #[serde(default)]
    pub(crate) entities: Vec<RawEntityIntegerId>,
}

// ─── TD-023 Hybrid typing schema (index-based) ───────────────────────────────
//
// Per [[load-bearing-invariants-at-emit-not-prompt]] the candidate-to-typing
// link is enforced STRUCTURALLY via a bounded integer index, NOT via name
// preservation. Avoids fuzzy-match band-aids in the parser.
//
// Required fields, NO #[serde(default)] per [[llm-output-parse-loudly]] —
// missing field = parse error so the fallback ladder can retry.

#[cfg_attr(not(feature = "ner"), allow(dead_code))]
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawHybridTyping {
    /// 0-based candidate index from the input list, bounded by schema enum.
    pub(crate) idx: u32,
    /// Integer entity type id, bounded by the active namespace registry.
    pub(crate) entity_type_id: u32,
}

#[cfg_attr(not(feature = "ner"), allow(dead_code))]
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct HybridTypingWrapper {
    #[serde(default)]
    pub(crate) typings: Vec<RawHybridTyping>,
}

/// Fact triplet as emitted by the IntegerIdLlmExtractor (stage 3).
///
/// `subject`/`predicate`/`object` carry **no** `#[serde(default)]` (TD-133
/// PREVENTION, per `llm-output-parse-loudly`): all three are content fields
/// that are always consumed downstream — `subject` → `subject_id`, `object` →
/// `object_id` (entity ref) or `object_value` (literal) in
/// `ingest/pipeline/{deferred,ingest_with}.rs`. kremory has NO unary/object-less
/// fact path, and an empty content field is filtered out as invalid. A default
/// would let a triplet with a MISSING field deserialise with `""` and then be
/// silently dropped — hiding the LLM misbehaviour that caused the empty
/// benchmark fact graph. Without the default, a missing field is a loud parse
/// error: the whole payload fails, `json_parse_fail` increments, and the
/// fallback ladder (FormatSchema→LlmJsonRepair→DelimitedTuple→PromptOnly)
/// retries — the same pattern as `BatchedNodeResolution`'s inner fields. A
/// present-but-empty field (`"object":""`) still deserialises and is filtered
/// by the empty-field guard in `parse_facts` — only a *missing* field fails.
///
/// `is_entity_ref` keeps its default: a missing hint means "treat object as a
/// literal value", a graceful non-corrupting default (no data loss).
/// `confidence` keeps its default: optional metadata per the rule.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawFact {
    pub(crate) subject: String,
    pub(crate) predicate: String,
    pub(crate) object: String,
    #[serde(default)]
    pub(crate) is_entity_ref: bool,
    #[serde(default = "default_confidence")]
    pub(crate) confidence: f64,
}

// ─── A6: confidence field round-trip (DoD, spec §1.4) ────────────────────────
//
// RawEntityIntegerId is pub(crate) so these live here, not in integration tests.
#[cfg(test)]
mod tests_a6_confidence_round_trip {
    use super::RawEntityIntegerId;

    #[test]
    fn confidence_with_value() {
        let json = r#"{"name":"Alice","entity_type_id":1,"confidence":0.9}"#;
        let parsed: RawEntityIntegerId =
            serde_json::from_str(json).expect("parse with confidence must succeed");
        assert_eq!(parsed.name, "Alice");
        assert_eq!(parsed.entity_type_id, 1);
        let conf = parsed.confidence.expect("confidence must be Some(0.9)");
        assert!(
            (conf - 0.9_f32).abs() < 1e-5,
            "confidence must be ~0.9; got {conf}"
        );
    }

    #[test]
    fn confidence_missing_is_none() {
        // Missing confidence field must deserialise to None, NOT a parse error.
        // This is the explicit exception to the loud-parse rule (spec §1.4).
        let json = r#"{"name":"Bob","entity_type_id":2}"#;
        let parsed: RawEntityIntegerId =
            serde_json::from_str(json).expect("parse without confidence must succeed");
        assert_eq!(parsed.name, "Bob");
        assert_eq!(parsed.entity_type_id, 2);
        assert!(
            parsed.confidence.is_none(),
            "missing confidence must be None; got {:?}",
            parsed.confidence
        );
    }

    #[test]
    fn confidence_explicit_null_is_none() {
        let json = r#"{"name":"Carol","entity_type_id":3,"confidence":null}"#;
        let parsed: RawEntityIntegerId =
            serde_json::from_str(json).expect("parse with null confidence must succeed");
        assert!(
            parsed.confidence.is_none(),
            "null confidence must be None; got {:?}",
            parsed.confidence
        );
    }

    #[test]
    fn name_required_loud_parse() {
        // Missing `name` must fail loudly — no #[serde(default)] on required fields.
        let json = r#"{"entity_type_id":1}"#;
        let result = serde_json::from_str::<RawEntityIntegerId>(json);
        assert!(
            result.is_err(),
            "missing `name` must be a parse error; got: {result:?}"
        );
    }

    #[test]
    fn entity_type_id_required_loud_parse() {
        // Missing `entity_type_id` must fail loudly.
        let json = r#"{"name":"Dave"}"#;
        let result = serde_json::from_str::<RawEntityIntegerId>(json);
        assert!(
            result.is_err(),
            "missing `entity_type_id` must be a parse error; got: {result:?}"
        );
    }
}
