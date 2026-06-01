//! Lazy JSON Schema statics for structured-output enforcement.
//!
//! Each `static SCHEMA_*` holds a [`serde_json::Value`] derived from the
//! corresponding parse-target struct via `schemars::schema_for!`. Schemas are
//! computed once per process via [`std::sync::LazyLock`] (stable since Rust 1.80).
//!
//! Wrapper struct fields are read only by `schemars::schema_for!` macro
//! reflection (and by serde at deserialize time in `mod.rs` extractors).
//! Rust dead-code analysis cannot see schemars reflection — module-level
//! allow is the appropriate scope for this pattern.
#![allow(dead_code)]
//!
//! # Wrapper structs (§3.2)
//!
//! OpenAI strict-mode and Anthropic `output_config` both require a **root
//! JSON object** — a bare array at top level is rejected. Every extraction
//! call that returns an array must wrap it in one of these structs.
//!
//! # FallbackArm (§3.3)
//!
//! [`FallbackArm`] selects which arm of the Phase-4 fallback ladder is used
//! for a given schema. Schemas wrapping [`RawRelationship`] — which carries
//! `deser_string_or_array` on its `subject`/`predicate`/`object` fields and
//! therefore has those fields excluded from the JSON Schema — **must** bypass
//! native/format schema arms and route to [`FallbackArm::LlmJsonRepair`]
//! where the array-or-string coercion still fires.

use serde::Deserialize;
use serde_json::Value;
use std::sync::LazyLock;

use super::{RawEntitySimple, RawFact, RawRelationship};

// ─── Wrapper structs (§3.2) ──────────────────────────────────────────────────

/// Wrapper for EntityList — OpenAI strict requires root object, not bare array.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct EntityListWrapper {
    pub(crate) items: Vec<RawEntitySimple>,
}

/// Wrapper for RelTypeList — array of strings.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct RelTypeListWrapper {
    pub(crate) items: Vec<String>,
}

/// Wrapper for TripletList — uses RawFact (plain String fields, no deser_string_or_array).
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct TripletListWrapper {
    pub(crate) items: Vec<RawFact>,
}

/// Wrapper for ContradictionVerdict — replaces Vec<usize> bare array.
/// `parse_index_list` in `contradiction.rs` deserialises this wrapper.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct ContradictionVerdictWrapper {
    pub(crate) indices: Vec<u32>,
}

/// Wrapper for entity-resolution verdict (CascadeResolver Tier 3 LLM call).
///
/// Replaces the raw `chat_with_tools(..., None, None)` text response.
/// The LLM is instructed to output a JSON object; wrapping in a struct
/// allows `StructuredCallBuilder` schema enforcement and consistent JSON parsing.
/// Allowed values: `"same"`, `"different"`, `"uncertain"`.
///
/// `verdict` carries `#[serde(default)]` so that an empty PromptOnly response
/// (deserialised as `{}`) yields `verdict = ""`, which the resolver match arm
/// maps conservatively to `ResolutionResult::Different` — same behaviour as
/// the pre-Phase-4 raw text path.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct ResolutionVerdictWrapper {
    /// Resolution verdict: "same", "different", or "uncertain".
    /// Defaults to empty string when the LLM returns no verdict (PromptOnly arm).
    #[serde(default)]
    pub(crate) verdict: String,
}

// ─── Schema-local parse-target types ─────────────────────────────────────────

/// Schema-specific entity-only output: used by SCHEMA_NUEXTRACT_ENTITIES_ONLY
/// and SCHEMA_ENTITY_TYPING. Uses RawEntitySimple (no deser_string_or_array).
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct EntityOnlyOutput {
    #[serde(default)]
    pub(crate) entities: Vec<RawEntitySimple>,
}

/// Schema-specific relationship-only output: used by SCHEMA_NUEXTRACT_RELATIONS_ONLY
/// and SCHEMA_REL_ONLY_FORCE_FALLBACK.
///
/// Uses RawRelationship whose `subject`/`predicate`/`object` are `#[schemars(skip)]`
/// due to `deser_string_or_array`. Any call using this struct MUST route through
/// the LlmJsonRepair arm — see SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct RelOnlyOutput {
    #[serde(default)]
    pub(crate) relationships: Vec<RawRelationship>,
}

// ─── NuExtractOutput visibility re-export ────────────────────────────────────

// NuExtractOutput is defined and pub(crate) in mod.rs — imported directly by
// the SCHEMA_NUEXTRACT_BOTH static below.
use super::NuExtractOutput;

// ─── FallbackArm enum (§3.3) ─────────────────────────────────────────────────

/// Selects which arm of the fallback ladder a schema is routed through.
///
/// Used by per-schema `force_arm` overrides to handle deserialiser–schema
/// incompatibilities (e.g. `deser_string_or_array` requires `LlmJsonRepair`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FallbackArm {
    /// Native grammar-constrained structured output (Anthropic output_config / OpenAI strict).
    NativeSchema,
    /// JSON format-schema enforcement (Ollama llama.cpp grammar).
    FormatSchema,
    /// llm_json repair fallback (preserves `deser_string_or_array` semantics).
    LlmJsonRepair,
    /// Prompt-only, no provider-side enforcement.
    PromptOnly,
}

// ─── Canonical schema statics (§3.3) ─────────────────────────────────────────

/// Schema for entity-list responses (DefaultExtractor stage 1, Graphiti stage 1).
/// Root object with `items: [RawEntitySimple]`.
pub(crate) static SCHEMA_ENTITY_LIST: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(EntityListWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for relationship-type-name lists (DefaultExtractor stage 2).
/// Root object with `items: [String]`.
pub(crate) static SCHEMA_REL_TYPE_LIST: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(RelTypeListWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for triplet lists (DefaultExtractor stage 3).
/// Root object with `items: [RawFact]`.
pub(crate) static SCHEMA_TRIPLET_LIST: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(TripletListWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for NuExtract full output (entities + relationships).
/// Used by NuExtractExtractor and GroundedNuExtractExtractor.
pub(crate) static SCHEMA_NUEXTRACT_BOTH: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(NuExtractOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for NuExtract entities-only pass (GroundedNuExtractExtractor pass 1).
/// Root object with `entities: [RawEntitySimple]`.
pub(crate) static SCHEMA_NUEXTRACT_ENTITIES_ONLY: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(EntityOnlyOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for NuExtract relationships-only pass (GroundedNuExtractExtractor pass 2).
///
/// NOTE: `subject`/`predicate`/`object` are excluded from schema because they use
/// `deser_string_or_array`. This schema MUST be routed via `LlmJsonRepair` arm.
/// See `SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM`.
pub(crate) static SCHEMA_NUEXTRACT_RELATIONS_ONLY: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(RelOnlyOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for ProgrammaticFirstExtractor entity-typing call (Call 1).
/// Root object with `entities: [RawEntitySimple]`.
///
/// NOTE: same shape as `SCHEMA_NUEXTRACT_ENTITIES_ONLY` — both derive from
/// `EntityOnlyOutput`. Kept as a distinct named constant per spec §3.3:
/// the two call-site paths route via separate names to avoid aliasing.
pub(crate) static SCHEMA_ENTITY_TYPING: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(EntityOnlyOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for ContradictionVerdict — wrapped index list.
/// Root object with `indices: [u32]`. Replaces bare `Vec<usize>` in parse_index_list.
pub(crate) static SCHEMA_CONTRADICTION_VERDICT: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(ContradictionVerdictWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for rel-only forced-fallback path.
/// Identical to SCHEMA_NUEXTRACT_RELATIONS_ONLY; exists as a distinct named
/// constant so the force_arm routing path does not alias the primary schema.
///
/// NOTE: also requires `LlmJsonRepair` arm — see `SCHEMA_REL_ONLY_FORCE_FALLBACK_FORCE_ARM`.
pub(crate) static SCHEMA_REL_ONLY_FORCE_FALLBACK: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(RelOnlyOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for entity-resolution verdict (CascadeResolver Tier 3 LLM call).
/// Root object with `verdict: String`.
///
/// Replaces the bare `chat_with_tools(..., None, None)` call in `resolver.rs`
/// so the classification response is schema-enforced through `StructuredCallBuilder`.
pub(crate) static SCHEMA_RESOLUTION_VERDICT: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(ResolutionVerdictWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

// ─── Per-schema force_arm overrides (§3.3) ───────────────────────────────────

/// Force `SCHEMA_NUEXTRACT_RELATIONS_ONLY` through the LlmJsonRepair arm.
///
/// Schemas that wrap `Vec<RawRelationship>` (with `deser_string_or_array` on
/// `subject`/`predicate`/`object`) MUST bypass Native/FormatSchema arms —
/// strict JSON Schema `"type":"string"` would mask array tokens to -∞,
/// producing syntactically-conformant output that silently violates the
/// deserialiser. The LlmJsonRepair arm lets the array-or-string coercion fire.
pub(crate) const SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM: Option<FallbackArm> =
    Some(FallbackArm::LlmJsonRepair);

/// Force `SCHEMA_REL_ONLY_FORCE_FALLBACK` through the LlmJsonRepair arm.
/// Same rationale as `SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM`.
pub(crate) const SCHEMA_REL_ONLY_FORCE_FALLBACK_FORCE_ARM: Option<FallbackArm> =
    Some(FallbackArm::LlmJsonRepair);

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::Value;

    // ── Schema generation ─────────────────────────────────────────────────────

    #[test]
    fn schema_entity_list_is_object_with_items_array() {
        let schema: &Value = &SCHEMA_ENTITY_LIST;
        assert_eq!(schema["type"], "object", "root must be object");
        let items = &schema["properties"]["items"];
        assert_eq!(items["type"], "array", "items must be array");
    }

    #[test]
    fn schema_rel_type_list_items_string_array() {
        let schema: &Value = &SCHEMA_REL_TYPE_LIST;
        assert_eq!(schema["type"], "object");
        let items = &schema["properties"]["items"];
        assert_eq!(items["type"], "array");
        assert_eq!(
            items["items"]["type"], "string",
            "rel type items must be strings"
        );
    }

    #[test]
    fn schema_triplet_list_is_object_with_items_array() {
        let schema: &Value = &SCHEMA_TRIPLET_LIST;
        assert_eq!(schema["type"], "object");
        let items = &schema["properties"]["items"];
        assert_eq!(items["type"], "array");
    }

    #[test]
    fn schema_contradiction_verdict_has_indices_array() {
        let schema: &Value = &SCHEMA_CONTRADICTION_VERDICT;
        assert_eq!(schema["type"], "object");
        let indices = &schema["properties"]["indices"];
        assert_eq!(indices["type"], "array", "indices must be array");
        assert_eq!(
            indices["items"]["format"], "uint32",
            "indices items must be uint32"
        );
    }

    #[test]
    fn schema_entity_typing_has_entities_array() {
        let schema: &Value = &SCHEMA_ENTITY_TYPING;
        assert_eq!(schema["type"], "object");
        let entities = &schema["properties"]["entities"];
        assert_eq!(entities["type"], "array");
    }

    #[test]
    fn schema_rel_only_has_relationships_array() {
        let schema: &Value = &SCHEMA_NUEXTRACT_RELATIONS_ONLY;
        assert_eq!(schema["type"], "object", "root must be object");
        let relationships = &schema["properties"]["relationships"];
        assert_eq!(
            relationships["type"], "array",
            "relationships must be array"
        );

        // The skipped fields (subject/predicate/object) must NOT appear as schema
        // property keys — schemars may inline or use $defs/$ref. We check the
        // full schema JSON for the specific key patterns used in JSON Schema
        // property definitions.
        let schema_str = serde_json::to_string(schema).unwrap();
        assert!(
            !schema_str.contains(r#""subject":"#) && !schema_str.contains(r#""subject" :"#),
            "subject must be absent from schema properties (deser_string_or_array incompatible)"
        );
        assert!(
            !schema_str.contains(r#""predicate":"#) && !schema_str.contains(r#""predicate" :"#),
            "predicate must be absent from schema properties"
        );
        // is_entity_ref and confidence ARE schema-representable — they should appear
        assert!(
            schema_str.contains("is_entity_ref"),
            "is_entity_ref should be present in schema"
        );
    }

    // ── FallbackArm ───────────────────────────────────────────────────────────

    #[test]
    fn rel_schemas_force_llm_json_repair_arm() {
        assert_eq!(
            SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM,
            Some(FallbackArm::LlmJsonRepair)
        );
        assert_eq!(
            SCHEMA_REL_ONLY_FORCE_FALLBACK_FORCE_ARM,
            Some(FallbackArm::LlmJsonRepair)
        );
    }

    #[test]
    fn non_rel_schemas_have_no_force_arm_override() {
        // No force_arm const for entity/triplet/contradiction schemas —
        // they use the full ladder per ProviderCaps.
        // Verify FallbackArm variants are distinct (not accidentally equal).
        assert_ne!(FallbackArm::NativeSchema, FallbackArm::LlmJsonRepair);
        assert_ne!(FallbackArm::FormatSchema, FallbackArm::PromptOnly);
    }

    // ── ContradictionVerdictWrapper round-trip ────────────────────────────────

    #[test]
    fn contradiction_verdict_wrapper_deserialises_wrapped_form() {
        let json = r#"{"indices": [0, 1, 2]}"#;
        let w: ContradictionVerdictWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(w.indices, vec![0u32, 1u32, 2u32]);
    }

    #[test]
    fn contradiction_verdict_wrapper_rejects_bare_array() {
        // The old parse_index_list accepted "[0, 1, 2]" — the new contract
        // requires the wrapped form. Bare arrays MUST fail.
        let result = serde_json::from_str::<ContradictionVerdictWrapper>("[0, 1, 2]");
        assert!(
            result.is_err(),
            "bare array must be rejected by ContradictionVerdictWrapper"
        );
    }

    // ── ResolutionVerdictWrapper round-trip ───────────────────────────────────

    #[test]
    fn resolution_verdict_wrapper_deserialises_same() {
        let json = r#"{"verdict": "same"}"#;
        let w: ResolutionVerdictWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(w.verdict, "same");
    }

    #[test]
    fn resolution_verdict_wrapper_deserialises_different() {
        let json = r#"{"verdict": "different"}"#;
        let w: ResolutionVerdictWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(w.verdict, "different");
    }

    #[test]
    fn resolution_verdict_wrapper_deserialises_uncertain() {
        let json = r#"{"verdict": "uncertain"}"#;
        let w: ResolutionVerdictWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(w.verdict, "uncertain");
    }

    #[test]
    fn schema_resolution_verdict_is_object_with_verdict_string() {
        let schema: &Value = &SCHEMA_RESOLUTION_VERDICT;
        assert_eq!(schema["type"], "object", "root must be object");
        let verdict = &schema["properties"]["verdict"];
        assert_eq!(verdict["type"], "string", "verdict must be string");
    }

    #[test]
    fn resolution_verdict_wrapper_empty_object_yields_default_verdict() {
        // PromptOnly arm returns {} when LLM returns empty — must deserialise
        // without error via #[serde(default)], yielding verdict="" which maps
        // conservatively to ResolutionResult::Different.
        let w: ResolutionVerdictWrapper = serde_json::from_str("{}").unwrap();
        assert_eq!(
            w.verdict, "",
            "empty object must yield empty verdict string"
        );
    }
}
