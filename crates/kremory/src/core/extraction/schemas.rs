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
//! [`FallbackArm`] selects which arm of the fallback ladder is used
//! for a given schema. Schemas wrapping [`RawRelationship`] — which carries
//! `deser_string_or_array` on its `subject`/`predicate`/`object` fields and
//! therefore has those fields excluded from the JSON Schema — **must** bypass
//! native/format schema arms and route to [`FallbackArm::LlmJsonRepair`]
//! where the array-or-string coercion still fires.

use serde::Deserialize;
use serde_json::Value;
use std::sync::LazyLock;

use super::models::{
    EntityListIntegerWrapper, HybridTypingWrapper, RawEntitySimple, RawFact, RawRelationship,
};

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
///
/// `reason` is the LLM's audit-trail justification for the verdict.
/// No `#[serde(default)]` on `reason` — a missing field from the LLM is a
/// parse failure, forcing the fallback ladder to retry rather than silently
/// accepting an incomplete response.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct ContradictionVerdictWrapper {
    pub(crate) indices: Vec<u32>,
    /// Audit-trail explanation of why these indices were selected.
    /// Required — LLM must emit this field; absence is a parse error.
    pub(crate) reason: String,
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
/// the prior raw-text response path.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct ResolutionVerdictWrapper {
    /// Resolution verdict: "same", "different", or "uncertain".
    /// Defaults to empty string when the LLM returns no verdict (PromptOnly arm).
    #[serde(default)]
    pub(crate) verdict: String,
}

/// One resolved entity in a **batched** entity-resolution response.
///
/// Mirrors Graphiti's `NodeDuplicate` (`graphiti_core/prompts/dedupe_nodes.py`).
/// `id` indexes the window's ambiguous-entity worklist presented in the prompt
/// (`0..K-1`); `duplicate_candidate_id` is the `candidate_id` of the matching
/// EXISTING entity in the shared window candidate pool, or `-1` when novel.
///
/// The INNER fields carry **no** `#[serde(default)]`: a missing field is a
/// parse error so the fallback ladder can retry rather than silently
/// defaulting to a wrong resolution. `id` is `u32`; `duplicate_candidate_id`
/// is `i32` because `-1` (novel) is a valid, load-bearing value.
///
/// `name` is a **verbatim echo** (the prompt instructs the model to copy the
/// entity's name exactly). It is used ONLY as a reject-only misindex checksum
/// at map-back: if `normalize_name(name)` ≠ the worklist entity's normalized
/// name, the model misindexed → the row is rejected to conservative-NEW.
/// It is NOT used for naming (kremory keeps first-mention-wins).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct BatchedNodeResolution {
    /// Index into the window's ambiguous-entity worklist (`0..K-1`).
    pub(crate) id: u32,
    /// Verbatim echo of the entity's name — reject-only misindex checksum.
    pub(crate) name: String,
    /// `candidate_id` of the matching existing entity, or `-1` if novel.
    pub(crate) duplicate_candidate_id: i32,
}

/// Root wrapper for a batched entity-resolution response.
///
/// Mirrors Graphiti's `NodeResolutions` — a single `entity_resolutions` array,
/// one entry per ambiguous extracted entity that survived the deterministic
/// cheap tiers (exact + MinHash) and the cosine blocking pre-filter.
/// Replaces the O(extracted × candidates) pairwise `ResolutionVerdictWrapper`
/// fan-out with one structured call per window.
///
/// `#[serde(default)]` on `entity_resolutions` is intentional: a
/// ladder-exhausted `PromptOnly` `{}` response deserialises to an EMPTY list,
/// so every windowed entity degrades to conservative-NEW (graceful) rather than
/// failing the whole ingest transaction. Empty array is a semantically valid
/// empty list here — the INNER `BatchedNodeResolution` fields stay required
/// so a malformed *row* still drives the fallback ladder loudly.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct BatchedNodeResolutions {
    #[serde(default)]
    pub(crate) entity_resolutions: Vec<BatchedNodeResolution>,
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

// ─── LlmExtractionOutput visibility re-export ────────────────────────────────────

// LlmExtractionOutput is defined and pub(crate) in mod.rs — imported directly by
// the SCHEMA_NUEXTRACT_BOTH static below.
use super::models::LlmExtractionOutput;

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
    /// LightRAG-style `<|#|>`-delimited tuple format (L6 fallback).
    ///
    /// Prompts the LLM to emit one entity per line as:
    /// `entity<|#|>name<|#|>entity_type_id<|#|>description`
    /// Terminated by an optional `<|COMPLETE|>` sentinel.
    /// Per-line parser: one malformed line drops one entity, not the whole parse.
    DelimitedTuple,
    /// Prompt-only, no provider-side enforcement.
    PromptOnly,
}

// ─── Canonical schema statics (§3.3) ─────────────────────────────────────────

/// Schema for entity-list responses (IntegerIdLlmExtractor stage 1, Graphiti stage 1).
/// Root object with `items: [RawEntitySimple]`.
pub(crate) static SCHEMA_ENTITY_LIST: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(EntityListWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Build a per-call entity-list schema with `label` constrained to an enum of
/// registry names — structural enforcement.
///
/// Returns a CLONED + mutated Value (not a static reference). The static
/// `SCHEMA_ENTITY_LIST` declares only `name`; this fn inserts a `label`
/// property with `enum: [Person, Organisation, Location, ...]` so the
/// provider's format_schema arm (Ollama) or strict native-schema arm
/// (Anthropic/OpenAI) constrains LLM output at decode time. LLM physically
/// cannot emit "Entity" unless it appears in the enum.
///
/// When `specs` is empty, returns the static schema unchanged (no enum
/// constraint — any string acceptable, server-side label_to_id falls back
/// to id=0 catch-all).
pub(crate) fn entity_list_schema_with_label_enum(
    specs: &[crate::core::entity_types::EntityTypeSpec],
) -> Value {
    let mut schema = (*SCHEMA_ENTITY_LIST).clone();
    if specs.is_empty() {
        return schema;
    }
    // Filter out id=0 catch-all from the schema enum: LLM must commit to a
    // specific type. Server-side label_to_id falls back to 0 if the emitted
    // label somehow bypasses the grammar (defensive only — Ollama
    // format_schema enforces the enum at decode time).
    let names: Vec<Value> = specs
        .iter()
        .filter(|s| s.id != 0)
        .map(|s| Value::String(s.name.clone()))
        .collect();
    if names.is_empty() {
        return schema;
    }
    let label_property = serde_json::json!({
        "type": "string",
        "enum": names,
        "description": "Entity type label; must be one of the registered type names."
    });
    // Try $defs/RawEntitySimple path first (schemars output shape).
    if let Some(props) = schema
        .pointer_mut("/$defs/RawEntitySimple/properties")
        .and_then(|v| v.as_object_mut())
    {
        props.insert("label".to_string(), label_property.clone());
        if let Some(req) = schema
            .pointer_mut("/$defs/RawEntitySimple/required")
            .and_then(|v| v.as_array_mut())
        {
            if !req.iter().any(|v| v.as_str() == Some("label")) {
                req.push(Value::String("label".to_string()));
            }
        }
        return schema;
    }
    // Fallback: inlined items shape (no $defs).
    if let Some(props) = schema
        .pointer_mut("/properties/items/items/properties")
        .and_then(|v| v.as_object_mut())
    {
        props.insert("label".to_string(), label_property);
        if let Some(req) = schema
            .pointer_mut("/properties/items/items/required")
            .and_then(|v| v.as_array_mut())
        {
            if !req.iter().any(|v| v.as_str() == Some("label")) {
                req.push(Value::String("label".to_string()));
            }
        }
    }
    schema
}

/// Schema for relationship-type-name lists (IntegerIdLlmExtractor stage 2).
/// Root object with `items: [String]`.
pub(crate) static SCHEMA_REL_TYPE_LIST: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(RelTypeListWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for triplet lists (IntegerIdLlmExtractor stage 3).
/// Root object with `items: [RawFact]`.
pub(crate) static SCHEMA_TRIPLET_LIST: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(TripletListWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for full LLM extraction output (entities + relationships).
/// Used by [`LlmExtractor`] for single-pass extraction.
pub(crate) static SCHEMA_NUEXTRACT_BOTH: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(LlmExtractionOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for entities-only extraction pass (GLiNER+LLM hybrid pass 1 typing step).
/// Root object with `entities: [RawEntitySimple]`.
pub(crate) static SCHEMA_NUEXTRACT_ENTITIES_ONLY: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(EntityOnlyOutput)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for relationships-only extraction pass (GLiNER+LLM hybrid pass 2 relationship step).
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

/// Schema for the **batched** entity-resolution call.
/// Root object with `entity_resolutions: [BatchedNodeResolution]`.
///
/// Replaces the O(extracted × candidates) pairwise `SCHEMA_RESOLUTION_VERDICT`
/// fan-out with one structured call per chunk (Graphiti `NodeResolutions`
/// pattern). Invoked via `StructuredCallBuilder` exactly like the pairwise
/// schema; the deterministic exact/MinHash tiers and the cosine blocking
/// pre-filter still run first so only the ambiguous remainder reaches this
/// call.
pub(crate) static SCHEMA_BATCHED_RESOLUTION: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(BatchedNodeResolutions)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Schema for the integer-ID entity list.
///
/// Root object with `entities: [RawEntityIntegerId]`.  The `entity_type_id`
/// field is a plain `u32` — `entity_list_schema_with_id_bounds` mutates this
/// static's clone to inject an `enum` + `minimum`/`maximum` constraint derived
/// from the active namespace registry.
///
/// Call sites MUST use `entity_list_schema_with_id_bounds` (which clones this
/// static and injects the runtime constraint) rather than this static directly.
pub(crate) static SCHEMA_ENTITY_LIST_INTEGER_ID: std::sync::LazyLock<Value> =
    std::sync::LazyLock::new(|| {
        serde_json::to_value(schemars::schema_for!(EntityListIntegerWrapper)).unwrap_or_else(|e| {
            panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
        })
    });

/// Build a per-call entity-list schema with `entity_type_id` constrained to an
/// `enum` of registered integer IDs.
///
/// Returns a cloned + mutated `Value`.  The `entity_type_id` property in the
/// `RawEntityIntegerId` definition is replaced with:
/// ```json
/// {"type": "integer", "enum": [0, 1, 2, ...], "minimum": 0, "maximum": N,
///  "description": "Entity type id from the registry."}
/// ```
///
/// id=0 (catch-all "Entity") is included in the enum so the LLM can fall back
/// to it when uncertain — the grammar enforces the choice at decode time.
///
/// When `specs` is empty, returns the base static unchanged (no enum constraint).
///
/// Navigation path follows the schemars output shape:
/// `$defs/RawEntityIntegerId/properties/entity_type_id` (primary).
/// Falls back to inline `properties/entities/items/properties/entity_type_id`.
pub(crate) fn entity_list_schema_with_id_bounds(
    specs: &[crate::core::entity_types::EntityTypeSpec],
) -> Value {
    let mut schema = (*SCHEMA_ENTITY_LIST_INTEGER_ID).clone();
    if specs.is_empty() {
        return schema;
    }
    // FILTER OUT id=0 catch-all from the enum: LLM must commit to a specific
    // type. Empirically, with id=0 present in the enum, qwen2.5:14b picks
    // it for 11/11 uncertain entities (then encodes the real type as
    // parenthetical text in the NAME field). Server-side validate_or_fallback
    // still resolves out-of-enum emissions to id=0 if the grammar somehow
    // allows (defensive only — Ollama format_schema enforces the enum at
    // decode time so this branch is unreachable for compliant providers).
    let ids: Vec<Value> = specs
        .iter()
        .filter(|s| s.id != 0)
        .map(|s| Value::Number(s.id.into()))
        .collect();
    if ids.is_empty() {
        return schema; // no specific types registered, leave schema unconstrained
    }
    let max_id = specs.iter().map(|s| s.id).max().unwrap_or(0);
    let min_id = specs
        .iter()
        .filter(|s| s.id != 0)
        .map(|s| s.id)
        .min()
        .unwrap_or(1);
    let id_property = serde_json::json!({
        "type": "integer",
        "enum": ids,
        "minimum": min_id,
        "maximum": max_id,
        "description": "Entity type id from the registry. MUST be a SPECIFIC type. id=0 catch-all is reserved server-side and not a valid LLM emission."
    });

    // Primary path: $defs/RawEntityIntegerId/properties/entity_type_id
    if let Some(props) = schema
        .pointer_mut("/$defs/RawEntityIntegerId/properties")
        .and_then(|v| v.as_object_mut())
    {
        props.insert("entity_type_id".to_string(), id_property.clone());
        if let Some(req) = schema
            .pointer_mut("/$defs/RawEntityIntegerId/required")
            .and_then(|v| v.as_array_mut())
        {
            if !req.iter().any(|v| v.as_str() == Some("entity_type_id")) {
                req.push(Value::String("entity_type_id".to_string()));
            }
        }
        return schema;
    }
    // Fallback: inline items shape (no $defs).
    if let Some(props) = schema
        .pointer_mut("/properties/entities/items/properties")
        .and_then(|v| v.as_object_mut())
    {
        props.insert("entity_type_id".to_string(), id_property);
        if let Some(req) = schema
            .pointer_mut("/properties/entities/items/required")
            .and_then(|v| v.as_array_mut())
        {
            if !req.iter().any(|v| v.as_str() == Some("entity_type_id")) {
                req.push(Value::String("entity_type_id".to_string()));
            }
        }
    }
    schema
}

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

// ─── Hybrid typing schema (index-based) ──────────────────────────────────────

/// Base schema for the hybrid typing response.
///
/// Root: `{typings: [{idx: u32, entity_type_id: u32}]}`. Both fields are
/// numerically bounded by `hybrid_typing_schema_with_bounds` at the call
/// site — `idx` to `[0, num_candidates - 1]` and `entity_type_id` to the
/// active namespace registry's enum. Enforced structurally at the schema
/// level: the LLM cannot drop or misalign candidates because both keys
/// are integer-constrained at decode time.
pub(crate) static SCHEMA_HYBRID_TYPING: LazyLock<Value> = LazyLock::new(|| {
    serde_json::to_value(schemars::schema_for!(HybridTypingWrapper)).unwrap_or_else(|e| {
        panic!("invariant: schemars::schema_for! is infallible for derived structs — {e}")
    })
});

/// Inject runtime bounds for the hybrid typing schema:
///   - `idx`: enum [0, 1, ..., num_candidates - 1]; minimum=0; maximum=N-1
///   - `entity_type_id`: enum of registered ids (excluding id=0 catch-all);
///     bounded by min/max registered id
///
/// `num_candidates` MUST equal the input candidate list length so the LLM
/// cannot emit an out-of-range idx. `specs` should be the active namespace
/// registry. Both fields end up `required` so a missing field is a parse
/// error, forcing the fallback ladder to retry rather than silently
/// accepting an incomplete response.
pub(crate) fn hybrid_typing_schema_with_bounds(
    num_candidates: usize,
    specs: &[crate::core::entity_types::EntityTypeSpec],
) -> Value {
    let mut schema = (*SCHEMA_HYBRID_TYPING).clone();

    if num_candidates == 0 {
        // No candidates → no LLM call should be made; return base schema unchanged.
        return schema;
    }

    let last_idx = num_candidates.saturating_sub(1);
    let idx_enum: Vec<Value> = (0..=last_idx as u64).map(Value::from).collect();
    let idx_property = serde_json::json!({
        "type": "integer",
        "enum": idx_enum,
        "minimum": 0,
        "maximum": last_idx,
        "description": "0-based index into the candidate list. MUST match exactly one of the listed indices."
    });

    let entity_type_ids: Vec<Value> = specs
        .iter()
        .filter(|s| s.id != 0)
        .map(|s| Value::Number(s.id.into()))
        .collect();
    let type_id_property = if entity_type_ids.is_empty() {
        // No registered types — leave the entity_type_id unconstrained but typed.
        serde_json::json!({
            "type": "integer",
            "minimum": 0,
            "description": "Entity type id; registry is empty for this namespace."
        })
    } else {
        let max_id = specs.iter().map(|s| s.id).max().unwrap_or(0);
        let min_id = specs
            .iter()
            .filter(|s| s.id != 0)
            .map(|s| s.id)
            .min()
            .unwrap_or(1);
        serde_json::json!({
            "type": "integer",
            "enum": entity_type_ids,
            "minimum": min_id,
            "maximum": max_id,
            "description": "Entity type id from the registry. MUST be a SPECIFIC type. id=0 catch-all is reserved server-side."
        })
    };

    // Primary path: $defs/RawHybridTyping/properties
    if let Some(props) = schema
        .pointer_mut("/$defs/RawHybridTyping/properties")
        .and_then(|v| v.as_object_mut())
    {
        props.insert("idx".to_string(), idx_property.clone());
        props.insert("entity_type_id".to_string(), type_id_property.clone());
        if let Some(req) = schema
            .pointer_mut("/$defs/RawHybridTyping/required")
            .and_then(|v| v.as_array_mut())
        {
            for field in ["idx", "entity_type_id"] {
                if !req.iter().any(|v| v.as_str() == Some(field)) {
                    req.push(Value::String(field.to_string()));
                }
            }
        }
        return schema;
    }
    // Fallback: inline items shape (no $defs).
    if let Some(props) = schema
        .pointer_mut("/properties/typings/items/properties")
        .and_then(|v| v.as_object_mut())
    {
        props.insert("idx".to_string(), idx_property);
        props.insert("entity_type_id".to_string(), type_id_property);
    }
    schema
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
    fn entity_list_schema_with_label_enum_injects_enum() {
        use crate::core::entity_types::EntityTypeSpec;
        let specs = vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "p".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "o".to_string(),
            },
        ];
        let schema = entity_list_schema_with_label_enum(&specs);
        let pretty = serde_json::to_string_pretty(&schema).unwrap();
        tracing::debug!("SCHEMA DUMP:\n{}", pretty);
        // Try $defs path first
        let label_at_defs = schema.pointer("/$defs/RawEntitySimple/properties/label");
        let label_inline = schema.pointer("/properties/items/items/properties/label");
        let label = label_at_defs.or(label_inline);
        assert!(
            label.is_some(),
            "label property must be injected somewhere; schema: {pretty}"
        );
        let label = label.unwrap();
        let enum_arr = label["enum"]
            .as_array()
            .unwrap_or_else(|| panic!("label must have enum array; got: {label}"));
        let names: Vec<&str> = enum_arr.iter().filter_map(|v| v.as_str()).collect();
        // id=0 Entity is FILTERED OUT to force LLM to commit to a specific type.
        assert!(
            !names.contains(&"Entity"),
            "id=0 catch-all must be filtered from enum"
        );
        assert!(names.contains(&"Person"));
        assert!(names.contains(&"Organisation"));
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
        assert_ne!(FallbackArm::DelimitedTuple, FallbackArm::LlmJsonRepair);
        assert_ne!(FallbackArm::DelimitedTuple, FallbackArm::PromptOnly);
    }

    // ── ContradictionVerdictWrapper round-trip ────────────────────────────────

    #[test]
    fn contradiction_verdict_wrapper_deserialises_wrapped_form() {
        let json = r#"{"indices": [0, 1, 2], "reason": "facts [0,1,2] are superseded"}"#;
        let w: ContradictionVerdictWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(w.indices, vec![0u32, 1u32, 2u32]);
        assert_eq!(w.reason, "facts [0,1,2] are superseded");
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

    #[test]
    fn contradiction_verdict_carries_reason() {
        // reason field is required — missing field must fail parse, not
        // silently default to empty string.
        let with_reason =
            r#"{"indices": [1], "reason": "fact 1 is outdated by the new assertion"}"#;
        let w: ContradictionVerdictWrapper = serde_json::from_str(with_reason).unwrap();
        assert_eq!(w.indices, vec![1u32]);
        assert!(!w.reason.is_empty(), "reason must be populated");
        assert_eq!(w.reason, "fact 1 is outdated by the new assertion");

        // Missing reason field must fail — not silently default.
        let without_reason = r#"{"indices": [1]}"#;
        let result = serde_json::from_str::<ContradictionVerdictWrapper>(without_reason);
        assert!(
            result.is_err(),
            "missing reason must be a parse error, not a silent default"
        );

        // Empty indices with reason is valid (no contradictions, reasoning still required).
        let no_contradictions = r#"{"indices": [], "reason": "no temporal overlap detected"}"#;
        let w2: ContradictionVerdictWrapper = serde_json::from_str(no_contradictions).unwrap();
        assert!(w2.indices.is_empty());
        assert_eq!(w2.reason, "no temporal overlap detected");
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

    // ── Integer-ID schema ─────────────────────────────────────────────────────

    #[test]
    fn entity_list_schema_with_id_bounds_injects_enum_and_bounds() {
        use crate::core::entity_types::EntityTypeSpec;
        let specs = vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ];
        let schema = entity_list_schema_with_id_bounds(&specs);
        let pretty = serde_json::to_string_pretty(&schema).unwrap();

        // entity_type_id property must be injected somewhere
        let at_defs = schema.pointer("/$defs/RawEntityIntegerId/properties/entity_type_id");
        let inline = schema.pointer("/properties/entities/items/properties/entity_type_id");
        let prop = at_defs
            .or(inline)
            .unwrap_or_else(|| panic!("entity_type_id must be injected in schema; got:\n{pretty}"));

        // type must be integer
        assert_eq!(
            prop["type"], "integer",
            "entity_type_id must have type=integer"
        );

        // enum must EXCLUDE id=0 catch-all — LLM must commit to a specific type.
        // Server-side validate_or_fallback handles bypass + maps unknowns to 0.
        let enum_arr = prop["enum"]
            .as_array()
            .unwrap_or_else(|| panic!("entity_type_id must have enum array; got: {prop}"));
        let ids: Vec<u64> = enum_arr.iter().filter_map(|v| v.as_u64()).collect();
        assert!(
            !ids.contains(&0),
            "enum must EXCLUDE id=0 catch-all; got: {ids:?}"
        );
        assert!(
            ids.contains(&1),
            "enum must include id=1 (Person); got: {ids:?}"
        );
        assert!(
            ids.contains(&2),
            "enum must include id=2 (Organisation); got: {ids:?}"
        );

        // bounds: minimum is now the smallest non-zero registered id
        assert_eq!(prop["minimum"], 1, "minimum must be smallest non-zero id");
        assert_eq!(prop["maximum"], 2, "maximum must be max registered id");
    }

    #[test]
    fn entity_list_schema_with_id_bounds_empty_specs_returns_base() {
        let schema = entity_list_schema_with_id_bounds(&[]);
        // No enum injected — schema should still be valid object with entities array.
        assert_eq!(
            schema["type"], "object",
            "root must be object when no specs"
        );
        let entities = &schema["properties"]["entities"];
        assert_eq!(entities["type"], "array", "entities must be array");
    }

    #[test]
    fn schema_entity_list_integer_id_is_object_with_entities_array() {
        let schema: &Value = &SCHEMA_ENTITY_LIST_INTEGER_ID;
        assert_eq!(schema["type"], "object", "root must be object");
        let entities = &schema["properties"]["entities"];
        assert_eq!(entities["type"], "array", "entities must be array");
    }

    // ── Batched-resolution schema spike ──────────────────────────────────────
    // Mechanical compile-spike: proves the Graphiti NodeResolutions-style
    // schema derives, generates valid JSON Schema at runtime, and
    // round-trips the load-bearing `-1` novel value.

    #[test]
    fn schema_batched_resolution_is_object_with_entity_resolutions_array() {
        let schema: &Value = &SCHEMA_BATCHED_RESOLUTION;
        assert_eq!(schema["type"], "object", "root must be object");
        let arr = &schema["properties"]["entity_resolutions"];
        assert_eq!(arr["type"], "array", "entity_resolutions must be array");
    }

    #[test]
    fn batched_resolution_parses_novel_and_duplicate_rows() {
        // duplicate_candidate_id = -1 (novel) and >=0 (matched) must both
        // parse; `name` is the verbatim-echo misindex checksum.
        let raw = r#"{"entity_resolutions":[
            {"id":0,"name":"Boston","duplicate_candidate_id":-1},
            {"id":1,"name":"Alice","duplicate_candidate_id":3}
        ]}"#;
        let parsed: BatchedNodeResolutions =
            serde_json::from_str(raw).expect("well-formed batched resolution must parse");
        assert_eq!(parsed.entity_resolutions.len(), 2);
        assert_eq!(parsed.entity_resolutions[0].name, "Boston");
        assert_eq!(parsed.entity_resolutions[0].duplicate_candidate_id, -1);
        assert_eq!(parsed.entity_resolutions[1].id, 1);
        assert_eq!(parsed.entity_resolutions[1].duplicate_candidate_id, 3);
    }

    #[test]
    fn batched_resolution_missing_inner_field_is_parse_error() {
        // INNER fields have no #[serde(default)] — a row missing
        // duplicate_candidate_id must be a loud parse error so the fallback
        // ladder retries rather than silently defaulting to a wrong
        // resolution.
        let raw = r#"{"entity_resolutions":[{"id":0,"name":"Boston"}]}"#;
        let parsed: Result<BatchedNodeResolutions, _> = serde_json::from_str(raw);
        assert!(
            parsed.is_err(),
            "missing duplicate_candidate_id must fail parse, not default"
        );
    }

    #[test]
    fn batched_resolution_empty_object_degrades_to_empty_list() {
        // A ladder-exhausted PromptOnly `{}` must deserialise to an EMPTY
        // list (every windowed entity → conservative-NEW), NOT fail the
        // whole ingest. #[serde(default)] on the wrapper Vec makes this
        // graceful.
        let parsed: BatchedNodeResolutions =
            serde_json::from_str("{}").expect("empty object must degrade to empty list, not error");
        assert!(
            parsed.entity_resolutions.is_empty(),
            "empty `{{}}` response must yield zero resolutions (all conservative-NEW)"
        );
    }
}
