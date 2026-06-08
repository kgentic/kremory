pub(crate) mod delimited_tuple;
pub mod factory;
#[cfg(feature = "ner")]
pub mod hybrid_typer;
pub mod prompts;
pub(crate) mod schemas;
pub(crate) mod structured;

pub use factory::{ExtractorSource, ProductionExtractor};

use std::sync::Arc;
use std::time::Instant;

use metrics::{counter, histogram};
use serde::Deserialize;
use tracing;

use crate::core::error::Result;

// ─── Serde helper: accept string or array ───────────────────────────────────

/// Deserialize a JSON string or array into a `String`.
///
/// Some LLMs (e.g. `llama3.2:3b`) emit arrays where the schema expects
/// a scalar: `"object": ["Python", "Rust", "Julia"]`.  This helper joins
/// array elements with `", "` so the extracted fact is still useful.
fn deser_string_or_array<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
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
fn default_string() -> String {
    String::new()
}

use crate::core::config::ContentType;
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};

// ─── Serde models for LLM JSON output coercion ──────────────────────────────

/// Top-level NuExtract output. Both fields are optional — LLMs may omit one.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct NuExtractOutput {
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

fn default_entity_label() -> String {
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

/// Entity array as emitted by the DefaultExtractor (stage 1).
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
// NOTE: `#[serde(default)]` is intentionally REMOVED from both fields (2026-06-04).
// Previously defaults swallowed truncated LLM output: a fragmented post-repair JSON
// like `[{"name":"Boston\", \"entity_type_id\":3}, {"}]` deserialized as a single
// entity with name=<garbage> and entity_type_id=0 (the default), collapsing all
// extractions to label="Entity". Without defaults, missing fields fail loudly so
// the fallback ladder (LlmJsonRepair → DelimitedTuple → PromptOnly) can retry.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawEntityIntegerId {
    pub(crate) name: String,
    /// Integer entity type id constrained to the active namespace registry.
    /// id=0 = "Entity" catch-all; id ≥ 1 = user-defined types.
    pub(crate) entity_type_id: u32,
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

/// Fact triplet as emitted by the DefaultExtractor (stage 3).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub(crate) struct RawFact {
    #[serde(default)]
    pub(crate) subject: String,
    #[serde(default)]
    pub(crate) predicate: String,
    #[serde(default)]
    pub(crate) object: String,
    #[serde(default)]
    pub(crate) is_entity_ref: bool,
    #[serde(default = "default_confidence")]
    pub(crate) confidence: f64,
}

// ─── DefaultExtractor ─────────────────────────────────────────────────────────

pub struct DefaultExtractor<L: ChatProvider> {
    llm: Arc<L>,
}

impl<L: ChatProvider> DefaultExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }
}

impl<L: ChatProvider> EntityExtractor for DefaultExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        // Stage 1: Extract entities.
        let known_hint = if ctx.known_entities.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = ctx
                .known_entities
                .iter()
                .map(|e| format!("{} ({})", e.name, e.label))
                .collect();
            format!("\nKnown entities (include these): {}\n", names.join(", "))
        };
        let stage1_prompt = format!(
            "{}{known_hint}",
            build_entity_prompt(text, ctx.allowed_entity_types, ctx.registry_specs)
        );
        let stage1_start = Instant::now();
        let stage1_msgs = vec![
            chat_msg_system("You are an entity extraction system. Extract named entities from text. Each entity must appear ONCE — no duplicates. Output valid JSON only."),
            chat_msg_user(stage1_prompt),
        ];
        // Build per-call schema with entity_type_id constrained to registered integer IDs.
        // Decode-time enforcement (TD-013 L1 / CLAUDE.md Rule 15): structural
        // grammar prevents LLM from emitting ids outside the registry enum.
        // Spike 1c/1d (2026-06-03): confirmed qwen2.5:14b emits id=0 on adversarial bypass.
        let stage1_schema = schemas::entity_list_schema_with_id_bounds(ctx.registry_specs);
        let stage1_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &stage1_schema,
            "EntityListIntegerId",
        )
        .messages(stage1_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::ExtractionStage {
            stage: "entities".to_string(),
            detail: e.to_string(),
        })?;
        let _ms = stage1_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "entities").record(_ms);
        tracing::info!(_ms, stage = "entities", "kremory.extraction.stage_ms");
        let stage1_text = serde_json::to_string(&stage1_value).unwrap_or_default();
        // L1: resolve integer ids → label strings via registry.
        // Build a temporary registry from the per-call specs; no DB round-trip needed here
        // since ingest_with already loaded and passed registry_specs into ExtractionContext.
        let stage1_registry =
            crate::core::entity_types::EntityTypeRegistry::from_specs(ctx.registry_specs.to_vec());
        let mut entities: Vec<ExtractedEntity> =
            parse_entities_integer(&stage1_text, &stage1_registry)?;

        // Apply exclusion filter.
        if !ctx.excluded_entity_types.is_empty() {
            entities.retain(|e| !ctx.excluded_entity_types.contains(&e.label));
        }

        // Stage 2: Extract relationship names (with stage 1 context).
        let stage2_prompt = build_relation_names_prompt(text, &entities, ctx.allowed_edge_types);
        let stage2_start = Instant::now();
        let stage2_msgs = vec![
            chat_msg_system("You are a relationship extraction system. Given entities found in text, identify relationship type names. Output valid JSON only."),
            chat_msg_user(stage2_prompt),
        ];
        let stage2_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_REL_TYPE_LIST,
            "RelTypeList",
        )
        .messages(stage2_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::ExtractionStage {
            stage: "relations".to_string(),
            detail: e.to_string(),
        })?;
        let _ms = stage2_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "relations").record(_ms);
        tracing::info!(_ms, stage = "relations", "kremory.extraction.stage_ms");
        let stage2_text = serde_json::to_string(&stage2_value).unwrap_or_default();
        let relation_names: Vec<String> = parse_relation_names(&stage2_text)?;

        // Stage 3: Extract full triplets (with stage 1 + 2 context).
        let stage3_prompt = build_triplet_prompt(text, &entities, &relation_names);
        let stage3_start = Instant::now();
        let stage3_msgs = vec![
            chat_msg_system("You are a knowledge graph extraction system. Extract (subject, predicate, object) triplets. Output valid JSON only."),
            chat_msg_user(stage3_prompt),
        ];
        let stage3_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_TRIPLET_LIST,
            "TripletList",
        )
        .messages(stage3_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = stage3_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "triplets").record(_ms);
        tracing::info!(_ms, stage = "triplets", "kremory.extraction.stage_ms");
        let stage3_text = serde_json::to_string(&stage3_value).unwrap_or_default();
        let facts: Vec<ExtractedFact> = parse_facts(&stage3_text)?;

        let entity_count = entities.len();
        let fact_count = facts.len();
        histogram!("rql.extraction.entity_count").record(entity_count as f64);
        histogram!("rql.extraction.fact_count").record(fact_count as f64);
        tracing::info!(
            entity_count,
            fact_count,
            extractor = "three_stage",
            "kremory.extraction.result"
        );

        Ok(ExtractionResult { entities, facts })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NuExtract Template-Based Extractor
// ═══════════════════════════════════════════════════════════════════════════════

/// NuExtract 2.0 default template — domain-agnostic.
/// Uses "verbatim-string" for all fields so the model discovers entity types freely.
///
/// For better extraction, callers should configure `PipelineConfig::allowed_entity_types`
/// with domain-specific types (e.g. `["Person", "Organisation", "Project"]` for meetings,
/// `["Species", "Habitat", "Gene"]` for biology). The `build_nuextract_template()` function
/// converts these into NuExtract enum constraints automatically.
const NUEXTRACT_TEMPLATE: &str = r#"{"entities": [{"name": "verbatim-string", "label": "verbatim-string"}], "relationships": [{"subject": "verbatim-string", "predicate": "verbatim-string", "object": "verbatim-string"}]}"#;

pub struct NuExtractExtractor<L: ChatProvider> {
    llm: Arc<L>,
}

impl<L: ChatProvider> NuExtractExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }
}

impl<L: ChatProvider> EntityExtractor for NuExtractExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        let template = build_nuextract_template(ctx.allowed_entity_types, ctx.allowed_edge_types);
        let format_hint = match ctx.content_type {
            ContentType::Message => "\n# Format: Conversational transcript with speaker labels. Extract all entities and relationships mentioned across speakers.\n",
            ContentType::Json => "\n# Format: Structured data fields. Extract entities and relationships from field values.\n",
            ContentType::Text | ContentType::Document => "",
        };
        let known_hint = if ctx.known_entities.is_empty() {
            String::new()
        } else {
            let names: Vec<&str> = ctx.known_entities.iter().map(|e| e.name.as_str()).collect();
            format!("\n# Known entities: {}\n", names.join(", "))
        };
        let prompt =
            format!("# Template:\n{template}{format_hint}{known_hint}\n# Context:\n{text}");

        let start = Instant::now();
        let nuextract_msgs = vec![chat_msg_user(prompt)];
        let nuextract_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_NUEXTRACT_BOTH,
            "NuExtractBoth",
        )
        .messages(nuextract_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "nuextract").record(_ms);
        tracing::info!(_ms, stage = "nuextract", "kremory.extraction.stage_ms");
        let resp_text = serde_json::to_string(&nuextract_value).unwrap_or_default();

        let (entities, facts) = parse_nuextract_response(&resp_text, ctx)?;

        let entity_count = entities.len();
        let fact_count = facts.len();
        histogram!("rql.extraction.entity_count").record(entity_count as f64);
        histogram!("rql.extraction.fact_count").record(fact_count as f64);
        tracing::info!(
            entity_count,
            fact_count,
            extractor = "nuextract",
            "kremory.extraction.result"
        );

        Ok(ExtractionResult { entities, facts })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GroundedNuExtractExtractor — two-pass extraction with entity constraint
// ═══════════════════════════════════════════════════════════════════════════════

pub struct GroundedNuExtractExtractor<L: ChatProvider> {
    llm: Arc<L>,
}

impl<L: ChatProvider> GroundedNuExtractExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }
}

impl<L: ChatProvider> EntityExtractor for GroundedNuExtractExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        // === Pass 1: Entities only ===
        let entity_template = build_entity_only_template(ctx.allowed_entity_types);
        let format_hint = match ctx.content_type {
            ContentType::Message => "\n# Format: Conversational transcript with speaker labels. Extract all entities mentioned across speakers.\n",
            ContentType::Json => "\n# Format: Structured data fields. Extract entities from field values.\n",
            ContentType::Text | ContentType::Document => "",
        };
        let known_hint = if ctx.known_entities.is_empty() {
            String::new()
        } else {
            let names: Vec<&str> = ctx.known_entities.iter().map(|e| e.name.as_str()).collect();
            format!("\n# Known entities: {}\n", names.join(", "))
        };
        let pass1_prompt =
            format!("# Template:\n{entity_template}{format_hint}{known_hint}\n# Context:\n{text}");

        let start = Instant::now();
        let pass1_msgs = vec![chat_msg_user(pass1_prompt)];
        let pass1_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_NUEXTRACT_ENTITIES_ONLY,
            "NuExtractEntitiesOnly",
        )
        .messages(pass1_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "grounded_entities").record(_ms);
        tracing::info!(
            _ms,
            stage = "grounded_entities",
            "kremory.extraction.stage_ms"
        );
        let pass1_text = serde_json::to_string(&pass1_value).unwrap_or_default();

        let (entities, _) = parse_nuextract_response(&pass1_text, ctx)?;

        if entities.is_empty() {
            counter!("rql.extraction.grounded_empty_pass1").increment(1);
            tracing::info!("kremory.extraction.grounded_empty_pass1");
            return Ok(ExtractionResult {
                entities: vec![],
                facts: vec![],
            });
        }

        // === Pass 2: Relationships constrained to extracted entities ===
        let entity_names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        let rel_template =
            build_grounded_relationship_template(&entity_names, ctx.allowed_edge_types);
        let pass2_prompt = format!("# Template:\n{rel_template}{format_hint}\n# Context:\n{text}");

        let start2 = Instant::now();
        let pass2_msgs = vec![chat_msg_user(pass2_prompt)];
        let mut pass2_builder = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_NUEXTRACT_RELATIONS_ONLY,
            "NuExtractRelationsOnly",
        )
        .messages(pass2_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms);
        if let Some(arm) = schemas::SCHEMA_NUEXTRACT_RELATIONS_ONLY_FORCE_ARM {
            pass2_builder = pass2_builder.force_arm(arm);
        }
        let pass2_value = pass2_builder
            .call()
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start2.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "grounded_relationships").record(_ms);
        tracing::info!(
            _ms,
            stage = "grounded_relationships",
            "kremory.extraction.stage_ms"
        );
        let pass2_text = serde_json::to_string(&pass2_value).unwrap_or_default();

        let (_, mut facts) = parse_nuextract_response(&pass2_text, ctx)?;

        // Fix is_entity_ref using Pass 1 entity set (parse_nuextract_response
        // computed it from an empty entities list since Pass 2 has no "entities" key)
        let entity_name_set: std::collections::HashSet<String> =
            entities.iter().map(|e| e.name.to_lowercase()).collect();
        for fact in &mut facts {
            fact.is_entity_ref = entity_name_set.contains(&fact.object.to_lowercase());
        }

        let entity_count = entities.len();
        let fact_count = facts.len();
        histogram!("rql.extraction.entity_count").record(entity_count as f64);
        histogram!("rql.extraction.fact_count").record(fact_count as f64);
        tracing::info!(
            entity_count,
            fact_count,
            extractor = "grounded_nuextract",
            "kremory.extraction.result"
        );

        Ok(ExtractionResult { entities, facts })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GraphitiStyleExtractor — Graphiti-quality prompts, 3-stage, unconstrained
// ═══════════════════════════════════════════════════════════════════════════════

/// 3-stage extractor with Graphiti-quality prompts: detailed instructions,
/// exclusion rules, worked examples, and the "Wikipedia test" quality filter.
/// Uses unconstrained generation + llm_json repair (no grammar constraints).
pub struct GraphitiStyleExtractor<L: ChatProvider> {
    llm: Arc<L>,
}

impl<L: ChatProvider> GraphitiStyleExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }
}

impl<L: ChatProvider> EntityExtractor for GraphitiStyleExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        // Stage 1: Extract entities with Graphiti-quality prompts.
        let known_hint = if ctx.known_entities.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = ctx
                .known_entities
                .iter()
                .map(|e| format!("{} ({})", e.name, e.label))
                .collect();
            format!(
                "\nAlready extracted (do not duplicate): {}\n",
                names.join(", ")
            )
        };
        let existing_block = prompts::render_existing_entities_block(ctx.existing_graph_entities);
        let stage1_prompt = format!(
            "{existing_block}{}{known_hint}",
            build_graphiti_entity_prompt(text, ctx.allowed_entity_types, ctx.registry_specs)
        );
        let stage1_start = Instant::now();
        let graphiti_s1_msgs = vec![
            chat_msg_system(GRAPHITI_ENTITY_SYSTEM),
            chat_msg_user(stage1_prompt),
        ];
        // L1: integer-ID schema — grammar constrains entity_type_id to registered enum at decode time.
        let graphiti_s1_schema = schemas::entity_list_schema_with_id_bounds(ctx.registry_specs);
        let graphiti_s1_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &graphiti_s1_schema,
            "EntityListIntegerId",
        )
        .messages(graphiti_s1_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = stage1_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "graphiti_entities").record(_ms);
        tracing::info!(
            _ms,
            stage = "graphiti_entities",
            "kremory.extraction.stage_ms"
        );
        let stage1_text = serde_json::to_string(&graphiti_s1_value).unwrap_or_default();
        // L1: resolve integer ids → label strings via registry.
        let graphiti_s1_registry =
            crate::core::entity_types::EntityTypeRegistry::from_specs(ctx.registry_specs.to_vec());
        let mut entities: Vec<ExtractedEntity> =
            parse_entities_integer(&stage1_text, &graphiti_s1_registry)?;

        if !ctx.excluded_entity_types.is_empty() {
            entities.retain(|e| !ctx.excluded_entity_types.contains(&e.label));
        }

        if entities.is_empty() {
            return Ok(ExtractionResult {
                entities: vec![],
                facts: vec![],
            });
        }

        // Stage 2: Extract relationships anchored to found entities.
        let stage2_prompt = build_graphiti_relationship_prompt(text, &entities);
        let stage2_start = Instant::now();
        let graphiti_s2_msgs = vec![
            chat_msg_system(GRAPHITI_RELATIONSHIP_SYSTEM),
            chat_msg_user(stage2_prompt),
        ];
        let graphiti_s2_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_TRIPLET_LIST,
            "TripletList",
        )
        .messages(graphiti_s2_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = stage2_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "graphiti_relationships").record(_ms);
        tracing::info!(
            _ms,
            stage = "graphiti_relationships",
            "kremory.extraction.stage_ms"
        );
        let stage2_text = serde_json::to_string(&graphiti_s2_value).unwrap_or_default();
        let mut facts: Vec<ExtractedFact> = parse_facts(&stage2_text)?;

        // Fix is_entity_ref using Stage 1 entity set.
        let entity_name_set: std::collections::HashSet<String> =
            entities.iter().map(|e| e.name.to_lowercase()).collect();
        for fact in &mut facts {
            fact.is_entity_ref = entity_name_set.contains(&fact.object.to_lowercase());
        }

        let entity_count = entities.len();
        let fact_count = facts.len();
        histogram!("rql.extraction.entity_count").record(entity_count as f64);
        histogram!("rql.extraction.fact_count").record(fact_count as f64);
        tracing::info!(
            entity_count,
            fact_count,
            extractor = "graphiti_style",
            "kremory.extraction.result"
        );

        Ok(ExtractionResult { entities, facts })
    }
}

const GRAPHITI_ENTITY_SYSTEM: &str = "\
You are a precise entity extraction system. Extract named entities from text and output valid JSON only.

Rules:
- Extract specific, identifiable entities that could have their own Wikipedia article or database entry.
- Always extract speaker names from dialogue lines (before ':') AND from self-introductions ('I'm X', 'my name is X', 'this is X speaking').
- Extract people mentioned by name even if they are not speakers.
- Use full names when available (e.g. 'Dr. Sarah Chen' not 'Sarah').
- When someone refers to a relative or associate by bare term (e.g. 'my dad'), qualify with the possessor ('James's dad'). Do NOT extract bare terms alone.

Do NOT extract:
- Generic nouns (government, team, company, system) unless explicitly named
- Adjectives, sentence fragments, or descriptive phrases
- Bare kinship terms (dad, mom, boss) or bare pet words (dog, cat) without qualification
- Actions, processes, or abstract concepts
- Duplicate references — extract each entity at most once";

/// Default broad entity types — used when caller provides no types.
/// Includes a catch-all "Entity" so no entity goes unextracted.
const DEFAULT_ENTITY_TYPES: &[&str] = &[
    "Person",
    "Organisation",
    "Location",
    "Technology",
    "Product",
    "Event",
    "Date",
];

fn build_graphiti_entity_prompt(
    text: &str,
    allowed_types: &[String],
    registry_specs: &[crate::core::entity_types::EntityTypeSpec],
) -> String {
    let types = if allowed_types.is_empty() {
        let defaults = DEFAULT_ENTITY_TYPES
            .iter()
            .map(|t| format!("\"{}\"", t))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Classify each entity using one of these types: {defaults}. If an entity doesn't fit any listed type, use \"Entity\" as the label.")
    } else {
        format!("Classify each entity using one of these types: {}. If an entity doesn't fit any type, use \"Entity\".", allowed_types.join(", "))
    };
    let l2_guidance = prompts::build_l2_guidance(registry_specs);

    format!("\
{types}

{l2_guidance}
<TEXT>
{text}
</TEXT>

Extract all named entities from the TEXT above. Output a JSON object: {{\"entities\": [{{\"name\": \"<literal name>\", \"entity_type_id\": <integer from registry>}}, ...]}}. Use the integer entity_type_id from the registry table above. Never include type information in the name field.

Examples (assuming registry has Person=1, Organisation=2, Location=3):
- Speaker line \"Dr. Patel: The test results...\" → {{\"name\": \"Dr. Patel\", \"entity_type_id\": 1}}
- Self-introduction \"Hi, I'm Ria\" → {{\"name\": \"Ria\", \"entity_type_id\": 1}}
- \"studied at Northeastern University\" → {{\"name\": \"Northeastern University\", \"entity_type_id\": 2}}
- \"works at Acme Corp\" → {{\"name\": \"Acme Corp\", \"entity_type_id\": 2}}
- \"originally from Morocco\" → {{\"name\": \"Morocco\", \"entity_type_id\": 3}}
- Do NOT extract \"the test results\" (generic noun phrase)
- Do NOT extract \"a few different internships\" (vague reference)")
}

const GRAPHITI_RELATIONSHIP_SYSTEM: &str = "\
You are a relationship extraction system. Given a list of entities found in text, extract the factual relationships between them. Output valid JSON only.

Rules:
- Every relationship must connect two entities from the provided list.
- Use specific, descriptive predicates (e.g. 'prescribed', 'works_at', 'reported_to', 'diagnosed_with').
- Include temporal relationships when stated (e.g. 'joined_in', 'left_on').
- Extract each distinct relationship once — do not repeat.
- Set is_entity_ref to true when the object is an entity from the list, false when it's a literal value.
- Set confidence between 0.0 and 1.0 based on how explicitly the relationship is stated.";

fn build_graphiti_relationship_prompt(text: &str, entities: &[ExtractedEntity]) -> String {
    let entity_list = entities
        .iter()
        .map(|e| format!("{} ({})", e.name, e.label))
        .collect::<Vec<_>>()
        .join(", ");

    format!("\
Entities found: [{entity_list}]

<TEXT>
{text}
</TEXT>

Extract all factual relationships between the entities above. Output a JSON array of objects with \"subject\", \"predicate\", \"object\", \"is_entity_ref\" (boolean), and \"confidence\" (0.0-1.0) fields.

Examples:
- {{\"subject\": \"Alice\", \"predicate\": \"works_at\", \"object\": \"Acme Corp\", \"is_entity_ref\": true, \"confidence\": 0.95}}
- {{\"subject\": \"Dr. Patel\", \"predicate\": \"prescribed\", \"object\": \"Metformin\", \"is_entity_ref\": true, \"confidence\": 0.9}}
- {{\"subject\": \"Alice\", \"predicate\": \"joined_in\", \"object\": \"Q2 2025\", \"is_entity_ref\": false, \"confidence\": 0.8}}")
}

fn build_entity_only_template(allowed_entity_types: &[String]) -> String {
    let label_type = if allowed_entity_types.is_empty() {
        "\"verbatim-string\"".to_string()
    } else {
        let opts: Vec<String> = allowed_entity_types
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect();
        format!("[{}]", opts.join(", "))
    };
    format!(r#"{{"entities": [{{"name": "verbatim-string", "label": {label_type}}}]}}"#)
}

fn build_grounded_relationship_template(
    entity_names: &[&str],
    allowed_edge_types: &[String],
) -> String {
    let entity_enum: Vec<String> = entity_names.iter().map(|n| format!("\"{n}\"")).collect();
    let entity_constraint = format!("[{}]", entity_enum.join(", "));
    let predicate_type = if allowed_edge_types.is_empty() {
        "\"verbatim-string\"".to_string()
    } else {
        let opts: Vec<String> = allowed_edge_types
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect();
        format!("[{}]", opts.join(", "))
    };
    format!(
        r#"{{"relationships": [{{"subject": {entity_constraint}, "predicate": {predicate_type}, "object": {entity_constraint}}}]}}"#
    )
}

fn build_nuextract_template(
    allowed_entity_types: &[String],
    allowed_edge_types: &[String],
) -> String {
    if allowed_entity_types.is_empty() && allowed_edge_types.is_empty() {
        return NUEXTRACT_TEMPLATE.to_string();
    }

    // Build a constrained template with enum types for labels/predicates.
    let label_type = if allowed_entity_types.is_empty() {
        "\"verbatim-string\"".to_string()
    } else {
        // NuExtract enum format: ["Option1", "Option2", ...]
        let opts: Vec<String> = allowed_entity_types
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect();
        format!("[{}]", opts.join(", "))
    };
    let predicate_type = if allowed_edge_types.is_empty() {
        "\"verbatim-string\"".to_string()
    } else {
        let opts: Vec<String> = allowed_edge_types
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect();
        format!("[{}]", opts.join(", "))
    };

    format!(
        r#"{{"entities": [{{"name": "verbatim-string", "label": {label_type}}}], "relationships": [{{"subject": "verbatim-string", "predicate": {predicate_type}, "object": "verbatim-string"}}]}}"#
    )
}

/// Pre-process NuExtract JSON to fix unclosed string values before `}`.
///
/// Some LLMs (e.g. `gemma4-e2b`) omit the closing `"` on string values when
/// followed immediately by `}`, producing output like:
///   `{"name": "Bob", "label": "Person}`
/// instead of the correct:
///   `{"name": "Bob", "label": "Person"}`
///
/// This pattern breaks `serde_json` parsing before `llm_json::repair_json` can
/// help, because the `}` gets absorbed into the unclosed string, mangling the
/// rest of the document structure.
///
/// Strategy: scan for `"}` where the `"` was intended as opening-quote-already-in-
/// progress. More concretely, find any byte sequence that matches `"<word>}` where
/// `<word>` contains no `"`, `\`, `{`, `}`, or newline — and insert the missing `"`.
/// Extract the first balanced JSON object `{...}` from `s`.
///
/// Some LLMs (e.g. llama3.2) emit valid JSON followed by prose ("Note: ..."),
/// or emit multiple JSON objects separated by whitespace. This function returns
/// a slice covering only the first complete `{...}` block. If no balanced
/// object is found, returns the original input unchanged.
fn extract_first_json_object(s: &str) -> &str {
    let bytes = s.as_bytes();
    let len = bytes.len();

    // Find the first `{`
    let start = match bytes.iter().position(|&b| b == b'{') {
        Some(pos) => pos,
        None => return s,
    };

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut i = start;

    while i < len {
        match bytes[i] {
            b'"' if !in_string => {
                in_string = true;
                i += 1;
            }
            b'"' if in_string => {
                in_string = false;
                i += 1;
            }
            b'\\' if in_string => {
                // skip escape sequence
                i += 2;
            }
            b'{' if !in_string => {
                depth += 1;
                i += 1;
            }
            b'}' if !in_string => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    return &s[start..i];
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // No balanced object found — return original.
    s
}

fn fix_unclosed_string_before_brace(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = Vec::with_capacity(len + 8);
    let mut i = 0;

    while i < len {
        // Look for the pattern: `"` followed by 1+ safe chars followed by `}`
        // where no closing `"` precedes the `}`.
        if bytes[i] == b'"' {
            // Scan forward to find the end of this potential string.
            let start = i; // points at opening `"`
            i += 1;
            let mut found_close = false;
            while i < len {
                match bytes[i] {
                    b'"' => {
                        // Properly closed string — copy verbatim up to and including `"`.
                        found_close = true;
                        i += 1;
                        break;
                    }
                    b'\\' => {
                        // Escape sequence — skip both chars.
                        i += 2;
                    }
                    b'}' if !found_close => {
                        // `}` inside an unclosed string — insert missing `"` before `}`.
                        out.extend_from_slice(&bytes[start..i]);
                        out.push(b'"'); // close the string
                        out.push(b'}'); // then the brace
                        i += 1;
                        found_close = true; // consumed this token
                        break;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            if !found_close && i >= len {
                // Ran off the end without closing — just copy remainder as-is.
                out.extend_from_slice(&bytes[start..i]);
            } else if found_close {
                // Copy up to current position if we broke on a real close quote.
                // (already pushed in the `}` branch above; for the `"` branch copy.)
                // For the `"` branch we need to push the range start..i.
                // Check: did we push via the `}` branch (already done)?
                // We can tell because `bytes[i-1]` would be `}` (pushed above).
                if i > 0 && bytes[i - 1] != b'}' {
                    out.extend_from_slice(&bytes[start..i]);
                }
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(out).unwrap_or_else(|_| s.to_owned())
}

fn parse_nuextract_response(
    json: &str,
    ctx: &ExtractionContext<'_>,
) -> anyhow::Result<(Vec<ExtractedEntity>, Vec<ExtractedFact>)> {
    let trimmed = json.trim();
    if trimmed.is_empty() {
        counter!("rql.extraction.json_parse_fail").increment(1);
        tracing::warn!(
            parser = "nuextract",
            "kremory.extraction.json_parse_fail empty response"
        );
        return Ok((vec![], vec![]));
    }

    // Extract the first balanced `{...}` block. Some LLMs (e.g. llama3.2) emit
    // valid JSON followed by prose ("Note: ...") or multiple JSON objects.
    // Feeding multiple objects or trailing prose to serde/repair_json produces
    // "invalid type: sequence, expected a string" errors.
    let first_obj = extract_first_json_object(trimmed);

    // If no JSON object found at all (pure prose response), return empty gracefully.
    if !first_obj.trim_start().starts_with('{') {
        counter!("rql.extraction.json_parse_fail").increment(1);
        tracing::warn!(
            parser = "nuextract",
            "kremory.extraction.json_parse_fail no JSON object in response"
        );
        return Ok((vec![], vec![]));
    }

    // Pre-process: some LLMs omit closing `"` before `}` in string values.
    // Fix this before attempting serde / llm_json repair.
    let preprocessed = fix_unclosed_string_before_brace(first_obj);
    let trimmed = preprocessed.as_ref();

    // Try direct parse, then llm_json repair if malformed.
    let output: NuExtractOutput = match serde_json::from_str(trimmed) {
        Ok(v) => {
            counter!("rql.extraction.json_parse_ok").increment(1);
            v
        }
        Err(_) => {
            // repair_to_array wraps in array — but NuExtract returns an object.
            // Try object-level repair first, then array-unwrap fallback.
            let repaired = llm_json::repair_json(trimmed, &llm_json::RepairOptions::default())
                .unwrap_or_else(|_| trimmed.to_owned());
            match serde_json::from_str::<NuExtractOutput>(&repaired) {
                Ok(v) => {
                    counter!("rql.extraction.json_parse_ok").increment(1);
                    v
                }
                Err(_) => {
                    // Last resort: repair_to_array may have wrapped the object.
                    let array_repaired = repair_to_array(trimmed);
                    match serde_json::from_str::<serde_json::Value>(&array_repaired) {
                        Ok(v) => {
                            // Unwrap single-element array back to object.
                            let obj = if let Some(arr) = v.as_array() {
                                if arr.len() == 1 {
                                    &arr[0]
                                } else {
                                    &v
                                }
                            } else {
                                &v
                            };
                            match serde_json::from_value::<NuExtractOutput>(obj.clone()) {
                                Ok(v) => {
                                    counter!("rql.extraction.json_parse_ok").increment(1);
                                    v
                                }
                                Err(e) => {
                                    counter!("rql.extraction.json_parse_fail").increment(1);
                                    tracing::warn!(error = %e, parser = "nuextract", "kremory.extraction.json_parse_fail after repair");
                                    eprintln!(
                                        "warn: failed to parse NuExtract JSON after repair: {e}"
                                    );
                                    return Ok((vec![], vec![]));
                                }
                            }
                        }
                        Err(e) => {
                            counter!("rql.extraction.json_parse_fail").increment(1);
                            tracing::warn!(error = %e, parser = "nuextract", "kremory.extraction.json_parse_fail array repair");
                            eprintln!("warn: failed to parse NuExtract JSON after repair: {e}");
                            return Ok((vec![], vec![]));
                        }
                    }
                }
            }
        }
    };

    // Convert serde structs → domain types, filtering empty names.
    // Store entity name in `properties["name"]` so the FTS index (which indexes
    // the `properties` column) can match queries against the entity's original
    // case name (e.g. "Alice"). Without this, FTS MATCH "Alice" returns nothing
    // because entity_id is UNINDEXED and label is the type ("Person"), not the name.
    let mut entities: Vec<ExtractedEntity> = output
        .entities
        .into_iter()
        .filter(|e| !e.name.is_empty())
        .map(|e| {
            let label = if e.label.is_empty() {
                "Entity".to_string()
            } else {
                e.label
            };
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            ExtractedEntity {
                name: e.name,
                label,
                properties: serde_json::Value::Object(props),
            }
        })
        .collect();

    // Apply exclusion filter.
    if !ctx.excluded_entity_types.is_empty() {
        entities.retain(|e| !ctx.excluded_entity_types.contains(&e.label));
    }

    // Build entity name set for is_entity_ref resolution.
    let entity_names: std::collections::HashSet<String> =
        entities.iter().map(|e| e.name.to_lowercase()).collect();

    // Convert relationships, filtering empty required fields.
    let facts: Vec<ExtractedFact> = output
        .relationships
        .into_iter()
        .filter(|r| !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty())
        .map(|r| {
            let is_entity_ref = entity_names.contains(&r.object.to_lowercase());
            ExtractedFact {
                subject: r.subject,
                predicate: r.predicate,
                object: r.object,
                is_entity_ref,
                confidence: r.confidence,
            }
        })
        .collect();

    Ok((entities, facts))
}

// ─── Prompt Builders ─────────────────────────────────────────────────────────

fn build_entity_prompt(
    text: &str,
    allowed_types: &[String],
    registry_specs: &[crate::core::entity_types::EntityTypeSpec],
) -> String {
    let type_hint = if allowed_types.is_empty() {
        String::new()
    } else {
        format!(
            "\nOnly extract entities of these types: {}",
            allowed_types.join(", ")
        )
    };
    let l2_guidance = prompts::build_l2_guidance(registry_specs);
    format!(
        "Extract all unique named entities from the following text. Each entity must appear exactly once.{type_hint}\n\n{l2_guidance}\nText: {text}\n\nOutput a JSON object: {{\"entities\": [{{\"name\": \"<literal entity name>\", \"entity_type_id\": <integer from the entity_type_id list shown above>}}, ...]}}. The entity_type_id MUST be a specific integer from the registry — never include type information in the name field. No duplicates."
    )
}

fn build_relation_names_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    allowed_edges: &[String],
) -> String {
    let entity_list = entities
        .iter()
        .map(|e| format!("{} ({})", e.name, e.label))
        .collect::<Vec<_>>()
        .join(", ");
    let edge_hint = if allowed_edges.is_empty() {
        String::new()
    } else {
        format!(
            "\nOnly extract these relationship types: {}",
            allowed_edges.join(", ")
        )
    };
    format!(
        "Given these entities: [{entity_list}]{edge_hint}\n\nWhat relationship types connect them in the following text?\n\nText: {text}\n\nOutput a JSON array of relationship name strings."
    )
}

fn build_triplet_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    relation_names: &[String],
) -> String {
    let entity_list = entities
        .iter()
        .map(|e| format!("{} ({})", e.name, e.label))
        .collect::<Vec<_>>()
        .join(", ");
    let rel_list = relation_names.join(", ");
    format!(
        "Given entities: [{entity_list}]\nRelationship types: [{rel_list}]\n\nExtract the key relationships from this text as (subject, predicate, object) triplets. Only include each distinct relationship once. Do not repeat.\n\nText: {text}\n\nOutput a concise JSON array of objects with \"subject\", \"predicate\", \"object\", \"is_entity_ref\" (boolean), and \"confidence\" (0.0-1.0) fields."
    )
}

// ─── JSON Repair ────────────────────────────────────────────────────────────

/// Repair messy LLM JSON into a parseable array.
/// Handles: markdown fences, missing array brackets, trailing commas,
/// single quotes, Python booleans (True/False/None).
fn repair_to_array(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip markdown code fences.
    if s.starts_with("```") {
        if let Some(first_nl) = s.find('\n') {
            s = s[first_nl + 1..].to_string();
        }
        if s.ends_with("```") {
            s.truncate(s.len() - 3);
            s = s.trim_end().to_string();
        }
    }

    let s = s.trim();
    if s.is_empty() || s == "[]" {
        return "[]".to_string();
    }

    // Wrap in array brackets if the output starts with `{` (objects without wrapper).
    let mut wrapped = if s.starts_with('{') {
        format!("[{s}]")
    } else {
        s.to_string()
    };

    // Strip trailing commas before `]` — llm_json mishandles these.
    while wrapped.contains(",]") {
        wrapped = wrapped.replace(",]", "]");
    }
    while wrapped.contains(", ]") {
        wrapped = wrapped.replace(", ]", "]");
    }

    // Use llm_json for fine-grained repairs (quotes, booleans, commas).
    let repaired =
        llm_json::repair_json(&wrapped, &llm_json::RepairOptions::default()).unwrap_or(wrapped);

    // llm_json sometimes reduces arrays to a single object — re-wrap if needed.
    let repaired = repaired.trim();
    if repaired.starts_with('{') {
        format!("[{repaired}]")
    } else {
        repaired.to_string()
    }
}

// ─── Parsers ─────────────────────────────────────────────────────────────────

fn parse_relation_names(json: &str) -> anyhow::Result<Vec<String>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Ok(vec![]);
    }

    let raw: Vec<serde_json::Value> = match serde_json::from_str(trimmed) {
        Ok(v) => {
            counter!("rql.extraction.json_parse_ok").increment(1);
            v
        }
        Err(_) => {
            let repaired = repair_to_array(trimmed);
            match serde_json::from_str(&repaired) {
                Ok(v) => {
                    counter!("rql.extraction.json_parse_ok").increment(1);
                    v
                }
                Err(e) => {
                    counter!("rql.extraction.json_parse_fail").increment(1);
                    tracing::warn!(error = %e, parser = "relation_names", "kremory.extraction.json_parse_fail");
                    eprintln!("warn: failed to parse relation names JSON after repair: {e}");
                    return Ok(vec![]);
                }
            }
        }
    };

    Ok(raw
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(s),
            _ => v.as_str().map(|s| s.to_string()),
        })
        .collect())
}

/// Legacy string-label entity parser (pre-TD-013 L1 path).
///
/// Production extractors now use `parse_entities_integer`.  This function is
/// retained as a test fixture for the string-label parse path (used by
/// `test_parse_entities_*` tests).  Rustc dead-code analysis doesn't count
/// `#[cfg(test)]` callers from the production-code vantage point.
#[allow(dead_code)]
fn parse_entities(json: &str) -> anyhow::Result<Vec<ExtractedEntity>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Ok(vec![]);
    }

    let raw: Vec<RawEntitySimple> = match serde_json::from_str(trimmed) {
        Ok(v) => {
            counter!("rql.extraction.json_parse_ok").increment(1);
            v
        }
        Err(_) => {
            let repaired = repair_to_array(trimmed);
            match serde_json::from_str(&repaired) {
                Ok(v) => {
                    counter!("rql.extraction.json_parse_ok").increment(1);
                    v
                }
                Err(e) => {
                    counter!("rql.extraction.json_parse_fail").increment(1);
                    tracing::warn!(error = %e, parser = "entities", "kremory.extraction.json_parse_fail");
                    eprintln!("warn: failed to parse entity JSON after repair: {e}");
                    return Ok(vec![]);
                }
            }
        }
    };

    Ok(raw
        .into_iter()
        .filter(|e| !e.name.is_empty() && !e.label.is_empty())
        .map(|e| {
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            ExtractedEntity {
                name: e.name,
                label: e.label,
                properties: serde_json::Value::Object(props),
            }
        })
        .collect())
}

/// Parse the integer-ID entity JSON emitted by the L1 extraction path (TD-013).
///
/// Accepts two input shapes:
/// 1. Wrapped: `{"entities": [{"name": "Alice", "entity_type_id": 1}, ...]}` — primary.
/// 2. Bare array: `[{"name": "Alice", "entity_type_id": 1}, ...]` — repair fallback.
/// 3. Single object: `{"name": "Alice", "entity_type_id": 1}` — wrapped via repair_to_array.
///
/// For each `RawEntityIntegerId`:
/// - Skips entries with empty `name`.
/// - Validates `entity_type_id` via `EntityTypeRegistry::validate_or_fallback`
///   (out-of-range or unknown ids → id=0 catch-all "Entity").
/// - Resolves the integer id to a label string via `EntityTypeRegistry::id_to_name`.
/// - Builds `ExtractedEntity { name, label, properties }` — downstream contract preserved.
fn parse_entities_integer(
    json: &str,
    registry: &crate::core::entity_types::EntityTypeRegistry,
) -> anyhow::Result<Vec<ExtractedEntity>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Ok(vec![]);
    }

    // KREMORY_DEBUG=1: emit raw stage1 LLM output to stderr for diagnosis (Rule 19).
    if std::env::var("KREMORY_DEBUG").is_ok() {
        eprintln!(
            "[KREMORY_DEBUG] parse_entities_integer raw input (len={}):\n{trimmed}\n[KREMORY_DEBUG end]",
            trimmed.len()
        );
    }

    // Try wrapped form first: {"entities": [...]}
    let raw: Vec<RawEntityIntegerId> = if let Ok(w) =
        serde_json::from_str::<EntityListIntegerWrapper>(trimmed)
    {
        counter!("rql.extraction.json_parse_ok", "path" => "wrapped").increment(1);
        w.entities
    } else {
        // Try bare array.
        match serde_json::from_str::<Vec<RawEntityIntegerId>>(trimmed) {
            Ok(v) => {
                counter!("rql.extraction.json_parse_ok", "path" => "bare_array").increment(1);
                v
            }
            Err(_) => {
                // repair_to_array handles: markdown fences, single-object, trailing commas.
                let repaired = repair_to_array(trimmed);
                match serde_json::from_str::<Vec<RawEntityIntegerId>>(&repaired) {
                    Ok(v) => {
                        // Per Rule 20: repair-path success is suspicious. Track separately
                        // so qwen-vs-haiku divergence and post-repair garbage are visible.
                        counter!("rql.extraction.json_parse_ok", "path" => "post_repair")
                            .increment(1);
                        v
                    }
                    Err(e) => {
                        counter!("rql.extraction.json_parse_fail").increment(1);
                        tracing::warn!(
                            error = %e,
                            parser = "entities_integer",
                            "kremory.extraction.json_parse_fail"
                        );
                        eprintln!("warn: failed to parse integer-id entity JSON after repair: {e}");
                        return Ok(vec![]);
                    }
                }
            }
        }
    };

    Ok(raw
        .into_iter()
        .filter_map(|e| {
            if e.name.is_empty() {
                counter!("rql.extraction.entity_rejected", "reason" => "empty_name").increment(1);
                return None;
            }
            // Shape-validate the name. Repair paths can splice JSON fragments
            // into the name field; reject any name containing JSON syntax chars
            // so garbage entities never reach persistence.
            if name_looks_like_json_fragment(&e.name) {
                counter!("rql.extraction.entity_rejected", "reason" => "name_json_fragment")
                    .increment(1);
                tracing::warn!(
                    raw_name = %e.name,
                    "kremory.extraction.entity_rejected.name_json_fragment"
                );
                return None;
            }
            let validated_id = registry.validate_or_fallback(e.entity_type_id);
            let label = registry.id_to_name(validated_id).to_string();
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            Some(ExtractedEntity {
                name: e.name,
                label,
                properties: serde_json::Value::Object(props),
            })
        })
        .collect())
}

/// Returns true if `name` contains characters that suggest it is a spliced JSON
/// fragment rather than a real entity name. Repair paths (`repair_to_array` +
/// `llm_json::repair_json`) can produce parseable output where a truncated
/// object's tail bleeds into the next entity's `name`. A real entity name will
/// never legitimately contain unescaped JSON structural characters.
fn name_looks_like_json_fragment(name: &str) -> bool {
    name.contains('"')
        || name.contains('\\')
        || name.contains('{')
        || name.contains('}')
        || name.contains("entity_type_id")
}

fn parse_facts(json: &str) -> anyhow::Result<Vec<ExtractedFact>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Ok(vec![]);
    }

    let raw: Vec<RawFact> = match serde_json::from_str(trimmed) {
        Ok(v) => {
            counter!("rql.extraction.json_parse_ok").increment(1);
            v
        }
        Err(_) => {
            let repaired = repair_to_array(trimmed);
            match serde_json::from_str(&repaired) {
                Ok(v) => {
                    counter!("rql.extraction.json_parse_ok").increment(1);
                    v
                }
                Err(e) => {
                    counter!("rql.extraction.json_parse_fail").increment(1);
                    tracing::warn!(error = %e, parser = "facts", "kremory.extraction.json_parse_fail");
                    eprintln!("warn: failed to parse fact JSON after repair: {e}");
                    return Ok(vec![]);
                }
            }
        }
    };

    Ok(raw
        .into_iter()
        .filter(|f| !f.subject.is_empty() && !f.predicate.is_empty() && !f.object.is_empty())
        .map(|f| ExtractedFact {
            subject: f.subject,
            predicate: f.predicate,
            object: f.object,
            is_entity_ref: f.is_entity_ref,
            confidence: f.confidence,
        })
        .collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════════════════════════
// SingleCallExtractor — free discovery: entities + relationships in one pass
// ═══════════════════════════════════════════════════════════════════════════════

/// Unconstrained single-call extractor that asks the LLM to discover all entities
/// and relationships in one pass. The LLM receives `known_entities` as a hint
/// (capped at 50 most recent) but is not constrained to a candidate list.
///
/// This is the primary extractor for the "Free Discovery + Programmatic Audit"
/// architecture. An OOV audit runs separately in `ingest_with()` after this call.
pub struct SingleCallExtractor<L: ChatProvider> {
    llm: Arc<L>,
    prompt_version: PromptVersion,
}

impl<L: ChatProvider> SingleCallExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self {
            llm,
            prompt_version: PromptVersion::default(),
        }
    }

    pub fn with_prompt_version(mut self, version: PromptVersion) -> Self {
        self.prompt_version = version;
        self
    }
}

const SINGLE_CALL_ENTITY_TYPES: &str =
    "Person, Organisation, Location, Technology, Product, Event, Date";

/// Prompt version identifier — bump when changing prompt text.
/// Used by ExtractionMetadata for provenance tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptVersion {
    /// V1: rules-heavy with worked example (~250 prompt tokens).
    /// Validated in spike at 76-80% recall on Qwen 3B.
    V1Rules,
    /// V2: schema-first, minimal rules, no worked example (~150 prompt tokens).
    /// Saves ~100 tokens of prefill budget vs V1. Omits the "no duplicates"
    /// instruction — viable for larger models (qwen2.5:14b+) that dedupe
    /// without being told, but causes intra-batch duplicate emissions on
    /// smaller models (observed 2026-05-28 with llama3.2:3b emitting
    /// `'VerbatimString'` and gemma4-e2b emitting `'car'` multiple times in
    /// a single extraction). Use explicitly when targeting large models with
    /// tight prefill budgets.
    V2SchemaLight,
    /// V3: schema-first + format hint + dedup line (~170 prompt tokens).
    /// Combines V2's schema-first structure (GoLLIE +13 F1 signal) with the
    /// explicit "No duplicates." instruction. Smaller cost than V1 (~80 tok
    /// saved) but reliably dedupes across the supported provider matrix.
    /// **Default since v0.1.4** — preferred for any model class because the
    /// ~20 token cost over V2 is dwarfed by avoided substrate dedup warnings
    /// and lost extraction information.
    #[default]
    V3SchemaHybrid,
}

impl std::fmt::Display for PromptVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1Rules => write!(f, "v1-rules"),
            Self::V2SchemaLight => write!(f, "v2-schema-light"),
            Self::V3SchemaHybrid => write!(f, "v3-schema-hybrid"),
        }
    }
}

fn build_known_hint(ctx: &ExtractionContext<'_>) -> String {
    if ctx.known_entities.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = ctx
            .known_entities
            .iter()
            .rev()
            .take(50) // cap at 50 most recent to prevent unbounded growth
            .map(|e| format!("{} ({})", e.name, e.label))
            .collect();
        format!("\nPreviously seen entities: {}\n", names.join(", "))
    }
}

/// V1: rules-heavy prompt with worked example.
/// ~250 prompt tokens. Validated at 76-80% recall (Qwen 3B, 8 fixtures).
fn build_single_call_prompt_v1(text: &str, ctx: &ExtractionContext<'_>) -> String {
    let known_hint = build_known_hint(ctx);
    let l2_guidance = prompts::build_l2_guidance(ctx.registry_specs);
    let existing_block = prompts::render_existing_entities_block(ctx.existing_graph_entities);

    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: Conversational transcript with speaker labels.\n",
        ContentType::Json => "Format: Structured data fields.\n",
        ContentType::Text | ContentType::Document => "",
    };

    format!(
        "Extract all unique entities and relationships from the text below.\n\n\
{format_hint}\
{existing_block}\
{l2_guidance}\n\
Rules:\n\
- Each entity must appear ONCE (no duplicates)\n\
- Classify entities as: {SINGLE_CALL_ENTITY_TYPES}, or Entity\n\
- Relationships must be unique triples\n\
- Use specific predicates: \"prescribed\", \"works_at\", \"located_in\", \"manages\", etc.\n\
{known_hint}\n\
Example input: \"Alice from Acme Corp met Bob. Alice manages the platform team.\"\n\
Example output:\n\
{{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},\
{{\"name\":\"Bob\",\"label\":\"Person\"}},{{\"name\":\"platform team\",\"label\":\"Organisation\"}}],\
\"relationships\":[{{\"subject\":\"Alice\",\"predicate\":\"works_at\",\"object\":\"Acme Corp\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"met\",\"object\":\"Bob\"}},\
{{\"subject\":\"Alice\",\"predicate\":\"manages\",\"object\":\"platform team\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with \"entities\" and \"relationships\" arrays. No duplicates."
    )
}

/// V2: schema-first, minimal prompt. No worked example, no verbose rules.
/// ~150 prompt tokens — saves ~100 tokens of prefill vs V1.
/// Hypothesis: schema alone is sufficient for small models trained on
/// instruction-following data. GoLLIE research shows schema-first gives
/// +13 F1 — the schema is the key signal, not the rules.
fn build_single_call_prompt_v2(text: &str, ctx: &ExtractionContext<'_>) -> String {
    let known_hint = build_known_hint(ctx);
    let l2_guidance = prompts::build_l2_guidance(ctx.registry_specs);
    let existing_block = prompts::render_existing_entities_block(ctx.existing_graph_entities);

    format!(
        "Extract all entities and relationships from the text.\n\n\
{existing_block}\
{l2_guidance}\n\
Output schema:\n\
{{\"entities\":[{{\"name\":\"<string>\",\"label\":\"<{SINGLE_CALL_ENTITY_TYPES}|Entity>\"}}],\
\"relationships\":[{{\"subject\":\"<entity name>\",\"predicate\":\"<verb>\",\"object\":\"<entity name or value>\"}}]}}\n\
{known_hint}\n\
<TEXT>\n{text}\n</TEXT>\n\n\
JSON:"
    )
}

/// V3: schema-first + format hint + dedup line. Best of V1 and V2.
/// ~170 prompt tokens. Keeps V2's schema-first structure (GoLLIE signal),
/// adds back V1's content-type hint and a single dedup instruction.
fn build_single_call_prompt_v3(text: &str, ctx: &ExtractionContext<'_>) -> String {
    let known_hint = build_known_hint(ctx);
    let l2_guidance = prompts::build_l2_guidance(ctx.registry_specs);
    let existing_block = prompts::render_existing_entities_block(ctx.existing_graph_entities);

    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: conversational transcript.\n",
        ContentType::Json => "Format: structured data.\n",
        ContentType::Text | ContentType::Document => "",
    };

    format!(
        "Extract all entities and relationships from the text.\n\
{format_hint}\n\
{existing_block}\
{l2_guidance}\n\
Output schema:\n\
{{\"entities\":[{{\"name\":\"<string>\",\"label\":\"<{SINGLE_CALL_ENTITY_TYPES}|Entity>\"}}],\
\"relationships\":[{{\"subject\":\"<entity name>\",\"predicate\":\"<verb>\",\"object\":\"<entity name or value>\"}}]}}\n\
No duplicates.\
{known_hint}\n\
<TEXT>\n{text}\n</TEXT>\n\n\
JSON:"
    )
}

/// Build the Stage 1 extraction prompt for a specific version.
pub(crate) fn build_single_call_prompt_versioned(
    text: &str,
    ctx: &ExtractionContext<'_>,
    version: PromptVersion,
) -> String {
    match version {
        PromptVersion::V1Rules => build_single_call_prompt_v1(text, ctx),
        PromptVersion::V2SchemaLight => build_single_call_prompt_v2(text, ctx),
        PromptVersion::V3SchemaHybrid => build_single_call_prompt_v3(text, ctx),
    }
}

impl<L: ChatProvider> EntityExtractor for SingleCallExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        let prompt = build_single_call_prompt_versioned(text, ctx, self.prompt_version);

        let start = Instant::now();
        let sc_msgs = vec![
            chat_msg_system("You are a knowledge graph extraction system. Output valid JSON only."),
            chat_msg_user(prompt),
        ];
        let sc_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_NUEXTRACT_BOTH,
            "NuExtractBoth",
        )
        .messages(sc_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "single_call").record(_ms);
        tracing::info!(_ms, stage = "single_call", "kremory.extraction.stage_ms");
        let resp_text = serde_json::to_string(&sc_value).unwrap_or_default();

        // Reuse the NuExtract parser — same JSON shape {"entities": [...], "relationships": [...]}
        let (entities, facts) = parse_nuextract_response(&resp_text, ctx)?;

        let entity_count = entities.len();
        let fact_count = facts.len();
        histogram!("rql.extraction.entity_count").record(entity_count as f64);
        histogram!("rql.extraction.fact_count").record(fact_count as f64);
        tracing::info!(
            entity_count,
            fact_count,
            extractor = "single_call",
            "kremory.extraction.result"
        );

        Ok(ExtractionResult { entities, facts })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ProgrammaticFirstExtractor — candidates first, LLM for typing + relationships
// ═══════════════════════════════════════════════════════════════════════════════

/// Serde model for LLM entity-typing response (Call 1).
#[derive(Debug, Default, Deserialize)]
struct EntityOnlyOutput {
    #[serde(default)]
    entities: Vec<RawEntity>,
}

/// Serde model for LLM relationship response (Call 2).
#[derive(Debug, Default, Deserialize)]
struct RelOnlyOutput {
    #[serde(default)]
    relationships: Vec<RawRelationship>,
}

/// Programmatic-first extractor: the pipeline generates candidates (zero LLM),
/// then two LLM calls type them and extract relationships.
///
/// Architecture (validated at 95% recall, <5ms programmatic, 2 LLM calls):
///   1. `OovAuditor::extract_candidates()` → ~25 candidates (OOV + PMI + scanner union)
///   2. LLM Call 1: confirm/type candidates + catch remaining ~5% the pipeline missed
///   3. LLM Call 2: extract relationships between confirmed entities
///
/// This is the PRIMARY extractor when an `OovAuditor` is configured.
/// Replaces the old "LLM discovers everything, OOV audits after" pattern.
pub struct ProgrammaticFirstExtractor<L: ChatProvider> {
    llm: Arc<L>,
    auditor: Arc<crate::core::text_utils::OovAuditor>,
    max_candidates: usize,
}

const PROG_ENTITY_TYPES: &str = "Person, Organisation, Location, Technology, Product, Event, Date";

impl<L: ChatProvider> ProgrammaticFirstExtractor<L> {
    pub fn new(
        llm: Arc<L>,
        auditor: Arc<crate::core::text_utils::OovAuditor>,
        max_candidates: usize,
    ) -> Self {
        Self {
            llm,
            auditor,
            max_candidates,
        }
    }
}

fn build_entity_typing_prompt(
    text: &str,
    candidates: &[String],
    ctx: &ExtractionContext<'_>,
) -> String {
    let cand_list = candidates
        .iter()
        .map(|c| format!("\"{}\"", c))
        .collect::<Vec<_>>()
        .join(", ");

    let known_hint = if ctx.known_entities.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = ctx
            .known_entities
            .iter()
            .rev()
            .take(25)
            .map(|e| format!("{} ({})", e.name, e.label))
            .collect();
        format!("\nPreviously seen entities: {}\n", names.join(", "))
    };

    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: Conversational transcript with speaker labels.\n",
        ContentType::Json => "Format: Structured data fields.\n",
        ContentType::Text | ContentType::Document => "",
    };

    let l2_guidance = prompts::build_l2_guidance(ctx.registry_specs);

    format!(
        "Extract all unique entities from the text below.\n\n\
{format_hint}\
{l2_guidance}\n\
Some candidate entities detected automatically: [{cand_list}]\n\
Confirm which are real entities, correct any errors, and add any the system missed.\n\n\
Rules:\n\
- Each entity must appear ONCE (no duplicates)\n\
- Classify as: {PROG_ENTITY_TYPES}, or Entity\n\
{known_hint}\n\
Example:\n\
Text: \"Alice from Acme Corp met Bob.\"\n\
Candidates: [\"Alice\", \"Acme Corp\"]\n\
Output: {{\"entities\":[{{\"name\":\"Alice\",\"label\":\"Person\"}},{{\"name\":\"Acme Corp\",\"label\":\"Organisation\"}},{{\"name\":\"Bob\",\"label\":\"Person\"}}]}}\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with an \"entities\" array only."
    )
}

fn build_relationship_prompt(text: &str, entities: &[ExtractedEntity]) -> String {
    let ent_list = entities
        .iter()
        .map(|e| format!("\"{}\"", e.name))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "Extract relationships between these entities from the text.\n\n\
Entities: [{ent_list}]\n\n\
Rules:\n\
- Unique (subject, predicate, object) triples\n\
- Use specific predicates: \"works_at\", \"prescribed\", \"located_in\", \"manages\", etc.\n\
- Set is_entity_ref=true when object is an entity, false when it's a literal value\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with a \"relationships\" array only."
    )
}

impl<L: ChatProvider> EntityExtractor for ProgrammaticFirstExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        // Step 1: Programmatic pipeline → candidates (zero LLM, <5ms)
        let pipeline_start = Instant::now();
        let candidates = self.auditor.extract_candidates(text, self.max_candidates);
        let _pipeline_ms = pipeline_start.elapsed().as_secs_f64() * 1000.0;
        let candidate_count = candidates.len() as u64;
        histogram!("rql.extraction.stage_ms", "stage" => "programmatic_pipeline")
            .record(_pipeline_ms);
        counter!("rql.extraction.programmatic_candidate_count").increment(candidate_count);
        tracing::info!(
            _pipeline_ms,
            candidate_count,
            stage = "programmatic_pipeline",
            "kremory.extraction.stage_ms"
        );

        // Step 2: LLM Call 1 — type/classify candidates + catch remaining ~5%
        let typing_prompt = build_entity_typing_prompt(text, &candidates, ctx);
        let typing_start = Instant::now();
        let typing_msgs = vec![
            chat_msg_system("You are a knowledge graph extraction system. Output valid JSON only."),
            chat_msg_user(typing_prompt),
        ];
        let typing_value = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_ENTITY_TYPING,
            "EntityTyping",
        )
        .messages(typing_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = typing_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "entity_typing").record(_ms);
        tracing::info!(_ms, stage = "entity_typing", "kremory.extraction.stage_ms");
        let typing_text = serde_json::to_string(&typing_value).unwrap_or_default();

        let entity_output: EntityOnlyOutput = parse_json_lenient(&typing_text).unwrap_or_default();
        let mut entities: Vec<ExtractedEntity> = entity_output
            .entities
            .into_iter()
            .filter(|e| !e.name.is_empty())
            .map(|e| {
                let mut props = serde_json::Map::new();
                props.insert(
                    "name".to_string(),
                    serde_json::Value::String(e.name.clone()),
                );
                ExtractedEntity {
                    name: e.name,
                    label: e.label,
                    properties: serde_json::Value::Object(props),
                }
            })
            .collect();

        // Apply exclusion filter
        if !ctx.excluded_entity_types.is_empty() {
            entities.retain(|e| !ctx.excluded_entity_types.contains(&e.label));
        }

        // Step 3: LLM Call 2 — extract relationships between confirmed entities
        let rel_prompt = build_relationship_prompt(text, &entities);
        let rel_start = Instant::now();
        let rel_msgs = vec![
            chat_msg_system("You are a knowledge graph extraction system. Output valid JSON only."),
            chat_msg_user(rel_prompt),
        ];
        let mut rel_builder = structured::StructuredCallBuilder::new(
            self.llm.as_ref(),
            &schemas::SCHEMA_REL_ONLY_FORCE_FALLBACK,
            "RelOnlyForceFallback",
        )
        .messages(rel_msgs)
        .model(self.llm.model())
        .ttft_budget_ms(ctx.arm_budget_ms);
        if let Some(arm) = schemas::SCHEMA_REL_ONLY_FORCE_FALLBACK_FORCE_ARM {
            rel_builder = rel_builder.force_arm(arm);
        }
        let rel_value = rel_builder
            .call()
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = rel_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "relationships").record(_ms);
        tracing::info!(_ms, stage = "relationships", "kremory.extraction.stage_ms");
        let rel_text = serde_json::to_string(&rel_value).unwrap_or_default();

        let rel_output: RelOnlyOutput = parse_json_lenient(&rel_text).unwrap_or_default();
        let facts: Vec<ExtractedFact> = rel_output
            .relationships
            .into_iter()
            .filter(|r| !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty())
            .map(|r| ExtractedFact {
                subject: r.subject,
                predicate: r.predicate,
                object: r.object,
                is_entity_ref: r.is_entity_ref,
                confidence: r.confidence,
            })
            .collect();

        let entity_count = entities.len();
        let fact_count = facts.len();
        histogram!("rql.extraction.entity_count").record(entity_count as f64);
        histogram!("rql.extraction.fact_count").record(fact_count as f64);
        tracing::info!(
            entity_count,
            fact_count,
            extractor = "programmatic_first",
            "kremory.extraction.result"
        );

        Ok(ExtractionResult { entities, facts })
    }
}

/// Lenient JSON parser: tries raw parse, then llm_json repair, then brace extraction.
fn parse_json_lenient<T: for<'de> serde::Deserialize<'de> + Default>(raw: &str) -> Option<T> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<T>(t) {
        return Some(v);
    }
    let repaired =
        llm_json::repair_json(t, &llm_json::RepairOptions::default()).unwrap_or(t.to_owned());
    if let Ok(v) = serde_json::from_str::<T>(&repaired) {
        return Some(v);
    }
    // Try extracting JSON object from surrounding text
    if let (Some(s), Some(e)) = (t.find('{'), t.rfind('}')) {
        if e > s {
            let slice = &t[s..=e];
            let repaired2 = llm_json::repair_json(slice, &llm_json::RepairOptions::default())
                .unwrap_or(slice.to_owned());
            if let Ok(v) = serde_json::from_str::<T>(&repaired2) {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::core::intelligence::ExtractionContext;
    use crate::core::provider::MockChatProvider;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use std::collections::HashMap;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    /// Build a MockChatProvider that returns staged responses:
    /// - prompt containing "Extract all unique named entities" → stage1_json
    /// - prompt containing "relationship types connect" → stage2_json
    /// - prompt containing "Extract the key relationships" → stage3_json
    fn staged_mock(stage1_json: &str, stage2_json: &str, stage3_json: &str) -> MockChatProvider {
        let mut map = HashMap::new();
        map.insert(
            "Extract all unique named entities".to_string(),
            stage1_json.to_string(),
        );
        map.insert(
            "relationship types connect".to_string(),
            stage2_json.to_string(),
        );
        map.insert(
            "Extract the key relationships".to_string(),
            stage3_json.to_string(),
        );
        MockChatProvider::new(map)
    }

    #[test]
    fn test_default_extractor_with_mock_llm() {
        // TD-013 L1: mock LLM emits integer-ID format; registry resolves ids to labels.
        let stage1 = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme Corp","entity_type_id":2}]}"#;
        let stage2 = r#"["works_at"]"#;
        let stage3 = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]"#;

        let specs = vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ];
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext {
            registry_specs: &specs,
            ..ExtractionContext::default()
        };
        let result = block_on(extractor.extract("Alice works at Acme Corp", &ctx)).unwrap();

        assert_eq!(result.entities.len(), 2, "should extract 2 entities");
        assert_eq!(result.facts.len(), 1, "should extract 1 fact");

        let alice = result.entities.iter().find(|e| e.name == "Alice").unwrap();
        assert_eq!(alice.label, "Person");

        let fact = &result.facts[0];
        assert_eq!(fact.subject, "Alice");
        assert_eq!(fact.predicate, "works_at");
        assert_eq!(fact.object, "Acme Corp");
        assert!(fact.is_entity_ref);
        assert!((fact.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_cascade_stage2_receives_stage1_entities() {
        // TD-013 L1: stage1 mock emits integer-ID format; registry resolves id=2 → "Organisation".
        let stage1 = r#"{"entities":[{"name":"GlobalCorp","entity_type_id":2}]}"#;
        let stage2 = r#"["founded_by"]"#;
        let stage3 = r#"[]"#;

        // The stage2 prompt is keyed on "relationship types connect" which is in build_relation_names_prompt.
        // We verify the function directly builds the prompt with entity info.
        let entities = vec![ExtractedEntity {
            name: "GlobalCorp".to_string(),
            label: "Organisation".to_string(),
            properties: serde_json::Value::Object(serde_json::Map::new()),
        }];
        let prompt = build_relation_names_prompt("some text", &entities, &[]);
        assert!(
            prompt.contains("GlobalCorp"),
            "stage2 prompt must contain entity name from stage1"
        );
        assert!(
            prompt.contains("Organisation"),
            "stage2 prompt must contain entity label from stage1"
        );

        // Also verify the full extractor runs without error.
        let specs = vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ];
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext {
            registry_specs: &specs,
            ..ExtractionContext::default()
        };
        let result = block_on(extractor.extract("GlobalCorp was founded.", &ctx)).unwrap();
        assert_eq!(result.entities.len(), 1);
    }

    #[test]
    fn test_excluded_entities_filtered() {
        // TD-013 L1: mock LLM emits integer-ID format; StopWord is id=3 and should
        // be filtered out by excluded_entity_types matching on the resolved label.
        let stage1 = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"StopWordInc","entity_type_id":3}]}"#;
        let stage2 = r#"[]"#;
        let stage3 = r#"[]"#;

        let specs = vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 3,
                name: "StopWord".to_string(),
                description: "A stop-word entity.".to_string(),
            },
        ];
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);

        let excluded = vec!["StopWord".to_string()];
        let ctx = ExtractionContext {
            excluded_entity_types: &excluded,
            registry_specs: &specs,
            ..ExtractionContext::default()
        };
        let result = block_on(extractor.extract("Alice and StopWordInc", &ctx)).unwrap();

        assert_eq!(
            result.entities.len(),
            1,
            "excluded entity type should be filtered out"
        );
        assert_eq!(result.entities[0].name, "Alice");
        assert!(
            result.entities.iter().all(|e| e.label != "StopWord"),
            "no StopWord entity should remain"
        );
    }

    #[test]
    fn test_empty_llm_response_graceful() {
        // Mock returns empty arrays for all stages.
        let mock = Arc::new(staged_mock("[]", "[]", "[]"));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let result = block_on(extractor.extract("Some text with no matches", &ctx));

        assert!(
            result.is_ok(),
            "empty LLM responses must not cause an error"
        );
        let result = result.unwrap();
        assert!(result.entities.is_empty(), "no entities expected");
        assert!(result.facts.is_empty(), "no facts expected");
    }

    #[test]
    fn test_malformed_llm_response_graceful() {
        // Mock returns malformed JSON that should be gracefully degraded.
        let mock = Arc::new(staged_mock("not valid json {{{", "also bad", "broken"));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let result = block_on(extractor.extract("Some text", &ctx));

        assert!(
            result.is_ok(),
            "malformed LLM response must not cause an error"
        );
        let result = result.unwrap();
        assert!(
            result.entities.is_empty(),
            "graceful degradation: empty entities"
        );
        assert!(result.facts.is_empty(), "graceful degradation: empty facts");
    }

    #[test]
    fn test_ontology_constraint_in_prompt() {
        let allowed = vec!["Person".to_string(), "Organisation".to_string()];
        let prompt = build_entity_prompt("Alice works at Acme.", &allowed, &[]);
        assert!(
            prompt.contains("Person"),
            "stage1 prompt must contain allowed entity type Person"
        );
        assert!(
            prompt.contains("Organisation"),
            "stage1 prompt must contain allowed entity type Organisation"
        );
        assert!(
            prompt.contains("Only extract entities of these types"),
            "stage1 prompt must include the type-constraint hint"
        );
    }

    #[test]
    fn test_parse_entities_handles_empty_array() {
        let result = parse_entities("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_facts_handles_empty_array() {
        let result = parse_facts("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_entities_valid_json() {
        let json = r#"[{"name":"Alice","label":"Person"},{"name":"Acme","label":"Organisation"}]"#;
        let entities = parse_entities(json).unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[0].label, "Person");
        assert_eq!(entities[1].name, "Acme");
        assert_eq!(entities[1].label, "Organisation");
    }

    #[test]
    fn test_parse_facts_valid_json() {
        let json = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.9}]"#;
        let facts = parse_facts(json).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "Alice");
        assert_eq!(facts[0].predicate, "works_at");
        assert!(facts[0].is_entity_ref);
        assert!((facts[0].confidence - 0.9).abs() < 0.001);
    }

    // ─── llm_json repair integration tests ──────────────────────────────

    #[test]
    fn test_parse_entities_repairs_objects_without_array_wrapper() {
        let messy = r#"{"name":"Alice","label":"Person"},{"name":"Bob","label":"Person"}"#;
        let entities = parse_entities(messy).unwrap();
        assert_eq!(
            entities.len(),
            2,
            "should repair unwrapped objects into array"
        );
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[1].name, "Bob");
    }

    #[test]
    fn test_parse_entities_repairs_trailing_comma() {
        let messy = r#"[{"name":"Alice","label":"Person"},{"name":"Bob","label":"Person"},]"#;
        let entities = parse_entities(messy).unwrap();
        assert_eq!(entities.len(), 2, "should handle trailing comma");
    }

    #[test]
    fn test_parse_entities_repairs_markdown_code_fence() {
        let messy = "```json\n[{\"name\":\"Alice\",\"label\":\"Person\"}]\n```";
        let entities = parse_entities(messy).unwrap();
        assert_eq!(entities.len(), 1, "should strip markdown fences");
        assert_eq!(entities[0].name, "Alice");
    }

    #[test]
    fn test_parse_facts_repairs_objects_without_array_wrapper() {
        let messy = r#"{"subject":"Alice","predicate":"works_at","object":"Acme","is_entity_ref":true,"confidence":0.9},{"subject":"Bob","predicate":"manages","object":"Team","is_entity_ref":false,"confidence":0.8}"#;
        let facts = parse_facts(messy).unwrap();
        assert_eq!(
            facts.len(),
            2,
            "should repair unwrapped fact objects into array"
        );
        assert_eq!(facts[0].subject, "Alice");
        assert_eq!(facts[1].subject, "Bob");
    }

    #[test]
    fn test_parse_facts_repairs_single_quotes() {
        let messy = r#"[{'subject':'Alice','predicate':'works_at','object':'Acme','is_entity_ref':true,'confidence':0.9}]"#;
        let facts = parse_facts(messy).unwrap();
        assert_eq!(facts.len(), 1, "should handle single-quoted keys/values");
        assert_eq!(facts[0].subject, "Alice");
    }

    #[test]
    fn test_parse_entities_repairs_python_style_booleans() {
        // Some models output Python-style True/False/None
        let messy = r#"[{"name":"Alice","label":"Person","active":True}]"#;
        let entities = parse_entities(messy).unwrap();
        assert_eq!(entities.len(), 1, "should handle Python-style booleans");
    }

    // ─── Integer-ID parse path tests (TD-013 L1) ──────────────────────────────

    fn make_test_registry() -> crate::core::entity_types::EntityTypeRegistry {
        crate::core::entity_types::EntityTypeRegistry::from_specs(vec![
            crate::core::entity_types::EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            crate::core::entity_types::EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
        ])
    }

    #[test]
    fn parse_entities_integer_resolves_id_to_label() {
        let registry = make_test_registry();
        let json = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme Corp","entity_type_id":2}]}"#;
        let entities = parse_entities_integer(json, &registry).unwrap();
        assert_eq!(entities.len(), 2);
        let alice = entities.iter().find(|e| e.name == "Alice").unwrap();
        assert_eq!(alice.label, "Person", "id=1 must resolve to Person");
        let acme = entities.iter().find(|e| e.name == "Acme Corp").unwrap();
        assert_eq!(
            acme.label, "Organisation",
            "id=2 must resolve to Organisation"
        );
    }

    #[test]
    fn parse_entities_integer_falls_back_to_zero_on_out_of_range() {
        let registry = make_test_registry();
        // id=99 is out of range for this registry (max=2); must fall back to id=0 → "Entity"
        let json = r#"{"entities":[{"name":"Unknown Thing","entity_type_id":99}]}"#;
        let entities = parse_entities_integer(json, &registry).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(
            entities[0].label, "Entity",
            "out-of-range id must fall back to Entity"
        );
    }

    #[test]
    fn parse_entities_integer_rejects_json_fragment_in_name() {
        // Lock the fix for the 2026-06-04 qwen2.5:14b benchmark failure: post-repair
        // garbage where a truncated entity's tail bleeds into the next entity's
        // `name` field (e.g. `Boston", "entity_type_id": 3}, {`). Without the
        // shape-validator these used to silently collapse to label="Entity".
        let registry = make_test_registry();
        let garbage =
            r#"{"entities":[{"name":"Boston\", \"entity_type_id\": 3}, {","entity_type_id":0}]}"#;
        let entities = parse_entities_integer(garbage, &registry).unwrap();
        assert!(
            entities.is_empty(),
            "names containing JSON syntax must be filtered, got: {entities:?}"
        );
    }

    #[test]
    fn parse_entities_integer_rejects_missing_entity_type_id() {
        // After stripping `#[serde(default)]` from RawEntityIntegerId, an entity
        // missing `entity_type_id` must fail deserialize loudly — not collapse
        // to id=0 → "Entity" via serde default. The fn returns `Ok(vec![])` on
        // total parse failure so the fallback ladder can attempt next arm.
        let registry = make_test_registry();
        let no_id = r#"{"entities":[{"name":"Alice"}]}"#;
        let entities = parse_entities_integer(no_id, &registry).unwrap();
        assert!(
            entities.is_empty(),
            "missing entity_type_id must NOT default to 0; got: {entities:?}"
        );
    }

    #[test]
    fn parse_entities_integer_rejects_missing_name() {
        // Symmetric to the above: missing `name` must also fail deserialize
        // rather than default to empty string + silently drop.
        let registry = make_test_registry();
        let no_name = r#"{"entities":[{"entity_type_id":1}]}"#;
        let entities = parse_entities_integer(no_name, &registry).unwrap();
        assert!(
            entities.is_empty(),
            "missing name must NOT default to empty; got: {entities:?}"
        );
    }

    // ─── Metrics assertion tests ─────────────────────────────────────────────

    type Snapshot = Vec<(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )>;

    /// Sum across all counter rows matching `name`, regardless of labels.
    ///
    /// Per [[observability-first-class]] cardinal failure mode #9, post-TD-013
    /// the extractor emits `rql.extraction.json_parse_ok` with `path=wrapped|
    /// bare_array|post_repair` labels alongside legacy unlabeled emission sites.
    /// Each path is its own counter row in the registry, so `find()` (returning
    /// the first match) would under-report. Tests asking "did N parses succeed?"
    /// want the aggregate, so sum.
    fn find_counter(snapshot: &Snapshot, name: &str) -> u64 {
        snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == name)
            .map(|(.., v)| match v {
                DebugValue::Counter(n) => *n,
                _ => 0,
            })
            .sum()
    }

    fn find_histogram(snapshot: &Snapshot, name: &str) -> Vec<f64> {
        snapshot
            .iter()
            .filter(|(k, ..)| k.key().name() == name)
            .flat_map(|(.., v)| match v {
                DebugValue::Histogram(vals) => {
                    vals.iter().map(|v| v.into_inner()).collect::<Vec<_>>()
                }
                _ => vec![],
            })
            .collect()
    }

    #[test]
    fn test_extraction_emits_metrics() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            // TD-013 L1: integer-ID format; registry resolves id=1→Person, id=2→Organisation.
            let stage1 = r#"{"entities":[{"name":"Alice","entity_type_id":1},{"name":"Acme Corp","entity_type_id":2}]}"#;
            let stage2 = r#"["works_at"]"#;
            let stage3 = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]"#;
            let specs = vec![
                crate::core::entity_types::EntityTypeSpec {
                    id: 0,
                    name: "Entity".to_string(),
                    description: "catch-all".to_string(),
                },
                crate::core::entity_types::EntityTypeSpec {
                    id: 1,
                    name: "Person".to_string(),
                    description: "A person.".to_string(),
                },
                crate::core::entity_types::EntityTypeSpec {
                    id: 2,
                    name: "Organisation".to_string(),
                    description: "An org.".to_string(),
                },
            ];
            let mock = Arc::new(staged_mock(stage1, stage2, stage3));
            let extractor = DefaultExtractor::new(mock);
            let ctx = ExtractionContext {
                registry_specs: &specs,
                ..ExtractionContext::default()
            };
            let _result = block_on(extractor.extract("Alice works at Acme Corp", &ctx)).unwrap();

            let snapshot = snapshotter.snapshot().into_vec();

            // 3 extraction stages should emit timing histograms
            let stage_timings = find_histogram(&snapshot, "rql.extraction.stage_ms");
            assert_eq!(
                stage_timings.len(),
                3,
                "should have 3 stage timings (entities, relations, triplets)"
            );
            assert!(
                stage_timings.iter().all(|&v| v >= 0.0),
                "all timings should be non-negative"
            );

            // Entity and fact count histograms
            let entity_counts = find_histogram(&snapshot, "rql.extraction.entity_count");
            assert_eq!(entity_counts.len(), 1);
            assert_eq!(entity_counts[0], 2.0, "should record 2 entities");

            let fact_counts = find_histogram(&snapshot, "rql.extraction.fact_count");
            assert_eq!(fact_counts.len(), 1);
            assert_eq!(fact_counts[0], 1.0, "should record 1 fact");

            // JSON parse success counters (stages 1 and 3 parse JSON)
            let parse_ok = find_counter(&snapshot, "rql.extraction.json_parse_ok");
            assert!(
                parse_ok >= 2,
                "at least 2 JSON parses should succeed (entities + facts)"
            );

            let parse_fail = find_counter(&snapshot, "rql.extraction.json_parse_fail");
            assert_eq!(parse_fail, 0, "no JSON parses should fail with mock data");
        });
    }

    #[test]
    fn test_extraction_metrics_on_json_failure() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            let mock = Arc::new(staged_mock("not valid json {{{", "also bad", "broken"));
            let extractor = DefaultExtractor::new(mock);
            let ctx = ExtractionContext::default();
            let _result = block_on(extractor.extract("Some text", &ctx)).unwrap();

            let snapshot = snapshotter.snapshot().into_vec();

            let parse_fail = find_counter(&snapshot, "rql.extraction.json_parse_fail");
            assert!(
                parse_fail >= 1,
                "should record at least 1 JSON parse failure"
            );

            let entity_counts = find_histogram(&snapshot, "rql.extraction.entity_count");
            assert_eq!(
                entity_counts[0], 0.0,
                "should record 0 entities on parse failure"
            );
        });
    }

    // ─── ProgrammaticFirstExtractor tests ────────────────────────────────────

    fn load_test_auditor() -> crate::core::text_utils::OovAuditor {
        let aff = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .expect("en_US.aff");
        let dic = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .expect("en_US.dic");
        let dict = zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .expect("build dictionary");
        let stops: std::collections::HashSet<String> =
            stop_words::get(stop_words::LANGUAGE::English)
                .into_iter()
                .map(|s| s.to_string())
                .collect();
        crate::core::text_utils::OovAuditor::new(dict, stops)
    }

    #[test]
    fn test_programmatic_first_with_mock_llm() {
        // Mock LLM: Call 1 (entity typing) returns confirmed entities,
        // Call 2 (relationships) returns a relationship triple.
        let mut map = HashMap::new();
        // Call 1 matches "Confirm which are real entities"
        map.insert(
            "Confirm which are real entities".to_string(),
            r#"{"entities":[{"name":"Alice","label":"Person"},{"name":"Acme Corp","label":"Organisation"}]}"#.to_string(),
        );
        // Call 2 matches "Extract relationships between these entities"
        map.insert(
            "Extract relationships between these entities".to_string(),
            r#"{"relationships":[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]}"#.to_string(),
        );
        let mock = Arc::new(MockChatProvider::new(map));
        let auditor = Arc::new(load_test_auditor());

        let extractor = ProgrammaticFirstExtractor::new(mock, auditor, 25);
        let ctx = ExtractionContext::default();

        let result =
            block_on(extractor.extract("Alice works at Acme Corp in the downtown office.", &ctx))
                .unwrap();

        assert!(
            result.entities.len() >= 2,
            "should have at least Alice and Acme Corp, got {:?}",
            result.entities.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
        assert!(
            !result.facts.is_empty(),
            "should have at least 1 relationship"
        );
        assert_eq!(result.facts[0].predicate, "works_at");
    }

    #[test]
    fn test_programmatic_first_empty_llm_still_runs() {
        // When LLM returns empty, extractor should return empty without error
        let map = HashMap::new();
        let mock = Arc::new(MockChatProvider::new(map));
        let auditor = Arc::new(load_test_auditor());

        let extractor = ProgrammaticFirstExtractor::new(mock, auditor, 25);
        let ctx = ExtractionContext::default();

        let result = block_on(extractor.extract("Alice works at Acme Corp.", &ctx)).unwrap();

        // LLM returned empty → no entities confirmed, no relationships
        assert!(
            result.entities.is_empty(),
            "empty LLM should yield no typed entities"
        );
    }

    #[test]
    fn test_parse_json_lenient_valid() {
        let parsed: Option<EntityOnlyOutput> =
            parse_json_lenient(r#"{"entities":[{"name":"Alice","label":"Person"}]}"#);
        assert!(parsed.is_some());
        assert_eq!(parsed.unwrap().entities.len(), 1);
    }

    #[test]
    fn test_parse_json_lenient_with_surrounding_text() {
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(
            r#"Here is the result: {"entities":[{"name":"Bob","label":"Person"}]} Done."#,
        );
        assert!(parsed.is_some());
        assert_eq!(parsed.unwrap().entities[0].name, "Bob");
    }

    #[test]
    fn test_parse_json_lenient_empty() {
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient("");
        assert!(parsed.is_none());
    }

    // ── fix_unclosed_string_before_brace ──────────────────────────────────────

    #[test]
    fn test_fix_unclosed_string_noop_on_valid_json() {
        // Well-formed JSON must pass through unchanged.
        let valid = r#"{"entities": [{"name": "Alice", "label": "Person"}]}"#;
        assert_eq!(fix_unclosed_string_before_brace(valid), valid);
    }

    #[test]
    fn test_fix_unclosed_string_repairs_gemma4_output() {
        // Reproduce the exact gemma4-e2b failure: labels missing closing `"` before `}`.
        let malformed = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Bob", "label": "Person}, {"name": "Stanford", "label": "Place}], "relationships": []}"#;
        let fixed = fix_unclosed_string_before_brace(malformed);
        let parsed: serde_json::Value =
            serde_json::from_str(&fixed).expect("fixed output must be valid JSON");
        let entities = parsed["entities"]
            .as_array()
            .expect("entities must be array");
        assert_eq!(entities.len(), 3, "all 3 entities must survive the fix");
        assert_eq!(entities[0]["name"], "Alice");
        assert_eq!(entities[1]["label"], "Person");
        assert_eq!(entities[2]["name"], "Stanford");
    }

    #[test]
    fn test_fix_unclosed_string_handles_empty() {
        assert_eq!(fix_unclosed_string_before_brace(""), "");
    }

    #[test]
    fn test_parse_nuextract_response_repairs_gemma4_unclosed_labels() {
        // Integration: verify parse_nuextract_response recovers from gemma4-e2b output.
        let malformed = r#"{"entities": [{"name": "Alice", "label": "Person"}, {"name": "Bob", "label": "Person}, {"name": "Stanford", "label": "Place}], "relationships": []}"#;
        let ctx = ExtractionContext {
            known_entities: &[],
            allowed_entity_types: &[],
            allowed_edge_types: &[],
            excluded_entity_types: &[],
            content_type: crate::core::config::ContentType::Text,
            registry_specs: &[],
            existing_graph_entities: &[],
            arm_budget_ms: 30_000,
        };
        let (entities, _facts) =
            parse_nuextract_response(malformed, &ctx).expect("must not return Err");
        assert_eq!(entities.len(), 3, "all 3 entities must be extracted");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Alice"), "Alice must be extracted");
        assert!(names.contains(&"Bob"), "Bob must be extracted");
        assert!(names.contains(&"Stanford"), "Stanford must be extracted");
    }

    // ── L1: Adversarial parse_json_lenient — placeholder-label inputs ────────
    //
    // These tests catch the exact TD-012 failure mode: valid JSON that parses
    // successfully but carries placeholder labels ("Entity", "UNKNOWN", "").
    // The parser itself accepts such inputs (it only checks syntax), so these
    // tests document and pin the parser's behaviour and drive the L2 validator
    // to be the correct rejection layer.

    #[test]
    fn parse_json_lenient_placeholder_entity_label_parses_but_is_flagged() {
        // TD-012 exact shape: valid JSON with label="Entity" (placeholder).
        // parse_json_lenient MUST parse it — the syntax is correct.
        // is_canonical_entity_type MUST reject it — it's a placeholder.
        let json = r#"{"entities":[{"name":"Alice","label":"Entity"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(
            parsed.is_some(),
            "parse_json_lenient must accept syntactically valid JSON even with placeholder label"
        );
        let entities = &parsed.unwrap().entities;
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].label, "Entity");
        // L2 validator must reject this label — it's the placeholder, not a type
        assert!(
            !is_canonical_entity_type(&entities[0].label),
            "is_canonical_entity_type must reject 'Entity' (TD-012 placeholder) — got: {}",
            entities[0].label
        );
    }

    #[test]
    fn parse_json_lenient_unknown_label_parses_but_is_flagged() {
        // "UNKNOWN" is another common placeholder emitted by poorly-constrained models.
        let json = r#"{"entities":[{"name":"OpenAI","label":"UNKNOWN"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(parsed.is_some(), "must parse syntactically valid JSON");
        let entities = &parsed.unwrap().entities;
        assert_eq!(entities[0].label, "UNKNOWN");
        assert!(
            !is_canonical_entity_type(&entities[0].label),
            "is_canonical_entity_type must reject 'UNKNOWN'"
        );
    }

    #[test]
    fn parse_json_lenient_empty_label_uses_default_and_is_flagged() {
        // When label field is absent the default is "Entity" — also a placeholder.
        let json = r#"{"entities":[{"name":"Alice"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(parsed.is_some(), "must parse JSON with missing label field");
        let entities = &parsed.unwrap().entities;
        // Default serde fills in "Entity"
        assert_eq!(
            entities[0].label, "Entity",
            "missing label must default to 'Entity'"
        );
        assert!(
            !is_canonical_entity_type(&entities[0].label),
            "default 'Entity' label must be rejected by is_canonical_entity_type"
        );
    }

    #[test]
    fn parse_json_lenient_mixed_canonical_and_placeholder_labels() {
        // When a batch has some canonical and some placeholder labels, the parser
        // accepts the whole batch and the validator rejects only the bad ones.
        let json =
            r#"{"entities":[{"name":"Alice","label":"Person"},{"name":"blob","label":"Entity"}]}"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        assert!(parsed.is_some(), "must parse mixed-label JSON");
        let entities = &parsed.unwrap().entities;
        assert_eq!(entities.len(), 2);
        assert!(
            is_canonical_entity_type(&entities[0].label),
            "Person must be canonical"
        );
        assert!(
            !is_canonical_entity_type(&entities[1].label),
            "'Entity' placeholder must be rejected"
        );
    }

    #[test]
    fn parse_json_lenient_malformed_braces_triggers_repair() {
        // Unbalanced braces — parser should attempt llm_json repair and succeed.
        let json = r#"{"entities":[{"name":"Alice","label":"Person""#;
        // The parser may or may not recover from this — document the behaviour.
        // If repair succeeds, the result must be Some with at least name "Alice".
        // If repair fails, None is returned — that is also acceptable.
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        if let Some(out) = parsed {
            // Repair succeeded: verify the entity is intact
            if !out.entities.is_empty() {
                assert_eq!(out.entities[0].name, "Alice");
            }
        }
        // None is also acceptable — parser did not fabricate data
    }

    #[test]
    fn parse_json_lenient_array_root_falls_back_to_default() {
        // Array at root doesn't match EntityOnlyOutput directly, but the JSON repair + brace
        // extraction path produces an empty-entities result (because EntityOnlyOutput's
        // `entities` field has `#[serde(default)]`).
        //
        // The important invariant: NO entities are returned from an array-root response.
        // This prevents an LLM that emits a raw array (instead of {"entities":[...]}) from
        // injecting untyped entries — the caller sees zero entities and retries / falls back.
        let json = r#"[{"name":"Alice","label":"Person"}]"#;
        let parsed: Option<EntityOnlyOutput> = parse_json_lenient(json);
        let entity_count = parsed.map(|o| o.entities.len()).unwrap_or(0);
        assert_eq!(
            entity_count, 0,
            "array-root response must yield zero entities from EntityOnlyOutput — \
             the shape mismatch should suppress extraction, not smuggle in unlabelled entities"
        );
    }

    // ── L2: ENTITY_TYPE_ALLOWLIST + is_canonical_entity_type tests ───────────

    #[test]
    fn is_canonical_entity_type_accepts_person() {
        assert!(
            is_canonical_entity_type("Person"),
            "Person must be in the canonical entity type allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_accepts_organisation() {
        assert!(
            is_canonical_entity_type("Organisation"),
            "Organisation must be in the canonical entity type allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_accepts_location() {
        assert!(
            is_canonical_entity_type("Location"),
            "Location must be in the canonical entity type allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_entity_placeholder() {
        assert!(
            !is_canonical_entity_type("Entity"),
            "'Entity' is a placeholder label (TD-012) — must be rejected by the allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_unknown_placeholder() {
        assert!(
            !is_canonical_entity_type("UNKNOWN"),
            "'UNKNOWN' is a placeholder label — must be rejected by the allowlist"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_empty_string() {
        assert!(
            !is_canonical_entity_type(""),
            "empty string is not a valid entity type — must be rejected"
        );
    }

    // ─── TD-013 PR1-corrected: mechanism semantics changed (2026-06-03) ──────
    //
    // is_canonical_entity_type previously required positive-allowlist match.
    // Now it requires structural validity + placeholder reject only — aligned
    // with Graphiti's `validate_node_labels` (regex-only) + peer ecosystem
    // consensus. Tests below assert NEW semantics.

    #[test]
    fn is_canonical_entity_type_accepts_arbitrary_valid_string() {
        // Novel labels NOT in any historical canonical list are ACCEPTED if
        // structurally valid. Aligns with Graphiti/Mem0/Cognee: no positive
        // allowlist; LLM is the type-discoverer.
        assert!(
            is_canonical_entity_type("SomeRandomType"),
            "structurally-valid novel types are accepted (no positive allowlist)"
        );
        assert!(
            is_canonical_entity_type("Court"),
            "ground-truth domain type 'Court' accepted (was missing from old allowlist)"
        );
        assert!(
            is_canonical_entity_type("Species"),
            "ground-truth domain type 'Species' accepted (was missing from old allowlist)"
        );
    }

    #[test]
    fn is_canonical_entity_type_accepts_case_variants() {
        // Case variants pass structural validity; canonical case-folding is
        // the responsibility of `normalize_label`, NOT this validator.
        assert!(
            is_canonical_entity_type("person"),
            "'person' (lowercase) is structurally valid; normalize_label maps to 'Person'"
        );
        assert!(
            is_canonical_entity_type("PERSON"),
            "'PERSON' (all-caps) is structurally valid; normalize_label maps to 'Person'"
        );
        assert!(
            is_canonical_entity_type("ORGANISATION"),
            "'ORGANISATION' (all-caps) is structurally valid"
        );
    }

    #[test]
    fn is_canonical_entity_type_rejects_structural_junk() {
        // Single-char, leading-digit, all-punctuation, or excessively-long
        // labels are structurally invalid (Graphiti-style Cypher safety pattern).
        assert!(!is_canonical_entity_type("A"), "single char rejected");
        assert!(
            !is_canonical_entity_type("123Type"),
            "leading digit rejected"
        );
        assert!(
            !is_canonical_entity_type("!!!"),
            "punctuation-only rejected"
        );
        assert!(
            !is_canonical_entity_type(&"X".repeat(100)),
            "overlong label rejected"
        );
    }

    #[test]
    fn normalize_label_canonicalizes_case() {
        assert_eq!(normalize_label("PERSON"), "Person");
        assert_eq!(normalize_label("person"), "Person");
        assert_eq!(normalize_label("Person"), "Person");
    }

    #[test]
    fn normalize_label_handles_us_uk_spelling() {
        // qwen2.5:14b emits "Organization" (US); kremory's canonical is "Organisation" (UK).
        // Same source of the original B3 mismatch documented in Vera review.
        assert_eq!(normalize_label("Organization"), "Organisation");
        assert_eq!(normalize_label("ORGANIZATION"), "Organisation");
        assert_eq!(normalize_label("Organisation"), "Organisation");
        // Also handle common abbreviations + synonyms.
        assert_eq!(normalize_label("Company"), "Organisation");
        assert_eq!(normalize_label("ORG"), "Organisation");
    }

    #[test]
    fn normalize_label_preserves_novel_labels() {
        // Novel labels not in alias map pass through unchanged (trimmed).
        assert_eq!(normalize_label("Court"), "Court");
        assert_eq!(normalize_label("Software"), "Software");
        assert_eq!(normalize_label("  Species  "), "Species");
    }

    #[test]
    fn canonical_forms_constant_pass_validation() {
        // ENTITY_TYPE_CANONICAL_FORMS contains the canonical reference set
        // for prompt interpolation. Each form must pass structural validity.
        for canonical in super::ENTITY_TYPE_CANONICAL_FORMS {
            assert!(
                is_canonical_entity_type(canonical),
                "canonical form '{canonical}' must pass is_canonical_entity_type"
            );
        }
    }

    // ─── L5: RawEntitySimple label array-coercion (qwen2.5:14b compat) ──────

    #[test]
    fn raw_entity_simple_label_coerces_array_to_string() {
        // qwen2.5:14b sometimes emits {"name": "Alice", "label": ["Person"]} —
        // deser_string_or_array coerces to RawEntitySimple { label: "Person", ... }
        let json = r#"{"name": "Alice", "label": ["Person"]}"#;
        let parsed: RawEntitySimple = serde_json::from_str(json)
            .expect("array-shaped label should coerce via deser_string_or_array");
        assert_eq!(parsed.name, "Alice");
        assert_eq!(parsed.label, "Person");
    }

    #[test]
    fn raw_entity_simple_label_accepts_string_unchanged() {
        let json = r#"{"name": "Bob", "label": "Person"}"#;
        let parsed: RawEntitySimple = serde_json::from_str(json).expect("string label");
        assert_eq!(parsed.name, "Bob");
        assert_eq!(parsed.label, "Person");
    }

    #[test]
    fn raw_entity_simple_label_joins_multiple_elements() {
        // deser_string_or_array joins array elements with ", " — matches
        // the same convention used for RawRelationship.subject/predicate/object.
        let json = r#"{"name": "Carol", "label": ["Person", "Politician"]}"#;
        let parsed: RawEntitySimple = serde_json::from_str(json)
            .expect("multi-element array should join via deser_string_or_array");
        assert_eq!(parsed.name, "Carol");
        assert_eq!(parsed.label, "Person, Politician");
    }
}
