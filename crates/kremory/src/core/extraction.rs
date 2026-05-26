use std::sync::Arc;
use std::time::Instant;

use metrics::{counter, histogram};
use serde::Deserialize;
use tracing;

use crate::core::error::Result;

use crate::core::config::ContentType;
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};

// ─── Serde models for LLM JSON output coercion ──────────────────────────────

/// Top-level NuExtract output. Both fields are optional — LLMs may omit one.
#[derive(Debug, Deserialize)]
struct NuExtractOutput {
    #[serde(default)]
    entities: Vec<RawEntity>,
    #[serde(default)]
    relationships: Vec<RawRelationship>,
}

/// Entity as emitted by the LLM. `label` defaults to "Entity" when missing.
#[derive(Debug, Deserialize)]
struct RawEntity {
    #[serde(default)]
    name: String,
    #[serde(default = "default_entity_label")]
    label: String,
}

fn default_entity_label() -> String {
    "Entity".to_string()
}

/// Relationship triplet as emitted by the LLM.
#[derive(Debug, Deserialize)]
struct RawRelationship {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    is_entity_ref: bool,
    #[serde(default = "default_confidence")]
    confidence: f64,
}

fn default_confidence() -> f64 {
    1.0
}

/// Entity array as emitted by the DefaultExtractor (stage 1).
#[derive(Debug, Deserialize)]
struct RawEntitySimple {
    #[serde(default)]
    name: String,
    #[serde(default)]
    label: String,
}

/// Fact triplet as emitted by the DefaultExtractor (stage 3).
#[derive(Debug, Deserialize)]
struct RawFact {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    is_entity_ref: bool,
    #[serde(default = "default_confidence")]
    confidence: f64,
}

// ─── JSON Schema Constants (for llguidance constrained decoding) ─────────────

/// Stage 1: Extract entity nodes — array of {name, label}.
pub const SCHEMA_ENTITY_LIST: &str = r#"{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"label":{"type":"string"}},"required":["name","label"],"additionalProperties":false}}"#;

/// Stage 2: Extract relationship names — array of strings.
pub const SCHEMA_RELATION_NAMES: &str = r#"{"type":"array","items":{"type":"string"}}"#;

/// Stage 3: Extract full triplets — array of {subject, predicate, object, is_entity_ref, confidence}.
pub const SCHEMA_TRIPLET_LIST: &str = r#"{"type":"array","items":{"type":"object","properties":{"subject":{"type":"string"},"predicate":{"type":"string"},"object":{"type":"string"},"is_entity_ref":{"type":"boolean"},"confidence":{"type":"number"}},"required":["subject","predicate","object","is_entity_ref","confidence"],"additionalProperties":false}}"#;

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
            build_entity_prompt(text, ctx.allowed_entity_types)
        );
        let stage1_start = Instant::now();
        let stage1_msgs = vec![
            chat_msg_system("You are an entity extraction system. Extract named entities from text. Each entity must appear ONCE — no duplicates. Output valid JSON only."),
            chat_msg_user(stage1_prompt),
        ];
        let stage1_resp = self
            .llm
            .chat_with_tools(&stage1_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::ExtractionStage {
                stage: "entities".to_string(),
                detail: e.to_string(),
            })?;
        let _ms = stage1_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "entities").record(_ms);
        tracing::info!(_ms, stage = "entities", "kremory.extraction.stage_ms");
        let stage1_text = stage1_resp.text().unwrap_or_default();
        let mut entities: Vec<ExtractedEntity> = parse_entities(&stage1_text)?;

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
        let stage2_resp = self
            .llm
            .chat_with_tools(&stage2_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::ExtractionStage {
                stage: "relations".to_string(),
                detail: e.to_string(),
            })?;
        let _ms = stage2_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "relations").record(_ms);
        tracing::info!(_ms, stage = "relations", "kremory.extraction.stage_ms");
        let stage2_text = stage2_resp.text().unwrap_or_default();
        let relation_names: Vec<String> = parse_relation_names(&stage2_text)?;

        // Stage 3: Extract full triplets (with stage 1 + 2 context).
        let stage3_prompt = build_triplet_prompt(text, &entities, &relation_names);
        let stage3_start = Instant::now();
        let stage3_msgs = vec![
            chat_msg_system("You are a knowledge graph extraction system. Extract (subject, predicate, object) triplets. Output valid JSON only."),
            chat_msg_user(stage3_prompt),
        ];
        let stage3_resp = self
            .llm
            .chat_with_tools(&stage3_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = stage3_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "triplets").record(_ms);
        tracing::info!(_ms, stage = "triplets", "kremory.extraction.stage_ms");
        let stage3_text = stage3_resp.text().unwrap_or_default();
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
        let resp = self
            .llm
            .chat_with_tools(&nuextract_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "nuextract").record(_ms);
        tracing::info!(_ms, stage = "nuextract", "kremory.extraction.stage_ms");
        let resp_text = resp.text().unwrap_or_default();

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
        let pass1_resp = self
            .llm
            .chat_with_tools(&pass1_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "grounded_entities").record(_ms);
        tracing::info!(
            _ms,
            stage = "grounded_entities",
            "kremory.extraction.stage_ms"
        );
        let pass1_text = pass1_resp.text().unwrap_or_default();

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
        let pass2_resp = self
            .llm
            .chat_with_tools(&pass2_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start2.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "grounded_relationships").record(_ms);
        tracing::info!(
            _ms,
            stage = "grounded_relationships",
            "kremory.extraction.stage_ms"
        );
        let pass2_text = pass2_resp.text().unwrap_or_default();

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
        let stage1_prompt = format!(
            "{}{known_hint}",
            build_graphiti_entity_prompt(text, ctx.allowed_entity_types)
        );
        let stage1_start = Instant::now();
        let graphiti_s1_msgs = vec![
            chat_msg_system(GRAPHITI_ENTITY_SYSTEM),
            chat_msg_user(stage1_prompt),
        ];
        let stage1_resp = self
            .llm
            .chat_with_tools(&graphiti_s1_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = stage1_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "graphiti_entities").record(_ms);
        tracing::info!(
            _ms,
            stage = "graphiti_entities",
            "kremory.extraction.stage_ms"
        );
        let stage1_text = stage1_resp.text().unwrap_or_default();
        let mut entities: Vec<ExtractedEntity> = parse_entities(&stage1_text)?;

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
        let stage2_resp = self
            .llm
            .chat_with_tools(&graphiti_s2_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = stage2_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "graphiti_relationships").record(_ms);
        tracing::info!(
            _ms,
            stage = "graphiti_relationships",
            "kremory.extraction.stage_ms"
        );
        let stage2_text = stage2_resp.text().unwrap_or_default();
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

fn build_graphiti_entity_prompt(text: &str, allowed_types: &[String]) -> String {
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

    format!("\
{types}

<TEXT>
{text}
</TEXT>

Extract all named entities from the TEXT above. Output a JSON array of objects with \"name\" and \"label\" fields.

Examples:
- Speaker line \"Dr. Patel: The test results...\" → {{\"name\": \"Dr. Patel\", \"label\": \"Person\"}}
- Self-introduction \"Hi, I'm Ria\" → {{\"name\": \"Ria\", \"label\": \"Person\"}}
- \"studied at Northeastern University\" → {{\"name\": \"Northeastern University\", \"label\": \"Organisation\"}}
- \"prescribed Metformin 500mg\" → {{\"name\": \"Metformin\", \"label\": \"Drug\"}}
- \"works at Acme Corp\" → {{\"name\": \"Acme Corp\", \"label\": \"Organisation\"}}
- \"originally from Morocco\" → {{\"name\": \"Morocco\", \"label\": \"Location\"}}
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
    let mut entities: Vec<ExtractedEntity> = output
        .entities
        .into_iter()
        .filter(|e| !e.name.is_empty())
        .map(|e| ExtractedEntity {
            name: e.name,
            label: if e.label.is_empty() {
                "Entity".to_string()
            } else {
                e.label
            },
            properties: serde_json::Value::Object(serde_json::Map::new()),
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

fn build_entity_prompt(text: &str, allowed_types: &[String]) -> String {
    let type_hint = if allowed_types.is_empty() {
        String::new()
    } else {
        format!(
            "\nOnly extract entities of these types: {}",
            allowed_types.join(", ")
        )
    };
    format!(
        "Extract all unique named entities from the following text. Each entity must appear exactly once.{type_hint}\n\nText: {text}\n\nOutput a JSON array of objects with \"name\" and \"label\" fields. No duplicates."
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
        .map(|e| ExtractedEntity {
            name: e.name,
            label: e.label,
            properties: serde_json::Value::Object(serde_json::Map::new()),
        })
        .collect())
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
    /// Saves ~100 tokens of prefill budget. Needs benchmarking.
    #[default]
    V2SchemaLight,
    /// V3: schema-first + format hint + dedup line (~170 prompt tokens).
    /// Combines V2's schema-first structure with V1's content-type awareness.
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

    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: Conversational transcript with speaker labels.\n",
        ContentType::Json => "Format: Structured data fields.\n",
        ContentType::Text | ContentType::Document => "",
    };

    format!(
        "Extract all unique entities and relationships from the text below.\n\n\
{format_hint}\
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

    format!(
        "Extract all entities and relationships from the text.\n\n\
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

    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: conversational transcript.\n",
        ContentType::Json => "Format: structured data.\n",
        ContentType::Text | ContentType::Document => "",
    };

    format!(
        "Extract all entities and relationships from the text.\n\
{format_hint}\n\
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
pub fn build_single_call_prompt_versioned(
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
        let resp = self
            .llm
            .chat_with_tools(&sc_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "single_call").record(_ms);
        tracing::info!(_ms, stage = "single_call", "kremory.extraction.stage_ms");
        let resp_text = resp.text().unwrap_or_default();

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

    format!(
        "Extract all unique entities from the text below.\n\n\
{format_hint}\
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
        let typing_resp = self
            .llm
            .chat_with_tools(&typing_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = typing_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "entity_typing").record(_ms);
        tracing::info!(_ms, stage = "entity_typing", "kremory.extraction.stage_ms");
        let typing_text = typing_resp.text().unwrap_or_default();

        let entity_output: EntityOnlyOutput = parse_json_lenient(&typing_text).unwrap_or_default();
        let mut entities: Vec<ExtractedEntity> = entity_output
            .entities
            .into_iter()
            .filter(|e| !e.name.is_empty())
            .map(|e| ExtractedEntity {
                name: e.name,
                label: e.label,
                properties: serde_json::json!({}),
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
        let rel_resp = self
            .llm
            .chat_with_tools(&rel_msgs, None, None)
            .await
            .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = rel_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "relationships").record(_ms);
        tracing::info!(_ms, stage = "relationships", "kremory.extraction.stage_ms");
        let rel_text = rel_resp.text().unwrap_or_default();

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
    use super::*;
    use crate::core::intelligence::ExtractionContext;
    use crate::core::provider::MockChatProvider;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use std::collections::HashMap;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
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
        let stage1 =
            r#"[{"name":"Alice","label":"Person"},{"name":"Acme Corp","label":"Organisation"}]"#;
        let stage2 = r#"["works_at"]"#;
        let stage3 = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]"#;

        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext::default();
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
        // Use a mock that records prompts via a capturing approach.
        // Since MockChatProvider matches by substring, we verify by checking that
        // a stage2 prompt is built containing the entity name from stage1.
        let stage1 = r#"[{"name":"GlobalCorp","label":"Organisation"}]"#;
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
        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);
        let ctx = ExtractionContext::default();
        let result = block_on(extractor.extract("GlobalCorp was founded.", &ctx)).unwrap();
        assert_eq!(result.entities.len(), 1);
    }

    #[test]
    fn test_excluded_entities_filtered() {
        let stage1 =
            r#"[{"name":"Alice","label":"Person"},{"name":"StopWordInc","label":"StopWord"}]"#;
        let stage2 = r#"[]"#;
        let stage3 = r#"[]"#;

        let mock = Arc::new(staged_mock(stage1, stage2, stage3));
        let extractor = DefaultExtractor::new(mock);

        let excluded = vec!["StopWord".to_string()];
        let ctx = ExtractionContext {
            excluded_entity_types: &excluded,
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
        let prompt = build_entity_prompt("Alice works at Acme.", &allowed);
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
    fn test_json_schemas_are_valid() {
        for (name, schema) in [
            ("SCHEMA_ENTITY_LIST", SCHEMA_ENTITY_LIST),
            ("SCHEMA_RELATION_NAMES", SCHEMA_RELATION_NAMES),
            ("SCHEMA_TRIPLET_LIST", SCHEMA_TRIPLET_LIST),
        ] {
            let parsed: serde_json::Value = serde_json::from_str(schema)
                .unwrap_or_else(|e| panic!("{name} is not valid JSON: {e}"));
            assert_eq!(parsed["type"], "array", "{name} must be an array schema");
        }
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

    // ─── Metrics assertion tests ─────────────────────────────────────────────

    type Snapshot = Vec<(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )>;

    fn find_counter(snapshot: &Snapshot, name: &str) -> u64 {
        snapshot
            .iter()
            .find(|(k, ..)| k.key().name() == name)
            .map(|(.., v)| match v {
                DebugValue::Counter(n) => *n,
                _ => 0,
            })
            .unwrap_or(0)
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
            let stage1 = r#"[{"name":"Alice","label":"Person"},{"name":"Acme Corp","label":"Organisation"}]"#;
            let stage2 = r#"["works_at"]"#;
            let stage3 = r#"[{"subject":"Alice","predicate":"works_at","object":"Acme Corp","is_entity_ref":true,"confidence":0.9}]"#;
            let mock = Arc::new(staged_mock(stage1, stage2, stage3));
            let extractor = DefaultExtractor::new(mock);
            let ctx = ExtractionContext::default();
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
            result.facts.len() >= 1,
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
}
