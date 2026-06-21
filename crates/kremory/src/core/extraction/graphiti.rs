//! LlmExtractor and Graphiti-quality prompt builders.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use metrics::histogram;
use tracing;

use super::parsers::{parse_entities_integer, parse_facts};
use super::{prompts, schemas, structured};
use crate::core::error::Result;
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};

// ═══════════════════════════════════════════════════════════════════════════════
// LlmExtractor — Graphiti-quality prompts, 3-stage, unconstrained
// ═══════════════════════════════════════════════════════════════════════════════

/// 3-stage extractor with Graphiti-quality prompts: detailed instructions,
/// exclusion rules, worked examples, and the "Wikipedia test" quality filter.
/// Uses unconstrained generation + llm_json repair (no grammar constraints).
pub struct LlmExtractor<L: ChatProvider> {
    llm: Arc<L>,
}

impl<L: ChatProvider> LlmExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }
}

impl<L: ChatProvider> EntityExtractor for LlmExtractor<L> {
    fn name(&self) -> &'static str {
        "llm"
    }

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

// ─── Graphiti prompt constants ────────────────────────────────────────────────

pub(super) const GRAPHITI_ENTITY_SYSTEM: &str = "\
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
pub(super) const DEFAULT_ENTITY_TYPES: &[&str] = &[
    "Person",
    "Organisation",
    "Location",
    "Technology",
    "Product",
    "Event",
    "Date",
];

pub(super) fn build_graphiti_entity_prompt(
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

pub(super) const GRAPHITI_RELATIONSHIP_SYSTEM: &str = "\
You are a relationship extraction system. Given a list of entities found in text, extract the factual relationships between them. Output valid JSON only.

Rules:
- Every relationship must connect two entities from the provided list.
- Use specific, descriptive predicates (e.g. 'prescribed', 'works_at', 'reported_to', 'diagnosed_with').
- Include temporal relationships when stated (e.g. 'joined_in', 'left_on').
- Extract each distinct relationship once — do not repeat.
- Set is_entity_ref to true when the object is an entity from the list, false when it's a literal value.
- Set confidence between 0.0 and 1.0 based on how explicitly the relationship is stated.";

pub(super) fn build_graphiti_relationship_prompt(
    text: &str,
    entities: &[ExtractedEntity],
) -> String {
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

// ─── Prompt builders (used by IntegerIdLlmExtractor + tests) ──────────────────────

/// Build entity extraction prompt for the 3-stage IntegerIdLlmExtractor pipeline.
pub(crate) fn build_entity_prompt(
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

/// Build relationship-type-names prompt for the 3-stage IntegerIdLlmExtractor pipeline.
pub(crate) fn build_relation_names_prompt(
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

/// Build full-triplet prompt for the 3-stage IntegerIdLlmExtractor pipeline.
pub(crate) fn build_triplet_prompt(
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
