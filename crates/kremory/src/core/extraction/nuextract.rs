//! NuExtract-based extractors.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use metrics::{counter, histogram};
use tracing;

use super::json_repair::parse_nuextract_response;
use super::{schemas, structured};
use crate::core::config::ContentType;
use crate::core::error::Result;
use crate::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use crate::core::provider::{chat_msg_user, ChatProvider};

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
pub(super) const NUEXTRACT_TEMPLATE: &str = r#"{"entities": [{"name": "verbatim-string", "label": "verbatim-string"}], "relationships": [{"subject": "verbatim-string", "predicate": "verbatim-string", "object": "verbatim-string"}]}"#;

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

// ─── NuExtract template builders ────────────────────────────────────────────

pub(super) fn build_entity_only_template(allowed_entity_types: &[String]) -> String {
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

pub(super) fn build_grounded_relationship_template(
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

pub(super) fn build_nuextract_template(
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
