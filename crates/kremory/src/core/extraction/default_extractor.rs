//! Three-stage default extractor.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use metrics::histogram;
use tracing;

use super::graphiti::{build_entity_prompt, build_relation_names_prompt, build_triplet_prompt};
use super::parsers::{parse_entities_integer, parse_facts, parse_relation_names};
use super::{schemas, structured};
use crate::core::error::Result;
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};

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
