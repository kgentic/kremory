//! ProgrammaticFirstExtractor — candidates first, LLM for typing + relationships.
//!
//! Split from `mod.rs` during a module reorganization.

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use metrics::{counter, histogram};
use serde::Deserialize;
use tracing;

use super::models::{RawEntity, RawRelationship};
use super::parsers::{emit_parse_yield_metrics, parse_items, ParsePath};
use super::{prompts, schemas, structured};
use crate::core::config::ContentType;
use crate::core::error::Result;
use crate::core::intelligence::{
    EntityExtractor, ExtractedEntity, ExtractedFact, ExtractionContext, ExtractionResult,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};

// ═══════════════════════════════════════════════════════════════════════════════
// ProgrammaticFirstExtractor — candidates first, LLM for typing + relationships
// ═══════════════════════════════════════════════════════════════════════════════

/// Serde model for LLM entity-typing response (Call 1).
#[derive(Debug, Default, Deserialize)]
pub(crate) struct EntityOnlyOutput {
    #[serde(default)]
    pub(crate) entities: Vec<RawEntity>,
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
    fn name(&self) -> &'static str {
        "programmatic"
    }

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
        .model(ctx.model())
        .ttft_budget_ms(ctx.arm_budget_ms)
        .call()
        .await
        .map_err(|e| crate::core::error::Error::Llm(e.to_string()))?;
        let _ms = typing_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.extraction.stage_ms", "stage" => "entity_typing").record(_ms);
        tracing::info!(_ms, stage = "entity_typing", "kremory.extraction.stage_ms");
        let typing_text = serde_json::to_string(&typing_value).unwrap_or_default();

        // Routed through the shared shape-tolerant `parse_items` helper:
        // the previous `parse_json_lenient::<
        // EntityOnlyOutput>` deserialized the whole `#[serde(default)]`
        // wrapper struct directly, which cannot distinguish a wrong/missing
        // "entities" key from a genuine empty list, and had zero raw-vs-
        // emitted observability. `ProgrammaticFirstExtractor` is dormant (not
        // wired into `ExtractorKind` production dispatch) but is fixed now
        // while the risk is low.
        let (raw_entities, raw_entity_count, entity_deserialize_ok, _entity_parse_path): (
            Vec<RawEntity>,
            usize,
            bool,
            ParsePath,
        ) = parse_items(&typing_text, "entities");
        if entity_deserialize_ok {
            counter!("rql.extraction.json_parse_ok").increment(1);
        } else {
            counter!("rql.extraction.json_parse_fail").increment(1);
            tracing::warn!(
                parser = "programmatic_entities",
                "kremory.extraction.json_parse_fail"
            );
        }
        let mut dropped_empty_name = 0u64;
        let mut entities: Vec<ExtractedEntity> = raw_entities
            .into_iter()
            .filter(|e| {
                let keep = !e.name.is_empty();
                if !keep {
                    dropped_empty_name += 1;
                }
                keep
            })
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
        if dropped_empty_name > 0 {
            counter!("rql.extraction.item_dropped", "parser" => "programmatic_entities", "reason" => "empty_name")
                .increment(dropped_empty_name);
        }
        emit_parse_yield_metrics("programmatic_entities", raw_entity_count, entities.len());

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
        .model(ctx.model())
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

        // Same shape-tolerant + observable routing as the entities call above.
        let (raw_rels, raw_rel_count, rel_deserialize_ok, _rel_parse_path): (
            Vec<RawRelationship>,
            usize,
            bool,
            ParsePath,
        ) = parse_items(&rel_text, "relationships");
        if rel_deserialize_ok {
            counter!("rql.extraction.json_parse_ok").increment(1);
        } else {
            counter!("rql.extraction.json_parse_fail").increment(1);
            tracing::warn!(
                parser = "programmatic_relationships",
                "kremory.extraction.json_parse_fail"
            );
        }
        let mut dropped_empty_field = 0u64;
        let facts: Vec<ExtractedFact> = raw_rels
            .into_iter()
            .filter(|r| {
                let keep = !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty();
                if !keep {
                    dropped_empty_field += 1;
                }
                keep
            })
            .map(|r| ExtractedFact {
                subject: r.subject,
                predicate: r.predicate,
                object: r.object,
                is_entity_ref: r.is_entity_ref,
                confidence: r.confidence,
                // `RawRelationship` carries no date, so this arm
                // cannot supply one. Always `None` ⇒ the persist sites fall back
                // to the episode `ref_time`.
                // Stated rather than left implicit: a payload that
                // falls back to this arm silently loses per-fact dates, so a
                // corpus with a high fallback rate will show `valid_at` uptake
                // far below what the primary arm achieves.
                valid_at: None,
            })
            .collect();
        if dropped_empty_field > 0 {
            counter!("rql.extraction.item_dropped", "parser" => "programmatic_relationships", "reason" => "empty_field")
                .increment(dropped_empty_field);
        }
        emit_parse_yield_metrics("programmatic_relationships", raw_rel_count, facts.len());

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

// ─── Lenient JSON parser ──────────────────────────────────────────────────────

/// Lenient JSON parser: tries raw parse, then llm_json repair, then brace extraction.
pub(crate) fn parse_json_lenient<T: for<'de> serde::Deserialize<'de> + Default>(
    raw: &str,
) -> Option<T> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<T>(t) {
        return Some(v);
    }
    let repaired =
        jsonrepair::repair_json(t, &jsonrepair::Options::default()).unwrap_or(t.to_owned());
    if let Ok(v) = serde_json::from_str::<T>(&repaired) {
        return Some(v);
    }
    // Try extracting JSON object from surrounding text
    if let (Some(s), Some(e)) = (t.find('{'), t.rfind('}')) {
        if e > s {
            let slice = &t[s..=e];
            let repaired2 = jsonrepair::repair_json(slice, &jsonrepair::Options::default())
                .unwrap_or(slice.to_owned());
            if let Ok(v) = serde_json::from_str::<T>(&repaired2) {
                return Some(v);
            }
        }
    }
    None
}
