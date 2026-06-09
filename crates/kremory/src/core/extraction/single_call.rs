//! SingleCallExtractor — free discovery: entities + relationships in one pass.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

use std::sync::Arc;
use std::time::Instant;

use metrics::histogram;
use tracing;

use super::json_repair::parse_nuextract_response;
use super::{prompts, schemas, structured};
use crate::core::config::ContentType;
use crate::core::error::Result;
use crate::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};

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

pub(super) const SINGLE_CALL_ENTITY_TYPES: &str =
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
    fn name(&self) -> &'static str {
        "single_call"
    }

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
