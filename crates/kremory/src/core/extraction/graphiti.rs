//! LlmExtractor and Graphiti-quality prompt builders.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
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
        .model(ctx.model())
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
        .model(ctx.model())
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
///
/// TD-187 (temporal grounding): `reference_time` is the caller-DECLARED
/// document anchor (`ExtractionContext::reference_time`) — and ONLY that.
/// When `Some`, a one-line date-grounding block is appended so the LLM can
/// resolve relative-time phrases ("yesterday", "last week") in `text` against
/// an absolute date. When `None` (the default — unchanged from before
/// TD-187), NOTHING extra is rendered: the returned prompt is byte-identical
/// to the pre-TD-187 prompt. This is load-bearing for VCR cassette replay —
/// see `ExtractionContext::reference_time`'s doc comment for why.
///
/// Args-as-object (TD-042): adding `reference_time` took this to 4 positional
/// params and tripped clippy `too_many_arguments` (4/3 — the project threshold
/// is 3 and `#[allow]` is banned in `src/`). Grouped into a params struct per
/// the ratified TD-042 pattern rather than suppressed.
pub(crate) struct TripletPromptParams<'a> {
    pub(crate) text: &'a str,
    pub(crate) entities: &'a [ExtractedEntity],
    pub(crate) relation_names: &'a [String],
    /// See [`crate::core::intelligence::ExtractionContext::reference_time`] —
    /// caller-DECLARED anchor only, never wall-clock.
    pub(crate) reference_time: Option<DateTime<Utc>>,
    /// ADR-080 — contents of the preceding turns of this conversation thread,
    /// oldest-first. Empty (the default) renders NOTHING and leaves the prompt
    /// byte-identical; see
    /// [`crate::core::intelligence::ExtractionContext::prior_turns`].
    pub(crate) prior_turns: &'a [String],
}

/// Total character budget for the replayed-turn block (ADR-080).
///
/// Unbounded replay is a real production hazard, not a theoretical one: the
/// depth is 10 episodes and an episode has no length limit, so a naive
/// concatenation can push the extraction prompt past the model's context window
/// — the same failure `MemoryBuilder::episode_content_warn_threshold` exists to
/// warn about, arriving by a different route. The budget is spent OLDEST-FIRST
/// and truncation drops whole turns from the FRONT, so the turns nearest the
/// text being extracted — the ones a reference is most likely to point at —
/// always survive.
pub(crate) const PRIOR_TURNS_MAX_CHARS: usize = 4_000;

pub(crate) fn build_triplet_prompt(params: TripletPromptParams<'_>) -> String {
    let TripletPromptParams {
        text,
        entities,
        relation_names,
        reference_time,
        prior_turns,
    } = params;
    let entity_list = entities
        .iter()
        .map(|e| format!("{} ({})", e.name, e.label))
        .collect::<Vec<_>>()
        .join(", ");
    let rel_list = relation_names.join(", ");
    // TD-187 round 2. The date rules and the `valid_at` output field are ONE
    // gated unit, both conditional on `reference_time`.
    //
    // Round 1 rendered only the rules, and the output-field sentence — which is
    // unconditional — still listed five fields, none a date. The model was told
    // to state a date with nowhere to put it, complied by dropping it, and the
    // change measured EXACTLY null: zero of 441 facts carried a date. An
    // instruction cannot create a schema slot.
    //
    // The field mention MUST stay inside this gate. It lives one sentence away
    // from the unconditional field list, and moving it there would re-fingerprint
    // all 303 committed chat cassettes (`record_replay.rs` hashes the rendered
    // prompt). With `None` this renders byte-identically to pre-TD-187, which the
    // guard test below asserts against a hardcoded pre-change literal.
    //
    // Wording is graphiti's (`graphiti_core/prompts/extract_edges.py`), adopted
    // deliberately rather than paraphrased — it is load-bearing, not cosmetic.
    // On a real failing corpus turn ("We had a blast last year at the Pride
    // fest", gold 2022) a thinner phrasing returned null and graphiti's
    // year-granularity rule returned 2022-01-01. The present-tense rule is an
    // explicit CHOICE, not an accident: an ongoing fact is anchored to the
    // document date, matching today's behaviour for such facts.
    let date_block = match reference_time {
        Some(ts) => format!(
            "\nDATE RULES for \"valid_at\" — when the fact became true:\n- The source document is dated {}. Resolve relative expressions (\"yesterday\", \"last week\", \"last year\", \"2 years ago\") against that date.\n- If only a year is resolvable, use January 1st of that year.\n- If the fact is ongoing (present tense), use the document date.\n- Leave \"valid_at\" null if no explicit or resolvable time is stated.\n- Do NOT hallucinate or infer dates from unrelated events.\n",
            ts.format("%Y-%m-%d")
        ),
        None => String::new(),
    };
    // The WHOLE field list is switched, not suffixed. Appending `, and "valid_at"`
    // to a list that already reads `..., and "confidence"` produces a double
    // conjunction ("and X, and Y") — sloppy prose in the one place we are asking a
    // language model to follow a spec precisely. Switching the whole clause keeps
    // both arms grammatical, and the `None` arm below is the VERBATIM pre-TD-187
    // literal, which is what keeps the 303 cassette fingerprints stable.
    let field_list = match reference_time {
        Some(_) => {
            "\"subject\", \"predicate\", \"object\", \"is_entity_ref\" (boolean), \"confidence\" (0.0-1.0), and \"valid_at\" (ISO 8601 date YYYY-MM-DD, or null)"
        }
        None => {
            "\"subject\", \"predicate\", \"object\", \"is_entity_ref\" (boolean), and \"confidence\" (0.0-1.0)"
        }
    };
    // ADR-080 (prior-turn replay). Two properties are load-bearing.
    //
    // 1. BYTE IDENTITY WHEN EMPTY. `prior_block` is spliced immediately before
    //    the existing `Text: ` marker and carries its OWN trailing `\n\n`, so
    //    the empty case leaves the surrounding literal exactly as it was. An
    //    empty section is only byte-safe if the separator lives inside the
    //    block, not around it — the same trick `date_block` uses, arrived at the
    //    other way round. The guard test asserts against the pre-ADR-080
    //    literal, and ~305 committed cassettes depend on it.
    //
    // 2. CONTEXT, NOT MATERIAL. The instruction forbidding extraction FROM the
    //    replayed turns is not politeness — without it every prior turn is
    //    re-extracted on every subsequent turn, so an N-turn conversation emits
    //    each early fact up to N times and the graph fills with duplicates that
    //    the resolver then has to merge. mem0 and Graphiti both pass previous
    //    turns as context only, for exactly this reason.
    let prior_block = if prior_turns.is_empty() {
        String::new()
    } else {
        // Spend the budget oldest-first but DROP from the front, so the turns
        // adjacent to `text` — the likely referents — always survive.
        let mut kept: Vec<&str> = Vec::new();
        let mut used = 0usize;
        for turn in prior_turns.iter().rev() {
            let cost = turn.chars().count() + 3; // "- " + newline
            if used + cost > PRIOR_TURNS_MAX_CHARS {
                break;
            }
            used += cost;
            kept.push(turn.as_str());
        }
        if kept.is_empty() {
            String::new()
        } else {
            kept.reverse();
            let body = kept
                .iter()
                .map(|t| format!("- {t}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "Earlier turns in this same conversation, oldest first. Use them ONLY to resolve references (pronouns, \"that one\", \"the same place\") appearing in the text below. Do NOT extract relationships from these earlier turns — they have already been processed.\n{body}\n\n"
            )
        }
    };
    format!(
        "Given entities: [{entity_list}]\nRelationship types: [{rel_list}]\n{date_block}\nExtract the key relationships from this text as (subject, predicate, object) triplets. Only include each distinct relationship once. Do not repeat.\n\n{prior_block}Text: {text}\n\nOutput a concise JSON array of objects with {field_list} fields."
    )
}

// ─── TD-187 unit tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod td_187_tests {
    use super::*;
    use chrono::TimeZone;

    fn sample_entities() -> Vec<ExtractedEntity> {
        vec![
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Alice".to_string(),
                properties: serde_json::Value::Null,
            },
            ExtractedEntity {
                label: "Person".to_string(),
                name: "Bob".to_string(),
                properties: serde_json::Value::Null,
            },
        ]
    }

    /// Guards all 303 committed VCR chat cassettes (`crates/kremory/tests/cassettes/`):
    /// the fingerprint hashes the rendered prompt (`record_replay.rs:225-263`), so
    /// `reference_time: None` MUST render byte-identically to the pre-TD-187 prompt.
    #[test]
    fn build_triplet_prompt_with_none_is_byte_identical_to_pre_td187_prompt() {
        let entities = sample_entities();
        let relation_names = vec!["met".to_string()];
        let text = "Alice met Bob yesterday.";

        let actual = build_triplet_prompt(TripletPromptParams {
            text,
            entities: &entities,
            relation_names: &relation_names,
            reference_time: None,
            prior_turns: &[],
        });

        // Hand-reconstructed from the pre-TD-187 format! literal (no `reference_time`
        // parameter, no date_block interpolation) — this is the exact string every
        // one of the 303 cassettes was recorded against.
        let expected = "Given entities: [Alice (Person), Bob (Person)]\nRelationship types: [met]\n\nExtract the key relationships from this text as (subject, predicate, object) triplets. Only include each distinct relationship once. Do not repeat.\n\nText: Alice met Bob yesterday.\n\nOutput a concise JSON array of objects with \"subject\", \"predicate\", \"object\", \"is_entity_ref\" (boolean), and \"confidence\" (0.0-1.0) fields.";

        assert_eq!(
            actual, expected,
            "None must render NOTHING extra — any deviation here breaks all 303 VCR cassette fingerprints"
        );
    }

    #[test]
    fn build_triplet_prompt_with_some_renders_date_grounding_block() {
        let entities = sample_entities();
        let relation_names = vec!["met".to_string()];
        let text = "Alice met Bob yesterday.";
        let ts = Utc.with_ymd_and_hms(2019, 3, 15, 12, 0, 0).unwrap();

        let actual = build_triplet_prompt(TripletPromptParams {
            text,
            entities: &entities,
            relation_names: &relation_names,
            reference_time: Some(ts),
            prior_turns: &[],
        });

        assert!(
            actual.contains("The source document is dated 2019-03-15."),
            "expected date-grounding line with formatted date, got: {actual}"
        );
        assert!(
            actual.contains("Resolve relative expressions"),
            "expected relative-time resolution instruction, got: {actual}"
        );
        // No time-of-day rendered — date-only per TD-187 spec (avoid fingerprint churn).
        assert!(
            !actual.contains("12:00:00"),
            "must not render time-of-day, got: {actual}"
        );
    }

    // ─── TD-187 round 2: the instruction must come WITH a slot ────────────────

    /// The round-1 defect, as an executable assertion.
    ///
    /// Round 1 shipped the date instruction while the output-field sentence still
    /// listed five fields, none a date. The model was told to state a date with
    /// nowhere to put it and dropped it: 0 of 441 facts carried one, and the
    /// change measured EXACTLY null. Asserting the two halves together is what
    /// makes that failure non-repeatable — either half alone passes while the
    /// feature does nothing.
    #[test]
    fn some_renders_the_date_instruction_and_the_valid_at_slot_together() {
        let entities = sample_entities();
        let relation_names = vec!["met".to_string()];
        let ts = Utc.with_ymd_and_hms(2023, 5, 8, 0, 0, 0).unwrap();

        let actual = build_triplet_prompt(TripletPromptParams {
            text: "Alice met Bob yesterday.",
            entities: &entities,
            relation_names: &relation_names,
            reference_time: Some(ts),
            prior_turns: &[],
        });

        assert!(
            actual.contains("Resolve relative expressions"),
            "instruction half missing, got: {actual}"
        );
        assert!(
            actual.contains(r#""valid_at" (ISO 8601 date YYYY-MM-DD, or null)"#),
            "SLOT half missing — this is exactly the round-1 defect: an instruction \
             to state a date with no field to state it in. Got: {actual}"
        );
    }

    /// The `None` arm must NOT mention `valid_at` anywhere.
    ///
    /// Byte-identity is already asserted above, but that test compares against one
    /// hardcoded literal; this one states the INVARIANT the literal exists to
    /// protect, so a future edit that changes both together still trips here.
    #[test]
    fn none_never_mentions_valid_at() {
        let entities = sample_entities();
        let relation_names = vec!["met".to_string()];

        let actual = build_triplet_prompt(TripletPromptParams {
            text: "Alice met Bob yesterday.",
            entities: &entities,
            relation_names: &relation_names,
            reference_time: None,
            prior_turns: &[],
        });

        assert!(
            !actual.contains("valid_at"),
            "None must render no date slot — mentioning it unconditionally \
             re-fingerprints all 303 committed cassettes. Got: {actual}"
        );
    }

    /// Grammar guard. The field list is switched wholesale rather than suffixed,
    /// because appending to a list ending `..., and "confidence"` yields a double
    /// conjunction. Caught in review of this very change; asserted so it stays fixed.
    #[test]
    fn field_list_has_no_double_conjunction_in_either_arm() {
        let entities = sample_entities();
        let relation_names = vec!["met".to_string()];
        let ts = Utc.with_ymd_and_hms(2023, 5, 8, 0, 0, 0).unwrap();

        for reference_time in [None, Some(ts)] {
            let actual = build_triplet_prompt(TripletPromptParams {
                text: "Alice met Bob yesterday.",
                entities: &entities,
                relation_names: &relation_names,
                reference_time,
                prior_turns: &[],
            });
            assert_eq!(
                actual.matches(", and ").count(),
                1,
                "field list must contain exactly one ', and ' — got: {actual}"
            );
        }
    }
}

// ─── ADR-080 prior-turn replay unit tests ────────────────────────────────────

#[cfg(test)]
mod adr_080_prior_turn_tests {
    use super::*;

    fn sample_entities() -> Vec<ExtractedEntity> {
        vec![ExtractedEntity {
            label: "Person".to_string(),
            name: "Alice".to_string(),
            properties: serde_json::Value::Null,
        }]
    }

    fn build(prior: &[String]) -> String {
        build_triplet_prompt(TripletPromptParams {
            text: "Alice liked that one.",
            entities: &sample_entities(),
            relation_names: &["liked".to_string()],
            reference_time: None,
            prior_turns: prior,
        })
    }

    /// The load-bearing one. An empty slice must render NOTHING, or every
    /// committed VCR cassette's fingerprint changes — the prompt is hashed
    /// (`core/provider/record_replay.rs`). The sibling test
    /// `td_187_tests::build_triplet_prompt_with_none_is_byte_identical_to_pre_td187_prompt`
    /// pins the full literal; this one pins the specific ADR-080 property.
    #[test]
    fn empty_prior_turns_render_nothing_and_keep_the_text_separator() {
        let actual = build(&[]);
        assert!(
            !actual.contains("Earlier turns"),
            "empty prior_turns must not render a replay block: {actual}"
        );
        // The separator must remain exactly `\n\n` before `Text: ` — the block
        // carries its OWN trailing newlines precisely so this stays true.
        assert!(
            actual.contains("Do not repeat.\n\nText: Alice liked that one."),
            "empty case changed the bytes around `Text:`: {actual}"
        );
    }

    #[test]
    fn non_empty_prior_turns_render_oldest_first_with_a_do_not_extract_instruction() {
        let prior = vec![
            "Alice: which jacket do you mean?".to_string(),
            "Bob: the blue one on the left.".to_string(),
        ];
        let actual = build(&prior);

        assert!(actual.contains("Earlier turns in this same conversation"));
        assert!(actual.contains("- Alice: which jacket do you mean?"));
        assert!(actual.contains("- Bob: the blue one on the left."));

        // Oldest-first: the first turn must appear before the second.
        let i0 = actual.find("which jacket").expect("turn 0 present");
        let i1 = actual.find("the blue one").expect("turn 1 present");
        assert!(i0 < i1, "prior turns rendered newest-first");

        // The block must sit BEFORE the text under extraction.
        let itext = actual.find("Text: Alice liked").expect("text present");
        assert!(i1 < itext, "replay block must precede the text");

        // CONTEXT, NOT MATERIAL. Without this instruction every prior turn is
        // re-extracted on every later turn and the graph fills with duplicates.
        assert!(
            actual.contains("Do NOT extract relationships from these earlier turns"),
            "missing the do-not-extract instruction: {actual}"
        );
    }

    /// Budget truncation must drop from the FRONT. The turns nearest the text
    /// are the likely referents of a pronoun in it, so they are the ones that
    /// must survive a squeeze.
    #[test]
    fn budget_drops_oldest_turns_first() {
        let big = "x".repeat(PRIOR_TURNS_MAX_CHARS / 2);
        let prior = vec![
            format!("OLDEST {big}"),
            format!("MIDDLE {big}"),
            "NEWEST short turn".to_string(),
        ];
        let actual = build(&prior);

        assert!(
            actual.contains("NEWEST short turn"),
            "the most recent turn must always survive truncation"
        );
        assert!(
            !actual.contains("OLDEST"),
            "oldest turn should have been dropped by the budget"
        );
        assert!(
            actual.len() < PRIOR_TURNS_MAX_CHARS * 2,
            "replay block blew the character budget: {} chars",
            actual.len()
        );
    }

    /// A single turn larger than the whole budget must not render a
    /// half-truncated fragment — it renders nothing, and the prompt stays
    /// byte-identical to the no-replay case.
    #[test]
    fn one_oversized_turn_degrades_to_no_block_not_a_fragment() {
        let huge = "y".repeat(PRIOR_TURNS_MAX_CHARS + 100);
        let actual = build(&[huge]);
        assert!(!actual.contains("Earlier turns"));
        assert_eq!(actual, build(&[]), "must equal the no-replay rendering");
    }
}
