//! L2 extraction prompt engineering — NEVER-list, cross-domain examples, registry injection.
//!
//! This module owns all prompt-string construction for entity extraction.
//! The functions here are called by the extractors in `extraction/mod.rs`.
//!
//! ## Design (TD-013 Phase 3)
//!
//! Three layers of guidance compose into a complete entity-extraction prompt:
//!
//! 1. **NEVER-list** — explicit enumeration of things that are never entities:
//!    pronouns, abstract concepts, generic nouns, placeholders.
//!    Pattern from Graphiti (`extract_nodes.py` anti-extraction rules) and
//!    Cognee ("Don't use too generic terms like Entity").
//!
//! 2. **Cross-domain few-shot examples** — three worked examples from different
//!    domains (technical/business, academic/legal, personal/everyday) so the
//!    model can generalise extraction across context types without over-fitting to
//!    one domain.
//!
//! 3. **Registry injection** — when an `EntityTypeRegistry` has been loaded for
//!    the active namespace, the prompt renders a structured table of registered
//!    types (`{id}: {name} — {description}`) so the LLM picks from known types.
//!    Includes anti-generic guidance: "If unsure, use entity_type_id=0 ('Entity').
//!    NEVER invent type names not in the registry."
//!
//! ## Fallback
//!
//! When the registry slice is empty (no types registered, or the registry hasn't
//! been loaded yet), the prompt omits the registry table and anti-generic block.
//! This maintains backward-compatibility with callers that don't pass registry
//! specs.

use crate::core::entity_types::{EntityTypeRegistry, EntityTypeSpec};
use crate::core::schema::{Entity, Episode};

// ─── NEVER-list ──────────────────────────────────────────────────────────────

/// The canonical NEVER-list: explicit enumeration of things that are NEVER entities.
///
/// Injected into every entity extraction prompt as a hard exclusion rule.
/// Pattern from Graphiti's `extract_nodes.py` anti-extraction rules and
/// Cognee's "Don't use too generic terms like Entity" guidance.
pub const NEVER_LIST: &str = "\
NEVER extract the following as entities:
- Pronouns (he, she, it, they, we, you, I, me, him, her, them, us, who, which, that)
- Abstract concepts (love, time, idea, freedom, happiness, justice, success, truth, knowledge)
- Generic nouns without a specific name (\"the company\", \"a person\", \"the system\", \"the team\", \"an item\", \"a thing\", \"an object\")
- Placeholder labels (\"entity\", \"object\", \"the X\", \"some X\", \"a X\")
- Bare relational roles without qualification (dad, mom, boss, friend, colleague — extract as \"Alice's dad\" only if the possessor is named)
- Common objects that are not specifically named (watch, car, book, phone — unless the object has a specific product name like \"iPhone 15\")";

// ─── Cross-domain few-shot examples ─────────────────────────────────────────

/// Three cross-domain worked examples covering technical/business, academic/legal,
/// and personal/everyday contexts.
///
/// Domain spread is intentional: prevents the model over-fitting to one context type
/// while demonstrating (a) what IS an entity, (b) what is NOT an entity.
///
/// Pattern from Graphiti few-shot examples and LightRAG cross-domain benchmarks.
pub const CROSS_DOMAIN_EXAMPLES: &str = r#"Examples (cross-domain):

Example 1 — Technical/Business:
  Text: "Alice works at Acme as a Software Engineer."
  Entities: Alice (Person), Acme (Organisation)
  NOT entities: "Software Engineer" (job title, not an entity)

Example 2 — Academic/Legal:
  Text: "Stanford's antitrust ruling cited the 1976 Sherman Act precedent."
  Entities: Stanford (Organisation), Sherman Act (LegalDocument)
  NOT entities: "antitrust" (adjective), "precedent" (abstract concept), "1976" (bare year without named context)

Example 3 — Personal/Everyday:
  Text: "Bob's mother gave him a watch for his birthday."
  Entities: Bob (Person)
  NOT entities: "mother" (bare relational role — extract as "Bob's mother" only if she's named), "watch" (generic object), "birthday" (common event noun)"#;

// ─── Registry injection ───────────────────────────────────────────────────────

/// Render the entity type registry as a structured table for the extraction prompt.
///
/// Each line: `{id}: {name} — {description}`
///
/// Includes anti-generic guidance per Cognee's pattern ("If unsure, use the
/// catch-all. NEVER invent type names not in the registry.").
///
/// Returns an empty string when `specs` is empty — no injection, no prompt bloat.
pub fn render_registry_block(specs: &[EntityTypeSpec]) -> String {
    if specs.is_empty() {
        return String::new();
    }

    let mut lines = Vec::with_capacity(specs.len() + 6);
    lines.push(
        "Entity types — set each entity's `entity_type_id` field to ONE of these integer ids:"
            .to_string(),
    );
    for spec in specs {
        lines.push(format!(
            "  {} — {} — {}",
            spec.id, spec.name, spec.description
        ));
    }
    lines.push(String::new()); // blank line separator
    lines.push("CRITICAL CLASSIFICATION RULES:".to_string());
    lines.push(
        "  • Every named entity MUST be assigned its most specific matching type from the list above.".to_string(),
    );
    lines.push(
        "  • Countries, cities, regions, geographic features → \"Location\" (e.g. France, Boston, Africa).".to_string(),
    );
    lines.push(
        "  • Companies, universities, agencies, named groups → \"Organisation\" (e.g. Amazon, Stanford, IMF).".to_string(),
    );
    lines.push(
        "  • Named individuals (first name, last name, full names) → \"Person\".".to_string(),
    );
    lines.push(
        "  • Use \"Entity\" ONLY as a last resort when NO other type fits. Choosing \"Entity\" for a country/company/person is INCORRECT.".to_string(),
    );
    lines.push("  • NEVER invent type names not listed above.".to_string());

    lines.join("\n")
}

// ─── Composed prompt section ─────────────────────────────────────────────────

/// Build the full L2 guidance block: NEVER-list + cross-domain examples + optional registry.
///
/// This block is injected into entity extraction prompts immediately before the
/// `<TEXT>` block. Empty when all inputs are empty (test/stub callers).
///
/// Callers must include a trailing newline after the returned string before their
/// `<TEXT>` tag.
pub fn build_l2_guidance(registry_specs: &[EntityTypeSpec]) -> String {
    let registry_block = render_registry_block(registry_specs);

    // Always include NEVER-list and cross-domain examples.
    // Registry block is conditional on non-empty specs.
    if registry_block.is_empty() {
        format!("{NEVER_LIST}\n\n{CROSS_DOMAIN_EXAMPLES}\n")
    } else {
        format!("{NEVER_LIST}\n\n{CROSS_DOMAIN_EXAMPLES}\n\n{registry_block}\n")
    }
}

// ─── L4' existing-entity injection ────────────────────────────────────────────

/// Maximum number of existing entities injected into the extraction prompt.
///
/// Injects only the top-N most-recently-accessed entities so the prompt stays
/// within reasonable token budgets for large registries.
pub const L4_PRIME_MAX_ENTITIES: usize = 50;

/// Render the existing-entity table for L4' prompt-time injection.
///
/// When the active `group_id` already contains canonical entities, this block
/// is prepended to the extraction prompt so the LLM reuses exact names rather
/// than inventing variant spellings (e.g. "Alice Johnson" vs "Alice J.").
///
/// Format:
/// ```text
/// ### Existing entities in this knowledge graph
///
/// The following entities already exist. REUSE these exact names when the text
/// mentions them — do not invent variant spellings. New entities that don't
/// match anything below are welcome.
///
/// | Name | Type |
/// |------|------|
/// | Alice Johnson | Person |
/// ...
/// ```
///
/// Returns an empty string when `entities` is empty (no prompt bloat for new
/// registries or when the group has no prior entities).
///
/// `entities` is a slice of `(name, label)` pairs ordered by descending
/// `access_count` (caller responsibility — this function does not sort).
///
/// Empirical pattern: Cognee `disambiguate_entities.py:122-138` uses a similar
/// candidate-list injection before asking the LLM to pick or create an entity.
/// kremory's L4' injects the same information at prompt-build time rather than
/// as a follow-up call, saving one LLM round-trip.
pub fn render_existing_entities_block(entities: &[(String, String)]) -> String {
    if entities.is_empty() {
        return String::new();
    }

    let capped: &[(String, String)] = if entities.len() > L4_PRIME_MAX_ENTITIES {
        &entities[..L4_PRIME_MAX_ENTITIES]
    } else {
        entities
    };

    let mut lines = Vec::with_capacity(capped.len() + 8);
    lines.push("### Existing entities in this knowledge graph".to_string());
    lines.push(String::new());
    lines.push(
        "The following entities already exist. REUSE these exact names when the text \
         mentions them — do not invent variant spellings. New entities that do not \
         match anything below are welcome."
            .to_string(),
    );
    lines.push(String::new());
    lines.push("| Name | Type |".to_string());
    lines.push("|------|------|".to_string());
    for (name, label) in capped {
        lines.push(format!("| {} | {} |", name, label));
    }

    lines.join("\n")
}

// ─── L6 DelimitedTuple prompt ─────────────────────────────────────────────────

/// Render the follow-up instruction for the L6 DelimitedTuple arm.
///
/// Appended as an additional user message after JSON-based extraction has failed.
/// Instructs the LLM to re-emit the same entities in the LightRAG pipe-delimited
/// format so the delimited-tuple parser can recover them.
///
/// Format (one entity per line):
/// ```text
/// entity<|#|>Name<|#|>entity_type_id<|#|>Description
/// <|COMPLETE|>
/// ```
///
/// - The `entity` marker is literal — always lowercase.
/// - `entity_type_id` is a non-negative integer (0 = generic catch-all).
/// - Lines with the wrong field count are silently skipped by the parser.
/// - `<|COMPLETE|>` is the optional end sentinel; parsing continues without it.
pub fn render_delimited_tuple_prompt() -> String {
    "Your previous response could not be parsed as JSON. \
     Please re-emit the entities using the pipe-delimited format — one per line:\n\
     \n\
     entity<|#|>Name<|#|>entity_type_id<|#|>Description\n\
     \n\
     Rules:\n\
     - The first field must be the literal word `entity` (lowercase).\n\
     - `entity_type_id` must be a non-negative integer (0 for unknown type).\n\
     - Do NOT include JSON, markdown, or any other formatting.\n\
     - End with `<|COMPLETE|>` on its own line.\n\
     \n\
     Example:\n\
     entity<|#|>Alice<|#|>1<|#|>A software engineer at Acme.\n\
     entity<|#|>Acme<|#|>2<|#|>A technology company.\n\
     <|COMPLETE|>"
        .to_string()
}

// ─── L7 reclassification prompt ──────────────────────────────────────────────

/// Render the L7 dream-phase entity reclassification prompt.
///
/// Shows the LLM:
/// 1. The entity's name and current properties.
/// 2. The last N episode contents that mention this entity (contextual evidence).
/// 3. The registered entity type table (id → name — description).
///
/// The LLM is asked to output a single `entity_type_id` integer from the
/// registered table.  Positive framing is used per peer-prompt-engineering
/// research (feedback_peer_prompt_engineering_positive_and_generic_examples):
/// present what the type *is*, not what to avoid.
///
/// ## Episode selection
///
/// The caller is responsible for selecting the relevant episodes (typically
/// those in the same `group_id` as the entity).  This function renders
/// whichever slice is passed.  At most the last 5 episodes are shown in the
/// prompt to bound token usage.
pub fn render_reclassify_prompt(
    entity: &Entity,
    related_episodes: &[Episode],
    registry: &EntityTypeRegistry,
) -> String {
    let entity_name = entity
        .properties
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&entity.id);

    let entity_desc = entity
        .properties
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Show at most the last 5 episodes to bound prompt length.
    let episode_window: &[Episode] = if related_episodes.len() > 5 {
        &related_episodes[related_episodes.len() - 5..]
    } else {
        related_episodes
    };

    let episodes_block: String = episode_window
        .iter()
        .enumerate()
        .map(|(i, ep)| format!("  [{}] {}", i + 1, ep.content.trim()))
        .collect::<Vec<_>>()
        .join("\n");

    let registry_table = render_registry_block(registry.specs());

    format!(
        "You are classifying an entity in a knowledge graph.\n\
         \n\
         Entity name: {entity_name}\n\
         {desc_line}\
         \n\
         Recent context (episodes that mention this entity):\n\
         {episodes_block}\n\
         \n\
         Available entity types:\n\
         {registry_table}\n\
         \n\
         Select the most specific matching entity_type_id from the table above.\n\
         Use id=0 (Entity) when none of the specific types clearly apply.\n\
         Output a JSON object with a single field: {{\"entity_type_id\": <integer>}}",
        entity_name = entity_name,
        desc_line = if entity_desc.is_empty() {
            String::new()
        } else {
            format!("Description: {entity_desc}\n")
        },
        episodes_block = episodes_block,
        registry_table = registry_table,
    )
}

// ─── TD-023 Hybrid GLiNER + LLM typing prompt ────────────────────────────────

/// Render the batched-typing prompt for the TD-023 hybrid extractor.
///
/// Uses INDEX-BASED mapping per [[load-bearing-invariants-at-emit-not-prompt]]:
/// the LLM emits `{typings: [{idx, entity_type_id}, ...]}` where `idx` is the
/// 0-based position of the candidate in the input list. Bounded small int =
/// hard to corrupt = no name-drift surface.
///
/// The schema enforces the index range (`minimum=0`, `maximum=N-1`) so the
/// model cannot emit an out-of-range index.
///
/// One LLM call total for typing, not one-per-entity. Designed for the
/// TD-022 empirical pivot: GLiNER does fast span discovery, LLM does
/// disambiguation where GLiNER's training corpus doesn't reach (e.g.
/// "Los Angeles Superior Court" → Court vs Organisation).
pub fn render_hybrid_typing_prompt(
    text: &str,
    candidate_names: &[&str],
    registry: &EntityTypeRegistry,
) -> String {
    let registry_table = render_registry_block(registry.specs());
    let candidates_block: String = candidate_names
        .iter()
        .enumerate()
        .map(|(idx, n)| format!("  [{idx}] {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    let last_idx = candidate_names.len().saturating_sub(1);

    format!(
        "You are classifying entities found in a knowledge-graph ingest.\n\
         \n\
         SOURCE TEXT:\n\
         {text}\n\
         \n\
         CANDIDATE ENTITIES (already extracted by a fast NER pass — your job is to assign a type to each):\n\
         {candidates_block}\n\
         \n\
         AVAILABLE ENTITY TYPES:\n\
         {registry_table}\n\
         \n\
         For each candidate above, assign the most specific matching entity_type_id from the table.\n\
         id=0 (Entity) is FORBIDDEN — it is a reserved server-side catch-all. You MUST pick a specific id ≥ 1.\n\
         When in doubt, pick the closest semantic match (e.g. 'Superior Court' is a Court when Court is listed; \
         otherwise an Organisation. A named medication is a Drug when Drug is listed; otherwise a Concept).\n\
         \n\
         Output a JSON object with this exact shape, where each value is a real integer (NOT placeholder text):\n\
         {{\n\
           \"typings\": [\n\
             {{\"idx\": 0, \"entity_type_id\": 1}},\n\
             {{\"idx\": 1, \"entity_type_id\": 2}}\n\
           ]\n\
         }}\n\
         \n\
         RULES:\n\
         - `idx` must be an integer from 0 to {last_idx} (inclusive). Use the integer prefix from `  [N] Name` above.\n\
         - `entity_type_id` must be an integer ≥ 1 chosen from the AVAILABLE ENTITY TYPES table above.\n\
         - Emit exactly one object per candidate. Do not skip any. Do not add extras.\n\
         - DO NOT include the candidate name, placeholder text like \"<integer>\", ellipsis, or any string values — only real integers.",
        text = text,
        candidates_block = candidates_block,
        registry_table = registry_table,
        last_idx = last_idx,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_specs() -> Vec<EntityTypeSpec> {
        vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "Catch-all for unclassified entities.".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A named human individual.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "A company, institution, or formal group.".to_string(),
            },
        ]
    }

    // ── NEVER-list ────────────────────────────────────────────────────────────

    #[test]
    fn never_list_contains_pronouns() {
        assert!(
            NEVER_LIST.contains("Pronouns"),
            "NEVER_LIST must mention Pronouns"
        );
        assert!(
            NEVER_LIST.contains("he, she, it, they"),
            "NEVER_LIST must list common pronouns"
        );
    }

    #[test]
    fn never_list_contains_abstract_concepts() {
        assert!(
            NEVER_LIST.contains("Abstract concepts"),
            "NEVER_LIST must mention abstract concepts"
        );
        assert!(
            NEVER_LIST.contains("love, time, idea"),
            "NEVER_LIST must list example abstract concepts"
        );
    }

    #[test]
    fn never_list_contains_generic_nouns() {
        assert!(
            NEVER_LIST.contains("Generic nouns"),
            "NEVER_LIST must mention generic nouns"
        );
        assert!(
            NEVER_LIST.contains("\"the company\""),
            "NEVER_LIST must give generic noun examples"
        );
    }

    #[test]
    fn never_list_contains_placeholder_labels() {
        assert!(
            NEVER_LIST.contains("Placeholder labels"),
            "NEVER_LIST must mention placeholder labels"
        );
        assert!(
            NEVER_LIST.contains("\"entity\""),
            "NEVER_LIST must list 'entity' as a placeholder"
        );
        assert!(
            NEVER_LIST.contains("\"object\""),
            "NEVER_LIST must list 'object' as a placeholder"
        );
    }

    // ── Cross-domain examples ─────────────────────────────────────────────────

    #[test]
    fn cross_domain_examples_contains_technical_business() {
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Example 1"),
            "must have Example 1 (technical/business)"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Alice works at Acme"),
            "Example 1 text must be present"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Alice (Person)"),
            "Example 1 must show Alice as Person"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Acme (Organisation)"),
            "Example 1 must show Acme as Organisation"
        );
    }

    #[test]
    fn cross_domain_examples_contains_academic_legal() {
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Example 2"),
            "must have Example 2 (academic/legal)"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Stanford"),
            "Example 2 must mention Stanford"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Sherman Act"),
            "Example 2 must mention Sherman Act"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("LegalDocument"),
            "Example 2 must show Sherman Act type as LegalDocument"
        );
    }

    #[test]
    fn cross_domain_examples_contains_personal_everyday() {
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Example 3"),
            "must have Example 3 (personal/everyday)"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Bob's mother gave him a watch"),
            "Example 3 text must be present"
        );
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("Bob (Person)"),
            "Example 3 must show Bob as the only entity"
        );
        // Negative assertions: mother and watch are NOT entities
        assert!(
            CROSS_DOMAIN_EXAMPLES.contains("\"mother\"")
                || CROSS_DOMAIN_EXAMPLES.contains("\"watch\""),
            "Example 3 must show that mother/watch are NOT entities"
        );
    }

    #[test]
    fn cross_domain_examples_spans_three_domains() {
        // Verify all three domain labels are present to prevent domain collapse.
        assert!(CROSS_DOMAIN_EXAMPLES.contains("Technical/Business"));
        assert!(CROSS_DOMAIN_EXAMPLES.contains("Academic/Legal"));
        assert!(CROSS_DOMAIN_EXAMPLES.contains("Personal/Everyday"));
    }

    // ── Registry block ────────────────────────────────────────────────────────

    #[test]
    fn render_registry_block_empty_specs_returns_empty_string() {
        let result = render_registry_block(&[]);
        assert!(
            result.is_empty(),
            "empty specs must produce empty string (no prompt bloat)"
        );
    }

    #[test]
    fn render_registry_block_renders_id_name_description() {
        let specs = make_test_specs();
        let result = render_registry_block(&specs);

        // TD-013 Phase 8: registry block renders names as quoted strings (LLM
        // emits string label per L2 schema; integer id is server-side via
        // registry.label_to_id lookup).
        assert!(result.contains("\"Entity\""), "must render Entity entry");
        assert!(result.contains("\"Person\""), "must render Person entry");
        assert!(
            result.contains("\"Organisation\""),
            "must render Organisation entry"
        );
        assert!(
            result.contains("Catch-all for unclassified entities"),
            "must render Entity description"
        );
        assert!(
            result.contains("A named human individual"),
            "must render Person description"
        );
    }

    #[test]
    fn render_registry_block_contains_anti_generic_guidance() {
        let specs = make_test_specs();
        let result = render_registry_block(&specs);

        assert!(
            result.contains("\"Entity\""),
            "registry block must reference Entity as fallback name"
        );
        assert!(
            result.contains("NEVER invent type names"),
            "registry block must include anti-generic guidance"
        );
    }

    #[test]
    fn render_registry_block_format_uses_dash_separator() {
        // TD-013 Phase 8 v3: format contract is `  {id} — {name} — {description}`.
        let specs = vec![EntityTypeSpec {
            id: 1,
            name: "Person".to_string(),
            description: "A human individual.".to_string(),
        }];
        let result = render_registry_block(&specs);
        assert!(
            result.contains("1 — Person — A human individual."),
            "registry block line format must be '{{id}} — {{name}} — {{description}}'; got: {result}"
        );
    }

    // ── build_l2_guidance ─────────────────────────────────────────────────────

    #[test]
    fn build_l2_guidance_without_registry_contains_never_list_and_examples() {
        let result = build_l2_guidance(&[]);

        assert!(
            result.contains("NEVER extract"),
            "L2 guidance must contain NEVER-list"
        );
        assert!(
            result.contains("Examples (cross-domain)"),
            "L2 guidance must contain cross-domain examples"
        );
        // No registry block when empty specs
        assert!(
            !result.contains("Entity types"),
            "L2 guidance without registry must not contain registry header"
        );
    }

    #[test]
    fn build_l2_guidance_with_registry_contains_all_four_sections() {
        let specs = make_test_specs();
        let result = build_l2_guidance(&specs);

        // Section 1: NEVER-list
        assert!(result.contains("NEVER extract"), "must contain NEVER-list");
        // Section 2: cross-domain examples
        assert!(
            result.contains("Examples (cross-domain)"),
            "must contain cross-domain examples"
        );
        // Section 3: registry table (TD-013 Phase 8: string-label format)
        assert!(
            result.contains("Entity types"),
            "must contain registry header"
        );
        assert!(
            result.contains("\"Person\""),
            "must contain Person registry entry"
        );
        // Section 4: anti-generic guidance
        assert!(
            result.contains("\"Entity\""),
            "must contain anti-generic guidance with Entity fallback"
        );
        assert!(
            result.contains("NEVER invent type names"),
            "must contain anti-generic guidance phrase"
        );
    }

    // ── L4' existing-entity block ─────────────────────────────────────────────

    #[test]
    fn render_existing_entities_block_empty_returns_empty() {
        let result = render_existing_entities_block(&[]);
        assert!(
            result.is_empty(),
            "empty entity list must produce empty string (no prompt bloat)"
        );
    }

    #[test]
    fn render_existing_entities_block_renders_table() {
        let entities = vec![
            ("Alice Johnson".to_string(), "Person".to_string()),
            ("Acme Corp".to_string(), "Organisation".to_string()),
        ];
        let result = render_existing_entities_block(&entities);

        assert!(
            result.contains("### Existing entities in this knowledge graph"),
            "must contain the section header"
        );
        assert!(
            result.contains("REUSE these exact names"),
            "must instruct the LLM to reuse names"
        );
        assert!(
            result.contains("| Name | Type |"),
            "must contain the table header row"
        );
        assert!(
            result.contains("| Alice Johnson | Person |"),
            "must contain Alice row"
        );
        assert!(
            result.contains("| Acme Corp | Organisation |"),
            "must contain Acme Corp row"
        );
    }

    #[test]
    fn render_existing_entities_block_caps_at_max() {
        // Build a list larger than L4_PRIME_MAX_ENTITIES.
        let entities: Vec<(String, String)> = (0..L4_PRIME_MAX_ENTITIES + 10)
            .map(|i| (format!("Entity{i}"), "Person".to_string()))
            .collect();
        let result = render_existing_entities_block(&entities);

        // The capped entity must be present; the one just past the cap must not.
        assert!(
            result.contains(&format!("Entity{}", L4_PRIME_MAX_ENTITIES - 1)),
            "must include the last entity within the cap"
        );
        assert!(
            !result.contains(&format!("Entity{}", L4_PRIME_MAX_ENTITIES)),
            "must NOT include entities beyond the cap"
        );
    }

    // ── DelimitedTuple prompt ─────────────────────────────────────────────────

    #[test]
    fn render_delimited_tuple_prompt_contains_format_instruction() {
        let prompt = render_delimited_tuple_prompt();
        assert!(
            prompt.contains("pipe-delimited format"),
            "must explain pipe-delimited format"
        );
        assert!(
            prompt.contains("entity<|#|>Name<|#|>entity_type_id<|#|>Description"),
            "must show the format schema with correct delimiters"
        );
        assert!(
            prompt.contains("<|COMPLETE|>"),
            "must mention the COMPLETE sentinel"
        );
        assert!(
            prompt.contains("non-negative integer"),
            "must specify entity_type_id as non-negative integer"
        );
    }

    #[test]
    fn render_delimited_tuple_prompt_contains_example() {
        let prompt = render_delimited_tuple_prompt();
        assert!(
            prompt.contains("Alice"),
            "must include a named entity example"
        );
        assert!(
            prompt.contains("Acme"),
            "must include an organisation example"
        );
    }
}
