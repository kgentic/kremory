use crate::config::ContentType;
use crate::intelligence::{ExtractedEntity, ExtractionContext};

pub fn build_known_hint(known: &[ExtractedEntity], cap: usize) -> String {
    if known.is_empty() {
        return String::new();
    }

    let names: Vec<String> = known
        .iter()
        .rev()
        .take(cap)
        .map(|entity| format!("{} ({})", entity.name, entity.label))
        .collect();
    format!("\nPreviously seen entities: {}\n", names.join(", "))
}

pub fn build_gleaning_prompt(
    text: &str,
    found_entities: &[ExtractedEntity],
    ctx: &ExtractionContext<'_>,
) -> String {
    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: Conversational transcript with speaker labels.\n",
        ContentType::Json => "Format: Structured data fields.\n",
        ContentType::Text | ContentType::Document => "",
    };

    format!(
        "Review the text and return only entities that were missed in the prior extraction.\n\n\
{format_hint}\
Already found entities: [{}]\n\
Rules:\n\
- Return ONLY newly found entities, not duplicates\n\
- Classify each entity with a label or use Entity when unsure\n\
- Output a single JSON object with an \"entities\" array\n\n\
<TEXT>\n{text}\n</TEXT>\n",
        found_entities
            .iter()
            .map(|entity| format!("\"{}\"", entity.name))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn build_typing_prompt(
    text: &str,
    candidates: &[String],
    ctx: &ExtractionContext<'_>,
) -> String {
    let candidate_list = candidates
        .iter()
        .map(|candidate| format!("\"{}\"", candidate))
        .collect::<Vec<_>>()
        .join(", ");

    let format_hint = match ctx.content_type {
        ContentType::Message => "Format: Conversational transcript with speaker labels.\n",
        ContentType::Json => "Format: Structured data fields.\n",
        ContentType::Text | ContentType::Document => "",
    };

    format!(
        "Type every candidate entity using the source text for context.\n\n\
{format_hint}\
Candidates: [{candidate_list}]\n\
{known_hint}\
Rules:\n\
- Return EVERY candidate exactly once\n\
- Preserve the candidate name when possible\n\
- Use labels Person, Organisation, Location, Technology, Product, Event, Date, or Entity\n\
- If unsure, use Entity instead of omitting the candidate\n\n\
<TEXT>\n{text}\n</TEXT>\n\n\
Output a single JSON object with an \"entities\" array.",
        known_hint = build_known_hint(ctx.known_entities, 25)
    )
}
