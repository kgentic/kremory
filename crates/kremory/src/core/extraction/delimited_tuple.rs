//! L6 DelimitedTuple parser — LightRAG `<|#|>` pipe-delimited entity format.
//!
//! ## Format
//!
//! ```text
//! entity<|#|>Name<|#|>entity_type_id<|#|>Description
//! entity<|#|>Stanford<|#|>2<|#|>A university in California.
//! <|COMPLETE|>
//! ```
//!
//! - Each line has exactly 4 fields separated by [`FIELD_SEP`].
//! - The first field MUST be the literal `"entity"` marker.
//! - `entity_type_id` is a `u32` integer; unparseable values cause the line to be skipped.
//! - `<|COMPLETE|>` ([`SENTINEL`]) is the optional end-of-output marker; parsing
//!   continues without it and stops at it when present.
//! - Malformed lines (wrong field count, non-`entity` marker, non-`u32` type id)
//!   are silently skipped. Each skip increments the
//!   `rql.extraction.delimited_tuple_skip_row` metric counter.
//!
//! ## Output
//!
//! Returns a `serde_json::Value` with shape:
//! ```json
//! { "entities": [ { "name": "...", "label": "...", "entity_type_id": N, "description": "..." } ] }
//! ```
//!
//! The `label` field is set to `"Entity"` (the generic catch-all) because the
//! delimited format carries an integer type-id, not a label string. Callers that
//! need type-name resolution should look up the id in the `entity_types` table.

use metrics::counter;
use serde_json::{json, Value};

// ─── Format constants ─────────────────────────────────────────────────────────

/// Field separator used by the LightRAG pipe-delimited format.
pub(crate) const FIELD_SEP: &str = "<|#|>";

/// Optional end-of-output sentinel. Lines equal to this string (after trimming)
/// stop parsing; subsequent content is ignored.
pub(crate) const SENTINEL: &str = "<|COMPLETE|>";

/// Literal marker that must appear as the first field on every entity line.
const ENTITY_MARKER: &str = "entity";

// ─── Parser ───────────────────────────────────────────────────────────────────

/// Parse a raw LLM response in the DelimitedTuple format into a JSON `Value`.
///
/// Returns `{"entities": [...]}` where each element has:
/// - `"name"`: the entity name string
/// - `"label"`: always `"Entity"` (type-id is captured separately)
/// - `"entity_type_id"`: the `u32` type identifier
/// - `"description"`: the entity description string
///
/// Empty input or a response containing only the sentinel returns
/// `{"entities": []}`.  Malformed lines are silently skipped (metric incremented).
pub(crate) fn parse_delimited_tuple_response(text: &str) -> Value {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return json!({ "entities": [] });
    }

    let mut entities: Vec<Value> = Vec::new();

    for line in trimmed.lines() {
        let line = line.trim();

        // Stop at the sentinel line.
        if line == SENTINEL {
            break;
        }

        // Skip blank lines.
        if line.is_empty() {
            continue;
        }

        match parse_entity_line(line) {
            Some(entity) => entities.push(entity),
            None => {
                counter!(
                    "rql.extraction.delimited_tuple_skip_row",
                    "reason" => "malformed",
                )
                .increment(1);
            }
        }
    }

    json!({ "entities": entities })
}

/// Parse a single entity line.
///
/// Returns `None` if the line is malformed:
/// - Wrong field count (must be exactly 4)
/// - First field is not the `"entity"` marker
/// - `entity_type_id` field is not a valid `u32`
fn parse_entity_line(line: &str) -> Option<Value> {
    let fields: Vec<&str> = line.splitn(5, FIELD_SEP).collect();

    // Must have exactly 4 fields. splitn(5, ...) with a 4-field line gives 4 parts;
    // a 5-part overflow gives 5 — reject both wrong-count cases.
    if fields.len() != 4 {
        return None;
    }

    let marker = fields[0].trim();
    let name = fields[1].trim();
    let type_id_str = fields[2].trim();
    let description = fields[3].trim();

    // Marker must be the literal "entity".
    if marker != ENTITY_MARKER {
        return None;
    }

    // entity_type_id must be a valid u32.
    let entity_type_id: u32 = type_id_str.parse().ok()?;

    Some(json!({
        "name": name,
        "label": "Entity",
        "entity_type_id": entity_type_id,
        "description": description,
    }))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_delimited_tuple_response ────────────────────────────────────────

    #[test]
    fn parse_three_valid_lines_returns_three_entities() {
        let input = "\
entity<|#|>Alice<|#|>1<|#|>A software engineer.\n\
entity<|#|>Stanford<|#|>2<|#|>A university in California.\n\
entity<|#|>Sherman Act<|#|>3<|#|>A US antitrust law.\n\
<|COMPLETE|>";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 3, "expected 3 entities");

        assert_eq!(entities[0]["name"], "Alice");
        assert_eq!(entities[0]["entity_type_id"], 1u32);
        assert_eq!(entities[0]["label"], "Entity");
        assert_eq!(entities[0]["description"], "A software engineer.");

        assert_eq!(entities[1]["name"], "Stanford");
        assert_eq!(entities[1]["entity_type_id"], 2u32);

        assert_eq!(entities[2]["name"], "Sherman Act");
        assert_eq!(entities[2]["entity_type_id"], 3u32);
    }

    #[test]
    fn parse_with_one_malformed_middle_line_returns_two_entities() {
        // Middle line has only 3 fields — should be skipped.
        let input = "\
entity<|#|>Alice<|#|>1<|#|>Description of Alice.\n\
MALFORMED_LINE_NO_FIELDS\n\
entity<|#|>Stanford<|#|>2<|#|>Description of Stanford.\n\
<|COMPLETE|>";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 2, "malformed middle line must be skipped");
        assert_eq!(entities[0]["name"], "Alice");
        assert_eq!(entities[1]["name"], "Stanford");
    }

    #[test]
    fn parse_empty_input_returns_zero_entities() {
        let result = parse_delimited_tuple_response("");
        let entities = result["entities"].as_array().unwrap();
        assert!(
            entities.is_empty(),
            "empty input must produce empty entities"
        );
    }

    #[test]
    fn parse_without_sentinel_still_parses() {
        // <|COMPLETE|> is optional — parser must work without it.
        let input = "\
entity<|#|>Alice<|#|>1<|#|>A software engineer.\n\
entity<|#|>Acme<|#|>2<|#|>A technology company.";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(
            entities.len(),
            2,
            "sentinel is optional; both lines must parse"
        );
        assert_eq!(entities[0]["name"], "Alice");
        assert_eq!(entities[1]["name"], "Acme");
    }

    #[test]
    fn parse_with_field_overflow_five_fields_skips_line() {
        // Line has 5 fields — splitn(5, ..) produces 5 parts → reject.
        let input = "entity<|#|>Alice<|#|>1<|#|>Description<|#|>ExtraField\n\
             entity<|#|>Acme<|#|>2<|#|>Good line.\n\
             <|COMPLETE|>";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(
            entities.len(),
            1,
            "5-field overflow line must be skipped; only valid line kept"
        );
        assert_eq!(entities[0]["name"], "Acme");
    }

    #[test]
    fn parse_wrong_marker_skips_line() {
        let input = "\
person<|#|>Alice<|#|>1<|#|>Wrong marker.\n\
entity<|#|>Acme<|#|>2<|#|>Correct marker.\n\
<|COMPLETE|>";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 1, "wrong marker must be skipped");
        assert_eq!(entities[0]["name"], "Acme");
    }

    #[test]
    fn parse_non_integer_type_id_skips_line() {
        let input = "\
entity<|#|>Alice<|#|>person<|#|>Type id is a string, not integer.\n\
entity<|#|>Acme<|#|>2<|#|>Valid type id.\n\
<|COMPLETE|>";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(
            entities.len(),
            1,
            "non-integer entity_type_id must cause line skip"
        );
        assert_eq!(entities[0]["name"], "Acme");
    }

    #[test]
    fn parse_only_sentinel_returns_zero_entities() {
        let result = parse_delimited_tuple_response("<|COMPLETE|>");
        let entities = result["entities"].as_array().unwrap();
        assert!(
            entities.is_empty(),
            "sentinel-only input must produce empty entities"
        );
    }

    #[test]
    fn parse_sentinel_stops_at_sentinel_ignores_trailing_lines() {
        let input = "\
entity<|#|>Alice<|#|>1<|#|>Before sentinel.\n\
<|COMPLETE|>\n\
entity<|#|>Bob<|#|>1<|#|>After sentinel — must be ignored.";

        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(
            entities.len(),
            1,
            "content after <|COMPLETE|> must be ignored"
        );
        assert_eq!(entities[0]["name"], "Alice");
    }

    #[test]
    fn parse_type_id_zero_is_valid_catch_all() {
        let input = "entity<|#|>SomeEntity<|#|>0<|#|>Unknown type — catch-all.\n<|COMPLETE|>";
        let result = parse_delimited_tuple_response(input);
        let entities = result["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["entity_type_id"], 0u32);
    }

    // ── parse_entity_line ─────────────────────────────────────────────────────

    #[test]
    fn parse_entity_line_returns_none_for_three_field_line() {
        // 3 fields — wrong count.
        let line = "entity<|#|>Alice<|#|>1";
        assert!(parse_entity_line(line).is_none());
    }

    #[test]
    fn parse_entity_line_returns_correct_fields() {
        let line = "entity<|#|>Alice<|#|>1<|#|>A named person.";
        let entity = parse_entity_line(line).unwrap();
        assert_eq!(entity["name"], "Alice");
        assert_eq!(entity["entity_type_id"], 1u32);
        assert_eq!(entity["description"], "A named person.");
        assert_eq!(entity["label"], "Entity");
    }

    #[test]
    fn parse_entity_line_trims_whitespace_from_fields() {
        let line = "entity<|#|>  Alice  <|#|>  1  <|#|>  Description.  ";
        let entity = parse_entity_line(line).unwrap();
        assert_eq!(entity["name"], "Alice");
        assert_eq!(entity["entity_type_id"], 1u32);
        assert_eq!(entity["description"], "Description.");
    }

    // ── Constant guards ───────────────────────────────────────────────────────

    #[test]
    fn field_sep_and_sentinel_are_correct_literals() {
        assert_eq!(
            FIELD_SEP, "<|#|>",
            "FIELD_SEP must be the LightRAG separator"
        );
        assert_eq!(
            SENTINEL, "<|COMPLETE|>",
            "SENTINEL must be the LightRAG end-of-output marker"
        );
    }
}
