//! JSON-to-domain parsers for extraction output.
//!
//! Split from `mod.rs` as part of TD-001 (E0-B).

// Items used only in #[cfg(test)] — suppress dead_code for non-test builds.
#![allow(dead_code)]

use metrics::counter;
use tracing;

use super::json_repair::repair_to_array;
use super::models::{EntityListIntegerWrapper, RawEntityIntegerId, RawEntitySimple, RawFact};
use crate::core::intelligence::{ExtractedEntity, ExtractedFact};

// ─── Relation name parser ─────────────────────────────────────────────────────

pub(crate) fn parse_relation_names(json: &str) -> anyhow::Result<Vec<String>> {
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

// ─── Entity parsers ───────────────────────────────────────────────────────────

/// Legacy string-label entity parser (pre-TD-013 L1 path).
///
/// Production extractors now use `parse_entities_integer`.  This function is
/// retained as a test fixture for the string-label parse path (used by
/// `test_parse_entities_*` tests).  Rustc dead-code analysis doesn't count
/// `#[cfg(test)]` callers from the production-code vantage point.
#[allow(dead_code)]
pub(crate) fn parse_entities(json: &str) -> anyhow::Result<Vec<ExtractedEntity>> {
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
        .collect())
}

/// Parse the integer-ID entity JSON emitted by the L1 extraction path (TD-013).
///
/// Accepts two input shapes:
/// 1. Wrapped: `{"entities": [{"name": "Alice", "entity_type_id": 1}, ...]}` — primary.
/// 2. Bare array: `[{"name": "Alice", "entity_type_id": 1}, ...]` — repair fallback.
/// 3. Single object: `{"name": "Alice", "entity_type_id": 1}` — wrapped via repair_to_array.
///
/// For each `RawEntityIntegerId`:
/// - Skips entries with empty `name`.
/// - Validates `entity_type_id` via `EntityTypeRegistry::validate_or_fallback`
///   (out-of-range or unknown ids → id=0 catch-all "Entity").
/// - Resolves the integer id to a label string via `EntityTypeRegistry::id_to_name`.
/// - Builds `ExtractedEntity { name, label, properties }` — downstream contract preserved.
pub(crate) fn parse_entities_integer(
    json: &str,
    registry: &crate::core::entity_types::EntityTypeRegistry,
) -> anyhow::Result<Vec<ExtractedEntity>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Ok(vec![]);
    }

    // KREMORY_DEBUG=1: emit raw stage1 LLM output to stderr for diagnosis (Rule 19).
    if std::env::var("KREMORY_DEBUG").is_ok() {
        eprintln!(
            "[KREMORY_DEBUG] parse_entities_integer raw input (len={}):\n{trimmed}\n[KREMORY_DEBUG end]",
            trimmed.len()
        );
    }

    // Try wrapped form first: {"entities": [...]}
    let raw: Vec<RawEntityIntegerId> = if let Ok(w) =
        serde_json::from_str::<EntityListIntegerWrapper>(trimmed)
    {
        counter!("rql.extraction.json_parse_ok", "path" => "wrapped").increment(1);
        w.entities
    } else {
        // Try bare array.
        match serde_json::from_str::<Vec<RawEntityIntegerId>>(trimmed) {
            Ok(v) => {
                counter!("rql.extraction.json_parse_ok", "path" => "bare_array").increment(1);
                v
            }
            Err(_) => {
                // repair_to_array handles: markdown fences, single-object, trailing commas.
                let repaired = repair_to_array(trimmed);
                match serde_json::from_str::<Vec<RawEntityIntegerId>>(&repaired) {
                    Ok(v) => {
                        // Per Rule 20: repair-path success is suspicious. Track separately
                        // so qwen-vs-haiku divergence and post-repair garbage are visible.
                        counter!("rql.extraction.json_parse_ok", "path" => "post_repair")
                            .increment(1);
                        v
                    }
                    Err(e) => {
                        counter!("rql.extraction.json_parse_fail").increment(1);
                        tracing::warn!(
                            error = %e,
                            parser = "entities_integer",
                            "kremory.extraction.json_parse_fail"
                        );
                        eprintln!("warn: failed to parse integer-id entity JSON after repair: {e}");
                        return Ok(vec![]);
                    }
                }
            }
        }
    };

    Ok(raw
        .into_iter()
        .filter_map(|e| {
            if e.name.is_empty() {
                counter!("rql.extraction.entity_rejected", "reason" => "empty_name").increment(1);
                return None;
            }
            // Shape-validate the name. Repair paths can splice JSON fragments
            // into the name field; reject any name containing JSON syntax chars
            // so garbage entities never reach persistence.
            if name_looks_like_json_fragment(&e.name) {
                counter!("rql.extraction.entity_rejected", "reason" => "name_json_fragment")
                    .increment(1);
                tracing::warn!(
                    raw_name = %e.name,
                    "kremory.extraction.entity_rejected.name_json_fragment"
                );
                return None;
            }
            let validated_id = registry.validate_or_fallback(e.entity_type_id);
            let label = registry.id_to_name(validated_id).to_string();
            let mut props = serde_json::Map::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String(e.name.clone()),
            );
            Some(ExtractedEntity {
                name: e.name,
                label,
                properties: serde_json::Value::Object(props),
            })
        })
        .collect())
}

/// Returns true if `name` contains characters that suggest it is a spliced JSON
/// fragment rather than a real entity name. Repair paths (`repair_to_array` +
/// `llm_json::repair_json`) can produce parseable output where a truncated
/// object's tail bleeds into the next entity's `name`. A real entity name will
/// never legitimately contain unescaped JSON structural characters.
pub(crate) fn name_looks_like_json_fragment(name: &str) -> bool {
    name.contains('"')
        || name.contains('\\')
        || name.contains('{')
        || name.contains('}')
        || name.contains("entity_type_id")
}

// ─── Fact parser ──────────────────────────────────────────────────────────────

pub(crate) fn parse_facts(json: &str) -> anyhow::Result<Vec<ExtractedFact>> {
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
