use anyhow::Result as AnyhowResult;
use serde::Deserialize;

use super::super::error::Result;
use super::super::intelligence::ExtractedEntity;
use super::super::resolver::normalize_name;

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct EntityListOutput {
    #[serde(default)]
    pub(crate) entities: Vec<HybridRawEntity>,
}

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub(crate) struct HybridRawEntity {
    #[serde(default)]
    pub(crate) name: String,
    #[serde(default = "default_entity_label")]
    pub(crate) label: String,
}

fn default_entity_label() -> String {
    "Entity".to_string()
}

pub(crate) fn parse_entity_list_response(raw: &str) -> Result<Vec<ExtractedEntity>> {
    let output: EntityListOutput = parse_json_lenient(raw).unwrap_or_default();
    Ok(output
        .entities
        .into_iter()
        .filter(|entity| !entity.name.is_empty())
        .map(|entity| ExtractedEntity {
            name: entity.name,
            label: entity.label,
            properties: serde_json::json!({}),
        })
        .collect())
}

pub(crate) fn parse_typed_orphans_response(
    raw: &str,
    candidates: &[String],
) -> Result<Vec<ExtractedEntity>> {
    let parsed = parse_entity_list_response(raw)?;
    let mut by_name = std::collections::HashMap::new();

    for entity in parsed {
        by_name.insert(normalize_name(&entity.name), entity);
    }

    let mut typed = Vec::new();
    for candidate in candidates {
        let normalized = normalize_name(candidate);
        if let Some(entity) = by_name.remove(&normalized) {
            typed.push(entity);
        } else {
            typed.push(ExtractedEntity {
                name: candidate.clone(),
                label: "Entity".to_string(),
                properties: serde_json::json!({"source": "hybrid_typing_fallback"}),
            });
        }
    }

    Ok(typed)
}

pub(crate) fn parse_json_lenient<T: for<'de> Deserialize<'de> + Default>(
    raw: &str,
) -> AnyhowResult<T> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(T::default());
    }

    if let Ok(value) = serde_json::from_str::<T>(trimmed) {
        return Ok(value);
    }

    let repaired = jsonrepair::repair_json(trimmed, &jsonrepair::Options::default())
        .unwrap_or_else(|_| trimmed.to_string());
    if let Ok(value) = serde_json::from_str::<T>(&repaired) {
        return Ok(value);
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if end > start {
            let slice = &trimmed[start..=end];
            let repaired = jsonrepair::repair_json(slice, &jsonrepair::Options::default())
                .unwrap_or_else(|_| slice.to_string());
            if let Ok(value) = serde_json::from_str::<T>(&repaired) {
                return Ok(value);
            }
        }
    }

    Ok(T::default())
}
