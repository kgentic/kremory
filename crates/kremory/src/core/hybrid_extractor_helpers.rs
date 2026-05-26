use serde_json::{Map, Value};

use super::super::grounding::GroundingChecker;
use super::super::intelligence::ExtractedEntity;
use super::super::resolver::normalize_name;

pub(crate) fn filter_new_entities(
    entities: Vec<ExtractedEntity>,
    known: &[ExtractedEntity],
) -> Vec<ExtractedEntity> {
    let known_names: std::collections::HashSet<String> = known
        .iter()
        .map(|entity| normalize_name(&entity.name))
        .collect();
    let mut seen = std::collections::HashSet::new();

    entities
        .into_iter()
        .filter(|entity| {
            let normalized = normalize_name(&entity.name);
            !known_names.contains(&normalized) && seen.insert(normalized)
        })
        .collect()
}

pub(crate) fn merge_entities_with_grounding(
    mut base: Vec<ExtractedEntity>,
    additive: Vec<ExtractedEntity>,
    source_text: &str,
    grounding_checker: &dyn GroundingChecker,
) -> Vec<ExtractedEntity> {
    base.extend(additive);
    base.sort_by_key(|e| normalize_name(&e.name));
    base.dedup_by(|a, b| normalize_name(&a.name) == normalize_name(&b.name));

    base.into_iter()
        .map(|entity| {
            let grounded = grounding_checker.is_grounded(&entity.name, source_text);
            with_grounding_flag(entity, grounded)
        })
        .collect()
}

fn with_grounding_flag(mut entity: ExtractedEntity, grounded: bool) -> ExtractedEntity {
    match entity.properties {
        Value::Object(mut map) => {
            map.insert("grounded".to_string(), Value::Bool(grounded));
            entity.properties = Value::Object(map);
        }
        _ => {
            let mut map = Map::new();
            map.insert("grounded".to_string(), Value::Bool(grounded));
            entity.properties = Value::Object(map);
        }
    }
    entity
}
