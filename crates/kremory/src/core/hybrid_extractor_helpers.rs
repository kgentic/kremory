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

/// Bundled parameters for [`merge_entities_with_grounding`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments).
pub(crate) struct MergeEntitiesWithGroundingParams<'a> {
    /// The base entity set; mutated and returned.
    pub base: Vec<ExtractedEntity>,
    /// Additional entities to merge into `base`.
    pub additive: Vec<ExtractedEntity>,
    /// Source text used to compute the grounding flag.
    pub source_text: &'a str,
    /// Grounding checker used to flag each entity.
    pub grounding_checker: &'a dyn GroundingChecker,
}

pub(crate) fn merge_entities_with_grounding(
    params: MergeEntitiesWithGroundingParams<'_>,
) -> Vec<ExtractedEntity> {
    let MergeEntitiesWithGroundingParams {
        mut base,
        additive,
        source_text,
        grounding_checker,
    } = params;
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
