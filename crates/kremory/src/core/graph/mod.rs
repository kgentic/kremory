use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::core::schema::{Entity, Fact};

mod entities;
mod entity_groups;
mod episodes;
mod facts;
mod namespace;
mod queries;

pub use facts::FactInsert;

#[cfg(test)]
mod tests;

/// Compute a hex-encoded SHA-256 content hash for a fact triple.
/// Hash input: `"{subject_id}\x00{predicate}\x00{object_key}"` where
/// `object_key` is `object_id` if set, otherwise `object_value`, otherwise `""`.
/// Story #209.
pub(super) fn fact_content_hash(
    subject_id: &str,
    predicate: &str,
    object_id: Option<&str>,
    object_value: Option<&str>,
) -> String {
    let object_key = object_id.or(object_value).unwrap_or("");
    let mut hasher = Sha256::new();
    hasher.update(subject_id.as_bytes());
    hasher.update(b"\x00");
    hasher.update(predicate.as_bytes());
    hasher.update(b"\x00");
    hasher.update(object_key.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone)]
pub struct SubGraph {
    pub entities: Vec<Entity>,
    pub facts: Vec<Fact>,
}

pub(super) fn parse_dt(s: &str) -> anyhow::Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(s)
        .map_err(|e| anyhow::anyhow!("bad timestamp '{}': {}", s, e))?
        .with_timezone(&Utc))
}

/// Expected columns from the canonical entity SELECT (LEFT JOIN entity_types):
/// 0: e.id, 1: COALESCE(et.name,'Entity') as label, 2: e.properties,
/// 3: e.recorded_at, 4: e.updated_at, 5: e.group_id, 6: e.access_count,
/// 7: e.entity_type_id
pub(super) fn row_to_entity(row: &libsql::Row) -> anyhow::Result<Entity> {
    let id: String = row.get::<String>(0)?;
    let label: String = row.get::<String>(1)?;
    let props_str: Option<String> = row.get::<Option<String>>(2)?;
    let created_str: String = row.get::<String>(3)?;
    let updated_str: Option<String> = row.get::<Option<String>>(4)?;
    let group_id: Option<String> = row.get::<Option<String>>(5)?;
    let access_count: i64 = row.get::<i64>(6)?;
    let entity_type_id_raw: i64 = row.get::<i64>(7)?;
    let entity_type_id: u32 = entity_type_id_raw.max(0) as u32;

    let properties: serde_json::Value = props_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);
    let recorded_at = parse_dt(&created_str)?;
    let updated_at = updated_str.as_deref().map(parse_dt).transpose()?;

    Ok(Entity {
        id,
        label,
        entity_type_id,
        properties,
        recorded_at,
        updated_at,
        group_id,
        access_count,
    })
}

pub(super) fn row_to_fact(row: &libsql::Row) -> anyhow::Result<Fact> {
    let id: i64 = row.get::<i64>(0)?;
    let subject_id: String = row.get::<String>(1)?;
    let predicate: String = row.get::<String>(2)?;
    let object_id: Option<String> = row.get::<Option<String>>(3)?;
    let object_value: Option<String> = row.get::<Option<String>>(4)?;
    let props_str: Option<String> = row.get::<Option<String>>(5)?;
    let valid_from_str: String = row.get::<String>(6)?;
    let valid_to_str: Option<String> = row.get::<Option<String>>(7)?;
    let recorded_str: String = row.get::<String>(8)?;
    let expired_str: Option<String> = row.get::<Option<String>>(9)?;
    let invalid_str: Option<String> = row.get::<Option<String>>(10)?;
    let group_id: Option<String> = row.get::<Option<String>>(11)?;
    let confidence: f64 = row.get::<f64>(12)?;
    let source_episode_id: Option<i64> = row.get::<Option<i64>>(13)?;
    let memory_type_str: Option<String> = row.get::<Option<String>>(14)?;
    let content_hash: Option<String> = row.get::<Option<String>>(15)?;
    let access_count: i64 = row.get::<i64>(16)?;

    let properties: Option<serde_json::Value> = props_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let valid_from = parse_dt(&valid_from_str)?;
    let valid_to = valid_to_str.as_deref().map(parse_dt).transpose()?;
    let recorded_at = parse_dt(&recorded_str)?;
    let expired_at = expired_str.as_deref().map(parse_dt).transpose()?;
    let invalid_at = invalid_str.as_deref().map(parse_dt).transpose()?;
    let memory_type = memory_type_str
        .as_deref()
        .and_then(|s| serde_json::from_str(&format!("\"{s}\"")).ok());

    Ok(Fact {
        id,
        subject_id,
        predicate,
        object_id,
        object_value,
        properties,
        valid_from,
        valid_to,
        recorded_at,
        expired_at,
        invalid_at,
        group_id,
        confidence,
        source_episode_id,
        memory_type,
        content_hash,
        access_count,
        // ADR-029b: composite FK fields — absent on pre-migration-004 rows;
        // populated by the migration 004 backfill. None on fresh rows until
        // the caller explicitly sets subject_group_id / object_group_id.
        subject_group_id: None,
        object_group_id: None,
    })
}
