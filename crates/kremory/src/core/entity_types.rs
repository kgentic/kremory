//! `EntityTypeSpec` + `EntityTypeRegistry` — per-namespace entity type registry.
//!
//! Implements L1 of the unified extraction architecture (TD-013):
//! - `EntityTypeSpec`: a single registered entity type (id, name, description).
//! - `EntityTypeRegistry`: per-namespace store loaded from the `entity_types` DB table.
//!
//! ## Integer-ID backbone (L1)
//!
//! The registry maps integer IDs ↔ type names within a group_id namespace.
//! id=0 is always "Entity" (catch-all). All other IDs are ≥ 1.
//!
//! ## L3 bounds validation
//!
//! `EntityTypeRegistry::validate_or_fallback` enforces that an LLM-emitted
//! `entity_type_id` belongs to the active namespace. Out-of-range or unknown
//! IDs are remapped to id=0 ("Entity") with a metrics counter increment.
//!
//! ## Loading
//!
//! Call `EntityTypeRegistry::load_for_group` to populate from the DB table
//! seeded by Migration 008.

use metrics::counter;

// ─── Default vocabulary ────────────────────────────────────────────────────────

/// Default OntoNotes-style entity type vocabulary.
///
/// Every namespace ships with this vocabulary so consumers can ingest without
/// pre-knowing domain-specific types. Override (`SourceParams.entity_types_override`)
/// augments for domain-specific vocabulary on top of these defaults.
///
/// id=0 is reserved for the "Entity" catch-all per spec §1.
/// All other ids start at 1 and correspond to the standard NER labels
/// carried by Graphiti, Cognee, and LightRAG.
pub const DEFAULT_ENTITY_TYPES: &[(u32, &str, &str)] = &[
    (0, "Entity", "Catch-all for entities that don't fit other types. Use ONLY when no other type applies — not for pronouns, generic nouns, or placeholder values."),
    (1, "Person", "A named individual identified by proper name. Example: 'Alice Chen', 'Dr. Martinez'."),
    (2, "Organisation", "A named company, institution, agency, or formal group. Example: 'Acme Corp', 'Stanford University'."),
    (3, "Location", "A named place, city, country, or geographic feature. Example: 'Boston', 'Great Barrier Reef'."),
    (4, "Date", "A calendar date or date range. Example: '15 March 2024', 'Q3 2023'."),
    (5, "Time", "A time of day or duration. Example: '3:30 PM', '20 minutes'."),
    (6, "Money", "A monetary amount with currency. Example: '$1.5 million', '€500'."),
    (7, "Quantity", "A measurement or count with units. Example: '300 metres', '12 samples'."),
    (8, "Event", "A named occurrence, meeting, or conference. Example: 'NeurIPS 2024', 'Annual Review Meeting'."),
    (9, "Concept", "A named abstract entity, theory, or methodology. Example: 'OAuth 2.0', 'Six Sigma'."),
    (10, "Court", "A court, tribunal, or judicial body. Example: 'Los Angeles Superior Court', 'Court of Appeal'."),
];

/// Seed the default entity_types vocabulary for `group_id` if not already present.
///
/// ## Semantics
///
/// - If `entity_types` has **any** row for `group_id` → no-op, return `Ok(0)`.
///   (Presence check: if ANY rows exist, the group already has a vocabulary.)
/// - If `entity_types` is empty for `group_id` → INSERT OR IGNORE each entry
///   from [`DEFAULT_ENTITY_TYPES`] and return the count of rows inserted.
///
/// ## Idempotency
///
/// `INSERT OR IGNORE` guards against PK conflicts on concurrent callers.
/// The presence-check short-circuit prevents redundant work on the common path.
///
/// ## Call sites
///
/// - **Migration 010** — backfills every `group_id` observed in `entities`
///   or `entity_types` at the time the migration runs.
/// - **`Engine::ingest_with`** — lazy seed for namespaces created after
///   Migration 010 ran (i.e., brand-new `group_id` values seen for the first time).
pub async fn ensure_default_types_seeded(
    conn: &libsql::Connection,
    group_id: &str,
) -> crate::core::error::Result<usize> {
    // Presence check: any rows for this group_id?
    let existing_count: i64 = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1",
                libsql::params![group_id],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "ensure_default_types_seeded presence check failed for group_id={}: {}",
                    group_id,
                    e
                ))
            })?;
        let row = rows.next().await.map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "ensure_default_types_seeded presence row read failed: {e}"
            ))
        })?;
        match row {
            Some(r) => r.get::<i64>(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "ensure_default_types_seeded count parse failed: {e}"
                ))
            })?,
            None => 0,
        }
    };

    if existing_count > 0 {
        return Ok(0);
    }

    // INSERT OR IGNORE each default type (no-op on PK conflict from concurrent callers).
    let mut inserted = 0usize;
    for (id, name, description) in DEFAULT_ENTITY_TYPES {
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (group_id, id, name, description) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![group_id, *id as i64, *name, *description],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "ensure_default_types_seeded insert failed (id={id}, group_id={group_id}): {e}"
            ))
        })?;
        inserted += 1;
    }

    counter!(
        "rql.entity_types.default_seed_applied",
        "group_id" => group_id.to_string()
    )
    .increment(1);

    Ok(inserted)
}

/// Single entry in the entity type registry.
///
/// Mirrors one row from `entity_types(group_id, id, name, description)`.
/// id=0 is the "Entity" catch-all sentinel; all user-defined types are id ≥ 1.
#[derive(Debug, Clone)]
pub struct EntityTypeSpec {
    /// Integer ID within this namespace. id=0 = "Entity" catch-all.
    pub id: u32,
    /// Canonical type name (e.g. "Person", "Organisation", "Court").
    pub name: String,
    /// Anti-junk description used in extraction prompts.
    pub description: String,
}

/// Per-namespace entity type registry.
///
/// Holds the full `entity_types` table slice for one `group_id`.
/// Constructed via [`EntityTypeRegistry::load_for_group`] on engine open.
///
/// Thread-safety: `EntityTypeRegistry` is `Clone` + `Send` + `Sync`; callers
/// can share it via `Arc<EntityTypeRegistry>`.
#[derive(Debug, Clone, Default)]
pub struct EntityTypeRegistry {
    /// All type entries for this namespace, ordered by id ascending.
    specs: Vec<EntityTypeSpec>,
}

impl EntityTypeRegistry {
    /// Construct an empty registry (useful for tests + default namespace init).
    pub fn empty() -> Self {
        Self { specs: Vec::new() }
    }

    /// Build a registry from a pre-loaded `Vec<EntityTypeSpec>`.
    ///
    /// The vec need not be sorted; the constructor sorts by id ascending.
    /// Always includes id=0 if present in the vec.
    pub fn from_specs(mut specs: Vec<EntityTypeSpec>) -> Self {
        specs.sort_by_key(|s| s.id);
        Self { specs }
    }

    /// Load registry from `entity_types` table for the given `group_id`.
    ///
    /// Returns an empty registry (not an error) when the table is empty for
    /// the namespace — Migration 008 seeds id=0 per-group_id so this should
    /// only occur on first-open of a brand-new namespace before migration runs.
    pub async fn load_for_group(
        conn: &libsql::Connection,
        group_id: &str,
    ) -> crate::core::error::Result<Self> {
        let mut rows = conn
            .query(
                "SELECT id, name, description FROM entity_types WHERE group_id = ?1 ORDER BY id ASC",
                libsql::params![group_id],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "EntityTypeRegistry::load_for_group query failed for group_id={}: {}",
                    group_id,
                    e
                ))
            })?;

        let mut specs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| crate::core::error::Error::Other(anyhow::anyhow!("row iteration: {e}")))?
        {
            let id: i64 = row.get(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!("entity_types.id read: {e}"))
            })?;
            let name: String = row.get(1).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!("entity_types.name read: {e}"))
            })?;
            let description: String = row.get(2).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "entity_types.description read: {e}"
                ))
            })?;
            // Guard: negative IDs are a data corruption signal; skip silently + metric.
            if id < 0 {
                counter!("rql.entity_type_registry.corrupt_id_skipped").increment(1);
                continue;
            }
            specs.push(EntityTypeSpec {
                id: id as u32,
                name,
                description,
            });
        }

        Ok(Self::from_specs(specs))
    }

    /// Resolve `entity_type_id` to a type name string.
    ///
    /// Returns the `name` for the matching id, or "Entity" (catch-all) when
    /// id=0 or the id is not registered.  Never panics.
    pub fn id_to_name(&self, id: u32) -> &str {
        for spec in &self.specs {
            if spec.id == id {
                return &spec.name;
            }
        }
        "Entity"
    }

    /// Resolve a type name to its integer id within this namespace.
    ///
    /// Case-sensitive match (canonical form after `normalize_label`).
    /// Returns `None` when the name is not registered.
    pub fn name_to_id(&self, name: &str) -> Option<u32> {
        self.specs.iter().find(|s| s.name == name).map(|s| s.id)
    }

    /// Return a slice of all registered types for this namespace.
    ///
    /// Used by Phase 3 (L2) extraction prompts: the prompt lists known types
    /// from this slice so the LLM can assign meaningful ids.
    pub fn specs(&self) -> &[EntityTypeSpec] {
        &self.specs
    }

    /// Return the maximum registered id.
    ///
    /// Needed by L3 bounds validation (validate_or_fallback): any LLM-emitted
    /// id > max_id is out-of-range and must fall back to id=0.
    pub fn max_id(&self) -> u32 {
        self.specs.iter().map(|s| s.id).max().unwrap_or(0)
    }

    /// L3 bounds validation: validate an LLM-emitted `entity_type_id`.
    ///
    /// ## Invariants
    ///
    /// - id=0 always passes (catch-all sentinel).
    /// - Any id registered in this namespace passes.
    /// - Any unknown or out-of-range id is remapped to 0 and a metric is fired.
    ///
    /// ## Metrics
    ///
    /// `rql.extraction.entity_type_id_fallback` with:
    /// - `reason = "out_of_range"` — emitted id exceeds max registered id
    /// - `reason = "unknown_id"` — emitted id ≤ max_id but not in registry (gap)
    ///
    /// ## Caller responsibility
    ///
    /// Call this function immediately after LLM extraction, before any DB write.
    /// Never write an unvalidated id to `entities.entity_type_id`.
    pub fn validate_or_fallback(&self, emitted_id: u32) -> u32 {
        // id=0 is always the catch-all — no validation needed.
        if emitted_id == 0 {
            return 0;
        }

        let max = self.max_id();
        if emitted_id > max {
            counter!(
                "rql.extraction.entity_type_id_fallback",
                "reason" => "out_of_range",
            )
            .increment(1);
            return 0;
        }

        // Check for gaps: id ≤ max_id but not in registry.
        if self.name_to_id_by_id(emitted_id).is_none() {
            counter!(
                "rql.extraction.entity_type_id_fallback",
                "reason" => "unknown_id",
            )
            .increment(1);
            return 0;
        }

        emitted_id
    }

    /// Internal: find a spec by id (for gap detection in validate_or_fallback).
    fn name_to_id_by_id(&self, id: u32) -> Option<&EntityTypeSpec> {
        self.specs.iter().find(|s| s.id == id)
    }

    /// Resolve a string label (from ingest pipeline) to its registered integer id.
    ///
    /// Used during the label → entity_type_id transition at the ingest boundary:
    /// after normalization, look up the label in this registry.
    /// - Returns the registered id when found.
    /// - Returns 0 ("Entity" catch-all) for placeholder labels ("Entity", "UNKNOWN", "")
    ///   OR when the label is not yet registered in this namespace.
    ///
    /// This is NOT an error path — unregistered labels gracefully fall to id=0.
    /// Phase 3 (L2 prompt engineering) ensures the LLM outputs known type names,
    /// so unregistered labels in production represent novel domains that haven't
    /// yet grown their registry.
    pub fn label_to_id(&self, label: &str) -> u32 {
        let trimmed = label.trim();
        // Placeholder labels always map to id=0.
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("entity")
            || trimmed.eq_ignore_ascii_case("unknown")
        {
            return 0;
        }
        self.name_to_id(trimmed).unwrap_or(0)
    }

    /// Check whether this registry has any entries (besides potentially id=0).
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

/// TD-021: Resolve a label to its registered integer id, registering a NEW
/// row in `entity_types` if the label is novel.
///
/// ## Semantics (open-vocabulary per TD-021 §2-B)
///
/// - Trim + placeholder labels ("Entity"/"UNKNOWN"/empty) → return 0 (catch-all).
/// - If `registry` already contains the label (case-insensitive) → return cached id.
/// - Otherwise: allocate next free id via `SELECT COALESCE(MAX(id), 0) + 1 FROM
///   entity_types WHERE group_id = ?`, INSERT OR IGNORE the new row, return the
///   id. The in-memory `registry` is NOT mutated — caller must re-load via
///   `load_for_group` on next use to see the new entry, OR pass a freshly-loaded
///   registry per call site (matches kremory's stateless-registry usage).
///
/// ## Race condition handling
///
/// Race safety depends on the `UNIQUE (group_id, name)` constraint at
/// `migrations.rs:1007`. If two concurrent callers register the same novel
/// label, INSERT OR IGNORE drops the second insert, then both callers
/// re-SELECT the actual id. No duplicate rows.
///
/// ## Metrics
///
/// - `rql.entity_types.registered_total{trigger="llm_emit"}` — net-new types.
/// - `rql.entity_types.label_cache_hit_total` — labels found in existing registry.
///
/// `group_id` is deliberately NOT a label — unbounded cardinality.
pub async fn label_to_id_or_register(
    conn: &libsql::Connection,
    group_id: &str,
    registry: &EntityTypeRegistry,
    label: &str,
) -> crate::core::error::Result<u32> {
    let trimmed = label.trim();
    // Placeholder labels always map to id=0 (catch-all).
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("entity")
        || trimmed.eq_ignore_ascii_case("unknown")
    {
        return Ok(0);
    }

    // Fast path: already in registry. Case-insensitive lookup — per Cognee+LightRAG
    // convergence, near-duplicate type names ("CustomType" / "Customtype" / "customtype")
    // are the same semantic type. normalize_label upstream can introduce case drift
    // (title-case flattens internal capitals like "CustomType" → "Customtype"); the
    // registry must absorb that variance, not register duplicates.
    if let Some(id) = registry
        .specs()
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case(trimmed))
        .map(|s| s.id)
    {
        counter!("rql.entity_types.label_cache_hit_total").increment(1);
        return Ok(id);
    }

    // Slow path: also check DB case-insensitively before allocating — the
    // in-memory registry may be stale relative to a concurrent writer.
    let existing_id: Option<i64> = {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 \
                 AND LOWER(name) = LOWER(?2) LIMIT 1",
                libsql::params![group_id, trimmed.to_string()],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register DB case-insensitive lookup failed: {e}"
                ))
            })?;
        match rows.next().await.map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "label_to_id_or_register DB lookup row read failed: {e}"
            ))
        })? {
            Some(r) => Some(r.get::<i64>(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register DB lookup id parse failed: {e}"
                ))
            })?),
            None => None,
        }
    };
    if let Some(id) = existing_id {
        counter!("rql.entity_types.label_cache_hit_total", "via" => "db_lookup").increment(1);
        return Ok(id as u32);
    }

    // Slow path: register. Allocate next free id for this namespace.
    let next_id: i64 = {
        let mut rows = conn
            .query(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM entity_types WHERE group_id = ?1",
                libsql::params![group_id],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register MAX(id) query failed: {e}"
                ))
            })?;
        let row = rows.next().await.map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "label_to_id_or_register MAX(id) row read failed: {e}"
            ))
        })?;
        match row {
            Some(r) => r.get::<i64>(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register MAX(id) parse failed: {e}"
                ))
            })?,
            None => 1,
        }
    };

    let description = format!(
        "LLM-discovered type (TD-021 open vocabulary). Surface form: '{}'.",
        trimmed
    );

    // INSERT OR IGNORE guards the race per UNIQUE(group_id, name) at migrations.rs:1007.
    conn.execute(
        "INSERT OR IGNORE INTO entity_types (group_id, id, name, description) \
         VALUES (?1, ?2, ?3, ?4)",
        libsql::params![group_id, next_id, trimmed.to_string(), description],
    )
    .await
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "label_to_id_or_register insert failed for label='{}': {e}",
            trimmed
        ))
    })?;

    // Re-SELECT actual id (handles race: another caller may have inserted
    // the same name with a different id between our MAX query and INSERT).
    let actual_id: i64 = {
        let mut rows = conn
            .query(
                "SELECT id FROM entity_types WHERE group_id = ?1 AND name = ?2",
                libsql::params![group_id, trimmed.to_string()],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register re-SELECT failed: {e}"
                ))
            })?;
        let row = rows
            .next()
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register re-SELECT row read failed: {e}"
                ))
            })?
            .ok_or_else(|| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "label_to_id_or_register: row vanished after INSERT for label='{}'",
                    trimmed
                ))
            })?;
        row.get::<i64>(0).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "label_to_id_or_register id parse failed: {e}"
            ))
        })?
    };

    counter!(
        "rql.entity_types.registered_total",
        "trigger" => "llm_emit",
    )
    .increment(1);

    Ok(actual_id as u32)
}

/// First-call persistence helper for the per-call `EntityTypeSpec` override
/// (spec §1 "Hybrid per-namespace persistent + caller override").
///
/// ## Semantics
///
/// - If `entity_types` has **any** row for `group_id` → no-op, return `Ok(0)`.
///   (Subsequent calls use the override ephemerally — for L2 prompt + L3
///   validation only — without overwriting registered types.)
/// - If `entity_types` is empty for `group_id` → INSERT OR IGNORE each spec,
///   return count of rows inserted.
///
/// Idempotent. Per spec §1: persistence is first-call only.
pub async fn upsert_entity_types(
    conn: &libsql::Connection,
    group_id: &str,
    specs: &[EntityTypeSpec],
) -> crate::core::error::Result<usize> {
    // Presence check: any rows for this group_id?
    let existing_count: i64 = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1",
                libsql::params![group_id],
            )
            .await
            .map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "upsert_entity_types presence check failed: {e}"
                ))
            })?;
        let row = rows.next().await.map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "upsert_entity_types row read failed: {e}"
            ))
        })?;
        match row {
            Some(r) => r.get::<i64>(0).map_err(|e| {
                crate::core::error::Error::Other(anyhow::anyhow!(
                    "upsert_entity_types count parse failed: {e}"
                ))
            })?,
            None => 0,
        }
    };

    if existing_count > 0 {
        return Ok(0);
    }

    // INSERT OR IGNORE each spec (no duplicate PRIMARY KEY on concurrent callers).
    let mut inserted = 0usize;
    for spec in specs {
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (group_id, id, name, description) \
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![
                group_id,
                spec.id as i64,
                spec.name.clone(),
                spec.description.clone()
            ],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "upsert_entity_types insert failed (id={}): {e}",
                spec.id
            ))
        })?;
        inserted += 1;
    }
    Ok(inserted)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── upsert_entity_types DB tests (TD-013 per-call override) ──────────────
    //
    // Tests #1-#3 call `upsert_entity_types` which does not yet exist.
    // Expected failure: E0425 (cannot find fn `upsert_entity_types` in module `super`).
    // Green phase adds the fn to this module; tests pass unchanged.

    /// Helper: open an in-memory libsql connection with all kremory migrations applied.
    ///
    /// Mirrors the pattern used by `TemporalGraph::open_in_memory` so migration 008
    /// (entity_types table) is present for upsert tests.
    async fn open_migrated_conn() -> libsql::Connection {
        let graph = crate::core::schema::TemporalGraph::open_in_memory()
            .await
            .expect("open_in_memory for entity_types upsert tests");
        // SAFETY: conn is pub on TemporalGraph (schema.rs:266).
        // We move it out via the Arc — tests hold the graph Arc so the
        // connection stays live throughout the test body.
        graph.conn.clone()
    }

    /// Count rows in entity_types for a given group_id.
    async fn count_rows(conn: &libsql::Connection, group_id: &str) -> usize {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id = ?1",
                libsql::params![group_id],
            )
            .await
            .expect("count query");
        let row = rows.next().await.expect("row iter").expect("row");
        let n: i64 = row.get(0).expect("count col");
        n as usize
    }

    /// #1 — upsert_entity_types inserts all specs on a fresh (empty) group_id.
    ///
    /// Setup: in-memory DB with migrations through 009. entity_types empty for "t1".
    /// Action: call upsert_entity_types(conn, "t1", &specs) with 3 specs.
    /// Assert: Ok(3) returned. entity_types has exactly 3 rows for "t1".
    ///
    /// COMPILE FAIL (Red): `upsert_entity_types` does not exist until Green phase.
    #[tokio::test]
    async fn test_upsert_entity_types_inserts_on_empty() {
        let conn = open_migrated_conn().await;

        // Ensure group "t1" has no rows (fresh DB after migrations should have none
        // unless migration 008 seeds a catch-all — which it only does for groups
        // that already have entity rows; "t1" has none so entity_types is empty for it).
        let before = count_rows(&conn, "t1").await;
        assert_eq!(
            before, 0,
            "pre-condition: entity_types must be empty for 't1'"
        );

        let specs = vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "Catch-all entity type.".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A human individual.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "A company or institution.".to_string(),
            },
        ];

        let inserted = upsert_entity_types(&conn, "t1", &specs)
            .await
            .expect("upsert_entity_types must succeed on empty group");

        assert_eq!(
            inserted, 3,
            "upsert_entity_types must return 3 (rows inserted) on a fresh group; got {inserted}"
        );

        let after = count_rows(&conn, "t1").await;
        assert_eq!(
            after, 3,
            "entity_types must have exactly 3 rows for 't1' after upsert; got {after}"
        );
    }

    /// #2 — upsert_entity_types is a no-op (returns 0) when the group already has rows.
    ///
    /// Persistence is "first-call" only per spec §1: once the group has any rows,
    /// the override is applied ephemerally for the current call only — DB is not
    /// clobbered. This test verifies that invariant at the upsert_entity_types level.
    ///
    /// Setup: in-memory DB. Manually insert 1 row (id=0) for "t2".
    /// Action: call upsert_entity_types with 5 specs (including id=0) for "t2".
    /// Assert: Ok(0) returned. entity_types still has exactly 1 row for "t2".
    ///
    /// COMPILE FAIL (Red): `upsert_entity_types` does not exist until Green phase.
    #[tokio::test]
    async fn test_upsert_entity_types_noop_when_rows_exist() {
        let conn = open_migrated_conn().await;

        // Pre-seed: insert one row so the group is non-empty.
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (group_id, id, name, description) \
             VALUES ('t2', 0, 'Entity', 'catch-all')",
            libsql::params![],
        )
        .await
        .expect("pre-seed insert");

        let before = count_rows(&conn, "t2").await;
        assert_eq!(
            before, 1,
            "pre-condition: exactly 1 row for 't2' before upsert"
        );

        let specs = vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
            EntityTypeSpec {
                id: 3,
                name: "Location".to_string(),
                description: "A place.".to_string(),
            },
            EntityTypeSpec {
                id: 4,
                name: "Event".to_string(),
                description: "An event.".to_string(),
            },
        ];

        let inserted = upsert_entity_types(&conn, "t2", &specs)
            .await
            .expect("upsert_entity_types must not error on existing group");

        assert_eq!(
            inserted, 0,
            "upsert_entity_types must return 0 (no-op) when group already has rows; got {inserted}"
        );

        let after = count_rows(&conn, "t2").await;
        assert_eq!(
            after, 1,
            "entity_types must still have exactly 1 row for 't2' after no-op upsert; got {after}"
        );
    }

    /// #3 — upsert_entity_types maintains per-group_id isolation.
    ///
    /// Inserting into "alpha" must not affect "beta" and vice versa.
    ///
    /// Setup: empty entity_types.
    /// Action: upsert into group_id="alpha" with 4 specs; upsert into "beta" with 3 specs.
    /// Assert: total 7 rows. alpha has 4, beta has 3.
    ///
    /// COMPILE FAIL (Red): `upsert_entity_types` does not exist until Green phase.
    #[tokio::test]
    async fn test_upsert_entity_types_per_group_isolated() {
        let conn = open_migrated_conn().await;

        // Pre-condition: both groups have zero rows on a fresh DB.
        assert_eq!(
            count_rows(&conn, "alpha").await,
            0,
            "alpha must start empty"
        );
        assert_eq!(count_rows(&conn, "beta").await, 0, "beta must start empty");

        let alpha_specs = vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A person.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "An org.".to_string(),
            },
            EntityTypeSpec {
                id: 3,
                name: "Location".to_string(),
                description: "A place.".to_string(),
            },
        ];
        let beta_specs = vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Court".to_string(),
                description: "A judicial court.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Statute".to_string(),
                description: "A legal statute.".to_string(),
            },
        ];

        let alpha_inserted = upsert_entity_types(&conn, "alpha", &alpha_specs)
            .await
            .expect("upsert alpha");
        let beta_inserted = upsert_entity_types(&conn, "beta", &beta_specs)
            .await
            .expect("upsert beta");

        assert_eq!(
            alpha_inserted, 4,
            "alpha must report 4 inserts; got {alpha_inserted}"
        );
        assert_eq!(
            beta_inserted, 3,
            "beta must report 3 inserts; got {beta_inserted}"
        );

        let alpha_count = count_rows(&conn, "alpha").await;
        let beta_count = count_rows(&conn, "beta").await;

        assert_eq!(alpha_count, 4, "alpha must have 4 rows; got {alpha_count}");
        assert_eq!(beta_count, 3, "beta must have 3 rows; got {beta_count}");

        // Cross-group isolation: alpha's types must not appear under beta and vice-versa.
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id IN ('alpha','beta')",
                libsql::params![],
            )
            .await
            .expect("total count query");
        let total_row = rows.next().await.expect("row iter").expect("row");
        let total: i64 = total_row.get(0).expect("count col");
        assert_eq!(
            total, 7,
            "total rows across both groups must be 7; got {total}"
        );
    }

    // ── ensure_default_types_seeded DB tests ─────────────────────────────────

    /// #D1 — ensure_default_types_seeded inserts 10 rows on a fresh (empty) group_id.
    ///
    /// Setup: in-memory DB with all migrations applied. entity_types empty for "fresh_grp".
    /// Action: call ensure_default_types_seeded(conn, "fresh_grp").
    /// Assert: Ok(10) returned. entity_types has exactly 10 rows for "fresh_grp".
    #[tokio::test]
    async fn test_ensure_default_types_seeded_seeds_ten_rows_on_empty() {
        let conn = open_migrated_conn().await;

        // Pre-condition: "fresh_grp" has no rows on a fresh DB.
        let before = count_rows(&conn, "fresh_grp").await;
        assert_eq!(
            before, 0,
            "pre-condition: entity_types must be empty for 'fresh_grp'"
        );

        let inserted = ensure_default_types_seeded(&conn, "fresh_grp")
            .await
            .expect("ensure_default_types_seeded must succeed on empty group");

        assert_eq!(
            inserted,
            DEFAULT_ENTITY_TYPES.len(),
            "ensure_default_types_seeded must return {} (count of DEFAULT_ENTITY_TYPES) on empty group; got {inserted}",
            DEFAULT_ENTITY_TYPES.len()
        );

        let after = count_rows(&conn, "fresh_grp").await;
        assert_eq!(
            after,
            DEFAULT_ENTITY_TYPES.len(),
            "entity_types must have exactly {} rows for 'fresh_grp' after seeding; got {after}",
            DEFAULT_ENTITY_TYPES.len()
        );
    }

    /// #D2 — ensure_default_types_seeded is a no-op (returns 0) when group already has rows.
    ///
    /// Presence check: if ANY rows exist, seeding must be skipped entirely.
    ///
    /// Setup: manually insert 1 row (id=0) for "existing_grp".
    /// Action: call ensure_default_types_seeded for "existing_grp".
    /// Assert: Ok(0) returned. entity_types still has exactly 1 row for "existing_grp".
    #[tokio::test]
    async fn test_ensure_default_types_seeded_noop_when_any_rows_exist() {
        let conn = open_migrated_conn().await;

        // Pre-seed: insert one row so the group is non-empty.
        conn.execute(
            "INSERT OR IGNORE INTO entity_types (group_id, id, name, description) \
             VALUES ('existing_grp', 0, 'Entity', 'catch-all')",
            libsql::params![],
        )
        .await
        .expect("pre-seed insert");

        let before = count_rows(&conn, "existing_grp").await;
        assert_eq!(
            before, 1,
            "pre-condition: exactly 1 row for 'existing_grp' before seeding"
        );

        let inserted = ensure_default_types_seeded(&conn, "existing_grp")
            .await
            .expect("ensure_default_types_seeded must not error on existing group");

        assert_eq!(
            inserted, 0,
            "ensure_default_types_seeded must return 0 (no-op) when group already has rows; got {inserted}"
        );

        let after = count_rows(&conn, "existing_grp").await;
        assert_eq!(
            after, 1,
            "entity_types must still have exactly 1 row for 'existing_grp' after no-op; got {after}"
        );
    }

    /// #D3 — ensure_default_types_seeded maintains per-group_id isolation.
    ///
    /// Seeding "grp_a" must not affect "grp_b" and vice versa.
    ///
    /// Setup: empty entity_types.
    /// Action: seed "grp_a" then "grp_b" (both fresh).
    /// Assert: each has DEFAULT_ENTITY_TYPES.len() rows; cross-group isolation confirmed.
    #[tokio::test]
    async fn test_ensure_default_types_seeded_per_group_isolated() {
        let conn = open_migrated_conn().await;

        // Pre-condition: both groups start empty.
        assert_eq!(
            count_rows(&conn, "grp_a").await,
            0,
            "grp_a must start empty"
        );
        assert_eq!(
            count_rows(&conn, "grp_b").await,
            0,
            "grp_b must start empty"
        );

        let a_inserted = ensure_default_types_seeded(&conn, "grp_a")
            .await
            .expect("seed grp_a");
        let b_inserted = ensure_default_types_seeded(&conn, "grp_b")
            .await
            .expect("seed grp_b");

        let expected = DEFAULT_ENTITY_TYPES.len();
        assert_eq!(
            a_inserted, expected,
            "grp_a must report {expected} inserts; got {a_inserted}"
        );
        assert_eq!(
            b_inserted, expected,
            "grp_b must report {expected} inserts; got {b_inserted}"
        );

        let a_count = count_rows(&conn, "grp_a").await;
        let b_count = count_rows(&conn, "grp_b").await;

        assert_eq!(
            a_count, expected,
            "grp_a must have {expected} rows; got {a_count}"
        );
        assert_eq!(
            b_count, expected,
            "grp_b must have {expected} rows; got {b_count}"
        );

        // Cross-group isolation: total rows = 2 * DEFAULT_ENTITY_TYPES.len().
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_types WHERE group_id IN ('grp_a', 'grp_b')",
                libsql::params![],
            )
            .await
            .expect("total count query");
        let total_row = rows.next().await.expect("row iter").expect("row");
        let total: i64 = total_row.get(0).expect("count col");
        assert_eq!(
            total as usize,
            expected * 2,
            "total rows across both groups must be {}; got {total}",
            expected * 2
        );
    }

    fn make_registry() -> EntityTypeRegistry {
        EntityTypeRegistry::from_specs(vec![
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: "catch-all".to_string(),
            },
            EntityTypeSpec {
                id: 1,
                name: "Person".to_string(),
                description: "A human individual.".to_string(),
            },
            EntityTypeSpec {
                id: 2,
                name: "Organisation".to_string(),
                description: "A company or institution.".to_string(),
            },
        ])
    }

    // ── id_to_name ────────────────────────────────────────────────────────────

    #[test]
    fn id_to_name_resolves_registered_id() {
        let reg = make_registry();
        assert_eq!(reg.id_to_name(0), "Entity");
        assert_eq!(reg.id_to_name(1), "Person");
        assert_eq!(reg.id_to_name(2), "Organisation");
    }

    #[test]
    fn id_to_name_unknown_id_returns_entity_fallback() {
        let reg = make_registry();
        assert_eq!(reg.id_to_name(99), "Entity");
    }

    // ── name_to_id ────────────────────────────────────────────────────────────

    #[test]
    fn name_to_id_returns_correct_id() {
        let reg = make_registry();
        assert_eq!(reg.name_to_id("Person"), Some(1));
        assert_eq!(reg.name_to_id("Organisation"), Some(2));
    }

    #[test]
    fn name_to_id_unknown_name_returns_none() {
        let reg = make_registry();
        assert!(reg.name_to_id("Court").is_none());
    }

    // ── validate_or_fallback (L3) ─────────────────────────────────────────────

    #[test]
    fn validate_or_fallback_passes_id_zero() {
        let reg = make_registry();
        assert_eq!(reg.validate_or_fallback(0), 0);
    }

    #[test]
    fn validate_or_fallback_passes_registered_id() {
        let reg = make_registry();
        assert_eq!(reg.validate_or_fallback(1), 1);
        assert_eq!(reg.validate_or_fallback(2), 2);
    }

    #[test]
    fn validate_or_fallback_out_of_range_returns_zero() {
        let reg = make_registry();
        // max_id = 2; 99 > 2 → out_of_range → fallback to 0
        assert_eq!(reg.validate_or_fallback(99), 0);
    }

    #[test]
    fn validate_or_fallback_empty_registry_any_nonzero_falls_back() {
        let reg = EntityTypeRegistry::empty();
        // empty registry has max_id = 0; any nonzero is out_of_range
        assert_eq!(reg.validate_or_fallback(0), 0);
        assert_eq!(reg.validate_or_fallback(1), 0);
    }

    // ── label_to_id ───────────────────────────────────────────────────────────

    #[test]
    fn label_to_id_known_label_returns_correct_id() {
        let reg = make_registry();
        assert_eq!(reg.label_to_id("Person"), 1);
        assert_eq!(reg.label_to_id("Organisation"), 2);
    }

    #[test]
    fn label_to_id_placeholder_labels_return_zero() {
        let reg = make_registry();
        assert_eq!(reg.label_to_id("Entity"), 0);
        assert_eq!(reg.label_to_id("UNKNOWN"), 0);
        assert_eq!(reg.label_to_id(""), 0);
        assert_eq!(reg.label_to_id("  "), 0);
    }

    #[test]
    fn label_to_id_unregistered_label_returns_zero() {
        let reg = make_registry();
        assert_eq!(reg.label_to_id("Court"), 0);
    }

    // ── max_id ────────────────────────────────────────────────────────────────

    #[test]
    fn max_id_correct() {
        let reg = make_registry();
        assert_eq!(reg.max_id(), 2);
    }

    #[test]
    fn max_id_empty_returns_zero() {
        let reg = EntityTypeRegistry::empty();
        assert_eq!(reg.max_id(), 0);
    }

    // ── sort invariant ────────────────────────────────────────────────────────

    #[test]
    fn from_specs_sorts_by_id() {
        let reg = EntityTypeRegistry::from_specs(vec![
            EntityTypeSpec {
                id: 2,
                name: "B".to_string(),
                description: String::new(),
            },
            EntityTypeSpec {
                id: 0,
                name: "Entity".to_string(),
                description: String::new(),
            },
            EntityTypeSpec {
                id: 1,
                name: "A".to_string(),
                description: String::new(),
            },
        ]);
        assert_eq!(reg.specs()[0].id, 0);
        assert_eq!(reg.specs()[1].id, 1);
        assert_eq!(reg.specs()[2].id, 2);
    }

    // ─── TD-021: label_to_id_or_register acceptance tests ────────────────────

    /// #L1 — novel label registers a new row + returns id > 0.
    ///
    /// Setup: in-memory DB, group "g1" seeded with DEFAULT_ENTITY_TYPES.
    /// Action: call label_to_id_or_register(&conn, "g1", &registry, "Court").
    /// Assert: returned id > 0, entity_types row count grows by 1.
    #[tokio::test]
    async fn test_label_to_id_or_register_grows_registry_on_novel_label() {
        let conn = open_migrated_conn().await;
        let inserted = upsert_entity_types(
            &conn,
            "g1",
            &DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(id, name, desc)| EntityTypeSpec {
                    id: *id,
                    name: name.to_string(),
                    description: desc.to_string(),
                })
                .collect::<Vec<_>>(),
        )
        .await
        .expect("seed defaults");
        assert!(inserted > 0, "defaults must seed");

        let before_count = count_rows(&conn, "g1").await;
        let registry = EntityTypeRegistry::load_for_group(&conn, "g1")
            .await
            .expect("load registry");

        // "Court" is now a default type (id=10); use "Statute" as a genuinely
        // novel label not present in DEFAULT_ENTITY_TYPES.
        let new_id = label_to_id_or_register(&conn, "g1", &registry, "Statute")
            .await
            .expect("register novel label");

        assert!(
            new_id > 0,
            "novel type must register with id > 0; got {new_id}"
        );
        let after_count = count_rows(&conn, "g1").await;
        assert_eq!(
            after_count,
            before_count + 1,
            "entity_types must grow by exactly 1 row for novel label; before={before_count} after={after_count}"
        );
    }

    /// #L2 — registering the same label twice returns the same id (idempotent).
    #[tokio::test]
    async fn test_label_to_id_or_register_idempotent_on_duplicate() {
        let conn = open_migrated_conn().await;
        upsert_entity_types(
            &conn,
            "g2",
            &DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(id, name, desc)| EntityTypeSpec {
                    id: *id,
                    name: name.to_string(),
                    description: desc.to_string(),
                })
                .collect::<Vec<_>>(),
        )
        .await
        .expect("seed defaults");

        let registry = EntityTypeRegistry::load_for_group(&conn, "g2")
            .await
            .expect("load registry");
        let first = label_to_id_or_register(&conn, "g2", &registry, "Drug")
            .await
            .expect("first register");
        let second = label_to_id_or_register(&conn, "g2", &registry, "Drug")
            .await
            .expect("second register");

        assert_eq!(
            first, second,
            "second registration of same label must return same id; first={first} second={second}"
        );
        let count_after = count_rows(&conn, "g2").await;
        assert_eq!(
            count_after,
            DEFAULT_ENTITY_TYPES.len() + 1,
            "exactly one row added despite duplicate registration"
        );
    }

    /// #L3 — concurrent registration of same novel label = same id (race safety).
    ///
    /// Two tasks concurrently call label_to_id_or_register for "Species". Both
    /// must succeed AND return the same id. Underlying INSERT OR IGNORE +
    /// re-SELECT pattern must collapse the race.
    #[tokio::test]
    async fn test_label_to_id_or_register_concurrent_same_label() {
        let conn = open_migrated_conn().await;
        upsert_entity_types(
            &conn,
            "g3",
            &DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(id, name, desc)| EntityTypeSpec {
                    id: *id,
                    name: name.to_string(),
                    description: desc.to_string(),
                })
                .collect::<Vec<_>>(),
        )
        .await
        .expect("seed defaults");

        let registry = std::sync::Arc::new(
            EntityTypeRegistry::load_for_group(&conn, "g3")
                .await
                .expect("load registry"),
        );

        // libsql::Connection is Clone (under-the-hood Arc).
        let conn_a = conn.clone();
        let conn_b = conn.clone();
        let reg_a = std::sync::Arc::clone(&registry);
        let reg_b = std::sync::Arc::clone(&registry);

        let h1 =
            tokio::spawn(
                async move { label_to_id_or_register(&conn_a, "g3", &reg_a, "Species").await },
            );
        let h2 =
            tokio::spawn(
                async move { label_to_id_or_register(&conn_b, "g3", &reg_b, "Species").await },
            );
        let id1 = h1.await.expect("h1 join").expect("h1 register");
        let id2 = h2.await.expect("h2 join").expect("h2 register");

        assert_eq!(
            id1, id2,
            "concurrent registration must yield same id; id1={id1} id2={id2}"
        );
        let count_after = count_rows(&conn, "g3").await;
        assert_eq!(
            count_after,
            DEFAULT_ENTITY_TYPES.len() + 1,
            "race must collapse to single row; got {count_after}"
        );
    }

    /// #L4 — placeholder labels ("Entity", "UNKNOWN", "") map to id=0 (catch-all).
    #[tokio::test]
    async fn test_label_to_id_or_register_placeholder_returns_zero() {
        let conn = open_migrated_conn().await;
        upsert_entity_types(
            &conn,
            "g4",
            &DEFAULT_ENTITY_TYPES
                .iter()
                .map(|(id, name, desc)| EntityTypeSpec {
                    id: *id,
                    name: name.to_string(),
                    description: desc.to_string(),
                })
                .collect::<Vec<_>>(),
        )
        .await
        .expect("seed defaults");
        let registry = EntityTypeRegistry::load_for_group(&conn, "g4")
            .await
            .expect("load registry");

        for label in [
            "Entity", "ENTITY", "entity", "UNKNOWN", "unknown", "", "   ",
        ] {
            let id = label_to_id_or_register(&conn, "g4", &registry, label)
                .await
                .unwrap_or_else(|e| {
                    panic!("placeholder '{label}' must map cleanly to 0; got error: {e}")
                });
            assert_eq!(
                id, 0,
                "placeholder label '{label}' must map to id=0 (catch-all); got {id}"
            );
        }
        // No new rows added for placeholders.
        let count_after = count_rows(&conn, "g4").await;
        assert_eq!(
            count_after,
            DEFAULT_ENTITY_TYPES.len(),
            "placeholder labels must NOT add rows; expected {} got {count_after}",
            DEFAULT_ENTITY_TYPES.len()
        );
    }
}
