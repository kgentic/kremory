//! Consumer INSPECT surface for the reversible-graph-mutations substrate
//! (arch-spec `reversible-graph-mutations-arch-spec-2026-07-10.md` §3 "Inspect
//! surface"). The **SEE** half of the see+fix story: a consumer asks "what did
//! `dream()` do to entity X?" and gets back a human/agent-readable
//! [`MutationRecord`] carrying the `mutation_id` needed to reverse it (via
//! `Memory::unmerge` / `restore_archived_fact` / `unsupersede`).
//!
//! DETERMINISTIC + READ-ONLY — no LLM, no writes, no txn. Every
//! [`MutationRecord`] is DERIVED from the stored `inputs` / `pre_state` snapshot
//! (§2.3) — the raw JSON internals are never dumped to the consumer; only the
//! affected-entity ids and a short summary are surfaced.
//!
//! ## `pub` + `#[doc(hidden)]` (MNT-002 precedent)
//!
//! `list_mutations` / `mutation_history` are `pub` + `#[doc(hidden)]` (re-exported
//! under `feature = "test-utils"` from `core/dream/mod.rs`) for the same E0365
//! reason as the reversal free-functions: external integration-test binaries
//! cannot import `pub(crate)` items. They are NOT the stable public API — the
//! consumer surface is the `Memory` facade (`facade/reverse.rs`). The DATA types
//! ([`MutationRecord`] / [`MutationFilter`] / [`super::MutationKind`]) ARE public
//! (re-exported from the facade) since they appear in the facade return type.

use metrics::counter;

use crate::core::error::{Error, Result};
use crate::core::schema::TemporalGraph;

use super::{MergeInputs, MutationKind};

// ─── consumer-facing view (§3 inspect surface) ──────────────────────────────

/// A consumer-facing, human/agent-readable view of one logged graph mutation —
/// the **SEE** half of the reversible-mutation story. Derived from the stored
/// `graph_mutation_log` snapshot (§2.3), NEVER the raw JSON internals.
///
/// `#[non_exhaustive]`: field additions (e.g. a per-kind detail struct) stay
/// non-breaking as later Stage-2 kinds gain inspect shapes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MutationRecord {
    /// The `graph_mutation_log.id` — pass this to `Memory::unmerge` (or the
    /// matching undo method for the kind) to reverse the mutation.
    pub mutation_id: i64,
    /// What kind of mutation this was (the `graph_mutation_log.kind` tag, typed).
    pub kind: MutationKind,
    /// RFC3339 timestamp when the mutation was applied.
    pub created_at: String,
    /// `true` once the mutation has been reversed (`graph_mutation_log.undone_at`
    /// is set) — an already-undone mutation is inert to a second undo.
    pub undone: bool,
    /// The namespace (group) this mutation scoped.
    pub group_id: String,
    /// Entity ids this mutation touched. For `entity_merge` this is
    /// `[keeper, loser]` (derived from `inputs`, §2.3) — the pair the consumer
    /// can locate by id.
    pub affected_entities: Vec<String>,
    /// A short human/agent-readable summary of what the mutation did, e.g.
    /// `merged 'alice j' into 'alice johnson' (site=canonicalize)`. Derived from
    /// the snapshot; contains no raw JSON.
    pub summary: String,
}

/// Filter for [`list_mutations`]. All fields optional. `include_undone` defaults
/// `false` — the default view shows only LIVE (still-reversible) mutations, the
/// actionable set; set it `true` for the full audit history.
#[derive(Debug, Clone, Default)]
pub struct MutationFilter {
    /// Restrict to one namespace (group). `None` → all namespaces.
    pub group_id: Option<String>,
    /// Restrict to one mutation kind. `None` → every kind.
    pub kind: Option<MutationKind>,
    /// Only mutations at/after this RFC3339 timestamp (`created_at >=`). `None`
    /// → no lower bound.
    pub since: Option<String>,
    /// Include already-undone mutations. Default `false` (live only).
    pub include_undone: bool,
}

// ─── derivation (§2.3 — inputs → affected + summary, no raw dump) ────────────

/// Derive the consumer-facing `(affected_entities, summary)` from the stored
/// per-kind `inputs` JSON (§2.3), WITHOUT exposing raw internals.
///
/// For `entity_merge` the `inputs` are our OWN structured emit, so a
/// deserialize failure is a hard `Error` (parse-loudly, §2.1) — a merge row we
/// cannot classify must surface, never silently render as a blank record. The
/// remaining Stage-1 kinds have no producer yet (§2.3 NOTE); until their writers
/// ship they render an HONEST generic summary (their id list is genuinely empty,
/// not a fabricated default).
fn derive_view(kind: MutationKind, inputs_json: &str) -> Result<(Vec<String>, String)> {
    match kind {
        MutationKind::EntityMerge => {
            let inputs: MergeInputs = serde_json::from_str(inputs_json).map_err(|e| {
                Error::Other(anyhow::anyhow!(
                    "inspect: deserialize entity_merge inputs (mutation un-classifiable): {e}"
                ))
            })?;
            let summary = format!(
                "merged '{}' into '{}' (site={})",
                inputs.loser,
                inputs.keeper,
                inputs.site.as_str()
            );
            // `[keeper, loser]` — the pair the consumer locates by id (§6.2 the
            // nogood is over the sorted pair; the human view keeps role order).
            Ok((vec![inputs.keeper, inputs.loser], summary))
        }
        other => Ok((Vec::new(), format!("{} mutation", other.as_tag()))),
    }
}

/// Build one [`MutationRecord`] from a `graph_mutation_log` row. Sync — pure
/// derivation. The row's column order is the `list_mutations` SELECT projection:
/// `(id, kind, group_id, created_at, undone_at, inputs)`.
fn build_record(row: &libsql::Row) -> Result<MutationRecord> {
    let mutation_id = row.get::<i64>(0)?;
    let kind = MutationKind::from_tag(&row.get::<String>(1)?)?;
    let group_id = row.get::<String>(2)?;
    let created_at = row.get::<String>(3)?;
    let undone_at = row.get::<Option<String>>(4)?;
    let inputs_json = row.get::<String>(5)?;
    let (affected_entities, summary) = derive_view(kind, &inputs_json)?;
    Ok(MutationRecord {
        mutation_id,
        kind,
        created_at,
        undone: undone_at.is_some(),
        group_id,
        affected_entities,
        summary,
    })
}

// ─── list_mutations (§3 inspect surface) ────────────────────────────────────

/// List logged graph mutations matching `filter`, newest-first
/// (`created_at DESC, id DESC` for a stable tie-break). Read-only, deterministic.
///
/// The `WHERE` clause is built from the optional filter fields; every value is a
/// BOUND parameter (no interpolation) — the `kind` tag binds as its `as_tag()`
/// string, matching what the executor wrote (§2.1). `include_undone = false`
/// (the default) adds `undone_at IS NULL` so only still-reversible mutations
/// surface.
#[doc(hidden)]
pub async fn list_mutations(
    graph: &TemporalGraph,
    filter: MutationFilter,
) -> Result<Vec<MutationRecord>> {
    let mut sql = String::from(
        "SELECT id, kind, group_id, created_at, undone_at, inputs \
         FROM graph_mutation_log WHERE 1 = 1",
    );
    let mut params: Vec<libsql::Value> = Vec::new();
    let mut n = 0;
    if let Some(g) = &filter.group_id {
        n += 1;
        sql.push_str(&format!(" AND group_id = ?{n}"));
        params.push(libsql::Value::from(g.clone()));
    }
    if let Some(k) = filter.kind {
        n += 1;
        sql.push_str(&format!(" AND kind = ?{n}"));
        params.push(libsql::Value::from(k.as_tag().to_string()));
    }
    if let Some(s) = &filter.since {
        n += 1;
        sql.push_str(&format!(" AND created_at >= ?{n}"));
        params.push(libsql::Value::from(s.clone()));
    }
    if !filter.include_undone {
        sql.push_str(" AND undone_at IS NULL");
    }
    sql.push_str(" ORDER BY created_at DESC, id DESC");

    let mut rows = graph.conn.query(&sql, params).await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push(build_record(&row)?);
    }
    tracing::debug!(
        target: "kremory.graph.provenance",
        group_id = ?filter.group_id,
        kind = ?filter.kind,
        include_undone = filter.include_undone,
        returned = out.len(),
        "kremory.graph.list_mutations"
    );
    counter!("kremory.graph.inspect_query_total", "op" => "list_mutations").increment(1);
    Ok(out)
}

// ─── mutation_history (§3 inspect surface) ──────────────────────────────────

/// Mutations that touched `entity_id` within `group_id`, newest-first. Includes
/// already-undone mutations (the FULL history of an entity, so the consumer can
/// SEE that a merge was already reversed — `undone = true`).
///
/// Matching is over the derived `affected_entities`: for `entity_merge` the
/// entity qualifies whether it was the loser OR the keeper (both appear in
/// `inputs`, §2.3). Deterministic, read-only — reuses [`list_mutations`] with a
/// group scope + `include_undone`, then filters on the derived id set.
#[doc(hidden)]
pub async fn mutation_history(
    graph: &TemporalGraph,
    entity_id: &str,
    group_id: &str,
) -> Result<Vec<MutationRecord>> {
    let all = list_mutations(
        graph,
        MutationFilter {
            group_id: Some(group_id.to_string()),
            kind: None,
            since: None,
            include_undone: true,
        },
    )
    .await?;
    let hits: Vec<MutationRecord> = all
        .into_iter()
        .filter(|r| r.affected_entities.iter().any(|e| e == entity_id))
        .collect();
    tracing::debug!(
        target: "kremory.graph.provenance",
        entity_id = %entity_id,
        group_id = %group_id,
        returned = hits.len(),
        "kremory.graph.mutation_history"
    );
    counter!("kremory.graph.inspect_query_total", "op" => "mutation_history").increment(1);
    Ok(hits)
}
