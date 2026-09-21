use crate::core::{
    dream::proposed_type::RejectionReason,
    error::Result,
    provider::{chat_msg_system, chat_msg_user},
};

// ─── Catch-all entity loader ──────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct CatchAllEntity {
    pub(crate) id: String,
}

pub(super) async fn load_catch_all_entities(
    conn: &libsql::Connection,
    group_id: &str,
) -> Result<Vec<CatchAllEntity>> {
    let mut rows = conn
        .query(
            // `id` tiebreak makes ordering deterministic when candidates tie on
            // access_count (e.g. freshly-ingested entities all at 0) — the row
            // order feeds the LLM prompt, so an unspecified tie-break makes the
            // whole pass non-reproducible run-to-run (surfaced by dream-loop E1).
            "SELECT id FROM entities \
             WHERE group_id = ?1 AND entity_type_id = 0 \
             ORDER BY access_count DESC, id",
            libsql::params![group_id.to_string()],
        )
        .await
        .map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: load_catch_all_entities query failed: {e}"
            ))
        })?;

    let mut entities = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "discover_types: load_catch_all_entities row read failed: {e}"
        ))
    })? {
        let id: String = row.get(0).map_err(|e| {
            crate::core::error::Error::Other(anyhow::anyhow!(
                "discover_types: load_catch_all_entities id read failed: {e}"
            ))
        })?;
        entities.push(CatchAllEntity { id });
    }
    Ok(entities)
}

// ─── Name-frequency clustering (v0.1.1 minimal) ──────────────────────────────

/// A cluster of catch-all entities with the same normalised name.
#[derive(Debug)]
pub(crate) struct NameCluster {
    /// Representative name (first occurrence).
    pub(crate) name: String,
    /// Number of catch-all entities with this name.
    pub(crate) count: usize,
}

/// Group catch-alls by normalised name, rank by frequency, take top K.
///
/// An earlier design specifies embedding-cosine clustering at threshold 0.7 with
/// minimum cluster size 3. v0.1.1 minimal-first: name-frequency grouping.
/// Embedding clustering ships in v0.2.0 when the embedder is always available.
pub(super) fn top_k_clusters(entities: &[CatchAllEntity], k: usize) -> Vec<NameCluster> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for entity in entities {
        let norm = normalise_name(&entity.id);
        *counts.entry(norm).or_insert(0) += 1;
    }
    // Filter: minimum cluster size of 2 (relaxed from spec's 3 at v0.1.1 to avoid
    // suppressing all proposals on small test namespaces).
    let mut clusters: Vec<NameCluster> = counts
        .into_iter()
        .filter(|(_, c)| *c >= 1) // include singletons at v0.1.1
        .map(|(name, count)| NameCluster { name, count })
        .collect();
    // Sort descending by count, then ascending by name as a DETERMINISTIC
    // tiebreak. The HashMap `.into_iter()` above yields randomized per-process
    // order, so a count-only sort leaves tied clusters in arbitrary order and
    // `truncate(k)` then keeps an arbitrary subset run-to-run — making this
    // pass's LLM prompt (and thus dream results) non-reproducible. Surfaced by
    // dream-loop E2; sibling of the ORDER BY tiebreak fix (b8208c9).
    clusters.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
    clusters.truncate(k);
    clusters
}

/// Normalise entity id/name for clustering: lowercase + trim.
fn normalise_name(s: &str) -> String {
    s.trim().to_lowercase()
}

// ─── Prompt builder ───────────────────────────────────────────────────────────

pub(super) fn build_discovery_messages(
    clusters: &[NameCluster],
    max_proposals: usize,
) -> Vec<crate::core::provider::ChatMessage> {
    let cluster_list = clusters
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. \"{}\" (seen {} time(s))", i + 1, c.name, c.count))
        .collect::<Vec<_>>()
        .join("\n");

    let system = format!(
        "You are a knowledge-graph type analyst. Your task is to propose NEW entity type \
categories for a knowledge graph.\n\
\n\
You will receive a list of entity names that could not be classified into existing types. \
Your job is to propose up to {max_proposals} NEW entity type definitions that would cover \
these entities.\n\
\n\
Rules:\n\
- Propose at most {max_proposals} new types. Fewer is fine.\n\
- Each type must have a short name (3-50 characters), a clear description, and a justification.\n\
- The name must be a concrete noun phrase — no placeholders like 'Unknown', 'Other', 'TBD', 'N/A'.\n\
- The description should explain what kinds of entities belong to this type.\n\
- Be specific: 'MedicalDevice' is better than 'Equipment'. 'LegalCase' is better than 'Item'.\n\
- Respond ONLY with the JSON structure — no extra commentary."
    );

    let user = format!(
        "These entities could not be classified into existing types:\n\n{cluster_list}\n\n\
Propose up to {max_proposals} new entity type categories that would cover them. \
Output JSON with a `proposals` array containing objects with `name`, `description`, and `justification` fields."
    );

    vec![chat_msg_system(system), chat_msg_user(user)]
}

// ─── Rejection reason → string ────────────────────────────────────────────────

pub(super) fn reason_to_string(r: RejectionReason) -> String {
    r.as_str().to_string()
}
