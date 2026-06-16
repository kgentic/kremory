//! DB helpers and LLM prompt builder for Dream Pass 4 (ADR-047).
//!
//! Contains all database reads/writes used by `consistency_check`:
//! - `load_type_registry` — entity_types table
//! - `load_candidates` — candidate entity rows for embed-prefilter
//! - `load_top3_facts` — top 3 facts per entity
//! - `load_source_episode` — most recent episode mentioning entity
//! - `apply_correction` — UPDATE entities on LLM-confirmed wrong type
//! - `write_audit_row` — INSERT dream_pass4_audit row
//! - `build_verify_messages` — LLM system+user messages for verify batch
//!
//! Extracted from the `consistency_check` monolith as part of TD-C split
//! (ADR-050 §5).

use crate::core::error::{Error, Result};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatMessage};

use super::CandidateRow;

// ─── DB read helpers ──────────────────────────────────────────────────────────

/// Load entity type registry: id → (name, description).
///
/// Both name and description are needed:
/// - description for embed-prefilter cosine comparison
/// - name for the LLM verify-prompt type menu (so LLM knows which ID = which type)
pub(super) async fn load_type_registry(
    db: &libsql::Connection,
) -> Result<std::collections::HashMap<i64, (String, String)>> {
    let mut rows = db
        .query("SELECT id, name, description FROM entity_types", ())
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_type_registry: {e}")))?;
    let mut map = std::collections::HashMap::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_type_registry row: {e}")))?
    {
        let id: i64 = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_id: {e}")))?;
        let name: String = row
            .get(1)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_name: {e}")))?;
        let desc: String = row
            .get(2)
            .map_err(|e| Error::Other(anyhow::anyhow!("type_desc: {e}")))?;
        map.insert(id, (name, desc));
    }
    Ok(map)
}

pub(super) async fn load_candidates(db: &libsql::Connection) -> Result<Vec<CandidateRow>> {
    let mut rows = db
        .query(
            "SELECT rowid, id, entity_type_id FROM entities \
             WHERE entity_type_id != 0 \
             AND entity_type_source NOT IN ('ConsumerPinned', 'DreamPass4')",
            (),
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_candidates: {e}")))?;
    let mut candidates = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_candidates row: {e}")))?
    {
        let rowid: i64 = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("rowid: {e}")))?;
        let name: String = row
            .get(1)
            .map_err(|e| Error::Other(anyhow::anyhow!("entity.id: {e}")))?;
        let type_id: i64 = row
            .get(2)
            .map_err(|e| Error::Other(anyhow::anyhow!("entity_type_id: {e}")))?;
        let top3_facts = load_top3_facts(db, &name).await?;
        let source_episode = load_source_episode(db, &name).await?;
        candidates.push(CandidateRow {
            rowid,
            name,
            entity_type_id: type_id,
            top3_facts,
            source_episode,
        });
    }
    Ok(candidates)
}

pub(super) async fn load_top3_facts(
    db: &libsql::Connection,
    entity_id: &str,
) -> Result<Vec<String>> {
    let mut rows = db
        .query(
            "SELECT object_value FROM facts \
             WHERE subject_id = ?1 AND object_value IS NOT NULL \
             AND expired_at IS NULL ORDER BY recorded_at DESC LIMIT 3",
            libsql::params![entity_id.to_string()],
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_top3_facts: {e}")))?;
    let mut facts = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_top3_facts row: {e}")))?
    {
        let val: String = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("fact: {e}")))?;
        facts.push(val);
    }
    Ok(facts)
}

/// Load the source-episode text for the most recent episode that mentioned
/// this entity. Returns `None` if no episodic edge exists.
///
/// Phase D iter 4 (2026-06-10) addition per ADR-047 amendment. The verify call
/// needs source-text context to disambiguate polyseme entities ("Apple emailed
/// me" vs "Apple is a fruit"). Without it the LLM operates on name + thin facts
/// alone and produces ~50% precision on polysemes per RISK-001 iter 3 evidence.
pub(super) async fn load_source_episode(
    db: &libsql::Connection,
    entity_id: &str,
) -> Result<Option<String>> {
    let mut rows = db
        .query(
            "SELECT e.content FROM episodes e \
             INNER JOIN episodic_edges ee ON ee.episode_id = e.id \
             WHERE ee.entity_id = ?1 \
             ORDER BY e.recorded_at DESC LIMIT 1",
            libsql::params![entity_id.to_string()],
        )
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_source_episode: {e}")))?;
    if let Some(row) = rows
        .next()
        .await
        .map_err(|e| Error::Other(anyhow::anyhow!("load_source_episode row: {e}")))?
    {
        let content: String = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("episode.content: {e}")))?;
        Ok(Some(content))
    } else {
        Ok(None)
    }
}

// ─── DB write helpers ─────────────────────────────────────────────────────────

pub(super) async fn apply_correction(
    db: &libsql::Connection,
    candidate: &CandidateRow,
    new_type_id: i64,
    now: &str,
) -> Result<()> {
    db.execute(
        "UPDATE entities SET entity_type_id = ?1, entity_type_source = 'DreamPass4', \
         entity_type_assigned_at = ?2, updated_at = ?2 WHERE rowid = ?3",
        libsql::params![new_type_id, now.to_string(), candidate.rowid],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "apply_correction rowid={}: {e}",
            candidate.rowid
        ))
    })?;
    Ok(())
}

/// Parameters for a single `dream_pass4_audit` row (IRREV-001).
/// Bundled to keep `write_audit_row` under the clippy 5-arg limit.
pub(super) struct AuditRowParams<'a> {
    pub(super) entity_rowid: i64,
    pub(super) pre_type_id: i64,
    pub(super) post_type_id: i64,
    pub(super) verify_confidence: f32,
    pub(super) verify_model: &'a str,
    pub(super) run_id: &'a str,
}

pub(super) async fn write_audit_row(
    db: &libsql::Connection,
    p: AuditRowParams<'_>,
) -> Result<()> {
    db.execute(
        "INSERT INTO dream_pass4_audit \
         (entity_id, pre_type_id, post_type_id, verify_confidence, verify_model, run_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        libsql::params![
            p.entity_rowid,
            p.pre_type_id,
            p.post_type_id,
            p.verify_confidence as f64,
            p.verify_model.to_string(),
            p.run_id.to_string()
        ],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "write_audit_row entity_id={}: {e}",
            p.entity_rowid
        ))
    })?;
    Ok(())
}

// ─── LLM prompt builder ───────────────────────────────────────────────────────

/// Build the LLM messages for a verify batch (ADR-047 §3).
///
/// The user message MUST include the full type registry (id → name) so the LLM
/// can assign valid `new_type_id` values when action=correct.  Without this menu
/// the LLM has no grounding for integer IDs and will produce corrections that
/// don't match any row in `entity_types`, yielding zero precision lift even when
/// the action decisions are semantically correct.
pub(super) fn build_verify_messages(
    candidates: &[&CandidateRow],
    type_map: &std::collections::HashMap<i64, (String, String)>,
) -> Vec<ChatMessage> {
    let system = "You are an entity-type verification assistant. \
        For EACH entity provided below you MUST output exactly one decision. \
        The `decisions` array length MUST equal the number of entities listed. \
        \
        Each decision MUST use these EXACT JSON field names (do NOT use synonyms): \
        - `entity_id` (integer) — copy verbatim from the entity provided \
        - `action` (string) — MUST be one of EXACTLY \"confirm\", \"correct\", or \"uncertain\" \
        - `new_type_id` (integer, REQUIRED when action=correct) — id from the registry below \
        - `confidence` (number 0.0-1.0) — your confidence in this decision \
        \
        Do NOT use the field name `decision` (use `action`). Do NOT use the field name \
        `reason` (use `confidence` as a number). Do NOT include extra fields like \
        `current_type_id`, `name`, or `reason` — they will cause schema rejection. \
        \
        Action semantics: confirm (current type is correct), correct (current type is \
        wrong — you MUST provide new_type_id from the registry), uncertain (insufficient \
        info to decide — prefer this over skipping). \
        \
        Respond with a JSON object matching the VerifyBatch schema exactly.";

    // Build the type registry menu so the LLM knows which ID corresponds to which type.
    let mut registry_lines: Vec<String> = type_map
        .iter()
        .filter(|(&id, _)| id != 0) // exclude catch-all
        .map(|(&id, (name, _desc))| format!("  id={id} name=\"{name}\""))
        .collect();
    registry_lines.sort(); // deterministic order
    let registry_block = registry_lines.join("\n");

    let entity_lines: Vec<String> = candidates
        .iter()
        .map(|c| {
            let (type_name, type_desc) = type_map
                .get(&c.entity_type_id)
                .map(|(n, d)| (n.as_str(), d.as_str()))
                .unwrap_or(("unknown", "unknown"));
            // Truncate source episode to avoid bloat (8000 chars ~= 2000 tokens
            // per entity; cap covers most natural-prose paragraphs).
            // Per sprint plan T1.4 — gbrain INJECTION_PATTERNS (basket #68):
            // sanitize before splice to prevent prompt-injection via user-supplied text.
            let source_excerpt: String = c
                .source_episode
                .as_deref()
                .map(|s| {
                    let truncated = if s.len() > 8000 {
                        format!("{}…[truncated]", &s[..8000])
                    } else {
                        s.to_string()
                    };
                    crate::core::extraction::injection_patterns::sanitize_for_verify_prompt(
                        &truncated,
                    )
                })
                .unwrap_or_else(|| "[no source episode available]".to_string());
            // Per ADR-047 amendment + arXiv:2605.29168 ontology-grounded post-extraction
            // correction precedent — source-episode text gives the LLM disambiguating
            // context for polysemes ("Apple emailed me" vs "Apple is a fruit").
            format!(
                "entity_id={} name=\"{}\" current_type_id={} current_type_name=\"{}\" \
                 type_description=\"{}\" facts=\"{}\" source_episode=\"\"\"{}\"\"\"",
                c.rowid,
                c.name,
                c.entity_type_id,
                type_name,
                type_desc,
                c.top3_facts.join("; "),
                source_excerpt
            )
        })
        .collect();

    vec![
        chat_msg_system(system),
        chat_msg_user(format!(
            "Available entity types:\n{registry_block}\n\nVerify these entity type assignments:\n{}",
            entity_lines.join("\n")
        )),
    ]
}
