//! Dream pass — instance acronym/nickname recall (ADR-063 spec §3, "Site #5").
//!
//! Periodic dream-phase pass that closes the acronym/nickname gap
//! `names_lexically_compatible` (`disambiguation/lexical.rs`) documents as an
//! inherent, unclosable-by-token-matching limit: pairs like `"IBM"` /
//! `"International Business Machines"` or `"Bob"` / `"Robert"` share zero
//! name tokens and, per R3, no embedding technique reliably discriminates
//! bare proper nouns either (cosine is not a usable identity signal for this
//! surface). Site #5 closes the gap with a deterministic, dictionary-free
//! structural pre-filter (initialism test OR graph co-occurrence) that
//! nominates candidate pairs for margin-triggered, batched LLM adjudication;
//! a deterministic write-gate (never the LLM alone) decides whether to merge.
//!
//! ## Pass shape (spec §3.0-§3.5)
//!
//! 1. Load all real entity ids for `group_id` (excludes nothing structurally
//!    — id=0 catch-all entities are still real entity rows here, unlike Site
//!    #3's `entity_types` id=0 sentinel).
//! 2. NAMED-HYBRID pre-filter (spec §3.1): a pair `(a, b)` is nominated if
//!    `initialism_candidate(a, b) OR cooccurs_in_graph(a, b)`. Both checks are
//!    deterministic, dictionary-free, embedder-independent.
//! 3. Nominated pairs are batched into ONE `IdentityVerdictBatch` LLM call
//!    per dream-pass invocation (shared schema, spec §2.1/§3.2). The boolean
//!    nomination IS the margin band — Site #5 has no continuous cosine score
//!    to band against (R3), so every nominated pair reaches the LLM.
//! 4. Each resolved pair is decided via the shared [`write_gate`] (spec §2.2)
//!    with `cosine = 0.0` (never a meaningful signal for this site) and
//!    `deterministic_signal = DeterministicSignal::from_structural_prefilter(true)`
//!    (spec §2.2.2 — the pre-filter's boolean nomination is Site #5's
//!    deterministic signal). Because `cosine` is always `0.0`, row 1 can
//!    never fire — every `Merge` for this site passes through an actual LLM
//!    verdict (row 5).
//! 5. `WriteDecision::Merge` reuses `canonicalization::apply_merge`'s exact
//!    destructive remap (spec §3.3) — the `identity_verdict_audit` row lands
//!    INSIDE the same `BEGIN IMMEDIATE` transaction (spec §5.1 RISK-003).
//!    `WriteDecision::PotentialAlias` reuses `insert_potential_alias_fact`
//!    (`disambiguation/mod.rs`), confidence sourced from the LLM verdict (no
//!    cosine exists for this site). `WriteDecision::Reject` writes nothing.
//!
//! ## Accepted residual gap (spec §3.1 ALT-001)
//!
//! Nickname pairs that (a) never co-occur anywhere in the graph AND (b) share
//! no structural (initial-letter) relationship — e.g. `Peggy`/`Margaret`,
//! `Jack`/`John` — are undetected BY DESIGN. Neither half of the hybrid
//! fires. This is an inherent limit, not a bug, and is not closeable by a
//! curated list (same honesty already applied to `names_lexically_compatible`'s
//! own documented gaps). A consumer who knows about such a pair in advance
//! uses the `.with_known_alias()` escape hatch (spec §3.4, D8) — DEFERRED in
//! this implementation pass; see the `// D8 escape hatch` marker below.
//!
//! ## Spike gating (spec §8)
//!
//! S1 (initialism precision/recall), S2 (LLM adjudication precision/recall),
//! and S6 (co-occurrence query cost) are NOT yet run. This pass therefore
//! ships behind `DreamOpts::include_acronym_nickname_recall`, DEFAULT
//! `false` (spec §8's hard constraint: no spike-gated number goes live
//! before its mapped spike shows PASS).
//!
//! ## Observability (spec §6)
//!
//! - `kremory.dream.acronym_recall.pairs_examined_total`
//! - `kremory.dream.acronym_recall.merges_applied_total`
//! - `kremory.dream.acronym_recall.rejected_total`
//! - `kremory.identity.candidate_nominated_total{site="site5_acronym_nickname"}`
//! - `kremory.identity.verdict_parse_fail_total{site="site5_acronym_nickname"}`
//! - `kremory.identity.llm_call_latency_ms_histogram{site="site5_acronym_nickname"}`
//! - `kremory.identity.write_gate_decision_total{site="site5_acronym_nickname",decision}`
//! - `kremory.identity.write_gate_llm_authorized_merge_total{site="site5_acronym_nickname"}`
//! - `KREMORY_DEBUG=1` raw-payload trace at target
//!   `kremory::dream::acronym_recall::raw_payload`

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use metrics::{counter, histogram};

use crate::core::error::{Error, Result};
use crate::core::extraction::structured::StructuredCallBuilder;
use crate::core::identity_verdict::{
    identity_verdict_batch_schema, write_gate, DeterministicSignal, IdentityVerdictBatch,
    IdentityVerdictItem, WriteDecision, WriteGateInputs,
};
use crate::core::provider::{chat_msg_system, chat_msg_user, ChatProvider};
use crate::core::schema::TemporalGraph;

/// Site label used on every shared `kremory.identity.*` counter (spec §6).
const SITE_LABEL: &str = "site5_acronym_nickname";

/// Stopwords that never contribute an initial letter in conventional acronym
/// formation (spec §3.1). Dictionary-free in the sense that this is a
/// closed, tiny structural-formation set — not a list of known acronyms or
/// nicknames.
const INITIALISM_STOPWORDS: &[&str] = &["of", "and", "the", "for"];

// ─── Report ────────────────────────────────────────────────────────────────

/// Summary returned by [`acronym_nickname_recall`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AcronymNicknameRecallReport {
    pub group_id: String,
    /// Number of entity pairs considered by the structural pre-filter,
    /// whether nominated or not (spec §6 — denominator for nomination-rate
    /// monitoring).
    pub pairs_examined: usize,
    /// Number of pairs nominated for LLM adjudication (initialism test OR
    /// co-occurrence fired).
    pub candidates_nominated: usize,
    /// Number of merges applied (loser entity remapped + removed).
    pub merges_applied: usize,
    /// Number of pairs that landed as a non-destructive `potential_alias`.
    pub potential_aliases: usize,
    /// Number of nominated pairs rejected (LLM said not-same, or verdict
    /// missing/parse-failed).
    pub rejected: usize,
}

// ─── Internal types ───────────────────────────────────────────────────────

/// A candidate pair nominated for the LLM-verify band, correlated back to
/// its `write_gate` inputs via index into the caller's `nominated` vec
/// (mirrors `IdentityVerdictItem::pair_id`, spec §2.1, and Site #3's
/// `NominatedPair`).
struct NominatedPair {
    a: String,
    b: String,
}

// ─── Public entry-point ───────────────────────────────────────────────────

/// Bundled parameters for [`acronym_nickname_recall`] — args-as-object per
/// TD-042 (rust-conventions §too_many_arguments, threshold 3). `llm` stays a
/// lead generic positional param (project convention, mirrors
/// `type_registry_collapse<L>` / `discover_types<L>`).
pub(crate) struct AcronymNicknameRecallParams<'a> {
    /// Full graph handle (not a bare connection) — needed because this pass
    /// reuses `canonicalization::apply_merge_with_audit` and
    /// `disambiguation::insert_potential_alias_fact`, both of which are
    /// `TemporalGraph`-typed methods (spec §3.3).
    pub(crate) graph: &'a TemporalGraph,
    pub(crate) group_id: &'a str,
    /// Concrete model id for capability detection (TD-094-style threading).
    /// Empty (`""`) → `PromptOnly` degrade.
    pub(crate) model_id: &'a str,
}

/// Run Site #5 instance acronym/nickname recall over `group_id` (ADR-063
/// spec §3).
///
/// Called by `mem.dream()` when `DreamOpts::include_acronym_nickname_recall
/// = true` (spike-gated, default `false` — spec §8). Hooked immediately
/// AFTER `resolve_pending_aliases` (L7) and BEFORE the reclassify pass (spec
/// §3.0).
pub(crate) async fn acronym_nickname_recall<L: ChatProvider>(
    llm: &L,
    params: AcronymNicknameRecallParams<'_>,
) -> Result<AcronymNicknameRecallReport> {
    let AcronymNicknameRecallParams {
        graph,
        group_id,
        model_id,
    } = params;
    let conn = &graph.conn;

    let mut report = AcronymNicknameRecallReport {
        group_id: group_id.to_string(),
        ..Default::default()
    };

    // ── Step 1: load entity ids for the group (spec §3.0) ────────────────────
    let ids = load_entity_ids(conn, group_id).await?;
    if ids.len() < 2 {
        return Ok(report);
    }

    // ── Step 2: structural pre-filter — named hybrid (spec §3.1) ─────────────
    let mut nominated: Vec<NominatedPair> = Vec::new();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            report.pairs_examined += 1;
            let a = &ids[i];
            let b = &ids[j];
            let is_nominated = initialism_candidate(a, b)
                || cooccurs_in_graph(CooccursInGraphParams {
                    conn,
                    group_id,
                    a,
                    b,
                })
                .await?;
            if is_nominated {
                nominated.push(NominatedPair {
                    a: a.clone(),
                    b: b.clone(),
                });
            }
        }
    }

    counter!("kremory.dream.acronym_recall.pairs_examined_total")
        .increment(report.pairs_examined as u64);
    report.candidates_nominated = nominated.len();
    for _ in &nominated {
        counter!(
            "kremory.identity.candidate_nominated_total",
            "site" => SITE_LABEL
        )
        .increment(1);
    }

    if nominated.is_empty() {
        return Ok(report);
    }

    // ── Step 3: margin-triggered (degenerate single-band) LLM adjudication ───
    // (spec §3.2) — the boolean nomination IS the band; every nominated pair
    // reaches the LLM.
    let verdicts_by_pair_id = adjudicate_batch(AdjudicateBatchParams {
        llm,
        model_id,
        conn,
        nominated: &nominated,
        group_id,
    })
    .await?;

    // ── Step 4/5: write_gate decision per pair + atomic writes (spec §3.3) ───
    let run_id = uuid::Uuid::new_v4().to_string();

    for (pair_id, pair) in nominated.iter().enumerate() {
        let verdict = verdicts_by_pair_id.get(&pair_id).cloned();
        let decision = write_gate(WriteGateInputs {
            // R3: no embedding technique discriminates bare proper nouns —
            // cosine is never a meaningful identity signal for this site.
            cosine: 0.0,
            // Site #5 always passes cosine = 0.0, so any positive threshold
            // keeps row 1 from ever firing (spec §3.3) — every Merge for
            // this site requires an actual LLM verdict (row 5).
            merge_threshold: 1.0,
            // spec §2.2.2: the structural pre-filter's boolean nomination
            // result IS the deterministic signal for this site.
            deterministic_signal: DeterministicSignal::from_structural_prefilter(true),
            llm_verdict: verdict.clone(),
            min_confidence_floor: None,
        });
        record_write_gate_decision(decision);

        match decision {
            WriteDecision::Merge => {
                // spec §3.3: reuse `canonicalization::apply_merge`'s exact
                // destructive remap. Keeper = whichever id `apply_merge`
                // treats as `keeper_id` — Site #5 has no cosine/description-
                // length signal to prefer one side, so the FIRST-seen id
                // (`pair.a`) is kept and `pair.b` is remapped onto it,
                // deterministically (upper-triangle enumeration order).
                let Some(v) = verdict.as_ref() else {
                    // write_gate cannot reach row 5 without a verdict; this
                    // branch is structurally unreachable but handled
                    // defensively (parse-loudly discipline — never silently
                    // merge on a missing verdict).
                    report.rejected += 1;
                    continue;
                };
                crate::core::canonicalization::apply_merge_with_audit(
                    graph,
                    crate::core::canonicalization::ApplyMergeWithAuditParams {
                        loser_id: &pair.b,
                        keeper_id: &pair.a,
                        audit: Some(crate::core::canonicalization::IdentityVerdictAuditRow {
                            site: SITE_LABEL,
                            group_id,
                            candidate_a: &pair.a,
                            candidate_b: &pair.b,
                            cosine: None,
                            structural_signal: true,
                            verdict: Some(v),
                            decision: "merge",
                            run_id: &run_id,
                        }),
                    },
                )
                .await?;
                report.merges_applied += 1;
            }
            WriteDecision::PotentialAlias => {
                // spec §3.3: reuse `insert_potential_alias_fact` — confidence
                // sourced from the LLM verdict (no cosine exists for this
                // site). An acronym pair landing here is re-evaluated by the
                // ordinary L7 `resolve_pending_aliases` flow on a later
                // dream cycle, exactly like any other potential-alias edge.
                let confidence = verdict.as_ref().map(|v| v.confidence).unwrap_or(0.0);
                let inserted = crate::core::disambiguation::insert_potential_alias_fact(
                    crate::core::disambiguation::InsertPotentialAliasFactParams {
                        graph,
                        new_entity_id: &pair.b,
                        existing_id: &pair.a,
                        similarity: confidence,
                        provenance: crate::core::disambiguation::AliasProvenance {
                            source_episode_id: None,
                            group_id: Some(group_id),
                        },
                    },
                )
                .await;
                if let Err(e) = inserted {
                    tracing::warn!(
                        target: "kremory::dream::acronym_recall",
                        error = %e,
                        group_id = %group_id,
                        "acronym_nickname_recall: potential_alias fact insert failed"
                    );
                }
                write_audit_row(WriteAuditRowParams {
                    conn,
                    group_id,
                    candidate_a: &pair.a,
                    candidate_b: &pair.b,
                    structural_signal: true,
                    verdict: verdict.as_ref(),
                    decision: "potential_alias",
                    run_id: &run_id,
                })
                .await?;
                report.potential_aliases += 1;
            }
            WriteDecision::Reject => {
                // No write — increment rejected counter (spec §3.3).
                report.rejected += 1;
            }
        }
    }

    counter!("kremory.dream.acronym_recall.merges_applied_total")
        .increment(report.merges_applied as u64);
    counter!("kremory.dream.acronym_recall.rejected_total").increment(report.rejected as u64);

    Ok(report)
}

fn record_write_gate_decision(decision: WriteDecision) {
    let label = match decision {
        WriteDecision::Merge => "merge",
        WriteDecision::PotentialAlias => "potential_alias",
        WriteDecision::Reject => "reject",
    };
    counter!(
        "kremory.identity.write_gate_decision_total",
        "site" => SITE_LABEL,
        "decision" => label
    )
    .increment(1);
    if decision == WriteDecision::Merge {
        counter!(
            "kremory.identity.write_gate_llm_authorized_merge_total",
            "site" => SITE_LABEL
        )
        .increment(1);
    }
}

// ─── Structural pre-filter (spec §3.1) ────────────────────────────────────

/// Deterministic string algorithm: every character of the shorter name
/// (case-insensitive) matches, in order, the first letter of a token in the
/// longer name, skipping [`INITIALISM_STOPWORDS`]. Symmetric — checks both
/// `(a as acronym of b)` and `(b as acronym of a)`. Pure function, no
/// DB/LLM access, `O(len(shorter) × tokens(longer))` (spec §3.1).
///
/// Dictionary-free: no list of known acronyms is consulted — only the
/// structural relationship between the two strings.
pub(crate) fn initialism_candidate(a: &str, b: &str) -> bool {
    let (shorter, longer) = if a.chars().count() <= b.chars().count() {
        (a, b)
    } else {
        (b, a)
    };
    is_initialism_of(shorter, longer)
}

/// One-directional check: is `shorter` an initialism of `longer`?
fn is_initialism_of(shorter: &str, longer: &str) -> bool {
    let shorter_chars: Vec<char> = shorter
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();
    if shorter_chars.is_empty() {
        return false;
    }
    let longer_tokens: Vec<String> = longer
        .split_whitespace()
        .filter(|t| !INITIALISM_STOPWORDS.contains(&t.to_lowercase().as_str()))
        .map(|t| t.to_lowercase())
        .collect();
    if longer_tokens.len() < shorter_chars.len() {
        // Not enough tokens in the longer name to match every initial letter.
        return false;
    }
    // Every character of `shorter`, in order, must match the first letter of
    // a token in `longer` — using the first `shorter_chars.len()` significant
    // tokens (standard acronym-formation order).
    shorter_chars
        .iter()
        .zip(longer_tokens.iter())
        .all(|(ch, token)| token.starts_with(*ch))
}

/// Bundled parameters for [`cooccurs_in_graph`] — args-as-object per TD-042
/// (rust-conventions §too_many_arguments, threshold 3).
struct CooccursInGraphParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    a: &'a str,
    b: &'a str,
}

/// Deterministic graph-structure query: TRUE if `a` and `b` share at least
/// one `episodic_edges.episode_id` (same episode mention) OR share a
/// graph-neighbor (a fact where one is subject/object and the other appears
/// as subject/object of a fact involving a common third entity within 1
/// hop). Bounded SQL scoped to `group_id`, not an in-memory full-graph
/// traversal (spec §3.1). Requires `idx_facts_subject` (migration 018) to
/// stay bounded-cost (spec §3.5 RISK-002).
async fn cooccurs_in_graph(params: CooccursInGraphParams<'_>) -> Result<bool> {
    let CooccursInGraphParams {
        conn,
        group_id,
        a,
        b,
    } = params;
    // Shared episode mention.
    let mut rows = conn
        .query(
            "SELECT 1 FROM episodic_edges ea \
             JOIN episodic_edges eb ON ea.episode_id = eb.episode_id \
             WHERE ea.entity_id = ?1 AND eb.entity_id = ?2 \
             AND ea.entity_group_id = ?3 AND eb.entity_group_id = ?3 \
             LIMIT 1",
            libsql::params![a.to_string(), b.to_string(), group_id.to_string()],
        )
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: cooccurs_in_graph episode query failed: {e}"
            ))
        })?;
    if rows
        .next()
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: cooccurs_in_graph episode row read failed: {e}"
            ))
        })?
        .is_some()
    {
        return Ok(true);
    }

    // 1-hop graph neighbor: a fact linking `a` to some entity `x`, and a
    // fact linking `b` to that SAME `x` (either direction, subject or
    // object), scoped to group_id and non-expired facts. Uses
    // idx_facts_subject (migration 018) + idx_facts_object to stay bounded.
    let mut rows = conn
        .query(
            "SELECT 1 FROM ( \
                 SELECT object_id AS neighbor FROM facts \
                 WHERE subject_id = ?1 AND expired_at IS NULL AND group_id = ?3 AND object_id IS NOT NULL \
                 UNION \
                 SELECT subject_id AS neighbor FROM facts \
                 WHERE object_id = ?1 AND expired_at IS NULL AND group_id = ?3 \
             ) AS neighbors_a \
             JOIN ( \
                 SELECT object_id AS neighbor FROM facts \
                 WHERE subject_id = ?2 AND expired_at IS NULL AND group_id = ?3 AND object_id IS NOT NULL \
                 UNION \
                 SELECT subject_id AS neighbor FROM facts \
                 WHERE object_id = ?2 AND expired_at IS NULL AND group_id = ?3 \
             ) AS neighbors_b \
             ON neighbors_a.neighbor = neighbors_b.neighbor \
             LIMIT 1",
            libsql::params![a.to_string(), b.to_string(), group_id.to_string()],
        )
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: cooccurs_in_graph neighbor query failed: {e}"
            ))
        })?;
    let found = rows
        .next()
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: cooccurs_in_graph neighbor row read failed: {e}"
            ))
        })?
        .is_some();
    Ok(found)
}

// D8 escape hatch (spec §3.4, deferred — see spec §7 fork "D8's exact
// consumer-facing API shape"): a future `.with_known_alias(a, b)`-style
// consumer-facing knob would hook HERE, treating a registered pair as a
// structural pre-filter nomination with `DeterministicSignal::
// from_structural_prefilter(true)` unconditionally (bypassing the
// initialism/co-occurrence test), still flowing through the SAME margin-
// triggered LLM adjudication + write_gate — never a direct Merge bypass.
// NOT implemented in this pass: no public API/napi surface added.

// ─── LLM adjudication (spec §3.2) ─────────────────────────────────────────

struct AdjudicateBatchParams<'a, L: ChatProvider> {
    llm: &'a L,
    model_id: &'a str,
    conn: &'a libsql::Connection,
    nominated: &'a [NominatedPair],
    group_id: &'a str,
}

/// Run ONE batched `IdentityVerdictBatch` LLM call adjudicating every
/// nominated pair (spec §3.2), returning verdicts keyed by `pair_id` (index
/// into `nominated`). Missing/parse-failed entries are simply absent from
/// the map — callers treat a missing `pair_id` as `llm_verdict = None`
/// (spec §2.3 failure-mode default).
async fn adjudicate_batch<L: ChatProvider>(
    params: AdjudicateBatchParams<'_, L>,
) -> Result<HashMap<usize, IdentityVerdictItem>> {
    let AdjudicateBatchParams {
        llm,
        model_id,
        conn,
        nominated,
        group_id,
    } = params;

    let messages = build_adjudication_messages(conn, nominated).await?;
    let schema = identity_verdict_batch_schema(nominated.len());

    let call_start = Instant::now();
    let raw_value = StructuredCallBuilder::new(llm, &schema, "IdentityVerdictBatch")
        .model(model_id)
        .messages(messages)
        .call()
        .await;
    let elapsed_ms = call_start.elapsed().as_millis() as f64;
    histogram!(
        "kremory.identity.llm_call_latency_ms_histogram",
        "site" => SITE_LABEL
    )
    .record(elapsed_ms);

    if std::env::var("KREMORY_DEBUG").is_ok() {
        tracing::debug!(
            target: "kremory::dream::acronym_recall::raw_payload",
            model_id = %model_id,
            group_id = %group_id,
            response = ?raw_value,
            "acronym_nickname_recall adjudication raw response"
        );
    }

    let raw_value = match raw_value {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "kremory::dream::acronym_recall",
                error = %e,
                group_id = %group_id,
                "acronym_nickname_recall: adjudication LLM call failed — all nominated pairs default to no-verdict"
            );
            return Ok(HashMap::new());
        }
    };

    // Batch envelope parse — deliberately loose (Vec<serde_json::Value>), then
    // per-item parse below (spec §2.1's "one malformed element ≠ whole batch loss").
    let batch: IdentityVerdictBatch = match serde_json::from_value(raw_value.clone()) {
        Ok(b) => b,
        Err(_) => {
            let repaired = if raw_value.is_array() {
                serde_json::json!({ "verdicts": raw_value })
            } else {
                raw_value.clone()
            };
            match serde_json::from_value::<IdentityVerdictBatch>(repaired) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(
                        target: "kremory::dream::acronym_recall",
                        error = %e,
                        group_id = %group_id,
                        "acronym_nickname_recall: failed to parse IdentityVerdictBatch — all nominated pairs default to no-verdict"
                    );
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(nominated.len() as u64);
                    return Ok(HashMap::new());
                }
            }
        }
    };

    let mut verdicts_by_pair_id: HashMap<usize, IdentityVerdictItem> = HashMap::new();
    for raw_item in &batch.verdicts {
        match serde_json::from_value::<IdentityVerdictItem>(raw_item.clone()) {
            Ok(item) => {
                if item.pair_id >= nominated.len() {
                    // Out-of-range pair_id — echoed id doesn't correlate to any
                    // nominated pair; drop it loudly.
                    counter!(
                        "kremory.identity.verdict_parse_fail_total",
                        "site" => SITE_LABEL
                    )
                    .increment(1);
                    tracing::warn!(
                        target: "kremory::dream::acronym_recall",
                        pair_id = item.pair_id,
                        nominated_len = nominated.len(),
                        "acronym_nickname_recall: verdict pair_id out of range — dropped"
                    );
                    continue;
                }
                verdicts_by_pair_id.entry(item.pair_id).or_insert(item);
            }
            Err(e) => {
                counter!(
                    "kremory.identity.verdict_parse_fail_total",
                    "site" => SITE_LABEL
                )
                .increment(1);
                tracing::warn!(
                    target: "kremory::dream::acronym_recall",
                    error = %e,
                    "acronym_nickname_recall: skipping malformed verdict item"
                );
            }
        }
    }

    Ok(verdicts_by_pair_id)
}

/// Build the LLM adjudication prompt for a batch of nominated entity pairs
/// (spec §3.2): both surface-form names, each entity's stored
/// `properties.description` (if present), and up to N=3 exemplar facts per
/// entity (reusing the `load_top3_facts`-style helper pattern already used
/// by `consistency_check/audit.rs`).
async fn build_adjudication_messages(
    conn: &libsql::Connection,
    nominated: &[NominatedPair],
) -> Result<Vec<crate::core::provider::ChatMessage>> {
    let system = "You are a knowledge-graph entity-identity analyst. You will be shown \
pairs of entity names (with any known description and recent facts) that a structural \
signal has flagged as POSSIBLY the same real-world entity — e.g. an initialism/acronym \
relationship (\"IBM\" / \"International Business Machines\") or graph co-occurrence \
(entities that appear together in the same episode or share a graph neighbor, such as \
\"Bob\" / \"Robert\"). For each pair, decide whether the two names refer to the SAME \
real-world entity. Respond ONLY with the JSON structure — no extra commentary."
        .to_string();

    let mut pair_lines = Vec::with_capacity(nominated.len());
    for (pair_id, pair) in nominated.iter().enumerate() {
        let desc_a = load_entity_description(conn, &pair.a).await?;
        let desc_b = load_entity_description(conn, &pair.b).await?;
        let facts_a = load_top3_facts(conn, &pair.a).await?;
        let facts_b = load_top3_facts(conn, &pair.b).await?;
        pair_lines.push(format!(
            "Pair {pair_id}:\n  A: name=\"{}\" description=\"{}\" facts={:?}\n  B: name=\"{}\" description=\"{}\" facts={:?}",
            pair.a, desc_a.unwrap_or_default(), facts_a,
            pair.b, desc_b.unwrap_or_default(), facts_b,
        ));
    }
    let pairs_list = pair_lines.join("\n\n");

    let user = format!(
        "Adjudicate the following entity-identity pairs. For each pair, output a \
verdict object with `pair_id` (matching the pair number below), `is_same_entity` \
(true if the two names refer to the same real-world entity), `confidence` \
(0.0-1.0), and `reasoning` (a short free-text justification).\n\n{pairs_list}"
    );

    Ok(vec![chat_msg_system(system), chat_msg_user(user)])
}

// ─── DB helpers ────────────────────────────────────────────────────────────

/// Load all entity ids for `group_id` (spec §3.0 — "raw entity population",
/// not scoped to id=0 catch-all only; every real entity is a candidate).
async fn load_entity_ids(conn: &libsql::Connection, group_id: &str) -> Result<Vec<String>> {
    let mut rows = conn
        .query(
            "SELECT id FROM entities WHERE group_id = ?1 ORDER BY id ASC",
            libsql::params![group_id],
        )
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: load entity ids failed for group_id={group_id}: {e}"
            ))
        })?;
    let mut ids = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "acronym_nickname_recall: entity id row read failed: {e}"
        ))
    })? {
        let id: String = row
            .get(0)
            .map_err(|e| Error::Other(anyhow::anyhow!("acronym_nickname_recall: id read: {e}")))?;
        if seen.insert(id.clone()) {
            ids.push(id);
        }
    }
    Ok(ids)
}

/// Load an entity's `properties.description` field, if present.
async fn load_entity_description(
    conn: &libsql::Connection,
    entity_id: &str,
) -> Result<Option<String>> {
    let mut rows = conn
        .query(
            "SELECT properties FROM entities WHERE id = ?1",
            libsql::params![entity_id.to_string()],
        )
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: load_entity_description query failed: {e}"
            ))
        })?;
    let Some(row) = rows.next().await.map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "acronym_nickname_recall: load_entity_description row read failed: {e}"
        ))
    })?
    else {
        return Ok(None);
    };
    let props_text: Option<String> = row.get(0).map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "acronym_nickname_recall: properties col read failed: {e}"
        ))
    })?;
    Ok(props_text
        .as_deref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| {
            v.get("description")
                .and_then(|d| d.as_str())
                .map(str::to_string)
        }))
}

/// Load up to 3 most-recent non-expired facts whose `object_value` is
/// present for `entity_id` (mirrors `consistency_check/audit.rs::
/// load_top3_facts` exactly).
async fn load_top3_facts(conn: &libsql::Connection, entity_id: &str) -> Result<Vec<String>> {
    let mut rows = conn
        .query(
            "SELECT object_value FROM facts \
             WHERE subject_id = ?1 AND object_value IS NOT NULL \
             AND expired_at IS NULL ORDER BY recorded_at DESC LIMIT 3",
            libsql::params![entity_id.to_string()],
        )
        .await
        .map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: load_top3_facts query failed: {e}"
            ))
        })?;
    let mut facts = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "acronym_nickname_recall: load_top3_facts row read failed: {e}"
        ))
    })? {
        let val: String = row.get(0).map_err(|e| {
            Error::Other(anyhow::anyhow!(
                "acronym_nickname_recall: fact value read failed: {e}"
            ))
        })?;
        facts.push(val);
    }
    Ok(facts)
}

struct WriteAuditRowParams<'a> {
    conn: &'a libsql::Connection,
    group_id: &'a str,
    candidate_a: &'a str,
    candidate_b: &'a str,
    structural_signal: bool,
    verdict: Option<&'a IdentityVerdictItem>,
    decision: &'a str,
    run_id: &'a str,
}

/// Insert one `identity_verdict_audit` row (spec §5.1). Used ONLY for
/// `PotentialAlias` decisions (audit-only, no destructive write) OUTSIDE any
/// merge transaction. Merge audit rows are written INSIDE `apply_merge_with_
/// audit`'s `BEGIN IMMEDIATE` (spec §5.1 RISK-003), never here. `cosine` is
/// always `NULL` for Site #5 (spec §5.1 — "NULL for Site #5, no meaningful
/// cosine").
async fn write_audit_row(params: WriteAuditRowParams<'_>) -> Result<()> {
    let WriteAuditRowParams {
        conn,
        group_id,
        candidate_a,
        candidate_b,
        structural_signal,
        verdict,
        decision,
        run_id,
    } = params;

    conn.execute(
        "INSERT INTO identity_verdict_audit \
         (site, group_id, candidate_a, candidate_b, cosine, structural_signal, \
          llm_is_same, llm_confidence, llm_reasoning, decision, run_id) \
         VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
        libsql::params![
            SITE_LABEL,
            group_id,
            candidate_a,
            candidate_b,
            structural_signal,
            verdict.map(|v| v.is_same_entity),
            verdict.map(|v| f64::from(v.confidence)),
            verdict.map(|v| v.reasoning.clone()),
            decision,
            run_id,
        ],
    )
    .await
    .map_err(|e| {
        Error::Other(anyhow::anyhow!(
            "acronym_nickname_recall: identity_verdict_audit insert failed: {e}"
        ))
    })?;
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────
//
// Deterministic — a scripted `ChatProvider` mirrors
// `type_registry_collapse.rs::ScriptedVerdictProvider` (no live LLM, runs in
// the default `cargo test` gate).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{
        InsertEntityWithGroupParams, InsertEpisodeParams, InsertEpisodicEdgeParams,
    };
    use crate::core::provider::{
        ChatMessage, ChatResponse, LLMError, MockChatResponse, StructuredOutputFormat, Tool,
    };
    use crate::core::schema::TemporalGraph;

    /// Scripted `ChatProvider` returning one fixed `IdentityVerdictBatch` JSON
    /// response, ignoring the prompt entirely (mirrors
    /// `type_registry_collapse.rs::ScriptedVerdictProvider`).
    #[derive(Debug)]
    struct ScriptedVerdictProvider {
        json: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for ScriptedVerdictProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> std::result::Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: self.json.clone(),
            }))
        }
    }

    #[allow(clippy::too_many_arguments)] // test helper — CLAUDE.md rule 5 test-exemption
    async fn insert_entity(graph: &TemporalGraph, id: &str, group_id: &str, description: &str) {
        let props = serde_json::json!({ "name": id, "description": description });
        graph
            .insert_entity_with_group(InsertEntityWithGroupParams {
                id,
                entity_type_id: 0u32,
                properties: props,
                group_id: Some(group_id),
            })
            .await
            .expect("insert entity");
    }

    async fn count_entities(conn: &libsql::Connection, group_id: &str) -> i64 {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entities WHERE group_id = ?1",
                libsql::params![group_id],
            )
            .await
            .expect("count query");
        rows.next()
            .await
            .expect("row")
            .expect("row present")
            .get::<i64>(0)
            .expect("count col")
    }

    // ── initialism_candidate pure-fn unit tests (spec §3.1) ──────────────────

    #[test]
    fn initialism_ibm_matches_full_name() {
        assert!(initialism_candidate(
            "IBM",
            "International Business Machines"
        ));
    }

    #[test]
    fn initialism_skips_stopwords() {
        // "UN" as an initialism of "United Nations" (no stopword needed here);
        // exercise the stopword-skip path with a name containing "of"/"the".
        assert!(initialism_candidate("USA", "United States of America"));
    }

    #[test]
    fn initialism_bob_robert_is_false() {
        // No structural (initial-letter) relationship — nickname, not acronym.
        assert!(!initialism_candidate("Bob", "Robert"));
    }

    #[test]
    fn initialism_is_symmetric() {
        assert_eq!(
            initialism_candidate("IBM", "International Business Machines"),
            initialism_candidate("International Business Machines", "IBM")
        );
    }

    // ── cooccurs_in_graph tests (spec §3.1) ───────────────────────────────────

    #[tokio::test]
    async fn cooccurs_true_when_entities_share_episode() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        insert_entity(&graph, "alpha", "g1", "").await;
        insert_entity(&graph, "beta", "g1", "").await;
        let ep = graph
            .insert_episode(InsertEpisodeParams {
                content: "Alpha and Beta appear together.",
                timestamp: chrono::Utc::now(),
                source_type: Some("transcript"),
                metadata: None,
            })
            .await
            .expect("episode");
        for ent in ["alpha", "beta"] {
            graph
                .insert_episodic_edge(InsertEpisodicEdgeParams {
                    episode_id: ep,
                    entity_id: ent,
                    entity_group_id: Some("g1"),
                    role: "mention",
                })
                .await
                .expect("edge");
        }
        assert!(cooccurs_in_graph(CooccursInGraphParams {
            conn: &conn,
            group_id: "g1",
            a: "alpha",
            b: "beta",
        })
        .await
        .expect("query"));
    }

    #[tokio::test]
    async fn cooccurs_false_when_unrelated() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        insert_entity(&graph, "gamma", "g2", "").await;
        insert_entity(&graph, "delta", "g2", "").await;
        assert!(!cooccurs_in_graph(CooccursInGraphParams {
            conn: &conn,
            group_id: "g2",
            a: "gamma",
            b: "delta",
        })
        .await
        .expect("query"));
    }

    // ── Full pass tests (write_gate ROW 5 / PotentialAlias / Reject) ─────────

    /// A nominated acronym pair + scripted LLM `is_same_entity=true,
    /// confidence=0.95` → merges_applied=1 (write_gate ROW 5 — the
    /// deterministic structural-prefilter signal fired AND the LLM agrees
    /// with sufficient confidence). Entity remapped, ONE merge audit row
    /// (site5, real verdict, cosine NULL).
    #[tokio::test]
    async fn nominated_acronym_pair_llm_true_merges_row5() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        insert_entity(&graph, "IBM", "g5", "A technology company.").await;
        insert_entity(
            &graph,
            "International Business Machines",
            "g5",
            "A technology company headquartered in New York.",
        )
        .await;

        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.95,"reasoning":"same company, acronym"}]}"#
                .to_string(),
        };

        let report = acronym_nickname_recall(
            &llm,
            AcronymNicknameRecallParams {
                graph: &graph,
                group_id: "g5",
                model_id: "test-model",
            },
        )
        .await
        .expect("acronym_nickname_recall must succeed");

        assert_eq!(
            report.candidates_nominated, 1,
            "IBM / International Business Machines is an initialism-nominated pair"
        );
        assert_eq!(
            report.merges_applied, 1,
            "nominated pair + LLM-true high-confidence must merge (write_gate row 5)"
        );
        assert_eq!(
            count_entities(&conn, "g5").await,
            1,
            "one entity survives after the merge"
        );

        let mut rows = conn
            .query(
                "SELECT cosine, structural_signal, llm_is_same, llm_confidence, decision \
                 FROM identity_verdict_audit WHERE group_id = 'g5' AND site = 'site5_acronym_nickname'",
                (),
            )
            .await
            .expect("audit query");
        let row = rows
            .next()
            .await
            .expect("row")
            .expect("exactly one merge audit row");
        let cosine: Option<f64> = row.get(0).expect("cosine col");
        let structural_signal: bool = row.get(1).expect("structural_signal col");
        let llm_is_same: bool = row.get(2).expect("llm_is_same col");
        let decision: String = row.get(4).expect("decision col");
        assert!(cosine.is_none(), "cosine must be NULL for Site #5");
        assert!(structural_signal, "structural signal must be recorded true");
        assert!(llm_is_same);
        assert_eq!(decision, "merge");
        assert!(
            rows.next().await.expect("row iter").is_none(),
            "exactly one audit row — no extras"
        );
    }

    /// A nominated pair with scripted LLM `is_same_entity=false` → Reject, no
    /// merge (write_gate row 3, honored without further checks).
    #[tokio::test]
    async fn nominated_pair_llm_false_rejects() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        insert_entity(&graph, "NASA", "g6", "A space agency.").await;
        insert_entity(
            &graph,
            "National Aeronautics and Space Administration",
            "g6",
            "The US space agency.",
        )
        .await;

        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[{"pair_id":0,"is_same_entity":false,"confidence":0.9,"reasoning":"not the same, hallucinated distinction"}]}"#
                .to_string(),
        };

        let report = acronym_nickname_recall(
            &llm,
            AcronymNicknameRecallParams {
                graph: &graph,
                group_id: "g6",
                model_id: "test-model",
            },
        )
        .await
        .expect("acronym_nickname_recall must succeed");

        assert_eq!(report.candidates_nominated, 1);
        assert_eq!(
            report.merges_applied, 0,
            "LLM false verdict must never merge"
        );
        assert_eq!(report.rejected, 1);
        assert_eq!(
            count_entities(&conn, "g6").await,
            2,
            "both entities survive — no merge"
        );
    }

    /// A nominated pair + LLM low-confidence (<0.7 floor) → PotentialAlias
    /// (write_gate row 4), potential_alias fact written, no merge.
    #[tokio::test]
    async fn nominated_pair_low_confidence_is_potential_alias() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let conn = graph.conn.clone();
        insert_entity(&graph, "FBI", "g7", "A law enforcement agency.").await;
        insert_entity(
            &graph,
            "Federal Bureau of Investigation",
            "g7",
            "A US federal law enforcement agency.",
        )
        .await;

        let llm = ScriptedVerdictProvider {
            json: r#"{"verdicts":[{"pair_id":0,"is_same_entity":true,"confidence":0.5,"reasoning":"plausible but unsure"}]}"#
                .to_string(),
        };

        let report = acronym_nickname_recall(
            &llm,
            AcronymNicknameRecallParams {
                graph: &graph,
                group_id: "g7",
                model_id: "test-model",
            },
        )
        .await
        .expect("acronym_nickname_recall must succeed");

        assert_eq!(report.candidates_nominated, 1);
        assert_eq!(
            report.merges_applied, 0,
            "low-confidence agreement must not merge (write_gate row 4)"
        );
        assert_eq!(report.potential_aliases, 1);
        assert_eq!(
            count_entities(&conn, "g7").await,
            2,
            "both entities survive — PotentialAlias is non-destructive"
        );

        // Confirm a potential_alias fact was written.
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM facts WHERE predicate = 'potential_alias' AND group_id = 'g7'",
                (),
            )
            .await
            .expect("facts query");
        let count: i64 = rows
            .next()
            .await
            .expect("row")
            .expect("row present")
            .get(0)
            .expect("count col");
        assert_eq!(count, 1, "one potential_alias fact must be written");
    }
}
