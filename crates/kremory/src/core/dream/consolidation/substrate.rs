//! Shared substrate for the dream CONSOLIDATION sub-phase (ADR-066 §2.5, spec P0).
//!
//! Three concerns, all zero-LLM + deterministic:
//!
//! - **`ConsolidationBudget`** (F-1, DoD-P0.1) — a single soft token-ceiling
//!   primitive. `check(projected)` is the pre-op gate (soft partial-abort: an op
//!   whose projected spend would push cumulative usage past the ceiling is skipped,
//!   later ops still run — matching kremory's warn-and-continue posture). `record`
//!   writes actual usage to the EXISTING `dream_pass_budget_usage` ledger table
//!   (migration 016).
//! - **Idempotency-key helpers** (F-2, DoD-P0.2) — canonicalized SHA-256 hashes for
//!   the richer objects consolidation keys on (a community's sorted member set, a
//!   sorted entity pair, a fact-archive event). Sorting makes the pair/member hashes
//!   order-independent, extending the proven ADR-050 `content_hash` shape.
//! - **`OpReport` / `ConsolidationSummary`** — the per-op result struct and the
//!   dispatcher's aggregate. Populated by the four ops (P1-P4); STUBS this phase.
//!
//! Spec: `.ai-docs/specs/adr-066-dream-consolidation-impl-spec-2026-07-03.md`
//! §3 (P0.1/P0.2) + ADR-066 §2.5 (F-1/F-2).

use sha2::{Digest, Sha256};

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

/// Unit separator (0x1F) between hash components — same delimiter the ADR-050
/// `canonical_entity_view` uses (`dream/idempotency.rs`), so field boundaries are
/// unambiguous and two distinct layouts can never collide.
// planned consumer: the idempotency helpers below (P1-P4 ops key on them).
#[allow(dead_code)]
const FIELD_SEP: char = '\u{1F}';

// ─── Budget primitive (F-1, DoD-P0.1) ──────────────────────────────────────────

/// A single SOFT token-ceiling for the consolidation sub-phase (ADR-066 §2.5 F-1).
///
/// Threaded mutably through `run_consolidation` so the four ops share ONE budget
/// (NOT per-subsystem fragmentation). `check` gates each op BEFORE it runs; ops
/// call `record` after spending to advance `used_tokens` and persist a ledger row.
///
/// **Scope honesty (first cut):** the **ledger** (`record`) is the load-bearing
/// half — it's free observability. The **ceiling** (`check`) is INERT in the
/// first cut: 3 of the 4 ops are zero-LLM and the only token-spending op
/// (supersession's LLM-nominate lane) is default-off, so nothing calls `record`
/// and `used_tokens` stays 0. The ceiling only bites once that lane ships — at
/// which point its real-token projection must be wired to `TokenTrackingChatProvider`
/// (which reads AutoAgents' `usage`) and validated. Cost is a DERIVED nicety:
/// local Ollama models are $0 (absent from `monitoring/provider-rates.toml`), so
/// the token count — not a dollar figure — is the meaningful bound for a
/// local-first background sweep.
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)`
/// so the deterministic supersession corpus harness can construct one under
/// `feature = "test-utils"` (external test binaries cannot import `pub(crate)`
/// items — E0365). NOT part of the stable public API.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ConsolidationBudget {
    /// Per-run ceiling (from `DreamOpts.consolidation_budget_tokens`). `u64::MAX`
    /// models an unbounded budget (`None` at the opts layer).
    pub ceiling_tokens: u64,
    /// Cumulative tokens spent by consolidation ops so far this run.
    pub used_tokens: u64,
}

impl ConsolidationBudget {
    /// Construct from an optional ceiling. `None` → unbounded (`u64::MAX`).
    pub fn new(ceiling_tokens: Option<u64>) -> Self {
        Self {
            ceiling_tokens: ceiling_tokens.unwrap_or(u64::MAX),
            used_tokens: 0,
        }
    }

    /// SOFT check: does `used + projected` stay within the ceiling?
    ///
    /// Returns `true` (op may run) when `used_tokens + projected <= ceiling_tokens`.
    /// Saturating add so a pathological `projected` can never wrap past the ceiling
    /// into a false allow.
    pub(crate) fn check(&self, projected: u64) -> bool {
        self.used_tokens.saturating_add(projected) <= self.ceiling_tokens
    }

    /// Record ACTUAL token spend for an op: advance `used_tokens` and append a row
    /// to the EXISTING `dream_pass_budget_usage` ledger (migration 016).
    ///
    /// `provider`/`model` are recorded for per-op source attribution (Rule 19). Cost
    /// is left NULL (Ollama dream passes report 0 cost; the ledger's `cost_usd_micro`
    /// is nullable). Non-fatal: a ledger-write failure returns `Err` for the caller's
    /// warn-and-continue handling, but `used_tokens` is advanced regardless so the
    /// in-memory ceiling stays honest even if persistence hiccups.
    // planned consumer: supersession LLM lane (P1) + communities (P4) — the two ops
    // that spend tokens call this after an LLM/graph pass. Exercised by unit test now.
    #[allow(dead_code)]
    pub(crate) async fn record(&mut self, params: BudgetRecordParams<'_>) -> Result<()> {
        let BudgetRecordParams {
            graph,
            pass_run_id,
            pass_name,
            provider,
            model,
            tokens_input,
            tokens_output,
        } = params;
        self.used_tokens = self
            .used_tokens
            .saturating_add(tokens_input.saturating_add(tokens_output));

        let now = chrono::Utc::now().timestamp();
        graph
            .conn
            .execute(
                "INSERT OR REPLACE INTO dream_pass_budget_usage \
                 (pass_run_id, pass_name, provider, model, tokens_input, tokens_output, \
                  cost_usd_micro, recorded_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7)",
                libsql::params![
                    pass_run_id,
                    pass_name,
                    provider,
                    model,
                    tokens_input as i64,
                    tokens_output as i64,
                    now,
                ],
            )
            .await?;
        Ok(())
    }
}

/// Bundled params for [`ConsolidationBudget::record`] — args-as-object per TD-042
/// (`too_many_arguments` threshold 3). `graph` is the receiver-like lead dep.
// planned consumer: constructed by supersession (P1) + communities (P4) at their
// `record` call sites. Constructed by the unit test now.
#[allow(dead_code)]
pub(crate) struct BudgetRecordParams<'a> {
    pub(crate) graph: &'a TemporalGraph,
    pub(crate) pass_run_id: &'a str,
    pub(crate) pass_name: &'a str,
    pub(crate) provider: &'a str,
    pub(crate) model: &'a str,
    pub(crate) tokens_input: u64,
    pub(crate) tokens_output: u64,
}

// ─── Idempotency-key helpers (F-2, DoD-P0.2) ────────────────────────────────────

/// SHA-256 hex of the arg — the shared digest used by every consolidation
/// idempotency helper. Mirrors the `dream/idempotency.rs::content_hash` hex loop.
fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// Idempotency hash for a community: SHA-256 over the entity-id members
/// **sorted** then separator-joined (ADR-066 §F-2). Sorting makes the hash a pure
/// function of the membership SET — a community whose members are unchanged
/// (regardless of discovery order) hashes identically and is not re-counted
/// (`communities_updated`, P4.3/P4.5).
// planned consumer: communities op (P4) keys community idempotency on this.
#[allow(dead_code)]
pub(crate) fn community_member_hash(member_ids: &[&str]) -> String {
    let mut sorted: Vec<&str> = member_ids.to_vec();
    sorted.sort_unstable();
    let joined = sorted.join(&FIELD_SEP.to_string());
    sha256_hex(&joined)
}

/// Idempotency hash for an entity/fact PAIR + op name: SHA-256 over the
/// **sorted** pair `(min, max)` then the op (ADR-066 §F-2). Sorting makes the hash
/// order-independent — `pair_hash(a, b, op) == pair_hash(b, a, op)` — so a pair
/// cross_episode/supersession already processed hashes identically on the next run.
// planned consumer: cross_episode (P3) + supersession (P1) key pair idempotency on this.
#[allow(dead_code)]
pub(crate) fn pair_hash(id_a: &str, id_b: &str, op: &str) -> String {
    let (lo, hi) = if id_a <= id_b {
        (id_a, id_b)
    } else {
        (id_b, id_a)
    };
    let joined = format!("{lo}{FIELD_SEP}{hi}{FIELD_SEP}{op}");
    sha256_hex(&joined)
}

/// Idempotency hash for a fact-archive event: SHA-256 over `fact_id` + its
/// `expired_at` (ADR-066 §F-2). Keys the archive move so a re-run of the archive op
/// on an already-moved fact is a no-op (P2.4).
// planned consumer: archive op (P2) keys the archive-move idempotency on this.
#[allow(dead_code)]
pub(crate) fn fact_archive_hash(fact_id: i64, expired_at: &str) -> String {
    let joined = format!("{fact_id}{FIELD_SEP}{expired_at}");
    sha256_hex(&joined)
}

// ─── Per-op report + aggregate summary ──────────────────────────────────────────

/// The result of a single consolidation op (P1-P4). One count (the op's own
/// source-attributed mutation total) + any non-fatal warnings it emitted + (for
/// cross_episode only) the merge decisions for the orchestrator to fan out as
/// `on_merge_proposed` events (ADR-070 Fork 5).
///
/// The four ops are fully implemented and populate their real counts (the inert path
/// — all-zero `default()` — is only returned when an op is disabled or errors).
///
/// **MNT-002 visibility (`pub` + `#[doc(hidden)]`):** promoted from `pub(crate)`
/// so the deterministic supersession corpus harness can read `count`/`warnings`
/// under `feature = "test-utils"`. NOT part of the stable public API.
/// One cross-episode merge decision (keeper ← loser) surfaced by `cross_episode` so
/// the orchestrator (`run_consolidation`) can fire the `on_merge_proposed` consumer
/// event from the layer that already holds the `EnrichmentEventSink` (ADR-070 Fork 5,
/// Risk #17 orchestrator-fires contingency — chosen over threading the sink INTO
/// `cross_episode`, which would push its arity past the `too_many_arguments` threshold).
/// Carries only the two entity ids; `dry_run` is known by the orchestrator (from the
/// `DreamOpts`), so it is not duplicated here.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossEpisodeMerge {
    pub keeper: String,
    pub loser: String,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpReport {
    /// The op's own mutation count (supersessions / archived / merges / communities
    /// updated — interpreted by the caller which op produced it).
    pub count: usize,
    /// Non-fatal notices (budget skip, degraded mode, …) folded into the
    /// `DreamSummary.warnings` surface (`facade/mod.rs`).
    pub warnings: Vec<String>,
    /// Cross-episode merge decisions this pass (keeper ← loser), for the orchestrator
    /// to fire `on_merge_proposed` (ADR-070 Fork 5). Populated ONLY by `cross_episode`;
    /// EMPTY for every other op (default). One entry per `merged` decision, whether
    /// shadowed or applied.
    pub merges: Vec<CrossEpisodeMerge>,
}

/// Aggregate of the four consolidation ops (ADR-066 §5). The dispatcher
/// (`run_consolidation`) fills each field from its op's [`OpReport`]; the facade
/// folds these into the four (already-existing) `DreamSummary` consolidation
/// fields.
///
/// `Default` = all-zero (the inert path when no op is enabled OR the whole
/// dispatcher errored — `unwrap_or_default()` in the facade).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ConsolidationSummary {
    pub(crate) communities_updated: usize,
    pub(crate) cross_episode_merges: usize,
    pub(crate) supersessions_recorded: usize,
    pub(crate) facts_archived: usize,
    /// Aggregated non-fatal warnings across all four ops.
    pub(crate) warnings: Vec<String>,
}

// ─── Uniform decision telemetry (ADR-070 Fork 2/3) ──────────────────────────────
//
// A single `emit_decision(DecisionRecord)` contract every consolidation op calls at
// its decision points. Structurally enforces the cardinality split (Fork 3): the
// LOW-cardinality discriminants (`op`/`mode`/`outcome`) go on the counter's labels;
// the HIGH-cardinality context (`group_id`/`entity_refs`/`debug_context`) goes ONLY
// on the trace event. `outcome: &'static str` makes the split a TYPE-level guarantee
// — an author cannot compile `format!("merged_{id}")` (a `String`) into a label.

/// Low-cardinality op discriminant (Fork 3) — fixed 4-variant enum, safe as a
/// counter label under any cardinality-safety rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConsolidationOpKind {
    Supersession,
    Archive,
    CrossEpisode,
    Communities,
}

impl ConsolidationOpKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Supersession => "supersession",
            Self::Archive => "archive",
            Self::CrossEpisode => "cross_episode",
            Self::Communities => "communities",
        }
    }
}

/// Low-cardinality mode discriminant (Fork 1/3) — 2 variants, safe as a counter
/// label. `Shadow` = decision computed, write skipped (dry_run). `Applied` = write
/// committed (or the op has no dry_run concept — always `Applied`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecisionMode {
    /// planned consumer: Phase C1 — the cross_episode shadow gate is the first
    /// non-test site to construct `Shadow`; until then only `Applied` is emitted.
    #[allow(dead_code)]
    Shadow,
    Applied,
}

impl DecisionMode {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Applied => "applied",
        }
    }
}

/// One consolidation op's decision at a single decision point (ADR-070 Fork 2).
/// `op`/`mode`/`outcome` are LOW-cardinality — safe as counter labels.
/// `group_id`/`entity_refs`/`debug_context` are HIGH-cardinality — trace fields
/// ONLY (Fork 3). [`emit_decision`] enforces this split structurally: there is no
/// code path by which a `DecisionRecord` field reaches the wrong sink.
pub(crate) struct DecisionRecord<'a> {
    pub(crate) op: ConsolidationOpKind,
    pub(crate) mode: DecisionMode,
    /// Fixed short string per op (e.g. cross_episode's `"merged"` / `"homonym_skip"`
    /// / `"hub_skip"` / `"group_size_skip"`). MUST be drawn from a small fixed
    /// enumeration per op — never an interpolated value (no entity id, no score).
    /// The `&'static str` type makes that structural: interpolated data is a
    /// `String` and will not compile here.
    pub(crate) outcome: &'static str,
    pub(crate) group_id: &'a str,
    pub(crate) entity_refs: &'a [&'a str],
    pub(crate) debug_context: Option<String>,
}

/// The ONE emission function every consolidation op calls at its decision point
/// (ADR-070 Fork 2/3). Structurally enforces the cardinality split: low-cardinality
/// fields go on the counter's labels; high-cardinality fields go ONLY on the trace
/// event.
pub(crate) fn emit_decision(record: DecisionRecord<'_>) {
    metrics::counter!(
        "kremory.dream.consolidation.decision_total",
        "op" => record.op.as_str(),
        "mode" => record.mode.as_str(),
        "outcome" => record.outcome,
    )
    .increment(1);

    tracing::info!(
        target: "kremory.dream.consolidation.decision",
        op = record.op.as_str(),
        mode = record.mode.as_str(),
        outcome = record.outcome,
        group_id = record.group_id,
        entity_refs = ?record.entity_refs,
        debug_context = record.debug_context.as_deref(),
        "consolidation decision"
    );
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::TemporalGraph;

    // ── Budget (DoD-P0.1) ───────────────────────────────────────────────────────

    #[test]
    fn budget_allows_within_ceiling_denies_over() {
        let b = ConsolidationBudget {
            ceiling_tokens: 1_000,
            used_tokens: 400,
        };
        // used + projected == ceiling → allowed (soft, inclusive).
        assert!(b.check(600), "used(400)+proj(600)=1000 == ceiling → allow");
        // used + projected > ceiling → denied.
        assert!(!b.check(601), "used(400)+proj(601)=1001 > ceiling → deny");
        assert!(
            b.check(0),
            "zero projected always allowed under a positive ceiling"
        );
    }

    #[test]
    fn budget_saturating_add_cannot_wrap_into_false_allow() {
        let b = ConsolidationBudget {
            ceiling_tokens: 1_000,
            used_tokens: 10,
        };
        // A pathological projected near u64::MAX must saturate, never wrap → deny.
        assert!(!b.check(u64::MAX), "saturating add must deny, not wrap");
    }

    #[test]
    fn budget_new_none_is_unbounded() {
        let b = ConsolidationBudget::new(None);
        assert_eq!(b.ceiling_tokens, u64::MAX);
        assert!(
            b.check(1_000_000_000),
            "unbounded budget allows any projected"
        );
    }

    #[tokio::test]
    async fn budget_record_writes_ledger_row_and_advances_used() {
        let graph = TemporalGraph::open_in_memory().await.expect("open");
        let mut budget = ConsolidationBudget::new(Some(10_000));

        budget
            .record(BudgetRecordParams {
                graph: &graph,
                pass_run_id: "run-1",
                pass_name: "consolidation_supersession",
                provider: "ollama",
                model: "gemma4:e4b",
                tokens_input: 120,
                tokens_output: 30,
            })
            .await
            .expect("record");

        assert_eq!(budget.used_tokens, 150, "used advanced by input+output");

        // Ledger row landed in the migration-016 table.
        let mut rows = graph
            .conn
            .query(
                "SELECT provider, model, tokens_input, tokens_output \
                 FROM dream_pass_budget_usage \
                 WHERE pass_run_id = 'run-1' AND pass_name = 'consolidation_supersession'",
                (),
            )
            .await
            .expect("query ledger");
        let row = rows.next().await.expect("row").expect("ledger row exists");
        let provider: String = row.get(0).expect("provider");
        let model: String = row.get(1).expect("model");
        let ti: i64 = row.get(2).expect("tokens_input");
        let to: i64 = row.get(3).expect("tokens_output");
        assert_eq!(provider, "ollama");
        assert_eq!(model, "gemma4:e4b");
        assert_eq!(ti, 120);
        assert_eq!(to, 30);
    }

    // ── Idempotency helpers (DoD-P0.2) ──────────────────────────────────────────

    #[test]
    fn community_member_hash_is_order_independent() {
        let a = community_member_hash(&["ent-c", "ent-a", "ent-b"]);
        let b = community_member_hash(&["ent-a", "ent-b", "ent-c"]);
        let c = community_member_hash(&["ent-b", "ent-c", "ent-a"]);
        assert_eq!(a, b, "sorted member hash must be order-independent");
        assert_eq!(b, c, "sorted member hash must be order-independent");
    }

    #[test]
    fn community_member_hash_changes_on_membership_change() {
        let base = community_member_hash(&["ent-a", "ent-b"]);
        let added = community_member_hash(&["ent-a", "ent-b", "ent-c"]);
        let removed = community_member_hash(&["ent-a"]);
        assert_ne!(base, added, "adding a member must change the hash");
        assert_ne!(base, removed, "removing a member must change the hash");
    }

    #[test]
    fn pair_hash_is_order_independent() {
        assert_eq!(
            pair_hash("ent-a", "ent-b", "cross_episode"),
            pair_hash("ent-b", "ent-a", "cross_episode"),
            "pair hash must be order-independent (sorted pair)"
        );
    }

    #[test]
    fn pair_hash_distinguishes_op_and_members() {
        let cross = pair_hash("ent-a", "ent-b", "cross_episode");
        let sup = pair_hash("ent-a", "ent-b", "supersession");
        let other_pair = pair_hash("ent-a", "ent-c", "cross_episode");
        assert_ne!(cross, sup, "op name must be part of the key");
        assert_ne!(cross, other_pair, "member set must be part of the key");
    }

    #[test]
    fn fact_archive_hash_keys_on_id_and_expired_at() {
        let base = fact_archive_hash(42, "2026-01-01T00:00:00Z");
        let diff_id = fact_archive_hash(43, "2026-01-01T00:00:00Z");
        let diff_time = fact_archive_hash(42, "2026-02-01T00:00:00Z");
        assert_eq!(
            base,
            fact_archive_hash(42, "2026-01-01T00:00:00Z"),
            "deterministic"
        );
        assert_ne!(base, diff_id, "fact_id is part of the key");
        assert_ne!(base, diff_time, "expired_at is part of the key");
    }

    #[test]
    fn hashes_are_sha256_hex_length() {
        for h in [
            community_member_hash(&["x"]),
            pair_hash("a", "b", "op"),
            fact_archive_hash(1, "t"),
        ] {
            assert_eq!(h.len(), 64, "sha256 hex is 64 chars");
            assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    // ── Property test: hash order-independence over random permutations ─────────

    #[test]
    fn prop_member_and_pair_hash_order_independent_over_permutations() {
        // DoD-P0.2 property: the pair/member hashes are invariant under any input
        // permutation. Deterministic pseudo-random permutations (no rng dep):
        // exhaustively permute a 4-member set + all 2-orderings of a pair.
        let members = ["ent-1", "ent-2", "ent-3", "ent-4"];
        let canonical = community_member_hash(&members);
        // All 24 permutations of the 4-member set must hash identically.
        let mut perms: Vec<Vec<&str>> = Vec::new();
        permute(&members, 0, &mut perms);
        assert_eq!(perms.len(), 24, "4! = 24 permutations");
        for p in &perms {
            let refs: Vec<&str> = p.clone();
            assert_eq!(
                community_member_hash(&refs),
                canonical,
                "permutation {p:?} must hash to the canonical member hash"
            );
        }

        // Pair hash: both orderings identical for every unordered pair drawn from
        // the member set.
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                assert_eq!(
                    pair_hash(members[i], members[j], "op"),
                    pair_hash(members[j], members[i], "op"),
                    "pair ({}, {}) must be order-independent",
                    members[i],
                    members[j]
                );
            }
        }
    }

    /// Recursive permutation generator (test-only, no rng dependency).
    fn permute<'a>(items: &[&'a str], k: usize, out: &mut Vec<Vec<&'a str>>) {
        if k == items.len() {
            out.push(items.to_vec());
            return;
        }
        let mut items = items.to_vec();
        for i in k..items.len() {
            items.swap(k, i);
            permute(&items, k + 1, out);
            items.swap(k, i);
        }
    }

    // ── Decision telemetry (ADR-070 Fork 2/3) ───────────────────────────────────

    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    #[test]
    fn op_kind_and_mode_render_fixed_low_cardinality_strings() {
        // The label domain is a fixed, code-reviewable enumeration (Fork 3).
        assert_eq!(ConsolidationOpKind::Supersession.as_str(), "supersession");
        assert_eq!(ConsolidationOpKind::Archive.as_str(), "archive");
        assert_eq!(ConsolidationOpKind::CrossEpisode.as_str(), "cross_episode");
        assert_eq!(ConsolidationOpKind::Communities.as_str(), "communities");
        assert_eq!(DecisionMode::Shadow.as_str(), "shadow");
        assert_eq!(DecisionMode::Applied.as_str(), "applied");
    }

    /// Does `decision_total` carry exactly the `(op, mode, outcome)` labels with the
    /// given count? Filters a `DebuggingRecorder` snapshot (the reuse pattern from
    /// `tests/consolidation_supersession_test.rs`).
    fn decision_counter(
        snapshotter: &metrics_util::debugging::Snapshotter,
        op: &str,
        mode: &str,
        outcome: &str,
    ) -> Option<u64> {
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .find_map(|(composite_key, _, _, value)| {
                let key = composite_key.key();
                if key.name() != "kremory.dream.consolidation.decision_total" {
                    return None;
                }
                let labels: std::collections::HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("op").copied() != Some(op)
                    || labels.get("mode").copied() != Some(mode)
                    || labels.get("outcome").copied() != Some(outcome)
                {
                    return None;
                }
                match value {
                    DebugValue::Counter(n) => Some(n),
                    _ => None,
                }
            })
    }

    #[test]
    fn emit_decision_fires_decision_total_with_low_cardinality_labels() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);
        emit_decision(DecisionRecord {
            op: ConsolidationOpKind::CrossEpisode,
            mode: DecisionMode::Shadow,
            outcome: "merged",
            // HIGH-cardinality context — must NOT reach the counter labels.
            group_id: "g-cardinality",
            entity_refs: &["keeper-1", "loser-2"],
            debug_context: Some("free-text detail".to_string()),
        });

        assert_eq!(
            decision_counter(&snapshotter, "cross_episode", "shadow", "merged"),
            Some(1),
            "decision_total{{op=cross_episode,mode=shadow,outcome=merged}} must fire once"
        );

        // Fork 3 structural guarantee: NO counter key carries the high-cardinality
        // group_id / entity id as a label value.
        let leaked = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .any(|(composite_key, _, _, _)| {
                let key = composite_key.key();
                key.labels()
                    .any(|l| l.value() == "g-cardinality" || l.value() == "keeper-1")
            });
        assert!(
            !leaked,
            "high-cardinality group_id/entity_refs must never appear as a counter label"
        );
    }

    #[test]
    fn emit_decision_applied_mode_renders_applied_label() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);
        emit_decision(DecisionRecord {
            op: ConsolidationOpKind::Archive,
            mode: DecisionMode::Applied,
            outcome: "archived",
            group_id: "g1",
            entity_refs: &["ent-a"],
            debug_context: None,
        });
        assert_eq!(
            decision_counter(&snapshotter, "archive", "applied", "archived"),
            Some(1),
            "decision_total{{op=archive,mode=applied,outcome=archived}} must fire once"
        );
    }
}
