use metrics::counter;

use crate::core::dream::provenance::{FactSupersedeInputs, FactSupersedePreState, MutationKind};
use crate::core::schema::Fact;

use super::*;

// ── SupersedeRequest ─────────────────────────────────────────────────────────

/// Consumer-facing supersede builder. Obtain via
/// `mem.supersede(fact_id)`.
///
/// Bounds a fact's WORLD-time `valid_to` window explicitly — the consumer
/// asserts "this fact's validity ends here" (e.g. a document was superseded by
/// a newer version, a price changed, a status expired). This is the
/// consumer-EXPLICIT half of the supersession story; auto-detected
/// supersession (the system inferring a supersession from new input) is
/// deliberately deferred — not built here.
///
/// # Mechanism
///
/// `.execute()` writes `facts.valid_to` via the NEW
/// [`TemporalGraph::bound_valid_to`] primitive (`core/graph/facts.rs`) — **NOT**
/// `invalidate_fact` / `invalidate_fact_with_reason`, which write `expired_at`/
/// `invalid_at` (system-time / domain-invalidation semantics) and would starve
/// the dream supersession sweep's `window_closeout` predicate (`valid_to IS NOT
/// NULL AND valid_to < now AND expired_at IS NULL AND invalid_at IS NULL`)
/// forever. This builder is the PRODUCER half of a two-phase chain:
///
/// 1. `mem.supersede(fact_id).at(valid_to).execute()` — bounds `valid_to`
///    (world-time) — THIS builder.
/// 2. The next `mem.dream()` call with `include_supersession_sweep: true` —
///    the CONSUMER (`core/dream/consolidation/supersession.rs`'s
///    `window_closeout`, unchanged) — observes the bounded `valid_to` and sets
///    `expired_at = valid_to` (system-time close), incrementing
///    `DreamSummary.supersessions_recorded`.
///
/// Must call `.execute()` explicitly — this is a mutating operation (mirrors
/// `ForgetRequest`'s "no accidental `.await`" discipline).
pub struct SupersedeRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) fact_id: i64,
    pub(super) namespace: Option<Namespace>,
    /// The bounded `valid_to` timestamp — REQUIRED before `.execute()`. No
    /// silent default: an unset `valid_to` fails LOUDLY at `.execute()` time
    /// rather than defaulting to `Utc::now()` (`llm-output-parse-loudly`'s
    /// "required fields carry no silent default" discipline, extended here to
    /// required consumer-builder input, not just LLM output).
    pub(super) valid_to: Option<DateTime<Utc>>,
    /// Optional human-readable reason. Threaded as a `tracing`/audit FIELD on
    /// the `bound_valid_to` call only — it does NOT route to
    /// `invalidate_fact_with_reason` (that primitive writes `invalid_at`, a
    /// domain-invalidation semantic that would break the `window_closeout`
    /// predicate).
    pub(super) reason: Option<String>,
    /// When `true`, `.execute()` runs the deterministic
    /// `window_closeout` sweep inline right after bounding `valid_to`, retiring
    /// already-past-dated bounds in one call. Set via [`Self::close_now`]. Default
    /// `false` — the retirement is otherwise deferred to the next `mem.dream()` with
    /// `include_supersession_sweep`.
    pub(super) close_now: bool,
}

impl<'a> SupersedeRequest<'a> {
    /// Set the bounded `valid_to` timestamp — REQUIRED before `.execute()`.
    pub fn at(mut self, valid_to: DateTime<Utc>) -> Self {
        self.valid_to = Some(valid_to);
        self
    }

    /// Optional human-readable reason, threaded as a tracing field only —
    /// never a metric label (cardinality discipline).
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the namespace scope for this supersede (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Retire the bound in ONE call instead of waiting
    /// for the next `mem.dream()` supersession sweep. After bounding `valid_to`,
    /// `.execute()` runs the deterministic `window_closeout` inline and returns
    /// [`SupersedeOutcome::Bounded`] carrying the retirement `retired` count.
    ///
    /// **Only ALREADY-PAST-DATED bounds retire** (the `valid_to < now` predicate):
    /// a bound at a FUTURE `valid_to` cannot be retired yet, so `close_now()` on it
    /// returns `Bounded { retired: 0 }` (its retirement is deferred to a later dream
    /// sweep once the window closes). The count is carried IN-BAND (`retired == 0` on
    /// a future-dated bound) rather than a doc caveat, so a caller can observe the
    /// deferral without inspecting the sweep — the same honesty applied to the
    /// variant name below.
    ///
    /// NOTE: `window_closeout` is namespace-scoped — the returned `retired` is the
    /// total retired in the resolved namespace this sweep, which includes any OTHER
    /// past-dated bounds, not solely this `fact_id`.
    pub fn close_now(mut self) -> Self {
        self.close_now = true;
        self
    }

    /// Execute the supersession. Bounds `fact_id`'s world-time `valid_to` via
    /// [`TemporalGraph::bound_valid_to`] — NOT
    /// `invalidate_fact`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `.at(...)` was never called (required-input
    /// discipline — no silent default), if namespace resolution fails, or if
    /// the underlying DB write fails. A missing/out-of-namespace `fact_id` is
    /// NOT an error — see [`SupersedeOutcome::NotFound`].
    pub async fn execute(self) -> Result<SupersedeOutcome> {
        // Required-input discipline: `.at()` must have been called. No silent
        // `Utc::now()` fallback — a missing bound is a caller bug, surfaced
        // loudly rather than silently mis-bounding the fact.
        let valid_to = self.valid_to.ok_or_else(|| {
            MemoryError::Other(
                "SupersedeRequest::execute requires .at(valid_to) — no silent default"
                    .into(),
            )
        })?;

        let ns = self.memory.resolve_namespace(self.namespace)?;
        // Lazy population: ensure namespace row exists before read.
        self.memory.ensure_namespace_policy(&ns).await?;
        let group_id = namespace_to_group_id(&ns);

        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::supersede requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        // AppendOnly enforcement: "valid_to set on facts" is a
        // named mutation — supersede bounds `facts.valid_to`, a mutation of an
        // existing row, and is prohibited on AppendOnly namespaces. Mirrors
        // `ForgetRequest::execute` (forget.rs). Without this
        // the mutation not only violates policy, it is permanently orphaned:
        // `dream()` itself refuses to run on AppendOnly namespaces, so
        // `window_closeout` could never complete the system-time close.
        let policy = tg
            .get_namespace_policy_cached(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        if let Some(p) = &policy {
            if p.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                return Err(MemoryError::Core(
                    crate::core::error::Error::NamespacePolicyViolation {
                        namespace: group_id.clone(),
                        operation: "supersede".to_string(),
                        policy: p.clone(),
                    },
                ));
            }
        }

        let Some(fact) = tg
            .get_fact_by_id(self.fact_id, &group_id)
            .await
            .map_err(MemoryError::Core)?
        else {
            counter!(
                "kremory.dream.consolidation.supersede_request_total",
                "outcome" => "not_found"
            )
            .increment(1);
            return Ok(SupersedeOutcome::NotFound);
        };

        // Time-inversion guard: a bound predating the fact's own
        // `valid_from` is nonsensical regardless of consumer intent. DB row
        // stays UNCHANGED — no partial write.
        if valid_to < fact.valid_from {
            counter!(
                "kremory.dream.consolidation.supersede_request_total",
                "outcome" => "rejected_time_inversion"
            )
            .increment(1);
            return Ok(SupersedeOutcome::RejectedTimeInversion);
        }

        // bound_valid_to + log the mutation, atomically. `bound_valid_to`'s own
        // UPDATE and the `graph_mutation_log` INSERT below share this txn for the
        // same reason TD-250's archive move does: a log row that commits
        // independently of the bound it describes is worse than no log row — it
        // would report a supersede that never durably happened, or (the gap this
        // closes) leave a real one unenumerable by `list_mutations`. Without this,
        // a retraction was reachable only by a consumer who already held the
        // `fact_id` from before the bound closed its validity window — the exact
        // gap that killed the proposed sixth MCP tool
        // (`.ai-docs/plans/mcp-agent-api-design-2026-09-14.md` §"the sixth tool").
        let guard = tg.begin_immediate_if_needed().await.map_err(MemoryError::Core)?;
        let write: crate::core::error::Result<()> = async {
            // bound_valid_to, NOT invalidate_fact — writes `valid_to`
            // (world-time), not `expired_at` (system-time). The dream supersession
            // sweep's `window_closeout` does the system-time close later.
            tg.bound_valid_to(self.fact_id, valid_to).await?;
            log_supersede_mutation(
                tg,
                LogSupersedeParams {
                    fact: &fact,
                    valid_to,
                    group_id: &group_id,
                },
            )
            .await
        }
        .await;
        match write {
            Ok(()) => guard.commit().await.map_err(MemoryError::Core)?,
            Err(e) => {
                let _ = guard.rollback().await;
                return Err(MemoryError::Core(e));
            }
        }

        // fact_id/group_id/reason are KREMORY_DEBUG-gated tracing FIELDS only —
        // never counter labels (cardinality discipline).
        if std::env::var("KREMORY_DEBUG").is_ok() {
            tracing::debug!(
                fact_id = self.fact_id,
                group_id = %group_id,
                reason = self.reason.as_deref().unwrap_or(""),
                "kremory.supersede.applied"
            );
        }

        counter!(
            "kremory.dream.consolidation.supersede_request_total",
            "outcome" => "bounded"
        )
        .increment(1);

        // `execute()` only BOUNDS `valid_to` — retirement is a separate step, so
        // the honest default is `retired: 0`. When `.close_now()` was requested, run
        // the deterministic window_closeout inline and carry the real count IN-BAND
        // (a future-dated bound retires nothing → `retired == 0`, observable without
        // inspecting the sweep).
        let retired = if self.close_now {
            crate::core::dream::consolidation::supersession::window_closeout(tg, &group_id)
                .await
                .map_err(MemoryError::Core)?
        } else {
            0
        };

        Ok(SupersedeOutcome::Bounded { retired })
    }
}

/// Args for [`log_supersede_mutation`] — an object rather than positional
/// parameters, which the workspace clippy threshold (3) rejects.
struct LogSupersedeParams<'a> {
    /// The fact row as read BEFORE this call's bound — its `valid_to`/
    /// `expired_at` are the pre-state a later undo restores.
    fact: &'a Fact,
    /// The bound this call is applying (not yet reflected in `fact`).
    valid_to: DateTime<Utc>,
    group_id: &'a str,
}

/// Write the `fact_supersede` row to `graph_mutation_log` (mirrors TD-250's
/// `log_archive_mutation`). Caller must already hold the bound write's
/// transaction — deliberately NOT self-contained, for the same reason the
/// archive log write isn't: a log row that commits independently of the bound
/// it describes is worse than no log row at all.
async fn log_supersede_mutation(
    graph: &TemporalGraph,
    p: LogSupersedeParams<'_>,
) -> crate::core::error::Result<()> {
    let LogSupersedeParams {
        fact,
        valid_to,
        group_id,
    } = p;
    let now = chrono::Utc::now().to_rfc3339();
    let pre_state = serde_json::to_string(&FactSupersedePreState {
        fact_id: fact.id,
        prior_valid_to: fact.valid_to.map(|t| t.to_rfc3339()),
        prior_expired_at: fact.expired_at.map(|t| t.to_rfc3339()),
    })
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "serialize fact_supersede pre_state: {e}"
        ))
    })?;
    let inputs = serde_json::to_string(&FactSupersedeInputs {
        fact_id: fact.id,
        subject_id: fact.subject_id.clone(),
        predicate: fact.predicate.clone(),
        object_id: fact.object_id.clone(),
        valid_to: valid_to.to_rfc3339(),
    })
    .map_err(|e| {
        crate::core::error::Error::Other(anyhow::anyhow!("serialize fact_supersede inputs: {e}"))
    })?;
    graph
        .conn
        .execute(
            "INSERT INTO graph_mutation_log (kind, group_id, created_at, pre_state, inputs) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                MutationKind::FactSupersede.as_tag(),
                group_id,
                now,
                pre_state,
                inputs
            ],
        )
        .await?;
    Ok(())
}

/// Typed outcome of a [`SupersedeRequest::execute`] call — NOT a bare
/// bool/count, per `llm-output-parse-loudly`'s "required fields have no
/// silent default" discipline extended to consumer-facing typed results: a
/// caller must be able to distinguish WHY a supersede didn't apply, not just
/// that it didn't. Exactly 3 variants, 1:1 with the `outcome` label values on
/// `kremory.dream.consolidation.supersede_request_total` (`bounded` /
/// `rejected_time_inversion` / `not_found`).
///
/// `#[non_exhaustive]` — the outcome surface is unreleased and may gain
/// variants; callers match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SupersedeOutcome {
    /// `valid_to` was BOUNDED successfully (an honest
    /// rename of the former `Applied`). `execute()` only writes `valid_to`; it does
    /// NOT retire the fact — so `retired` is `0` unless `.close_now()` was called,
    /// in which case it carries the inline `window_closeout` retirement count (which
    /// is still `0` for a FUTURE-dated bound whose window has not yet closed).
    Bounded {
        /// Facts retired by an inline `.close_now()` sweep (`0` for a plain
        /// `execute()` or a future-dated bound). Namespace-scoped total.
        retired: usize,
    },
    /// The supplied `valid_to` predates the fact's own `valid_from` — a
    /// nonsensical bound, rejected structurally. DB row unchanged.
    RejectedTimeInversion,
    /// `fact_id` does not exist, or does not exist in the resolved namespace.
    NotFound,
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod supersede_tests {
    use std::sync::Arc;

    use chrono::Duration;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::{Memory, SupersedeOutcome};

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    /// Plant a subject entity + fact directly via the graph, mirroring
    /// `consolidation_supersession_test.rs`'s `run_one_row` seeding pattern.
    /// Returns the inserted `fact_id`.
    async fn seed_fact(
        mem: &Memory,
        ns: &Namespace,
        valid_from: chrono::DateTime<chrono::Utc>,
    ) -> i64 {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let group_id = mem.group_id_for_test(ns);

        tg.insert_entity_with_group(crate::core::graph::InsertEntityWithGroupParams {
            id: "entity-supersede-subject",
            entity_type_id: 0,
            properties: serde_json::json!({}),
            group_id: Some(&group_id),
        })
        .await
        .expect("seed subject entity");

        tg.insert_fact_with_group(
            crate::core::graph::FactInsert::new("entity-supersede-subject", "status", valid_from)
                .object_value("active"),
            Some(&group_id),
        )
        .await
        .expect("seed fact")
    }

    /// AC — happy path: `.at(t)` where `t` is after the fact's `valid_from`
    /// applies cleanly; the DB row's `valid_to` matches.
    #[tokio::test]
    async fn supersede_applies_bounded_valid_to() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-supersede-applies");
        let now = chrono::Utc::now();
        let valid_from = now - Duration::days(10);
        let fact_id = seed_fact(&mem, &ns, valid_from).await;

        let bound_at = now - Duration::days(1);
        let outcome = mem
            .supersede(fact_id)
            .in_namespace(ns.clone())
            .at(bound_at)
            .execute()
            .await
            .expect("supersede must succeed");

        // execute() only BOUNDS — never retires — so retired is 0.
        assert_eq!(outcome, SupersedeOutcome::Bounded { retired: 0 });

        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let group_id = mem.group_id_for_test(&ns);
        let fact = tg
            .get_fact_by_id(fact_id, &group_id)
            .await
            .expect("get_fact_by_id must succeed")
            .expect("fact must exist");
        assert_eq!(
            fact.valid_to.map(|t| t.timestamp()),
            Some(bound_at.timestamp()),
            "DB valid_to must match the supplied bound"
        );
    }

    /// `.close_now()` on a PAST-dated bound runs the
    /// inline window_closeout and returns `Bounded { retired: >0 }`, while a
    /// FUTURE-dated bound in the same namespace is left un-retired (its window has
    /// not closed). Proves the honest in-band count: the deferral is observable from
    /// the return (`retired`), not a doc caveat.
    #[tokio::test]
    async fn close_now_retires_past_bound_and_leaves_future_bound() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-supersede-close-now");
        let now = chrono::Utc::now();
        let valid_from = now - Duration::days(30);

        // Seed ONE subject entity + two DISTINCT facts (distinct predicates so the
        // content hashes differ — the entity PK can only be inserted once).
        let (past_fact, future_fact) = {
            let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
            let group_id = mem.group_id_for_test(&ns);
            tg.insert_entity_with_group(crate::core::graph::InsertEntityWithGroupParams {
                id: "entity-close-now-subject",
                entity_type_id: 0,
                properties: serde_json::json!({}),
                group_id: Some(&group_id),
            })
            .await
            .expect("seed subject entity");
            let past = tg
                .insert_fact_with_group(
                    crate::core::graph::FactInsert::new(
                        "entity-close-now-subject",
                        "status",
                        valid_from,
                    )
                    .object_value("active"),
                    Some(&group_id),
                )
                .await
                .expect("seed past fact");
            let future = tg
                .insert_fact_with_group(
                    crate::core::graph::FactInsert::new(
                        "entity-close-now-subject",
                        "role",
                        valid_from,
                    )
                    .object_value("member"),
                    Some(&group_id),
                )
                .await
                .expect("seed future fact");
            (past, future)
        };

        // Bound the FUTURE fact first (no close) — its window is still open.
        let future_bound = now + Duration::days(10);
        let out_future = mem
            .supersede(future_fact)
            .in_namespace(ns.clone())
            .at(future_bound)
            .execute()
            .await
            .expect("future supersede must succeed");
        assert_eq!(
            out_future,
            SupersedeOutcome::Bounded { retired: 0 },
            "plain execute() bounds only — retired 0"
        );

        // Bound the PAST fact WITH close_now — window is already closed, so the
        // inline window_closeout retires it. The namespace-scoped sweep retires only
        // past-dated bounds → exactly the one past fact (future one stays open).
        let past_bound = now - Duration::days(1);
        let out_past = mem
            .supersede(past_fact)
            .in_namespace(ns.clone())
            .at(past_bound)
            .close_now()
            .execute()
            .await
            .expect("past supersede with close_now must succeed");
        assert_eq!(
            out_past,
            SupersedeOutcome::Bounded { retired: 1 },
            "close_now on a past-dated bound retires exactly the one closed window"
        );

        // Assert DB state: past fact retired (expired_at set), future fact NOT.
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let group_id = mem.group_id_for_test(&ns);
        let past = tg
            .get_fact_by_id(past_fact, &group_id)
            .await
            .expect("get past")
            .expect("past fact exists");
        assert!(
            past.expired_at.is_some(),
            "past-dated bound must be retired (expired_at set) by close_now"
        );
        let future = tg
            .get_fact_by_id(future_fact, &group_id)
            .await
            .expect("get future")
            .expect("future fact exists");
        assert_eq!(
            future.expired_at, None,
            "future-dated bound must be left un-retired (window not yet closed)"
        );
    }

    /// `.close_now()` on a FUTURE-dated bound returns `Bounded { retired: 0 }`
    /// (nothing to retire yet). The honest zero, observable in-band.
    #[tokio::test]
    async fn close_now_on_future_bound_retires_zero() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-supersede-close-now-future");
        let now = chrono::Utc::now();
        let valid_from = now - Duration::days(10);
        let fact_id = seed_fact(&mem, &ns, valid_from).await;

        let future_bound = now + Duration::days(5);
        let outcome = mem
            .supersede(fact_id)
            .in_namespace(ns.clone())
            .at(future_bound)
            .close_now()
            .execute()
            .await
            .expect("close_now on future bound must succeed");
        assert_eq!(
            outcome,
            SupersedeOutcome::Bounded { retired: 0 },
            "future-dated bound retires nothing yet — retired 0, deferred to a later sweep"
        );
    }

    /// AC — a `valid_to` predating the fact's own `valid_from` is rejected
    /// structurally; the DB row is UNCHANGED (no partial write).
    #[tokio::test]
    async fn supersede_rejects_time_inversion() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-supersede-inversion");
        let now = chrono::Utc::now();
        let valid_from = now - Duration::days(10);
        let fact_id = seed_fact(&mem, &ns, valid_from).await;

        let inverted_bound = valid_from - Duration::days(1);
        let outcome = mem
            .supersede(fact_id)
            .in_namespace(ns.clone())
            .at(inverted_bound)
            .execute()
            .await
            .expect("supersede must succeed (typed rejection, not Err)");

        assert_eq!(outcome, SupersedeOutcome::RejectedTimeInversion);

        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let group_id = mem.group_id_for_test(&ns);
        let fact = tg
            .get_fact_by_id(fact_id, &group_id)
            .await
            .expect("get_fact_by_id must succeed")
            .expect("fact must exist");
        assert_eq!(
            fact.valid_to, None,
            "DB row must be UNCHANGED — no partial write on time-inversion rejection"
        );
    }

    /// AC — supersede bounds `facts.valid_to`, a mutation of
    /// an existing row, so it MUST be blocked on AppendOnly namespaces
    /// ("valid_to set on facts" is a prohibited mutation there). Mirrors
    /// `forget.rs::by_source_id_appendonly_blocks_forget`. The DB row must stay
    /// unchanged.
    #[tokio::test]
    async fn supersede_appendonly_blocks_mutation() {
        use crate::memory::types::{ImmutabilityLevel, NamespacePolicy};

        let mem = make_memory().await;
        let ns_inner = Namespace::new("test-supersede-appendonly");
        let now = chrono::Utc::now();
        let valid_from = now - Duration::days(10);

        // Register AppendOnly (mandates forgettable=false + dream_eligible=false
        // for policy coherence), then seed a fact (INSERT is permitted on
        // AppendOnly — only mutation of existing rows is blocked).
        let policy = NamespacePolicy {
            immutability: ImmutabilityLevel::AppendOnly,
            forgettable: false,
            dream_eligible: false,
            ..NamespacePolicy::default()
        };
        let ns_with_policy = ns_inner.clone().with_policy(policy).expect("policy");
        mem.register_namespace(ns_with_policy)
            .await
            .expect("register_namespace must succeed");
        let fact_id = seed_fact(&mem, &ns_inner, valid_from).await;

        let result = mem
            .supersede(fact_id)
            .in_namespace(ns_inner.clone())
            .at(now - Duration::days(1))
            .execute()
            .await;

        let err = result.expect_err("AppendOnly must block supersede (a valid_to mutation)");
        let msg = err.to_string();
        assert!(
            msg.contains("policy") || msg.contains("Policy") || msg.contains("AppendOnly"),
            "error must reference the policy violation; got: {msg}"
        );

        // DB row unchanged — the blocked supersede wrote nothing.
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let group_id = mem.group_id_for_test(&ns_inner);
        let fact = tg
            .get_fact_by_id(fact_id, &group_id)
            .await
            .expect("get_fact_by_id must succeed")
            .expect("fact must exist");
        assert_eq!(
            fact.valid_to, None,
            "AppendOnly-blocked supersede must not write valid_to"
        );
    }

    /// AC — a nonexistent `fact_id` returns a typed `NotFound`, not an `Err`.
    #[tokio::test]
    async fn supersede_not_found_for_missing_fact_id() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-supersede-not-found");
        // No fact seeded — namespace still needs to exist for resolve to work,
        // but get_fact_by_id legitimately returns None.
        let outcome = mem
            .supersede(999_999)
            .in_namespace(ns)
            .at(chrono::Utc::now())
            .execute()
            .await
            .expect("supersede must succeed even with no match");

        assert_eq!(outcome, SupersedeOutcome::NotFound);
    }
}
