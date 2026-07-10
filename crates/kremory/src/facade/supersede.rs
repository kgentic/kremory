use metrics::counter;

use super::*;

// ── SupersedeRequest ─────────────────────────────────────────────────────────

/// Consumer-facing supersede builder (**ADR-071 §Item 3**, TD-070). Obtain via
/// `mem.supersede(fact_id)`.
///
/// Bounds a fact's WORLD-time `valid_to` window explicitly — the consumer
/// asserts "this fact's validity ends here" (e.g. a document was superseded by
/// a newer version, a price changed, a status expired). This is the
/// consumer-EXPLICIT half of TD-070/065's supersession gap; auto-detected
/// supersession (the system inferring a supersession from new input) is
/// deliberately deferred (tracked as TD-P1-AUTO — not built here).
///
/// # Mechanism (**Amendment C** —
/// `.ai-docs/adrs/adr-071-adr070-reconciliation-amendment-2026-07-09.md`)
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
    /// predicate — Amendment C point 4).
    pub(super) reason: Option<String>,
}

impl<'a> SupersedeRequest<'a> {
    /// Set the bounded `valid_to` timestamp — REQUIRED before `.execute()`.
    pub fn at(mut self, valid_to: DateTime<Utc>) -> Self {
        self.valid_to = Some(valid_to);
        self
    }

    /// Optional human-readable reason, threaded as a tracing field only —
    /// never a metric label (cardinality discipline, ADR-071 §Item 5).
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the namespace scope for this supersede (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Execute the supersession. Bounds `fact_id`'s world-time `valid_to` via
    /// [`TemporalGraph::bound_valid_to`] (ADR-071 §Item 3, Amendment C) — NOT
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
                "SupersedeRequest::execute requires .at(valid_to) — no silent default \
                 (ADR-071 §Item 3 required-input discipline)"
                    .into(),
            )
        })?;

        let ns = self.memory.resolve_namespace(self.namespace)?;
        // ADR-029a lazy population: ensure namespace row exists before read.
        self.memory.ensure_namespace_policy(&ns).await?;
        let group_id = namespace_to_group_id(&ns);

        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::supersede requires a Memory constructed via the builder/providers \
                 path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        // AppendOnly enforcement (ADR-029b §3.1 — "valid_to set on facts" is a
        // named mutation): supersede bounds `facts.valid_to`, a mutation of an
        // existing row, and is prohibited on AppendOnly namespaces. Mirrors
        // `ForgetRequest::execute` (forget.rs) — Quinn Item-3 HIGH. Without this
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

        // Time-inversion guard (§3b step 3): a bound predating the fact's own
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

        // Amendment C: bound_valid_to, NOT invalidate_fact — writes `valid_to`
        // (world-time), not `expired_at` (system-time). The dream supersession
        // sweep's `window_closeout` does the system-time close later.
        tg.bound_valid_to(self.fact_id, valid_to)
            .await
            .map_err(MemoryError::Core)?;

        // fact_id/group_id/reason are KREMORY_DEBUG-gated tracing FIELDS only —
        // never counter labels (cardinality discipline, ADR-071 §Item 5).
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
            "outcome" => "applied"
        )
        .increment(1);

        Ok(SupersedeOutcome::Applied)
    }
}

/// Typed outcome of a [`SupersedeRequest::execute`] call — NOT a bare
/// bool/count, per `llm-output-parse-loudly`'s "required fields have no
/// silent default" discipline extended to consumer-facing typed results: a
/// caller must be able to distinguish WHY a supersede didn't apply, not just
/// that it didn't. Exactly 3 variants, 1:1 with the `outcome` label values on
/// `kremory.dream.consolidation.supersede_request_total` (ADR-071 §Item 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupersedeOutcome {
    /// `valid_to` was bounded successfully.
    Applied,
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
    async fn seed_fact(mem: &Memory, ns: &Namespace, valid_from: chrono::DateTime<chrono::Utc>) -> i64 {
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

        assert_eq!(outcome, SupersedeOutcome::Applied);

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

    /// AC (Quinn Item-3 HIGH) — supersede bounds `facts.valid_to`, a mutation of
    /// an existing row, so it MUST be blocked on AppendOnly namespaces (ADR-029b
    /// §3.1 names "valid_to set on facts" as a prohibited mutation). Mirrors
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
