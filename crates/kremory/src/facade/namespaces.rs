//! Namespace registration + policy on `Memory` (TD-043 split out of
//! `facade/mod.rs`).
//!
//! A namespace is the TENANCY boundary, so everything here is operator-side: create
//! one, seed its entity types, upgrade its immutability policy, and the lazy
//! `ensure_namespace_policy` every mutating request calls before it writes. None of
//! it is reachable from the MCP tool surface, deliberately — see
//! `crates/kremory-mcp/surface-decisions.toml`.

use super::*;

impl Memory {
    /// Register a namespace + its policy explicitly, ahead of any writes.
    ///
    /// # Idempotency
    ///
    /// Calling `register_namespace` with the SAME `(group_id, policy)` pair
    /// returns `Ok(())`. Calling with a DIFFERENT policy on an existing
    /// `group_id` returns
    /// `Err(MemoryError::Core(Error::NamespacePolicyImmutable { ... }))`.
    /// This makes startup code safe to re-execute (idempotent against
    /// persisted state).
    ///
    /// # Validation
    ///
    /// The policy attached to `namespace` is validated via
    /// [`NamespacePolicy::validate`] before persistence. Incoherent policies
    /// surface as `Err(MemoryError::Core(Error::InvalidPolicy(...)))`.
    ///
    /// # No enforcement at v0.1.4
    ///
    /// The policy is PERSISTED but not yet enforced on
    /// `dream()` / `forget()` / mutation operations. Enforcement lands in
    /// v0.1.5+. Every non-default policy registration emits
    /// a `tracing::warn!` on target `kremory.namespace` to make the
    /// declaration vs enforcement gap visible.
    ///
    /// # Race semantics — atomic via `BEGIN IMMEDIATE`
    ///
    /// `register_namespace` wraps the SELECT + INSERT pair in a
    /// `BEGIN IMMEDIATE` transaction (the write_lock invariant). This
    /// acquires SQLite's RESERVED write lock before reading, serializing
    /// against concurrent `remember(...)` calls that would implicitly create
    /// the namespace with default policy.
    ///
    /// The recommended pattern is `register_namespace` AT STARTUP before any
    /// `remember(...)`. See the match arms below for the three race outcomes.
    pub async fn register_namespace(&self, namespace: Namespace) -> Result<()> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::register_namespace requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let policy = namespace.policy.clone().unwrap_or_default();
        policy
            .validate()
            .map_err(|e| MemoryError::Core(CoreError::InvalidPolicy(e)))?;

        let group_id = namespace_to_group_id(&namespace);
        let is_non_default = policy != NamespacePolicy::default();

        // Atomic INSERT-or-compare via BEGIN IMMEDIATE.
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(MemoryError::Core)?;
        let stored = tg
            .get_namespace_policy(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        let outcome: Result<()> = match stored {
            Some(existing) if existing == policy => Ok(()),
            Some(existing) => Err(MemoryError::Core(CoreError::NamespacePolicyImmutable {
                namespace: group_id.clone(),
                stored: existing,
                attempted: policy.clone(),
            })),
            None => tg
                .set_namespace_policy(&group_id, &policy)
                .await
                .map_err(MemoryError::Core),
        };
        match &outcome {
            Ok(()) => {
                guard.commit().await.map_err(MemoryError::Core)?;
            }
            Err(_) => {
                guard.rollback().await.map_err(MemoryError::Core)?;
            }
        }

        // Operational visibility: every non-default policy DECLARATION emits
        // warn (NOT info). Default
        // policies are silent (they would be the existing behaviour).
        if outcome.is_ok() && is_non_default {
            tracing::warn!(
                target: "kremory.namespace",
                group_id = %group_id,
                policy = ?policy,
                "kremory.namespace.policy_declared: POLICY DECLARED BUT NOT \
                 ENFORCED at v0.1.4 — enforcement lands v0.1.5+. \
                 See https://docs.rs/kremory/0.1.4/kremory/#adr-029a"
            );
        }

        outcome
    }

    /// Register a namespace policy + seed its entity-type registry in one atomic
    /// operation (spec custom-entity-type-registry §5.2.2).
    ///
    /// Seeds the namespace's `entity_types` table per the `seed` instruction,
    /// and registers the (default) namespace policy — both inside the SAME
    /// `BEGIN IMMEDIATE` transaction (no TOCTOU window between policy write and
    /// seed write; §5.8 invariant).
    ///
    /// # Semantics (D9 / D9a — brownfield safety)
    ///
    /// | namespace state         | `Default` / `Augment` | `Replace`                              |
    /// |-------------------------|-----------------------|----------------------------------------|
    /// | no rows (fresh)         | seed → `Seeded`       | seed → `Seeded`                        |
    /// | has rows, seed MATCHES  | `AlreadySeeded`       | `AlreadySeeded` (D9a, idempotent boot) |
    /// | has rows, seed DIFFERS  | `AlreadySeeded`       | `Err(AlreadyPopulated { group_id })`   |
    ///
    /// - id=0 "Entity" catch-all is ALWAYS present after a successful call.
    /// - `Replace` NEVER mutates a populated namespace — it fails loud with NO
    ///   DB write, so existing entities' `entity_type_id` can never be orphaned
    ///   (ASMP-003). To add types to an already-populated namespace use
    ///   [`assert_entity_type`](Self::assert_entity_type) (the sanctioned
    ///   incremental-add path).
    ///
    /// This is the PREFERRED startup pattern for domain-specific namespaces:
    /// call at startup BEFORE the first `remember()` for the target namespace.
    /// The existing [`register_namespace`](Self::register_namespace) remains for
    /// callers that want the default seed.
    pub async fn register_namespace_with_seed(
        &self,
        namespace: Namespace,
        seed: crate::core::entity_types::NamespaceSeed,
    ) -> std::result::Result<
        crate::core::entity_types::SeedOutcome,
        crate::core::entity_types::NamespaceRegistrationError,
    > {
        use crate::core::entity_types::NamespaceRegistrationError;

        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            NamespaceRegistrationError::Store(CoreError::Other(anyhow::anyhow!(
                "Memory::register_namespace_with_seed requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
            )))
        })?;

        let policy = namespace.policy.clone().unwrap_or_default();
        policy
            .validate()
            .map_err(|e| NamespaceRegistrationError::Store(CoreError::InvalidPolicy(e)))?;

        let group_id = namespace_to_group_id(&namespace);
        let is_non_default = policy != NamespacePolicy::default();

        // Single BEGIN IMMEDIATE wrapping: namespace policy write + seed
        // application. Presence check + seed live inside this txn (no TOCTOU).
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(NamespaceRegistrationError::Store)?;

        let outcome: std::result::Result<
            crate::core::entity_types::SeedOutcome,
            NamespaceRegistrationError,
        > = async {
            // Policy: INSERT-or-compare (mirrors register_namespace).
            let stored = tg
                .get_namespace_policy(&group_id)
                .await
                .map_err(NamespaceRegistrationError::Store)?;
            match stored {
                Some(existing) if existing == policy => {}
                Some(existing) => {
                    return Err(NamespaceRegistrationError::Store(
                        CoreError::NamespacePolicyImmutable {
                            namespace: group_id.clone(),
                            stored: existing,
                            attempted: policy.clone(),
                        },
                    ));
                }
                None => {
                    tg.set_namespace_policy(&group_id, &policy)
                        .await
                        .map_err(NamespaceRegistrationError::Store)?;
                }
            }

            // Seed (D9 / D9a) inside the same txn.
            crate::core::entity_types::apply_namespace_seed(&tg.conn, &group_id, &seed).await
        }
        .await;

        match &outcome {
            Ok(_) => {
                guard
                    .commit()
                    .await
                    .map_err(NamespaceRegistrationError::Store)?;
            }
            Err(_) => {
                guard
                    .rollback()
                    .await
                    .map_err(NamespaceRegistrationError::Store)?;
            }
        }

        if outcome.is_ok() && is_non_default {
            tracing::warn!(
                target: "kremory.namespace",
                group_id = %group_id,
                policy = ?policy,
                "kremory.namespace.policy_declared: POLICY DECLARED BUT NOT \
                 ENFORCED at v0.1.4 — enforcement lands v0.1.5+."
            );
        }

        outcome
    }

    /// Monotonically upgrade a namespace's immutability from `Mutable` to
    /// `AppendOnly`.
    ///
    /// This is a **one-way ratchet**: `Mutable → AppendOnly` is the only
    /// allowed direction. Attempting to downgrade (`AppendOnly → Mutable`)
    /// returns `Err(MemoryError::Core(Error::NamespacePolicyImmutable))`.
    /// Calling on an already-`AppendOnly` namespace is idempotent (`Ok(())`).
    ///
    /// # Atomicity
    ///
    /// The read-decide-write sequence is wrapped in a `BEGIN IMMEDIATE`
    /// transaction to prevent races with concurrent `register_namespace` or
    /// `upgrade_namespace_policy` calls.
    ///
    /// # Errors
    ///
    /// - `MemoryError::Other` — `Memory` not constructed via the builder path.
    /// - `MemoryError::Core(Error::NamespacePolicyImmutable)` — downgrade
    ///   attempted or policy mismatch.
    /// - `MemoryError::Core(Error::Other)` — substrate failure.
    pub async fn upgrade_namespace_policy(&self, namespace: Namespace) -> Result<()> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::upgrade_namespace_policy requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;

        let group_id = namespace_to_group_id(&namespace);
        let guard = tg
            .begin_immediate_if_needed()
            .await
            .map_err(MemoryError::Core)?;

        let stored = tg
            .get_namespace_policy(&group_id)
            .await
            .map_err(MemoryError::Core)?;

        let target_policy = crate::memory::types::NamespacePolicy::APPEND_ONLY;

        let outcome: Result<()> = match stored {
            Some(ref existing)
                if existing.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly =>
            {
                // Already AppendOnly — idempotent.
                Ok(())
            }
            Some(ref existing)
                if existing.immutability == crate::memory::types::ImmutabilityLevel::Mutable =>
            {
                // Upgrade Mutable → AppendOnly.
                tg.set_namespace_policy_with_upgraded_at(&group_id, &target_policy)
                    .await
                    .map_err(MemoryError::Core)
            }
            Some(existing) => {
                // Unexpected policy state — treat as immutable conflict.
                Err(MemoryError::Core(CoreError::NamespacePolicyImmutable {
                    namespace: group_id.clone(),
                    stored: existing,
                    attempted: target_policy,
                }))
            }
            None => {
                // Namespace not yet registered — create directly as AppendOnly.
                tg.set_namespace_policy_with_upgraded_at(&group_id, &target_policy)
                    .await
                    .map_err(MemoryError::Core)
            }
        };

        match &outcome {
            Ok(()) => {
                guard.commit().await.map_err(MemoryError::Core)?;
                // Invalidate cache so next read reflects the new AppendOnly policy.
                tg.invalidate_policy_cache(&group_id);
                tracing::info!(
                    target: "kremory.namespace",
                    group_id = %group_id,
                    "kremory.namespace.policy_upgraded: namespace upgraded to AppendOnly"
                );
            }
            Err(_) => {
                guard.rollback().await.map_err(MemoryError::Core)?;
            }
        }
        outcome
    }

    /// Lazy-population helper: ensure a default-policy row exists for the
    /// `namespace` if it has not been observed yet. Invoked from the first-
    /// encounter paths (`remember`, `recall`, `forget`, `dream`).
    ///
    /// Best-effort: when `Memory` is constructed without a direct
    /// `Arc<TemporalGraph>` (e.g. test-only stub-handle path) this is a no-op.
    /// Errors from the substrate are converted to `MemoryError::Core` and
    /// returned so the call site can decide whether to fail the user request.
    pub(crate) async fn ensure_namespace_policy(&self, namespace: &Namespace) -> Result<()> {
        let Some(tg) = self.temporal_graph.as_ref() else {
            return Ok(());
        };
        let group_id = namespace_to_group_id(namespace);
        tg.ensure_namespace_policy_row(&group_id)
            .await
            .map_err(MemoryError::Core)
    }
}
