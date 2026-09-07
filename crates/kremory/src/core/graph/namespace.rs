use metrics::histogram;
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::TemporalGraph;

impl TemporalGraph {
    /// Read the persisted [`crate::memory::types::NamespacePolicy`] for a
    /// `group_id`, or `None` if the namespace has not been observed.
    ///
    /// Used by `register_namespace` to detect idempotency vs immutable-conflict.
    /// Substrate-only.
    pub(crate) async fn get_namespace_policy(
        &self,
        group_id: &str,
    ) -> Result<Option<crate::memory::types::NamespacePolicy>> {
        let started = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT policy_json FROM namespaces WHERE group_id = ? LIMIT 1",
                libsql::params![group_id.to_string()],
            )
            .await?;
        let policy = if let Some(row) = rows.next().await? {
            let json: String = row.get(0)?;
            let parsed: crate::memory::types::NamespacePolicy = serde_json::from_str(&json)?;
            Some(parsed)
        } else {
            None
        };
        histogram!("kremory_core_namespace_policy_get_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(policy)
    }

    /// Write a `NamespacePolicy` for a `group_id` if not already present.
    ///
    /// Substrate-only. Caller (`register_namespace`) holds the BEGIN IMMEDIATE
    /// guard for race safety; this method does NOT manage its own transaction.
    /// `ON CONFLICT DO NOTHING` keeps the operation idempotent at the SQL level
    /// when called concurrently.
    pub(crate) async fn set_namespace_policy(
        &self,
        group_id: &str,
        policy: &crate::memory::types::NamespacePolicy,
    ) -> Result<()> {
        let started = Instant::now();
        let json = serde_json::to_string(policy)?;
        self.conn
            .execute(
                "INSERT INTO namespaces (group_id, policy_json, schema_version) \
                 VALUES (?, ?, 1) \
                 ON CONFLICT(group_id) DO NOTHING",
                libsql::params![group_id.to_string(), json],
            )
            .await?;
        histogram!("kremory_core_namespace_policy_set_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Write an updated `NamespacePolicy` for an existing `group_id`, stamping
    /// `upgraded_at = now()`. Used by `Memory::upgrade_namespace_policy` for
    /// the monotonic Mutable → AppendOnly upgrade.
    ///
    /// Caller holds the `BEGIN IMMEDIATE` guard. This method does NOT open a
    /// transaction — it is meant to be called inside the caller's atomic block.
    ///
    /// Sets `upgraded_at` to the current UTC time in RFC3339 format.
    pub(crate) async fn set_namespace_policy_with_upgraded_at(
        &self,
        group_id: &str,
        policy: &crate::memory::types::NamespacePolicy,
    ) -> Result<()> {
        let started = Instant::now();
        let json = serde_json::to_string(policy)?;
        let now = chrono::Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO namespaces (group_id, policy_json, upgraded_at, schema_version) \
                 VALUES (?, ?, ?, 1) \
                 ON CONFLICT(group_id) DO UPDATE SET \
                   policy_json = excluded.policy_json, \
                   upgraded_at = excluded.upgraded_at",
                libsql::params![group_id.to_string(), json, now],
            )
            .await?;
        histogram!("kremory_core_namespace_policy_upgrade_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Ensure a default-policy row exists for `group_id` on implicit
    /// observation. Called by `remember`/`recall`/`forget`/`dream` on first
    /// encounter with a previously-unregistered namespace.
    ///
    /// # Race safety
    ///
    /// This method opens its own `BEGIN IMMEDIATE` guard via
    /// `begin_immediate_if_needed`, which is a no-op when nested under an
    /// existing outer transaction. The wrapping serializes against concurrent
    /// `register_namespace` calls. `INSERT OR IGNORE` is itself idempotent —
    /// the guard ensures the surrounding read-decide sequence in
    /// `register_namespace` stays consistent.
    pub(crate) async fn ensure_namespace_policy_row(&self, group_id: &str) -> Result<()> {
        let started = Instant::now();
        // Default policy serialized inline — matches NamespacePolicy::default().
        const DEFAULT_POLICY_JSON: &str =
            r#"{"immutability":"mutable","forgettable":true,"dream_eligible":true}"#;
        let guard = self.begin_immediate_if_needed().await?;
        let result = self
            .conn
            .execute(
                "INSERT OR IGNORE INTO namespaces (group_id, policy_json, schema_version) \
                 VALUES (?, ?, 1)",
                libsql::params![group_id.to_string(), DEFAULT_POLICY_JSON.to_string()],
            )
            .await;
        match result {
            Ok(_) => {
                guard.commit().await?;
                histogram!("kremory_core_namespace_policy_ensure_seconds")
                    .record(started.elapsed().as_secs_f64());
                Ok(())
            }
            Err(e) => {
                guard.rollback().await?;
                Err(e.into())
            }
        }
    }

    /// Read the namespace policy using the per-handle LRU cache.
    ///
    /// Cache hit: returns the cached policy immediately (no DB read).
    /// Cache miss: reads from `namespaces` table, populates cache, returns result.
    ///
    /// The cache is a per-handle LRU (capacity 256) so different `TemporalGraph`
    /// instances have independent caches — no cross-handle invalidation needed.
    pub(crate) async fn get_namespace_policy_cached(
        &self,
        group_id: &str,
    ) -> Result<Option<crate::memory::types::NamespacePolicy>> {
        // Cache hit — peek does not update LRU recency on a miss path.
        if let Some(policy) = self.policy_cache.peek(group_id) {
            return Ok(Some(policy));
        }
        // Cache miss — read from DB and populate.
        let policy = self.get_namespace_policy(group_id).await?;
        if let Some(ref p) = policy {
            self.policy_cache.put(group_id, p.clone());
        }
        Ok(policy)
    }

    /// Evict the policy cache entry for `group_id`.
    ///
    /// Called by `upgrade_namespace_policy` after committing the upgrade so
    /// subsequent reads reflect the new AppendOnly policy.
    pub(crate) fn invalidate_policy_cache(&self, group_id: &str) {
        self.policy_cache.invalidate(group_id);
    }
}
