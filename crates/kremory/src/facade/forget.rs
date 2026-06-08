use super::*;

// ── ForgetRequest ─────────────────────────────────────────────────────────────

/// Forget (delete) request builder. Obtain via `mem.forget()`.
///
/// Must call `.execute()` explicitly — this is a destructive operation.
pub struct ForgetRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) namespace: Option<Namespace>,
    /// G8 — narrow forget to entities derived from episodes with this
    /// `source_id`. Composes with `in_namespace`. Vera F17 shared-entity
    /// preservation applies: an entity is deleted only when all of its
    /// `episodic_edges` resolve to episodes matching the filter.
    pub(super) source_id: Option<String>,
}

impl<'a> ForgetRequest<'a> {
    /// Set the namespace scope for deletion (overrides Memory default).
    pub fn in_namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// G8 — narrow forget to entities derived from episodes matching this
    /// `source_id` (the column added in G1). Composes with `in_namespace`.
    ///
    /// # Shared-entity preservation (Vera F17)
    ///
    /// Entities referenced by ANY episode outside this `source_id` are NOT
    /// deleted. Only entities whose entire `episodic_edges` set falls within
    /// the matched episodes are removed. This prevents cross-source data loss
    /// when a single entity ("Acme Corp") is mentioned in multiple ingested
    /// documents.
    ///
    /// # AppendOnly enforcement
    ///
    /// Per ADR-029b §3.1, AppendOnly enforcement applies regardless of
    /// `by_source_id` scope — narrowing the forget set does not weaken the
    /// policy gate.
    pub fn by_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Execute the deletion. Returns the count of deleted entity rows.
    ///
    /// This is the only terminal for `ForgetRequest` — there is no implicit
    /// `.await` to prevent accidental destructive operations.
    ///
    /// # AppendOnly enforcement (ADR-029b §3.1)
    ///
    /// If the namespace has `AppendOnly` policy, returns
    /// `Err(MemoryError::Core(CoreError::NamespacePolicyViolation))`.
    pub async fn execute(self) -> Result<u64> {
        let ns = self.memory.resolve_namespace(self.namespace)?;
        // ADR-029a lazy population: ensure namespace row exists before read.
        self.memory.ensure_namespace_policy(&ns).await?;

        // ADR-029b §3.1: AppendOnly enforcement — forget is a mutation.
        let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::forget requires a Memory constructed via the builder/providers path \
                 (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let group_id = namespace_to_group_id(&ns);
        let policy = tg
            .get_namespace_policy_cached(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        if let Some(p) = &policy {
            if p.immutability == crate::memory::types::ImmutabilityLevel::AppendOnly {
                // ADR-029b §3.1 enforcement — v0.1.5 closure of the v0.1.4
                // declare-but-don't-enforce contract. ForgetRequest is a
                // mutating operation and is prohibited on AppendOnly
                // namespaces. Returns the canonical CoreError variant so
                // callers can pattern-match on the policy violation.
                return Err(MemoryError::Core(
                    crate::core::error::Error::NamespacePolicyViolation {
                        namespace: group_id.clone(),
                        operation: "forget".to_string(),
                        policy: p.clone(),
                    },
                ));
            }
        }

        // G8 — narrowed source_id forget path. Walks the
        // episodes(source_id) → episodic_edges(episode_id, entity_id) chain to
        // collect candidate entities, then applies Vera F17 shared-entity
        // preservation: an entity is deleted only when EVERY one of its
        // episodic_edges row falls inside the matched episode set. Entities
        // with edges to any episode outside the filter are pinned.
        if let Some(sid) = self.source_id {
            let conn = &tg.conn;
            // Step 1: candidate entity_ids = entities that have at least one
            // edge to an episode matching (source_id, group_id).
            let mut cand_rows = conn
                .query(
                    "SELECT DISTINCT ee.entity_id \
                     FROM episodic_edges ee \
                     JOIN episodes e ON e.id = ee.episode_id \
                     WHERE e.source_id = ?1 AND e.group_id = ?2",
                    libsql::params![sid.clone(), group_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?;
            let mut candidates: Vec<String> = Vec::new();
            while let Some(row) = cand_rows.next().await.map_err(CoreError::Database)? {
                candidates.push(row.get::<String>(0).map_err(CoreError::Database)?);
            }
            // Quinn C2 — N+1 visibility: log candidate count so a regression
            // (e.g. document with 200+ entities) surfaces in tracing before
            // the v0.1.7 SQL pre-filter promotion lands.
            tracing::debug!(
                source_id = %sid,
                namespace = %group_id,
                candidate_count = candidates.len(),
                "forget by_source_id: candidate entities collected"
            );
            // Step 2: pin any candidate that has ANY edge to an episode
            // OUTSIDE the matched set (Vera F17 shared-entity preservation).
            let mut to_delete: Vec<String> = Vec::with_capacity(candidates.len());
            for entity_id in candidates {
                let mut count_rows = conn
                    .query(
                        // De Morgan: NOT (source_id = ?2 AND group_id = ?3)
                        // expands to (source_id IS NULL OR source_id != ?2 OR
                        // group_id != ?3). NULL guard is load-bearing —
                        // episodes seeded before G1 land with source_id=NULL
                        // and must count as "outside" the filter. DO NOT
                        // "simplify" this clause without re-running the Vera
                        // F17 shared-entity preservation tests.
                        "SELECT COUNT(*) FROM episodic_edges ee \
                         JOIN episodes e ON e.id = ee.episode_id \
                         WHERE ee.entity_id = ?1 \
                           AND (e.source_id IS NULL OR e.source_id != ?2 OR e.group_id != ?3)",
                        libsql::params![entity_id.clone(), sid.clone(), group_id.clone()],
                    )
                    .await
                    .map_err(CoreError::Database)?;
                let outside_count: i64 = count_rows
                    .next()
                    .await
                    .map_err(CoreError::Database)?
                    .ok_or_else(|| {
                        MemoryError::Other(
                            "forget by_source_id: COUNT(*) returned no rows".to_string(),
                        )
                    })?
                    .get::<i64>(0)
                    .map_err(CoreError::Database)?;
                if outside_count == 0 {
                    to_delete.push(entity_id);
                }
            }
            let entities_deleted = if to_delete.is_empty() {
                0
            } else {
                tg.batch_forget(&to_delete)
                    .await
                    .map_err(MemoryError::Core)?
            };

            // Quinn C3 — spec §G8 says "only the episode row(s) AND edges
            // exclusively owned by this source_id are removed". Episode rows
            // are 1:1 with source_id (not shared across consumers), so
            // delete the matched episode rows after entity cleanup. The
            // FK (episodic_edges.episode_id → episodes.id) means we must
            // also drop any remaining episodic_edges pointing to these
            // episodes first (entities sharing with other sources stayed
            // pinned, but their edges to THIS source's episodes go).
            conn.execute(
                "DELETE FROM episodic_edges WHERE episode_id IN \
                 (SELECT id FROM episodes WHERE source_id = ?1 AND group_id = ?2)",
                libsql::params![sid.clone(), group_id.clone()],
            )
            .await
            .map_err(CoreError::Database)?;
            conn.execute(
                "DELETE FROM episodes WHERE source_id = ?1 AND group_id = ?2",
                libsql::params![sid, group_id],
            )
            .await
            .map_err(CoreError::Database)?;
            return Ok(entities_deleted);
        }

        // Wire to substrate: list entities in group, then batch_forget.
        let entities = tg
            .list_entities_in_group(&group_id)
            .await
            .map_err(MemoryError::Core)?;
        if entities.is_empty() {
            return Ok(0);
        }
        let ids: Vec<String> = entities.into_iter().map(|e| e.id).collect();
        tg.batch_forget(&ids).await.map_err(MemoryError::Core)
    }
}

#[cfg(test)]
mod forget_by_source_id_tests {
    use std::sync::Arc;

    use crate::core::provider::{DynEmbeddingProvider, MockChatProvider, NullEmbeddingProvider};
    use crate::memory::types::Namespace;

    use super::Memory;

    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn DynEmbeddingProvider> = Arc::new(NullEmbeddingProvider { dim: 384 });
        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("Memory must build")
    }

    /// Seed: episode + entity + episodic_edge. Returns the inserted
    /// episode_id (SQLite AUTOINCREMENT).
    async fn seed_link(mem: &Memory, source_id: &str, ns: &Namespace, entity_id: &str) -> i64 {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;

        // Insert entity (idempotent via INSERT OR IGNORE on PRIMARY KEY).
        // Migration 009 dropped entities.label — use entity_type_id=0 (catch-all).
        conn.execute(
            "INSERT OR IGNORE INTO entities (id, entity_type_id, properties, recorded_at, group_id) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                entity_id,
                0i64,
                "{}",
                "2026-05-29T12:00:00Z",
                ns.namespace.as_str()
            ],
        )
        .await
        .expect("entity seed must succeed");

        // Insert episode with explicit source_id.
        conn.execute(
            "INSERT INTO episodes (content, timestamp, source_type, metadata, group_id, source_id, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            libsql::params![
                "seed",
                "2026-05-29T12:00:00Z",
                "Document",
                None::<String>,
                ns.namespace.as_str(),
                source_id,
                "2026-05-29T12:00:00Z"
            ],
        )
        .await
        .expect("episode seed must succeed");

        // Recover the inserted episode id.
        let mut rows = conn
            .query(
                "SELECT id FROM episodes WHERE source_id = ?1 AND group_id = ?2 \
                 ORDER BY id DESC LIMIT 1",
                libsql::params![source_id, ns.namespace.as_str()],
            )
            .await
            .expect("episode id lookup must succeed");
        let episode_id: i64 = rows
            .next()
            .await
            .expect("episode id row_next must succeed")
            .expect("episode id row must exist")
            .get::<i64>(0)
            .expect("episode id column");

        // Wire the episodic_edge. Migration 006 enforces composite FK
        // (entity_id, entity_group_id) → entities(id, group_id), so we must
        // bind entity_group_id explicitly.
        conn.execute(
            "INSERT INTO episodic_edges (episode_id, entity_id, entity_group_id, role, recorded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                episode_id,
                entity_id,
                ns.namespace.as_str(),
                "mentioned",
                "2026-05-29T12:00:00Z"
            ],
        )
        .await
        .expect("episodic_edge seed must succeed");

        episode_id
    }

    async fn entity_exists(mem: &Memory, entity_id: &str) -> bool {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        let mut rows = conn
            .query(
                "SELECT 1 FROM entities WHERE id = ?1",
                libsql::params![entity_id],
            )
            .await
            .expect("entity exist check must succeed");
        rows.next().await.expect("row_next must succeed").is_some()
    }

    /// AC.8 — happy path: entity referenced only by source_A is forgotten
    /// when forget(by_source_id="A") runs; entity referenced only by source_B
    /// is preserved.
    #[tokio::test]
    async fn by_source_id_deletes_only_matched_unique_entities() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-unique");
        seed_link(&mem, "doc-A", &ns, "entity-alpha").await;
        seed_link(&mem, "doc-B", &ns, "entity-beta").await;

        let deleted = mem
            .forget()
            .in_namespace(ns.clone())
            .by_source_id("doc-A")
            .execute()
            .await
            .expect("forget by_source_id must succeed");

        assert_eq!(deleted, 1, "only entity-alpha must be deleted");
        assert!(
            !entity_exists(&mem, "entity-alpha").await,
            "alpha must be gone"
        );
        assert!(entity_exists(&mem, "entity-beta").await, "beta must remain");
    }

    /// AC.8 / Vera F17 — shared-entity preservation: entity referenced by
    /// BOTH source_A and source_B must NOT be deleted when only source_A is
    /// forgotten.
    #[tokio::test]
    async fn by_source_id_preserves_shared_entity_vera_f17() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-shared");
        // Same entity linked from two distinct sources.
        seed_link(&mem, "doc-A", &ns, "entity-shared").await;
        seed_link(&mem, "doc-B", &ns, "entity-shared").await;

        let deleted = mem
            .forget()
            .in_namespace(ns.clone())
            .by_source_id("doc-A")
            .execute()
            .await
            .expect("forget by_source_id must succeed");

        assert_eq!(
            deleted, 0,
            "shared entity must NOT be deleted (Vera F17 preservation)"
        );
        assert!(
            entity_exists(&mem, "entity-shared").await,
            "shared entity must remain (still referenced by doc-B)"
        );
    }

    async fn episode_count_for(mem: &Memory, source_id: &str, ns: &Namespace) -> i64 {
        let tg = mem.temporal_graph.as_ref().expect("temporal_graph");
        let conn = &tg.conn;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM episodes WHERE source_id = ?1 AND group_id = ?2",
                libsql::params![source_id, ns.namespace.as_str()],
            )
            .await
            .expect("episode count must succeed");
        rows.next()
            .await
            .expect("row_next")
            .expect("row")
            .get::<i64>(0)
            .expect("count column")
    }

    /// AC.8 / Quinn C3 — episode rows for the matched source_id are
    /// removed (not just entities). Spec: "only the episode row(s) and
    /// edges exclusively owned by this source_id are removed."
    #[tokio::test]
    async fn by_source_id_deletes_matched_episode_rows() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-episodes");
        seed_link(&mem, "doc-A", &ns, "entity-foo").await;
        seed_link(&mem, "doc-B", &ns, "entity-bar").await;

        assert_eq!(episode_count_for(&mem, "doc-A", &ns).await, 1);
        assert_eq!(episode_count_for(&mem, "doc-B", &ns).await, 1);

        mem.forget()
            .in_namespace(ns.clone())
            .by_source_id("doc-A")
            .execute()
            .await
            .expect("forget must succeed");

        assert_eq!(
            episode_count_for(&mem, "doc-A", &ns).await,
            0,
            "doc-A episodes must be deleted (Quinn C3)"
        );
        assert_eq!(
            episode_count_for(&mem, "doc-B", &ns).await,
            1,
            "doc-B episodes must be preserved (out of scope)"
        );
    }

    /// AC.8 / Quinn C4 — AppendOnly policy gate still fires when
    /// `by_source_id` is set. Narrowing scope does not weaken the gate.
    #[tokio::test]
    async fn by_source_id_appendonly_blocks_forget() {
        use crate::memory::types::{ImmutabilityLevel, NamespacePolicy};

        let mem = make_memory().await;
        let ns_inner = Namespace::new("test-g8-appendonly");

        // Register AppendOnly policy. AppendOnly mandates forgettable=false
        // AND dream_eligible=false (coherence check rejects mutating ops).
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

        // Seed an episode so the source_id path has a candidate (even
        // though the policy gate fires first and the entity stays).
        seed_link(&mem, "doc-A", &ns_inner, "entity-zeta").await;

        let result = mem
            .forget()
            .in_namespace(ns_inner)
            .by_source_id("doc-A")
            .execute()
            .await;

        let err = result.expect_err("AppendOnly must block forget by_source_id");
        let msg = err.to_string();
        assert!(
            msg.contains("policy") || msg.contains("Policy") || msg.contains("AppendOnly"),
            "error must reference policy violation; got: {msg}"
        );
    }

    /// AC.8 — unknown source_id returns 0, no Err.
    #[tokio::test]
    async fn by_source_id_no_match_returns_zero() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g8-empty");
        seed_link(&mem, "doc-A", &ns, "entity-x").await;

        let deleted = mem
            .forget()
            .in_namespace(ns.clone())
            .by_source_id("does-not-exist-zzz")
            .execute()
            .await
            .expect("forget by_source_id must succeed even with no matches");

        assert_eq!(deleted, 0, "no match must return 0, not Err");
        assert!(
            entity_exists(&mem, "entity-x").await,
            "non-matched entity must remain"
        );
    }
}
