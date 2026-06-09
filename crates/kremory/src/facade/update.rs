use super::*;

// ── UpdateSourceUriRequest ────────────────────────────────────────────────────

/// Request to update an episode's `source_uri`. Built via [`Memory::update_source_uri`].
///
/// Must call `.to(new_uri)` before `.await`. Errors if no episodes match the
/// given `source_id`. Does NOT mutate facts, bi-temporal axes (`valid_from`,
/// `valid_to`, `recorded_at`), or other episode columns.
///
/// # Substrate-purity
///
/// `source_id` and `source_uri` are substrate-generic universal identifiers —
/// chat-grain, doc-grain, and code-grain consumers all use the same surface.
#[must_use = "UpdateSourceUriRequest must be .await-ed after calling .to(new_uri)"]
pub struct UpdateSourceUriRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) source_id: String,
    pub(super) new_uri: Option<String>,
}

impl<'a> UpdateSourceUriRequest<'a> {
    /// Set the new `source_uri` value.
    pub fn to(mut self, new_uri: impl Into<String>) -> Self {
        self.new_uri = Some(new_uri.into());
        self
    }
}

impl<'a> IntoFuture for UpdateSourceUriRequest<'a> {
    type Output = Result<u64>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let new_uri = self.new_uri.ok_or_else(|| {
                MemoryError::Other(
                    "update_source_uri: .to(new_uri) must be called before .await".into(),
                )
            })?;
            let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
                MemoryError::Other(
                    "Memory::update_source_uri requires a Memory constructed via \
                     the builder/providers path (no Arc<TemporalGraph> attached)"
                        .into(),
                )
            })?;
            let conn = &tg.conn;
            // Verify at least one episode with this source_id exists.
            let count: i64 = conn
                .query(
                    "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
                    libsql::params![self.source_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?
                .next()
                .await
                .map_err(CoreError::Database)?
                .ok_or_else(|| {
                    MemoryError::Other(
                        "update_source_uri: COUNT query returned no rows".to_string(),
                    )
                })?
                .get(0)
                .map_err(CoreError::Database)?;
            if count == 0 {
                return Err(MemoryError::Other(format!(
                    "update_source_uri: no episode found with source_id={}",
                    self.source_id
                )));
            }
            let updated = conn
                .execute(
                    "UPDATE episodes SET source_uri = ?1 WHERE source_id = ?2",
                    libsql::params![new_uri, self.source_id],
                )
                .await
                .map_err(CoreError::Database)?;
            Ok(updated)
        })
    }
}

// ── Memory::update_source_uri entry point ─────────────────────────────────────

impl Memory {
    /// Update the `source_uri` for all episode(s) with the given `source_id`.
    ///
    /// Does NOT mutate facts, bi-temporal axes (`valid_from`, `valid_to`,
    /// `recorded_at`), or any other episode column. Returns `Err` if no
    /// episodes match the given `source_id`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kremory::Memory;
    /// # async fn ex(mem: Memory) -> kremory::memory::Result<()> {
    /// let n = mem.update_source_uri("doc-abc").to("path/v2").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "UpdateSourceUriRequest must be .await-ed after calling .to(new_uri)"]
    pub fn update_source_uri<'a>(
        &'a self,
        source_id: impl Into<String> + 'a,
    ) -> UpdateSourceUriRequest<'a> {
        UpdateSourceUriRequest {
            memory: self,
            source_id: source_id.into(),
            new_uri: None,
        }
    }
}

// ── UpdateEpisodeMetadataRequest builder ──────────────────────────────────────

/// Request to shallow-merge a JSON patch into episode metadata.
/// Built via [`Memory::update_episode_metadata`]. See that method for
/// merge semantics (shallow, arrays-replace, non-object rejected).
pub struct UpdateEpisodeMetadataRequest<'a> {
    pub(super) memory: &'a Memory,
    pub(super) source_id: String,
    pub(super) patch: Option<serde_json::Value>,
    pub(super) pending_error: Option<MemoryError>,
}

impl<'a> UpdateEpisodeMetadataRequest<'a> {
    /// Set the JSON patch to merge.
    ///
    /// The patch MUST be a JSON object (`Value::Object(_)`). Non-object
    /// patches (numbers, strings, arrays, booleans, null) cause `.await`
    /// to return `Err(MemoryError::Other(...))`.
    pub fn patch(mut self, patch: serde_json::Value) -> Self {
        if !patch.is_object() {
            self.pending_error = Some(MemoryError::Other(format!(
                "update_episode_metadata: patch must be a JSON object, got: {patch}"
            )));
        } else {
            self.patch = Some(patch);
        }
        self
    }
}

impl<'a> std::future::IntoFuture for UpdateEpisodeMetadataRequest<'a> {
    type Output = Result<usize>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            if let Some(err) = self.pending_error {
                return Err(err);
            }
            let patch_value = self.patch.ok_or_else(|| {
                MemoryError::Other(
                    "update_episode_metadata: .patch(value) must be called before .await"
                        .to_string(),
                )
            })?;
            let tg = self.memory.temporal_graph.as_ref().ok_or_else(|| {
                MemoryError::Other(
                    "Memory::update_episode_metadata requires a Memory constructed via \
                     the builder/providers path (no Arc<TemporalGraph> attached)"
                        .to_string(),
                )
            })?;
            let conn = &tg.conn;

            // Verify at least one episode with this source_id exists.
            let count: i64 = conn
                .query(
                    "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
                    libsql::params![self.source_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?
                .next()
                .await
                .map_err(CoreError::Database)?
                .ok_or_else(|| {
                    MemoryError::Other(
                        "update_episode_metadata: COUNT query returned no rows".to_string(),
                    )
                })?
                .get(0)
                .map_err(CoreError::Database)?;

            if count == 0 {
                return Err(MemoryError::Other(format!(
                    "update_episode_metadata: no episode found with source_id={}",
                    self.source_id
                )));
            }

            // Read existing metadata TEXT for the source_id.
            let existing_text: Option<String> = conn
                .query(
                    "SELECT metadata FROM episodes WHERE source_id = ?1",
                    libsql::params![self.source_id.clone()],
                )
                .await
                .map_err(CoreError::Database)?
                .next()
                .await
                .map_err(CoreError::Database)?
                .ok_or_else(|| {
                    MemoryError::Other(
                        "update_episode_metadata: metadata SELECT returned no rows".to_string(),
                    )
                })?
                .get(0)
                .map_err(CoreError::Database)?;

            // Parse existing (NULL → empty object).
            let mut existing_obj: serde_json::Map<String, serde_json::Value> = match existing_text {
                Some(ref s) => serde_json::from_str(s).map_err(|e| {
                    MemoryError::Other(format!(
                        "update_episode_metadata: failed to parse existing metadata as JSON: {e}"
                    ))
                })?,
                None => serde_json::Map::new(),
            };

            // Shallow-merge: iterate patch top-level keys, overwrite/insert.
            // Arrays are replaced, not merged — this is automatic because we
            // overwrite the top-level key, not recurse into nested structures.
            if let Some(patch_obj) = patch_value.as_object() {
                for (k, v) in patch_obj {
                    existing_obj.insert(k.clone(), v.clone());
                }
            }

            // Serialize and UPDATE.
            let merged_text = serde_json::to_string(&existing_obj).map_err(|e| {
                MemoryError::Other(format!(
                    "update_episode_metadata: failed to serialize merged metadata: {e}"
                ))
            })?;

            let updated = conn
                .execute(
                    "UPDATE episodes SET metadata = ?1 WHERE source_id = ?2",
                    libsql::params![merged_text, self.source_id],
                )
                .await
                .map_err(CoreError::Database)?;

            Ok(updated as usize)
        })
    }
}

// ── Memory::update_episode_metadata entry point ───────────────────────────────

impl Memory {
    /// Update the `metadata` JSON column for the episode(s) with the given
    /// `source_id` via shallow-merge of `patch` into existing metadata.
    ///
    /// # Semantics
    ///
    /// - **Shallow merge**: patch keys at the top level merge with existing
    ///   metadata; nested objects are NOT recursively merged.
    /// - **Arrays REPLACE not merge**: a patch `{"refs": [b]}` FULLY REPLACES
    ///   an existing `{"refs": [a]}` — no concatenation, no de-duplication.
    /// - **NULL → empty object**: if `episode.metadata` is NULL, treats as
    ///   `{}` and merges.
    /// - **Patch must be Object**: a non-object patch (`json!(42)`, array,
    ///   string) returns `Err` at `.await` time.
    /// - **Does NOT mutate**: facts, bi-temporal axes (`valid_from`,
    ///   `valid_to`, `recorded_at` on facts), `source_id`, `source_uri`,
    ///   `recorded_at` on episodes — only the `metadata` column changes.
    ///
    /// # Substrate-purity
    ///
    /// `metadata` is opaque JSON. The substrate enforces no schema on its
    /// contents — consumers carry their own taxonomy.
    #[must_use = "UpdateEpisodeMetadataRequest must be .await-ed or have a terminal called"]
    pub fn update_episode_metadata<'a>(
        &'a self,
        source_id: impl Into<String> + 'a,
    ) -> UpdateEpisodeMetadataRequest<'a> {
        UpdateEpisodeMetadataRequest {
            memory: self,
            source_id: source_id.into(),
            patch: None,
            pending_error: None,
        }
    }
}

// ── Memory::recall_by_source_id (G5) ──────────────────────────────────────────

impl Memory {
    /// Direct lookup of episodes matching `source_id`. Returns episodes
    /// ordered by `recorded_at DESC` (newest first).
    ///
    /// # Namespace scope (Vera F11 fold-in)
    ///
    /// When `namespace` is `Some`, the query is restricted to that namespace
    /// only. When `None`, the Memory's default namespace is used; if no
    /// default is set, results span all namespaces (the only path where
    /// cross-namespace results can leak — opt-in via explicit `None` with no
    /// builder-default).
    ///
    /// # Returns
    ///
    /// `Ok(Vec<Episode>)` — empty when no episode rows match the filter.
    /// Never returns Err for "no match"; only DB errors propagate.
    pub async fn recall_by_source_id(
        &self,
        source_id: impl AsRef<str>,
        namespace: Option<Namespace>,
    ) -> Result<Vec<crate::core::schema::Episode>> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::recall_by_source_id requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .to_string(),
            )
        })?;
        let conn = &tg.conn;

        let source_id = source_id.as_ref().to_string();
        let group_filter: Option<String> = namespace
            .or_else(|| self.default_namespace.clone())
            .map(|ns| ns.namespace);

        // `(? IS NULL OR group_id = ?)` evaluates to TRUE when filter is NULL
        // → no namespace scope applied.
        // TD-003 Phase G: SELECT now projects source_id (idx 9), source_uri (idx 10),
        // content_hash (idx 11) — closing Episode struct ↔ table column asymmetry.
        // Previously content_hash was hardcoded None; source_id/source_uri were absent.
        let mut rows = conn
            .query(
                "SELECT id, content, timestamp, source_type, metadata, group_id, \
                        saga_id, sequence_number, recorded_at, source_id, source_uri, \
                        content_hash \
                 FROM episodes \
                 WHERE source_id = ?1 \
                   AND (?2 IS NULL OR group_id = ?2) \
                 ORDER BY recorded_at DESC",
                libsql::params![source_id, group_filter],
            )
            .await
            .map_err(CoreError::Database)?;

        let mut out: Vec<crate::core::schema::Episode> = Vec::new();
        while let Some(row) = rows.next().await.map_err(CoreError::Database)? {
            let id = row.get::<i64>(0).map_err(CoreError::Database)?;
            let content = row.get::<String>(1).map_err(CoreError::Database)?;
            let timestamp_text = row.get::<String>(2).map_err(CoreError::Database)?;
            let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_text)
                .map_err(|e| {
                    MemoryError::Other(format!(
                        "recall_by_source_id: episode.timestamp not RFC3339: {e}"
                    ))
                })?
                .with_timezone(&chrono::Utc);
            let source_type = row.get::<Option<String>>(3).map_err(CoreError::Database)?;
            let metadata_text = row.get::<Option<String>>(4).map_err(CoreError::Database)?;
            let metadata = match metadata_text {
                Some(t) => Some(serde_json::from_str::<serde_json::Value>(&t).map_err(|e| {
                    MemoryError::Other(format!(
                        "recall_by_source_id: episode.metadata not JSON: {e}"
                    ))
                })?),
                None => None,
            };
            let group_id = row.get::<Option<String>>(5).map_err(CoreError::Database)?;
            let saga_id = row.get::<Option<String>>(6).map_err(CoreError::Database)?;
            let sequence_number = row.get::<Option<i64>>(7).map_err(CoreError::Database)?;
            let recorded_at = row.get::<Option<String>>(8).map_err(CoreError::Database)?;
            let source_id_col = row.get::<Option<String>>(9).map_err(CoreError::Database)?;
            let source_uri = row.get::<Option<String>>(10).map_err(CoreError::Database)?;
            let content_hash = row.get::<Option<String>>(11).map_err(CoreError::Database)?;

            out.push(crate::core::schema::Episode {
                id,
                content,
                timestamp,
                source_type,
                metadata,
                group_id,
                saga_id,
                sequence_number,
                recorded_at,
                source_id: source_id_col,
                source_uri,
                content_hash,
            });
        }
        Ok(out)
    }
}

// ── G2 Red tests (AC.3) ───────────────────────────────────────────────────────

#[cfg(test)]
mod update_source_uri_tests {
    use std::sync::Arc;

    use crate::{
        core::provider::{MockChatProvider, NullEmbeddingProvider},
        memory::types::Namespace,
    };

    use super::Memory;

    /// Build an in-memory `Memory` instance wired with null providers.
    ///
    /// Uses `TemporalGraph::open_with_dim(":memory:", 384)` under the hood
    /// via `providers::open_graph`. Both LLM and embedder are null — the G2
    /// invariant under test (source_uri update) requires neither.
    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });

        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("in-memory Memory construction must not fail")
    }

    /// Seed an episode row directly via SQL, bypassing the facade ingest path.
    ///
    /// Required because `RememberRequest::with_source_id` / `with_source_uri`
    /// do not yet exist (G1 partial — struct fields added, facade builder
    /// methods deferred to G5). The `conn_for_test()` accessor is gated behind
    /// `#[cfg(any(test, feature = "test-utils"))]`.
    ///
    /// Returns the `group_id` used for the episode row.
    async fn seed_episode_with_source(
        mem: &Memory,
        source_id: &str,
        source_uri: &str,
        namespace: &Namespace,
    ) -> String {
        use crate::memory::engine_handle::namespace_to_group_id;

        let group_id = namespace_to_group_id(namespace);
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some for in-memory Memory");
        let conn = tg.conn_for_test();

        conn.execute(
            "INSERT INTO episodes \
             (group_id, content, timestamp, recorded_at, source_id, source_uri) \
             VALUES (?1, ?2, unixepoch('now'), datetime('now'), ?3, ?4)",
            libsql::params![
                group_id.clone(),
                "test content for source identity.",
                source_id,
                source_uri,
            ],
        )
        .await
        .expect("seed_episode_with_source INSERT must succeed");

        group_id
    }

    // ── AC.3 happy path ───────────────────────────────────────────────────────

    /// AC.3 — happy path: `update_source_uri` updates `source_uri` on the
    /// target episode without mutating any facts or bi-temporal axes.
    ///
    /// Invariants asserted:
    /// - `source_uri` changed to the new value on the episode row.
    /// - `source_id` is unchanged.
    /// - `recorded_at` on the episode row is unchanged.
    /// - Fact count in the `facts` table is unchanged (zero facts were
    ///   inserted by the seed, so the count remains zero).
    /// - No facts rows have `valid_from` mutated (vacuously true for zero
    ///   facts, but the assertion pattern confirms the contract explicitly).
    ///
    /// NOTE: This test will fail to compile until the Green agent ships
    /// `Memory::update_source_uri` + `UpdateSourceUriRequest`.
    #[tokio::test]
    async fn update_source_uri_does_not_mutate_facts_or_timestamps() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g2-happy");

        seed_episode_with_source(&mem, "doc-a", "path/old", &ns).await;

        // Capture pre-update state from episodes table.
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let pre_row = conn
            .query(
                "SELECT source_uri, source_id, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("pre-update query must succeed")
            .next()
            .await
            .expect("pre-update query row_next must succeed")
            .expect("pre-update episode row must exist");

        let pre_source_uri: String = pre_row.get(0).expect("source_uri column");
        let pre_source_id: String = pre_row.get(1).expect("source_id column");
        let pre_recorded_at: String = pre_row.get(2).expect("recorded_at column");

        assert_eq!(
            pre_source_uri, "path/old",
            "seed: source_uri must be path/old before update"
        );
        assert_eq!(pre_source_id, "doc-a", "seed: source_id must be doc-a");

        // Capture pre-update fact count. NullLlm produces zero facts — assert
        // zero and confirm it stays zero after the update.
        let pre_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts f \
                 JOIN episodes e ON e.id = f.subject_id \
                 WHERE e.source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("pre-update fact count query must succeed")
            .next()
            .await
            .expect("fact count row_next must succeed")
            .expect("fact count row must exist")
            .get(0)
            .expect("fact count column");

        // ── G2 builder call ──
        let updated_count: u64 = mem
            .update_source_uri("doc-a")
            .to("path/new")
            .await
            .expect("update_source_uri must succeed for existing source_id");

        assert!(
            updated_count >= 1,
            "at least one episode row must be updated"
        );

        // Post-update: verify source_uri changed, source_id unchanged.
        let post_row = conn
            .query(
                "SELECT source_uri, source_id, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("post-update query must succeed")
            .next()
            .await
            .expect("post-update query row_next must succeed")
            .expect("post-update episode row must exist");

        let post_source_uri: String = post_row.get(0).expect("source_uri column post");
        let post_source_id: String = post_row.get(1).expect("source_id column post");
        let post_recorded_at: String = post_row.get(2).expect("recorded_at column post");

        assert_eq!(
            post_source_uri, "path/new",
            "source_uri must be updated to path/new"
        );
        assert_eq!(
            post_source_id, "doc-a",
            "source_id must be unchanged after update"
        );
        assert_eq!(
            post_recorded_at, pre_recorded_at,
            "recorded_at must not be mutated by update_source_uri"
        );

        // Post-update: fact count must be unchanged.
        let post_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts f \
                 JOIN episodes e ON e.id = f.subject_id \
                 WHERE e.source_id = ?1",
                libsql::params!["doc-a"],
            )
            .await
            .expect("post-update fact count query must succeed")
            .next()
            .await
            .expect("post-update fact count row_next must succeed")
            .expect("post-update fact count row must exist")
            .get(0)
            .expect("post-update fact count column");

        assert_eq!(
            post_fact_count, pre_fact_count,
            "fact count must not change after update_source_uri"
        );
    }

    // ── AC.3 error path ───────────────────────────────────────────────────────

    /// AC.3 — error path: `update_source_uri` with a `source_id` that does not
    /// exist in any episode must return `Err`.
    #[tokio::test]
    async fn update_source_uri_missing_source_id_returns_err() {
        let mem = make_memory().await;

        let result = mem
            .update_source_uri("nonexistent-doc-12345")
            .to("any/path")
            .await;

        assert!(
            result.is_err(),
            "update_source_uri with unknown source_id must return Err, got: {:?}",
            result
        );
    }

    // ── AC.3 idempotency ──────────────────────────────────────────────────────

    /// AC.3 — idempotency: calling `update_source_uri` with the same URI as
    /// the current value must return `Ok` without error and leave the row
    /// unchanged.
    #[tokio::test]
    async fn update_source_uri_idempotent_same_uri() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g2-idempotent");

        seed_episode_with_source(&mem, "x", "p", &ns).await;

        let result = mem.update_source_uri("x").to("p").await;

        assert!(
            result.is_ok(),
            "update_source_uri with same uri must return Ok, got: {:?}",
            result
        );

        // Verify the source_uri row is still "p" — no spurious mutation.
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT source_uri FROM episodes WHERE source_id = ?1",
                libsql::params!["x"],
            )
            .await
            .expect("idempotency check query must succeed")
            .next()
            .await
            .expect("idempotency check row_next must succeed")
            .expect("idempotency check row must exist");

        let uri: String = row.get(0).expect("source_uri column in idempotency check");
        assert_eq!(
            uri, "p",
            "source_uri must remain 'p' after idempotent re-run"
        );
    }
}

// ── G3 Red tests (AC.4 + AC.10) ──────────────────────────────────────────────

#[cfg(test)]
mod update_episode_metadata_tests {
    use std::sync::Arc;

    use serde_json::json;

    use crate::{
        core::provider::{MockChatProvider, NullEmbeddingProvider},
        memory::types::Namespace,
    };

    use super::Memory;

    /// Build an in-memory `Memory` instance wired with null providers.
    async fn make_memory() -> Memory {
        let llm: Arc<dyn crate::memory::ChatProvider> = Arc::new(MockChatProvider::null());
        let embedder: Arc<dyn crate::core::provider::DynEmbeddingProvider> =
            Arc::new(NullEmbeddingProvider { dim: 384 });

        Memory::open(":memory:")
            .with_llm(llm)
            .with_embedder(embedder)
            .await
            .expect("in-memory Memory construction must not fail")
    }

    /// Seed an episode row with a given `source_id` and `metadata` JSON text
    /// directly via SQL, bypassing the facade ingest path.
    async fn seed_episode_with_metadata(
        mem: &Memory,
        source_id: &str,
        source_uri: &str,
        metadata_json: Option<&str>,
        namespace: &Namespace,
    ) -> String {
        use crate::memory::engine_handle::namespace_to_group_id;

        let group_id = namespace_to_group_id(namespace);
        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some for in-memory Memory");
        let conn = tg.conn_for_test();

        match metadata_json {
            Some(json_text) => {
                conn.execute(
                    "INSERT INTO episodes \
                     (group_id, content, timestamp, recorded_at, source_id, source_uri, metadata) \
                     VALUES (?1, ?2, unixepoch('now'), datetime('now'), ?3, ?4, ?5)",
                    libsql::params![
                        group_id.clone(),
                        "test episode content for metadata patch.",
                        source_id,
                        source_uri,
                        json_text,
                    ],
                )
                .await
                .expect("seed_episode_with_metadata INSERT (with metadata) must succeed");
            }
            None => {
                conn.execute(
                    "INSERT INTO episodes \
                     (group_id, content, timestamp, recorded_at, source_id, source_uri) \
                     VALUES (?1, ?2, unixepoch('now'), datetime('now'), ?3, ?4)",
                    libsql::params![
                        group_id.clone(),
                        "test episode content for metadata patch.",
                        source_id,
                        source_uri,
                    ],
                )
                .await
                .expect("seed_episode_with_metadata INSERT (NULL metadata) must succeed");
            }
        }

        group_id
    }

    // ── AC.4 happy path: shallow merge preserves existing keys ───────────────

    /// AC.4 — shallow merge: patching one key must update that key while
    /// preserving all other top-level keys in the existing metadata object.
    #[tokio::test]
    async fn update_metadata_shallow_merge_preserves_existing_keys() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-shallow-merge");

        seed_episode_with_metadata(
            &mem,
            "doc-b",
            "path/b",
            Some(r#"{"status":"draft","count":1}"#),
            &ns,
        )
        .await;

        mem.update_episode_metadata("doc-b")
            .patch(json!({"status": "accepted"}))
            .await
            .expect("update_episode_metadata shallow merge must succeed");

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-b"],
            )
            .await
            .expect("post-patch metadata query must succeed")
            .next()
            .await
            .expect("post-patch metadata row_next must succeed")
            .expect("post-patch episode row must exist");

        let metadata_text: String = row.get(0).expect("metadata column");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        assert_eq!(
            metadata,
            json!({"status": "accepted", "count": 1}),
            "shallow merge must update status and preserve count"
        );
    }

    // ── AC.10: arrays REPLACE, not append ────────────────────────────────────

    /// AC.10 — array replace semantics: patching an array-valued key must
    /// REPLACE the existing array entirely, not merge or append to it.
    #[tokio::test]
    async fn update_metadata_array_replace_not_append() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-array-replace");

        seed_episode_with_metadata(
            &mem,
            "doc-c",
            "path/c",
            Some(r#"{"refs":[{"id":"a","rel":"depends_on"}]}"#),
            &ns,
        )
        .await;

        mem.update_episode_metadata("doc-c")
            .patch(json!({"refs": [{"id": "b", "rel": "supersedes"}]}))
            .await
            .expect("update_episode_metadata array replace must succeed");

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-c"],
            )
            .await
            .expect("post-patch metadata query must succeed")
            .next()
            .await
            .expect("post-patch metadata row_next must succeed")
            .expect("post-patch episode row must exist");

        let metadata_text: String = row.get(0).expect("metadata column");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        let refs = metadata.get("refs").expect("refs key must exist");
        let refs_arr = refs.as_array().expect("refs must be a JSON array");

        assert_eq!(
            refs_arr.len(),
            1,
            "refs array must contain exactly one entry (REPLACE, not append); got: {refs_arr:?}"
        );
        assert_eq!(
            refs_arr[0],
            json!({"id": "b", "rel": "supersedes"}),
            "refs[0] must be the patched entry (id=b), not the original (id=a)"
        );
    }

    // ── AC.4 edge: NULL metadata column creates empty object then merges ──────

    /// AC.4 edge — NULL metadata: patching an episode with a NULL metadata
    /// column must treat existing state as `{}` and merge the patch into it.
    #[tokio::test]
    async fn update_metadata_null_metadata_creates_and_merges() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-null-metadata");

        seed_episode_with_metadata(&mem, "doc-d", "path/d", None, &ns).await;

        mem.update_episode_metadata("doc-d")
            .patch(json!({"first": "value"}))
            .await
            .expect("update_episode_metadata on NULL metadata column must succeed");

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-d"],
            )
            .await
            .expect("post-patch metadata query must succeed")
            .next()
            .await
            .expect("post-patch metadata row_next must succeed")
            .expect("post-patch episode row must exist");

        let metadata_text: String = row
            .get(0)
            .expect("metadata column must be non-NULL after patch");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        assert_eq!(
            metadata,
            json!({"first": "value"}),
            "NULL metadata + patch must produce the patch object itself"
        );
    }

    // ── AC.4 input validation: non-object patch is rejected ──────────────────

    /// AC.4 — input validation: calling `.patch(value)` with a non-Object JSON
    /// value must return `Err` at build/await time.
    #[tokio::test]
    async fn update_metadata_non_object_patch_rejects() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-non-object-patch");

        seed_episode_with_metadata(
            &mem,
            "doc-e-reject",
            "path/e-reject",
            Some(r#"{"existing":"value"}"#),
            &ns,
        )
        .await;

        let result = mem
            .update_episode_metadata("doc-e-reject")
            .patch(json!(42))
            .await;

        assert!(
            result.is_err(),
            "update_episode_metadata with non-object patch must return Err, got: {:?}",
            result
        );
    }

    // ── AC.4 error path: missing source_id returns Err ───────────────────────

    /// AC.4 error path — missing source_id: patching a `source_id` that does
    /// not exist must return `Err`.
    #[tokio::test]
    async fn update_metadata_missing_source_id_returns_err() {
        let mem = make_memory().await;

        let result = mem
            .update_episode_metadata("nonexistent-source-99999")
            .patch(json!({"any": "patch"}))
            .await;

        assert!(
            result.is_err(),
            "update_episode_metadata with unknown source_id must return Err, got: {:?}",
            result
        );
    }

    // ── AC.4 non-mutation: facts, source_uri, recorded_at are unchanged ───────

    /// AC.4 non-mutation guarantee: `update_episode_metadata` must only mutate
    /// the `metadata` column.
    #[tokio::test]
    async fn update_metadata_does_not_mutate_facts_or_uri() {
        let mem = make_memory().await;
        let ns = Namespace::new("test-g3-non-mutation");

        seed_episode_with_metadata(&mem, "doc-f", "path/f", Some(r#"{"status":"draft"}"#), &ns)
            .await;

        let tg = mem
            .temporal_graph
            .as_ref()
            .expect("temporal_graph must be Some");
        let conn = tg.conn_for_test();

        let pre_row = conn
            .query(
                "SELECT source_uri, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-f"],
            )
            .await
            .expect("pre-patch episode query must succeed")
            .next()
            .await
            .expect("pre-patch episode row_next must succeed")
            .expect("pre-patch episode row must exist");

        let pre_source_uri: String = pre_row.get(0).expect("source_uri column pre");
        let pre_recorded_at: String = pre_row.get(1).expect("recorded_at column pre");

        let pre_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts WHERE source_episode_id IN \
                 (SELECT id FROM episodes WHERE source_id = ?1)",
                libsql::params!["doc-f"],
            )
            .await
            .expect("pre-patch fact count query must succeed")
            .next()
            .await
            .expect("pre-patch fact count row_next must succeed")
            .expect("pre-patch fact count row must exist")
            .get(0)
            .expect("fact count column pre");

        mem.update_episode_metadata("doc-f")
            .patch(json!({"status": "accepted"}))
            .await
            .expect("update_episode_metadata non-mutation test must succeed");

        let post_row = conn
            .query(
                "SELECT source_uri, recorded_at FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-f"],
            )
            .await
            .expect("post-patch episode query must succeed")
            .next()
            .await
            .expect("post-patch episode row_next must succeed")
            .expect("post-patch episode row must exist");

        let post_source_uri: String = post_row.get(0).expect("source_uri column post");
        let post_recorded_at: String = post_row.get(1).expect("recorded_at column post");

        assert_eq!(
            post_source_uri, pre_source_uri,
            "source_uri must not be mutated by update_episode_metadata"
        );
        assert_eq!(
            post_recorded_at, pre_recorded_at,
            "recorded_at must not be mutated by update_episode_metadata"
        );

        let post_fact_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM facts WHERE source_episode_id IN \
                 (SELECT id FROM episodes WHERE source_id = ?1)",
                libsql::params!["doc-f"],
            )
            .await
            .expect("post-patch fact count query must succeed")
            .next()
            .await
            .expect("post-patch fact count row_next must succeed")
            .expect("post-patch fact count row must exist")
            .get(0)
            .expect("fact count column post");

        assert_eq!(
            post_fact_count, pre_fact_count,
            "fact count must not change after update_episode_metadata"
        );

        let meta_row = conn
            .query(
                "SELECT metadata FROM episodes WHERE source_id = ?1",
                libsql::params!["doc-f"],
            )
            .await
            .expect("post-patch metadata sanity query must succeed")
            .next()
            .await
            .expect("post-patch metadata sanity row_next must succeed")
            .expect("post-patch metadata sanity row must exist");

        let metadata_text: String = meta_row.get(0).expect("metadata column sanity");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata must be valid JSON");

        assert_eq!(
            metadata.get("status").and_then(|v| v.as_str()),
            Some("accepted"),
            "metadata status must be updated to 'accepted' (sanity: patch was not a no-op)"
        );
    }
}
