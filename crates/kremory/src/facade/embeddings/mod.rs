//! Embedding backfill + re-embed maintenance on `Memory` (TD-043 split out of
//! `facade/mod.rs`).
//!
//! Six operator entrypoints over the same shape — walk a table in pages, embed each
//! item's text, store the vector, count success/failure — across episodes, entities
//! and facts, in a `backfill` (fill NULLs only) and a `reembed_all` (rewrite every
//! row) variant. They shared ~700 lines of `facade/mod.rs` with the consumer-facing
//! API for no reason other than history: nothing here is part of the everyday
//! remember/recall path, and every one of them is something you run deliberately,
//! against a copy, after changing an embedder.
//!
//! The whole module is gated on `content-search` — the columns it fills only exist
//! there.

use super::*;

/// Tally returned by
/// [`Memory::backfill_episode_embeddings`] and, also by
/// [`Memory::reembed_all_episode_embeddings`],
/// [`Memory::reembed_all_entity_embeddings`], and
/// [`Memory::reembed_all_fact_embeddings`] — all four drive the same
/// embed+store shape (embed a page item's text, persist the vector, count
/// success/failure) over a different table/page-source, so they share this
/// tally shape rather than each declaring an identical `{embedded, failed}`
/// struct.
/// Field names stay table-agnostic ("items", not "episodes") accordingly.
/// Feature-gated behind `content-search` (the whole embedding path only
/// exists there).
#[cfg(feature = "content-search")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpisodeEmbeddingBackfill {
    /// Items (episodes/entities/facts, depending on which method returned
    /// this tally) whose text was embedded + stored this run.
    pub embedded: u64,
    /// Items skipped due to a per-item embed/store failure (WARN-logged;
    /// re-run to retry them).
    pub failed: u64,
}

/// Args-as-object for [`Memory::embed_and_store_episode_page`] per
/// (`clippy.toml` `too-many-arguments-threshold = 3`, `self` counts).
/// Private — an internal seam shared by
/// [`Memory::backfill_episode_embeddings`] and
/// [`Memory::reembed_all_episode_embeddings`], not part of the public API.
#[cfg(feature = "content-search")]
struct EmbedEpisodePageParams<'a> {
    tg: &'a TemporalGraph,
    batch: Vec<(i64, String)>,
    stats: &'a mut EpisodeEmbeddingBackfill,
    op: &'static str,
}

/// Args-as-object for [`Memory::embed_and_store_entity_page`] —
/// sibling of [`EmbedEpisodePageParams`], same rationale.
#[cfg(feature = "content-search")]
struct EmbedEntityPageParams<'a> {
    tg: &'a TemporalGraph,
    batch: Vec<crate::core::graph::EntityReembedRow>,
    stats: &'a mut EpisodeEmbeddingBackfill,
    op: &'static str,
}

/// Args-as-object for [`Memory::embed_and_store_fact_page`] — sibling
/// of [`EmbedEpisodePageParams`], same rationale.
#[cfg(feature = "content-search")]
struct EmbedFactPageParams<'a> {
    tg: &'a TemporalGraph,
    batch: Vec<(i64, String)>,
    stats: &'a mut EpisodeEmbeddingBackfill,
    op: &'static str,
}

impl Memory {
    /// Backfill `episodes.embedding` for every
    /// episode that has none, over the existing corpus — NO re-ingest, NO LLM.
    ///
    /// Selects NULL-embedding episodes in pages of `batch_size`, embeds each
    /// episode's `content` with the SAME embedder the graph already uses for
    /// entity/fact embeddings, and UPDATEs the `embedding` column (populating
    /// the `episodes_vec_idx` DiskANN index from Migration 026). Idempotent +
    /// resumable: an already-embedded episode is skipped (its `embedding` is
    /// non-NULL), so re-running only fills the remaining gap. Feature-gated
    /// behind `content-search` (the column only exists there).
    ///
    /// Returns the run [`EpisodeEmbeddingBackfill`] tally. Per-episode embed
    /// failures are counted (`failed`) + WARN-logged but do NOT abort the run —
    /// a transient embedder hiccup on one episode must not lose the whole
    /// backfill (re-run to retry the failures).
    ///
    /// Intended as an operator/maintenance entrypoint (e.g. the
    /// `kremory-http backfill-episode-embeddings` subcommand) — run it against a
    /// COPY of the DB before measuring the dense arm, so the pre-existing corpus
    /// is dense-searchable without a full re-ingest.
    #[cfg(feature = "content-search")]
    pub async fn backfill_episode_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::backfill_episode_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        // Guard against a zero page size (an infinite no-progress loop).
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        loop {
            let batch = tg
                .episodes_missing_embedding(batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            self.embed_and_store_episode_page(EmbedEpisodePageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "backfill_episode_embeddings",
            })
            .await;
            // If a whole page was all-failures we would loop forever on the same
            // NULL rows (the `WHERE embedding IS NULL` predicate never drops a
            // failed row out of the next page) — bail once we've made no forward
            // progress on a full page.
            if stats.embedded == 0 && stats.failed > 0 {
                tracing::warn!(
                    failed = stats.failed,
                    "backfill_episode_embeddings: first page all-failed — aborting (check the embedder)"
                );
                break;
            }
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.backfill_episode_embeddings complete"
        );
        Ok(stats)
    }

    /// Re-embed
    /// **every** episode's `content`, overwriting any embedding already
    /// stored — the remedy for an embedding-CONFIG change (flipping
    /// [`SearchConfig::embed_task_prefix_enabled`](crate::core::config::SearchConfig::embed_task_prefix_enabled),
    /// swapping the embedder model, or changing the embedding dimension),
    /// none of which [`backfill_episode_embeddings`](Self::backfill_episode_embeddings)
    /// can serve — that method's `WHERE embedding IS NULL` paging can only
    /// FILL a gap, it can never RE-embed a row that already has a vector.
    ///
    /// ⚠️ This rewrites every `episodes.embedding` value in the database. Run
    /// it against a COPY of the DB before measuring — see the
    /// `embed_task_prefix_enabled` doc for the full safe sequence (flip the
    /// knob on a fresh copy, re-embed in full, THEN measure). Entity/fact
    /// embeddings are NOT touched by this method (they need a full re-ingest,
    /// or — for entities specifically — the merge-time re-embed covers
    /// only the alias-merge path, not a bulk config-change re-embed).
    ///
    /// Pages via an id-cursor over ALL episode rows
    /// ([`TemporalGraph::episodes_after_id`]), not the NULL-only predicate
    /// [`backfill_episode_embeddings`](Self::backfill_episode_embeddings)
    /// uses — see that query's doc for why a plain `LIMIT` loop over an
    /// unfiltered page source would never terminate. Idempotent: safe to
    /// re-run (e.g. to retry any per-episode failures from a prior run — see
    /// [`EpisodeEmbeddingBackfill::failed`]).
    ///
    /// Feature-gated behind `content-search` (the column only exists there).
    #[cfg(feature = "content-search")]
    pub async fn reembed_all_episode_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::reembed_all_episode_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        // Guard against a zero page size (an infinite no-progress loop).
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        let mut after_id: i64 = 0;
        loop {
            let batch = tg
                .episodes_after_id(after_id, batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            // Advance the cursor to the last id in THIS page before consuming
            // `batch` below — unlike the NULL-predicate backfill, a row stays
            // in this unfiltered result set after being re-embedded, so the
            // cursor (not the predicate) is what makes the loop terminate.
            after_id = batch.last().map(|(id, _)| *id).unwrap_or(after_id);
            self.embed_and_store_episode_page(EmbedEpisodePageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "reembed_all_episode_embeddings",
            })
            .await;
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.reembed_all_episode_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared embed step: document-prefix `text` (WRITE side —
    /// always [`document_embed_text`](crate::core::embed_prefix::document_embed_text),
    /// never the query-side prefix) and hand it to the configured embedder.
    ///
    /// Shared by all THREE bulk re-embed page loops
    /// ([`embed_and_store_episode_page`](Self::embed_and_store_episode_page),
    /// [`embed_and_store_entity_page`](Self::embed_and_store_entity_page),
    /// [`embed_and_store_fact_page`](Self::embed_and_store_fact_page)) — the
    /// one piece of "embed+write" body that is byte-identical across all
    /// three. The write-BACK half deliberately stays in each caller instead
    /// of behind a generic/closure dispatch: the id type differs (`i64` for
    /// episodes/facts, the entity's TEXT slug for entities) and so does the
    /// setter (`set_episode_embedding` / `set_entity_embedding` /
    /// `set_fact_embedding`) — a fully generic write dispatch would need
    /// either a discriminated-union id type or a closure-per-call-site, more
    /// machinery than the ~6 lines of per-caller match-arm plumbing it would
    /// save.
    #[cfg(feature = "content-search")]
    async fn embed_document_text(&self, text: &str) -> crate::core::error::Result<Vec<f32>> {
        let embed_task_prefix_enabled = self.search_config().embed_task_prefix_enabled;
        let prefixed_content =
            crate::core::embed_prefix::document_embed_text(text, embed_task_prefix_enabled);
        self.embedder.embed_dyn(&prefixed_content).await
    }

    /// Shared per-page embed+store body for
    /// [`backfill_episode_embeddings`](Self::backfill_episode_embeddings) and
    /// [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)
    /// — same embed call ([`embed_document_text`](Self::embed_document_text)),
    /// same per-episode failure handling; the two callers differ only in
    /// which paging query selected `batch`. `op` labels the WARN log lines so
    /// a failure can be attributed to the caller that hit it. Args-as-object
    /// Args-as-object (`clippy.toml` `too-many-arguments-threshold = 3`).
    #[cfg(feature = "content-search")]
    async fn embed_and_store_episode_page(&self, params: EmbedEpisodePageParams<'_>) {
        let EmbedEpisodePageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for (episode_id, content) in batch {
            match self.embed_document_text(&content).await {
                Ok(embedding) => match tg.set_episode_embedding(episode_id, &embedding).await {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(
                            error = %e,
                            episode_id,
                            op,
                            "set_episode_embedding failed"
                        );
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, episode_id, op, "embedder failed");
                }
            }
        }
    }

    /// Re-embed
    /// **every** entity's display name, overwriting any embedding already
    /// stored. Sibling of [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)
    /// — same shape, same document-prefix routing, different page
    /// source ([`TemporalGraph::entities_after_id`], composite-`(id,
    /// group_id)`-cursored — see that fn's doc for why).
    ///
    /// This is the remedy for TWO distinct staleness sources: (1) a live
    /// correctness bug — after a dream-phase merge/alias, a de-duplicated
    /// entity's identity may have changed (a new canonical name) while its
    /// stored embedding still encodes the pre-merge surface form, and no
    /// production path re-persisted it in bulk before this method existed
    /// (the merge-time re-embed at `core::canonicalization::apply_merge_with_audit`
    /// only covers the LIVE merge sites, not a full-corpus catch-up); and (2)
    /// an embedding-CONFIG change (flipping
    /// [`SearchConfig::embed_task_prefix_enabled`](crate::core::config::SearchConfig::embed_task_prefix_enabled),
    /// swapping the embedder model, or changing the embedding dimension).
    ///
    /// ⚠️ This rewrites every `entities.embedding` value in the database. Run
    /// it against a COPY of the DB before measuring — see
    /// [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)'s
    /// doc for the full safe sequence. Idempotent: safe to re-run (e.g. to
    /// retry any per-entity failures from a prior run).
    ///
    /// Feature-gated behind `content-search` (mirrors the episode/fact
    /// siblings — all three bulk re-embed paths ship together, even though
    /// entity/fact embeddings do not themselves depend on the content-RAG
    /// column; the `content-search` gate is this method family's existing
    /// convention, not a new dependency).
    #[cfg(feature = "content-search")]
    pub async fn reembed_all_entity_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::reembed_all_entity_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        let mut after_id = String::new();
        let mut after_group_id = String::new();
        loop {
            let batch = tg
                .entities_after_id(crate::core::graph::EntitiesAfterIdParams {
                    after_id: &after_id,
                    after_group_id: &after_group_id,
                    limit: batch_size,
                })
                .await
                .map_err(MemoryError::Core)?;
            // Advance the composite cursor to the LAST row in THIS page
            // before consuming `batch` below — same rationale as
            // `reembed_all_episode_embeddings`'s cursor advance (an
            // unfiltered page source never self-consumes). No `.expect()`
            // (banned in src/): an empty page ends the loop via the
            // `let-else`, same as a `.is_empty()` break would, without
            // needing a fallible unwrap on `.last()` right after.
            let Some(last) = batch.last() else {
                break;
            };
            after_id.clone_from(&last.id);
            after_group_id.clone_from(&last.group_id);
            self.embed_and_store_entity_page(EmbedEntityPageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "reembed_all_entity_embeddings",
            })
            .await;
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.reembed_all_entity_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared per-page embed+store body for
    /// [`reembed_all_entity_embeddings`](Self::reembed_all_entity_embeddings)
    /// — mirrors [`embed_and_store_episode_page`](Self::embed_and_store_episode_page);
    /// differs only in the id type (`&str` slug, not `i64`) and setter
    /// ([`TemporalGraph::set_entity_embedding`]). Args-as-object.
    #[cfg(feature = "content-search")]
    async fn embed_and_store_entity_page(&self, params: EmbedEntityPageParams<'_>) {
        let EmbedEntityPageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for row in batch {
            match self.embed_document_text(&row.embed_text).await {
                // Scoped by the row's OWN `group_id`.
                // `EntityReembedRow` carries it precisely because `id` alone is
                // not the entity key — `entities_after_id` documents the same
                // composite-cursor reason. Unscoped, a full re-embed wrote each
                // name's vector across every namespace, so the last row
                // processed silently won for all of them.
                Ok(embedding) => match tg
                    .set_entity_embedding_in_group(crate::core::graph::SetEntityEmbeddingParams {
                        id: &row.id,
                        group_id: &row.group_id,
                        embedding: &embedding,
                    })
                    .await
                {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(
                            error = %e,
                            entity_id = %row.id,
                            op,
                            "set_entity_embedding failed"
                        );
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, entity_id = %row.id, op, "embedder failed");
                }
            }
        }
    }

    /// Re-embed
    /// **every** fact's `subject predicate object` triple text, overwriting
    /// any embedding already stored. Sibling of
    /// [`reembed_all_episode_embeddings`](Self::reembed_all_episode_embeddings)
    /// — same shape, same document-prefix routing, different page
    /// source ([`TemporalGraph::facts_after_id`] — see that fn's doc for how
    /// the subject/object text is reconstructed from stored entity rows,
    /// since the raw extraction strings themselves are not persisted).
    ///
    /// De-confounds a fact's dependency on its subject/object entities: when
    /// [`reembed_all_entity_embeddings`](Self::reembed_all_entity_embeddings)
    /// changes an entity's resolved display name (e.g. its `properties.name`
    /// was corrected), facts referencing that entity should be re-embedded
    /// too so their triple text stays in sync — run entity re-embed BEFORE
    /// fact re-embed when both are needed (the `reembed-all-embeddings`
    /// `kremory-http` subcommand does this in the right order).
    ///
    /// ⚠️ This rewrites every `facts.embedding` value in the database. Run it
    /// against a COPY of the DB before measuring. Idempotent: safe to re-run.
    /// Feature-gated behind `content-search`.
    #[cfg(feature = "content-search")]
    pub async fn reembed_all_fact_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::reembed_all_fact_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        let mut after_id: i64 = 0;
        loop {
            let batch = tg
                .facts_after_id(after_id, batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            after_id = batch.last().map(|(id, _)| *id).unwrap_or(after_id);
            self.embed_and_store_fact_page(EmbedFactPageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "reembed_all_fact_embeddings",
            })
            .await;
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.reembed_all_fact_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared per-page embed+store body for
    /// [`reembed_all_fact_embeddings`](Self::reembed_all_fact_embeddings) —
    /// mirrors [`embed_and_store_episode_page`](Self::embed_and_store_episode_page);
    /// differs only in the setter ([`TemporalGraph::set_fact_embedding`]).
    /// Args-as-object.
    #[cfg(feature = "content-search")]
    async fn embed_and_store_fact_page(&self, params: EmbedFactPageParams<'_>) {
        let EmbedFactPageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for (fact_id, fact_text) in batch {
            match self.embed_document_text(&fact_text).await {
                Ok(embedding) => match tg.set_fact_embedding(fact_id, &embedding).await {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(error = %e, fact_id, op, "set_fact_embedding failed");
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, fact_id, op, "embedder failed");
                }
            }
        }
    }

    /// Fill the
    /// entity embedding gap — the entity sibling of
    /// [`backfill_episode_embeddings`](Self::backfill_episode_embeddings),
    /// same shape exactly: same constructed-via-builder guard, same `op:`
    /// naming convention on WARN logs, same [`EpisodeEmbeddingBackfill`]
    /// return/tally shape, same namespace-scoping discipline (via
    /// [`TemporalGraph::backfill_entity_embedding`]'s composite `(id,
    /// group_id)` key).
    ///
    /// Pages via [`TemporalGraph::entities_missing_embeddings`]'s `WHERE
    /// embedding IS NULL` predicate — unlike
    /// [`reembed_all_entity_embeddings`](Self::reembed_all_entity_embeddings),
    /// this can ONLY fill a gap; it never touches a row that already carries
    /// an embedding (re-run to retry the earlier tally's `failed` rows).
    ///
    /// Feature-gated behind `content-search` (the column only exists there).
    #[cfg(feature = "content-search")]
    pub async fn backfill_entity_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::backfill_entity_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        // Guard against a zero page size (an infinite no-progress loop).
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        loop {
            let batch = tg
                .entities_missing_embeddings(batch_size)
                .await
                .map_err(MemoryError::Core)?;
            if batch.is_empty() {
                break;
            }
            self.backfill_and_store_entity_page(EmbedEntityPageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "backfill_entity_embeddings",
            })
            .await;
            // Mirrors `backfill_episode_embeddings`'s early-abort: if a whole
            // page was all-failures we would loop forever on the same NULL
            // rows (the `WHERE embedding IS NULL` predicate never drops a
            // failed row out of the next page) — bail once we've made no
            // forward progress on a full page.
            if stats.embedded == 0 && stats.failed > 0 {
                tracing::warn!(
                    failed = stats.failed,
                    "backfill_entity_embeddings: first page all-failed — aborting (check the embedder)"
                );
                break;
            }
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.backfill_entity_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared per-page embed+store body for
    /// [`backfill_entity_embeddings`](Self::backfill_entity_embeddings) —
    /// mirrors [`embed_and_store_entity_page`](Self::embed_and_store_entity_page)
    /// exactly (including the namespace-scoped write), except the
    /// write-back calls [`TemporalGraph::backfill_entity_embedding`] (the
    /// new NULL-only setter) rather than
    /// [`TemporalGraph::set_entity_embedding_in_group`] — mirrors the
    /// `set_fact_embedding` / `backfill_fact_embedding` split already present
    /// at the fact layer. Reuses [`EmbedEntityPageParams`] since the two
    /// bodies differ only in which setter they call, not in their data shape.
    #[cfg(feature = "content-search")]
    async fn backfill_and_store_entity_page(&self, params: EmbedEntityPageParams<'_>) {
        let EmbedEntityPageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for row in batch {
            match self.embed_document_text(&row.embed_text).await {
                Ok(embedding) => match tg
                    .backfill_entity_embedding(crate::core::graph::SetEntityEmbeddingParams {
                        id: &row.id,
                        group_id: &row.group_id,
                        embedding: &embedding,
                    })
                    .await
                {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(
                            error = %e,
                            entity_id = %row.id,
                            op,
                            "backfill_entity_embedding failed"
                        );
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, entity_id = %row.id, op, "embedder failed");
                }
            }
        }
    }

    /// Fill the fact embedding gap by driving the pre-existing Story
    /// #214 crash-recovery primitives
    /// ([`TemporalGraph::facts_missing_embeddings`],
    /// [`TemporalGraph::backfill_fact_embedding`]) for the first time from a
    /// production code path — both were previously reachable only from their
    /// own definitions and from `graph/tests.rs`.
    ///
    /// `facts_missing_embeddings` has no `LIMIT`/cursor of its own (it
    /// already returns every missing-embedding fact in one query), so
    /// `batch_size` here controls how many rows this loop hands to
    /// [`backfill_and_store_fact_page`](Self::backfill_and_store_fact_page)
    /// per iteration — the same early-abort intent as the episode/entity
    /// siblings (a broken embedder is caught after the first `batch_size`
    /// failures, not after every row), applied client-side since the query
    /// itself can't be paged.
    ///
    /// Feature-gated behind `content-search` (mirrors the episode/entity
    /// siblings — all bulk re-embed/backfill paths ship together).
    #[cfg(feature = "content-search")]
    pub async fn backfill_fact_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<EpisodeEmbeddingBackfill> {
        let tg = self.temporal_graph.as_ref().ok_or_else(|| {
            MemoryError::Other(
                "Memory::backfill_fact_embeddings requires a Memory constructed via the \
                 builder/providers path (no Arc<TemporalGraph> attached)"
                    .into(),
            )
        })?;
        let batch_size = batch_size.max(1);

        let mut stats = EpisodeEmbeddingBackfill::default();
        // `facts_missing_embeddings` has no `LIMIT`/cursor of its own — it
        // already returns every missing-embedding fact in one query (see its
        // own doc comment). `batch_size` chunks the in-memory result for the
        // per-page helper below, purely to bound how many rows are embedded
        // before the early-abort check runs (see the loop body).
        let missing = tg
            .facts_missing_embeddings()
            .await
            .map_err(MemoryError::Core)?;
        for chunk in missing.chunks(batch_size) {
            let batch: Vec<(i64, String)> = chunk
                .iter()
                .map(
                    |(fact_id, subject_id, predicate, object_value, object_id)| {
                        // Text built from the raw subject_id/object_id
                        // (entity slugs), NOT a properties.name lookup — unlike
                        // `facts_after_id`'s reconstruction (used by
                        // `reembed_all_fact_embeddings`), `facts_missing_embeddings`
                        // does not JOIN `entities` for display names, and this
                        // method must not rewrite that primitive. The unscoped
                        // `TemporalGraph::get_entity` could supply a display
                        // name, but it matches on `id` alone (no `group_id`) —
                        // exactly the cross-namespace bug this codebase
                        // already fixed elsewhere. Falling back to the raw
                        // id/slug is safe (it IS `entity_display_name`'s own
                        // fallback value) at the cost of missing a
                        // `properties.name` override — proven by
                        // `backfill_fact_embeddings_builds_text_from_raw_entity_ids`.
                        let object_text = object_id
                            .clone()
                            .or_else(|| object_value.clone())
                            .unwrap_or_default();
                        (*fact_id, format!("{subject_id} {predicate} {object_text}"))
                    },
                )
                .collect();
            self.backfill_and_store_fact_page(EmbedFactPageParams {
                tg,
                batch,
                stats: &mut stats,
                op: "backfill_fact_embeddings",
            })
            .await;
            // Mirrors the episode/entity siblings' early-abort: bail once a
            // full chunk made no forward progress (broken embedder), rather
            // than burning through every remaining chunk on guaranteed
            // failures.
            if stats.embedded == 0 && stats.failed > 0 {
                tracing::warn!(
                    failed = stats.failed,
                    "backfill_fact_embeddings: first page all-failed — aborting (check the embedder)"
                );
                break;
            }
        }
        tracing::info!(
            embedded = stats.embedded,
            failed = stats.failed,
            "kremory.backfill_fact_embeddings complete"
        );
        Ok(stats)
    }

    /// Shared per-page embed+store body for
    /// [`backfill_fact_embeddings`](Self::backfill_fact_embeddings) — mirrors
    /// [`embed_and_store_fact_page`](Self::embed_and_store_fact_page) exactly,
    /// except the write-back calls [`TemporalGraph::backfill_fact_embedding`]
    /// (Story #214's crash-recovery setter) rather than
    /// [`TemporalGraph::set_fact_embedding`] — this backfill path is what gives that
    /// primitive its first production caller. Reuses [`EmbedFactPageParams`]
    /// since the two bodies differ only in which setter they call, not in
    /// their data shape.
    #[cfg(feature = "content-search")]
    async fn backfill_and_store_fact_page(&self, params: EmbedFactPageParams<'_>) {
        let EmbedFactPageParams {
            tg,
            batch,
            stats,
            op,
        } = params;
        for (fact_id, fact_text) in batch {
            match self.embed_document_text(&fact_text).await {
                Ok(embedding) => match tg.backfill_fact_embedding(fact_id, &embedding).await {
                    Ok(()) => stats.embedded += 1,
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!(error = %e, fact_id, op, "backfill_fact_embedding failed");
                    }
                },
                Err(e) => {
                    stats.failed += 1;
                    tracing::warn!(error = %e, fact_id, op, "embedder failed");
                }
            }
        }
    }

}

#[cfg(all(test, feature = "content-search"))]
mod tests;
