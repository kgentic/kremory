use chrono::{DateTime, Utc};
use metrics::histogram;
use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use crate::core::error::Result;
use crate::core::schema::{Fact, TemporalGraph};

use super::{row_to_fact, SubGraph};

/// Bundled parameters for [`TemporalGraph::get_neighbours_at`] — args-as-object
/// to satisfy a too-many-arguments lint (the receiver plus 3
/// positional params trips the project's 3-arg threshold).
/// What a `batch_forget` actually removed, per table.
///
/// `entities` alone is a misleading success signal: shared-entity preservation
/// pins any subject that also appears elsewhere, so a correct erasure routinely
/// deletes facts and edges while removing ZERO entities. A caller branching on
/// `entities > 0` concludes nothing happened (TD-247).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchForgetCounts {
    pub entities: u64,
    pub facts: u64,
    pub edges: u64,
}

/// Bundled params for [`TemporalGraph::get_facts_by_subject_predicate`] —
/// args-as-object (`too_many_arguments` threshold 3, `&self` counts).
pub struct GetFactsBySubjectPredicateParams<'a> {
    pub subject_id: &'a str,
    pub predicate: &'a str,
    pub group_id: &'a str,
}

pub struct GetNeighboursAtParams<'a> {
    pub entity_id: &'a str,
    pub hops: u32,
    pub as_of: Option<DateTime<Utc>>,
    /// In-BFS visited-entity
    /// cap. `None` = unbounded (today's behaviour). `Some(cap)` early-exits the
    /// BFS once `visited_entities.len() >= cap`, bounding hub-explosion on the
    /// widened multi-hop path (spec R1). The caller-side per-seed neighbour cap
    /// alone is insufficient at `hops >= 2` — every neighbour is
    /// enqueued+queried before the caller can trim (see `context.rs`). Placed at
    /// the top of the BFS loop so `hops == 1` stays byte-identical: a seed's
    /// direct neighbours are all enqueued in the seed's own iteration, so the
    /// cap only fires on the NEXT iteration (after every hop-1 neighbour is
    /// already visited) — it bites only the hop>=2 expansion.
    ///
    /// DETERMINISM CAVEAT: at `hops >= 2`, WHICH entities are
    /// visited before the cap trips depends on the per-hop fact-query row order,
    /// and that query has no `ORDER BY` — so the surviving visited SET can vary
    /// run-to-run for an identical `hops >= 2` query. At the default `hops == 1`
    /// this is fully deterministic (every direct neighbour is visited before the
    /// cap fires). When `expansion_hop_bound` is actually raised past 1, add an
    /// `ORDER BY id` to the per-hop query for a stable frontier.
    pub max_visited: Option<usize>,
}

impl TemporalGraph {
    pub async fn get_neighbours(&self, entity_id: &str, hops: u32) -> Result<SubGraph> {
        let _db_start = Instant::now();
        let mut visited_entities: HashSet<String> = HashSet::new();
        let mut collected_facts: Vec<Fact> = Vec::new();
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();

        visited_entities.insert(entity_id.to_string());
        queue.push_back((entity_id.to_string(), 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }

            // Find all non-expired facts where this entity is subject or object
            let current_id_str = current_id.clone();
            let mut rows = self
                .conn
                .query(
                    "SELECT id, subject_id, predicate, object_id, object_value, properties,
                            valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                     FROM facts
                     WHERE (subject_id = ?1 OR object_id = ?1)
                       AND expired_at IS NULL",
                    libsql::params![current_id_str],
                )
                .await?;

            let mut batch: Vec<Fact> = Vec::new();
            while let Some(row) = rows.next().await? {
                batch.push(row_to_fact(&row)?);
            }

            for fact in batch {
                // Collect connected entity IDs we haven't visited
                let neighbour_id = if fact.subject_id == current_id {
                    fact.object_id.clone()
                } else {
                    Some(fact.subject_id.clone())
                };

                collected_facts.push(fact);

                if let Some(nid) = neighbour_id {
                    if !visited_entities.contains(&nid) {
                        visited_entities.insert(nid.clone());
                        queue.push_back((nid, depth + 1));
                    }
                }
            }
        }

        // Deduplicate facts by id
        collected_facts.sort_by_key(|f| f.id);
        collected_facts.dedup_by_key(|f| f.id);

        // Load all discovered entities
        let mut entities = Vec::new();
        for eid in &visited_entities {
            if let Some(entity) = self.get_entity(eid).await? {
                entities.push(entity);
            }
        }

        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let entity_count = entities.len();
        let fact_count = collected_facts.len();
        histogram!("rql.db.get_neighbours_entities").record(entity_count as f64);
        histogram!("rql.db.get_neighbours_facts").record(fact_count as f64);
        histogram!("rql.db.get_neighbours_ms").record(_ms);
        tracing::info!(_ms, entity_count, fact_count, "kremory.db.get_neighbours");
        Ok(SubGraph {
            entities,
            facts: collected_facts,
        })
    }

    /// Temporal-bounded sibling of [`Self::get_neighbours`]. `as_of: None` runs
    /// the copy-identical per-hop `WHERE (subject_id = ?1 OR object_id = ?1)
    /// AND expired_at IS NULL` query `get_neighbours` runs today;
    /// `as_of: Some(t)` adds the SAME
    /// valid-time predicate `facts_at`/`entity_facts_at` already use:
    /// `valid_from <= ?t AND (valid_to IS NULL OR valid_to > ?t)`.
    /// `expired_at IS NULL` stays unconditional (system-time hard-retirement,
    /// orthogonal to `as_of`); `invalid_at` is deliberately NOT checked — a
    /// fact later flagged by the contradiction resolver but still
    /// valid-time-in-window at T must still surface for `as_of(T)`; that's
    /// the entire point of bi-temporal audit (prove what the record showed as
    /// true at T, even after correction).
    ///
    /// [`Self::get_neighbours`] is intentionally left UNTOUCHED above — this
    /// is a new sibling fn, not a modified shared primitive: other call sites,
    /// e.g. `speculative_cache.rs`'s prefetch, have no
    /// reason to carry a now-mandatory `as_of` parameter through their
    /// signatures.
    pub async fn get_neighbours_at(&self, params: GetNeighboursAtParams<'_>) -> Result<SubGraph> {
        let GetNeighboursAtParams {
            entity_id,
            hops,
            as_of,
            max_visited,
        } = params;
        let _db_start = Instant::now();
        let mut visited_entities: HashSet<String> = HashSet::new();
        let mut collected_facts: Vec<Fact> = Vec::new();
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();

        visited_entities.insert(entity_id.to_string());
        queue.push_back((entity_id.to_string(), 0));

        let as_of_str = as_of.map(|t| t.to_rfc3339());

        while let Some((current_id, depth)) = queue.pop_front() {
            // In-BFS fan-out cap. Checked at
            // the TOP of the loop (after pop, before processing) so `hops == 1`
            // is byte-identical — a seed's direct neighbours are all enqueued in
            // the seed's OWN iteration, so this only trips on a later iteration,
            // after every hop-1 neighbour is already visited; at `hops >= 2` it
            // stops the expansion BEFORE any depth-2 query, bounding hub-explosion.
            if let Some(cap) = max_visited {
                if visited_entities.len() >= cap {
                    break;
                }
            }
            if depth >= hops {
                continue;
            }

            let current_id_str = current_id.clone();
            let mut rows = match &as_of_str {
                None => {
                    self.conn
                        .query(
                            "SELECT id, subject_id, predicate, object_id, object_value, properties,
                                    valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                                memory_type, content_hash, access_count
                             FROM facts
                             WHERE (subject_id = ?1 OR object_id = ?1)
                               AND expired_at IS NULL",
                            libsql::params![current_id_str],
                        )
                        .await?
                }
                Some(t) => {
                    self.conn
                        .query(
                            "SELECT id, subject_id, predicate, object_id, object_value, properties,
                                    valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                                memory_type, content_hash, access_count
                             FROM facts
                             WHERE (subject_id = ?1 OR object_id = ?1)
                               AND expired_at IS NULL
                               AND valid_from <= ?2
                               AND (valid_to IS NULL OR valid_to > ?2)",
                            libsql::params![current_id_str, t.clone()],
                        )
                        .await?
                }
            };

            let mut batch: Vec<Fact> = Vec::new();
            while let Some(row) = rows.next().await? {
                batch.push(row_to_fact(&row)?);
            }

            for fact in batch {
                let neighbour_id = if fact.subject_id == current_id {
                    fact.object_id.clone()
                } else {
                    Some(fact.subject_id.clone())
                };

                collected_facts.push(fact);

                if let Some(nid) = neighbour_id {
                    if !visited_entities.contains(&nid) {
                        visited_entities.insert(nid.clone());
                        queue.push_back((nid, depth + 1));
                    }
                }
            }
        }

        // Deduplicate facts by id
        collected_facts.sort_by_key(|f| f.id);
        collected_facts.dedup_by_key(|f| f.id);

        // Load all discovered entities
        let mut entities = Vec::new();
        for eid in &visited_entities {
            if let Some(entity) = self.get_entity(eid).await? {
                entities.push(entity);
            }
        }

        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        let entity_count = entities.len();
        let fact_count = collected_facts.len();
        // Own histograms (not shared with get_neighbours) mirroring facts_at's
        // rql.db.facts_at_ms/_count pattern exactly (observability-first-class).
        histogram!("rql.db.get_neighbours_at_ms").record(_ms);
        histogram!("rql.db.get_neighbours_at_count").record(fact_count as f64);
        tracing::info!(
            _ms,
            entity_count,
            fact_count,
            as_of_applied = as_of.is_some(),
            "kremory.db.get_neighbours_at"
        );
        Ok(SubGraph {
            entities,
            facts: collected_facts,
        })
    }

    // === Embedding ===

    /// Return all active (non-expired) `potential_alias` facts in `group_id`.
    ///
    /// Used by the L7 dream-phase `resolve_pending_aliases` pass to identify
    /// alias candidates for confirmation or revocation.
    pub async fn get_alias_facts_in_group(&self, group_id: &str) -> Result<Vec<Fact>> {
        let _db_start = Instant::now();
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id,
                        confidence, source_episode_id, memory_type, content_hash, access_count
                 FROM facts
                 WHERE predicate = 'potential_alias'
                   AND group_id = ?1
                   AND expired_at IS NULL",
                libsql::params![group_id],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        let _ms = _db_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("rql.db.get_alias_facts_in_group_ms").record(_ms);
        tracing::info!(_ms, group_id, "kremory.db.get_alias_facts_in_group");
        Ok(facts)
    }

    /// Get active (non-expired) facts for a subject + predicate combination,
    /// scoped to `group_id` (TD-254). `subject_id` alone is NOT unique across
    /// namespaces — entities use a composite `(id, group_id)` primary key
    /// specifically so the same name can exist independently in different
    /// namespaces. Before this fix, this query had no `group_id` filter at
    /// all, so contradiction detection's candidate pool (the caller of this
    /// fn) could pull in and potentially invalidate an unrelated namespace's
    /// facts whenever a subject/predicate name collided across namespaces —
    /// a real cross-namespace data leak, not a theoretical one (a common
    /// short name like "status" or "the team" is enough).
    pub async fn get_facts_by_subject_predicate(
        &self,
        params: GetFactsBySubjectPredicateParams<'_>,
    ) -> Result<Vec<Fact>> {
        let GetFactsBySubjectPredicateParams {
            subject_id,
            predicate,
            group_id,
        } = params;
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE subject_id = ?1 AND predicate = ?2 AND expired_at IS NULL AND group_id = ?3",
                libsql::params![subject_id, predicate, group_id],
            )
            .await?;
        let mut facts = Vec::new();
        while let Some(row) = rows.next().await? {
            facts.push(row_to_fact(&row)?);
        }
        Ok(facts)
    }

    // === petgraph export ===

    pub async fn to_petgraph(&self) -> Result<petgraph::graph::DiGraph<String, (String, i64)>> {
        use petgraph::graph::DiGraph;
        use std::collections::HashMap;

        let entities = self.list_entities().await?;
        let mut graph: DiGraph<String, (String, i64)> = DiGraph::new();
        let mut node_index: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

        for entity in &entities {
            let idx = graph.add_node(entity.id.clone());
            node_index.insert(entity.id.clone(), idx);
        }

        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_id, object_value, properties,
                        valid_from, valid_to, recorded_at, expired_at, invalid_at, group_id, confidence, source_episode_id,
                        memory_type, content_hash, access_count
                 FROM facts
                 WHERE object_id IS NOT NULL
                   AND expired_at IS NULL",
                (),
            )
            .await?;

        while let Some(row) = rows.next().await? {
            let fact = row_to_fact(&row)?;
            if let Some(ref oid) = fact.object_id {
                let src = node_index.get(&fact.subject_id);
                let dst = node_index.get(oid);
                if let (Some(&s), Some(&d)) = (src, dst) {
                    graph.add_edge(s, d, (fact.predicate.clone(), fact.id));
                }
            }
        }

        Ok(graph)
    }

    /// Find all facts that have no embedding and return their IDs + object_value.
    ///
    /// # SQLite-first ordering invariant
    ///
    /// SQLite commit MUST precede vector write. If a crash occurs
    /// between SQLite commit and vector write, all facts with NULL embedding can
    /// be identified via this function and re-embedded by the caller. The reverse
    /// ordering (vector-first) has NO recovery path — this is the entire
    /// justification for the SQLite-first contract.
    /// Return all non-expired facts that are missing a vector embedding.
    ///
    /// Tuple layout: `(fact_id, subject_id, predicate, object_value, object_id)`.
    ///
    /// Both `object_value` and `object_id` are included so callers can build the
    /// text-to-embed with the best available object representation:
    /// prefer `object_id` (entity reference) over `object_value` (literal string)
    /// when constructing the embedding input.
    pub async fn facts_missing_embeddings(
        &self,
    ) -> Result<Vec<(i64, String, String, Option<String>, Option<String>)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, subject_id, predicate, object_value, object_id FROM facts WHERE embedding IS NULL AND expired_at IS NULL",
                (),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: i64 = row.get(0)?;
            let subject_id: String = row.get(1)?;
            let predicate: String = row.get(2)?;
            let object_value: Option<String> = row.get(3)?;
            let object_id: Option<String> = row.get(4)?;
            out.push((id, subject_id, predicate, object_value, object_id));
        }
        Ok(out)
    }

    /// Delete an entity and ALL dependent rows atomically.
    ///
    /// Deletes in dependency order within a single `BEGIN IMMEDIATE` transaction:
    /// 1. `entities_fts` — standalone FTS5 virtual table; no FK cascade.
    /// 2. `episodic_edges` — FK → `entities(id)` (no ON DELETE CASCADE).
    /// 3. `facts` — FK → `entities(id)` as subject/object (no ON DELETE CASCADE).
    /// 4. `entities` — parent row.
    ///
    /// If any step fails the transaction is rolled back and no rows are changed.
    ///
    /// Returns `true` when the entity existed and was deleted; `false` when the
    /// entity was not found (idempotent — not an error).
    pub async fn forget_entity(&self, entity_id: &str) -> Result<bool> {
        let id = libsql::Value::Text(entity_id.to_owned());
        let guard = self.begin_immediate_if_needed().await?;

        // 1. FTS — standalone FTS5; no FK cascade, must delete first.
        let fts_result = self
            .conn
            .execute(
                "DELETE FROM entities_fts WHERE entity_id = ?1",
                libsql::params![id.clone()],
            )
            .await;

        // 2. Episodic edges referencing this entity.
        let edges_result = if fts_result.is_ok() {
            self.conn
                .execute(
                    "DELETE FROM episodic_edges WHERE entity_id = ?1",
                    libsql::params![id.clone()],
                )
                .await
        } else {
            fts_result
        };

        // 3. Facts where this entity is subject or object.
        let facts_result = if edges_result.is_ok() {
            self.conn
                .execute(
                    "DELETE FROM facts WHERE subject_id = ?1 OR object_id = ?1",
                    libsql::params![id.clone()],
                )
                .await
        } else {
            edges_result
        };

        // 4. Entity row itself.
        if let Err(e) = facts_result {
            guard.rollback().await?;
            return Err(e.into());
        }

        let deleted = self
            .conn
            .execute("DELETE FROM entities WHERE id = ?1", libsql::params![id])
            .await;

        match deleted {
            Err(e) => {
                guard.rollback().await?;
                Err(e.into())
            }
            Ok(n) => {
                let found = n > 0;
                guard.commit().await?;
                tracing::info!(entity_id, found, "kremory.db.forget_entity");
                Ok(found)
            }
        }
    }

    /// Delete up to 250 entities in transactional 100-item chunks.
    ///
    /// Each chunk of up to 100 IDs is wrapped in its own `BEGIN IMMEDIATE`
    /// transaction. Deletion order per chunk: entities_fts → episodic_edges →
    /// facts_fts → facts → entities. The `facts_fts` step closes a
    /// pre-existing gap: this hard-delete path previously cleaned
    /// `entities_fts` but not `facts_fts` — the shadow was only ever
    /// purged by the dream archive path (`archive.rs::move_fact`).
    /// It MUST run before the `facts` DELETE so the resolving subquery still
    /// sees the rows about to be removed.
    ///
    /// Returns what was actually deleted, per table — not just the entity count.
    /// A caller reporting "0 removed" while facts and edges went with them is
    /// telling a true fact about the wrong noun (TD-247).
    pub async fn batch_forget(&self, entity_ids: &[String]) -> Result<BatchForgetCounts> {
        const CHUNK_SIZE: usize = 100;
        let mut counts = BatchForgetCounts::default();

        for chunk in entity_ids.chunks(CHUNK_SIZE) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let params: Vec<libsql::Value> = chunk
                .iter()
                .map(|s| libsql::Value::Text(s.clone()))
                .collect();

            let guard = self.begin_immediate_if_needed().await?;

            // Execute the four DELETE statements; collect the first error.
            macro_rules! try_delete {
                ($sql:expr, $p:expr) => {
                    match self.conn.execute(&$sql, $p).await {
                        Ok(n) => n,
                        Err(e) => {
                            guard.rollback().await?;
                            return Err(e.into());
                        }
                    }
                };
            }

            // 1. FTS.
            try_delete!(
                format!("DELETE FROM entities_fts WHERE entity_id IN ({placeholders})"),
                params.clone()
            );

            // 2. Episodic edges.
            let edges = try_delete!(
                format!("DELETE FROM episodic_edges WHERE entity_id IN ({placeholders})"),
                params.clone()
            );

            // 2.5. Facts FTS shadow rows — closes the batch_forget↔facts_fts
            //      gap: facts_fts was previously cleaned ONLY by the dream
            //      archive path (`core/dream/consolidation/archive.rs::move_fact`),
            //      never by this hard-delete path. MUST run BEFORE step 3's
            //      `facts` DELETE below — the subquery needs the `facts` rows
            //      to still exist to resolve which fact ids are about to be
            //      purged. Two IN clauses (mirrors step 3) → params doubled.
            let sql_facts_fts = format!(
                "DELETE FROM facts_fts WHERE CAST(fact_id AS INTEGER) IN \
                 (SELECT id FROM facts WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders}))"
            );
            let mut doubled_for_fts = params.clone();
            doubled_for_fts.extend_from_slice(&params);
            try_delete!(sql_facts_fts, doubled_for_fts);

            // 3. Facts (subject or object).
            //    Two IN clauses → params must be doubled.
            let sql_facts = format!(
                "DELETE FROM facts WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})"
            );
            let mut doubled = params.clone();
            doubled.extend_from_slice(&params);
            let facts = try_delete!(sql_facts, doubled);

            // 4. Entity rows.
            let n = try_delete!(
                format!("DELETE FROM entities WHERE id IN ({placeholders})"),
                params
            );

            guard.commit().await?;
            counts.entities += n;
            counts.facts += facts;
            counts.edges += edges;
        }

        tracing::info!(
            count = entity_ids.len(),
            entities = counts.entities,
            facts = counts.facts,
            edges = counts.edges,
            "kremory.db.batch_forget"
        );
        Ok(counts)
    }

    // ── Namespace policy (v0.1.4) ─────────────────────────────────────────────
    //
    // All three methods are `pub(crate)` — only the
    // `facade` layer (kremory::Memory::register_namespace) and the lazy-population
    // wiring should call them. External consumers go through the facade and get
    // validation + idempotency + tracing + race-safety.
}
