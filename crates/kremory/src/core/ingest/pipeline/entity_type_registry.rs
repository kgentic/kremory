//! Which entity types this ingest is allowed to use (TD-045 split out of
//! `ingest_with.rs`).
//!
//! Two sources, and the precedence between them is the whole content of this
//! module: a caller-supplied override, or whatever the namespace already has in
//! the database. The override is PERSISTED on first use and then reconciled on
//! every later call, so a caller who passes the same list twice does not create a
//! second copy, and a caller who passes a DIFFERENT list learns that the namespace
//! already disagrees with them.

use crate::core::entity_types::{EntityTypeRegistry, EntityTypeSpec};
use crate::core::error::Result;
use crate::core::ingest::Engine;
use crate::core::provider::{ChatProvider, EmbeddingProvider};

impl<L: ChatProvider + 'static, Emb: EmbeddingProvider> Engine<L, Emb> {
    /// Resolve the entity-type registry for this ingest.
    ///
    /// `overrides = None` reads the namespace's persisted types. `Some(specs)` seeds
    /// them on a first call and reconciles against them afterwards.
    pub(super) async fn resolve_entity_type_registry(
        &self,
        overrides: Option<&[EntityTypeSpec]>,
        effective_gid: &str,
    ) -> Result<EntityTypeRegistry> {
        let registry = if let Some(override_specs) = overrides {
            let db_registry =
                EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?;
            if db_registry.is_empty() {
                // First-call persistence: seed DB from override.
                crate::core::entity_types::upsert_entity_types(
                    &self.graph.conn,
                    effective_gid,
                    override_specs,
                )
                .await?;
                metrics::counter!(
                    "rql.ingest.registry_override_applied",
                    "persisted" => "true",
                )
                .increment(1);
            } else {
                // Additive merge: persist any override types missing from DB.
                //
                // Was previously "ephemeral, do not touch DB" but that
                // created an entity-type/JOIN hole: an entity row stored with
                // entity_type_id = override-only id had no entity_types row →
                // SQL COALESCE(et.name, 'Entity') resolved label='Entity' at
                // read time regardless of the stored integer id. The right fix is
                // additive merge — INSERT OR IGNORE each missing override
                // spec, preserving previously-stored rows.
                let mut newly_persisted: usize = 0;
                for spec in override_specs.iter() {
                    if db_registry.name_to_id(&spec.name).is_none() {
                        self.graph
                            .conn
                            .execute(
                                "INSERT OR IGNORE INTO entity_types \
                                 (group_id, id, name, description) \
                                 VALUES (?1, ?2, ?3, ?4)",
                                libsql::params![
                                    effective_gid,
                                    spec.id as i64,
                                    spec.name.clone(),
                                    spec.description.clone()
                                ],
                            )
                            .await
                            .map_err(|e| {
                                crate::core::error::Error::Other(anyhow::anyhow!(
                                    "additive override persist failed for spec '{}': {e}",
                                    spec.name
                                ))
                            })?;
                        newly_persisted += 1;
                    }
                }
                metrics::counter!(
                    "rql.ingest.registry_override_applied",
                    "persisted" => if newly_persisted > 0 { "additive" } else { "false" },
                )
                .increment(1);
                metrics::histogram!("rql.ingest.registry_override_additive_count")
                    .record(newly_persisted as f64);
            }
            EntityTypeRegistry::from_specs(override_specs.to_vec())
        } else {
            let db_registry =
                EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?;
            // Builder-seed: if the builder set `allowed_entity_types`,
            // any type not already in the registry for this group_id is registered
            // additively via `label_to_id_or_register`. This runs AFTER
            // `ensure_default_types_seeded` so the registry is never empty here;
            // the merge is additive-only (INSERT OR IGNORE via the same race-safe
            // MAX(id)+1 path that Pass 0 uses). No caller-specified id → SQLite
            // assigns the next free id, avoiding position-based collisions with
            // future Pass 0 writes.
            if !self.config.allowed_entity_types.is_empty() {
                let mut new_types_seeded: usize = 0;
                for name in &self.config.allowed_entity_types {
                    // Use case-insensitive match here so we don't fire the
                    // seed branch when the DB already has the canonical form under a
                    // different case (e.g. builder has "court", DB has "Court").
                    // `label_to_id_or_register` does a case-insensitive DB lookup
                    // (COLLATE NOCASE path), so a case-sensitive `name_to_id` guard
                    // would count a "skip that never inserts" as a seed — a
                    // lying-counter failure mode.
                    let already_registered = db_registry
                        .specs()
                        .iter()
                        .any(|s| s.name.eq_ignore_ascii_case(name));
                    if !already_registered {
                        crate::core::entity_types::label_to_id_or_register(
                            crate::core::entity_types::LabelToIdOrRegisterParams {
                                conn: &self.graph.conn,
                                group_id: effective_gid,
                                registry: &db_registry,
                                label: name,
                            },
                        )
                        .await?;
                        new_types_seeded += 1;
                    }
                }
                if new_types_seeded > 0 {
                    metrics::counter!(
                        "rql.ingest.registry_builder_seed_applied",
                        "namespace" => effective_gid.to_string(),
                    )
                    .increment(1);
                    // Histogram gains namespace label matching the paired counter
                    // so operators can disaggregate by namespace.
                    metrics::histogram!(
                        "rql.ingest.registry_builder_seed_count",
                        "namespace" => effective_gid.to_string(),
                    )
                    .record(new_types_seeded as f64);
                    // Reload the registry so the derived allowed_entity_types_live
                    // below includes the newly-seeded builder types.
                    EntityTypeRegistry::load_for_group(&self.graph.conn, effective_gid).await?
                } else {
                    db_registry
                }
            } else {
                db_registry
            }
        };
        Ok(registry)
    }
}
