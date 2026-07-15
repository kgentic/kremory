// ─── Migration 024 ─────────────────────────────────────────────────────────

/// Migration 024 (TD-117, Vera M2 adversarial review,
/// `.ai-docs/tech-debt/tech-debt-register.md`): hard-error when the
/// caller-supplied `embedding_dim` disagrees with the embedding dimension
/// this store already contains data for.
///
/// ## The bug
///
/// `embedding_dim` is a caller-supplied `TemporalGraph::open_with_dim`
/// parameter with NO persisted-vs-current consistency check. Reopening a
/// pre-existing, populated DB with a different `embedding_dim` than the
/// vectors were originally written with (e.g. swapping embedding models from
/// a 384-dim one to a 768-dim one) silently propagates stale-dim byte blobs
/// into a differently-typed `F32_BLOB(new_dim)` column — the RAW `INSERT ...
/// SELECT embedding` copy in `migrate_023_vector_index_column_type`
/// (`defs_j.rs`) is spike-verified bit-preserving but was NEVER
/// length-checked against the target dim — producing either silent garbage
/// nearest-neighbour search results or a confusing low-level failure deep in
/// a query, instead of a clear, early, loud error at DB-open time, when the
/// mismatch is knowable.
///
/// ## Two independent guards
///
/// 1. **Sample check.** Before any DDL runs, sample one existing
///    `entities.embedding` / `facts.embedding` row (if either table already
///    exists and has a non-NULL embedding) and compare its `LENGTH(...)`
///    against `dim * 4` (4 bytes per f32 component — embeddings are always
///    stored as `Vec<f32>`). Catches a populated pre-existing DB reopened
///    with the wrong dim, regardless of whether the registry (below) has
///    ever been seeded — this is the load-bearing check for the exact bug
///    scenario above, and it runs BEFORE `migrate_023` can rebuild anything.
/// 2. **Persisted registry.** `embedding_dim_registry` is a single-row table
///    recording the `embedding_dim` this store was opened with the first
///    time this migration ever ran on it — which is, in practice, the dim
///    baked into the very first `CREATE TABLE entities (embedding
///    F32_BLOB(dim))`, since this migration is called before that DDL (see
///    the call site in `TemporalGraph::run_migrations`). Every subsequent
///    open compares the caller's `dim` against the persisted value and
///    hard-errors on mismatch. Catches an EMPTY (zero-row) DB reopened with a
///    different dim, which the sample check (1) cannot see (nothing to
///    sample).
///
/// Both guards return `Error::EmbeddingDimMismatch` — the open path aborts
/// before any table is created or rebuilt with the wrong dim.
///
/// ## Idempotency
///
/// `CREATE TABLE IF NOT EXISTS`. The very first run on a given DB seeds the
/// registry row (matching whatever dim this call used) and returns `Ok`.
/// Every subsequent run on a consistent dim is a pure read + compare (no
/// writes). Running this migration twice in the same process (e.g. via
/// `run_migrations_again_for_test`) with an unchanged `embedding_dim` is a
/// clean no-op.
pub(crate) async fn migrate_024_verify_embedding_dim(
    conn: &libsql::Connection,
    dim: usize,
) -> crate::core::error::Result<()> {
    async fn table_exists(
        conn: &libsql::Connection,
        name: &str,
    ) -> std::result::Result<bool, libsql::Error> {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                libsql::params![name],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    }

    fn other_err(context: &str, e: impl std::fmt::Display) -> crate::core::error::Error {
        crate::core::error::Error::Other(anyhow::anyhow!(
            "migrate_024_verify_embedding_dim: {context}: {e}"
        ))
    }

    let expected_bytes: i64 = i64::try_from(dim)
        .ok()
        .and_then(|d| d.checked_mul(4))
        .ok_or_else(|| {
            crate::core::error::Error::Config(format!(
                "embedding_dim {dim} is too large to validate (overflow computing dim * 4)"
            ))
        })?;

    // ── Guard 1: sample an existing stored embedding's byte length ─────────
    for table in ["entities", "facts"] {
        if !table_exists(conn, table)
            .await
            .map_err(|e| other_err(&format!("checking existence of '{table}'"), e))?
        {
            continue;
        }

        let mut rows = conn
            .query(
                &format!(
                    "SELECT LENGTH(embedding) FROM {table} WHERE embedding IS NOT NULL LIMIT 1"
                ),
                (),
            )
            .await
            .map_err(|e| other_err(&format!("sampling embedding length from '{table}'"), e))?;

        let Some(row) = rows
            .next()
            .await
            .map_err(|e| other_err(&format!("reading sampled row from '{table}'"), e))?
        else {
            continue;
        };

        let actual_bytes: i64 = row
            .get(0)
            .map_err(|e| other_err(&format!("reading LENGTH(embedding) from '{table}'"), e))?;

        if actual_bytes != expected_bytes {
            return Err(crate::core::error::Error::EmbeddingDimMismatch {
                requested_dim: dim,
                detail: format!(
                    "an existing '{table}.embedding' row is {actual_bytes} bytes long, which \
                     is not {expected_bytes} bytes (embedding_dim {dim} * 4)"
                ),
            });
        }
    }

    // ── Guard 2: persisted embedding_dim registry ───────────────────────────
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_dim_registry (
            id             INTEGER PRIMARY KEY CHECK (id = 1),
            embedding_dim  INTEGER NOT NULL
        )",
        (),
    )
    .await
    .map_err(|e| other_err("creating embedding_dim_registry", e))?;

    let mut rows = conn
        .query(
            "SELECT embedding_dim FROM embedding_dim_registry WHERE id = 1",
            (),
        )
        .await
        .map_err(|e| other_err("reading embedding_dim_registry", e))?;

    match rows
        .next()
        .await
        .map_err(|e| other_err("iterating embedding_dim_registry", e))?
    {
        Some(row) => {
            let persisted_raw: i64 = row
                .get(0)
                .map_err(|e| other_err("reading embedding_dim_registry.embedding_dim", e))?;
            let persisted = usize::try_from(persisted_raw).map_err(|e| {
                other_err(
                    "embedding_dim_registry.embedding_dim is negative — corrupt store",
                    e,
                )
            })?;
            if persisted != dim {
                return Err(crate::core::error::Error::EmbeddingDimMismatch {
                    requested_dim: dim,
                    detail: format!(
                        "this store's embedding_dim_registry records embedding_dim={persisted} \
                         from when it was first opened"
                    ),
                });
            }
        }
        None => {
            let dim_i64 = i64::try_from(dim)
                .map_err(|e| other_err("embedding_dim does not fit in i64 for persistence", e))?;
            conn.execute(
                "INSERT INTO embedding_dim_registry (id, embedding_dim) VALUES (1, ?1)",
                libsql::params![dim_i64],
            )
            .await
            .map_err(|e| other_err("seeding embedding_dim_registry", e))?;
        }
    }

    Ok(())
}
