#![allow(clippy::unwrap_used, clippy::expect_used)]
//! TD-117 regression guard (Vera M2 adversarial review,
//! `.ai-docs/tech-debt/tech-debt-register.md`): `TemporalGraph::open_with_dim`
//! must hard-error when reopening an existing, populated DB with a different
//! `embedding_dim` than the one its stored vectors were written with.
//!
//! ## The bug
//!
//! `embedding_dim` is a caller-supplied `TemporalGraph::open_with_dim`
//! parameter with NO persisted-vs-current consistency check. Reopening a
//! pre-existing DB with a DIFFERENT `embedding_dim` than the vectors were
//! originally written with (e.g. swapping embedding models from a 384-dim one
//! to a 768-dim one) used to propagate stale-dim byte blobs into a
//! differently-typed `F32_BLOB(new_dim)` column, completely unvalidated —
//! producing either silent garbage nearest-neighbour search results or a
//! confusing low-level failure deep in a query, instead of a clear, early,
//! loud error at DB-open time (when the mismatch is knowable).
//!
//! Deterministic — no LLM calls needed.

use kremory::core::schema::TemporalGraph;

#[tokio::test]
async fn reopen_with_different_embedding_dim_hard_errors() {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("td117-dim-mismatch.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");

    // Phase A: create + populate the DB at embedding_dim=384 with a real
    // 384-dim embedding on a real row, then drop the handle to close the
    // connection before reopening the same file.
    {
        let graph = TemporalGraph::open_with_dim(path_str, 384)
            .await
            .expect("first open at embedding_dim=384 must succeed on a fresh DB");

        let now = chrono::Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO entities (id, group_id, entity_type_id, properties, recorded_at) \
                 VALUES ('e1', 'default', 0, '{}', ?1)",
                libsql::params![now],
            )
            .await
            .expect("insert entity");

        let embedding: Vec<f32> = vec![0.1_f32; 384];
        graph
            .set_entity_embedding("e1", &embedding)
            .await
            .expect("set 384-dim embedding on real row");
    }

    // Phase B: reopen the SAME file with embedding_dim=768. Must hard-error —
    // never silently succeed with a wrong-length blob planted in a 768-dim
    // column, and never panic.
    let reopened = TemporalGraph::open_with_dim(path_str, 768).await;

    match reopened {
        Err(kremory::CoreError::EmbeddingDimMismatch {
            requested_dim,
            detail,
        }) => {
            assert_eq!(
                requested_dim, 768,
                "requested_dim must reflect the reopen call's dim"
            );
            assert!(
                !detail.is_empty(),
                "detail must name which guard fired and the concrete numbers involved"
            );
        }
        Ok(_) => panic!(
            "expected Err(EmbeddingDimMismatch) when reopening a 384-dim DB with \
             embedding_dim=768, got Ok(_)"
        ),
        Err(other) => panic!(
            "expected Err(EmbeddingDimMismatch) when reopening a 384-dim DB with \
             embedding_dim=768, got a different error variant: {other}"
        ),
    }
}

/// A DB opened with embedding_dim=768 seeds the persisted registry at 768
/// even with zero rows written. Reopening the SAME (still-empty) file with
/// embedding_dim=384 must still hard-error — the sample-length guard can't
/// see anything (no rows exist to sample), but the persisted-registry guard
/// must catch it.
#[tokio::test]
async fn reopen_empty_db_with_different_embedding_dim_hard_errors() {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("td117-empty-dim-mismatch.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");

    {
        let _graph = TemporalGraph::open_with_dim(path_str, 768)
            .await
            .expect("first open at embedding_dim=768 must succeed on a fresh, empty DB");
    }

    let reopened = TemporalGraph::open_with_dim(path_str, 384).await;

    match reopened {
        Err(kremory::CoreError::EmbeddingDimMismatch {
            requested_dim,
            detail,
        }) => {
            assert_eq!(
                requested_dim, 384,
                "requested_dim must reflect the reopen call's dim"
            );
            assert!(
                !detail.is_empty(),
                "detail must name which guard fired and the concrete numbers involved"
            );
        }
        Ok(_) => panic!(
            "expected Err(EmbeddingDimMismatch) when reopening an empty 768-dim DB with \
             embedding_dim=384, got Ok(_)"
        ),
        Err(other) => panic!(
            "expected Err(EmbeddingDimMismatch) when reopening an empty 768-dim DB with \
             embedding_dim=384, got a different error variant: {other}"
        ),
    }
}

/// Reopening with the SAME embedding_dim must remain a clean no-op — this is
/// the overwhelming common case and must never regress.
#[tokio::test]
async fn reopen_with_same_embedding_dim_succeeds() {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join("td117-dim-consistent.db");
    let path_str = path.to_str().expect("tempdir path must be valid UTF-8");

    {
        let graph = TemporalGraph::open_with_dim(path_str, 384)
            .await
            .expect("first open at embedding_dim=384 must succeed on a fresh DB");

        let now = chrono::Utc::now().to_rfc3339();
        graph
            .conn
            .execute(
                "INSERT INTO entities (id, group_id, entity_type_id, properties, recorded_at) \
                 VALUES ('e1', 'default', 0, '{}', ?1)",
                libsql::params![now],
            )
            .await
            .expect("insert entity");

        let embedding: Vec<f32> = vec![0.1_f32; 384];
        graph
            .set_entity_embedding("e1", &embedding)
            .await
            .expect("set 384-dim embedding on real row");
    }

    TemporalGraph::open_with_dim(path_str, 384)
        .await
        .expect("reopen with the same embedding_dim must succeed");
}
