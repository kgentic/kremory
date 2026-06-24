#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Phase G integration tests — TD-003 Episode struct ↔ table column symmetry.
//!
//! ## Acceptance criteria
//!
//! AC.1 — `recall_by_source_id` returns Episode with `source_id` field populated
//!         (matches the slug used at ingest). Regression: field was always None before G-2.
//! AC.2 — `recall_by_source_id` returns Episode with `content_hash` field populated
//!         (was hardcoded None before G-1 Migration 011). Must be a 64-hex-char SHA-256.
//! AC.3 — Migration 011 is idempotent: calling `run_migrations` twice must not error
//!         and must leave `content_hash` column present exactly once in `episodes`.
//! AC.4 — `content_hash` index exists after Migration 011 runs.

use std::sync::Arc;

use kremory::core::schema::TemporalGraph;
use kremory::{DynEmbeddingProvider, Memory, Namespace, SourceKind};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_episode_g_{tag}_{}_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

async fn make_memory(tag: &str) -> (Memory, std::path::PathBuf) {
    let path = unique_db_path(tag);
    let mem = Memory::open(path.clone())
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open must succeed");
    (mem, path)
}

async fn open_graph(tag: &str) -> (TemporalGraph, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir creation must succeed");
    let path = tmp.path().join(format!("kremory-mig-011-{tag}.db"));
    let path_str = path.to_str().expect("path must be valid UTF-8");
    let graph = TemporalGraph::open(path_str)
        .await
        .expect("TemporalGraph::open must succeed on fresh DB");
    (graph, tmp)
}

async fn table_columns(graph: &TemporalGraph, table: &str) -> Vec<String> {
    let sql = format!("PRAGMA table_info('{table}')");
    let mut rows = graph
        .conn
        .query(&sql, ())
        .await
        .unwrap_or_else(|_| panic!("PRAGMA table_info('{table}') must succeed"));
    let mut cols = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .expect("PRAGMA table_info row iteration must not error")
    {
        let name: String = row.get(1).expect("column name at index 1");
        cols.push(name);
    }
    cols
}

// ── AC.1: source_id field populated after recall_by_source_id ────────────────

/// `recall_by_source_id` must return Episodes with `source_id` == the slug
/// used at ingest. Regression: before G-2 the SELECT projection omitted
/// `source_id` and the field was always None.
#[tokio::test]
async fn episode_struct_carries_source_id_after_ingest() {
    let ns = Namespace::new("test-episode-rt-source-id");
    let (mem, _path) = make_memory("source_id_field").await;
    let slug = "g-test-source-id-001";

    mem.remember("Phase G round-trip test content for source_id field.")
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction()
        .await
        .expect("remember().from_source() must succeed");

    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");

    assert!(
        !episodes.is_empty(),
        "recall_by_source_id must return at least one episode"
    );

    let ep = &episodes[0];
    assert_eq!(
        ep.source_id.as_deref(),
        Some(slug),
        "Episode.source_id must equal the slug used at ingest \
         (regression: G-2 SELECT projection was missing source_id column)"
    );
}

// ── AC.2: content_hash field populated (64-hex SHA-256) ──────────────────────

/// `recall_by_source_id` must return Episodes with `content_hash` populated
/// as a 64-hex-character SHA-256 string. Regression: before G-1+G-2, the
/// column did not exist and the field was hardcoded None.
#[tokio::test]
async fn episode_struct_carries_content_hash_after_ingest() {
    let ns = Namespace::new("test-episode-rt-content-hash");
    let (mem, _path) = make_memory("content_hash_field").await;
    let slug = "g-test-content-hash-001";

    mem.remember("Phase G content hash round-trip test.")
        .from_source(slug, SourceKind::Document)
        .in_namespace(ns.clone())
        .skip_extraction()
        .await
        .expect("remember().from_source() must succeed");

    let episodes = mem
        .recall_by_source_id(slug, Some(ns))
        .await
        .expect("recall_by_source_id must not error");

    assert!(
        !episodes.is_empty(),
        "recall_by_source_id must return at least one episode"
    );

    let ep = &episodes[0];
    let hash = ep.content_hash.as_deref().unwrap_or_else(|| {
        panic!(
            "Episode.content_hash must be Some after Migration 011 backfill \
             (regression: field was hardcoded None before G-1)"
        )
    });
    assert_eq!(
        hash.len(),
        64,
        "content_hash must be a 64-hex-character SHA-256 string; got {hash:?}"
    );
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "content_hash must contain only hex digits; got {hash:?}"
    );
}

// ── AC.3: Migration 011 idempotency ──────────────────────────────────────────

/// Running `run_migrations` twice must be a no-op for Migration 011:
/// no errors and `content_hash` column must remain present.
#[tokio::test]
async fn migration_011_idempotent_double_apply() {
    let (graph, _tmp) = open_graph("idempotent").await;

    // Verify content_hash column present after first open().
    let cols_first = table_columns(&graph, "episodes").await;
    assert!(
        cols_first.iter().any(|c| c == "content_hash"),
        "pre-condition: episodes must have content_hash after first open(); \
         found columns: {cols_first:?}"
    );

    // Second migration run via test hook — must not error.
    graph.run_migrations_again_for_test().await.expect(
        "second run_migrations must be idempotent for Migration 011 — \
             PRAGMA table_info gate must detect content_hash already present and skip ALTER TABLE",
    );

    // Column shape unchanged.
    let cols_second = table_columns(&graph, "episodes").await;
    assert!(
        cols_second.iter().any(|c| c == "content_hash"),
        "episodes must still have content_hash after second migration run; \
         found: {cols_second:?}"
    );
}

// ── AC.4: content_hash index exists after Migration 011 ──────────────────────

/// `idx_episodes_content_hash` index must exist on the `episodes` table
/// after Migration 011 runs.
#[tokio::test]
async fn migration_011_content_hash_index_exists() {
    let (graph, _tmp) = open_graph("index_check").await;

    let mut rows = graph
        .conn
        .query(
            "SELECT name FROM sqlite_master \
             WHERE type='index' AND name='idx_episodes_content_hash'",
            (),
        )
        .await
        .expect("sqlite_master index query must succeed");

    let row = rows.next().await.expect("row iteration must not error");

    assert!(
        row.is_some(),
        "idx_episodes_content_hash index must exist on episodes table after Migration 011"
    );
}

// ── Static source-code gate ───────────────────────────────────────────────────

/// The migrations module must statically contain the Migration 011 function and
/// the key DDL strings that make the spec claims verifiable in CI.
///
/// TD-045: the former monolithic `migrations.rs` was split into the
/// `migrations/` module directory (`mod.rs` + `defs_*.rs`). This gate now
/// concatenates every `.rs` file in that directory so it is resilient to which
/// `defs_*.rs` holds Migration 011.
#[test]
fn migration_011_source_gates_present() {
    use std::path::Path;
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mig_dir = manifest_dir.join("src/core/migrations");
    let mut content = String::new();
    for entry in std::fs::read_dir(&mig_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", mig_dir.display()))
    {
        let path = entry.expect("migrations dir entry").path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            content.push_str(
                &std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display())),
            );
        }
    }

    assert!(
        content.contains("migrate_011_episodes_content_hash"),
        "migrations module must define `migrate_011_episodes_content_hash`"
    );
    assert!(
        content.contains("ALTER TABLE episodes ADD COLUMN content_hash TEXT"),
        "migrations module must contain the content_hash ALTER TABLE DDL"
    );
    assert!(
        content.contains("idx_episodes_content_hash"),
        "migrations module must contain the idx_episodes_content_hash index DDL"
    );
}
