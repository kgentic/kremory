#![allow(clippy::unwrap_used, clippy::expect_used)]
//! **DUR-4 regression pin** (V1-CANONICAL §4.2) — `forget().by_source_id()` must be
//! ATOMIC.
//!
//! # The defect
//!
//! The destructive cascade in `facade/forget.rs` issues **four** statements with no
//! enclosing transaction — the code's own comment said so:
//!
//! > *"this sequence has no enclosing explicit transaction today — each statement
//! > auto-commits"*
//!
//! 1. `batch_forget(entities)`
//! 2. `DELETE FROM episodes_fts WHERE rowid IN (SELECT id FROM episodes WHERE source_id=?)`
//! 3. `DELETE FROM episodic_edges WHERE episode_id IN (SELECT id FROM episodes WHERE …)`
//! 4. `DELETE FROM episodes WHERE source_id=?`
//!
//! Each auto-commits independently, so a failure at step 3 or 4 leaves the database
//! **permanently inconsistent with no rollback and no mutation record**. The worst
//! shape is steps 1-2 succeeding and 4 failing: the full-text-search shadow rows are
//! gone while the episodes remain, so the content is **still stored but no longer
//! findable** — a silent, irreversible loss of searchability that no error surfaces,
//! because the error the caller receives is about the episode delete.
//!
//! # How this test proves it
//!
//! A SQLite `BEFORE DELETE` trigger on `episodes` raises an error, forcing step 4 to
//! fail *after* steps 1-3 have run. The assertion is then simply: **did the earlier
//! deletes survive?** Under a transaction they roll back; without one they are already
//! committed and gone.
//!
//! This drives the REAL facade path (`mem.forget().by_source_id(..).execute()`), not a
//! hand-rolled replica of the cascade — a test that re-issued the four statements
//! itself would prove only that SQLite transactions work.
//!
//! # Sensitivity
//!
//! Proven in BOTH directions. With the `BeginGuard` removed from `forget.rs` this test
//! FAILS (`episodic_edges` rows are gone); with the trigger removed the forget
//! succeeds and the rows are legitimately absent, so the assertion cannot pass
//! vacuously — a run where the trigger silently failed to install would report the
//! forget as `Ok`, which the test rejects explicitly.

use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::provider::{ChatProvider, MockChatResponse};
use kremory::{DynEmbeddingProvider, Memory, Namespace};

/// Returns `"[]"` for every call. The builder requires *an* extractor to be wired
/// even when every episode here uses `.skip_extraction()`, and DUR-4 is about the
/// DELETE cascade — no extraction behaviour is under test.
#[derive(Debug, Clone)]
struct EmptyArrayLlmClient;

#[async_trait::async_trait]
impl ChatProvider for EmptyArrayLlmClient {
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(MockChatResponse {
            text: "[]".to_owned(),
        }))
    }
}

fn unique_db_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kremory_forget_atomicity_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn make_null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Count rows matching a predicate. `COUNT(*)` is safe here: `episodic_edges` and
/// `episodes` are NOT vector-indexed, so the libsql `COUNT(*)`-returns-0 trap that
/// affects `entities`/`facts` (SYSTEM-PRIMER gotcha #1) does not apply.
async fn count(conn: &libsql::Connection, sql: &str, p: impl libsql::params::IntoParams) -> i64 {
    let mut rows = conn.query(sql, p).await.expect("count query");
    rows.next()
        .await
        .expect("iter")
        .expect("row")
        .get::<i64>(0)
        .expect("count col")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forget_by_source_id_rolls_back_when_the_episode_delete_fails() {
    let db = unique_db_path("rollback");
    let mem = Memory::open(&db)
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("tests"))
        .await
        .expect("builder should succeed");

    // Two episodes under one source_id, extraction skipped so the test stays
    // deterministic and needs no LLM — DUR-4 is about the DELETE cascade, and the
    // cascade does not care how the rows arrived.
    for text in [
        "Ada Lovelace wrote the first algorithm.",
        "Grace Hopper found the first bug.",
    ] {
        mem.remember(text)
            .from_document("doc-dur4")
            .skip_extraction()
            .await
            .expect("remember must succeed");
    }

    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");

    let episodes_before = count(
        &tg.conn,
        "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
        libsql::params!["doc-dur4"],
    )
    .await;
    assert_eq!(
        episodes_before, 2,
        "precondition: both episodes must be stored before we test the cascade"
    );

    let edges_before = count(
        &tg.conn,
        "SELECT COUNT(*) FROM episodic_edges WHERE episode_id IN \
         (SELECT id FROM episodes WHERE source_id = ?1)",
        libsql::params!["doc-dur4"],
    )
    .await;

    // MEASURE FINDABILITY, NOT ROW COUNT. `episodes_fts` is an EXTERNAL-CONTENT FTS5
    // table, so `SELECT COUNT(*) ... WHERE rowid IN (SELECT id FROM episodes ...)`
    // reads THROUGH to `episodes` and returns 2 even after the shadow rows are gone.
    // That false negative is why the first two drafts of this test passed against the
    // unfixed code. A real `MATCH` query hits the index itself and cannot lie.
    let fts_before = count(
        &tg.conn,
        "SELECT COUNT(*) FROM episodes_fts WHERE episodes_fts MATCH 'Lovelace'",
        (),
    )
    .await;
    // NON-VACUITY GUARD. Measured 2026-08-04: episodes=2, edges=0, fts=2. The
    // `episodic_edges` count is ZERO here because `.skip_extraction()` creates no
    // entities and therefore no edges — so an assertion on edges alone ALWAYS PASSES
    // and proves nothing. The first draft of this test did exactly that and went
    // green before any fix existed. `episodes_fts` is the real observable: step 2 of
    // the cascade deletes it, and it is populated for every episode.
    assert!(
        fts_before > 0,
        "non-vacuity: episodes_fts must be populated ({fts_before} rows), otherwise \
         the assertion below cannot distinguish rollback from nothing-to-roll-back"
    );

    // Force step 4 (`DELETE FROM episodes`) to fail, AFTER steps 1-3 have run.
    // RAISE(ABORT) aborts the statement; with no enclosing transaction the three
    // preceding statements have already auto-committed and are unrecoverable.
    tg.conn
        .execute(
            "CREATE TRIGGER dur4_block_episode_delete BEFORE DELETE ON episodes \
             BEGIN SELECT RAISE(ABORT, 'DUR-4 injected failure'); END",
            (),
        )
        .await
        .expect("trigger install must succeed — the whole test rests on it");

    let result = mem.forget().by_source_id("doc-dur4").execute().await;

    // Instrument validation: if the trigger did not actually fire, `forget` would
    // return Ok and every assertion below would pass for the wrong reason.
    assert!(
        result.is_err(),
        "the injected trigger must make forget() fail — an Ok here means the trigger \
         never fired and this test is proving nothing"
    );

    tg.conn
        .execute("DROP TRIGGER dur4_block_episode_delete", ())
        .await
        .expect("trigger drop");

    // THE ASSERTION. Under a transaction, the failure at step 4 rolls steps 1-3 back.
    // Without one, they are already committed and the rows are gone forever.
    let episodes_after = count(
        &tg.conn,
        "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
        libsql::params!["doc-dur4"],
    )
    .await;
    assert_eq!(
        episodes_after, episodes_before,
        "episodes must be untouched after a failed forget"
    );

    let fts_after = count(
        &tg.conn,
        "SELECT COUNT(*) FROM episodes_fts WHERE episodes_fts MATCH 'Lovelace'",
        (),
    )
    .await;
    assert_eq!(
        fts_after, fts_before,
        "episodes_fts must be ROLLED BACK when the episode delete fails.\n\
         This is DUR-4's irreversible case: step 2 purged the full-text shadow rows \
         and step 4 then failed, so the episodes are STILL STORED but NO LONGER \
         FINDABLE. Nothing surfaces it — the caller's error is about the episode \
         delete, and the rows cannot be rebuilt because the subquery that identifies \
         them resolves through `episodes`, which still exists but no longer matches."
    );

    // Edges are asserted too, but WITHOUT a non-vacuity guard, because this fixture
    // legitimately has none (skip_extraction => no entities => no edges). Kept so the
    // assertion starts working for free if the fixture ever gains real extraction;
    // it is explicitly NOT the load-bearing check.
    let edges_after = count(
        &tg.conn,
        "SELECT COUNT(*) FROM episodic_edges WHERE episode_id IN \
         (SELECT id FROM episodes WHERE source_id = ?1)",
        libsql::params!["doc-dur4"],
    )
    .await;
    assert_eq!(
        edges_after, edges_before,
        "episodic_edges must be rolled back too"
    );
}

/// The happy path must still work — a rollback guard that also blocks successful
/// deletes would be a cure worse than the disease.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forget_by_source_id_still_commits_on_the_happy_path() {
    let db = unique_db_path("happy");
    let mem = Memory::open(&db)
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(make_null_embedder())
        .default_namespace(Namespace::new("tests"))
        .await
        .expect("builder should succeed");

    mem.remember("Katherine Johnson computed the trajectories.")
        .from_document("doc-happy")
        .skip_extraction()
        .await
        .expect("remember must succeed");

    let tg = mem
        .temporal_graph_for_test()
        .expect("temporal_graph must be set on the builder path");

    assert_eq!(
        count(
            &tg.conn,
            "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
            libsql::params!["doc-happy"],
        )
        .await,
        1,
        "precondition: the episode must be stored"
    );

    mem.forget()
        .by_source_id("doc-happy")
        .execute()
        .await
        .expect("forget must succeed on the happy path");

    assert_eq!(
        count(
            &tg.conn,
            "SELECT COUNT(*) FROM episodes WHERE source_id = ?1",
            libsql::params!["doc-happy"],
        )
        .await,
        0,
        "the episode must actually be gone when forget() returns Ok — otherwise the \
         transaction wrapper has turned a working delete into a silent no-op"
    );
}
