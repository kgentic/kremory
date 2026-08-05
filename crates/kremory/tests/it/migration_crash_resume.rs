#![allow(clippy::unwrap_used, clippy::expect_used)]
//! **DUR-5 regression pin** (V1-CANONICAL §4.2) — a crash midway through a migration's
//! table swap must not brick the database on the next `open()`.
//!
//! # The defect
//!
//! Several migrations replace a table by rebuilding it:
//!
//! ```text
//! CREATE TABLE X_new (…)          -- idempotent
//! INSERT INTO X_new SELECT … FROM X
//! DROP TABLE X                    -- ← bare, no IF EXISTS
//! ALTER TABLE X_new RENAME TO X
//! ```
//!
//! The window between the `DROP` and the `RENAME` is not covered by a transaction (DDL
//! here runs with `PRAGMA foreign_keys = OFF`, statement-at-a-time). A crash there — a
//! `SIGKILL`, an OOM, a laptop lid closing — leaves `X` gone and `X_new` holding the
//! only copy of the data. What the next `open()` then did depended on the migration,
//! and none of the three outcomes was acceptable:
//!
//! | migration | resume gate | behaviour on the next `open()` before the fix |
//! |---|---|---|
//! | `migrate_004` (`defs_a.rs:172`) | present | skips the copy, reaches the bare `DROP`, **errors — `open()` fails forever** |
//! | `migrate_006` (`defs_h.rs:95,100`) | present | identical |
//! | `migrate_013` (`defs_d.rs`) | **absent** | proceeds as if fresh, **drops `entities_new`** — the only copy — then errors |
//!
//! `migrate_013` is the severe one: the data is destroyed by the RECOVERY attempt, not
//! by the crash. Its `else` arm treated a MISSING `entities` table as "not yet
//! migrated", which is indistinguishable from "half-swapped" without a second check.
//!
//! # How these tests prove it
//!
//! Each test manufactures the exact post-crash state — `X_new` populated, `X` dropped —
//! against a real database that has already been through every migration, then
//! **re-opens it through `Memory::open()`**, the actual consumer entry point.
//!
//! Driving the real `open()` rather than calling a migration function directly is the
//! point: it is the only way to prove the recovery works where it has to work, and it
//! covers the whole migration chain re-running over the repaired schema rather than one
//! function in isolation.
//!
//! # Sensitivity
//!
//! Proven by restoring the broken behaviour. With `IF EXISTS` reverted to a bare
//! `DROP TABLE entities` in `defs_a.rs`, `entities_survives_a_crash_between_drop_and_rename`
//! fails with `no such table: entities`. With `migrate_013`'s `resuming_partial_swap`
//! gate removed, the same test fails after the row count has already gone to zero.
//!
//! Each test also asserts a NON-VACUITY precondition — that rows existed before the
//! simulated crash — because "no rows lost" is trivially true of an empty table, and an
//! assertion that cannot distinguish recovery from nothing-to-recover proves nothing.

use std::sync::Arc;

use autoagents_llm::chat::{ChatMessage, ChatResponse, StructuredOutputFormat, Tool};
use autoagents_llm::error::LLMError;
use kremory::core::provider::{ChatProvider, MockChatResponse};
use kremory::{DynEmbeddingProvider, Memory, Namespace};

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
        "kremory_dur5_crash_resume_{}_{}_{}.db",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        seq
    ))
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

async fn open_memory(db: &std::path::Path) -> Memory {
    Memory::open(db)
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(null_embedder())
        .default_namespace(Namespace::new("dur5"))
        .await
        .expect("Memory::open must succeed")
}

/// `COUNT(*)` is unusable on `entities` and `facts` — the libsql vector index makes it
/// return 0 on populated tables (SYSTEM-PRIMER gotcha #1). Count rows by iterating a
/// non-aggregate SELECT instead, which is what the primer prescribes.
async fn row_count(conn: &libsql::Connection, table: &str) -> usize {
    let mut rows = conn
        .query(&format!("SELECT rowid FROM {table}"), ())
        .await
        .expect("select must succeed");
    let mut n = 0usize;
    while rows.next().await.expect("row iter").is_some() {
        n += 1;
    }
    n
}

async fn table_exists(conn: &libsql::Connection, name: &str) -> bool {
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
            libsql::params![name],
        )
        .await
        .expect("sqlite_master query");
    rows.next().await.expect("row iter").is_some()
}

/// Manufacture the post-crash state for `table`: a populated `{table}_new` scratch with
/// the table's REAL DDL, and no `{table}`. Returns the number of rows stranded.
///
/// # Why the DDL must be cloned exactly
///
/// The obvious shortcut, `CREATE TABLE {table}_new AS SELECT * FROM {table}`, is an
/// UNFAITHFUL simulation and the first draft of this test used it. `AS SELECT` copies
/// rows but discards column types, constraints and indexes, so the recovered table came
/// back structurally different from what a real crash would have left — and every test
/// failed for reasons that had nothing to do with DUR-5:
///
/// | table | failure | actual cause |
/// |---|---|---|
/// | `facts` | `migrate_023`: *"unexpected vector column type"* | `embedding`'s `F32_BLOB(n)` degraded to untyped |
/// | `episodic_edges` | `migrate_017`: *"no such column: entity_group_id"* | constraints/indexes lost |
/// | `entities` | `DROP TABLE` → *"SQL logic error"* | the vector index still referenced it |
///
/// A real crash leaves `{table}_new` exactly as the migration created it — with full
/// DDL — because the migration writes it with an explicit `CREATE TABLE (…)`. Cloning
/// the live table's `sql` from `sqlite_master` reproduces that faithfully, which is the
/// difference between testing the real system and testing a model of it.
async fn simulate_crash_between_drop_and_rename(conn: &libsql::Connection, table: &str) -> usize {
    let before = row_count(conn, table).await;
    assert!(
        before > 0,
        "NON-VACUITY: `{table}` must hold rows before the simulated crash, otherwise \
         'no rows lost' is trivially true and this test cannot distinguish a successful \
         recovery from an empty database"
    );

    // The real migrations wrap their swap in `PRAGMA foreign_keys = OFF` — without it
    // `DROP TABLE entities` fails with `FOREIGN KEY constraint failed` (787), because
    // `facts` and `episodic_edges` hold composite FKs into it. The simulation has to do
    // the same or it cannot reach the state being simulated.
    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .expect("fk off");

    // Clone the exact DDL, renaming only the table itself.
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
            libsql::params![table],
        )
        .await
        .expect("sqlite_master DDL query");
    let ddl: String = rows
        .next()
        .await
        .expect("row iter")
        .expect("the table must have DDL in sqlite_master")
        .get(0)
        .expect("sql column");
    drop(rows);
    // SQLite re-emits the name QUOTED after an `ALTER TABLE … RENAME` — and every table
    // here reached its current shape via exactly that — so the stored DDL reads
    // `CREATE TABLE "entities" (…)`, not `CREATE TABLE entities (…)`. Try the quoted
    // form first. (The unquoted arm is not dead: a table created directly by the base
    // DDL and never renamed keeps the bare form.)
    let scratch_ddl = if ddl.contains(&format!("TABLE \"{table}\"")) {
        ddl.replacen(
            &format!("TABLE \"{table}\""),
            &format!("TABLE \"{table}_new\""),
            1,
        )
    } else {
        ddl.replacen(&format!("TABLE {table}"), &format!("TABLE {table}_new"), 1)
    };
    // INSTRUMENT VALIDATION. This assertion has already earned its place: the first
    // draft matched only the unquoted form and silently produced a rewrite that was a
    // no-op, which would otherwise have surfaced as a baffling downstream error.
    assert!(
        scratch_ddl.contains(&format!("{table}_new")),
        "INSTRUMENT VALIDATION: the DDL rewrite did not take. Original was:\n{ddl}"
    );
    conn.execute(&scratch_ddl, ())
        .await
        .expect("scratch table creation must succeed");
    conn.execute(
        &format!("INSERT INTO {table}_new SELECT * FROM {table}"),
        (),
    )
    .await
    .expect("copying rows into the scratch table must succeed");

    // Indexes must go before the table does — libsql refuses to drop a table a vector
    // index still references ("SQL logic error"), which is exactly why the real
    // migrations issue their `DROP INDEX` statements first.
    let mut idx_rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?1 \
             AND sql IS NOT NULL",
            libsql::params![table],
        )
        .await
        .expect("index enumeration");
    let mut index_names: Vec<String> = Vec::new();
    while let Some(row) = idx_rows.next().await.expect("index row iter") {
        index_names.push(row.get(0).expect("index name"));
    }
    drop(idx_rows);
    for name in index_names {
        let _ = conn
            .execute(&format!("DROP INDEX IF EXISTS {name}"), ())
            .await;
    }

    conn.execute(&format!("DROP TABLE {table}"), ())
        .await
        .expect("dropping the live table must succeed");
    conn.execute("PRAGMA foreign_keys = ON", ())
        .await
        .expect("fk on");

    // INSTRUMENT VALIDATION: confirm the crash state is actually what we think it is.
    // If the DROP silently did nothing, every assertion afterwards would pass for the
    // wrong reason.
    assert!(
        !table_exists(conn, table).await,
        "the simulated crash must actually remove `{table}` — it is still present, so \
         this test is proving nothing"
    );
    assert_eq!(
        row_count(conn, &format!("{table}_new")).await,
        before,
        "the scratch table must hold every row before we test recovery"
    );

    before
}

/// Ingest enough to populate `entities`, `facts` and `episodic_edges`.
///
/// `.skip_extraction()` is deliberately NOT used: it creates episodes only, and the
/// tables under test would then be empty, tripping the non-vacuity guard above.
async fn seed(mem: &Memory) {
    for (s, p, o) in [
        ("Ada Lovelace", "wrote algorithms for", "Analytical Engine"),
        ("Grace Hopper", "found", "the first bug"),
    ] {
        mem.remember(format!("{s} {p} {o}."))
            .from_document("dur5-seed")
            .with_facts(vec![kremory::memory::types::StructuredFact {
                subject: s.to_owned(),
                predicate: p.to_owned(),
                object: o.to_owned(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .await
            .expect("remember must succeed");
    }
}

/// `migrate_004` (`defs_a.rs:237`) + `migrate_013` (`defs_d.rs:221`) both swap
/// `entities`. This is the severe case: before the fix, `migrate_013` DESTROYED
/// `entities_new` on the recovery attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn entities_survives_a_crash_between_drop_and_rename() {
    let db = unique_db_path("entities");
    let stranded;
    {
        let mem = open_memory(&db).await;
        seed(&mem).await;
        let tg = mem.temporal_graph_for_test().expect("temporal_graph");
        stranded = simulate_crash_between_drop_and_rename(&tg.conn, "entities").await;
    }

    // THE ASSERTION. Re-open through the real consumer entry point.
    let mem = Memory::open(&db)
        .with_llm(Arc::new(EmptyArrayLlmClient) as Arc<dyn ChatProvider>)
        .with_embedder(null_embedder())
        .default_namespace(Namespace::new("dur5"))
        .await;
    let mem = mem.expect(
        "DUR-5: `open()` must RECOVER from a crash between DROP and RENAME. A failure \
         here means the database is permanently unopenable — the migration chain cannot \
         complete and there is no path forward for the user short of deleting their data.",
    );

    let tg = mem.temporal_graph_for_test().expect("temporal_graph");
    assert!(
        table_exists(&tg.conn, "entities").await,
        "DUR-5: `entities` must exist after recovery — the rename never completed"
    );
    assert_eq!(
        row_count(&tg.conn, "entities").await,
        stranded,
        "DUR-5: every row stranded in `entities_new` must survive. A zero here is \
         migrate_013's catastrophic path: its gate read the missing `entities` as \
         'not yet migrated', so it ran `DROP TABLE IF EXISTS entities_new` and \
         destroyed the only surviving copy — data loss caused by the RECOVERY, not \
         by the crash."
    );
}

/// `migrate_006` (`defs_h.rs:184`) swaps `facts`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facts_survives_a_crash_between_drop_and_rename() {
    let db = unique_db_path("facts");
    let stranded;
    {
        let mem = open_memory(&db).await;
        seed(&mem).await;
        let tg = mem.temporal_graph_for_test().expect("temporal_graph");
        stranded = simulate_crash_between_drop_and_rename(&tg.conn, "facts").await;
    }

    let mem = open_memory(&db).await;
    let tg = mem.temporal_graph_for_test().expect("temporal_graph");
    assert!(
        table_exists(&tg.conn, "facts").await,
        "DUR-5: `facts` must exist after recovery"
    );
    assert_eq!(
        row_count(&tg.conn, "facts").await,
        stranded,
        "DUR-5: every fact stranded in `facts_new` must survive the recovery"
    );
}

/// `migrate_006` (`defs_h.rs:319`) swaps `episodic_edges`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn episodic_edges_survives_a_crash_between_drop_and_rename() {
    let db = unique_db_path("edges");
    let stranded;
    {
        let mem = open_memory(&db).await;
        seed(&mem).await;
        let tg = mem.temporal_graph_for_test().expect("temporal_graph");
        stranded = simulate_crash_between_drop_and_rename(&tg.conn, "episodic_edges").await;
    }

    let mem = open_memory(&db).await;
    let tg = mem.temporal_graph_for_test().expect("temporal_graph");
    assert!(
        table_exists(&tg.conn, "episodic_edges").await,
        "DUR-5: `episodic_edges` must exist after recovery"
    );
    assert_eq!(
        row_count(&tg.conn, "episodic_edges").await,
        stranded,
        "DUR-5: every edge stranded in `episodic_edges_new` must survive the recovery"
    );
}

/// Manufacture the OTHER crash window: `{table}_new` exists but is **EMPTY**, because
/// the run died between `CREATE TABLE {table}_new` and the `INSERT … SELECT` that fills
/// it. Also lays down the `{table}_bak_{tag}` snapshot the real migration would have
/// written first, since the resume must repair from that.
///
/// This is the case a resume that trusts EXISTENCE rather than COMPLETENESS gets wrong,
/// and it is more likely than the post-DROP window — the copy is the long statement.
async fn simulate_crash_before_the_copy(
    conn: &libsql::Connection,
    table: &str,
    bak_suffix: &str,
) -> usize {
    let before = row_count(conn, table).await;
    assert!(
        before > 0,
        "NON-VACUITY: `{table}` must hold rows before the simulated crash"
    );

    conn.execute("PRAGMA foreign_keys = OFF", ())
        .await
        .expect("fk off");

    // The snapshot the real migration takes before it touches anything.
    //
    // It usually ALREADY EXISTS — the migration ran for real during the first `open()`
    // and leaves its `_bak_` table behind as a rollback artifact. Replace it so the
    // snapshot reflects the rows we just seeded; without the drop this errors with
    // `table … already exists`, which is how the first draft failed.
    conn.execute(
        &format!("DROP TABLE IF EXISTS {table}_bak_{bak_suffix}"),
        (),
    )
    .await
    .expect("snapshot reset");
    conn.execute(
        &format!("CREATE TABLE {table}_bak_{bak_suffix} AS SELECT * FROM {table}"),
        (),
    )
    .await
    .expect("snapshot creation");

    // The scratch table — created, never filled.
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
            libsql::params![table],
        )
        .await
        .expect("ddl query");
    let ddl: String = rows
        .next()
        .await
        .expect("row iter")
        .expect("ddl row")
        .get(0)
        .expect("sql col");
    drop(rows);
    let scratch_ddl = if ddl.contains(&format!("TABLE \"{table}\"")) {
        ddl.replacen(
            &format!("TABLE \"{table}\""),
            &format!("TABLE \"{table}_new\""),
            1,
        )
    } else {
        ddl.replacen(&format!("TABLE {table}"), &format!("TABLE {table}_new"), 1)
    };
    assert!(
        scratch_ddl.contains(&format!("{table}_new")),
        "INSTRUMENT VALIDATION: DDL rewrite did not take:\n{ddl}"
    );
    conn.execute(&scratch_ddl, ())
        .await
        .expect("scratch create");

    conn.execute("PRAGMA foreign_keys = ON", ())
        .await
        .expect("fk on");

    // INSTRUMENT VALIDATION: the scratch must really be empty, or this test is
    // indistinguishable from the already-covered fully-populated case.
    assert_eq!(
        row_count(conn, &format!("{table}_new")).await,
        0,
        "the scratch must be EMPTY — that is the whole point of this fixture"
    );
    before
}

/// **Quinn REL-002 — ⚠️ THIS TEST PASSES VACUOUSLY. Read before trusting it.**
///
/// It asserts the right thing (an incomplete `entities_new` must never be renamed over
/// live data) but it **cannot currently reach the code that guarantees it**, and that
/// was MEASURED, not assumed: disabling `migrate_004`'s content-based repair
/// (`defs_a.rs`, `if scratch < snapshot`) leaves this test GREEN.
///
/// Why: `migrate_004`'s G1 gate (`defs_a.rs:146-161`) returns `Ok(())` as soon as
/// `entities` carries the composite PK. This fixture's database has been through every
/// migration, so G1 short-circuits and the resume branch is never entered — the empty
/// scratch is simply ignored, and the data survives for a reason that has nothing to do
/// with the repair.
///
/// Reaching it needs a genuinely **pre-004** database (no composite PK) carrying a
/// leftover `entities_new`, which this fixture cannot cheaply construct. The repair is
/// therefore **reasoned and unverified** — kept because the hazard is real for anyone
/// upgrading from a pre-004 version, and labelled because a green test nobody has seen
/// go red is not evidence.
///
/// Its sibling `episodic_edges_survives_a_crash_before_the_copy_completed` IS
/// non-vacuous — proven red by disabling `defs_h`'s repair — so REL-001, the regression
/// actually introduced by this change, is genuinely covered.
///
/// ## ✅ The gap this comment described is CLOSED — 2026-08-05
///
/// The repair is no longer "reasoned and unverified". It is driven directly, at unit
/// level, by `core::migrations::tests::an_empty_scratch_is_repopulated_from_the_snapshot_
/// before_the_swap` — which builds a genuinely **pre-004** `entities` (single-column PK,
/// no children tables to drag composite FKs along) so G1 cannot short-circuit, and is
/// **RED-PROVEN**: inverting `defs_a.rs`'s `if scratch < snapshot` fails it with
/// `left: 0, right: 3`. Two siblings pin the other directions — a COMPLETE scratch must
/// be swapped as-is (so the guard is not always-true), and a scratch with no snapshot
/// must refuse loudly with the live table intact.
///
/// The earlier note that a pre-004 fixture "cannot be cheaply constructed" was true of
/// THIS integration fixture and false of the unit path: `migrate_004_composite_pk_entities`
/// takes only a `Connection`. **This test is kept** — it still pins the ordinary
/// already-migrated case (a stale scratch on a modern database must stay inert), which is
/// the case real users hit. It simply is not, and never was, the REL-002 proof.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_incomplete_entities_scratch_is_not_renamed_over_live_data() {
    let db = unique_db_path("entities-partial");
    let seeded;
    {
        let mem = open_memory(&db).await;
        seed(&mem).await;
        let tg = mem.temporal_graph_for_test().expect("temporal_graph");
        seeded = simulate_crash_before_the_copy(&tg.conn, "entities", "004").await;
    }

    let mem = open_memory(&db).await;
    let tg = mem.temporal_graph_for_test().expect("temporal_graph");
    assert_eq!(
        row_count(&tg.conn, "entities").await,
        seeded,
        "REL-002: every entity must survive. A zero here means the resume trusted that \
         `entities_new` EXISTED rather than checking it was COMPLETE, dropped the live \
         table, and renamed an empty scratch over it — total silent data loss, reported \
         as a successful migration."
    );
}

/// **Quinn REL-001** — `migrate_006` must not rename an EMPTY `episodic_edges_new` over
/// live data.
///
/// This is the hazard the G1/G4 gate reordering CREATED: a stale scratch used to be
/// permanently unreachable behind G1's early return, so hoisting G4 without a
/// completeness check would have converted an inert leftover into a data-loss trigger.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn episodic_edges_survives_a_crash_before_the_copy_completed() {
    let db = unique_db_path("edges-partial");
    let seeded;
    {
        let mem = open_memory(&db).await;
        seed(&mem).await;
        let tg = mem.temporal_graph_for_test().expect("temporal_graph");
        seeded = simulate_crash_before_the_copy(&tg.conn, "episodic_edges", "006").await;
    }

    let mem = open_memory(&db).await;
    let tg = mem.temporal_graph_for_test().expect("temporal_graph");
    assert_eq!(
        row_count(&tg.conn, "episodic_edges").await,
        seeded,
        "REL-001: every presence edge must survive. A zero here is the regression the \
         G1/G4 reordering introduced — and `PRAGMA foreign_key_check` cannot catch it, \
         because fewer rows means fewer violations."
    );
}

/// The ORDINARY path must be unaffected. A resume gate that mistakes a normal run for a
/// crashed one would skip the rebuild and leave the schema unmigrated — a cure worse
/// than the disease, and invisible because everything still opens.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clean_reopen_is_still_a_clean_reopen() {
    let db = unique_db_path("clean");
    let seeded;
    {
        let mem = open_memory(&db).await;
        seed(&mem).await;
        let tg = mem.temporal_graph_for_test().expect("temporal_graph");
        seeded = row_count(&tg.conn, "entities").await;
        assert!(seeded > 0, "non-vacuity: the seed must create entities");
    }

    let mem = open_memory(&db).await;
    let tg = mem.temporal_graph_for_test().expect("temporal_graph");
    assert_eq!(
        row_count(&tg.conn, "entities").await,
        seeded,
        "a clean re-open must preserve every entity — the DUR-5 resume gate must not \
         fire on a database that never crashed"
    );
    assert!(
        !table_exists(&tg.conn, "entities_new").await,
        "a clean re-open must leave no scratch table behind"
    );
}
