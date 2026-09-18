#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Two OS PROCESSES over one database file. Does a write in one become visible
//! in the other?
//!
//! ## Why this exists
//!
//! `examples/two_handles_one_database.rs` claimed to cover "a web process and a
//! background worker" and covered no such thing: it opened two handles inside
//! ONE process. The claim shipped inside the published crate, and
//! `docs/deployment.md` §2 independently records the real state as
//! *"cross-process is UNTESTED"* — so the example asserted something the docs
//! said nobody had checked.
//!
//! The mechanisms are genuinely different. kremory's write lock is an
//! in-process `AsyncMutex` and it is **per handle** — `TemporalGraph::
//! open_with_dim` mints a fresh one (`core/schema.rs:460`) — so it excludes
//! tasks inside one handle and nothing else. Between processes there is no
//! shared runtime at all: arbitration is SQLite/libsql file locking, separate
//! WAL index mappings, separate page caches, and `PRAGMA busy_timeout = 5000`
//! (`core/schema.rs:456`).
//!
//! ## How it spans processes
//!
//! The test binary re-invokes ITSELF via `std::env::current_exe()`, selecting a
//! normally-`#[ignore]`d entrypoint with `--exact … --ignored`. The child is a
//! genuinely separate OS process — its own address space, its own libsql, its
//! own everything — parameterised entirely through the environment. There is no
//! helper binary because a `src/bin` target would ship inside the published
//! crate, and an extra `examples/` entry would join the published example list.
//!
//! Two false passes are guarded explicitly, because both are silent:
//!
//! - **Never actually left the parent.** Each child prints its pid and the
//!   parent asserts it differs from `std::process::id()`.
//! - **Child ran no test at all.** A `--exact` path that matches nothing exits
//!   **0** with `running 0 tests` (verified by hand). So the parent parses the
//!   child's `XPROC_*` lines and panics when they are absent, rather than
//!   reading a clean exit code as a clean result.
//!
//! DETERMINISTIC, zero-LLM, but NOT fast-tier cheap: **~9.8 s per child**, so
//! ~20 s per test. That is not this test's overhead — it is the cost of one
//! `Memory::open` (migrations + integrity check) in a debug build, measured
//! identically on an unrelated existing test in this same binary. Two children
//! per test is the floor for a claim about two processes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, StructuredFact};

/// Selects the child entrypoint below. Absent ⇒ we are the parent.
const ROLE_ENV: &str = "KREMORY_XPROC_ROLE";
/// Database file both processes open.
const DB_ENV: &str = "KREMORY_XPROC_DB";
/// Unique token the writer stores and the reader looks for. Unique per run so a
/// stale row from an earlier run can never make this pass.
const TOKEN_ENV: &str = "KREMORY_XPROC_TOKEN";
/// The reader child opens THIS path when set, so the RED verification can point
/// it at a different file without touching the assertion it is testing.
const READER_DB_OVERRIDE_ENV: &str = "KREMORY_XPROC_READER_DB";

/// Path of the ignored test the children run. Must match the `#[test]` fn's
/// full module path or `--exact` selects nothing and the child exits 0 having
/// run no tests — a false pass this file explicitly guards against.
const CHILD_TEST_PATH: &str = "cross_process_one_database::cross_process_child_entrypoint";

const NAMESPACE: &str = "xproc";

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

fn stub_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

async fn open_mem(db: &Path, ns: &Namespace) -> Memory {
    Memory::open(db.to_str().unwrap())
        .default_namespace(ns.clone())
        .with_llm(stub_llm())
        .with_embedder(null_embedder())
        .await
        .unwrap()
}

/// What a child did, as reported on its stdout. Parsed by the parent so a child
/// that silently ran nothing cannot be mistaken for a child that passed.
struct ChildReport {
    pid: u32,
    passages: usize,
}

fn parse_child_report(stdout: &str) -> Option<ChildReport> {
    let mut pid = None;
    let mut passages = None;
    for line in stdout.lines() {
        if let Some(v) = line.strip_prefix("XPROC_PID=") {
            pid = v.trim().parse::<u32>().ok();
        }
        if let Some(v) = line.strip_prefix("XPROC_PASSAGES=") {
            passages = v.trim().parse::<usize>().ok();
        }
    }
    Some(ChildReport {
        pid: pid?,
        passages: passages?,
    })
}

/// Everything a child needs, as one object rather than a widening argument
/// list (`clippy::too_many_arguments` is denied workspace-wide, and an `allow`
/// here would be the band-aid).
struct ChildSpec<'a> {
    role: &'a str,
    db: &'a Path,
    token: &'a str,
    /// Only meaningful for the reader. `Some` sends it at a DIFFERENT file,
    /// which is how the sensitivity guard below breaks the mechanism.
    reader_db: Option<&'a Path>,
}

/// Spawn a child process running `cross_process_child_entrypoint`.
fn spawn_child(spec: ChildSpec<'_>) -> std::process::Output {
    let exe: PathBuf = std::env::current_exe().expect("test binary must know its own path");
    let mut cmd = Command::new(&exe);
    cmd.args(["--exact", CHILD_TEST_PATH, "--ignored", "--nocapture"])
        .env(ROLE_ENV, spec.role)
        .env(DB_ENV, spec.db)
        .env(TOKEN_ENV, spec.token);
    if let Some(p) = spec.reader_db {
        cmd.env(READER_DB_OVERRIDE_ENV, p);
    }
    cmd.output().unwrap_or_else(|e| {
        panic!(
            "failed to spawn child {} from {}: {e}",
            spec.role,
            exe.display()
        )
    })
}

// ── The child ───────────────────────────────────────────────────────────────

/// Runs ONLY in a child process. `#[ignore]` keeps it out of the normal suite;
/// the parent selects it explicitly with `--exact … --ignored`.
///
/// It is also defensive about being run by hand (`--run-ignored all`): with no
/// role in the environment there is nothing to do, so it returns rather than
/// failing on a missing variable.
#[test]
#[ignore = "child process entrypoint — driven by cross_process_write_is_visible_to_a_second_process"]
fn cross_process_child_entrypoint() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        eprintln!("{ROLE_ENV} unset — not a child invocation, nothing to do");
        return;
    };
    let db = PathBuf::from(std::env::var(DB_ENV).expect("child needs a database path"));
    let token = std::env::var(TOKEN_ENV).expect("child needs a token");

    println!("XPROC_PID={}", std::process::id());

    let rt = tokio::runtime::Runtime::new().expect("child runtime");
    rt.block_on(async move {
        let ns = Namespace::new(NAMESPACE);
        match role.as_str() {
            "writer" => {
                let mem = open_mem(&db, &ns).await;
                mem.remember(format!("Cross-process sentinel {token}."))
                    .in_namespace(ns.clone())
                    .with_facts(vec![StructuredFact {
                        subject: "sentinel".into(),
                        predicate: "token".into(),
                        object: token.clone(),
                        valid_from: None,
                        valid_to: None,
                        memory_type: None,
                    }])
                    .skip_extraction()
                    .await
                    .expect("writer child must be able to write");
                mem.close().await.expect("writer close");
                println!("XPROC_PASSAGES=0");
            }
            "reader" => {
                // The override exists so the RED verification can point this
                // child at a DIFFERENT file without weakening the assertion.
                let target = std::env::var(READER_DB_OVERRIDE_ENV)
                    .map(PathBuf::from)
                    .unwrap_or(db);
                let mem = open_mem(&target, &ns).await;
                let found = mem
                    .recall(&token)
                    .in_namespace(ns.clone())
                    .content()
                    .await
                    .expect("reader child recall");
                println!("XPROC_PASSAGES={}", found.len());
                for p in &found {
                    println!("XPROC_SNIPPET={}", p.snippet.replace('\n', " "));
                }
                mem.close().await.expect("reader close");
            }
            other => panic!("unknown {ROLE_ENV}={other}"),
        }
    });
}

// ── The test ────────────────────────────────────────────────────────────────

/// A write committed by one OS process must be readable by a second one over
/// the same file.
///
/// This is the claim `examples/two_handles_one_database.rs` used to make and
/// could not back, and the one `docs/deployment.md` §2 records as untested.
#[test]
fn cross_process_write_is_visible_to_a_second_process() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("xproc.db");
    let token = format!("tok{}", uuid::Uuid::new_v4().simple());

    // ── Process 1: write ────────────────────────────────────────────────────
    let w = spawn_child(ChildSpec {
        role: "writer",
        db: &db,
        token: &token,
        reader_db: None,
    });
    assert!(
        w.status.success(),
        "writer child failed ({:?})\nstdout:\n{}\nstderr:\n{}",
        w.status,
        String::from_utf8_lossy(&w.stdout),
        String::from_utf8_lossy(&w.stderr)
    );
    let w_out = String::from_utf8_lossy(&w.stdout).into_owned();
    let w_report = parse_child_report(&w_out).unwrap_or_else(|| {
        panic!("writer child produced no report — did `--exact {CHILD_TEST_PATH}` select nothing?\nstdout:\n{w_out}")
    });

    // ── Process 2: read ─────────────────────────────────────────────────────
    let r = spawn_child(ChildSpec {
        role: "reader",
        db: &db,
        token: &token,
        reader_db: None,
    });
    assert!(
        r.status.success(),
        "reader child failed ({:?})\nstdout:\n{}\nstderr:\n{}",
        r.status,
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    );
    let r_out = String::from_utf8_lossy(&r.stdout).into_owned();
    let r_report = parse_child_report(&r_out).unwrap_or_else(|| {
        panic!("reader child produced no report — did `--exact {CHILD_TEST_PATH}` select nothing?\nstdout:\n{r_out}")
    });

    // ── This really was three processes ─────────────────────────────────────
    //
    // Checked before the interesting assertion, because a "cross-process" test
    // that never left the parent is exactly the defect being repaired here.
    let me = std::process::id();
    assert_ne!(
        w_report.pid, me,
        "the writer must be a separate OS process; got the parent's own pid {me}"
    );
    assert_ne!(
        r_report.pid, me,
        "the reader must be a separate OS process; got the parent's own pid {me}"
    );
    assert_ne!(
        w_report.pid, r_report.pid,
        "writer and reader must be different processes; both reported {}",
        w_report.pid
    );

    // ── The claim ───────────────────────────────────────────────────────────
    assert!(
        r_report.passages > 0,
        "a write committed by pid {} must be visible to pid {} over the same \
         database file. If this fails, kremory cannot be shared across \
         processes and docs/deployment.md §2 needs to say so outright rather \
         than 'untested'.\nreader stdout:\n{r_out}",
        w_report.pid,
        r_report.pid
    );
}

/// Sensitivity check for the test above.
///
/// The visibility assertion is only evidence if it can fail. Here the reader is
/// pointed at a DIFFERENT database file — same code path, same token, same
/// process spawning — and must find nothing. If this one ever starts finding
/// passages, the test above is passing for some reason other than
/// cross-process visibility and must not be trusted.
#[test]
fn cross_process_reader_on_a_different_file_finds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("xproc-written.db");
    let elsewhere = dir.path().join("xproc-elsewhere.db");
    let token = format!("tok{}", uuid::Uuid::new_v4().simple());

    let w = spawn_child(ChildSpec {
        role: "writer",
        db: &db,
        token: &token,
        reader_db: None,
    });
    assert!(
        w.status.success(),
        "writer child failed ({:?})\nstderr:\n{}",
        w.status,
        String::from_utf8_lossy(&w.stderr)
    );

    let r = spawn_child(ChildSpec {
        role: "reader",
        db: &db,
        token: &token,
        reader_db: Some(&elsewhere),
    });
    assert!(
        r.status.success(),
        "reader child failed ({:?})\nstderr:\n{}",
        r.status,
        String::from_utf8_lossy(&r.stderr)
    );
    let r_out = String::from_utf8_lossy(&r.stdout).into_owned();
    let r_report = parse_child_report(&r_out)
        .unwrap_or_else(|| panic!("reader child produced no report\nstdout:\n{r_out}"));

    assert_eq!(
        r_report.passages, 0,
        "a reader on a DIFFERENT file must find nothing — if it does, the \
         visibility test above proves nothing.\nreader stdout:\n{r_out}"
    );
}
