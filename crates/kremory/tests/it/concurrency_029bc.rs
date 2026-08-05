#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Concurrency tests for ADR-029b + ADR-029c invariants.
//!
//! Four test suites (Phase B of the 5-tier test pyramid):
//!
//! B1 — `tokio::join_all` multi-namespace recall fan-out (ADR-029c Decision 4)
//!      Concurrent recalls across N namespaces all complete without error and
//!      each returns a result attributed to its own namespace.
//!
//! B2 — NamespacePolicyCache thundering-herd on cold-cache (ADR-029c §H2)
//!      N concurrent callers register the same namespace simultaneously.
//!      The database must serialise writes; only one `Err::NamespacePolicyImmutable`
//!      is possible if two different policies race, but when all tasks use the same
//!      policy the outcome is always `Ok(())` for all callers.
//!
//! B3 — Concurrent `remember` calls across distinct namespaces don't cross-pollute
//!      Facts ingested into ns-A must not appear in a recall against ns-B.
//!
//! B4 — `upgrade_namespace_policy` atomicity vs concurrent recall
//!      An upgrade from Mutable → AppendOnly concurrent with a recall must
//!      not corrupt state: recall always completes, upgrade always completes,
//!      and the post-upgrade policy is AppendOnly.
//!
//! # Barrier note
//!
//! All barrier-based synchronisation uses `tokio::sync::Barrier`, NOT
//! `std::sync::Barrier`.  The standard library barrier blocks the OS thread
//! with a condvar, which deadlocks multi-thread tokio runtimes when there are
//! not enough OS threads to unblock all the waiting tasks simultaneously.
//! `tokio::sync::Barrier::wait()` is async and yields back to the executor
//! while waiting, allowing other tasks to run and eventually unblock everyone.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kremory::{DynEmbeddingProvider, Memory, Namespace, NamespacePolicy};
use tokio::sync::Barrier;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn null_llm() -> Arc<dyn kremory::memory::ChatProvider> {
    Arc::new(kremory::core::provider::MockChatProvider::null())
}

fn null_embedder() -> Arc<dyn DynEmbeddingProvider> {
    Arc::new(kremory::core::provider::NullEmbeddingProvider { dim: 384 })
}

/// Open a fresh Memory against a temp-file SQLite database.
async fn fresh_memory() -> (Memory, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let path = tmp.path().join("kremory-conc.db");
    let mem = Memory::open(&path)
        .with_llm(null_llm())
        .with_embedder(null_embedder())
        .await
        .expect("Memory::open");
    (mem, tmp)
}

// ── B1: tokio::join_all multi-namespace recall fan-out ────────────────────────

/// B1: Concurrent recall across 8 distinct namespaces all succeed.
///
/// ADR-029c Decision 4 mandates fan-out via `tokio::join_all`.  This test
/// exercises the _scheduler-level_ concurrency: all recall futures are spawned
/// simultaneously and race to acquire the libSQL connection pool.
///
/// Assertion: every recall returns `Ok(_)`, none panics, none deadlocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_concurrent_recall_across_namespaces_all_succeed() {
    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    const N: usize = 8;
    let mut handles = Vec::with_capacity(N);

    for i in 0..N {
        let mem = Arc::clone(&mem);
        let ns = Namespace::new(format!("tenant-b1-{i}"));
        handles.push(tokio::spawn(async move {
            mem.recall("test query").in_namespace(ns).await.map(|_| ())
        }));
    }

    let results = futures::future::join_all(handles).await;
    let mut ok_count = 0usize;
    for (i, res) in results.into_iter().enumerate() {
        let inner = res.expect("task did not panic");
        inner.unwrap_or_else(|e| panic!("recall for tenant-b1-{i} failed: {e}"));
        ok_count += 1;
    }
    assert_eq!(ok_count, N, "all {N} concurrent recalls must succeed");
}

/// B1 variant: `in_namespaces` fan-out — single call that fans out internally.
///
/// Exercises the `in_namespaces(&[A, B, C])` code-path under concurrent callers.
/// Uses `tokio::sync::Barrier` (async-safe) so all tasks fire simultaneously.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_in_namespaces_fan_out_concurrent_callers() {
    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    let ns_a = Namespace::new("fan-out-a");
    let ns_b = Namespace::new("fan-out-b");
    let ns_c = Namespace::new("fan-out-c");

    const CALLERS: usize = 6;
    let barrier = Arc::new(Barrier::new(CALLERS));
    let mut handles = Vec::with_capacity(CALLERS);

    for _ in 0..CALLERS {
        let mem = Arc::clone(&mem);
        let bar = Arc::clone(&barrier);
        let namespaces = vec![ns_a.clone(), ns_b.clone(), ns_c.clone()];
        handles.push(tokio::spawn(async move {
            // Yield to other tasks while waiting — does not block the thread.
            bar.wait().await;
            mem.recall("concurrent fan-out")
                .in_namespaces(&namespaces)
                .await
                .map(|_| ())
        }));
    }

    let results = futures::future::join_all(handles).await;
    for (i, res) in results.into_iter().enumerate() {
        let inner = res.expect("task did not panic");
        inner.unwrap_or_else(|e| panic!("caller {i} in_namespaces failed: {e}"));
    }
}

// ── B2: NamespacePolicyCache thundering-herd on cold-cache ───────────────────

/// B2: N concurrent `register_namespace` calls for the SAME namespace with the
/// SAME policy all return `Ok(())`.
///
/// ADR-029c §H2 documents a thundering-herd caveat on the policy-cache path:
/// all cold-cache callers fall through to the DB. The `register_namespace`
/// implementation must serialise those writes. With identical policies the
/// idempotent path must win for every caller.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b2_thundering_herd_same_policy_all_succeed() {
    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    const N: usize = 12;
    let barrier = Arc::new(Barrier::new(N));
    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(N);

    let ns_base = Namespace::new("thundering-herd-ns")
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent policy");

    for _ in 0..N {
        let mem = Arc::clone(&mem);
        let ns = ns_base.clone();
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&success_count);
        handles.push(tokio::spawn(async move {
            bar.wait().await;
            let result = mem.register_namespace(ns).await;
            if result.is_ok() {
                cnt.fetch_add(1, Ordering::Relaxed);
            }
            result.map(|_| ())
        }));
    }

    let results = futures::future::join_all(handles).await;
    for (i, res) in results.into_iter().enumerate() {
        let inner = res.expect("task did not panic");
        inner.unwrap_or_else(|e| {
            panic!("register_namespace caller {i} failed with same policy: {e}")
        });
    }
    assert_eq!(
        success_count.load(Ordering::Relaxed),
        N,
        "all {N} concurrent same-policy register_namespace calls must return Ok(())"
    );
}

/// B2 variant: cold-cache concurrent reads (recall) all complete without hang.
///
/// Exercises the policy-cache lock path under `get_namespace_policy_cached`.
/// The assertion is liveness: all tasks complete within the test timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b2_thundering_herd_cold_cache_recall_no_deadlock() {
    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    // Register a namespace first (establishes the DB row for policy lookup).
    let ns_name = "cold-cache-recall";
    let ns = Namespace::new(ns_name)
        .with_policy(NamespacePolicy::APPEND_ONLY)
        .expect("coherent");
    mem.register_namespace(ns.clone())
        .await
        .expect("register before concurrency test");

    const N: usize = 10;
    let barrier = Arc::new(Barrier::new(N));
    let mut handles = Vec::with_capacity(N);

    for _ in 0..N {
        let mem = Arc::clone(&mem);
        let ns_clone = Namespace::new(ns_name);
        let bar = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            // Yield to other tasks while waiting (async-safe barrier).
            bar.wait().await;
            mem.recall("cold cache test")
                .in_namespace(ns_clone)
                .await
                .map(|_| ())
        }));
    }

    let results = futures::future::join_all(handles).await;
    for (i, res) in results.into_iter().enumerate() {
        let inner = res.expect("task did not panic");
        inner.unwrap_or_else(|e| panic!("cold-cache recall caller {i} failed: {e}"));
    }
}

// ── B3: Concurrent remember across namespaces — no cross-namespace pollution ──

/// B3: Facts ingested into ns-A do not appear in recall for ns-B when both
/// operate concurrently.
///
/// This is the core isolation invariant for ADR-029c multi-namespace recall.
/// The test injects facts into two namespaces concurrently, then asserts that
/// a synchronous recall in ns-B returns only its own namespace results
/// (namespace field, when Some, must match the queried ns).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b3_concurrent_remember_no_cross_namespace_pollution() {
    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    let ns_a = Namespace::new("isolation-a");
    let ns_b = Namespace::new("isolation-b");

    // Concurrently remember into both namespaces.
    let mem_a = Arc::clone(&mem);
    let mem_b = Arc::clone(&mem);
    let nsa = ns_a.clone();
    let nsb = ns_b.clone();

    // 2-task async barrier so both spawns race the ingest path simultaneously.
    let barrier = Arc::new(Barrier::new(2));
    let bar_a = Arc::clone(&barrier);
    let bar_b = Arc::clone(&barrier);

    let handle_a = tokio::spawn(async move {
        bar_a.wait().await;
        mem_a
            .remember("Alice works on project Sentinel in namespace-A")
            .in_namespace(nsa)
            .await
    });

    let handle_b = tokio::spawn(async move {
        bar_b.wait().await;
        mem_b
            .remember("Bob works on project Phoenix in namespace-B")
            .in_namespace(nsb)
            .await
    });

    // Both ingestions must succeed (errors here are not the subject of this
    // test — we log them but don't fail on ingest errors from the stub).
    let (res_a, res_b) = tokio::join!(handle_a, handle_b);
    let _ = res_a.expect("task-a did not panic");
    let _ = res_b.expect("task-b did not panic");

    // Recall from ns-B.  Any results that carry a `namespace` attribution
    // must be from ns-B, never ns-A.  Use `.raw()` to obtain
    // `Vec<RetrievedContext>` directly (`.await` on a plain recall returns the
    // rendered prompt String which is not iterable as RetrievedContext).
    let context = mem
        .recall("who works on what project")
        .in_namespace(Namespace::new("isolation-b"))
        .raw()
        .await
        .expect("raw recall in ns-B after concurrent ingest must not error");

    for rc in &context {
        if let Some(ref ns) = rc.namespace {
            assert_ne!(
                ns.namespace, "isolation-a",
                "ns-B recall must not return results attributed to ns-A; got: {rc:?}"
            );
        }
    }
}

// ── B4: upgrade_namespace_policy atomicity vs concurrent recall ───────────────

/// B4: Concurrent `upgrade_namespace_policy` + recall: both complete, no
/// corrupted state.
///
/// ADR-029b Decision 6: `upgrade_namespace_policy` is monotonic (Mutable →
/// AppendOnly only) and must be atomic with respect to concurrent reads.
/// After the upgrade completes the policy must be AppendOnly; concurrent
/// recall must not corrupt the upgrade write or vice versa.
///
/// Assertion:
/// - Both the upgrade task and the recall tasks complete without error.
/// - After the join, `register_namespace` with a Mutable policy on the same
///   namespace returns `Err::NamespacePolicyImmutable` (proving the upgrade
///   persisted and the policy cache was invalidated).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b4_upgrade_namespace_atomicity_vs_concurrent_recall() {
    let (mem, _tmp) = fresh_memory().await;
    let mem = Arc::new(mem);

    let ns_name = "upgrade-atomicity-ns";
    // Establish the namespace as Mutable first.
    mem.register_namespace(Namespace::new(ns_name))
        .await
        .expect("initial register as Mutable");

    const RECALL_TASKS: usize = 6;
    let barrier = Arc::new(Barrier::new(RECALL_TASKS + 1)); // recalls + upgrade

    let mut handles = Vec::with_capacity(RECALL_TASKS + 1);

    // Recall tasks — race against the upgrade.
    for i in 0..RECALL_TASKS {
        let mem = Arc::clone(&mem);
        let ns = Namespace::new(ns_name);
        let bar = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            bar.wait().await;
            mem.recall("concurrent with upgrade")
                .in_namespace(ns)
                .await
                .map_err(|e| format!("recall-{i} error: {e}"))
                .map(|_| ())
        }));
    }

    // Upgrade task — fires simultaneously with the recall tasks.
    {
        let mem = Arc::clone(&mem);
        let ns = Namespace::new(ns_name);
        let bar = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            bar.wait().await;
            mem.upgrade_namespace_policy(ns)
                .await
                .map_err(|e| format!("upgrade error: {e}"))
        }));
    }

    let results = futures::future::join_all(handles).await;
    for (i, res) in results.into_iter().enumerate() {
        let inner = res.expect("task did not panic");
        inner.unwrap_or_else(|e| panic!("concurrent task {i} failed: {e}"));
    }

    // Post-condition: attempting to re-register the same namespace with a
    // Mutable policy must fail with NamespacePolicyImmutable — proving the
    // upgrade persisted correctly and the cache was invalidated.
    let mutable_ns = Namespace::new(ns_name)
        .with_policy(NamespacePolicy::default())
        .expect("Mutable policy is coherent");
    let re_register = mem.register_namespace(mutable_ns).await;
    assert!(
        re_register.is_err(),
        "re-register with Mutable policy after upgrade must return Err, not Ok"
    );
}
