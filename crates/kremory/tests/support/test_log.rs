//! Per-test tracing harness — `init_test_log` (v0.2.4 Component 3, spec §6).
//!
//! Governing spec: `.ai-docs/specs/v0-2-4-test-infra-o11y-harness-arch-spec-2026-06-15.md` §6.
//!
//! Installs a per-test `tracing` subscriber that writes to
//! `target/test-logs/<sanitized-test-name>-<unix-ts>.log` via a non-blocking
//! file appender. The subscriber is installed with
//! `tracing::subscriber::set_default` (a THREAD-LOCAL `DefaultGuard`), NOT
//! `set_global_default`/`fmt::init` (once-per-process) so dozens of test fns in
//! one binary can each install + swap their own subscriber without panicking
//! (Decision D4, §6.2).
//!
//! # Scope is the caller thread ONLY (spec §6.3 — load-bearing)
//!
//! `set_default` installs the subscriber on the CURRENT thread. Under
//! `#[tokio::test(flavor = "multi_thread")]` futures may execute on spawned
//! worker threads where the thread-local subscriber is absent — events emitted
//! there escape the log file. Traced tests therefore MUST use the current-thread
//! flavor (plain `#[tokio::test]`).
//!
//! Even current-thread flavor is NECESSARY-BUT-INSUFFICIENT for kremory's
//! Phase-2 capture (ASMP-001): the `BackgroundIngestor` runs enrichment on a
//! DEDICATED `std::thread` driving its OWN `new_multi_thread().worker_threads(1)`
//! runtime (`deferred_pipeline.rs:321`). A thread-local subscriber can never
//! reach that worker thread, and `parent_span.enter()` (`ingestor.rs:103`)
//! propagates span CONTEXT, not the subscriber. So this appender captures only
//! the synchronous caller-thread work (Phase-1 extraction on the await point +
//! the recall path). Assert Phase-2 correctness via the `EnrichmentEventSink`
//! (the PRIMARY Phase-2 oracle, §6.4) — NOT trace output.

use tracing::subscriber::DefaultGuard;
use tracing_appender::non_blocking::WorkerGuard;

/// Holds BOTH guards for the test's lifetime.
///
/// Dropping it restores the prior subscriber (`DefaultGuard`) AND flushes +
/// shuts down the non-blocking appender's background worker (`WorkerGuard`).
/// Both MUST be held for the whole test — dropping the `WorkerGuard` early
/// drops the flush thread and loses buffered events (spec §6.1).
#[must_use = "TestLogGuard must be bound for the whole test; an unbound guard \
              drops immediately and captures nothing"]
pub struct TestLogGuard {
    _default: DefaultGuard,
    _worker: WorkerGuard,
}

/// Install a per-test tracing subscriber writing to
/// `target/test-logs/<sanitized-test-name>-<unix-ts>.log`.
///
/// `test_name` is sanitized (`::` and `/` → `_`) so module-path-style names map
/// to a single-segment filename. The `target/test-logs` directory is created if
/// absent. Returns a [`TestLogGuard`] that MUST be bound for the test body — see
/// the `#[must_use]` note.
///
/// # Panics
/// Panics (loudly, per test convention) if the log directory or file cannot be
/// created — a test that cannot write its trace log has lost its debugging
/// surface and should fail visibly rather than run blind.
pub fn init_test_log(test_name: &str) -> TestLogGuard {
    let sanitized = test_name.replace("::", "_").replace('/', "_");

    // Resolve the log dir relative to the workspace target dir. CARGO_TARGET_DIR
    // wins if set; otherwise fall back to the crate-local `target/`.
    let log_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"))
        .join("test-logs");

    std::fs::create_dir_all(&log_dir)
        .unwrap_or_else(|e| panic!("init_test_log: create_dir_all {}: {e}", log_dir.display()));

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let log_path = log_dir.join(format!("{sanitized}-{ts}.log"));

    let file = std::fs::File::create(&log_path)
        .unwrap_or_else(|e| panic!("init_test_log: create {}: {e}", log_path.display()));

    let (non_blocking, worker) = tracing_appender::non_blocking(file);

    let subscriber = tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(true)
        .with_level(true)
        .finish();

    let default = tracing::subscriber::set_default(subscriber);

    TestLogGuard {
        _default: default,
        _worker: worker,
    }
}
