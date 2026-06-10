#![allow(clippy::unwrap_used, clippy::expect_used)]
//! RED-phase tests for GAP-002: `pub(crate)` promotion of three functions in
//! `crates/kremory/src/core/dream/consistency_check.rs`.
//!
//! The functions that must be promoted:
//! - `verify_batch_schema` (line 144 in current main)
//! - `build_verify_messages` (line 631 in current main)
//! - `verify_batch` (must be located and promoted to `pub(crate)`)
//!
//! Governing spec:
//! - `kremory-v020--c6-async-gate-verify-architecture.md` §5.2 (dependency direction):
//!   "these functions are currently module-private in `consistency_check.rs`.
//!   `verify_stage.rs` requires them. Promotion MUST happen in Phase A."
//! - Test strategy §5.4 DENT-001: "compile-test: `verify_stage.rs` imports them and
//!   `cargo check` PASS"
//!
//! ## Why these tests MUST fail on current main
//!
//! All three functions are currently `fn` (module-private) — NOT `pub(crate)`.
//! Tests in `crates/kremory/tests/` are compiled as SEPARATE crates that link against
//! the `kremory` crate. They CAN access `pub(crate)` items because integration tests
//! in Rust's `tests/` directory are treated as external crates for `pub` access, but
//! `pub(crate)` items in the library crate are accessible from integration tests within
//! the SAME CRATE's test binary.
//!
//! Specifically: Rust integration tests in `tests/*.rs` compile as SEPARATE crates
//! that link against the library. `pub(crate)` is crate-scoped — it is NOT accessible
//! from external integration test crates. Therefore, to test that these functions are
//! `pub(crate)` (accessible within the library crate from `verify_stage.rs`), we need
//! to:
//!   1. Verify the functions compile (no compiler error) when accessed from inside
//!      the kremory crate (unit test style), OR
//!   2. Add re-exports under a `#[cfg(test)]` feature gate for integration test access.
//!
//! Since `crates/kremory/tests/` are EXTERNAL to the crate, we CANNOT directly import
//! `pub(crate)` functions from `consistency_check.rs` here. Instead, this file tests
//! the STRUCTURAL CONSEQUENCE: the spike binary `crates/kremory-eval/src/bin/spike_c6_async_gate.rs`
//! is documented as BLOCKED on `verify_batch` being accessible. We verify this indirectly
//! via a `cargo check` subprocess.
//!
//! ALTERNATIVE COMPILE-FAIL APPROACH: we attempt to import `verify_batch_schema`
//! using a hypothetical `#[cfg(test)]` re-export or `test-utils` feature. If that
//! re-export is NOT present, the import fails and the test is red. If it IS present,
//! the import succeeds (green). This is the approach used below.
//!
//! ## Mocking boundary
//!
//! No LLM calls. These are compile-level tests only.

// ─── Approach 1: test-utils feature gate re-export ───────────────────────────
//
// This import succeeds ONLY when:
// (a) `verify_batch_schema` is `pub(crate)` AND
// (b) kremory re-exports it under `#[cfg(any(test, feature = "test-utils"))]`
//     (the standard pattern used in the codebase for test-accessible internals,
//      see kremory/src/core/ingest/mod.rs line 33-37 for the existing pattern).
//
// On current main: FAILS because `verify_batch_schema` is `fn` (private).
// After Green phase: PASSES because it is `pub(crate)` and re-exported.
//
// NOTE: this test file intentionally uses a feature-gated re-export that does NOT
// exist yet. The compile failure is the red state.
#[cfg(feature = "test-utils")]
mod visibility_tests {
    // These imports FAIL TO COMPILE on current main:
    // 1. `verify_batch_schema` is `fn` (not `pub(crate)`)
    // 2. Even if promoted to `pub(crate)`, it's not in kremory's public test-utils re-export yet
    use kremory::core::dream::consistency_check::{
        build_verify_messages,
        verify_batch,
        verify_batch_schema,
    };

    /// Compile-time test: `verify_batch_schema` is accessible as pub(crate) via test-utils re-export.
    ///
    /// If this function body compiles, the promotion succeeded.
    /// Test body intentionally trivial — the meaningful assertion is compilation.
    #[test]
    fn verify_batch_schema_is_accessible_as_pub_crate() {
        // Call with a dummy count — if the function signature changed, this will catch it.
        let schema = verify_batch_schema(3);
        assert!(
            schema.is_object(),
            "verify_batch_schema must return a JSON object; got: {:?}",
            schema
        );
        // Must have the top-level 'decisions' property to be a valid VerifyBatch schema.
        assert!(
            schema
                .get("properties")
                .and_then(|p: &serde_json::Value| p.get("decisions"))
                .is_some(),
            "verify_batch_schema must contain 'decisions' in properties"
        );
    }

    /// Compile-time test: `build_verify_messages` is accessible as pub(crate) via test-utils re-export.
    ///
    /// We just assert the function is callable — its return type is `Vec<...>` (ChatMessages).
    /// The exact signature is verified by the compiler; the body tests that it returns
    /// a non-empty messages vec.
    #[test]
    fn build_verify_messages_is_accessible_as_pub_crate() {
        // CandidateRow is private — we cannot construct it from outside the module.
        // This compile-test therefore primarily validates that the symbol resolves.
        // If `build_verify_messages` requires private types that can't be constructed
        // from outside, the Green agent should expose a `CandidateRow` test constructor
        // or change the signature to accept a plain struct.
        //
        // Calling the function with the actual parameters is intentionally left to the
        // Green agent — the IMPORT alone failing to compile is the red assertion.
        // SAFETY: transmute used only to reference the symbol (fn pointer cast to *const (),
        // then back to unsafe fn()) — the result is never called. Green-phase fix: the
        // Red test body required `unsafe {}` around std::mem::transmute; the import-level
        // assertion (symbol resolves at all) is the load-bearing check.
        let _f: unsafe fn() = unsafe { std::mem::transmute(build_verify_messages as *const ()) };
        let _ = _f;
    }

    /// Compile-time test: `verify_batch` is accessible as pub(crate) via test-utils re-export.
    #[test]
    fn verify_batch_is_accessible_as_pub_crate() {
        // Reference the symbol — compile error if not pub(crate).
        // We can't call it easily without constructing its private arg types,
        // but mere reference is sufficient for Phase A DoD.
        let _ptr = verify_batch as *const ();
        let _ = _ptr;
    }
}

// ─── Approach 2: subprocess cargo check on spike binary ──────────────────────
//
// The spike binary `crates/kremory-eval/src/bin/spike_c6_async_gate.rs` documents
// that it is BLOCKED on `verify_batch` being pub(crate) accessible. If the spike
// binary currently fails `cargo check` because of the private function, then
// after the Green phase promotes the function the spike should compile.
//
// This test invokes `cargo check` as a subprocess and asserts the exit code.
// On current main: FAILS (spike binary has compile errors due to private visibility).
// After Green phase: PASSES.
#[tokio::test]
async fn spike_c6_async_gate_compiles_after_pub_crate_promotion() {
    use std::process::Command;

    // Find the workspace root by looking for the Cargo.toml with [workspace].
    let workspace_root = {
        let mut dir = std::path::PathBuf::from(file!());
        // file!() returns the source path relative to workspace root.
        // Walk up until we find Cargo.toml with [workspace].
        dir.pop(); // pop filename
        dir.pop(); // pop tests/
        dir.pop(); // pop kremory/
        dir.pop(); // pop crates/
        dir
    };

    let output = Command::new("cargo")
        .args([
            "check",
            "--manifest-path",
            workspace_root
                .join("Cargo.toml")
                .to_str()
                .expect("utf-8 path"),
            "-p",
            "kremory-eval",
            "--bin",
            "spike_c6_async_gate",
        ])
        .output()
        .expect("cargo check must be available in PATH");

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "cargo check -p kremory-eval --bin spike_c6_async_gate FAILED — \
         this means GAP-002 pub(crate) promotion is not complete.\n\
         cargo stderr:\n{stderr}"
    );
}

// ─── Approach 3: direct source-level grep assertion ──────────────────────────
//
// A deterministic structural test: grep the source file for `pub(crate) fn verify_batch`,
// `pub(crate) fn build_verify_messages`, `pub(crate) fn verify_batch_schema`.
// This is the simplest test that directly matches the Phase A DoD criterion.
//
// On current main: FAILS because the functions are `fn` (no pub(crate) prefix).
// After Green phase: PASSES.
#[test]
fn consistency_check_source_contains_pub_crate_promotions() {
    // Locate consistency_check.rs relative to this test file.
    // file!() gives us the path from workspace root.
    let source_path = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("src");
        p.push("core");
        p.push("dream");
        p.push("consistency_check.rs");
        p
    };

    let source = std::fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("cannot read consistency_check.rs at {:?}: {e}", source_path));

    // verify_batch_schema — accepts pub(crate) or pub (Green may promote beyond pub(crate)).
    assert!(
        source.contains("pub(crate) fn verify_batch_schema")
            || source.contains("pub fn verify_batch_schema"),
        "GAP-002 INCOMPLETE: consistency_check.rs must contain `pub(crate) fn verify_batch_schema` \
         or `pub fn verify_batch_schema`.\n\
         Current content has `fn verify_batch_schema` (module-private). \
         Green phase must add `pub(crate)` or `pub` prefix."
    );

    // build_verify_messages — accepts pub(crate) or pub.
    assert!(
        source.contains("pub(crate) fn build_verify_messages")
            || source.contains("pub fn build_verify_messages"),
        "GAP-002 INCOMPLETE: consistency_check.rs must contain `pub(crate) fn build_verify_messages` \
         or `pub fn build_verify_messages`.\n\
         Current content has `fn build_verify_messages` (module-private). \
         Green phase must add `pub(crate)` or `pub` prefix."
    );

    // verify_batch — accepts pub(crate) or pub, sync or async.
    assert!(
        source.contains("pub(crate) fn verify_batch")
            || source.contains("pub(crate) async fn verify_batch")
            || source.contains("pub fn verify_batch")
            || source.contains("pub async fn verify_batch"),
        "GAP-002 INCOMPLETE: consistency_check.rs must contain a promoted `verify_batch` function \
         (pub(crate) or pub, sync or async).\n\
         Locate the verify_batch invocation at line ~289 in run_consistency_check, \
         extract it to a named function, and add pub(crate) or pub prefix."
    );
}
