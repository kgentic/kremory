//! G6 / Story #9 — Zero `unwrap()` / `expect()` enforcement — RED gate.
//!
//! ## What these tests verify
//!
//! 1. `test_cargo_toml_has_deny_lints` — the `[lints.clippy]` section exists in
//!    `crates/kremory/Cargo.toml` with `unwrap_used`, `expect_used`, and
//!    `warnings` all set to `"deny"`.  Fails until GREEN adds the section.
//!
//! 2. `test_clippy_unwrap_expect_clean` — `cargo clippy -p kremory --all-features
//!    -- -D clippy::unwrap_used -D clippy::expect_used` exits 0.  Fails until
//!    GREEN replaces all 11 sites with `panic!("invariant: …")`.
//!
//! ## Why no per-site Err-path tests
//!
//! All 11 unwrap/expect sites are INVARIANT (see `story-9-red.md` inventory).
//! None changes the function return type to `Result`.  The AC only requires
//! per-error-path tests for FALLIBLE sites; there are zero such sites.
//!
//! **Story lock**: #9 (zero-unwrap / -D warnings policy, gate G6 + G7).

use std::path::Path;
use std::process::Command;

/// Path to the manifest under test, resolved at runtime from CARGO_MANIFEST_DIR
/// (set by Cargo for integration tests).
fn kremory_cargo_toml() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is set to `crates/kremory` by Cargo during `cargo test`.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set — run via `cargo test`");
    Path::new(&manifest_dir).join("Cargo.toml")
}

// ---------------------------------------------------------------------------
// Gate 1: Cargo.toml lints section
// ---------------------------------------------------------------------------

/// Asserts the `[lints.clippy]` deny section is present in Cargo.toml.
///
/// Fails until GREEN adds:
/// ```toml
/// [lints.clippy]
/// unwrap_used  = "deny"
/// expect_used  = "deny"
/// warnings     = "deny"  # or [lints] warn = "deny" equivalent
/// ```
#[test]
fn test_cargo_toml_has_deny_lints() {
    let cargo_toml_path = kremory_cargo_toml();
    let contents = std::fs::read_to_string(&cargo_toml_path)
        .unwrap_or_else(|e| panic!("failed to read {:?}: {}", cargo_toml_path, e));

    assert!(
        contents.contains("[lints.clippy]"),
        "Cargo.toml must contain a `[lints.clippy]` section (Story #9 AC2).\n\
         File: {:?}",
        cargo_toml_path,
    );

    assert!(
        contents.contains("unwrap_used") && contents.contains("deny"),
        "Cargo.toml `[lints.clippy]` must set `unwrap_used = \"deny\"` (Story #9 AC2).\n\
         File: {:?}",
        cargo_toml_path,
    );

    assert!(
        contents.contains("expect_used"),
        "Cargo.toml `[lints.clippy]` must set `expect_used = \"deny\"` (Story #9 AC2).\n\
         File: {:?}",
        cargo_toml_path,
    );
}

// ---------------------------------------------------------------------------
// Gate 2: clippy clean with deny flags
// ---------------------------------------------------------------------------

/// Asserts that `cargo clippy -p kremory --all-features -- -D clippy::unwrap_used
/// -D clippy::expect_used` exits 0.
///
/// This test spawns a child `cargo clippy` process.  It will:
///   - FAIL (exit non-zero) in RED phase — 11 unwrap/expect sites exist.
///   - PASS (exit 0)        in GREEN phase — all sites replaced with
///     `panic!("invariant: …")` and/or `?` propagation.
///
/// The test captures stderr so the output appears in `cargo test` when it fails.
#[test]
fn test_clippy_unwrap_expect_clean() {
    // Resolve the workspace root from CARGO_MANIFEST_DIR (crates/kremory → ../../)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set — run via `cargo test`");
    let workspace_root = Path::new(&manifest_dir)
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .expect("could not resolve workspace root from CARGO_MANIFEST_DIR");

    let output = Command::new("cargo")
        .args([
            "clippy",
            "-p",
            "kremory",
            "--all-features",
            "--",
            "-D",
            "clippy::unwrap_used",
            "-D",
            "clippy::expect_used",
        ])
        .current_dir(workspace_root)
        .output()
        .expect("failed to spawn `cargo clippy` — is cargo in PATH?");

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "cargo clippy reported unwrap/expect violations (Story #9 AC1+AC2).\n\
         GREEN phase must replace all sites with `panic!(\"invariant: …\")` or `?`.\n\
         \n--- clippy stderr ---\n{}\n--- end ---",
        stderr,
    );
}
