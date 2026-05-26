//! Story #156 — RED phase acceptance tests
//!
//! Two tests:
//! 1. `test_no_legacy_rql_prefix_in_src` — grep crates/kremory/src/ for Rql/Rqlm/Rqlc
//!    type/ident occurrences (excluding allowlisted metric strings and file paths);
//!    count must be 0.
//! 2. `test_error_handling_policy_exists` — assert docs/error-handling-policy.md
//!    exists and contains all required sections.
//!
//! Both tests FAIL before the GREEN phase renames are applied.

use std::path::Path;
use std::process::Command;

/// Verify no legacy Rql/Rqlm/Rqlc type or identifier names remain in src/.
///
/// Allowlisted (not renamed per gate G10 / spec §6):
/// - `"rql.*"` metric name string literals (rql.ingest.total_ms, etc.)
/// - Comments documenting the historical rename itself
/// - Module comments referencing the old crate name in historical context
///
/// The grep pattern matches:
/// - `RqlError`, `RqlmError`, `RqlcConfig`, `RqlGraph`, `RqlmTelemetryConfig`
/// - Any CamelCase starting with `Rql`, `Rqlm`, `Rqlc`
/// - Excludes lines that are purely `rql.` metric strings (allowlisted)
/// - Excludes `.ai-docs/` path strings
#[test]
fn test_no_legacy_rql_prefix_in_src() {
    // Find the workspace root relative to this test file's manifest location
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_dir = Path::new(manifest_dir).join("src");

    // Use grep to find remaining occurrences.
    // Pattern: word starting with Rql, Rqlm, or Rqlc (type-level — uppercase R).
    // Exclude metric string literals (rql. with lowercase r inside quotes).
    let output = Command::new("grep")
        .args([
            "-rn",
            // Match lines with Rql/Rqlm/Rqlc as type-level identifiers
            r"Rql[A-Za-z]",
            src_dir.to_str().expect("src path is valid UTF-8"),
        ])
        .output()
        .expect("grep must be available on PATH");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Filter out allowlisted lines:
    // 1. Lines containing only rql.* metric name strings (these use lowercase rql.)
    // 2. Lines that are comments documenting the historical rename (containing
    //    "was `rql-core`" / "was `rql-memory`" / "former `rqlc`" patterns)
    let violations: Vec<&str> = stdout
        .lines()
        .filter(|line| {
            // Remove lines that are ONLY a metric string (the allowlisted pattern
            // `"rql.something"` appears in string literals — these use lowercase `rql.`
            // not uppercase `Rql`). Since we grep for `Rql[A-Za-z]` (uppercase R),
            // metric strings will never match anyway. This filter is belt-and-suspenders.
            !line.contains("\"rql.")
                // Remove lines that are historical rename comments
                && !line.contains("was `rql-core`")
                && !line.contains("was `rql-memory`")
                && !line.contains("former `rqlc`")
                && !line.contains(".ai-docs/adrs/rql/")
        })
        .collect();

    assert!(
        violations.is_empty(),
        "Legacy Rql/Rqlm/Rqlc identifiers found in crates/kremory/src/ ({} violations):\n{}",
        violations.len(),
        violations.join("\n")
    );
}

/// Verify the error-handling policy document exists and contains all required sections.
#[test]
fn test_error_handling_policy_exists() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    // docs/ is at the workspace root (two levels up from crates/kremory/)
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("crates/kremory has parent")
        .parent()
        .expect("crates has parent (workspace root)");

    let policy_path = workspace_root.join("docs").join("error-handling-policy.md");

    assert!(
        policy_path.exists(),
        "docs/error-handling-policy.md must exist (Story #156 Concern B). \
         Expected at: {}",
        policy_path.display()
    );

    let content =
        std::fs::read_to_string(&policy_path).expect("error-handling-policy.md must be readable");

    // Required sections (case-insensitive substring match):
    let required_sections = [
        ("User-input failures", "## User-input"),
        ("Contract violations", "## Contract violations"),
        ("Test code exemption", "## Test code"),
        ("Decision rubric", "## Decision rubric"),
        ("Examples", "## Examples"),
    ];

    let mut missing = Vec::new();
    for (label, marker) in &required_sections {
        // Allow any heading level (##, ###) and case variation
        let found = content.lines().any(|l| {
            l.to_lowercase()
                .contains(&marker.to_lowercase().replace("## ", ""))
        });
        if !found {
            missing.push(*label);
        }
    }

    assert!(
        missing.is_empty(),
        "docs/error-handling-policy.md is missing required sections: {:?}\n\
         All required: User-input failures, Contract violations, Test code exemption, \
         Decision rubric, Examples.",
        missing
    );

    // Verify the panic message prefix convention is documented
    assert!(
        content.contains("invariant:"),
        "docs/error-handling-policy.md must document the `invariant:` panic prefix convention"
    );

    // Verify at least 2 do/don't example pairs
    let example_count = content.matches("```rust").count();
    assert!(
        example_count >= 2,
        "docs/error-handling-policy.md must contain at least 2 code examples (do/don't pairs), \
         found: {} rust code blocks",
        example_count
    );
}
