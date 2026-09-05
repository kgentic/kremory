// `expect_used`/`unwrap_used` are denied package-wide by `[lints.clippy]`, which
// applies to test targets too. Tests are the surface where an unmet setup
// precondition SHOULD panic loudly with its message, so this scopes the
// production-only lint the same way the sibling guard
// `no_workspace_terminology_in_lib.rs` already does. Not a suppression of a real
// finding: every site below is test scaffolding whose failure means the harness
// itself is broken, which must abort rather than be handled.
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Guard — the PUBLIC-SHIPPING surface must carry no operator-local absolute
//! path and no private product codename.
//!
//! # Why this exists
//!
//! kremory 0.7.0 shipped both defects to crates.io. `crates/*/src/**` and
//! `crates/*/Cargo.toml` are inside the crate's `include` allowlist, so a code
//! comment and a test-fixture string carrying a private product codename went
//! out verbatim. A separate pre-public-repo audit then found 66 lines
//! containing the operator's home directory across 39 files, three of which
//! named unrelated private projects. Neither class was caught by review; both
//! are mechanically detectable in milliseconds.
//!
//! # Design notes, each one earned rather than assumed
//!
//! * **Walks the filesystem instead of shelling out to `grep`.** Measured
//!   2026-09-05: `git grep -- 'crates/*/src'` does NOT match
//!   `crates/kremory/src/core/graph/tests.rs`, so an audit run through that
//!   pathspec reported the surface clean while a real hit sat inside the scan
//!   set. A guard is only as good as its instrument.
//! * **Classifies binary explicitly and FAILS on it rather than skipping.**
//!   `grep` skips any NUL-containing file silently and still exits 0, which
//!   converts "did not look" into "no matches" — indistinguishable from a pass.
//! * **Asserts a non-vacuity floor.** If the walk ever breaks (a directory
//!   renamed, a root moved) the scan set empties and every assertion below
//!   passes trivially. A vacuous pass looks exactly like an earned one, so the
//!   floor is what makes this test's green mean something.
//! * **Reads private terms from a file rather than hardcoding them.** The list
//!   lives in `.ai-docs/`, which is stripped from the public export, so naming
//!   private products in order to ban them does not itself publish them.

use std::path::{Path, PathBuf};

/// Repo root, derived from the compile-time manifest dir rather than the
/// process CWD (which differs between `cargo test` and `cargo nextest`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("manifest dir should be <root>/crates/kremory")
        .to_path_buf()
}

/// Every TRACKED file that reaches a public consumer.
///
/// The set is derived from `git ls-files`, not from a filesystem walk. That is
/// the difference between "what ships" and "what happens to be on this disk":
/// an earlier filesystem-walking version of this guard descended into
/// `bench/locomo/.venv/`, `__pycache__/` and a gitignored results dump and
/// produced 15,477 findings across 18,914 files — none of which ship. A guard
/// that fires on ordinary work gets disabled, so the false-positive rate is a
/// correctness property here, not a tuning preference.
///
/// `git ls-files` is invoked with NO pathspec. All filtering happens below in
/// Rust with explicit prefix tests, because git's pathspec globbing is the
/// exact trap this guard was written after: `git grep -- 'crates/*/src'` was
/// measured (2026-09-05) NOT to match `crates/kremory/src/core/graph/tests.rs`.
fn shipping_files(root: &Path) -> Vec<PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .expect("git ls-files should run");
    assert!(out.status.success(), "git ls-files failed: {:?}", out.status);

    let listing = String::from_utf8(out.stdout).expect("tracked paths are UTF-8");
    listing
        .split('\0')
        .filter(|p| !p.is_empty())
        .filter(|p| is_public_surface(p))
        .map(|p| root.join(p))
        .filter(|p| p.is_file())
        .collect()
}

/// True when a repo-relative tracked path is published — via the crates.io
/// package (`include`: `src/**`, the manifest) or via the repository itself.
/// The two sets are not the same, and the union is what the public can read.
fn is_public_surface(rel: &str) -> bool {
    // Root-level markdown (README, CHANGELOG, CONTRIBUTING, SECURITY, ...).
    if !rel.contains('/') && rel.ends_with(".md") {
        return true;
    }
    if let Some(rest) = rel.strip_prefix("crates/") {
        // `crates/<name>/Cargo.toml` and `crates/<name>/src/**/*.rs`.
        if let Some((_, tail)) = rest.split_once('/') {
            if tail == "Cargo.toml" {
                return true;
            }
            if tail.starts_with("src/") && tail.ends_with(".rs") {
                return true;
            }
        }
        return false;
    }
    PUBLIC_REPO_DIRS
        .iter()
        .any(|d| rel.starts_with(&format!("{d}/")))
}

/// Directories published via the repository but not via the crate package.
/// `e2e-consumer/` is why this list exists: two example commands there carried
/// an absolute home path and were invisible to a guard scoped to `crates/`.
const PUBLIC_REPO_DIRS: [&str; 6] = [
    "docs",
    "bench",
    "scripts",
    "e2e-consumer",
    "e2e-consumer-napi",
    "monitoring",
];

/// Read as UTF-8, or report why not. Never silently skips: a binary file inside
/// the hand-authored shipping surface is itself a finding.
fn read_text(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("unreadable: {e}"))?;
    if bytes.contains(&0) {
        return Err("contains NUL bytes — binary content in the shipping surface".into());
    }
    String::from_utf8(bytes).map_err(|_| "not valid UTF-8".to_string())
}

/// Absolute home-directory prefixes. Any of these in a shipped file exposes the
/// author's machine layout, and in practice the names of their other projects.
const LOCAL_PATH_MARKERS: [&str; 2] = ["/Users/", "/home/"];

fn forbidden_terms(root: &Path) -> Option<Vec<String>> {
    let list = root.join(".ai-docs/public-surface-forbidden-terms.txt");
    // A present `.ai-docs/` means this is the private repo, where the list MUST
    // exist. Absent (the public export strips it) the term half legitimately
    // does not apply — but the path half below still runs unconditionally.
    if !root.join(".ai-docs").is_dir() {
        return None;
    }
    let body = std::fs::read_to_string(&list).unwrap_or_else(|e| {
        panic!("{} is required in the private repo but is unreadable: {e}", list.display())
    });
    let terms: Vec<String> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_lowercase())
        .collect();
    assert!(!terms.is_empty(), "{} has no terms — guard would be vacuous", list.display());
    Some(terms)
}

#[test]
fn shipping_surface_carries_no_local_paths_or_private_codenames() {
    let root = repo_root();
    let files = shipping_files(&root);

    // Non-vacuity floor. Its job is to catch a BROKEN walk (which yields near
    // zero), not to track the true count — set well below it so that deleting a
    // few source files cannot false-positive, since a guard that fires on
    // ordinary work gets disabled and then protects nothing. Verified
    // 2026-09-05: the walk finds exactly 204, matching an independent count of
    // 5 root markdown + 5 manifests + 181 crate sources + 13 published docs.
    assert!(
        files.len() >= 100,
        "scanned only {} files — the walk is broken and this test would pass vacuously",
        files.len()
    );

    let terms = forbidden_terms(&root);
    let mut findings: Vec<String> = Vec::new();

    for path in &files {
        let rel = path.strip_prefix(&root).unwrap_or(path).display().to_string();
        let text = match read_text(path) {
            Ok(t) => t,
            Err(why) => {
                findings.push(format!("{rel}: {why}"));
                continue;
            }
        };
        let lower = text.to_lowercase();

        for (n, line) in text.lines().enumerate() {
            for marker in LOCAL_PATH_MARKERS {
                if line.contains(marker) {
                    findings.push(format!(
                        "{rel}:{}: operator-local absolute path ({marker})",
                        n + 1
                    ));
                }
            }
        }

        if let Some(terms) = &terms {
            for term in terms {
                if lower.contains(term.as_str()) {
                    findings.push(format!("{rel}: private term '{term}'"));
                }
            }
        }
    }

    assert!(
        findings.is_empty(),
        "public-shipping surface is not clean ({} scanned, {} findings):\n  {}",
        files.len(),
        findings.len(),
        findings.join("\n  ")
    );
}

/// Sensitivity, both directions. A guard proven only on the cases it should
/// block is untested: this asserts it FIRES on a synthetic leak and stays
/// SILENT on text that merely resembles one.
#[test]
fn detector_fires_on_a_leak_and_not_on_lookalikes() {
    let leaks = [
        "let p = \"/Users/someone/.cache/model.gguf\";",
        "# see /home/ci/build/out.log",
    ];
    for l in leaks {
        assert!(
            LOCAL_PATH_MARKERS.iter().any(|m| l.contains(m)),
            "detector missed a real leak: {l}"
        );
    }

    // False positives are a correctness property, not an acceptable cost: a
    // guard that fires on ordinary work gets disabled, and then protects nothing.
    let clean = [
        "let p = dirs::home_dir().join(\".cache/model.gguf\");",
        "documented at ~/.config/kremory/config.toml",
        "the users table stores home addresses",
        "GET /users/:id returns the profile",
    ];
    for c in clean {
        assert!(
            !LOCAL_PATH_MARKERS.iter().any(|m| c.contains(m)),
            "detector false-positived on ordinary text: {c}"
        );
    }
}
