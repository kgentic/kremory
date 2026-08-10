//! Asserts the premise that four `cargo audit` advisories are ignored on.
//!
//! `.cargo/audit.toml` accepts RUSTSEC-2026-0104 / -0098 / -0099 / -0049
//! (`rustls-webpki` 0.102.8, pinned transitively by `libsql = "=0.9.30"`) on
//! the grounds that **kremory never opens a remote libsql connection**, so the
//! TLS certificate-chain code those advisories live in is unreachable.
//!
//! That premise must be DERIVED, not DECLARED. A comment saying "we don't use
//! remote libsql" is attested by whoever wrote it and rots the moment someone
//! adds replica support — at which point four accepted advisories silently
//! become live and nothing says so (CLAUDE.md Rule 37 derive-over-declare,
//! Rule 20 audit-what-guards-mask).
//!
//! So this test reads the source and fails if the premise stops holding.
//!
//! # Why a Rust test rather than a grep in CI
//!
//! `grep`/`rg`/`git grep` classify any file containing a NUL byte as binary
//! and skip it **silently, with a success exit code** — a sweep can report
//! "no matches" when the truth is "did not look" (CLAUDE.md Rule 38, and this
//! repo has hit exactly that). `fs::read` has no such behaviour: it returns
//! the bytes or it returns an error. The instrument cannot report a false
//! negative here, which is the whole point of the check.
//!
//! # Scope
//!
//! Deliberately narrow: the crate whose reachability claim is being made
//! (`crates/kremory/src`). `kremory-eval` is a dev/bench crate that is never
//! published and never reaches a consumer, so it is out of scope; if it ever
//! grows a remote path, that is a bench concern, not a published-surface one.

use std::path::{Path, PathBuf};

/// libsql APIs that open a REMOTE (TLS) connection. Using any of these makes
/// `rustls-webpki`'s certificate-validation path reachable.
///
/// This is a formal API surface — a fixed, enumerable set of function and
/// builder names owned by a dependency — NOT an attempt to pattern-match
/// intent. Matching on a closed formal set is the legitimate use of a
/// literal scan (CLAUDE.md Rule 29's discriminator).
const REMOTE_LIBSQL_APIS: &[&str] = &[
    "new_remote",
    "new_remote_replica",
    "new_synced_database",
    "new_local_replica",
    "sync_url",
    "remote_writes",
];

fn kremory_src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `dir`, recursively.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read an entry of {}: {e}", dir.display()))
            .path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn kremory_never_opens_a_remote_libsql_connection() {
    let root = kremory_src_root();
    let mut sources = Vec::new();
    rust_sources(&root, &mut sources);

    // Non-vacuity guard: a walk that found nothing would pass this test
    // unconditionally and tell us nothing. `kremory/src` is a large crate;
    // anything near zero means the walk broke, not that the crate is clean.
    // (A green test you have never seen fail is an unvalidated instrument.)
    assert!(
        sources.len() > 50,
        "source walk found only {} .rs files under {} — the WALK is broken, \
         so this test proves nothing. Fix the walk before trusting a pass.",
        sources.len(),
        root.display()
    );

    let mut hits: Vec<String> = Vec::new();
    for path in &sources {
        // Bytes, not `read_to_string`: a non-UTF-8 file must surface as a
        // hard error rather than being skipped.
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let text = String::from_utf8_lossy(&bytes);
        for (lineno, line) in text.lines().enumerate() {
            for api in REMOTE_LIBSQL_APIS {
                if line.contains(api) {
                    hits.push(format!(
                        "  {}:{} -> `{api}`\n      {}",
                        path.strip_prefix(&root).unwrap_or(path).display(),
                        lineno + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        hits.is_empty(),
        "kremory now reaches a REMOTE libsql API, which makes the \
         `rustls-webpki` TLS certificate-validation path REACHABLE.\n\n\
         Four advisories are currently accepted in `.cargo/audit.toml` on the \
         explicit premise that this code path is dead:\n\
         \x20 RUSTSEC-2026-0104 (reachable panic in CRL parsing)\n\
         \x20 RUSTSEC-2026-0098 / -0099 (name-constraint bypasses)\n\
         \x20 RUSTSEC-2026-0049 (CRL Distribution Point matching)\n\n\
         That premise no longer holds. Do NOT silence this test — go to \
         `.cargo/audit.toml` and re-decide those four, which now means either \
         bumping past the `libsql = \"=0.9.30\"` pin or accepting a LIVE \
         vulnerability knowingly.\n\n\
         Found {} occurrence(s):\n{}",
        hits.len(),
        hits.join("\n")
    );
}
