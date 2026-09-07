// `expect_used`/`unwrap_used` are denied package-wide by `[lints.clippy]`, which
// applies to test targets too. Every site below is test scaffolding whose
// failure means the harness itself is broken (git missing, repo root wrong), and
// that must abort loudly rather than be silently handled. Same scoping, and same
// reasoning, as the sibling guards `public_surface_hygiene.rs` and
// `no_workspace_terminology_in_lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Guard — every `ADR-NNN` cited under `crates/` must resolve to a real ADR file.
//!
//! # Why this exists
//!
//! A citation to a decision record is only worth anything if the record can be
//! found. On 2026-09-06 a new ADR was numbered 079 and the collision was caught
//! only by accident — a test name (`adr079_...`) clashed with an `-E test(/adr079/)`
//! filter. Had the clash not been mechanical, two unrelated decisions would now
//! share a number. This test makes the check mechanical instead of lucky.
//!
//! # The failure this guard is built to avoid: looking in ONE directory
//!
//! TD-236 recorded ADR-079 as MISSING and asked for it to be written from
//! scratch. It was never missing. It sits in `.ai-docs/adrs/rql/`, and the
//! register's own text cites that exact path. The claim came from listing a
//! single directory, and acting on it would have produced a SECOND, conflicting
//! ADR-079 — the precise harm the row was filed to prevent.
//!
//! ADR files are spread across at least ten directories (`.ai-docs/adrs/`,
//! `.ai-docs/adrs/rql/`, `.ai-docs/adrs/rql/adr-029-namespace-policy/`,
//! `.ai-docs/decisions/`, `.ai-docs/specs/`, `.ai-docs/plans/`, `docs/adr/`, …).
//! So resolution here matches on FILENAME ANYWHERE IN THE REPO, never on a
//! directory allowlist. A directory list would have to be updated by whoever
//! adds the eleventh location, which is exactly the person who will not know it
//! exists. Measured 2026-09-06: a directory-scoped check reports ADR-074 and
//! ADR-079 missing; both resolve.
//!
//! # Design notes, each one measured rather than assumed
//!
//! * **Exactly three digits, with a non-digit boundary.** Without the boundary
//!   test, `ADR-2026-05-20` (a real date-slugged citation in
//!   `core/background/deferred_pipeline.rs`) is read as a dangling three-digit
//!   reference and reported. That is a pure false positive on ordinary work,
//!   and a guard that fires on ordinary work gets deleted.
//!
//!   ⚠️ The bad three-digit form is DESCRIBED here rather than written out,
//!   deliberately. This scanner walks `git ls-files`, so it scans ITS OWN
//!   SOURCE — and it did not notice, because while this file was untracked it
//!   was outside the scan set. It went green, was committed, and failed on the
//!   very next run, citing its own prose. Any example ADR number written
//!   literally in this file becomes a citation the guard must then resolve.
//!
//!   Generalises past this file: a document that DESCRIBES a string must keep
//!   that string out of the path of anything that scans for it. The same shape
//!   bit twice more today — a blanket ADR renumber rewrote eight unrelated
//!   citations, and a blanket npm-scope rename rewrote the register entry that
//!   was documenting the rename.
//! * **Scans every tracked file under `crates/`, with no extension allowlist.**
//!   An allowlist rots the moment a citation lands in a file type nobody listed.
//!   The obvious counter-risk is a recorded VCR cassette in which a model
//!   hallucinates an ADR number; measured 2026-09-06, the 582 fixture/cassette
//!   files under `crates/` contain **zero** occurrences of even the substring
//!   `ADR`, so the risk is theoretical. If it ever fires it is loud and cheap to
//!   exempt, whereas a silent gap is neither.
//! * **Uses `git ls-files -z` with NO pathspec, filtering in Rust.** Follows the
//!   reasoning already written down in `public_surface_hygiene.rs`: git's
//!   pathspec globbing was measured (2026-09-05) to silently miss real files,
//!   and a filesystem walk descends into `.venv/` and gitignored dumps. `-z`
//!   additionally makes the record count independent of newlines in filenames.
//! * **Treats binary and non-UTF-8 as findings, never as skips.** `grep` skips a
//!   NUL-containing file silently and still exits 0, converting "did not look"
//!   into "no matches". Measured 2026-09-06: of 988 tracked files under
//!   `crates/`, zero contain NUL and zero are invalid UTF-8, so failing on them
//!   costs nothing today and refuses to go quiet later.
//! * **Asserts non-vacuity floors.** If the walk or the extractor breaks, the
//!   scan set empties and every assertion passes trivially. A vacuous pass is
//!   indistinguishable from an earned one, so the floors are what make green
//!   mean something here.
//!
//! # Scope
//!
//! `crates/` only, per TD-236. `.ai-docs/` prose legitimately discusses ADR
//! numbers that were proposed, renumbered or never written, and pointing this at
//! documentation would fire constantly on ordinary writing.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Repo root, derived from the compile-time manifest dir rather than the process
/// CWD (which differs between `cargo test` and `cargo nextest`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("manifest dir should be <root>/crates/kremory")
        .to_path_buf()
}

/// Every path tracked by git, repo-relative.
///
/// Invoked with NO pathspec; all filtering happens in Rust below. See the module
/// docs for why the pathspec is not trusted to do it.
fn tracked_paths(root: &Path) -> Vec<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .expect("git ls-files should run");
    assert!(
        out.status.success(),
        "git ls-files failed: {:?}",
        out.status
    );
    let listing = String::from_utf8(out.stdout).expect("tracked paths are UTF-8");
    listing
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Parse a leading `adr-NNN` from a filename, requiring a non-digit boundary so
/// a hypothetical `adr-0791-*.md` cannot be mistaken for ADR-079.
fn adr_number_from_filename(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let rest = lower.strip_prefix("adr-")?;
    let digits: String = rest.chars().take(3).collect();
    if digits.len() != 3 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if rest.chars().nth(3).is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(digits)
}

/// Every ADR number for which a file exists ANYWHERE in the repo.
fn available_adr_numbers(paths: &[String]) -> BTreeSet<String> {
    paths
        .iter()
        .filter_map(|p| p.rsplit('/').next())
        .filter_map(adr_number_from_filename)
        .collect()
}

/// Extract every `ADR-NNN` citation from a body of text.
///
/// Case-insensitive so both the prose form (`ADR-079`) and the slug form
/// (`adr-079-contradiction-…`) resolve. Requires the preceding byte to be a
/// non-identifier character so an ADR number embedded in a longer word is not
/// invented, and requires exactly three digits so `ADR-2026-05-20` is left alone.
fn extract_adr_refs(text: &str) -> BTreeSet<String> {
    // `to_ascii_lowercase` maps only A-Z to a-z, so byte indices stay aligned
    // with the original. Both are indexed as bytes, never sliced as `str`, so
    // multi-byte UTF-8 cannot panic here.
    let raw = text.as_bytes();
    let lower = text.to_ascii_lowercase();
    let low = lower.as_bytes();

    let mut found = BTreeSet::new();
    for i in 0..low.len().saturating_sub(6) {
        if &low[i..i + 4] != b"adr-" {
            continue;
        }
        // Reject a match glued to the end of an identifier or word.
        if i > 0 && (raw[i - 1].is_ascii_alphanumeric() || raw[i - 1] == b'_') {
            continue;
        }
        let digits = &low[i + 4..i + 7];
        if !digits.iter().all(u8::is_ascii_digit) {
            continue;
        }
        // Exactly three digits — this is what excludes `ADR-2026-05-20`.
        if low.get(i + 7).is_some_and(u8::is_ascii_digit) {
            continue;
        }
        found.insert(String::from_utf8_lossy(digits).into_owned());
    }
    found
}

/// ADR numbers cited under `crates/` that legitimately have no `adr-NNN-*` file.
///
/// These are v0.1.0-era decisions that were recorded as ROWS in the architect
/// index rather than promoted to standalone files. They are real, ratified and
/// still load-bearing in the code that cites them — the record simply lives in
/// a different artifact shape, so demanding a file would be demanding a
/// rewrite of settled history.
///
/// Spelled out one number at a time, each with the line that holds it, rather
/// than expressed as a range (`< 026`) — a range would silently absorb any
/// future dangling reference in that span, which is the failure this guard
/// exists to catch. `exempt_numbers_are_all_still_needed` below fails if any
/// entry stops being necessary, so an exemption cannot outlive its reason.
///
/// ADR-019 and ADR-020 were removed from this list on 2026-09-07: the same
/// comment-hygiene pass stripped every citation to each under `crates/` (both
/// had cited only doc comments, none of which resolved to a file — exactly
/// the exemption this list existed for), so `exempt_numbers_are_all_still_needed`
/// correctly caught both exemptions as dead and this list follows its own rule.
const EXEMPT_ADR_NUMBERS: [(&str, &str); 1] = [(
    "022",
    "Mutex<()> write-serialiser closing DENT-002 — recorded at \
     .ai-docs/architecture/kremory-v010-architect-INDEX-2026-05-26.md:244, never a file",
)];

/// The doc that holds the exempt decisions above. If it moves, the justifications
/// above become unverifiable and the exemptions must be re-grounded.
const EXEMPTION_SOURCE_DOC: &str =
    ".ai-docs/architecture/kremory-v010-architect-INDEX-2026-05-26.md";

/// Read as UTF-8, or report why not. Never silently skips: an unreadable file in
/// the scan set is a finding, because a skip is indistinguishable from a pass.
fn read_text(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("unreadable: {e}"))?;
    if bytes.contains(&0) {
        return Err("contains NUL bytes — binary file in the ADR-citation scan set".into());
    }
    String::from_utf8(bytes).map_err(|_| "not valid UTF-8".to_string())
}

/// Collect `(adr_number -> citing sites)` for everything tracked under `crates/`,
/// alongside any file that could not be read.
/// This file's own repo-relative path. See the exclusion in `scan_crates`.
///
/// Asserted to exist by `self_path_is_accurate` below, so a rename cannot
/// silently turn the exclusion into a no-op — at which point the guard would
/// start failing on its own fixtures again, which is precisely how it first
/// broke.
const SELF_PATH: &str = "crates/kremory/tests/it/adr_reference_integrity.rs";

fn scan_crates(root: &Path, tracked: &[String]) -> (Vec<(String, String)>, Vec<String>, usize) {
    let mut refs: Vec<(String, String)> = Vec::new();
    let mut unreadable: Vec<String> = Vec::new();
    let mut scanned = 0usize;

    for rel in tracked.iter().filter(|p| p.starts_with("crates/")) {
        // The scanner does not scan ITSELF. This is not a convenience
        // exemption — it is structural.
        //
        // This file necessarily contains fake ADR numbers: they are the
        // fixtures for `dangling_reference_is_detected` and the boundary
        // cases in the design notes. Scanning them makes the guard cite its
        // own test data and fail permanently.
        //
        // It went undetected in exactly the way that matters: the scan set is
        // `git ls-files`, so while this file was UNTRACKED it was invisible to
        // itself and the suite went green. It failed on the first run after
        // being committed. A guard whose own arrival breaks the build is worse
        // than no guard, because the obvious fix is to delete it.
        //
        // Cost of the exemption, stated plainly: a REAL ADR citation written
        // in this file would not be checked. That is acceptable — this file
        // guards ADR citations, it does not make architectural decisions, so
        // it has no business citing one. If that ever changes, cite it from
        // the code that implements the decision instead.
        if rel == SELF_PATH {
            continue;
        }
        let path = root.join(rel);
        if !path.is_file() {
            continue;
        }
        match read_text(&path) {
            Ok(text) => {
                scanned += 1;
                for n in extract_adr_refs(&text) {
                    refs.push((n, rel.clone()));
                }
            }
            Err(why) => unreadable.push(format!("{rel}: {why}")),
        }
    }
    (refs, unreadable, scanned)
}

#[test]
fn every_adr_cited_under_crates_has_a_file() {
    let root = repo_root();
    let tracked = tracked_paths(&root);
    let available = available_adr_numbers(&tracked);
    let (refs, unreadable, scanned) = scan_crates(&root, &tracked);

    // Non-vacuity floors. Their job is to catch a BROKEN instrument (which
    // yields ~zero), not to track the true counts, so each sits well below the
    // measured value: 2026-09-06 saw 988 files scanned, 56 distinct citations
    // and 63 ADR files. Set loosely enough that ordinary deletions cannot
    // false-positive.
    assert!(
        scanned >= 300,
        "scanned only {scanned} files under crates/ — the walk is broken and this \
         test would pass vacuously"
    );
    assert!(
        available.len() >= 30,
        "found only {} ADR files in the repo — the filename matcher is broken and \
         this test would fail spuriously for every citation",
        available.len()
    );
    let distinct: BTreeSet<&String> = refs.iter().map(|(n, _)| n).collect();
    assert!(
        distinct.len() >= 20,
        "extracted only {} distinct ADR citations from {scanned} files — the \
         extractor is broken and this test would pass vacuously",
        distinct.len()
    );

    assert!(
        unreadable.is_empty(),
        "files in the ADR-citation scan set could not be read, so they were NOT \
         checked ({} of {scanned}):\n  {}",
        unreadable.len(),
        unreadable.join("\n  ")
    );

    let exempt: BTreeSet<&str> = EXEMPT_ADR_NUMBERS.iter().map(|(n, _)| *n).collect();
    let mut findings: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (number, site) in &refs {
        if available.contains(number) || exempt.contains(number.as_str()) {
            continue;
        }
        if seen.insert(number.clone()) {
            findings.push(format!(
                "ADR-{number} cited at {site} — no adr-{number}-* file exists"
            ));
        }
    }

    assert!(
        findings.is_empty(),
        "ADR citations under crates/ do not resolve to any ADR file ({} of {} \
         distinct citations, {scanned} files scanned).\nEither the ADR was never \
         written, or it exists under a number this citation gets wrong. Search the \
         WHOLE repo before concluding it is missing — ADR files live in ~10 \
         directories, and assuming otherwise is what produced TD-236's false alarm.\n  {}",
        findings.len(),
        distinct.len(),
        findings.join("\n  ")
    );
}

/// Sensitivity, both directions. A guard proven only on what it should catch is
/// untested — this pins that it FIRES on real citations and stays SILENT on the
/// lookalikes that would otherwise make it a nuisance.
#[test]
fn extractor_finds_real_citations_and_ignores_lookalikes() {
    for (text, want) in [
        ("per ADR-079 rev.2 the default is ON", "079"),
        ("slug: adr-068-as-of-temporal-recall-2026-07-03", "068"),
        ("(ADR-074 / TD-116)", "074"),
        ("see ADR-001.", "001"),
    ] {
        let got = extract_adr_refs(text);
        assert!(
            got.contains(want),
            "extractor missed a real citation {want} in {text:?} (got {got:?})"
        );
    }

    // False positives are a correctness property here, not an acceptable cost.
    // `ADR-2026-05-20` is the load-bearing case: it is a real date-slugged
    // citation in `core/background/deferred_pipeline.rs`, and a naive
    // three-digit match reads it as the non-existent ADR-202.
    for text in [
        "triple-emit (ADR-2026-05-20 D1)",
        "ADR-07 is too short",
        "BADR-123 is part of a word",
        "snake_adr-123 is an identifier",
        "ADR-abc is not a number",
    ] {
        assert!(
            extract_adr_refs(text).is_empty(),
            "extractor false-positived on ordinary text: {text:?} -> {:?}",
            extract_adr_refs(text)
        );
    }
}

/// An exemption rots the moment its reason stops holding: it survives, covers
/// nothing, and quietly widens the guard. This fails the build in BOTH
/// directions — if an exempt ADR gains a file (exemption now stale), and if it
/// stops being cited under `crates/` (exemption now dead).
#[test]
fn exempt_numbers_are_all_still_needed() {
    let root = repo_root();
    let tracked = tracked_paths(&root);
    let available = available_adr_numbers(&tracked);
    let (refs, _, _) = scan_crates(&root, &tracked);
    let cited: BTreeSet<&String> = refs.iter().map(|(n, _)| n).collect();

    assert!(
        root.join(EXEMPTION_SOURCE_DOC).is_file(),
        "{EXEMPTION_SOURCE_DOC} is gone — it holds the decisions the exemptions \
         below point at, so every justification is now unverifiable"
    );

    for (number, why) in EXEMPT_ADR_NUMBERS {
        assert!(
            !available.contains(number),
            "ADR-{number} is exempt but an adr-{number}-* file now exists — delete \
             the exemption, the guard can resolve it ({why})"
        );
        assert!(
            cited.contains(&number.to_string()),
            "ADR-{number} is exempt but nothing under crates/ cites it any more — \
             delete the dead exemption ({why})"
        );
    }
}

/// The self-exclusion is keyed on a hard-coded path, so a file rename would
/// turn it into a silent no-op and the guard would resume citing its own
/// fixtures. This pins it.
#[test]
fn self_path_is_accurate() {
    let root = repo_root();
    assert!(
        root.join(SELF_PATH).is_file(),
        "SELF_PATH no longer names this file ({SELF_PATH}). The self-exclusion in \
         scan_crates is now a no-op and the guard will start reporting its own test \
         fixtures as dangling ADR citations. Update SELF_PATH to the new path."
    );
}
