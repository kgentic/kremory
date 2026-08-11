//! ADR-029d ratchet — every `entities` WRITE must filter on `group_id`.
//!
//! ## Why this exists
//!
//! The entity key is the COMPOSITE `(id, group_id)` — the `facts` FK says so
//! (`REFERENCES entities(id, group_id)`), and ADR-029d requires every read and
//! write to filter on namespace. A write keyed on `id` alone hits **every**
//! namespace holding that name.
//!
//! TD-206 was exactly this, on `entities.embedding`, on the live ingest path, at
//! FIVE call sites. The instructive part is how it was found: NOT by reading.
//! Two rounds of `grep` over `src/` concluded "no production callers" — a
//! conclusion produced by a `| head -5` that truncated the real callers off the
//! end of the list. What enumerated them exhaustively was making the unscoped
//! method stop existing, so the COMPILER had to name every caller.
//!
//! This test is the generalisation of that lesson. It cannot rely on a human
//! grepping carefully, because that is the step already proven to fail.
//!
//! ## What it does and does NOT claim
//!
//! It does NOT claim the class is empty. **It is not** — there are 15 known
//! unscoped writes, enumerated in `ALLOWLIST` below with a classification each.
//! What it claims is narrower and checkable: **the class cannot GROW.** A new
//! unscoped `entities` write fails this test with its file, line and SQL.
//!
//! A count-only ceiling was rejected: fixing one site while adding another nets
//! zero and passes silently. The allowlist is keyed on the SQL itself, so both
//! movements are visible.
//!
//! ## Legitimate exemptions
//!
//! * **Migrations** — operate on the whole database by definition; a migration
//!   scoped to one namespace would be a bug.
//! * **Test modules** — single-namespace fixtures where the distinction cannot
//!   arise.
//! * **`WHERE id = ?` on a globally-unique key** — does NOT apply to `entities`
//!   (composite key), but DOES apply to `facts.id` (`INTEGER PRIMARY KEY
//!   AUTOINCREMENT`), which is why this lint targets `entities` only.

#![cfg(feature = "test-utils")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

/// (file suffix, SQL prefix, classification) — every currently-known unscoped
/// `entities` write. A row here is a STATEMENT ABOUT THE SITE, not a blessing:
/// `NEEDS-FIX` rows are tracked as TD-207 and are expected to shrink.
const ALLOWLIST: &[(&str, &str, &str)] = &[
    // ── Migrations — whole-DB by design ─────────────────────────────────────
    // NOTE: prefixes are matched against a SINGLE SOURCE LINE, because that is
    // what the violation reporter prints. A prefix copied from the normalised
    // multi-line statement will never match — the first cut of this file made
    // exactly that mistake and `allowlist_has_no_stale_entries` caught it, which
    // is the staleness check doing its job on its own author.
    (
        "core/migrations/defs_c.rs",
        "SET entity_type_source = 'Phase1Ner',",
        "OK: migration — backfills every namespace by design",
    ),
    (
        "core/migrations/defs_e.rs",
        "WHERE entity_type_source = 'DreamPass4'",
        "OK: migration — whole-DB correction by design",
    ),
    (
        "core/migrations/defs_b.rs",
        "UPDATE entities",
        "OK: migration — backfills entity_type_id across every namespace by design. \
         (Formatted as a raw multi-line string with no `\\` continuations, so the \
         statement window ends at line 1 — the identity is necessarily coarse here, \
         and this file contains exactly one `UPDATE entities`.)",
    ),
    // ── Test fixtures ───────────────────────────────────────────────────────
    (
        "core/canonicalization.rs",
        "UPDATE entities SET ner_confidence = 0.6 WHERE id =",
        "OK: #[cfg(test)] fixture (mod tests begins ~:1417), single namespace",
    ),
    (
        "core/canonicalization.rs",
        "UPDATE entities SET ner_confidence = 0.8 WHERE id =",
        "OK: #[cfg(test)] fixture, single namespace",
    ),
    (
        "core/canonicalization.rs",
        "UPDATE entities SET access_count = 7 WHERE id =",
        "OK: #[cfg(test)] fixture, single namespace",
    ),
    (
        "core/graph/entities.rs",
        "UPDATE entities SET embedding = vector(?1) WHERE id = ?2",
        "OK: `set_entity_embedding` is #[cfg(any(test, feature = \"test-utils\"))] \
         (TD-206) — unrepresentable in a production build",
    ),
    // ── NEEDS-FIX — real production writes, tracked as TD-207 ────────────────
    (
        "core/search.rs",
        "UPDATE entities SET access_count = access_count + 1 WHERE id IN",
        "NEEDS-FIX (TD-207): inflates access_count across namespaces. Low harm \
         today — `graph_degree_bonus` reads `degree`, not `access_count`, so it \
         does not reach ranking — but it is the same defect class.",
    ),
    (
        "core/graph/entities.rs",
        "UPDATE entities SET embedding = {} WHERE id = ?1",
        "NEEDS-FIX (TD-207): second embedding-write path; same class as TD-206.",
    ),
    (
        "core/graph/entities.rs",
        "UPDATE entities SET properties = ?1, updated_at = ?2 WHERE id = ?3",
        "NEEDS-FIX (TD-207): properties overwrite hits every namespace.",
    ),
    (
        "core/graph/entities.rs",
        "UPDATE entities SET entity_type_id = ?1, updated_at = ?2 WHERE id = ?3",
        "NEEDS-FIX (TD-207): entity type overwrite hits every namespace.",
    ),
    (
        "core/graph/entities.rs",
        "DELETE FROM entities WHERE id IN",
        "NEEDS-FIX (TD-207): DELETE — highest-severity shape in this list.",
    ),
    (
        "core/graph/entities.rs",
        "DELETE FROM entities WHERE id LIKE ?1",
        "NEEDS-FIX (TD-207): prefix DELETE, unscoped.",
    ),
    (
        "core/graph/queries.rs",
        "DELETE FROM entities WHERE id = ?1",
        "NEEDS-FIX (TD-207): DELETE, unscoped.",
    ),
    (
        "core/graph/queries.rs",
        "DELETE FROM entities WHERE id IN",
        "NEEDS-FIX (TD-207): DELETE, unscoped.",
    ),
    (
        "core/dream/consistency_check/audit.rs",
        "UPDATE entities SET entity_type_id = ?1, entity_type_source = 'DreamPass4',",
        "NEEDS-FIX (TD-207): dream retype hits every namespace.",
    ),
    // Migration SCRATCH tables. `entities_new` / `entities_new_023` are transient
    // rebuild tables, not the entity store — matched here only because they are
    // prefixed `entities`. The word-boundary check in `is_entities_write` already
    // excludes them; these rows exist so the staleness check does not flag the
    // exclusion as unused if the boundary logic is ever loosened.
    (
        "core/migrations/defs_a.rs",
        "DELETE FROM entities_new",
        "OK: migration scratch table, not the entity store",
    ),
    (
        "core/migrations/defs_j.rs",
        "DELETE FROM entities_new_023",
        "OK: migration scratch table, not the entity store",
    ),
];

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let p = entry.expect("entry").path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// Collapse whitespace / line-continuations into one line.
fn normalise(s: &str) -> String {
    s.replace("\\\n", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Does `line` mention a write to the `entities` table specifically?
///
/// Word-boundary matched on purpose. The first cut of this lint matched
/// `entities` as a substring and fired on `DELETE FROM entities_new_023` — a
/// migration SCRATCH table, not the entity store. A lint that fires on correct
/// code gets disabled, and a disabled lint protects nothing
/// (`over-blocking-is-a-security-failure`), so the false positive is a defect in
/// the lint, not a row for the allowlist.
fn is_entities_write(line: &str) -> bool {
    let u = line.to_uppercase();
    for verb in ["UPDATE ", "DELETE FROM "] {
        let mut from = 0usize;
        while let Some(rel) = u[from..].find(verb) {
            let after = from + rel + verb.len();
            let tail = u[after..].trim_start();
            if let Some(rest) = tail.strip_prefix("ENTITIES") {
                // Word boundary: reject `ENTITIES_FTS`, `ENTITIES_NEW`, etc.
                if !rest.starts_with(|c: char| c == '_' || c.is_ascii_alphanumeric()) {
                    return true;
                }
            }
            from = after;
        }
    }
    false
}

/// A `//`-comment line. SQL in this crate never lives in a comment, and comment
/// prose routinely QUOTES SQL while discussing it — the "doing vs mentioning"
/// confusion that any text-matching guard has by construction.
fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

#[test]
fn no_new_unscoped_entities_writes() {
    let root = src_root();
    let mut files = Vec::new();
    rs_files(&root, &mut files);
    assert!(
        files.len() > 20,
        "PRECONDITION: expected to scan the whole src tree, found only {} files — \
         if this ever reads 0 the lint passes VACUOUSLY, which is the failure mode \
         it exists to prevent",
        files.len()
    );

    let mut violations: Vec<String> = Vec::new();

    for path in &files {
        let src = std::fs::read_to_string(path).expect("read source");
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        // Line-oriented, NOT string-literal-parsing. The first cut walked `"`
        // pairs and mis-paired on an apostrophe inside a doc comment, slurping
        // ~400 characters of prose and reporting it as SQL. Rust SQL here is
        // always a multi-line literal with `\` continuations, so a window of the
        // following lines is both simpler and more faithful.
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if is_comment(line) || !is_entities_write(line) {
                continue;
            }
            // Take the WHOLE statement, not a fixed number of lines. A fixed
            // 4-line window truncated `discover_types.rs:870` and
            // `reclassify.rs:634` before their `WHERE id = ?3 AND group_id = ?4`
            // and reported two CORRECTLY-scoped writes as violations. A lint that
            // fires on correct code trains people to switch it off
            // (`over-blocking-is-a-security-failure`), so the window follows the
            // literal to its close: kremory's SQL uses `\` continuations, so the
            // statement ends on the first line that does NOT end with a backslash.
            let mut window: Vec<&str> = Vec::new();
            for l in &lines[i..(i + 15).min(lines.len())] {
                if is_comment(l) {
                    break;
                }
                window.push(l);
                if !l.trim_end().ends_with('\\') {
                    break;
                }
            }
            let window = window.join(" ");
            let stmt = normalise(&window);
            if stmt.to_uppercase().contains("GROUP_ID") {
                continue;
            }
            // Identity is the normalised STATEMENT WINDOW, not the single line.
            // kremory formats SQL as `"UPDATE entities \` with the SET clause on
            // the following line, so a line-keyed identity reads `UPDATE entities`
            // for every site — indistinguishable, and useless as an allowlist key.
            let sql = stmt
                .replace('"', " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let allowed = ALLOWLIST
                .iter()
                .any(|(f, prefix, _)| rel.ends_with(f) && sql.contains(prefix));
            if !allowed {
                violations.push(format!("  {rel}:{}\n      {sql}", i + 1));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "ADR-029d: {} NEW unscoped `entities` write(s) — every write must filter on \
         `group_id`, because the entity key is the composite `(id, group_id)`.\n\n{}\n\n\
         TD-206 was this defect on the ingest path, at five call sites, and TWO rounds \
         of grep declared it absent. If the write is genuinely whole-DB (a migration) \
         or test-only, add it to ALLOWLIST with that classification. Do NOT add it \
         unclassified.",
        violations.len(),
        violations.join("\n")
    );
}

/// The allowlist is a ratchet, so a STALE entry is a silent hole: it would keep
/// excusing a site that has since been fixed or moved, and would excuse a NEW
/// site that happened to reuse the SQL. Fail if an entry matches nothing.
#[test]
fn allowlist_has_no_stale_entries() {
    let root = src_root();
    let mut files = Vec::new();
    rs_files(&root, &mut files);

    let mut unmatched: Vec<&str> = Vec::new();
    for (f, sql, _reason) in ALLOWLIST {
        let found = files.iter().any(|p| {
            let rel = p.to_string_lossy().replace('\\', "/");
            if !rel.ends_with(f) {
                return false;
            }
            // Normalise the WHOLE file: kremory's SQL spans continuation lines,
            // so a per-line search cannot see a prefix that straddles them.
            let src = std::fs::read_to_string(p).unwrap_or_default();
            let flat = src
                .replace('"', " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            flat.contains(*sql)
        });
        if !found {
            unmatched.push(sql);
        }
    }

    assert!(
        unmatched.is_empty(),
        "ADR-029d allowlist has {} entry(ies) matching nothing — the site was fixed, \
         moved or reworded. Remove them, so the ratchet keeps ratcheting:\n{}",
        unmatched.len(),
        unmatched
            .iter()
            .map(|s| format!("  - {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
