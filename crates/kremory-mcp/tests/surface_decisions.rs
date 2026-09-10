#![allow(clippy::unwrap_used, clippy::expect_used)]
//! **PAR-G3** — every public `Memory` operation must carry a recorded DECISION about
//! whether an MCP agent can reach it.
//!
//! # This is NOT a parity gate, and that distinction is the point
//!
//! `kremory-napi` mirrors the facade 1:1 (ADR-031) because a Node consumer is the
//! same consumer in another language. An MCP client is an **LLM agent**, and the
//! surface audit
//! (`.ai-docs/research/kremory-mcp-tool-surface-audit--mcp-tool-design-best-practice-2026-07-13.md`)
//! ratified a deliberately **small, waved** surface for it — semantically overlapping
//! tools cost the agent disambiguation. That audit *collapsed two inherited tools into
//! one* on exactly that principle.
//!
//! So 41 of 46 facade operations are deliberately NOT tools, and a gate demanding 1:1
//! parity would enforce the opposite of a researched decision. What this gate asserts
//! is that the non-exposure is **decided**, not accidental.
//!
//! # The drift class it catches
//!
//! A new capability ships and nobody asks whether an agent should reach it. That is
//! not hypothetical here: ADR-078 flipped `content-search` on by default, nobody
//! re-checked the consumer surfaces, and E2E-2 (a broken consumer journey) then
//! survived weeks of green builds because nothing looked.
//!
//! # Bidirectional
//!
//! - facade op with no row      → FAIL (undecided capability)
//! - row naming no facade op    → FAIL (stale after a rename/removal)
//! - `exposed_as` naming a tool that does not exist in `lib.rs` → FAIL (claimed but unwired)
//!
//! The third is the one that would otherwise let this file *claim* coverage it does
//! not have — a decision record that certifies a tool into existence.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/kremory-mcp
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root is two levels above crates/kremory-mcp")
        .to_path_buf()
}

/// Public operations on EVERY `impl Memory` block under `crates/kremory/src/facade/`
/// — the tool-shaped entry points. Builder methods (`RecallRequest::as_of` etc.) are
/// deliberately excluded: tools map to OPERATIONS, and a builder knob is a parameter
/// on one, not a tool.
///
/// ⚠️ This walks the whole DIRECTORY, not `facade/mod.rs` alone. It used to read that
/// one file, which made the guard one refactor away from decoration: splitting
/// `facade/mod.rs` (TD-043) moved six embedding operations into `facade/embeddings/`
/// and the guard reported them as REMOVED. That false alarm was the harmless
/// direction. The silent one is the reason this changed — a NEW `Memory` method added
/// in any sibling file would never have been required to carry a decision at all, and
/// nothing would have said so.
fn facade_operations(root: &Path) -> Vec<String> {
    let dir = root.join("crates/kremory/src/facade");
    let mut files = Vec::new();
    collect_rs_files(&dir, &mut files);
    assert!(
        !files.is_empty(),
        "no .rs files under {} — the facade moved and this guard is now scanning nothing",
        dir.display()
    );

    let mut ops = Vec::new();
    for file in &files {
        let src = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", file.display()));
        ops.extend(memory_impl_ops(&src));
    }
    assert!(
        ops.contains(&"remember".to_string()) && ops.contains(&"recall".to_string()),
        "the walker found no `remember`/`recall` — it is not reading the impl blocks it \
         thinks it is, and an empty scan passes every assertion below"
    );
    ops.sort();
    ops.dedup();
    ops
}

/// Every `.rs` file under `dir`, recursively.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry must be readable").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `true` for `impl Memory {` and nothing else.
///
/// The prefix test alone (`starts_with("impl Memory")`) also matches
/// `impl MemoryBuilder<…>`, which is how the first version of this walker reported
/// `with_llm` / `with_embedder` as undecided facade OPERATIONS — they are builder
/// knobs, exactly the category the doc above says is excluded.
fn is_memory_impl_header(line: &str) -> bool {
    let t = line.trim_start();
    match t.strip_prefix("impl Memory") {
        Some(rest) => rest.trim_start().starts_with('{'),
        None => false,
    }
}

/// `pub fn` / `pub async fn` names inside every `impl Memory` block in one file.
///
/// `impl Memory` only — `impl std::fmt::Debug for Memory` and the request-builder
/// impls do not match the prefix, which is the intended exclusion.
fn memory_impl_ops(src: &str) -> Vec<String> {
    let lines: Vec<&str> = src.lines().collect();
    let mut ops = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if !is_memory_impl_header(lines[i]) {
            i += 1;
            continue;
        }
        let mut depth: i32 = 0;
        let mut entered = false;
        while i < lines.len() {
            let line = lines[i];
            depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
            if !entered {
                if line.contains('{') {
                    entered = true;
                }
                i += 1;
                continue;
            }
            if depth <= 0 {
                i += 1;
                break;
            }
            let t = line.trim_start();
            i += 1;
            if t.starts_with("//") {
                continue;
            }
            // `pub fn foo(` / `pub async fn foo(`
            let rest = match t.strip_prefix("pub async fn ").or_else(|| t.strip_prefix("pub fn ")) {
                Some(r) => r,
                None => continue,
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            // `*_for_test` accessors are `#[cfg(test/test-utils)]`-gated and explicitly
            // documented as outside the stable public API.
            if !name.is_empty() && !name.ends_with("_for_test") {
                ops.push(name);
            }
        }
    }
    ops
}

/// Tool names actually registered in `kremory-mcp/src/lib.rs`.
fn registered_tools(root: &Path) -> HashSet<String> {
    let src = std::fs::read_to_string(root.join("crates/kremory-mcp/src/lib.rs"))
        .expect("kremory-mcp/src/lib.rs must be readable");
    let mut out = HashSet::new();
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("//") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("name = \"") {
            if let Some(end) = rest.find('"') {
                let n = &rest[..end];
                if n.starts_with("kremory_") {
                    out.insert(n.to_string());
                }
            }
        }
    }
    out
}

/// `op -> (exposed_as, deferred)` from `surface-decisions.toml`.
///
/// Hand-parsed rather than pulling in a `toml` dev-dependency for a 50-line file of
/// one fixed shape — and a hand parser silently matching nothing is exactly what the
/// non-vacuity floor below exists to catch.
fn decisions(root: &Path) -> HashMap<String, (Option<String>, Option<String>)> {
    let src = std::fs::read_to_string(root.join("crates/kremory-mcp/surface-decisions.toml"))
        .expect("surface-decisions.toml must be readable");
    let mut out = HashMap::new();
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with('#') || t.is_empty() || t.starts_with('[') {
            continue;
        }
        let Some((key, val)) = t.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            continue;
        }
        let field = |name: &str| -> Option<String> {
            val.find(&format!("{name} = \""))
                .map(|i| &val[i + name.len() + 4..])
                .and_then(|r| r.find('"').map(|e| r[..e].to_string()))
        };
        out.insert(key.to_string(), (field("exposed_as"), field("deferred")));
    }
    out
}

#[test]
fn every_facade_operation_has_a_recorded_mcp_surface_decision() {
    let root = repo_root();
    let ops = facade_operations(&root);
    let decided = decisions(&root);
    let tools = registered_tools(&root);

    // ── NON-VACUITY FLOORS ───────────────────────────────────────────────────
    // Each scan is a hand-rolled parser, and a hand-rolled parser that matches
    // NOTHING reports "all decided" having examined nothing — a control that
    // certifies by not looking. Measured 2026-08-05: 46 ops, 5 tools, 46 rows.
    assert!(
        ops.len() >= 40,
        "facade scan found only {} operations (expected >= 40) — the SCAN is broken, \
         not the code. Do not read this as 'all decided'. Check: did `impl Memory` \
         move out of facade/mod.rs, or change formatting?",
        ops.len()
    );
    assert!(
        tools.len() >= 5,
        "tool scan found only {} registered tools (expected >= 5) — the scan is broken. \
         Check the `name = \"kremory_…\"` shape in kremory-mcp/src/lib.rs.",
        tools.len()
    );
    assert!(
        decided.len() >= 40,
        "decision-file scan found only {} rows (expected >= 40) — the TOML parse is \
         broken, not the file.",
        decided.len()
    );

    // ── 1. Every facade op must be DECIDED ───────────────────────────────────
    let undecided: Vec<&String> = ops.iter().filter(|o| !decided.contains_key(*o)).collect();
    assert!(
        undecided.is_empty(),
        "these public `Memory` operations have NO recorded MCP surface decision:\n  {}\n\n\
         A new capability shipped and nobody decided whether an LLM agent should reach \
         it. Add a row to crates/kremory-mcp/surface-decisions.toml with either\n  \
         `exposed_as = \"kremory_x\"` or `deferred = \"why not\"`.\n\
         `deferred` is the NORMAL answer — the MCP surface is deliberately small \
         (see the tool-surface audit). What is not acceptable is leaving it undecided.",
        undecided
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    // ── 2. Every row must name a REAL facade op (stale-row / rename guard) ────
    let op_set: HashSet<&str> = ops.iter().map(String::as_str).collect();
    let stale: Vec<&String> = decided
        .keys()
        .filter(|k| !op_set.contains(k.as_str()))
        .collect();
    assert!(
        stale.is_empty(),
        "these rows name operations that no longer exist on `Memory`:\n  {}\n\n\
         A rename or removal left the decision behind. Update or delete the row — a \
         decision about a vanished capability reads as coverage and is not.",
        stale.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n  ")
    );

    // ── 3. Every `exposed_as` must name a tool that ACTUALLY EXISTS ──────────
    // Without this the file could claim exposure it does not have — a decision
    // record certifying a tool into existence.
    let mut phantom = Vec::new();
    for (op, (exposed, _)) in &decided {
        if let Some(tool) = exposed {
            if !tools.contains(tool) {
                phantom.push(format!("{op} -> {tool}"));
            }
        }
    }
    assert!(
        phantom.is_empty(),
        "these rows claim an MCP tool that is NOT registered in kremory-mcp/src/lib.rs:\n  {}\n\n\
         Registered tools: {:?}",
        phantom.join("\n  "),
        {
            let mut t: Vec<&str> = tools.iter().map(String::as_str).collect();
            t.sort_unstable();
            t
        }
    );

    // ── 4. Every row must carry EXACTLY ONE of the two fields ────────────────
    let malformed: Vec<String> = decided
        .iter()
        .filter(|(_, (e, d))| e.is_some() == d.is_some())
        .map(|(op, _)| op.clone())
        .collect();
    assert!(
        malformed.is_empty(),
        "these rows have both `exposed_as` and `deferred`, or neither:\n  {}",
        malformed.join("\n  ")
    );
}
