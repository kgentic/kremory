// File-level clippy exemption for test invariants:
//
// This file is a TEST file. Per `~/.claude/rules/testing-policy.md` test files
// are explicitly exempted from `clippy::expect_used` / `clippy::unwrap_used`.
// In this file, `.expect()` is used exclusively for test setup invariants
// (CARGO_MANIFEST_DIR existence, workspace layout, syn::parse_file of files
// known to be syntactically valid Rust). A panic at any of these sites is the
// CORRECT failure mode — it means the test harness or workspace is broken in
// a way that this test must catch loudly. Each `.expect()` carries an inline
// message identifying which invariant was violated.
//
// This is a file-scope `allow`, not per-call band-aids, so the rationale is
// documented once and applies uniformly to the invariant-style asserts below.
#![allow(clippy::expect_used)]

//! ADR-031 Layer 3 — mechanical parity test between `kremory` facade and `kremory-napi` binding.
//!
//! Uses `syn` to parse the substrate facade + napi source files, extracts public
//! symbols, and asserts that every substrate symbol either has a corresponding
//! napi mirror OR appears in `parity-skip.toml` with a documented reason.
//!
//! Fails CI with an actionable message identifying any symbol that drifts without
//! a skip entry.
//!
//! # Scope
//!
//! Tracked substrate types (all public methods/fields under these):
//! - `Memory` — pub fn on the impl block
//! - `MemoryBuilder` — pub fn on the impl block
//! - `RememberRequest` — pub fn on the impl block
//! - `RememberBatchBuilder` — pub fn on the impl block
//! - `EpisodeEntryBuilder` — pub fn on the impl block
//! - `RecallRequest` — pub fn on the impl block
//! - `ForgetRequest` — pub fn on the impl block
//! - `DreamRequest` — pub fn on the impl block
//! - `RecallTemplate` — pub fn on the impl block
//! - `DreamSummary` — pub fields on the struct
//!
//! Napi mirror surface extracted from `lib.rs` + `convert.rs`:
//! - `#[napi]` methods on `JsMemory` impl
//! - `#[napi(object)]` struct fields on option/result structs

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use syn::{ImplItem, Item};

// ── Skip-list parsing (manual — avoids toml crate !Send issues in test harness) ──

/// Parse `parity-skip.toml` into a map of `symbol → reason` without using
/// the `toml` crate's serde integration (which has thread-safety issues in the
/// test harness due to `toml::value::Table` using `Arc` internally).
///
/// We parse manually: look for `[[skip]]` section headers, then extract
/// `symbol = "..."` and `reason = "..."` key-value pairs.
fn load_skip_list(toml_path: &std::path::Path) -> HashMap<String, String> {
    let raw = std::fs::read_to_string(toml_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", toml_path.display()));

    let mut result: HashMap<String, String> = HashMap::new();
    let mut cur_symbol: Option<String> = None;
    let mut cur_reason: Option<String> = None;

    for line in raw.lines() {
        let line = line.trim();

        if line == "[[skip]]" {
            // Flush previous entry if complete.
            if let (Some(sym), Some(reason)) = (cur_symbol.take(), cur_reason.take()) {
                result.insert(sym, reason);
            } else {
                // Reset partial state.
                cur_symbol = None;
                cur_reason = None;
            }
            continue;
        }

        // Skip comments and empty lines.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(val) = extract_toml_string_value(line, "symbol") {
            cur_symbol = Some(val);
        } else if let Some(val) = extract_toml_string_value(line, "reason") {
            cur_reason = Some(val);
        }
        // `revisit` is governance metadata — not needed for the test logic.
    }

    // Flush last entry.
    if let (Some(sym), Some(reason)) = (cur_symbol, cur_reason) {
        result.insert(sym, reason);
    }

    result
}

/// Extract the string value from a TOML line of the form `key = "value"`.
/// Returns `None` if the line doesn't match.
fn extract_toml_string_value(line: &str, key: &str) -> Option<String> {
    let prefix = format!("{key} = \"");
    if !line.starts_with(&prefix) {
        return None;
    }
    let rest = &line[prefix.len()..];
    // Find the closing quote, handling trivial escape (no multi-line strings in this file).
    let end = rest.rfind('"')?;
    Some(rest[..end].to_string())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn repo_root() -> PathBuf {
    // Justification for .expect() chain below:
    //   CARGO_MANIFEST_DIR is guaranteed-set by the cargo test harness; absence
    //   means cargo is broken or the test is being run via a non-cargo runner,
    //   neither of which we support. The two .parent() chain is an invariant of
    //   the kremory workspace layout (crates/kremory-napi → crates → repo root);
    //   any breakage indicates a workspace restructure that this test must catch
    //   loudly, not paper over with default fallbacks.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo test runner");
    PathBuf::from(manifest_dir)
        .parent()
        .expect("kremory-napi has a parent crates/ directory")
        .parent()
        .expect("crates/ has a parent repo root")
        .to_path_buf()
}

fn parse_file_to_string(path: &std::path::Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Check whether an item has an attribute matching the given name.
fn has_attr(attrs: &[syn::Attribute], attr_name: &str) -> bool {
    attrs.iter().any(|a| {
        a.path()
            .segments
            .last()
            .is_some_and(|s| s.ident == attr_name)
    })
}

/// Check whether an item is gated by `#[cfg(test)]` specifically.
///
/// PAR-G2 (V1-CANONICAL §4.2): the walker previously called `has_attr(attrs, "cfg")`,
/// which skips **every** `#[cfg(...)]` method — so the gate was blind exactly on the
/// feature axis it most needed to police. Its own comment said it meant `#[cfg(test)]`;
/// the code said "any cfg". Proven live by `backfill_episode_embeddings`, which is
/// `#[cfg(feature = "content-search")]` and had neither a napi mirror nor a skip entry,
/// yet the gate stayed GREEN.
///
/// Only a literal `#[cfg(test)]` is a genuine test-only item. Feature gates
/// (`#[cfg(feature = "...")]`) are part of the consumer surface and MUST be walked.
fn is_cfg_test_gated(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if a.path().segments.last().is_none_or(|s| s.ident != "cfg") {
            return false;
        }
        // `#[cfg(test)]` parses as a single path token `test`; anything else
        // (`feature = "x"`, `all(..)`, `not(..)`) is a real conditional-compilation
        // gate on shipped API and must not be skipped.
        a.parse_args::<syn::Path>()
            .is_ok_and(|p| p.is_ident("test"))
    })
}

/// Check whether `#[napi(object)]` is present (napi attribute with `object` arg).
fn has_napi_object(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        let is_napi = a.path().segments.last().is_some_and(|s| s.ident == "napi");
        if !is_napi {
            return false;
        }
        // Check tokens contain "object"
        use quote::ToTokens;
        a.to_token_stream().to_string().contains("object")
    })
}

// ── Substrate extraction ──────────────────────────────────────────────────────

/// Tracked type names and the impl-type prefix used in skip-list entries.
fn tracked_impl_types() -> &'static [(&'static str, &'static str)] {
    &[
        ("Memory", "Memory"),
        ("MemoryBuilder", "MemoryBuilder"),
        ("RememberRequest", "RememberRequest"),
        ("RememberBatchBuilder", "RememberBatchBuilder"),
        ("EpisodeEntryBuilder", "EpisodeEntryBuilder"),
        ("RecallRequest", "RecallRequest"),
        ("RecallTemplate", "RecallTemplate"),
        ("ForgetRequest", "ForgetRequest"),
        ("DreamRequest", "DreamRequest"),
    ]
}

/// Struct types whose public *fields* are tracked (not methods).
fn tracked_struct_types() -> &'static [(&'static str, &'static str)] {
    &[("DreamSummary", "DreamSummary")]
}

/// Extract `TypePrefix::method_name` for all `pub fn` in tracked impl blocks,
/// and `StructType::field_name` for tracked struct public fields.
/// Takes source strings to avoid holding `syn::File` (which is `!Send`).
/// Accepts multiple sources so callers can pass both `facade/mod.rs` and
/// `facade/builder.rs` after the TD-015 MemoryBuilder extraction.
fn extract_substrate_symbols(sources: &[&str]) -> Vec<String> {
    let mut symbols: Vec<String> = Vec::new();

    let tracked_impls: HashMap<&str, &str> = tracked_impl_types().iter().copied().collect();
    let tracked_structs: HashMap<&str, &str> = tracked_struct_types().iter().copied().collect();

    for src in sources {
        let file = syn::parse_file(src).expect("failed to parse kremory facade source file");

        for item in &file.items {
            match item {
                Item::Impl(impl_block) => {
                    // Determine the self-type name. Handles plain types and generic types like
                    // `MemoryBuilder<L, E>`.
                    let type_name_owned: Option<String> = match impl_block.self_ty.as_ref() {
                        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
                        _ => None,
                    };

                    let Some(ref type_name) = type_name_owned else {
                        continue;
                    };

                    let Some(&prefix) = tracked_impls.get(type_name.as_str()) else {
                        continue;
                    };

                    for impl_item in &impl_block.items {
                        let ImplItem::Fn(method) = impl_item else {
                            continue;
                        };
                        // Only `pub` visibility; skip private helpers and `pub(crate)`.
                        if !matches!(method.vis, syn::Visibility::Public(_)) {
                            continue;
                        }
                        // Skip genuinely test-only items (`#[cfg(test)]`) and deprecated
                        // ones. PAR-G2: this used to skip EVERY `#[cfg(...)]`, which made
                        // the gate blind on the feature axis — see `is_cfg_test_gated`.
                        if is_cfg_test_gated(&method.attrs)
                            || has_attr(&method.attrs, "deprecated")
                        {
                            continue;
                        }
                        let fn_name = method.sig.ident.to_string();
                        symbols.push(format!("{prefix}::{fn_name}"));
                    }
                }

                Item::Struct(s) => {
                    let struct_name = s.ident.to_string();
                    let Some(&prefix) = tracked_structs.get(struct_name.as_str()) else {
                        continue;
                    };
                    if let syn::Fields::Named(fields) = &s.fields {
                        for field in &fields.named {
                            if !matches!(field.vis, syn::Visibility::Public(_)) {
                                continue;
                            }
                            if let Some(ident) = &field.ident {
                                symbols.push(format!("{prefix}::{ident}"));
                            }
                        }
                    }
                }

                _ => {}
            }
        }
    } // end for src in sources

    symbols
}

// ── Napi surface extraction ───────────────────────────────────────────────────

/// Collect napi method names from `#[napi]` impl blocks and field names from
/// `#[napi(object)]` structs. Takes source strings to avoid holding `syn::File`.
fn extract_napi_symbols(sources: &[&str]) -> HashSet<String> {
    let mut all: HashSet<String> = HashSet::new();

    for src in sources {
        let file = syn::parse_file(src).expect("failed to parse kremory-napi source file");

        for item in &file.items {
            match item {
                Item::Impl(impl_block) => {
                    if !has_attr(&impl_block.attrs, "napi") {
                        continue;
                    }
                    for impl_item in &impl_block.items {
                        let ImplItem::Fn(method) = impl_item else {
                            continue;
                        };
                        if !matches!(method.vis, syn::Visibility::Public(_)) {
                            continue;
                        }
                        all.insert(method.sig.ident.to_string());
                    }
                }

                Item::Struct(s) => {
                    if !has_napi_object(&s.attrs) {
                        continue;
                    }
                    if let syn::Fields::Named(named) = &s.fields {
                        for field in &named.named {
                            if !matches!(field.vis, syn::Visibility::Public(_)) {
                                continue;
                            }
                            if let Some(ident) = &field.ident {
                                all.insert(ident.to_string());
                            }
                        }
                    }
                }

                _ => {}
            }
        }
    }

    all
}

// ── Name-matching (substrate snake_case → napi camelCase) ────────────────────

/// Convert a substrate `snake_case` name to `camelCase` for napi surface matching.
fn snake_to_camel(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut upper_next = false;
    for ch in s.chars() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next {
            result.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

/// Check whether a substrate method/field name has a plausible napi mirror.
///
/// Matching rules (in priority order):
/// 1. Exact match (snake_case method → same name)
/// 2. camelCase conversion match
/// 3. Suffix-based match for abbreviated napi names (e.g., `update_episode_metadata` →
///    napi `updateMetadata` matches the `metadata` suffix words)
fn has_napi_mirror(substrate_fn: &str, napi_symbols: &HashSet<String>) -> bool {
    if napi_symbols.contains(substrate_fn) {
        return true;
    }
    let camel = snake_to_camel(substrate_fn);
    if napi_symbols.contains(&camel) {
        return true;
    }
    // Suffix-based: check camelCase of any 1..N-word suffix of the snake words.
    let words: Vec<&str> = substrate_fn.split('_').collect();
    for start in 1..words.len() {
        let suffix_camel = snake_to_camel(&words[start..].join("_"));
        if napi_symbols.contains(&suffix_camel) {
            return true;
        }
    }
    false
}

// ── Main test ─────────────────────────────────────────────────────────────────

#[test]
fn napi_surface_matches_substrate_or_skip_list() {
    let root = repo_root();

    // Read sources as strings immediately — do NOT hold syn::File values
    // (syn::File is !Send and would cause issues in the threaded test harness).
    // facade/mod.rs + facade/builder.rs both scanned: MemoryBuilder was extracted
    // to builder.rs in TD-015; both files carry tracked impl blocks.
    // PAR-G1 (V1-CANONICAL §4.2): the walker read only `mod.rs` + `builder.rs`, but the
    // 10 tracked types are defined across SEVEN files — `Memory` impls also live in
    // `update.rs`; `RememberRequest`/`RememberBatchBuilder`/`EpisodeEntryBuilder` in
    // `remember.rs`; `RecallRequest` in `recall.rs`; `ForgetRequest` in `forget.rs`;
    // `DreamRequest` in `dream.rs`. Five of seven were never parsed, so most of the
    // tracked request-builder surface had never been parity-checked at all while the
    // gate reported GREEN.
    let facade_mod_src = parse_file_to_string(&root.join("crates/kremory/src/facade/mod.rs"));
    let facade_builder_src =
        parse_file_to_string(&root.join("crates/kremory/src/facade/builder.rs"));
    let facade_update_src = parse_file_to_string(&root.join("crates/kremory/src/facade/update.rs"));
    let facade_remember_src =
        parse_file_to_string(&root.join("crates/kremory/src/facade/remember.rs"));
    let facade_recall_src = parse_file_to_string(&root.join("crates/kremory/src/facade/recall.rs"));
    let facade_forget_src = parse_file_to_string(&root.join("crates/kremory/src/facade/forget.rs"));
    let facade_dream_src = parse_file_to_string(&root.join("crates/kremory/src/facade/dream.rs"));
    let napi_lib_src = parse_file_to_string(&root.join("crates/kremory-napi/src/lib.rs"));
    let napi_convert_src = parse_file_to_string(&root.join("crates/kremory-napi/src/convert.rs"));
    let skip_list_path = root.join("crates/kremory-napi/parity-skip.toml");

    let skip_list = load_skip_list(&skip_list_path);
    let napi_symbols = extract_napi_symbols(&[&napi_lib_src, &napi_convert_src]);
    let substrate_symbols = extract_substrate_symbols(&[
        &facade_mod_src,
        &facade_builder_src,
        &facade_update_src,
        &facade_remember_src,
        &facade_recall_src,
        &facade_forget_src,
        &facade_dream_src,
    ]);

    // Governance check: no duplicate skip-list entries.
    {
        let mut seen: HashSet<&str> = HashSet::new();
        for key in skip_list.keys() {
            assert!(
                seen.insert(key.as_str()),
                "parity-skip.toml has duplicate entry for symbol '{key}'. \
                 Each symbol must appear at most once."
            );
        }
    }

    // Enforce: skip-list count must not exceed 88 (sanity cap — over-finding guard).
    // ⚠️ THIS NOTE HAS NOW ROTTED TWICE — corrected again 2026-08-12.
    //   - It read "83" until 2026-08-06 while the assertion said 86.
    //   - It was then rewritten to say "matches the `skip_count <= 86` assert" —
    //     but the assert was later raised to 88 (line ~484) and this line was not,
    //     so the comment written to FIX a doc-rot instance became one itself.
    // The enforced value is the assert, and only the assert: `skip_count <= 88`.
    // The lead line above and this note are now both 88. If you raise the assert
    // again, grep this file for the OLD number before you finish — twice now the
    // raise was made and the prose left behind.
    // ADR-031 acceptance gate 2: if this grows large it means the walker is too
    // aggressive or the skip list is being used as an escape hatch.
    // Lowered 100 → 90 by TD-053 (2026-06-25) after pruning 13 doc-rot entries, then
    // 90 → 80 after the boy-scout pass pruned 14 more redundant "already mirrored"
    // entries (forget/dream/RecallRequest::*/DreamSummary::*). Raised 80 → 81
    // (reranker latency spike, 2026-07-28) for exactly one new legitimate entry —
    // `MemoryBuilder::with_rerank_candidate_max_chars` — which follows the SAME
    // Tier-2-builder-deferred pattern as its 7 immediate siblings
    // (with_content_stream_weight / with_rrf_k / with_episode_dense_enabled /
    // with_fact_dense_enabled / with_embed_task_prefix_enabled /
    // with_proximity_weight), all already skip-listed for the identical ADR-030
    // Form B reason. The register was already sitting exactly at the prior cap
    // (80) before this addition — not a symptom of walker over-finding. Raised
    // 81 → 83 (TD-112, 2026-07-28) for two new legitimate entries —
    // `Memory::reembed_all_entity_embeddings` / `Memory::reembed_all_fact_embeddings`
    // — which follow the SAME one-shot-CLI-maintenance-entrypoint pattern as
    // their already-skipped sibling `Memory::reembed_all_episode_embeddings`.
    // Raised 83 → 84 (DOC-1, 2026-08-03) for exactly one entry —
    // `MemoryBuilder::with_provider_rates_path` — which is the SAME Tier-2
    // builder-knob pattern as its already-skipped siblings
    // (with_content_stream_weight / with_rrf_k / with_temporal_weight): the whole
    // builder surface is deferred to JS together per ADR-030 Form B, since
    // `Memory.open` is the single JS construction entry point. Raised deliberately
    // and on the record rather than by reshaping what the guard measures — the
    // register was already sitting exactly at the prior cap (83) before this
    // addition, so this is register growth, not walker over-finding.
    // Raised 84 → 86 (PAR-G1/PAR-G2, 2026-08-03) for exactly the two symbols the
    // repaired walker surfaced on its first run: `Memory::group_id_for_test` (test-only,
    // same gate as the already-skipped `temporal_graph_for_test`) and
    // `Memory::backfill_episode_embeddings` (one-shot CLI maintenance, same shape as its
    // already-skipped sibling `reembed_all_episode_embeddings`). Both are register growth
    // from FIXING the guard, not walker over-finding — the walker had been reading 2 of
    // the 7 files holding tracked types and skipping every `#[cfg(...)]` method, so this
    // is the first time either symbol was ever examined. Measured before the fix and
    // confirmed after: 2 new entries, not the ~59 initially estimated.
    // Raised 86 → 88 (TD-172, 2026-08-12) for exactly TWO entries —
    // `MemoryBuilder::with_contradiction_detection_enabled` — the NINTH and last
    // missing per-knob override setter, deferred for the identical ADR-030 Form B
    // reason as its eight already-skipped siblings. The register was sitting exactly
    // at the prior cap (86) before this addition, so this is register growth, not
    // walker over-finding — and it is raised on the record rather than by pruning a
    // legitimate entry to make room, which is how a ratchet quietly becomes decoration.
    let skip_count = skip_list.len();
    assert!(
        skip_count <= 88,
        "parity-skip.toml has {skip_count} entries which exceeds the sanity cap of 88. \
         This indicates the parity walker is over-finding substrate symbols. \
         Refine tracked_impl_types() / tracked_struct_types() scope rather than \
         inflating the skip list."
    );

    // Main drift check.
    let mut drift_detected: Vec<String> = Vec::new();

    for symbol in &substrate_symbols {
        let fn_name = symbol.split("::").nth(1).unwrap_or(symbol.as_str());
        let in_skip = skip_list.contains_key(symbol.as_str());
        let has_mirror = has_napi_mirror(fn_name, &napi_symbols);

        if !in_skip && !has_mirror {
            drift_detected.push(symbol.clone());
        }
    }

    if !drift_detected.is_empty() {
        let mut msg = String::from(
            "ADR-031 Layer 3: napi surface drift detected.\n\
             The following kremory substrate symbols have no napi mirror and no skip-list entry.\n\
             For each symbol, EITHER add a matching napi method to kremory-napi/src/lib.rs \
             OR add a skip entry to kremory-napi/parity-skip.toml with a `reason` and `revisit`.\n\n",
        );
        for sym in &drift_detected {
            let fn_name = sym.split("::").nth(1).unwrap_or(sym.as_str());
            let camel = snake_to_camel(fn_name);
            msg.push_str(&format!(
                "  DRIFT: '{sym}' — no napi mirror for '{fn_name}' \
                 (expected camelCase: '{camel}') \
                 and no skip-list entry in parity-skip.toml\n"
            ));
        }
        panic!("{msg}");
    }

    // Non-fatal: warn on orphaned skip-list entries (substrate symbol no longer exists).
    let substrate_set: HashSet<&str> = substrate_symbols.iter().map(|s| s.as_str()).collect();
    for key in skip_list.keys() {
        if !substrate_set.contains(key.as_str()) {
            eprintln!(
                "parity-skip.toml orphan: '{key}' is in the skip list but was not found \
                 in the substrate facade. Consider removing the entry."
            );
        }
    }

    // Emit stats for CI visibility.
    eprintln!(
        "api_parity: {} substrate symbols checked, {} skip-list entries, {} napi surface symbols",
        substrate_symbols.len(),
        skip_count,
        napi_symbols.len()
    );
}

// ── Unit tests for helpers ────────────────────────────────────────────────────

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn snake_to_camel_cases() {
        assert_eq!(snake_to_camel("recall_by_source_id"), "recallBySourceId");
        assert_eq!(
            snake_to_camel("update_episode_metadata"),
            "updateEpisodeMetadata"
        );
        assert_eq!(snake_to_camel("close"), "close");
        assert_eq!(snake_to_camel("open"), "open");
        assert_eq!(snake_to_camel("in_namespace"), "inNamespace");
    }

    #[test]
    fn has_napi_mirror_exact_match() {
        let mut napi = HashSet::new();
        napi.insert("close".to_string());
        assert!(has_napi_mirror("close", &napi));
        assert!(!has_napi_mirror("open", &napi));
    }

    #[test]
    fn has_napi_mirror_camel_match() {
        let mut napi = HashSet::new();
        napi.insert("recallBySourceId".to_string());
        assert!(has_napi_mirror("recall_by_source_id", &napi));
    }

    #[test]
    fn has_napi_mirror_suffix_match() {
        let mut napi = HashSet::new();
        // Suffix of `recall_by_source_id` words[2..] = "source_id" → camel "sourceId"
        napi.insert("sourceId".to_string());
        assert!(has_napi_mirror("recall_by_source_id", &napi));
        // Words[3..] = "id" also matches "id" directly if napi had "id" — but here we
        // verify the two-word suffix "source_id" → "sourceId" is the primary signal.
        assert!(!has_napi_mirror("update_episode_metadata", &napi));
    }

    #[test]
    fn skip_list_parser_handles_standard_entry() {
        let toml = r#"
[[skip]]
symbol = "Memory::open"
reason = "Tier-2 builder deferred"
revisit = "v0.2.0"

[[skip]]
symbol = "Memory::auto"
reason = "Auto constructor"
revisit = "never"
"#;
        // Write to a temp file.
        let dir = std::env::temp_dir();
        let path = dir.join("api_parity_test_skip.toml");
        std::fs::write(&path, toml).expect("write temp skip toml");
        let map = load_skip_list(&path);
        assert_eq!(
            map.get("Memory::open").map(|s| s.as_str()),
            Some("Tier-2 builder deferred")
        );
        assert_eq!(
            map.get("Memory::auto").map(|s| s.as_str()),
            Some("Auto constructor")
        );
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn extract_toml_string_value_basic() {
        assert_eq!(
            extract_toml_string_value(r#"symbol = "Memory::open""#, "symbol"),
            Some("Memory::open".to_string())
        );
        assert_eq!(
            extract_toml_string_value(r#"reason = "some reason""#, "reason"),
            Some("some reason".to_string())
        );
        assert_eq!(extract_toml_string_value("other = \"x\"", "symbol"), None);
    }
}
