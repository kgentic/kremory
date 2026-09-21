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
//! Napi mirror surface extracted from `lib.rs` + `convert/{ingest,recall,dream,mutations}.rs`:
//! - `#[napi]` methods on `JsMemory` impl
//! - `#[napi(object)]` struct fields on option/result structs

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use syn::{ImplItem, Item};

// ── Skip-list parsing ─────────────────────────────────────────────────────────

/// One `[[skip]]` entry: a substrate symbol with no Node equivalent.
#[derive(serde::Deserialize)]
struct SkipEntry {
    /// Substrate symbol being skipped, e.g. `MemoryBuilder::with_temporal_weight`.
    symbol: String,
    /// Why it has no Node equivalent. Surfaced in the orphan report, and REQUIRED:
    /// a skip with no stated reason is indistinguishable from an oversight.
    reason: String,
    /// When the deferral is due to be revisited. REQUIRED: a skip with no revisit
    /// is a permanent gap wearing a temporary label.
    revisit: String,
}

#[derive(serde::Deserialize)]
struct SkipFile {
    skip: Vec<SkipEntry>,
}

/// Parse `parity-skip.toml` into a map of `symbol → reason`.
///
/// ⚠️ **This used to hand-parse the file line-by-line, and that is why the file was
/// MALFORMED and nobody knew.** One entry carried two `revisit` keys — which real
/// TOML rejects outright — while the hand parser skipped `revisit` entirely and so
/// could not see it. A second entry had silently lost its own `revisit` value to
/// the same drift. The `toml` dependency needed to catch this had been declared in
/// `Cargo.toml` the whole time and never used; the only mention of it in this file
/// was a comment explaining why it was not being used.
///
/// The original comment claimed the `toml` crate had `!Send` problems in the test
/// harness via `toml::value::Table`'s internal `Arc`. That applies to the dynamic
/// `toml::Value` API; deserializing straight into owned structs, as below, never
/// constructs one.
fn load_skip_list(toml_path: &std::path::Path) -> HashMap<String, String> {
    let raw = std::fs::read_to_string(toml_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", toml_path.display()));

    let parsed: SkipFile = toml::from_str(&raw).unwrap_or_else(|e| {
        panic!(
            "parity-skip.toml is not valid TOML: {e}\n\
             A malformed skip-list silently changes which symbols the parity gate \
             believes are deliberately absent, so this is a hard failure rather than \
             a best-effort parse."
        )
    });

    let mut result: HashMap<String, String> = HashMap::with_capacity(parsed.skip.len());
    for entry in parsed.skip {
        assert!(
            !entry.reason.trim().is_empty(),
            "parity-skip.toml entry '{}' has an empty `reason`. A skip with no stated \
             reason cannot be told apart from an oversight.",
            entry.symbol
        );
        assert!(
            !entry.revisit.trim().is_empty(),
            "parity-skip.toml entry '{}' has an empty `revisit`. Every deferral needs \
             a date or a named sweep, or it is permanent by default.",
            entry.symbol
        );
        // Duplicate detection has to happen HERE, at insert time. The governance
        // check that used to live further down iterated the finished HashMap's keys,
        // which cannot contain duplicates by construction — a green control that
        // could never fire.
        if let Some(previous) = result.insert(entry.symbol.clone(), entry.reason) {
            panic!(
                "parity-skip.toml has a duplicate entry for symbol '{}'. Each symbol \
                 must appear at most once; the later entry silently replaced:\n  {previous}",
                entry.symbol
            );
        }
    }

    result
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
                        if is_cfg_test_gated(&method.attrs) || has_attr(&method.attrs, "deprecated")
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
    // `convert.rs` was split into `convert/{ingest,recall,dream,mutations}.rs` (TD-243,
    // file-size ratchet — the flat file was WATCHED at 1987 lines). `convert/mod.rs` only
    // declares `mod`/`pub use` (no `#[napi]` items) and `convert/tests.rs` is cfg(test)-only,
    // so neither carries FFI surface; the four domain files are the full replacement set.
    let napi_convert_ingest_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/convert/ingest.rs"));
    let napi_convert_recall_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/convert/recall.rs"));
    let napi_convert_dream_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/convert/dream.rs"));
    let napi_convert_mutations_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/convert/mutations.rs"));
    // `lib.rs`'s own `impl JsMemory` (38 `#[napi]` methods, no test module) was split the
    // same way into `memory/{lifecycle,ingest,dream,recall,mutations,admin}.rs` — each file
    // carries its own `#[napi] impl JsMemory { ... }` block (multiple inherent impl blocks
    // for one type, across files, is normal Rust; napi-rs registers annotated items
    // independently, not per-file). `memory/mod.rs` only declares the `JsMemory` struct +
    // `mod`s (no `#[napi]` impl), so it is NOT added here — mirrors why `convert/mod.rs`
    // above is skipped too. `napi_lib_src` above is kept for symmetry even though `lib.rs`
    // no longer contains any `#[napi]` item post-split; it costs nothing to leave wired.
    let napi_memory_lifecycle_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/memory/lifecycle.rs"));
    let napi_memory_ingest_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/memory/ingest.rs"));
    let napi_memory_dream_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/memory/dream.rs"));
    let napi_memory_recall_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/memory/recall.rs"));
    let napi_memory_mutations_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/memory/mutations.rs"));
    let napi_memory_admin_src =
        parse_file_to_string(&root.join("crates/kremory-napi/src/memory/admin.rs"));
    let skip_list_path = root.join("crates/kremory-napi/parity-skip.toml");

    let skip_list = load_skip_list(&skip_list_path);
    let napi_symbols = extract_napi_symbols(&[
        &napi_lib_src,
        &napi_convert_ingest_src,
        &napi_convert_recall_src,
        &napi_convert_dream_src,
        &napi_convert_mutations_src,
        &napi_memory_lifecycle_src,
        &napi_memory_ingest_src,
        &napi_memory_dream_src,
        &napi_memory_recall_src,
        &napi_memory_mutations_src,
        &napi_memory_admin_src,
    ]);
    let substrate_symbols = extract_substrate_symbols(&[
        &facade_mod_src,
        &facade_builder_src,
        &facade_update_src,
        &facade_remember_src,
        &facade_recall_src,
        &facade_forget_src,
        &facade_dream_src,
    ]);

    // Duplicate skip-list entries are rejected inside `load_skip_list`, at insert
    // time. A check here would iterate the finished map's keys and could never fire.

    // Enforce: skip-list count must not exceed 93 (sanity cap — over-finding guard).
    // ⚠️ THIS NOTE ROTTED TWICE before 2026-08-12 (see the history below for the
    // exact sequence). Kept in sync again on 2026-09-02 (88 → 89, TD-231).
    // The enforced value is the assert, and only the assert: `skip_count <= 93`.
    // The lead line above and this note are now both 93. If you raise the assert
    // again, grep this file for the OLD number before you finish — this note has
    // already rotted twice from someone skipping that step.
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
    // Raised 88 → 89 (TD-231, 2026-09-02) for exactly ONE entry —
    // `MemoryBuilder::extraction_arm_budget_ms` — the TENTH per-knob override
    // setter (a new Rust-side builder method added the same day, closing a
    // gap where local-model consumers had no reachable path to raise this
    // timeout at all). Same ADR-030 Form B deferral as its nine siblings.
    // The register was sitting exactly at the prior cap (88) before this
    // addition, so this is register growth, not walker over-finding.
    // Raised 89 -> 90 (ADR-080, 2026-09-06) for exactly ONE entry —
    // `MemoryBuilder::prior_turn_replay_depth` — the ELEVENTH per-knob override
    // setter, deferred to JS under the identical ADR-030 Form B reason as its
    // ten siblings. The register was sitting exactly at the prior cap (89)
    // before this addition, so this is register growth, not walker
    // over-finding; raised on the record rather than by pruning a legitimate
    // entry to make room.
    // Raised 90 -> 91 (public-docs-and-api-surface-audit Phase 2, F17,
    // 2026-09-07) for exactly ONE entry — `MemoryBuilder::with_graph_degree_weight`
    // — the TWELFTH per-knob `SearchConfig` override setter, added to close the
    // one axis (`graph_degree_weight`) that had no public setter at all (the
    // mirror image of TD-231, which found the missing knob for
    // `extraction_arm_budget_ms`). Deferred to JS under the identical ADR-030
    // Form B reason as its eleven siblings. The register was sitting exactly at
    // the prior cap (90) before this addition, so this is register growth, not
    // walker over-finding; raised on the record rather than by pruning a
    // legitimate entry to make room.
    // Raised 92 -> 93 (TD-061 secret-scan, 2026-09-21): with_secret_scan_enabled
    // and with_secret_scan_mode are per-knob MemoryBuilder setters, deferred under
    // ADR-030 Form B exactly as their thirteen siblings are. The gate caught them
    // because the feature shipped the methods without a mirror OR a skip entry.
    // Raised 91 -> 92 (public-docs-and-api-surface-audit quality-review B2,
    // 2026-09-07) for exactly ONE entry — `DreamRequest::execute` — a new
    // method added because `dream()` now requires an explicit `.execute()`
    // terminal (matching `forget()`/`undo()`/etc.'s existing destructive-op
    // convention, per the quality-review B2 finding). Same shape as the
    // already-skipped `ForgetRequest::execute` sibling immediately above in
    // parity-skip.toml — an implementation detail the napi binding calls
    // internally, not a JS-visible gap. The register was sitting exactly at
    // the prior cap (91) before this addition, so this is register growth,
    // not walker over-finding. NOTE TO THE NEXT PERSON: this file's lead
    // comment and the assert must BOTH say 93 now — grep for the old number
    // before you finish, because this note has already rotted twice.
    let skip_count = skip_list.len();
    assert!(
        skip_count <= 93,
        "parity-skip.toml has {skip_count} entries which exceeds the sanity cap of 93. \
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

    /// Each parser test writes its OWN file. A shared fixed filename in the temp
    /// dir races sibling tests under nextest's parallel pool, and now that the
    /// parser PANICS on malformed input, a race would fail an unrelated test.
    fn write_skip_fixture(name: &str, body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("api_parity_test_{name}.toml"));
        std::fs::write(&path, body).expect("write temp skip toml");
        path
    }

    #[test]
    #[should_panic(expected = "not valid TOML")]
    fn malformed_toml_is_rejected_rather_than_best_effort_parsed() {
        // A duplicate key — the EXACT malformation that sat undetected in the real
        // file, because the previous hand-rolled parser never looked at `revisit`.
        let path = write_skip_fixture(
            "malformed",
            r#"
[[skip]]
symbol = "Memory::open"
reason = "Tier-2 builder deferred"
revisit = "v0.2.0"
revisit = "stray"
"#,
        );
        let _ = load_skip_list(&path);
    }

    #[test]
    #[should_panic(expected = "duplicate entry for symbol")]
    fn a_duplicated_symbol_is_rejected_instead_of_silently_collapsing() {
        // Two entries, one symbol. The previous check iterated the finished
        // HashMap's keys, so this collapsed to ONE entry and quietly CREATED
        // headroom under the cap.
        let path = write_skip_fixture(
            "duplicate",
            r#"
[[skip]]
symbol = "Memory::open"
reason = "first"
revisit = "v0.2.0"

[[skip]]
symbol = "Memory::open"
reason = "second"
revisit = "v0.2.0"
"#,
        );
        let _ = load_skip_list(&path);
    }

    #[test]
    #[should_panic(expected = "empty `revisit`")]
    fn a_deferral_with_no_revisit_is_rejected() {
        let path = write_skip_fixture(
            "no_revisit",
            r#"
[[skip]]
symbol = "Memory::open"
reason = "Tier-2 builder deferred"
revisit = "   "
"#,
        );
        let _ = load_skip_list(&path);
    }

    #[test]
    #[should_panic(expected = "empty `reason`")]
    fn a_skip_with_no_reason_is_rejected() {
        let path = write_skip_fixture(
            "no_reason",
            r#"
[[skip]]
symbol = "Memory::open"
reason = ""
revisit = "v0.2.0"
"#,
        );
        let _ = load_skip_list(&path);
    }

    #[test]
    fn the_real_skip_list_parses_and_every_entry_is_complete() {
        // The shipped file itself, through the real loader — so a malformation
        // committed tomorrow fails HERE with a parser message, not later with a
        // confusing parity mismatch.
        let map = load_skip_list(&repo_root().join("crates/kremory-napi/parity-skip.toml"));
        assert!(
            map.len() > 50,
            "the real skip list should be substantial; got {} entries — a near-empty \
             map means the loader silently dropped entries",
            map.len()
        );
    }
}
