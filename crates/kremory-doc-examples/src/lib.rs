//! Phase 0 doc-compile audit harness.
//!
//! `docs/specs/public-docs-and-api-surface-audit/` (spec RULE-001, RULE-002;
//! plan.md "Phase 0"; tasks.md T0.1-T0.4). NOT part of kremory's public API —
//! this whole crate is `publish = false` and exists only to compile every
//! fenced ```rust block in `docs/*.md` + `README.md` against the real
//! `kremory` crate.
//!
//! ## How to run
//!
//! ```text
//! python3 crates/kremory-doc-examples/generate_docs.py   # regenerate ./generated/*.md
//! cargo test --doc -p kremory-doc-examples                # compile them all
//! ```
//!
//! Each `mod` below corresponds to one source doc file. The generator
//! (`generate_docs.py`) groups that file's fenced blocks into per-section
//! compilation units and writes ONE generated markdown file containing one
//! fenced block per unit; `#[doc = include_str!(...)]` on a dummy module is
//! how that generated markdown gets fed to rustdoc's doctest runner --
//! confirmed by a real spike (T0.1, 2026-09-07) against a 2-block fragment
//! lifted from `docs/api.md`'s own Quickstart section, BEFORE this file was
//! written: `kremory::` imports resolved cleanly, and `no_run` /
//! `compile_fail` markers behaved exactly per rustdoc's documented semantics.
//! See README.md in this directory for the full design writeup + the
//! placeholder-prelude rationale.
//!
//! Adding a new doc file: add it to `DOC_FILES` in `generate_docs.py`, add a
//! matching `mod` below pointing at the new `generated/<name>.generated.md`.
//! `check_doc_files_complete()` in `generate_docs.py` asserts this module list
//! against `DOC_FILES` and `DOC_FILES` against the real doc tree, so a file
//! missing from either place fails loudly instead of silently dropping out of
//! the compile set. RULE-002's `>= 60` block floor is a separate, weaker check
//! — it catches a broken extractor, not a single dropped file.

#![allow(unused)]

#[doc = include_str!("../generated/README.md.generated.md")]
mod readme_md {}

#[doc = include_str!("../generated/docs__api__index.md.generated.md")]
mod docs_api_index_md {}

#[doc = include_str!("../generated/docs__api__setup.md.generated.md")]
mod docs_api_setup_md {}

#[doc = include_str!("../generated/docs__api__namespaces.md.generated.md")]
mod docs_api_namespaces_md {}

#[doc = include_str!("../generated/docs__api__ingest.md.generated.md")]
mod docs_api_ingest_md {}

#[doc = include_str!("../generated/docs__api__recall.md.generated.md")]
mod docs_api_recall_md {}

#[doc = include_str!("../generated/docs__api__bi-temporal.md.generated.md")]
mod docs_api_bi_temporal_md {}

#[doc = include_str!("../generated/docs__api__dream.md.generated.md")]
mod docs_api_dream_md {}

#[doc = include_str!("../generated/docs__api__reversibility.md.generated.md")]
mod docs_api_reversibility_md {}

#[doc = include_str!("../generated/docs__api__async-and-events.md.generated.md")]
mod docs_api_async_and_events_md {}

#[doc = include_str!("../generated/docs__api__advanced.md.generated.md")]
mod docs_api_advanced_md {}

#[doc = include_str!("../generated/docs__api__feature-flags.md.generated.md")]
mod docs_api_feature_flags_md {}

#[doc = include_str!("../generated/docs__api__node-binding.md.generated.md")]
mod docs_api_node_binding_md {}

#[doc = include_str!("../generated/docs__releases__upgrade-guide.md.generated.md")]
mod docs_releases_upgrade_guide_md {}

#[doc = include_str!("../generated/docs__getting-started.md.generated.md")]
mod docs_getting_started_md {}

#[doc = include_str!("../generated/docs__benchmarks.md.generated.md")]
mod docs_benchmarks_md {}

#[doc = include_str!("../generated/docs__comparison.md.generated.md")]
mod docs_comparison_md {}

#[doc = include_str!("../generated/docs__error-handling-policy.md.generated.md")]
mod docs_error_handling_policy_md {}

#[doc = include_str!("../generated/docs__eval-fixtures.md.generated.md")]
mod docs_eval_fixtures_md {}

#[doc = include_str!("../generated/docs__eval.md.generated.md")]
mod docs_eval_md {}

#[doc = include_str!("../generated/docs__observability.md.generated.md")]
mod docs_observability_md {}

#[doc = include_str!("../generated/docs__testing.md.generated.md")]
mod docs_testing_md {}
