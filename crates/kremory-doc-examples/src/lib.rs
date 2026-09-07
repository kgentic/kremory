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
//! RULE-002's `>= 60` extracted-block guard (in `generate_docs.py`) is the
//! safety net if this list ever silently drifts out of sync with the real
//! doc tree.

#![allow(unused)]

#[doc = include_str!("../generated/README.md.generated.md")]
mod readme_md {}

#[doc = include_str!("../generated/docs__api.md.generated.md")]
mod docs_api_md {}

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
