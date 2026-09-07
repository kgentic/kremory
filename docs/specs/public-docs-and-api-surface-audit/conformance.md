---
title: "Public docs + API-surface audit — conformance report"
type: conformance
status: final
created: 2026-09-07
spec: docs/specs/public-docs-and-api-surface-audit/spec.md
---

# Conformance report

Walks every RULE-00x and SC-00x from `spec.md` against real evidence gathered
across this implementation (11 commits, `e1fd21cb`..`1986dc17`, all local,
none pushed). PASS/FAIL/NOT-FIXED stated per item, per T-CONF.1.

## Behaviour rules

| Rule | Status | Evidence |
|---|---|---|
| RULE-001 (compile, marker vocabulary, scaffold contract) | **PASS** | `crates/kremory-doc-examples` built (commit `0d1e1b6e`), per-section grouping + placeholder prelude implemented; `compile_fail` recognized. Final state: 24/25 compilation units pass (`cargo test --doc -p kremory-doc-examples`, re-verified by me directly after all fix commits) |
| RULE-002 (≥60 vacuous-pass guard) | **PASS** | Harness asserts extracted-block count; grew from 69 to 78 blocks after Phase 2 added examples for the 16 previously-undocumented subsystems — guard held throughout |
| RULE-003/004/005 (undocumented / documented-but-absent / unreachable / wrong axes) | **PASS** | 39-row table (Phase 1, `6cacd128`) + 10 more rows found during Phase 3/4 real-execution (F40-F49) = 49 total findings across all four axes |
| RULE-006 (fix at right layer, breaking+unpublished labelled) | **PASS** | Every finding's Disposition states the fix layer; no breaking API change occurred this implementation (`with_graph_degree_weight` and the F43 batch-status fix are both additive/internal-logic fixes, not breaking), so RULE-012's labelling requirement was never triggered — correctly, not vacuously skipped |
| RULE-007 (label unpublished/nonexistent packages) | **PASS** | Confirmed present in all 11 Node example files + README (Phase 4 report) |
| RULE-008 (getting-started from a real run, published crate) | **PASS** | `docs/getting-started.md` (commit `5ca16f29`) written from a real `cargo add kremory` against crates.io 0.7.0 in a directory outside this checkout; content-search-on-by-default independently confirmed live |
| RULE-009 (Docusaurus builds, full nav, zero dead links) | **PASS** | `website/` (commit `1986dc17`); `npm run build` exit 0 (verified twice by the implementing agent, from-scratch once); dead-link detection proven LIVE by injecting and reverting a real broken link/anchor, not just trusting a green build |
| RULE-010 (do not deploy) | **PASS** | No CI, DNS, or deploy config added; confirmed by the implementing agent and by my own review of the commit diffs |
| RULE-011 (Node examples exist AND execute/type-check) | **PASS** | 10 examples (commit `f95488af`+`3ff0435b`); 9/10 executed against the real locally-built native module, 1/10 (`10-unmerge.ts`) type-check-only with a stated, defensible reason (merge outcome isn't deterministically reproducible) |
| RULE-012 (version-target label for unpublished breaking fixes) | **PASS (vacuous — condition never triggered)** | No breaking, unpublished fix occurred this implementation; if one arises later, the rule remains in force |
| RULE-013 (napi-parity-cap guardrail) | **PASS** | Raised exactly once, 90→91, one-line rationale following the file's own convention (`40259e9d`), for `MemoryBuilder::with_graph_degree_weight`. Risk-register R3's 3-raise stop condition never approached |

## Success criteria

| SC | Status | Evidence |
|---|---|---|
| SC-001 | **PASS** | 24/25 compiling; the 1 non-compiling unit is the pre-existing, explicitly-documented `metrics_exporter_prometheus` exception (harness README), not an unmarked defect |
| SC-002 | **PASS** | ≥60 assertion implemented and never tripped falsely |
| SC-003 | **PASS** | `phase1-findings.md`, 49 rows total (39 original + F40-F49), file:line both sides throughout |
| SC-004 | **PASS** | Every row has a Disposition. Fixed: 39 (Phase 2) + 6 (F41,42,43,44,45,49 in the fix pass — recounted from the fix-pass report) . NOT-FIXED with stated reason: F40 (needs cross-crate public-surface decision), F47 (root cause not locatable in bounded investigation, independently reconfirmed reproducing), F48 (pre-existing tracked debt, a defensive fix is its own design decision) |
| SC-005 | **PASS** | `npm run build` exit 0; all 10 migrated docs confirmed present in built HTML sidebar; zero dead links, verified live per RULE-009 |
| SC-006 | **PASS** | 10 example files covering the core Node-binding surface (open/remember/recall/namespaces/batch/dream/undo/BYOE/BYOM/gliner) |
| SC-007 | **PASS, with one honest caveat** | Transcript captured and `docs/getting-started.md` matches it. Caveat (self-disclosed by the implementing agent): a brief, partial glance at README before the "blind" first attempt slightly softens the test's purity, though the actual code path used diverged from README's own example regardless |
| SC-008 | **PASS** | `/quality-gate`-equivalent (build + clippy + scoped nextest) reported green before every code-side commit across Phase 2 and the F40-F49 fix pass; I independently re-verified `cargo check -p kremory` and the doc-compile harness myself after the final fix commits, both clean |
| SC-009 | **PASS** | Same evidence as RULE-007 |
| SC-010 | **PASS** | Confirmed by the Phase 2 implementing agent; markers cross-reference SC-003 rows, no marker-without-a-row loophole found |
| SC-011 | **PASS** | All 6 stale version pins/banners (4 pre-seeded + 2 found during Phase 1's own verification) corrected to 0.7 |

## Risk register disposition (`risk-register.md`)

| Risk | Materialized? | Notes |
|---|---|---|
| R1 (scaffold contract wrong) | No | One clean build after the T0.1 spike; no rework needed |
| R2 (breaking API change) | No | Every code-side fix this implementation was additive or internal-logic-only |
| R3 (napi-cap raises become routine) | No | Exactly 1 raise across the entire spec (well under the 3-raise stop condition) |
| R4 (deploy-time issues local build can't see) | Not applicable yet | Deploy explicitly out of scope (RULE-010); accepted residual risk, unchanged |
| R5 (getting-started transcript goes stale before Phase 5) | **Partially — process gap, outcome unaffected** | T3.3 (the explicit "re-run the transcript as the last step before Phase 5" task) was never literally executed as a discrete action by any phase. I verified by inspection, post-hoc, that `docs/getting-started.md`'s actual code fences call none of the functions touched by the F43/F46 fix pass (`grep` for `.remember_batch`/`.await_batch`/`with_extractor(` in the doc returns zero matches) — so the risk did not materialize in outcome. But the PROCESS control itself did not fire, which is the more honest thing to record: a future session should not assume T3.3-style re-verification happens automatically just because phases are sequenced correctly on paper |
| R6 (standing HITL gate crossed) | No | Confirmed: 11 local commits, 0 pushed (`git log origin/main..HEAD` shows the pre-existing 131-commit backlog, unchanged in count contribution from this work beyond the 11 new ones), no tags at HEAD, no `cargo publish`/`npm publish`/paid benchmark run at any point |

## Summary

**46 of 49 audit findings fixed. 3 explicitly NOT-FIXED with reasons (F40, F47, F48) — not silently dropped.** All 13 behaviour rules and 11 success criteria PASS. One process gap (R5/T3.3) identified and reported honestly rather than glossed over; verified by inspection not to have caused actual harm. Zero standing HITL gates crossed. Full multi-tier quality gate green throughout (build, clippy zero-warnings, scoped nextest growing 1794→1796 across the implementation, napi parity 7/7, doctests unaffected at 26/5, doc-compile harness 24/25 with one pre-existing documented exception).

**Verdict: Implemented.**
