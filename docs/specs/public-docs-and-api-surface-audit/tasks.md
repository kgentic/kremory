---
title: "Public docs + API-surface audit — tasks"
type: tasks
status: draft
created: 2026-09-07
spec: docs/specs/public-docs-and-api-surface-audit/spec.md
plan: docs/specs/public-docs-and-api-surface-audit/plan.md
---

# Tasks

Ordered; each task cites the RULE(s)/SC(s) it discharges. Phase 0→1→2 strictly
sequential (each is the prior's work list). Phase 3/4 parallel once Phase 2 is
stable. Phase 5 depends on Phase 2's doc-side commits.

## Phase 0 — Compile harness

- **T0.1** [RULE-001, RULE-002] Spike `#[doc = include_str!(...)]` +
  `cargo test --doc` against one real doc file (`docs/api.md`) with the
  per-section grouping + placeholder prelude from RULE-001. Confirm it
  actually resolves `kremory::` imports before committing to this wiring over
  the bare `rustdoc --test --extern` alternative.
- **T0.2** [RULE-001] Write the extraction script: walk `docs/*.md` +
  `README.md`, group fenced ` ```rust ` blocks by enclosing `##`/`###`
  section, detect existing markers (`no_run`, `compile_fail`, "illustrative",
  "pseudo-code" in surrounding prose), inject the documented prelude
  (`my_llm`, `my_embedder`, `MySink`, and any others discovered necessary
  during T0.1's spike).
- **T0.3** [RULE-002] Add the vacuous-pass assertion (≥ 60 extracted blocks) as
  a hard failure in the harness, not a warning.
- **T0.4** [SC-001, SC-002] Run the harness. Produce the `N of <total> compile,
  M fail` count with `file:line` per failure. This IS the Phase 1 work list —
  no separate step needed to produce it.
- **T0.5** [SC-010] For every failure resolved via marker rather than fix,
  confirm it also lands as an SC-003 row (Phase 1 will create SC-003; this
  task is a cross-check once Phase 1 exists, not blocking T0.4's completion).

## Phase 1 — Semantic surface audit

- **T1.1** [RULE-003, RULE-004, RULE-005] Enumerate the public API surface of
  `kremory` (via `cargo public-api` or a manual `pub` grep if that crate isn't
  already a dependency — check before hand-rolling, per
  `evaluate-3p-before-handrolling`) and diff against `docs/api.md` +
  doc-comments for the four axes.
- **T1.2** [RULE-003..005] Same for `kremory-napi`, cross-referencing
  `crates/kremory-napi/tests/api_parity.rs` which already enumerates the 1:1
  surface — don't re-derive what that gate already knows.
- **T1.3** [SC-011] Grep every in-scope doc for version pins/banners; flag any
  not matching the target version (published 0.7.0, or workspace HEAD per
  doc).
- **T1.4** Fold in the two pre-seeded findings from `plan.md` (the
  `docs/api.md:1158` self-contradiction; the stale version pins found during
  spec review) rather than rediscovering them from scratch.
- **T1.5** [SC-003] Produce the full table: axis, `file:line` both sides,
  proposed fix layer, for every row from T1.1-T1.4 plus T0.4's Phase-0
  failures not resolved by a legitimate marker.

## Phase 2 — Fix

- **T2.1** [RULE-006] Per SC-003 row: implement the fix at the stated layer.
  Code-side fixes get `/quality-gate` green (SC-008) before commit. Additive
  accessor pattern (TD-231 precedent) preferred over invasive changes.
- **T2.2** [RULE-012] For any fix that is breaking + unpublished, add the
  version-target label everywhere the affected capability is documented.
- **T2.3** [RULE-013] If a fix needs a `kremory-napi` parity-skip-list entry
  beyond the current cap (90), raise the cap by exactly the needed amount with
  a one-line rationale, per the file's own convention. Track against
  risk-register.md R3 (stop and flag if this is the 3rd such raise in this
  spec).
- **T2.4** [SC-004] Every SC-003 row gets a Fixed/NOT-FIXED disposition with a
  reason if NOT-FIXED (e.g. maps to a larger pre-existing TD per the
  Non-goals disambiguation).

## Phase 3 — Getting started (parallel with Phase 4)

- **T3.1** [RULE-008] Fresh directory outside the repo checkout.
  `cargo add kremory` against published 0.7.0. Walk the real first-run path.
  Capture the transcript verbatim.
- **T3.2** [SC-007] Write/rewrite `docs/getting-started.md` to match the
  transcript exactly. Any friction becomes a new SC-003 row (back to Phase 2),
  not prose smoothing.
- **T3.3** [risk-register R5] Re-run T3.1's transcript as the LAST step before
  Phase 5 content migration, after all Phase 2 commits have landed — not just
  once mid-Phase-3.

## Phase 4 — Node examples (parallel with Phase 3)

- **T4.1** [RULE-011, SC-006, SC-009] Write JS/TS examples for every
  non-deferred Node capability; label `@kgentic-ai/kremory-node` unpublished
  (0.4.1) wherever mentioned.
- **T4.2** [RULE-011 strengthened] Actually run each example against the
  locally-built native module via the existing
  `crates/kremory-napi/__test__/*.test.mjs` harness, or type-check via `tsc`
  against `index.d.ts` (mirroring `types.check.ts`) for anything needing a
  live LLM/embedder.

## Phase 5 — Docusaurus scaffold

- **T5.1** Scaffold a new Docusaurus site (directory TBD at implementation
  time — `website/` unless that collides with something).
- **T5.2** [RULE-009, SC-005] Migrate existing (post-Phase-2-fix) doc content
  into the scaffold. Nav structure follows existing doc structure per the
  spec's non-goal on IA decisions.
- **T5.3** [RULE-009] `npm run build` exits 0; verify nav reaches every
  existing doc; run a link-checker (find a maintained one before hand-rolling,
  per `evaluate-3p-before-handrolling`) for zero dead internal links.
- **T5.4** Do NOT deploy (RULE-010) — Phase 6 is out of scope, maintainer-gated.

## Conformance

- **T-CONF.1** Walk every RULE-00x and SC-00x; state PASS / FAIL / NOT-FIXED
  with the concrete evidence (file:line, test output, transcript) for each.
- **T-CONF.2** Explicitly confirm no standing HITL gate was crossed (no push,
  no publish, no tag, no paid benchmark) — risk-register.md R6.
- **T-CONF.3** Stamp spec.md status `Approved for Implementation → Implemented`.
