---
title: "Public docs + API-surface audit — plan"
type: plan
status: draft
created: 2026-09-07
spec: docs/specs/public-docs-and-api-surface-audit/spec.md
---

# Public docs + API-surface audit — technical approach

Six phases, each with a mechanical exit criterion. No phase exits on "looks
good" — every exit is a count, an assertion, or a diff.

## Affected areas

| Area | Packages/files | Nature of change |
|---|---|---|
| Doc compile harness | new `crates/kremory/tests/doc_examples/` (or equivalent), extraction script | New test infrastructure |
| Rust doc content | `docs/*.md`, `README.md`, inline `///` doctests | Fixes to examples found broken |
| Public API surface | `crates/kremory/src/**` (wherever Phase 1 finds gaps) | Additive accessors/builders (TD-231-class fixes), never behaviour changes to match aspirational docs |
| Node binding docs + examples | `crates/kremory-napi/`, new `examples/` or `docs/` under it | New JS/TS example files |
| Docusaurus site | new top-level `website/` (or `docs-site/`) directory | New static-site scaffold |
| Getting-started doc | `docs/getting-started.md` (new or rewritten) | Rewritten from a real run transcript |

## Phase 0 — Compile every documented example

Extract every ```rust block from `docs/*.md` and `README.md` into a generated
test target and compile it against the workspace HEAD crate (not the published
one — this phase is about catching drift in the tree being worked on).

- **Mechanism — SPIKED 2026-09-07, confirmed viable, no hand-rolled extractor
  needed** (per `evaluate-3p-before-handrolling`): `rustdoc --test <file>.md`
  natively compiles every fenced ` ```rust ` block in an arbitrary markdown
  file as a doctest — no external crate required. Verified against
  `rustc 1.93.1` with a 3-block fixture: a plain block ran and passed, a
  ` ```rust,no_run ` block compiled-only ("- compile ... ok", never executed),
  and a ` ```rust,ignore ` block was skipped and reported as `ignored` rather
  than silently dropped (all three outcomes are visible in the test count,
  which is what RULE-002's vacuous-pass guard needs). This directly gives
  RULE-001's compile requirement and the marker semantics for "legitimately
  non-compiling" blocks, with zero custom parsing.
  Two ways to wire it to the real crate, either viable — pick at implementation
  time based on which links most cleanly against workspace HEAD:
  (a) invoke `rustdoc --test docs/api.md --edition 2021 --extern
  kremory=target/debug/libkremory.rlib -L target/debug/deps` directly per doc
  file, or (b) the more common crate idiom: a tiny `#[doc = include_str!("../../docs/api.md")]`
  module in a test-only crate, exercised via plain `cargo test --doc`, which
  gets crate linking for free from cargo instead of manual `--extern`/`-L`
  flags. Prefer (b) unless it proves awkward with multiple doc files, since it
  needs no flag-plumbing at all.
- **Exit**: `N of <total> compile, M fail`, each failure attributed to
  `file:line`. RULE-002's vacuous-pass guard: assert extracted count ≥ 60
  before trusting any pass/fail number.
- Blocks that are legitimately non-compiling (illustrative fragments, runtime
  examples needing a live embedder/LLM) get an explicit marker (`no_run` or an
  equivalent convention for non-doctest fences) — the marker itself is counted,
  distinguishing "marked, understood non-compiling" from "unmarked, therefore a
  defect."
- Output feeds Phase 2 directly as a work list — no judgement needed to
  produce it.

## Phase 1 — Semantic surface audit

For each public surface (kremory crate, kremory-napi Node binding), compare
what the code exposes against what the docs claim. Adversarial framing: *"what
would a new user try that does not work"*, not *"is the doc accurate"*.

Four axes, each producing rows in one table:
- **Undocumented** — public + reachable, mentioned nowhere in the docs.
- **Documented but absent** — the TD-231 class; prose promises a method that
  does not exist. Highest severity.
- **Unreachable** — implemented and configured but with no public path to set
  it (TD-231's mirror image).
- **Wrong** — signature, default, or behaviour differs from the description.

Mechanism: grep/AST-walk the public API surface (`pub fn`, `pub struct` fields,
builder methods) against `docs/api.md` + README + doc comments; cross-reference
against the API-parity test (`crates/kremory-napi/tests/api_parity.rs`) for the
Node side, since that gate already enumerates the 1:1 surface.

Exit: the table itself, `file:line` cited both sides, every row.

## Phase 2 — Fix, at the right layer

Per finding: decide code-wrong or doc-wrong, state which before fixing (per
RULE-006). Doc-side fixes are docs-only commits. Code-side fixes go through the
project's standard gate: `/quality-gate` green before commit (SC-008), Quinn
review for anything touching `src/`, per this repo's phase-boundary discipline
in the root `CLAUDE.md`.

Breaking API changes (RULE-006's WHERE clause): implement + commit, do not
publish. Flag in the findings table for the maintainer.

Exit: every Phase 0/1 row is fixed (layer stated) or explicitly NOT FIXED with
a reason (SC-004).

## Phase 3 — Getting started, written from a real first run

Fresh directory outside this repo checkout. `cargo add kremory` against the
published crate (0.7.0, re-verified live 2026-09-07). Walk the actual
first-run experience — no assumptions, no reading the source and imagining it.

- Capture the transcript (commands + output) verbatim.
- Any friction becomes a Phase 2 row (SC-004), not prose written to route
  around it.
- Exit: transcript + doc that matches it exactly (SC-007).

## Phase 4 — Node binding examples

RULE-011 / SC-006: at least one runnable JS/TS example per non-deferred
Node-binding capability. Written and actually run against the built
`kremory-napi` native module locally (not just typed and assumed correct —
same "walk it for real" discipline as Phase 3, scaled down).

Label `@kgentic-ai/kremory-node` as unpublished (0.4.1) everywhere it's
mentioned (SC-009) — no example should imply `npm install` works today.

## Phase 5 — Docusaurus scaffold

Site structure, nav, content migration into the new scaffold directory. Reuse
existing `docs/*.md` content (post-Phase-2 fixes) rather than rewriting.

- Exit: `npm run build` succeeds locally (SC-005); every existing doc reachable
  from nav; zero dead internal links (a link-checker pass, e.g.
  `markdown-link-check` or Docusaurus's own broken-link detection at build
  time — check for a maintained option before hand-rolling one, per
  `evaluate-3p-before-handrolling`).
- Not deployed (RULE-010) — deploy is Phase 6, blocked, listed for visibility
  only, not scheduled here.

## Phase 6 — Deploy (BLOCKED — maintainer, out of scope for implementation)

Needs repo-public (unblocks free CI) + a DNS record on `kremory.dev`. Not
touched by this plan's `implement` step. Listed so it stays visible.

## Risks

| Risk | Mitigation |
|---|---|
| The compile harness extracts nothing and reports success | RULE-002 assert extracted-count ≥ 60 |
| Phase 2 "fixes" docs to match broken behaviour instead of fixing code | Every row states which layer was wrong BEFORE the fix (RULE-006) |
| Doc examples need a live LLM/embedder to compile | Compile ≠ run. Phase 0 is a type-check; runtime examples get an explicit non-compiling marker |
| Docs rewritten mid-flight as the API changes under them | Deliberate ordering: Phase 0/1/2 (code+docs stabilize) precede Phase 3/4/5 (new content) |
| Documenting unpublished packages as though they ship | Explicit status labels (SC-009), checked per-surface |
| Hand-rolling a markdown-doctest extractor when `rustdoc --test` already does most of it | Spike `rustdoc --test` compatibility first (see Phase 0 mechanism note) before building a bespoke extractor |
| Standing HITL gates (publish/push/tag) accidentally crossed during "yolo" autonomous execution | Explicitly named as out-of-scope in spec Guardrails; every phase's exit criterion stops at "committed locally," never at "pushed" |

## Findings pre-seeded from spec review (fold into the Phase 1 SC-003 table — do not treat as new discoveries requiring separate credit)

- `docs/api.md:1158` vs `docs/api.md:1167`: self-contradiction, same file, 13
  lines apart. Line 1158 says "the crate has an explicit **empty** default
  feature set"; line 1167 (§13 heading) says "kremory's `default` feature set
  is `["content-search"]` (ADR-078, 2026-07-28 — it was previously empty)".
  Line 1158 is stale (describes the pre-ADR-078 state as current). Axis:
  **Wrong**. Fix layer: doc (line 1158's table row needs updating to match
  §13, which is correct).
- Stale `version = "0.6"` Cargo.toml snippets: `docs/api.md:398,1187`,
  `docs/observability.md:288`, `README.md:429`; plus a stale `v0.6.0` banner at
  `docs/api.md:3`. Axis: covered by SC-011, not the four RULE-003/004/005 API
  axes (these are version pins, not API surface). Fix layer: doc, update to
  0.7.0 (or a `>= 0.7` range, maintainer's call).

## Sequencing / dependencies

Phase 0 → Phase 1 → Phase 2 must run in that order (each depends on the prior
phase's output as its work list). Phase 3 and Phase 4 can run in parallel with
each other once Phase 2 is stable (both are independent "walk it for real"
exercises on different surfaces). Phase 5 depends on Phase 2's doc-side fixes
being committed (it migrates content, not source-of-truth).
